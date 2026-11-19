from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError

from core.infrastructure.database.models import SystemUser, WebSession

PASSWORD_ITERATIONS = 600_000
PASSWORD_MIN_LENGTH = 6
SESSION_TTL_MS = 8 * 60 * 60 * 1000
SESSION_COOKIE = "efg_session"


class AuthenticationError(ValueError):
    pass


class DuplicateUsernameError(ValueError):
    pass


class LastActiveUserError(ValueError):
    pass


class UserNotFoundError(ValueError):
    pass


@dataclass(frozen=True)
class AuthenticatedUser:
    id: int
    username: str
    must_change_password: bool


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS, dklen=32)
    salt_text = base64.urlsafe_b64encode(salt).decode("ascii").rstrip("=")
    digest_text = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${salt_text}${digest_text}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations_text, salt_text, expected_text = encoded.split("$")
        iterations = int(iterations_text)
        if algorithm != "pbkdf2_sha256" or not 100_000 <= iterations <= 2_000_000:
            return False
        salt = base64.urlsafe_b64decode(salt_text + "=" * (-len(salt_text) % 4))
        expected = base64.urlsafe_b64decode(expected_text + "=" * (-len(expected_text) % 4))
        if len(salt) != 16 or len(expected) != 32:
            return False
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations, dklen=len(expected))
        return hmac.compare_digest(actual, expected)
    except (AttributeError, TypeError, ValueError):
        return False


_DUMMY_PASSWORD_HASH = hash_password("not-a-real-account-password")


class AuthService:
    def __init__(self, database):
        self.database = database

    def ensure_initial_admin(self) -> None:
        now = int(time.time() * 1000)
        with self.database.sessions.begin() as session:
            exists = session.scalar(select(SystemUser.id).limit(1))
            if exists is None:
                session.add(SystemUser(
                    username="admin",
                    password_hash=hash_password("admin"),
                    is_active=True,
                    must_change_password=True,
                    created_at_ms=now,
                    updated_at_ms=now,
                ))

    def authenticate(self, username: str, password: str) -> AuthenticatedUser | None:
        with self.database.sessions() as session:
            user = session.scalar(select(SystemUser).where(SystemUser.username == username))
            password_valid = verify_password(password, user.password_hash if user is not None else _DUMMY_PASSWORD_HASH)
            if user is None or not user.is_active or not password_valid:
                return None
            return AuthenticatedUser(user.id, user.username, user.must_change_password)

    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(token.encode("ascii")).hexdigest()

    def _new_session(self, session, user_id: int, now: int) -> str:
        token = secrets.token_urlsafe(32)
        session.add(WebSession(
            token_hash=self._token_hash(token),
            user_id=user_id,
            created_at_ms=now,
            expires_at_ms=now + SESSION_TTL_MS,
        ))
        return token

    def create_session(self, user_id: int) -> str:
        now = int(time.time() * 1000)
        with self.database.sessions.begin() as session:
            user = session.get(SystemUser, user_id)
            if user is None or not user.is_active:
                raise UserNotFoundError
            session.execute(delete(WebSession).where(WebSession.expires_at_ms <= now))
            token = self._new_session(session, user_id, now)
        return token

    def user_for_session(self, token: str | None) -> AuthenticatedUser | None:
        if not token:
            return None
        try:
            token_hash = self._token_hash(token)
        except UnicodeEncodeError:
            return None
        now = int(time.time() * 1000)
        with self.database.sessions.begin() as session:
            web_session = session.get(WebSession, token_hash)
            if web_session is None:
                return None
            user = session.get(SystemUser, web_session.user_id)
            if web_session.expires_at_ms <= now or user is None or not user.is_active:
                session.delete(web_session)
                return None
            return AuthenticatedUser(user.id, user.username, user.must_change_password)

    def revoke_session(self, token: str | None) -> None:
        if not token:
            return
        try:
            token_hash = self._token_hash(token)
        except UnicodeEncodeError:
            return
        with self.database.sessions.begin() as session:
            session.execute(delete(WebSession).where(WebSession.token_hash == token_hash))

    def change_password(self, user_id: int, current_password: str, new_password: str) -> str:
        now = int(time.time() * 1000)
        with self.database.sessions.begin() as session:
            user = session.get(SystemUser, user_id)
            if user is None or not user.is_active:
                raise UserNotFoundError
            if not verify_password(current_password, user.password_hash):
                raise AuthenticationError
            user.password_hash = hash_password(new_password)
            user.must_change_password = False
            user.updated_at_ms = now
            session.execute(delete(WebSession).where(WebSession.user_id == user_id))
            return self._new_session(session, user_id, now)

    def list_users(self) -> list[dict]:
        with self.database.sessions() as session:
            users = session.scalars(select(SystemUser).order_by(SystemUser.username)).all()
            return [
                {
                    "id": user.id,
                    "username": user.username,
                    "is_active": user.is_active,
                    "must_change_password": user.must_change_password,
                    "created_at_ms": user.created_at_ms,
                }
                for user in users
            ]

    def create_user(self, username: str, password: str) -> dict:
        now = int(time.time() * 1000)
        user = SystemUser(
            username=username,
            password_hash=hash_password(password),
            is_active=True,
            must_change_password=True,
            created_at_ms=now,
            updated_at_ms=now,
        )
        try:
            with self.database.sessions.begin() as session:
                session.add(user)
                session.flush()
                return {
                    "id": user.id,
                    "username": user.username,
                    "is_active": user.is_active,
                    "must_change_password": user.must_change_password,
                    "created_at_ms": user.created_at_ms,
                }
        except IntegrityError as exc:
            raise DuplicateUsernameError from exc

    def update_user(self, user_id: int, username: str, is_active: bool, password: str | None) -> dict:
        now = int(time.time() * 1000)
        try:
            with self.database.sessions.begin() as session:
                user = session.get(SystemUser, user_id)
                if user is None:
                    raise UserNotFoundError
                if not is_active and user.is_active:
                    active_count = session.scalar(
                        select(func.count()).select_from(SystemUser).where(SystemUser.is_active.is_(True))
                    )
                    if active_count <= 1:
                        raise LastActiveUserError
                user.username = username
                user.is_active = is_active
                if password:
                    user.password_hash = hash_password(password)
                    user.must_change_password = True
                user.updated_at_ms = now
                if not is_active or password:
                    session.execute(delete(WebSession).where(WebSession.user_id == user_id))
                session.flush()
                return {
                    "id": user.id,
                    "username": user.username,
                    "is_active": user.is_active,
                    "must_change_password": user.must_change_password,
                    "created_at_ms": user.created_at_ms,
                }
        except IntegrityError as exc:
            raise DuplicateUsernameError from exc

    def delete_user(self, user_id: int) -> None:
        with self.database.sessions.begin() as session:
            user = session.get(SystemUser, user_id)
            if user is None:
                raise UserNotFoundError
            if user.is_active:
                active_count = session.scalar(
                    select(func.count()).select_from(SystemUser).where(SystemUser.is_active.is_(True))
                )
                if active_count <= 1:
                    raise LastActiveUserError
            session.delete(user)
