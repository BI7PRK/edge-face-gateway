import threading
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from core.infrastructure.auth import SESSION_COOKIE, verify_password
from core.infrastructure.database.database import Database
from core.infrastructure.database.models import SystemUser, WebSession
from core.runtime.web import create_app


def _engine(database):
    return SimpleNamespace(
        database=database,
        output_dir=Path("."),
        host_started_mono=0.0,
        runtimes=[],
        stop_event=threading.Event(),
    )


def test_default_admin_is_hashed_and_forced_to_change_password(tmp_path: Path):
    db = Database(tmp_path / "gateway.db")
    db.migrate()
    app = create_app(_engine(db))
    client = TestClient(app, follow_redirects=False)

    with db.sessions() as session:
        admin = session.query(SystemUser).filter_by(username="admin").one()
        assert admin.password_hash != "admin"
        assert verify_password("admin", admin.password_hash)
        assert admin.must_change_password is True
        assert session.query(WebSession).count() == 0

    login_page = client.get("/login")
    assert login_page.status_code == 200
    assert "系统登录" in login_page.text
    assert "Edge Face Gateway" in login_page.text
    assert "<nav>" not in login_page.text
    assert client.get("/").status_code == 303
    assert client.get("/api/users").status_code == 401
    assert client.post("/api/auth/login", json={"username": "admin", "password": "wrong"}).status_code == 401
    login = client.post("/api/auth/login", json={"username": "admin", "password": "admin"})
    assert login.status_code == 200
    assert login.json()["must_change_password"] is True
    assert "httponly" in login.headers["set-cookie"].lower()
    token = client.cookies.get(SESSION_COOKIE)
    with db.sessions() as session:
        web_session = session.query(WebSession).one()
        assert web_session.token_hash != token
    assert client.get("/api/status").status_code == 428
    assert client.get("/").headers["location"] == "/change-password"
    assert client.post("/api/auth/change-password", json={
        "current_password": "admin",
        "new_password": "short",
    }).status_code == 422

    changed = client.post("/api/auth/change-password", json={
        "current_password": "admin",
        "new_password": "New-Admin-Passphrase-2026!",
    })
    assert changed.status_code == 200
    users_page = client.get("/users")
    assert users_page.status_code == 200
    assert "添加系统用户" in users_page.text
    assert client.get("/api/users").status_code == 200
    assert client.post("/api/auth/logout", headers={"Origin": "http://attacker.example"}).status_code == 403
    assert client.post("/api/auth/logout").status_code == 200
    assert client.get("/api/users").status_code == 401
    assert client.post("/api/auth/login", json={"username": "admin", "password": "admin"}).status_code == 401

    with db.sessions() as session:
        admin = session.query(SystemUser).filter_by(username="admin").one()
        assert admin.must_change_password is False
        assert verify_password("New-Admin-Passphrase-2026!", admin.password_hash)
    db.close()


def test_expired_web_session_is_rejected(tmp_path: Path, authenticate_client):
    db = Database(tmp_path / "gateway.db")
    db.migrate()
    app = create_app(_engine(db))
    client = authenticate_client(TestClient(app))
    client.post("/api/auth/login", json={
        "username": "admin",
        "password": "Gateway-Test-Passphrase-1!",
    })
    with db.sessions.begin() as session:
        session.query(WebSession).update({WebSession.expires_at_ms: 0})
    assert client.get("/api/users").status_code == 401
    db.close()


def test_user_management_resets_password_revokes_sessions_and_keeps_active_admin(tmp_path: Path, authenticate_client):
    db = Database(tmp_path / "gateway.db")
    db.migrate()
    app = create_app(_engine(db))
    admin_client = authenticate_client(TestClient(app))
    assert admin_client.post("/api/users", json={
        "username": "too-short",
        "password": "12345",
    }).status_code == 422
    created = admin_client.post("/api/users", json={
        "username": "operator",
        "password": "123456",
    })
    assert created.status_code == 201
    user_id = created.json()["id"]
    assert created.json()["must_change_password"] is True
    with db.sessions() as session:
        user = session.get(SystemUser, user_id)
        assert user.password_hash != "123456"
        assert verify_password("123456", user.password_hash)

    user_client = TestClient(app)
    login = user_client.post("/api/auth/login", json={
        "username": "operator",
        "password": "123456",
    })
    assert login.status_code == 200
    assert login.json()["must_change_password"] is True
    assert user_client.get("/api/users").status_code == 428
    changed = user_client.post("/api/auth/change-password", json={
        "current_password": "123456",
        "new_password": "Operator-Second-Passphrase-2!",
    })
    assert changed.status_code == 200
    assert user_client.get("/api/users").status_code == 200

    updated = admin_client.put(f"/api/users/{user_id}", json={
        "username": "operator-renamed",
        "is_active": True,
        "password": "Operator-Reset-Passphrase-3!",
    })
    assert updated.status_code == 200
    assert updated.json()["must_change_password"] is True
    assert user_client.get("/api/users").status_code == 401
    reset_login = user_client.post("/api/auth/login", json={
        "username": "operator-renamed",
        "password": "Operator-Reset-Passphrase-3!",
    })
    assert reset_login.status_code == 200
    assert reset_login.json()["must_change_password"] is True

    disabled = admin_client.put(f"/api/users/{user_id}", json={
        "username": "operator-renamed",
        "is_active": False,
    })
    assert disabled.status_code == 200
    assert user_client.get("/api/users").status_code == 401
    assert admin_client.delete(f"/api/users/{user_id}").status_code == 200
    admin = admin_client.get("/api/users").json()["items"][0]
    assert admin["username"] == "admin"
    assert admin_client.delete(f"/api/users/{admin['id']}").status_code == 409
    db.close()
