from __future__ import annotations
import os
from pathlib import Path
from cryptography.fernet import Fernet

class SecretCipher:
    """Encrypt camera stream URLs; key is kept outside SQLite in a persistent secret file."""
    def __init__(self, key_path: str | Path):
        configured = os.getenv("FACE_SECRET_KEY")
        path = Path(key_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if configured:
            key = configured.encode("ascii")
        elif path.exists():
            key = path.read_bytes().strip()
        else:
            key = Fernet.generate_key()
            path.write_bytes(key + b"\n")
            try:
                path.chmod(0o600)
            except OSError:
                pass
        self._fernet = Fernet(key)
    def encrypt(self, value: str | None) -> str | None:
        if not value:
            return None
        return self._fernet.encrypt(value.encode()).decode()
    def decrypt(self, value: str | None) -> str | None:
        if not value:
            return None
        return self._fernet.decrypt(value.encode()).decode()
