import pytest


@pytest.fixture
def authenticate_client():
    def authenticate(client):
        response = client.post("/api/auth/login", json={"username": "admin", "password": "admin"})
        assert response.status_code == 200
        if response.json()["must_change_password"]:
            response = client.post("/api/auth/change-password", json={
                "current_password": "admin",
                "new_password": "Gateway-Test-Passphrase-1!",
            })
            assert response.status_code == 200
        return client

    return authenticate
