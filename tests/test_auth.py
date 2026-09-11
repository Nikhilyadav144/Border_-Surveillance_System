"""Authentication and authorisation, end to end through the real API.

Driven with FastAPI's TestClient against the actual application, so what is
asserted is what a browser or a curl command would get - not what a mocked
dependency would have returned. A test that stubs out the thing it is
checking proves nothing about the thing it is checking.

The pipeline is never started here. These tests are about the gate in front
of it, and the gate has no opinion about YOLO.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from core.auth import (
    NORMAL_USER,
    SUPER_ADMIN,
    hash_password,
    looks_hashed,
    verify_password,
)

ADMIN_ID = "ADMIN-001"
ADMIN_PW = "correct-horse-battery-staple"
USER_ID = "USER-002"
USER_PW = "operator-password-2026"


@pytest.fixture
def api(tmp_path, monkeypatch):
    """The real app, against a throwaway database.

    api.server is imported once per process and holds module-level state, so
    the database and the auth dependency are repointed at a temporary file
    for the duration of each test rather than the project's real one.
    """
    import api.deps as deps
    import api.server as server
    from database.database import Database

    db = Database(tmp_path / "auth.db")
    monkeypatch.setattr(server, "db", db)
    deps.configure(db, deps.AuthSettings(
        idle_minutes=30, absolute_hours=12, warn_before_seconds=120,
        max_failed_attempts=3, lockout_minutes=15,
        camera_control_requires_admin=True,
    ))

    db.create_user(ADMIN_ID, "Nikhil", hash_password(ADMIN_PW), SUPER_ADMIN)
    db.create_user(USER_ID, "Operator-01", hash_password(USER_PW), NORMAL_USER)

    with TestClient(server.app) as client:
        yield client, db
    db.close()


def sign_in(client, user_id, password):
    return client.post("/api/auth/login",
                       json={"user_id": user_id, "password": password})


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


# ==========================================================================
# 1-4  Signing in
# ==========================================================================

class TestSignIn:

    def test_super_admin_can_sign_in(self, api):
        client, _ = api
        response = sign_in(client, ADMIN_ID, ADMIN_PW)
        assert response.status_code == 200
        body = response.json()
        assert body["user"]["role"] == SUPER_ADMIN
        assert body["user"]["is_admin"] is True
        assert body["token"]

    def test_normal_user_can_sign_in(self, api):
        client, _ = api
        body = sign_in(client, USER_ID, USER_PW).json()
        assert body["user"]["role"] == NORMAL_USER
        assert body["user"]["is_admin"] is False

    def test_the_wrong_password_is_refused(self, api):
        client, _ = api
        response = sign_in(client, ADMIN_ID, "not-the-password")
        assert response.status_code == 401
        assert "token" not in response.json()

    def test_an_unknown_user_and_a_wrong_password_read_the_same(self, api):
        """Telling them apart enumerates accounts one request at a time."""
        client, _ = api
        unknown = sign_in(client, "NOBODY-999", "whatever")
        wrong = sign_in(client, ADMIN_ID, "not-the-password")
        assert unknown.status_code == wrong.status_code == 401
        assert unknown.json()["detail"] == wrong.json()["detail"]

    def test_a_disabled_account_cannot_sign_in(self, api):
        client, db = api
        db.update_user(USER_ID, is_active=0)
        response = sign_in(client, USER_ID, USER_PW)
        assert response.status_code == 403
        assert "disabled" in response.json()["detail"].lower()

    def test_the_user_id_is_not_case_sensitive(self, api):
        client, _ = api
        assert sign_in(client, "admin-001", ADMIN_PW).status_code == 200


# ==========================================================================
# 5-9  The authorisation boundary
# ==========================================================================

#: Every admin route. Listed explicitly so that adding one without a test is
#: a visible omission rather than a silent gap.
ADMIN_ROUTES = [
    ("GET", "/api/admin/users"),
    ("POST", "/api/admin/users"),
    ("PUT", "/api/admin/users/USER-002"),
    ("DELETE", "/api/admin/users/USER-002"),
    ("GET", "/api/admin/audit-logs"),
]


class TestAuthorisation:

    def test_super_admin_reaches_the_admin_api(self, api):
        client, _ = api
        token = sign_in(client, ADMIN_ID, ADMIN_PW).json()["token"]
        response = client.get("/api/admin/users", headers=bearer(token))
        assert response.status_code == 200
        assert {u["user_id"] for u in response.json()} == {ADMIN_ID, USER_ID}

    @pytest.mark.parametrize("method,path", ADMIN_ROUTES)
    def test_a_normal_user_is_refused_every_admin_route(self, api, method, path):
        """Typing the URL is the attack this defends against.

        The interface hides these pages from an operator, but hiding is not
        access control - the request has to be refused even when it is made
        by hand, which is what this asserts.
        """
        client, _ = api
        token = sign_in(client, USER_ID, USER_PW).json()["token"]
        response = client.request(method, path, headers=bearer(token), json={})
        assert response.status_code == 403
        assert response.json()["detail"] == "Super Admin access required"

    @pytest.mark.parametrize("method,path", ADMIN_ROUTES)
    def test_no_token_at_all_is_refused(self, api, method, path):
        client, _ = api
        assert client.request(method, path, json={}).status_code == 401

    def test_a_normal_user_cannot_start_a_camera(self, api):
        """Camera control is camera management, and management is admin."""
        client, _ = api
        token = sign_in(client, USER_ID, USER_PW).json()["token"]
        response = client.post("/api/cameras/CAM-01/start",
                               headers=bearer(token))
        assert response.status_code == 403

    def test_a_normal_user_can_still_read_the_surveillance_api(self, api):
        """Restricting an operator must not blind them."""
        client, _ = api
        token = sign_in(client, USER_ID, USER_PW).json()["token"]
        for path in ("/api/cameras", "/api/dashboard", "/api/alerts",
                     "/api/incidents", "/api/events", "/api/plates"):
            assert client.get(path, headers=bearer(token)).status_code == 200, path

    def test_a_normal_user_cannot_forge_a_role(self, api):
        """The role travels in the database row, not in the request.

        A client that sends its own role is sending an opinion. These calls
        try every obvious way to state one and none of them change anything,
        because the backend never reads a role from the request at all.
        """
        client, _ = api
        token = sign_in(client, USER_ID, USER_PW).json()["token"]
        for headers in (
            {**bearer(token), "X-Role": "SUPER_ADMIN"},
            {**bearer(token), "Role": "SUPER_ADMIN"},
        ):
            assert client.get("/api/admin/users",
                              headers=headers).status_code == 403
        response = client.get("/api/admin/users?role=SUPER_ADMIN",
                              headers=bearer(token))
        assert response.status_code == 403

    def test_a_normal_user_cannot_promote_themselves(self, api):
        client, _ = api
        token = sign_in(client, USER_ID, USER_PW).json()["token"]
        response = client.put(f"/api/admin/users/{USER_ID}",
                              headers=bearer(token),
                              json={"role": SUPER_ADMIN})
        assert response.status_code == 403


# ==========================================================================
# 10-11  Sessions
# ==========================================================================

class TestSessions:

    def test_logout_kills_the_token(self, api):
        client, _ = api
        token = sign_in(client, ADMIN_ID, ADMIN_PW).json()["token"]
        assert client.get("/api/cameras", headers=bearer(token)).status_code == 200
        assert client.post("/api/auth/logout",
                           headers=bearer(token)).status_code == 200
        # The same token, replayed. Server-side revocation is what makes this
        # a real logout rather than the client agreeing to forget.
        assert client.get("/api/cameras", headers=bearer(token)).status_code == 401
        assert client.get("/api/auth/me", headers=bearer(token)).status_code == 401

    def test_an_idle_session_expires(self, api, monkeypatch):
        import api.deps as deps
        client, db = api
        token = sign_in(client, ADMIN_ID, ADMIN_PW).json()["token"]
        assert client.get("/api/cameras", headers=bearer(token)).status_code == 200

        # Wind the session's last-seen stamp back past the idle window
        # rather than sleeping for half an hour.
        from core.auth import token_fingerprint
        stale = time.time() - deps.get_settings().idle_seconds - 5
        db.touch_auth_session(token_fingerprint(token), stale)

        assert client.get("/api/cameras", headers=bearer(token)).status_code == 401

    def test_an_expired_session_is_recorded(self, api):
        import api.deps as deps
        from core.auth import token_fingerprint
        client, db = api
        token = sign_in(client, ADMIN_ID, ADMIN_PW).json()["token"]
        db.touch_auth_session(token_fingerprint(token),
                              time.time() - deps.get_settings().idle_seconds - 5)
        client.get("/api/cameras", headers=bearer(token))
        actions = [r["action"] for r in db.get_audit_logs()]
        assert "SESSION_EXPIRED" in actions

    def test_a_made_up_token_is_refused(self, api):
        client, _ = api
        assert client.get("/api/cameras",
                          headers=bearer("a" * 43)).status_code == 401

    def test_disabling_an_account_ends_its_open_session(self, api):
        """Disable has to mean disabled NOW, not at the next expiry."""
        client, db = api
        token = sign_in(client, USER_ID, USER_PW).json()["token"]
        assert client.get("/api/cameras", headers=bearer(token)).status_code == 200
        db.update_user(USER_ID, is_active=0)
        assert client.get("/api/cameras", headers=bearer(token)).status_code == 401

    def test_me_reports_the_backends_view_of_the_caller(self, api):
        client, _ = api
        token = sign_in(client, USER_ID, USER_PW).json()["token"]
        body = client.get("/api/auth/me", headers=bearer(token)).json()
        assert body["user"]["user_id"] == USER_ID
        assert body["user"]["role"] == NORMAL_USER
        assert body["session_idle_seconds"] > 0


# ==========================================================================
# 12, 15  Failed attempts
# ==========================================================================

class TestLoginProtection:

    def test_a_failed_attempt_is_recorded(self, api):
        client, db = api
        sign_in(client, ADMIN_ID, "wrong")
        rows = [r for r in db.get_audit_logs() if r["action"] == "LOGIN_FAILED"]
        assert len(rows) == 1
        assert rows[0]["status"] == "FAILED"

    def test_repeated_failures_trigger_a_cooldown(self, api):
        client, _ = api
        for _ in range(3):                      # the threshold in the fixture
            assert sign_in(client, ADMIN_ID, "wrong").status_code == 401
        blocked = sign_in(client, ADMIN_ID, ADMIN_PW)
        assert blocked.status_code == 429
        assert "try again" in blocked.json()["detail"].lower()

    def test_the_cooldown_is_temporary_not_a_permanent_lock(self, api):
        """Nobody is ever locked out with no way back."""
        client, db = api
        for _ in range(3):
            sign_in(client, ADMIN_ID, "wrong")
        assert sign_in(client, ADMIN_ID, ADMIN_PW).status_code == 429
        row = db.get_user(ADMIN_ID)
        assert row["locked_until"] is not None
        assert float(row["locked_until"]) > time.time()      # it lapses

    def test_an_admin_can_clear_a_lockout_immediately(self, api):
        client, db = api
        for _ in range(3):
            sign_in(client, USER_ID, "wrong")
        assert sign_in(client, USER_ID, USER_PW).status_code == 429

        admin_token = sign_in(client, ADMIN_ID, ADMIN_PW).json()["token"]
        assert client.post(f"/api/admin/users/{USER_ID}/unlock",
                           headers=bearer(admin_token)).status_code == 200
        assert sign_in(client, USER_ID, USER_PW).status_code == 200

    def test_one_success_clears_the_counter(self, api):
        """A weekly typo must never accumulate into a lock."""
        client, db = api
        sign_in(client, ADMIN_ID, "wrong")
        sign_in(client, ADMIN_ID, "wrong")
        sign_in(client, ADMIN_ID, ADMIN_PW)
        assert int(db.get_user(ADMIN_ID)["failed_attempts"]) == 0


# ==========================================================================
# 13-14  Creating users, and what gets stored
# ==========================================================================

class TestUserAdministration:

    def test_an_admin_can_create_a_user(self, api):
        client, db = api
        token = sign_in(client, ADMIN_ID, ADMIN_PW).json()["token"]
        response = client.post("/api/admin/users", headers=bearer(token),
                               json={"user_id": "USER-003",
                                     "username": "Operator-02",
                                     "password": "another-good-password",
                                     "role": NORMAL_USER})
        assert response.status_code == 201
        assert response.json()["role"] == NORMAL_USER
        assert sign_in(client, "USER-003",
                       "another-good-password").status_code == 200

    def test_the_password_is_stored_as_a_hash(self, api):
        """The single most important assertion in this file."""
        client, db = api
        token = sign_in(client, ADMIN_ID, ADMIN_PW).json()["token"]
        client.post("/api/admin/users", headers=bearer(token),
                    json={"user_id": "USER-004", "username": "Operator-03",
                          "password": "plaintext-must-not-survive"})
        row = db.get_user("USER-004")
        stored = row["password_hash"]
        assert looks_hashed(stored)
        assert "plaintext-must-not-survive" not in stored
        # And nowhere else in the row either.
        assert "plaintext-must-not-survive" not in str(dict(row))
        assert verify_password("plaintext-must-not-survive", stored)

    def test_no_response_ever_returns_a_hash(self, api):
        client, _ = api
        token = sign_in(client, ADMIN_ID, ADMIN_PW).json()["token"]
        body = client.get("/api/admin/users", headers=bearer(token)).text
        assert "password" not in body.lower()
        assert "scrypt" not in body

    def test_two_identical_passwords_do_not_share_a_hash(self, api):
        """A per-password salt is what makes a lookup table useless."""
        client, db = api
        token = sign_in(client, ADMIN_ID, ADMIN_PW).json()["token"]
        for uid in ("USER-005", "USER-006"):
            client.post("/api/admin/users", headers=bearer(token),
                        json={"user_id": uid, "username": uid,
                              "password": "the-very-same-password"})
        assert (db.get_user("USER-005")["password_hash"]
                != db.get_user("USER-006")["password_hash"])

    def test_a_short_password_is_refused(self, api):
        client, _ = api
        token = sign_in(client, ADMIN_ID, ADMIN_PW).json()["token"]
        response = client.post("/api/admin/users", headers=bearer(token),
                               json={"user_id": "USER-007",
                                     "username": "x", "password": "short"})
        assert response.status_code == 400

    def test_a_duplicate_user_id_is_refused(self, api):
        client, _ = api
        token = sign_in(client, ADMIN_ID, ADMIN_PW).json()["token"]
        response = client.post("/api/admin/users", headers=bearer(token),
                               json={"user_id": USER_ID, "username": "dupe",
                                     "password": "a-good-password"})
        assert response.status_code == 409

    def test_the_last_admin_cannot_be_disabled(self, api):
        """The change that would leave a system nobody can administer."""
        client, _ = api
        token = sign_in(client, ADMIN_ID, ADMIN_PW).json()["token"]
        response = client.put(f"/api/admin/users/{ADMIN_ID}",
                              headers=bearer(token), json={"is_active": False})
        assert response.status_code == 409
        assert "only active super admin" in response.json()["detail"].lower()

    def test_the_last_admin_cannot_be_demoted(self, api):
        client, _ = api
        token = sign_in(client, ADMIN_ID, ADMIN_PW).json()["token"]
        response = client.put(f"/api/admin/users/{ADMIN_ID}",
                              headers=bearer(token), json={"role": NORMAL_USER})
        assert response.status_code == 409

    def test_an_admin_can_be_disabled_once_another_exists(self, api):
        client, _ = api
        token = sign_in(client, ADMIN_ID, ADMIN_PW).json()["token"]
        client.post("/api/admin/users", headers=bearer(token),
                    json={"user_id": "ADMIN-002", "username": "Second",
                          "password": "second-admin-password",
                          "role": SUPER_ADMIN})
        assert client.put(f"/api/admin/users/{ADMIN_ID}",
                          headers=bearer(token),
                          json={"is_active": False}).status_code == 200

    def test_an_admin_cannot_delete_their_own_account(self, api):
        client, _ = api
        token = sign_in(client, ADMIN_ID, ADMIN_PW).json()["token"]
        response = client.delete(f"/api/admin/users/{ADMIN_ID}",
                                 headers=bearer(token))
        assert response.status_code == 400

    def test_a_password_reset_ends_that_users_sessions(self, api):
        client, _ = api
        user_token = sign_in(client, USER_ID, USER_PW).json()["token"]
        admin_token = sign_in(client, ADMIN_ID, ADMIN_PW).json()["token"]
        client.put(f"/api/admin/users/{USER_ID}", headers=bearer(admin_token),
                   json={"password": "a-brand-new-password"})
        assert client.get("/api/cameras",
                          headers=bearer(user_token)).status_code == 401
        assert sign_in(client, USER_ID, "a-brand-new-password").status_code == 200


# ==========================================================================
# The audit log
# ==========================================================================

class TestAuditLog:

    def test_the_admin_can_read_it_and_a_normal_user_cannot(self, api):
        client, _ = api
        admin = sign_in(client, ADMIN_ID, ADMIN_PW).json()["token"]
        user = sign_in(client, USER_ID, USER_PW).json()["token"]
        assert client.get("/api/admin/audit-logs",
                          headers=bearer(admin)).status_code == 200
        assert client.get("/api/admin/audit-logs",
                          headers=bearer(user)).status_code == 403

    def test_a_refused_admin_request_is_recorded(self, api):
        """The interesting event, not the boring one."""
        client, db = api
        user = sign_in(client, USER_ID, USER_PW).json()["token"]
        client.get("/api/admin/users", headers=bearer(user))
        denied = [r for r in db.get_audit_logs()
                  if r["action"] == "ACCESS_DENIED"]
        assert denied
        assert denied[0]["user_id"] == USER_ID
        assert denied[0]["status"] == "DENIED"
        assert "/api/admin/users" in denied[0]["resource"]

    def test_the_expected_actions_are_all_recorded(self, api):
        client, db = api
        admin = sign_in(client, ADMIN_ID, ADMIN_PW).json()["token"]
        sign_in(client, USER_ID, "wrong")
        client.post("/api/admin/users", headers=bearer(admin),
                    json={"user_id": "USER-009", "username": "Nine",
                          "password": "nine-nine-nine-nine"})
        client.put("/api/admin/users/USER-009", headers=bearer(admin),
                   json={"is_active": False})
        client.post("/api/auth/logout", headers=bearer(admin))

        actions = {r["action"] for r in db.get_audit_logs(limit=100)}
        for expected in ("LOGIN_SUCCESS", "LOGIN_FAILED", "LOGOUT",
                         "USER_CREATED", "USER_DISABLED"):
            assert expected in actions, expected

    def test_no_credential_ever_reaches_the_audit_log(self, api):
        client, db = api
        sign_in(client, ADMIN_ID, ADMIN_PW)
        sign_in(client, ADMIN_ID, "a-very-distinctive-wrong-password")
        blob = str([dict(r) for r in db.get_audit_logs(limit=100)])
        assert "a-very-distinctive-wrong-password" not in blob
        assert ADMIN_PW not in blob


# ==========================================================================
# What stays open
# ==========================================================================

class TestOpenRoutes:

    def test_health_needs_no_session(self, api):
        """The login screen has to be able to say whether the backend is up
        BEFORE anybody has a token. A health check that requires a session
        cannot answer the one question it exists for."""
        client, _ = api
        assert client.get("/api/health").status_code == 200

    def test_health_reveals_nothing_about_the_site(self, api):
        client, _ = api
        body = client.get("/api/health").json()
        assert set(body) <= {"status", "service"}
