"""Who is calling, and may they.

The whole authorisation model is three dependencies:

    current_user   - a valid session, or 401
    require_user   - the same, as a route guard
    require_admin  - a valid session belonging to a SUPER_ADMIN, or 403

Every decision is made from the database row the token resolves to. Nothing
here reads a role, a user id or a permission from the request body, a query
parameter or a custom header, because all three are written by the client and
a client that can name its own role has no authorisation at all.

This module sits between the HTTP layer and the surveillance pipeline and
touches neither. It imports nothing from ``core.pipeline`` and nothing in the
detection path imports it.
"""

from __future__ import annotations

import time
from typing import Optional

from fastapi import Depends, HTTPException, Request, status

from core.auth import (
    AuditAction,
    AuditStatus,
    Principal,
    SUPER_ADMIN,
    token_fingerprint,
)


class AuthSettings:
    """Timeouts and thresholds, read once from config at startup.

    Held on an object rather than as module constants so the tests can build
    one with short timeouts without reaching into module globals.
    """

    def __init__(
        self,
        idle_minutes: float = 30.0,
        absolute_hours: float = 12.0,
        warn_before_seconds: float = 120.0,
        max_failed_attempts: int = 5,
        lockout_minutes: float = 15.0,
        camera_control_requires_admin: bool = True,
    ) -> None:
        self.idle_seconds = float(idle_minutes) * 60.0
        self.absolute_seconds = float(absolute_hours) * 3600.0
        self.warn_before_seconds = float(warn_before_seconds)
        self.max_failed_attempts = int(max_failed_attempts)
        self.lockout_seconds = float(lockout_minutes) * 60.0
        self.camera_control_requires_admin = bool(camera_control_requires_admin)

    @classmethod
    def from_config(cls, config) -> "AuthSettings":
        return cls(
            idle_minutes=config.get("auth.session_idle_minutes", 30),
            absolute_hours=config.get("auth.session_absolute_hours", 12),
            warn_before_seconds=config.get("auth.warn_before_expiry_seconds", 120),
            max_failed_attempts=config.get("auth.max_failed_attempts", 5),
            lockout_minutes=config.get("auth.lockout_minutes", 15),
            camera_control_requires_admin=config.get(
                "auth.camera_control_requires_admin", True),
        )

    def to_public(self) -> dict:
        """What the interface needs to run its own countdown."""
        return {
            "session_idle_seconds": self.idle_seconds,
            "warn_before_expiry_seconds": self.warn_before_seconds,
            "camera_control_requires_admin": self.camera_control_requires_admin,
        }


# Installed by api/server.py at import time. Kept as module state because a
# FastAPI dependency has no other way to reach the application's database and
# settings without threading them through every route signature.
_db = None
_settings = AuthSettings()


def configure(db, settings: AuthSettings) -> None:
    global _db, _settings
    _db = db
    _settings = settings


def get_settings() -> AuthSettings:
    return _settings


def get_db():
    if _db is None:                                  # pragma: no cover
        raise RuntimeError("api.deps.configure() was never called")
    return _db


# --------------------------------------------------------------------------
# Reading the token
# --------------------------------------------------------------------------

def _bearer_token(request: Request) -> Optional[str]:
    """Pull the session token out of the request.

    `Authorization: Bearer <token>` is the normal path. The `X-Session-Token`
    header is accepted as well because an MJPEG stream is loaded by an
    `<img>` tag, and a browser will not attach an Authorization header to
    one; the stream route reads a query parameter for the same reason.
    """
    header = request.headers.get("authorization") or ""
    if header.lower().startswith("bearer "):
        token = header[7:].strip()
        if token:
            return token
    alt = request.headers.get("x-session-token")
    if alt:
        return alt.strip()
    return None


class AuthError(HTTPException):
    """401 with a machine-readable reason.

    The `code` lets the interface tell three situations apart that all mean
    "sign in again" but read very differently to a person: never signed in,
    signed in but timed out, and signed in on an account that has since been
    disabled.
    """

    def __init__(self, detail: str, code: str) -> None:
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )
        self.code = code


def resolve_principal(request: Request) -> Optional[Principal]:
    """Identify the caller, or return None. Never raises.

    Also enforces both timeouts and refreshes the idle clock, so any
    authenticated request keeps a working session alive and no request can
    revive an expired one.
    """
    db = get_db()
    token = _bearer_token(request)
    if not token:
        return None

    row = db.get_auth_session(token_fingerprint(token))
    if row is None:
        return None

    now = time.time()

    if int(row["revoked"] or 0):
        return None
    if not int(row["is_active"] or 0):
        # Disabled between requests. The session is killed here rather than
        # merely refused, so the account cannot be used again if it is
        # re-enabled later with the same token still in a browser tab.
        db.revoke_user_sessions(row["user_id"])
        return None
    if now >= float(row["expires_ts"]):
        db.revoke_auth_session(row["token_hash"])
        return None
    if now - float(row["last_seen_ts"]) > _settings.idle_seconds:
        db.revoke_auth_session(row["token_hash"])
        db.insert_audit(AuditAction.SESSION_EXPIRED, user_id=row["user_id"],
                        status=AuditStatus.SUCCESS,
                        detail={"reason": "idle timeout"})
        return None

    db.touch_auth_session(row["token_hash"], now)
    principal = Principal(
        user_id=row["user_id"],
        username=row["username"],
        role=row["role"],
    )
    # Stashed so a route can audit an action against the caller without
    # taking the Principal as a second parameter.
    request.state.principal = principal
    request.state.token_hash = row["token_hash"]
    return principal


# --------------------------------------------------------------------------
# The guards
# --------------------------------------------------------------------------

def require_user(request: Request) -> Principal:
    """Any signed-in account. 401 otherwise."""
    principal = resolve_principal(request)
    if principal is None:
        raise AuthError("Not authenticated. Please sign in.",
                        code="NOT_AUTHENTICATED")
    return principal


def require_admin(request: Request,
                  principal: Principal = Depends(require_user)) -> Principal:
    """A SUPER_ADMIN. 403 for anyone else.

    The refusal is audited, because somebody reaching an admin endpoint
    without the role is exactly the event a security log exists to capture -
    whether it is a curious operator typing a URL or something worse.
    """
    if principal.role != SUPER_ADMIN:
        get_db().insert_audit(
            AuditAction.ACCESS_DENIED,
            user_id=principal.user_id,
            resource=str(request.url.path),
            status=AuditStatus.DENIED,
            detail={"method": request.method, "role": principal.role},
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Super Admin access required",
        )
    return principal


def require_camera_control(
    request: Request,
    principal: Principal = Depends(require_user),
) -> Principal:
    """Whoever may start and stop a camera.

    Administrators by default: starting and stopping a feed is camera
    management, and an operator stopping a camera mid-incident is a real
    failure mode. Flipping `auth.camera_control_requires_admin` to false in
    application.yaml hands it to every signed-in user without a code change.
    """
    if _settings.camera_control_requires_admin and principal.role != SUPER_ADMIN:
        get_db().insert_audit(
            AuditAction.ACCESS_DENIED,
            user_id=principal.user_id,
            resource=str(request.url.path),
            status=AuditStatus.DENIED,
            detail={"method": request.method, "role": principal.role,
                    "reason": "camera control is restricted to Super Admin"},
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Super Admin access required",
        )
    return principal


def optional_user(request: Request) -> Optional[Principal]:
    """Identify the caller if possible, without refusing the request."""
    return resolve_principal(request)


def principal_from_token(token: str) -> Optional[Principal]:
    """Resolve a bare token, for the one caller that cannot send a header.

    The MJPEG stream is loaded by an `<img>` tag and browsers attach no
    Authorization header to image requests. Every check below is the same as
    `resolve_principal` makes; only where the token came from differs.
    """
    if not token:
        return None
    db = get_db()
    row = db.get_auth_session(token_fingerprint(token))
    if row is None:
        return None

    now = time.time()
    if int(row["revoked"] or 0) or not int(row["is_active"] or 0):
        return None
    if now >= float(row["expires_ts"]):
        db.revoke_auth_session(row["token_hash"])
        return None
    if now - float(row["last_seen_ts"]) > _settings.idle_seconds:
        db.revoke_auth_session(row["token_hash"])
        return None

    db.touch_auth_session(row["token_hash"], now)
    return Principal(user_id=row["user_id"], username=row["username"],
                     role=row["role"])
