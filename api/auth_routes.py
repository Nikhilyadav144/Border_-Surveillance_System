"""Sign-in, sign-out and administration endpoints.

Two routers, mounted by api/server.py:

    auth_router   /api/auth/*    open to anyone; login is how you get a token
    admin_router  /api/admin/*   every route behind require_admin

The split is the security boundary. `admin_router` declares its dependency
once, on the router, so a route added to it later is protected whether or not
whoever adds it remembers to think about authorisation. Protection that has
to be remembered per route is protection that will eventually be forgotten.
"""

from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from api import deps
from core.auth import (
    AuditAction,
    AuditStatus,
    NORMAL_USER,
    PasswordError,
    Principal,
    SUPER_ADMIN,
    hash_password,
    new_token,
    token_fingerprint,
    validate_password,
    validate_role,
    validate_user_id,
    validate_username,
    verify_password,
)

auth_router = APIRouter(prefix="/api/auth", tags=["auth"])
admin_router = APIRouter(
    prefix="/api/admin",
    tags=["admin"],
    dependencies=[Depends(deps.require_admin)],
)


# ==========================================================================
# Request bodies
# ==========================================================================

class LoginRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class CreateUserRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=64)
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)
    # Defaulted to NORMAL_USER and validated below. A body that omits the
    # field can never accidentally mint an administrator.
    role: str = NORMAL_USER
    is_active: bool = True


class UpdateUserRequest(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None
    password: Optional[str] = None


# ==========================================================================
# Sign in
# ==========================================================================

#: What a failed sign-in says, whatever actually went wrong.
#
# One message for "no such user" and for "wrong password" on purpose. Telling
# the two apart hands an attacker a way to enumerate valid accounts one
# request at a time, and tells a legitimate user nothing they can act on.
INVALID_CREDENTIALS = "Invalid User ID or password."


@auth_router.post("/login")
def login(payload: LoginRequest, request: Request):
    """Exchange credentials for a session token."""
    db = deps.get_db()
    settings = deps.get_settings()
    now = time.time()
    user_id = (payload.user_id or "").strip().upper()

    row = db.get_user(user_id)

    if row is None:
        # No such account. Audited without a user_id, because there is no
        # user to attribute it to - the attempted id goes in the detail.
        db.insert_audit(AuditAction.LOGIN_FAILED, user_id=None,
                        resource=user_id, status=AuditStatus.FAILED,
                        detail={"reason": "unknown user"})
        # Still spend roughly the time a real verification costs. Returning
        # instantly for unknown accounts and slowly for real ones is an
        # account-enumeration oracle measured with a stopwatch.
        verify_password(payload.password, "scrypt$32768$8$1$00$00")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail=INVALID_CREDENTIALS)

    locked_until = row["locked_until"]
    if locked_until is not None and now < float(locked_until):
        remaining = int(float(locked_until) - now)
        db.insert_audit(AuditAction.LOGIN_BLOCKED, user_id=user_id,
                        status=AuditStatus.DENIED,
                        detail={"seconds_remaining": remaining})
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(f"Too many failed attempts. Try again in "
                    f"{max(1, remaining // 60)} minute(s)."),
        )

    if not verify_password(payload.password, row["password_hash"]):
        attempts = db.record_login_failure(
            user_id, settings.max_failed_attempts,
            settings.lockout_seconds, now)
        db.insert_audit(AuditAction.LOGIN_FAILED, user_id=user_id,
                        status=AuditStatus.FAILED,
                        detail={"consecutive_failures": attempts})
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail=INVALID_CREDENTIALS)

    # The password was right. Whether the account may sign in is a separate
    # question, and it is answered after the check so that a disabled
    # account cannot be used to test passwords against.
    if not int(row["is_active"] or 0):
        db.insert_audit(AuditAction.LOGIN_FAILED, user_id=user_id,
                        status=AuditStatus.DENIED,
                        detail={"reason": "account disabled"})
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account has been disabled. "
                   "Contact the Super Administrator.",
        )

    token = new_token()
    db.create_auth_session(
        token_hash=token_fingerprint(token),
        user_id=user_id,
        created_ts=now,
        expires_ts=now + settings.absolute_seconds,
        user_agent=request.headers.get("user-agent"),
    )
    db.record_login_success(user_id, now)
    db.insert_audit(AuditAction.LOGIN_SUCCESS, user_id=user_id,
                    status=AuditStatus.SUCCESS)
    db.purge_expired_sessions(now - 86400)

    principal = Principal(user_id=user_id, username=row["username"],
                          role=row["role"])
    return {
        "token": token,
        "user": principal.to_public(),
        **settings.to_public(),
    }


@auth_router.post("/logout")
def logout(request: Request,
           principal: Principal = Depends(deps.require_user)):
    """End this session on the server, not just in the browser.

    The row is revoked, so the token is dead the moment this returns. A
    client that keeps a copy gains nothing by replaying it.
    """
    db = deps.get_db()
    token_hash = getattr(request.state, "token_hash", None)
    if token_hash:
        db.revoke_auth_session(token_hash)
    db.insert_audit(AuditAction.LOGOUT, user_id=principal.user_id,
                    status=AuditStatus.SUCCESS)
    return {"status": "signed out"}


@auth_router.get("/me")
def me(principal: Principal = Depends(deps.require_user)):
    """Who the backend believes is calling.

    The interface calls this on load to restore a session, and the answer is
    the backend's, not the browser's: a tampered local copy of the role
    changes nothing, because this route reads the database.
    """
    return {"user": principal.to_public(), **deps.get_settings().to_public()}


# ==========================================================================
# User administration
# ==========================================================================

def _public_user(row) -> dict:
    """One user row as the admin screen reads it. No hash, ever."""
    locked_until = row["locked_until"] if "locked_until" in row.keys() else None
    now = time.time()
    return {
        "user_id": row["user_id"],
        "username": row["username"],
        "role": row["role"],
        "is_active": bool(row["is_active"]),
        "status": "ACTIVE" if row["is_active"] else "DISABLED",
        "locked": bool(locked_until and now < float(locked_until)),
        "failed_attempts": int(row["failed_attempts"] or 0),
        "last_login_ts": row["last_login_ts"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


@admin_router.get("/users")
def list_users():
    return [_public_user(r) for r in deps.get_db().list_users()]


@admin_router.post("/users", status_code=status.HTTP_201_CREATED)
def create_user(payload: CreateUserRequest,
                principal: Principal = Depends(deps.require_admin)):
    db = deps.get_db()
    try:
        user_id = validate_user_id(payload.user_id)
        username = validate_username(payload.username)
        role = validate_role(payload.role)
        validate_password(payload.password)
    except PasswordError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=str(exc))

    if db.get_user(user_id) is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail=f"User {user_id} already exists.")

    db.create_user(user_id, username, hash_password(payload.password),
                   role, payload.is_active)
    db.insert_audit(AuditAction.USER_CREATED, user_id=principal.user_id,
                    resource=user_id,
                    detail={"role": role, "is_active": payload.is_active})
    return _public_user(db.get_user(user_id))


@admin_router.put("/users/{target_id}")
def update_user(target_id: str, payload: UpdateUserRequest,
                principal: Principal = Depends(deps.require_admin)):
    db = deps.get_db()
    target_id = (target_id or "").strip().upper()
    row = db.get_user(target_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="No such user.")

    updates: dict = {}
    actions: list[str] = []

    try:
        if payload.username is not None:
            updates["username"] = validate_username(payload.username)
        if payload.role is not None:
            updates["role"] = validate_role(payload.role)
        if payload.password is not None:
            validate_password(payload.password)
            updates["password_hash"] = hash_password(payload.password)
            actions.append(AuditAction.PASSWORD_RESET)
    except PasswordError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=str(exc))

    if payload.is_active is not None:
        updates["is_active"] = 1 if payload.is_active else 0
        actions.append(AuditAction.USER_ENABLED if payload.is_active
                       else AuditAction.USER_DISABLED)

    # The lockout check. Disabling or demoting the last active administrator
    # would leave a system nobody can administer and no way back in short of
    # editing the database by hand, so it is refused - including when an
    # administrator does it to their own account by accident.
    _guard_last_admin(db, row, updates)

    if not updates:
        return _public_user(row)

    db.update_user(target_id, **updates)

    # A changed password or a disabled account must not leave that person
    # signed in somewhere on an old token.
    if "password_hash" in updates or updates.get("is_active") == 0 \
            or "role" in updates:
        db.revoke_user_sessions(target_id)

    for action in actions or [AuditAction.USER_UPDATED]:
        db.insert_audit(action, user_id=principal.user_id, resource=target_id,
                        detail={"fields": sorted(
                            k for k in updates if k != "password_hash")})
    if not actions:
        pass
    return _public_user(db.get_user(target_id))


@admin_router.delete("/users/{target_id}")
def delete_user(target_id: str,
                principal: Principal = Depends(deps.require_admin)):
    db = deps.get_db()
    target_id = (target_id or "").strip().upper()
    row = db.get_user(target_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="No such user.")
    if target_id == principal.user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot delete the account you are signed in with.")
    _guard_last_admin(db, row, {"is_active": 0})

    db.delete_user(target_id)
    db.insert_audit(AuditAction.USER_DELETED, user_id=principal.user_id,
                    resource=target_id, detail={"role": row["role"]})
    return {"status": "deleted", "user_id": target_id}


def _guard_last_admin(db, row, updates: dict) -> None:
    """Refuse a change that would remove the last usable administrator."""
    if row["role"] != SUPER_ADMIN:
        return
    losing_admin = (
        updates.get("is_active") == 0
        or (updates.get("role") and updates["role"] != SUPER_ADMIN)
    )
    if not losing_admin:
        return
    if not int(row["is_active"] or 0):
        return          # already inactive; it was not the last usable one
    if db.count_admins(active_only=True) <= 1:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This is the only active Super Admin. Create another "
                   "before disabling or demoting this one.",
        )


@admin_router.post("/users/{target_id}/unlock")
def unlock_user(target_id: str,
                principal: Principal = Depends(deps.require_admin)):
    """Clear a failed-attempt lockout without waiting for the cooldown."""
    db = deps.get_db()
    target_id = (target_id or "").strip().upper()
    if db.get_user(target_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="No such user.")
    db.clear_lockout(target_id)
    db.insert_audit(AuditAction.PRIVILEGED_ACTION, user_id=principal.user_id,
                    resource=target_id, detail={"action": "lockout cleared"})
    return _public_user(db.get_user(target_id))


# ==========================================================================
# Audit log
# ==========================================================================

@admin_router.get("/audit-logs")
def audit_logs(user_id: Optional[str] = None, action: Optional[str] = None,
               status_filter: Optional[str] = None, limit: int = 200):
    import json as _json
    rows = deps.get_db().get_audit_logs(
        user_id=user_id, action=action, status=status_filter,
        limit=max(1, min(int(limit), 1000)),
    )
    out = []
    for row in rows:
        detail = None
        if row["detail"]:
            try:
                detail = _json.loads(row["detail"])
            except (TypeError, ValueError):
                detail = None
        ts_text = str(row["ts_text"] or "")
        out.append({
            "id": row["id"],
            "ts": row["ts"],
            "ts_text": ts_text,
            "time": ts_text.split(" ")[-1] if ts_text else "",
            "user_id": row["user_id"],
            "action": row["action"],
            "resource": row["resource"],
            "status": row["status"],
            "detail": detail,
        })
    return out
