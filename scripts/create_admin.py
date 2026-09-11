#!/usr/bin/env python3
"""Create or reset the initial Super Admin account.

    python scripts/create_admin.py

Prompts for a password without echoing it. For an unattended setup, read it
from the environment instead:

    SUPER_ADMIN_USER=ADMIN-001 \
    SUPER_ADMIN_NAME="Nikhil" \
    SUPER_ADMIN_PASSWORD='...' \
    python scripts/create_admin.py

WHY THIS IS A SCRIPT AND NOT A CONFIG VALUE
-------------------------------------------
A password in config/application.yaml is a password in Git, and a password in
Git is public the moment the repository is shared with a judge, a teammate or
a marketplace. It cannot be un-shared by deleting it later, because the
history keeps it. This script never writes the password anywhere: it hashes
it, stores the hash, and forgets the plain text when the process exits.

The password is also never echoed, so it does not end up in a terminal
scrollback or a screen recording of the demonstration.
"""

from __future__ import annotations

import getpass
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.auth import (                                    # noqa: E402
    AuditAction,
    PasswordError,
    SUPER_ADMIN,
    hash_password,
    validate_password,
    validate_user_id,
    validate_username,
)
from core.config import load_config                        # noqa: E402
from database.database import Database                     # noqa: E402


def read_password() -> str:
    """From the environment if set, otherwise prompted twice, never echoed."""
    from_env = os.environ.get("SUPER_ADMIN_PASSWORD")
    if from_env:
        print("Using SUPER_ADMIN_PASSWORD from the environment.")
        return from_env

    while True:
        first = getpass.getpass("Password (not shown): ")
        try:
            validate_password(first)
        except PasswordError as exc:
            print(f"  {exc}")
            continue
        second = getpass.getpass("Confirm password: ")
        if first != second:
            print("  Passwords do not match. Try again.")
            continue
        return first


def main() -> int:
    config = load_config()
    db = Database(config.path("database_file"))

    try:
        user_id = validate_user_id(
            os.environ.get("SUPER_ADMIN_USER")
            or input("Super Admin User ID [ADMIN-001]: ").strip()
            or "ADMIN-001"
        )
        username = validate_username(
            os.environ.get("SUPER_ADMIN_NAME")
            or input(f"Display name [{user_id}]: ").strip()
            or user_id
        )
    except PasswordError as exc:
        print(f"\n{exc}")
        return 2

    existing = db.get_user(user_id)
    if existing is not None:
        print(f"\n{user_id} already exists (role {existing['role']}, "
              f"{'active' if existing['is_active'] else 'disabled'}).")
        answer = input("Reset its password and re-enable it? [y/N]: ")
        if answer.strip().lower() not in ("y", "yes"):
            print("Nothing changed.")
            return 0

    password = read_password()

    try:
        password_hash = hash_password(password)
    except PasswordError as exc:
        print(f"\n{exc}")
        return 2
    finally:
        # Not security theatre: it shortens the window in which the plain
        # password sits in this process's memory.
        del password

    if existing is None:
        db.create_user(user_id, username, password_hash, SUPER_ADMIN, True)
        db.insert_audit(AuditAction.USER_CREATED, user_id=user_id,
                        resource=user_id,
                        detail={"role": SUPER_ADMIN, "via": "create_admin.py"})
        print(f"\nCreated Super Admin {user_id} ({username}).")
    else:
        db.update_user(user_id, password_hash=password_hash,
                       username=username, role=SUPER_ADMIN, is_active=1,
                       failed_attempts=0, locked_until=None)
        # Any session opened under the old password stops working now.
        db.revoke_user_sessions(user_id)
        db.insert_audit(AuditAction.PASSWORD_RESET, user_id=user_id,
                        resource=user_id, detail={"via": "create_admin.py"})
        print(f"\nReset Super Admin {user_id} ({username}).")

    print("The password was hashed with scrypt and is not stored anywhere "
          "in readable form.")
    print(f"Sign in at the dashboard with User ID: {user_id}")
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
