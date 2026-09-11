"""Passwords and session tokens.

Deliberately built on the standard library alone. The venv this runs in has
no bcrypt, no passlib and no JWT library, and the machine is on Python 3.14
where a C-extension build can fail at the worst possible moment. Adding a
dependency that must compile the night before a demonstration is a risk with
no security benefit here, because everything needed is already in `hashlib`
and `secrets`.

Two decisions worth stating plainly:

* **scrypt, not SHA-256.** A bare hash of a password is broken by design:
  modern hardware tries billions of SHA-256 guesses a second, so a stolen
  table of them is a stolen table of passwords. scrypt is deliberately slow
  AND deliberately memory-hungry, which is what defeats the GPU and ASIC
  attacks that make plain hashing useless. It has been in the standard
  library since 3.6.

* **Opaque server-side tokens, not JWT.** A JWT cannot be revoked: it stays
  valid until it expires, whatever the server thinks, so "log out" becomes a
  polite request the client may ignore. Making logout real means keeping a
  server-side record to check against - and once that record exists, the
  signed token adds nothing. A random token plus a row is simpler, shorter,
  and actually revocable.

Nothing in this module imports from `core.pipeline` or any detection code,
and nothing in the detection path imports this. That separation is the point.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from dataclasses import dataclass
from typing import Optional

# --------------------------------------------------------------------------
# Roles
# --------------------------------------------------------------------------
# Two, and only two. Every access decision in the system reduces to "is this
# SUPER_ADMIN", so a third role would mean revisiting every check.

SUPER_ADMIN = "SUPER_ADMIN"
NORMAL_USER = "NORMAL_USER"
ROLES = (SUPER_ADMIN, NORMAL_USER)

ROLE_LABELS = {
    SUPER_ADMIN: "SUPER ADMIN",
    NORMAL_USER: "NORMAL USER",
}

ROLE_TAGLINES = {
    SUPER_ADMIN: "FULL SYSTEM ACCESS",
    NORMAL_USER: "OPERATOR ACCESS",
}


# --------------------------------------------------------------------------
# Password hashing
# --------------------------------------------------------------------------

#: scrypt cost parameters. n is the work factor, r the block size, p the
#: parallelisation. 2**15 with r=8 costs roughly 32 MB and ~100 ms per hash
#: on a laptop - slow enough to make guessing expensive, fast enough that a
#: sign-in feels instant. Raising n later does not strand existing accounts:
#: the parameters travel inside each stored hash, so old ones keep verifying
#: under the values they were made with.
SCRYPT_N = 2 ** 15
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_DKLEN = 64
SALT_BYTES = 16

#: Rejected outright rather than silently truncated. scrypt has no practical
#: length limit, but an unbounded input is an easy way to make the server do
#: 32 MB of work per request, which is a denial of service with extra steps.
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 256


class PasswordError(ValueError):
    """A password that cannot be accepted, with a reason fit to show a user."""


def validate_password(password: str) -> None:
    """Raise PasswordError if this password may not be set.

    Length only. Composition rules ("one capital, one symbol") push people
    towards predictable substitutions and shorter passwords; length is what
    actually costs an attacker.
    """
    if not isinstance(password, str) or not password:
        raise PasswordError("A password is required.")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise PasswordError(
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    if len(password) > MAX_PASSWORD_LENGTH:
        raise PasswordError(
            f"Password must be at most {MAX_PASSWORD_LENGTH} characters.")


def hash_password(password: str) -> str:
    """Hash a password for storage.

    Returns a self-describing string:

        scrypt$<n>$<r>$<p>$<salt hex>$<digest hex>

    The salt is fresh per password, so two people who choose the same one do
    not share a hash, and a precomputed table is useless against either.
    """
    validate_password(password)
    salt = secrets.token_bytes(SALT_BYTES)
    digest = hashlib.scrypt(
        password.encode("utf-8"), salt=salt,
        n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=SCRYPT_DKLEN,
        maxmem=SCRYPT_N * SCRYPT_R * 256,
    )
    return (f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}$"
            f"{salt.hex()}${digest.hex()}")


def verify_password(password: str, stored: str) -> bool:
    """Is this the password behind that stored hash?

    Compared with `hmac.compare_digest`, not `==`. An ordinary comparison
    returns as soon as two bytes differ, and the time it took leaks how much
    of the guess was right - enough, over many attempts, to recover a hash
    byte by byte.

    A malformed or empty stored value returns False rather than raising: a
    corrupt row should fail the sign-in, not take the endpoint down.
    """
    if not password or not stored:
        return False
    try:
        algorithm, n, r, p, salt_hex, digest_hex = stored.split("$")
        if algorithm != "scrypt":
            return False
        n, r, p = int(n), int(r), int(p)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
    except (ValueError, AttributeError):
        return False

    try:
        candidate = hashlib.scrypt(
            password.encode("utf-8"), salt=salt,
            n=n, r=r, p=p, dklen=len(expected),
            maxmem=n * r * 256,
        )
    except (ValueError, MemoryError):
        return False
    return hmac.compare_digest(candidate, expected)


def looks_hashed(value: str) -> bool:
    """Is this string a stored hash rather than a plain password?

    Used by the tests that assert nothing readable reached the database.
    """
    return isinstance(value, str) and value.startswith("scrypt$") \
        and value.count("$") == 5


# --------------------------------------------------------------------------
# Session tokens
# --------------------------------------------------------------------------

#: 32 bytes from the OS CSPRNG, url-safe encoded. Guessing one is not a
#: realistic attack at this size, which is why the token carries no structure
#: and no claims: it is a lookup key, and the row it finds is the authority.
TOKEN_BYTES = 32


def new_token() -> str:
    """A fresh, unguessable session token."""
    return secrets.token_urlsafe(TOKEN_BYTES)


def token_fingerprint(token: str) -> str:
    """What gets stored for a token. Never the token itself.

    A session token is a bearer credential: whoever holds it IS the user
    until it expires. Storing them in plain text would mean a copy of the
    database is a set of working logins. Storing the SHA-256 instead makes
    the table useless to anyone who reads it, and costs one hash per request
    to check.

    Plain SHA-256 is right here where it would be wrong for a password: the
    input is 32 random bytes, so there is no guessing to slow down.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Identifiers
# --------------------------------------------------------------------------

#: ADMIN-001, USER-002, OPERATOR-07. Constrained so a user id cannot carry
#: anything surprising into a log line, a header or a query.
USER_ID_PATTERN = re.compile(r"^[A-Z][A-Z0-9_-]{2,31}$")
USERNAME_PATTERN = re.compile(r"^[\w .'-]{2,64}$", re.UNICODE)


def validate_user_id(user_id: str) -> str:
    """Normalise and check a user id, or raise PasswordError with a reason."""
    candidate = (user_id or "").strip().upper()
    if not USER_ID_PATTERN.match(candidate):
        raise PasswordError(
            "User ID must be 3-32 characters: letters, digits, hyphen or "
            "underscore, starting with a letter.")
    return candidate


def validate_username(username: str) -> str:
    candidate = (username or "").strip()
    if not USERNAME_PATTERN.match(candidate):
        raise PasswordError("Username must be 2-64 ordinary characters.")
    return candidate


def validate_role(role: str) -> str:
    candidate = (role or "").strip().upper()
    if candidate not in ROLES:
        raise PasswordError(f"Role must be one of: {', '.join(ROLES)}.")
    return candidate


# --------------------------------------------------------------------------
# The authenticated caller
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Principal:
    """Who the backend has decided is making this request.

    Built only from a database row reached through a valid session token.
    Nothing here is ever taken from a request body or a header the client
    controls - that is the whole point of the type. A client can claim to be
    an admin; it cannot produce one of these.
    """

    user_id: str
    username: str
    role: str

    @property
    def is_admin(self) -> bool:
        return self.role == SUPER_ADMIN

    @property
    def role_label(self) -> str:
        return ROLE_LABELS.get(self.role, self.role)

    @property
    def role_tagline(self) -> str:
        return ROLE_TAGLINES.get(self.role, "")

    def to_public(self) -> dict:
        """What the frontend is told about the signed-in user."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "role": self.role,
            "role_label": self.role_label,
            "role_tagline": self.role_tagline,
            "is_admin": self.is_admin,
        }


# --------------------------------------------------------------------------
# Audit actions
# --------------------------------------------------------------------------
# Named constants rather than free strings, so a typo cannot quietly create a
# new action nobody ever filters for.

class AuditAction:
    LOGIN_SUCCESS = "LOGIN_SUCCESS"
    LOGIN_FAILED = "LOGIN_FAILED"
    LOGIN_BLOCKED = "LOGIN_BLOCKED"
    LOGOUT = "LOGOUT"
    SESSION_EXPIRED = "SESSION_EXPIRED"
    USER_CREATED = "USER_CREATED"
    USER_UPDATED = "USER_UPDATED"
    USER_DISABLED = "USER_DISABLED"
    USER_ENABLED = "USER_ENABLED"
    USER_DELETED = "USER_DELETED"
    PASSWORD_RESET = "PASSWORD_RESET"
    CAMERA_CONFIGURATION_CHANGED = "CAMERA_CONFIGURATION_CHANGED"
    CAMERA_STARTED = "CAMERA_STARTED"
    CAMERA_STOPPED = "CAMERA_STOPPED"
    SYSTEM_SETTING_CHANGED = "SYSTEM_SETTING_CHANGED"
    PRIVILEGED_ACTION = "PRIVILEGED_ACTION"
    ACCESS_DENIED = "ACCESS_DENIED"
    EVIDENCE_EXPORTED = "EVIDENCE_EXPORTED"


class AuditStatus:
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    DENIED = "DENIED"
