"""Roles and permissions.

Three roles, and the boundaries between them are security decisions rather than UI convenience:

  analyst   triage, record verdicts, read alerts. Cannot close incidents, cannot create
            suppressions, cannot see raw IP addresses.
  senior    everything an analyst can do, plus closing incidents, creating suppression rules, and
            **promoting verdicts into the retraining pool**.
  admin     everything, plus PII re-identification, key rotation and model promotion.

Two of those splits are load-bearing:

**Verdict promotion requires `senior`.** The analyst feedback loop is a poisoning vector we
introduced ourselves - a compromised analyst account can label its own traffic benign until the
model learns to ignore it (THREAT_MODEL T1). Separating "record a verdict" from "let that verdict
train the model" is what makes the attack require two compromised accounts instead of one.

**Suppression requires `senior` and always carries an expiry.** A permanent suppression is a
permanent blind spot. The vulnerability scanner allowlisted in March is the C2 channel missed in
September.

Row-level scoping: an analyst sees only incidents on network segments assigned to their team. Not a
UI filter - it is applied in the repository query, so an analyst who calls the API directly gets the
same answer as one using the console.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Role(StrEnum):
    ANALYST = "analyst"
    SENIOR = "senior"
    ADMIN = "admin"


class Permission(StrEnum):
    READ_ALERTS = "alerts:read"
    READ_INCIDENTS = "incidents:read"
    TRIAGE = "incidents:triage"
    RECORD_VERDICT = "verdict:record"

    CLOSE_INCIDENT = "incidents:close"
    PROMOTE_VERDICT = "verdict:promote"  # let a verdict enter the retraining pool
    CREATE_SUPPRESSION = "suppression:create"

    REIDENTIFY_PII = "pii:reidentify"
    PROMOTE_MODEL = "model:promote"
    CHANGE_THRESHOLD = "threshold:change"
    ROTATE_KEY = "key:rotate"
    READ_AUDIT = "audit:read"


_ANALYST: frozenset[Permission] = frozenset(
    {
        Permission.READ_ALERTS,
        Permission.READ_INCIDENTS,
        Permission.TRIAGE,
        Permission.RECORD_VERDICT,
    }
)

_SENIOR: frozenset[Permission] = _ANALYST | frozenset(
    {
        Permission.CLOSE_INCIDENT,
        Permission.PROMOTE_VERDICT,
        Permission.CREATE_SUPPRESSION,
        Permission.READ_AUDIT,
    }
)

_ADMIN: frozenset[Permission] = _SENIOR | frozenset(
    {
        Permission.REIDENTIFY_PII,
        Permission.PROMOTE_MODEL,
        Permission.CHANGE_THRESHOLD,
        Permission.ROTATE_KEY,
    }
)

ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.ANALYST: _ANALYST,
    Role.SENIOR: _SENIOR,
    Role.ADMIN: _ADMIN,
}


@dataclass(frozen=True)
class Principal:
    """An authenticated caller."""

    username: str
    role: Role
    # Network segments this principal may see. Empty means all - which is why it is empty only for
    # senior and admin.
    segments: frozenset[str] = field(default_factory=frozenset)

    def can(self, permission: Permission) -> bool:
        return permission in ROLE_PERMISSIONS[self.role]

    def may_see_segment(self, segment: str | None) -> bool:
        """Row-level scoping. No segment restriction means unrestricted."""
        if not self.segments:
            return True
        return segment is None or segment in self.segments


class PermissionDenied(PermissionError):
    def __init__(self, principal: Principal, permission: Permission) -> None:
        self.principal = principal
        self.permission = permission
        super().__init__(
            f"role {principal.role.value!r} lacks {permission.value!r}. "
            f"Required role: {required_role(permission).value}"
        )


def require(principal: Principal, permission: Permission) -> None:
    """Raise unless the principal holds the permission."""
    if not principal.can(permission):
        raise PermissionDenied(principal, permission)


def required_role(permission: Permission) -> Role:
    """Lowest role that holds this permission, for error messages that are actually helpful."""
    for role in (Role.ANALYST, Role.SENIOR, Role.ADMIN):
        if permission in ROLE_PERMISSIONS[role]:
            return role
    return Role.ADMIN


def matrix() -> str:
    """The permission matrix, rendered. Goes on the governance page."""
    perms = sorted(Permission, key=lambda p: p.value)
    width = max(len(p.value) for p in perms) + 2
    lines = [
        f"  {'permission':<{width}} {'analyst':>9} {'senior':>8} {'admin':>7}",
        f"  {'-' * width} {'-' * 9} {'-' * 8} {'-' * 7}",
    ]
    for perm in perms:
        marks = [
            "  yes  " if perm in ROLE_PERMISSIONS[r] else "   -   "
            for r in (Role.ANALYST, Role.SENIOR, Role.ADMIN)
        ]
        lines.append(f"  {perm.value:<{width}} {marks[0]:>9} {marks[1]:>8} {marks[2]:>7}")
    lines += [
        "",
        "  Two boundaries that are security decisions rather than UI choices:",
        "    verdict:promote    a compromised analyst account cannot teach the model on its own",
        "    suppression:create a permanent blind spot needs an owner, and every rule has an expiry",
    ]
    return "\n".join(lines)
