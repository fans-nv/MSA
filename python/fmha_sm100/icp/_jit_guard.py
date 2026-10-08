"""Runtime-JIT policy: in a prewarmed image, a compile while serving is an alert.

Every JIT entry point in this package calls :func:`on_compile` immediately
before it actually compiles (never on a cache hit). The policy is
``ICP_RUNTIME_JIT``:

    allow   log at INFO and compile (default when no prewarm manifest exists)
    warn    log a WARNING and compile (default when the active cache root has a
            prewarm manifest for this arch, i.e. inside a prewarmed image)
    fail    raise :class:`RuntimeJitError` instead of compiling

``fmha_sm100.icp.prewarm`` runs with ``allow``. :func:`events` returns every
compile this process attempted, for a serving stack to report.
"""

from __future__ import annotations

import logging
import os
import threading

POLICY_ENV = "ICP_RUNTIME_JIT"
POLICIES = ("allow", "warn", "fail")

logger = logging.getLogger("fmha_sm100.icp.jit")
_events: list[dict[str, str]] = []
_lock = threading.Lock()


class RuntimeJitError(RuntimeError):
    """A kernel was about to be JIT-compiled while ``ICP_RUNTIME_JIT=fail``."""


def policy(arch: str | None = None) -> str:
    value = os.environ.get(POLICY_ENV, "").strip().lower()
    if value:
        if value not in POLICIES:
            raise ValueError(f"{POLICY_ENV}={value!r}; expected one of {POLICIES}")
        return value
    from . import _cache  # noqa: PLC0415

    try:
        prewarmed = _cache.manifest_path(arch).is_file()
    except Exception:  # noqa: BLE001 - an unreadable root is not a prewarmed one
        prewarmed = False
    return "warn" if prewarmed else "allow"


def on_compile(component: str, name: str, *, arch: str | None = None,
               where: str = "") -> None:
    """Record, and apply the policy to, one imminent compile."""
    mode = policy(arch)
    event = {"component": component, "name": name, "arch": arch or "",
             "where": where, "policy": mode}
    with _lock:
        _events.append(event)
    message = (f"[fmha_sm100.icp] runtime JIT of {component}:{name}"
               f"{' for sm_' + arch if arch else ''}{' in ' + where if where else ''}")
    if mode == "fail":
        raise RuntimeJitError(
            f"{message} refused ({POLICY_ENV}=fail): the prewarmed cache has no "
            "artifact for these sources and arch. Run `python -m "
            "fmha_sm100.icp.prewarm verify` to see which component is missing.")
    if mode == "warn":
        logger.warning("%s: not in the prewarmed cache; compiling on the "
                       "serving path", message)
    else:
        logger.info("%s", message)


def events() -> list[dict[str, str]]:
    with _lock:
        return [dict(e) for e in _events]
