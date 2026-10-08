"""``python -m fmha_sm100.icp.abi`` -- run the vendored ABI's self-check.

No torch, no CUDA, no device. It prints the summary and the recomputed digest
and exits non-zero if either the self-check fails or the digest has moved
without :data:`fmha_sm100.icp.abi.ABI_DIGEST` moving with it.
"""

from __future__ import annotations

import sys

from . import ABI_DIGEST, abi_digest, self_check, to_json


def main() -> int:
    summary = self_check()
    print(to_json())
    print(f"checks: {summary.checks}")
    digest = abi_digest()
    print(f"digest: {digest}")
    if digest != ABI_DIGEST:
        print(
            f"DIGEST MISMATCH: expected {ABI_DIGEST}. The vendored ABI has "
            "been edited; bump ABI_VERSION and re-review with every dependent "
            "owner, or restore the frozen copy.",
            file=sys.stderr,
        )
        return 1
    print("SELF-CHECK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
