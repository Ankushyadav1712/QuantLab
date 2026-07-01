"""Python entry point for the `alphatest` CLI.

Forwards into `cli.__main__:main`.  Existed before as separate scripts
(`shuffle_test.py`, `verify_alphas.py`) — they remain for back-compat,
but new docs point here.
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from cli.__main__ import main  # noqa: E402 — sys.path tweak above is intentional

if __name__ == "__main__":
    sys.exit(main())
