#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NIBI_ROOT = REPO_ROOT / "nibi_model_compare"
for candidate in (str(NIBI_ROOT), str(REPO_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from postprocess_framewise_runner import main


if __name__ == "__main__":
    raise SystemExit(main())
