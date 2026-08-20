#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NIBI_ROOT = REPO_ROOT / "nibi_model_compare"
for candidate in (str(NIBI_ROOT), str(REPO_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import postprocess_framewise_runner as _runner


main = _runner.main


def __getattr__(name: str):
    """Preserve the legacy helper API while this file remains a CLI wrapper.

    Several tests and downstream scripts import helper functions from this module.
    Delegate those lookups dynamically so runtime-populated globals such as
    ``encode_binary_mask_to_rle`` remain in sync with the consolidated runner.
    """
    return getattr(_runner, name)


if __name__ == "__main__":
    raise SystemExit(main())
