#!/usr/bin/env python3
"""CLI wrapper for ``zxing-cpp-sr``."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zxing_cpp_sr.decoder import main


if __name__ == "__main__":
    raise SystemExit(main())
