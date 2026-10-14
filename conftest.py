"""Pytest configuration and environment setup."""

import sys
from pathlib import Path

# Ensure src/ is on sys.path for test discovery
root_dir = Path(__file__).resolve().parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
