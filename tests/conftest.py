# QRNG Project — conftest.py
# Adds the project root to sys.path so imports resolve correctly.

import sys
from pathlib import Path

# Add qrng-project to sys.path
sys.path.insert(0, str(Path(__file__).parent))
