"""Pytest configuration for KGK AI tests."""

import sys
from pathlib import Path

# Ensure project root is in Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
