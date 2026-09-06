"""KGK AI — Environment Setup Script.

Verifies Python version, creates virtual environment, installs dependencies,
and creates .env from template.

Usage:
    python scripts/setup.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def check_python_version() -> bool:
    """Verify Python version is 3.10+."""
    version = sys.version_info
    if version < (3, 10):
        print(f"ERROR: Python 3.10+ required, found {version.major}.{version.minor}")
        return False
    print(f"OK: Python {version.major}.{version.minor}.{version.micro}")
    return True


def create_venv(project_root: Path) -> bool:
    """Create virtual environment if it doesn't exist."""
    venv_path = project_root / "venv"
    if venv_path.exists():
        print("OK: Virtual environment already exists")
        return True

    print("Creating virtual environment...")
    result = subprocess.run(
        [sys.executable, "-m", "venv", str(venv_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: Failed to create venv: {result.stderr}")
        return False
    print("OK: Virtual environment created")
    return True


def install_deps(project_root: Path) -> bool:
    """Install dependencies."""
    req_file = project_root / "requirements-dev.txt"
    if not req_file.exists():
        req_file = project_root / "requirements.txt"

    print(f"Installing dependencies from {req_file.name}...")
    pip_exe = str(project_root / "venv" / "Scripts" / "pip")
    if not os.path.exists(pip_exe):
        pip_exe = str(project_root / "venv" / "bin" / "pip")

    result = subprocess.run(
        [pip_exe, "install", "-r", str(req_file)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: Failed to install dependencies: {result.stderr[-500:]}")
        return False
    print("OK: Dependencies installed")
    return True


def create_env_file(project_root: Path) -> bool:
    """Create .env from .env.example if it doesn't exist."""
    env_path = project_root / ".env"
    example_path = project_root / ".env.example"

    if env_path.exists():
        print("OK: .env already exists")
        return True

    if not example_path.exists():
        print("WARNING: .env.example not found, skipping .env creation")
        return True

    env_path.write_text(example_path.read_text(), encoding="utf-8")
    print("OK: .env created from .env.example (edit with your settings)")
    return True


def create_directories(project_root: Path) -> None:
    """Ensure required directories exist."""
    dirs = [
        "knowledge/documents",
        "knowledge/processed",
        "knowledge/vectorstore",
        "data/conversations",
        "data/datasets",
        "logs",
    ]
    for d in dirs:
        path = project_root / d
        path.mkdir(parents=True, exist_ok=True)
    print("OK: Directories verified")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent

    print("=" * 50)
    print("KGK AI — Environment Setup")
    print("=" * 50)

    steps = [
        ("Checking Python version", check_python_version),
        ("Creating directories", lambda: create_directories(project_root)),
        ("Creating .env file", lambda: create_env_file(project_root)),
    ]

    for name, func in steps:
        print(f"\n--- {name} ---")
        if not func():
            print("\nSetup incomplete. Please fix the errors above.")
            sys.exit(1)

    # Virtual environment and deps (optional if user manages their own)
    print("\n--- Creating virtual environment ---")
    if create_venv(project_root):
        print("\n--- Installing dependencies ---")
        install_deps(project_root)

    print("\n" + "=" * 50)
    print("Setup complete!")
    print("=" * 50)
    print("\nNext steps:")
    print("  1. Edit .env with your settings")
    print("  2. Activate the virtual environment:")
    print("     Windows:  venv\\Scripts\\activate")
    print("     Linux:    source venv/bin/activate")
    print("  3. Run:  python -m app.main")
    print()


if __name__ == "__main__":
    main()
