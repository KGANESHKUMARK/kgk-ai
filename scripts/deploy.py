"""KGK AI — Deployment Helper Script.

Assists with deploying KGK AI to Hugging Face Spaces.

Usage:
    python scripts/deploy.py --space-id your-username/kgk-ai
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="KGK AI — Deployment Helper")
    parser.add_argument("--space-id", required=True, help="Hugging Face Space ID (username/space-name)")
    parser.add_argument("--token", default=None, help="Hugging Face token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    print("=" * 50)
    print("KGK AI — Hugging Face Deployment")
    print("=" * 50)

    print(f"\nTarget Space: {args.space_id}")
    print("\nDeployment steps:")
    print("  1. Create a Space on Hugging Face (SDK: Gradio, Hardware: ZeroGPU)")
    print("  2. Add this repo as a remote:")
    print(f"     git remote add space https://huggingface.co/spaces/{args.space_id}")
    print("  3. Push to the Space:")
    print("     git push space main")
    print("  4. Set environment variables in Space Settings → Variables and secrets")
    print("  5. Monitor build logs in the Space's Logs tab")

    print("\nDeployment helper will be fully implemented in Phase 11 (HF Deployment).")
    print("This script is a placeholder for Phase 1.")


if __name__ == "__main__":
    main()
