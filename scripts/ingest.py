"""KGK AI — Document Ingestion Script.

Loads documents from knowledge/documents/, processes them, and builds
the FAISS vector store for RAG retrieval.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --dir custom/docs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def main() -> None:
    parser = argparse.ArgumentParser(description="KGK AI — Document Ingestion")
    parser.add_argument(
        "--dir",
        default="knowledge/documents",
        help="Directory containing source documents",
    )
    args = parser.parse_args()

    docs_dir = project_root / args.dir

    if not docs_dir.exists():
        print(f"ERROR: Directory not found: {docs_dir}")
        sys.exit(1)

    # Count documents
    extensions = {".txt", ".md", ".pdf", ".json"}
    files = [f for f in docs_dir.rglob("*") if f.suffix.lower() in extensions]

    if not files:
        print(f"No documents found in {docs_dir}")
        print(f"Supported formats: {', '.join(extensions)}")
        print("Place documents in knowledge/documents/ and re-run.")
        sys.exit(0)

    print(f"Found {len(files)} document(s) in {docs_dir}")
    for f in files:
        print(f"  - {f.relative_to(project_root)}")

    print("\nIngestion pipeline will be implemented in Phase 6 (RAG).")
    print("This script is a placeholder for Phase 1.")


if __name__ == "__main__":
    main()
