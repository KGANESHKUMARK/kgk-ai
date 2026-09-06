"""KGK AI — Document Ingestion Script.

Loads documents from knowledge/documents/, processes them, and builds
the FAISS vector store for RAG retrieval.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --dir custom/docs
    python scripts/ingest.py --force
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild (ignore cached vector store)",
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

    print("\nStarting ingestion pipeline...")
    try:
        from app.logging_config import setup_logging
        setup_logging()

        from app.rag.pipeline import RAGPipeline

        pipeline = RAGPipeline()
        chunk_count = pipeline.ingest_directory(str(docs_dir))

        if chunk_count > 0:
            print(f"\nIngestion complete: {chunk_count} chunks stored in vector store.")
            print(f"Vector store saved to: {project_root / 'knowledge' / 'vectorstore'}")
        else:
            print("\nNo chunks were generated. Check document content.")

    except ImportError as e:
        print(f"\nERROR: Missing dependency: {e}")
        print("Install with: pip install sentence-transformers faiss-cpu PyPDF2")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Ingestion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
