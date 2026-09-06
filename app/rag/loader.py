"""KGK AI Document Loader — Multi-format document loading.

Supports TXT, MD, PDF, and JSON files. Returns raw text with metadata
for downstream chunking and embedding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from app.logging_config import get_logger

logger = get_logger("rag.loader")

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".json"}


@dataclass
class LoadedDocument:
    """A loaded document with raw text and metadata.

    Attributes:
        content: Raw text content of the document.
        source: Absolute path to the source file.
        filename: Name of the source file.
        document_id: Unique identifier (based on filename).
        file_type: File extension (e.g. '.txt', '.pdf').
        metadata: Additional metadata.
    """

    content: str
    source: str = ""
    filename: str = ""
    document_id: str = ""
    file_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentLoader:
    """Loads documents from the filesystem.

    Supports TXT, MD, PDF, and JSON formats. Each format has its own
    extraction logic. Unknown formats are skipped with a warning.
    """

    def load_file(self, file_path: str | Path) -> Optional[LoadedDocument]:
        """Load a single document from a file path.

        Args:
            file_path: Path to the document file.

        Returns:
            LoadedDocument instance or None if loading fails.
        """
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"File not found: {path}")
            return None

        if not path.is_file():
            logger.warning(f"Not a file: {path}")
            return None

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            logger.warning(f"Unsupported file type '{ext}': {path}")
            return None

        loader_fn = {
            ".txt": self._load_text,
            ".md": self._load_text,
            ".pdf": self._load_pdf,
            ".json": self._load_json,
        }.get(ext)

        if loader_fn is None:
            return None

        try:
            content = loader_fn(path)
            if not content or not content.strip():
                logger.warning(f"Empty content from: {path}")
                return None

            doc = LoadedDocument(
                content=content,
                source=str(path.resolve()),
                filename=path.name,
                document_id=path.stem,
                file_type=ext,
                metadata={"file_size": path.stat().st_size},
            )
            logger.info(
                f"Loaded: {path.name} ({len(content)} chars)",
                extra={"component": "rag.loader", "filename": path.name, "chars": len(content)},
            )
            return doc
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            return None

    def load_directory(self, dir_path: str | Path) -> list[LoadedDocument]:
        """Load all supported documents from a directory (recursive).

        Args:
            dir_path: Path to the directory.

        Returns:
            List of LoadedDocument instances.
        """
        path = Path(dir_path)
        if not path.exists() or not path.is_dir():
            logger.warning(f"Directory not found: {path}")
            return []

        documents = []
        for ext in SUPPORTED_EXTENSIONS:
            for file_path in path.rglob(f"*{ext}"):
                doc = self.load_file(file_path)
                if doc is not None:
                    documents.append(doc)

        logger.info(
            f"Loaded {len(documents)} document(s) from {path}",
            extra={"component": "rag.loader", "count": len(documents), "dir": str(path)},
        )
        return documents

    def _load_text(self, path: Path) -> str:
        """Load a plain text or Markdown file."""
        return path.read_text(encoding="utf-8", errors="replace")

    def _load_pdf(self, path: Path) -> str:
        """Load a PDF file using PyPDF2."""
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(str(path))
            texts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    texts.append(text)
            return "\n\n".join(texts)
        except ImportError:
            logger.error("PyPDF2 not installed. Install with: pip install PyPDF2")
            raise

    def _load_json(self, path: Path) -> str:
        """Load a JSON file and extract text content.

        Handles both:
        - Plain text strings in JSON
        - Objects with 'text', 'content', or 'body' fields
        - Arrays of the above
        """
        import json

        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return self._extract_json_text(data)

    def _extract_json_text(self, data: Any) -> str:
        """Recursively extract text from JSON data.

        Args:
            data: Parsed JSON data (str, dict, list, etc.).

        Returns:
            Extracted text content.
        """
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            for key in ("text", "content", "body", "description"):
                if key in data and isinstance(data[key], str):
                    return data[key]
            # If no text field, flatten all string values
            parts = []
            for value in data.values():
                if isinstance(value, str):
                    parts.append(value)
            return "\n".join(parts) if parts else str(data)
        elif isinstance(data, list):
            parts = []
            for item in data:
                text = self._extract_json_text(item)
                if text:
                    parts.append(text)
            return "\n\n".join(parts)
        else:
            return str(data)
