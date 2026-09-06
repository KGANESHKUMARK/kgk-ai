"""KGK AI Memory Stores — Concrete implementations of BaseMemoryStore.

Provides in-memory and file-based persistent storage for conversation memory.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any

from app.logging_config import get_logger
from app.memory.manager import BaseMemoryStore, MemoryItem

logger = get_logger("memory.stores")


class InMemoryStore(BaseMemoryStore):
    """In-memory storage backend for conversation memory.

    Thread-safe using a lock. Data is lost when the process exits.
    """

    def __init__(self) -> None:
        self._store: dict[str, list[MemoryItem]] = {}
        self._lock = threading.Lock()

    def save(self, key: str, item: MemoryItem) -> None:
        with self._lock:
            if key not in self._store:
                self._store[key] = []
            self._store[key].append(item)

    def retrieve(self, key: str, limit: int = 10) -> list[MemoryItem]:
        with self._lock:
            items = self._store.get(key, [])
            return items[-limit:]

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    @property
    def size(self) -> int:
        with self._lock:
            return sum(len(items) for items in self._store.values())


class FileMemoryStore(BaseMemoryStore):
    """File-based persistent storage for conversation memory.

    Stores each conversation's memory items as a JSON file on disk.
    Survives process restarts.
    """

    def __init__(self, base_dir: str) -> None:
        self._base_dir = base_dir
        self._lock = threading.Lock()
        os.makedirs(base_dir, exist_ok=True)
        logger.info(f"FileMemoryStore initialized at: {base_dir}")

    def _get_path(self, key: str) -> str:
        safe_key = key.replace("/", "_").replace("\\", "_").replace(":", "_")
        return os.path.join(self._base_dir, f"{safe_key}.json")

    def _load(self, key: str) -> list[MemoryItem]:
        path = self._get_path(key)
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [MemoryItem(**item) for item in data]
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.warning(f"Failed to load memory for key '{key}': {e}")
            return []

    def _save(self, key: str, items: list[MemoryItem]) -> None:
        path = self._get_path(key)
        data = [
            {
                "content": item.content,
                "timestamp": item.timestamp,
                "metadata": item.metadata,
            }
            for item in items
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def save(self, key: str, item: MemoryItem) -> None:
        with self._lock:
            items = self._load(key)
            items.append(item)
            self._save(key, items)

    def retrieve(self, key: str, limit: int = 10) -> list[MemoryItem]:
        with self._lock:
            items = self._load(key)
            return items[-limit:]

    def delete(self, key: str) -> bool:
        with self._lock:
            path = self._get_path(key)
            if os.path.exists(path):
                os.remove(path)
                return True
            return False

    def clear(self) -> None:
        with self._lock:
            for filename in os.listdir(self._base_dir):
                if filename.endswith(".json"):
                    filepath = os.path.join(self._base_dir, filename)
                    os.remove(filepath)
