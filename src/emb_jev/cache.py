from __future__ import annotations

import sqlite3
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .types import normalized_vectors


class VectorCache:
    """Bounded memory cache, optionally backed by SQLite. Stores candidates only."""

    def __init__(self, path: str | Path | None, dimension: int, capacity: int = 1024):
        self.dimension = dimension
        self.capacity = capacity
        self.memory: OrderedDict[str, np.ndarray] = OrderedDict()
        self.connection = None
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(path)
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vector BLOB NOT NULL)"
            )

    def get(self, key: str) -> np.ndarray | None:
        if key in self.memory:
            self.memory.move_to_end(key)
            return self.memory[key]
        if self.connection is not None:
            row = self.connection.execute(
                "SELECT vector FROM vectors WHERE key = ?", (key,)
            ).fetchone()
            if row is not None:
                try:
                    value = normalized_vectors(
                        np.frombuffer(row[0], dtype="<f4").reshape(1, -1),
                        rows=1,
                        dimension=self.dimension,
                    )[0]
                except ValueError as error:
                    raise ValueError("candidate cache is corrupt; delete the cache file") from error
                self._remember(key, value)
                return value
        return None

    def put(self, key: str, value: np.ndarray) -> None:
        value = normalized_vectors(value.reshape(1, -1), rows=1, dimension=self.dimension)[0]
        if self.connection is not None:
            with self.connection:
                self.connection.execute(
                    "INSERT OR REPLACE INTO vectors VALUES (?, ?)",
                    (key, value.astype("<f4").tobytes()),
                )
        self._remember(key, value)

    def _remember(self, key: str, value: np.ndarray) -> None:
        value.setflags(write=False)
        self.memory[key] = value
        self.memory.move_to_end(key)
        while len(self.memory) > self.capacity:
            self.memory.popitem(last=False)

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        self.memory.clear()
