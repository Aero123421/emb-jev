from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

PREFIX = "task: classification | query: "


@dataclass(frozen=True)
class State:
    """One input: text and/or one local image and/or one local audio clip.

    Mixed inputs put the text first, then image, then audio. Media positions are
    assembled by the SDK; callers must not provide model placeholder tokens.
    """

    text: str | None = None
    image: str | Path | None = None
    audio: str | Path | None = None

    def __post_init__(self) -> None:
        if self.text is not None:
            if not isinstance(self.text, str) or not self.text.strip():
                raise ValueError("text must be a non-empty string")
            if any(token in self.text for token in ("<|image|>", "<|audio|>", "<|video|>")):
                raise ValueError("media placeholder tokens are assembled by the SDK")
        if self.text is None and self.image is None and self.audio is None:
            raise ValueError("provide text, image, or audio")
        for name in ("image", "audio"):
            path = getattr(self, name)
            if path is not None:
                resolved = Path(path).expanduser().resolve()
                if not resolved.is_file():
                    raise FileNotFoundError(resolved)
                object.__setattr__(self, name, resolved)

    def cache_key(self, identity: str) -> str:
        contents = {"text": self.text, "backend": identity, "format": 1}
        for name in ("image", "audio"):
            path = getattr(self, name)
            if path is not None:
                with path.open("rb") as stream:
                    contents[name] = hashlib.file_digest(stream, "sha256").hexdigest()
        encoded = json.dumps(contents, ensure_ascii=False, sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()

    def model_input(self) -> str | dict:
        text = PREFIX + self.text if self.text is not None else ""
        if self.image is None and self.audio is None:
            return text
        result = {}
        for name in ("image", "audio"):
            path = getattr(self, name)
            if path is not None:
                text += f" <|{name}|>"
                result[name] = str(path)
        result["text"] = text.strip()
        return result


@dataclass(frozen=True)
class Decision:
    type: Literal["choice", "boolean", "score"]
    value: str | bool | float | None
    status: Literal["decided", "abstained"]
    selected_key: str
    similarity: float
    margin: float
    similarities: dict[str, float]
    weights: dict[str, float]
    expected_score: float | None = None

    def to_dict(self) -> dict:
        """JSON-safe output. Weights are relative similarities, not calibrated probabilities."""
        return asdict(self)


def normalized_vectors(vectors, *, rows: int, dimension: int) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.shape != (rows, dimension):
        raise ValueError(f"expected embedding shape {(rows, dimension)}, got {array.shape}")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if not np.isfinite(array).all() or not np.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError("backend returned non-finite or zero embeddings")
    return array / norms
