from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .backend import Backend, EmbeddingGemmaBackend
from .cache import VectorCache
from .types import Decision, State, normalized_vectors


@dataclass(frozen=True)
class PreparedTask:
    _engine: Engine
    type: Literal["choice", "boolean", "score"]
    keys: tuple[str, ...]
    _vectors: np.ndarray
    _score_values: tuple[float, ...] | None
    min_similarity: float | None
    min_margin: float | None
    temperature: float

    def decide(self, *, text=None, image=None, audio=None) -> Decision:
        return self._engine.decide({"result": self}, text=text, image=image, audio=audio)["result"]

    def _score(self, scores: np.ndarray) -> Decision:
        winner = int(np.argmax(scores))
        top = float(scores[winner])
        margin = float(top - np.partition(scores, -2)[-2])
        weights = np.exp((scores.astype(np.float64) - top) / self.temperature)
        weights /= weights.sum()
        abstained = (self.min_similarity is not None and top < self.min_similarity) or (
            self.min_margin is not None and margin < self.min_margin
        )
        expected = None
        value: str | bool | float | None = self.keys[winner]
        if self.type == "boolean":
            value = self.keys[winner] == "true"
        elif self.type == "score":
            value = self._score_values[winner]
            if not abstained:
                expected = float(np.dot(weights, self._score_values))
        return Decision(
            type=self.type,
            value=None if abstained else value,
            status="abstained" if abstained else "decided",
            selected_key=self.keys[winner],
            similarity=top,
            margin=margin,
            similarities=dict(zip(self.keys, map(float, scores), strict=True)),
            weights=dict(zip(self.keys, map(float, weights), strict=True)),
            expected_score=expected,
        )


class Engine:
    """Training-free candidate comparison. Instances are intended for one worker/thread.

    Questions share a task-independent input embedding. Candidate descriptions
    must contain the meaning of the decision; this is not instruction following.
    """

    def __init__(
        self,
        *,
        backend: Backend | None = None,
        modalities: Sequence[str] = ("text",),
        cache_path=None,
        **model_options,
    ):
        if backend is not None and model_options:
            raise ValueError("model options cannot be combined with a custom backend")
        self.backend = backend or EmbeddingGemmaBackend(modalities=modalities, **model_options)
        self._cache = VectorCache(cache_path, self.backend.dimension)
        self._closed = False

    def prepare(
        self,
        *,
        type: Literal["choice", "boolean", "score"],
        options: Mapping[str, str | State],
        score_values: Mapping[str, float] | None = None,
        min_similarity: float | None = None,
        min_margin: float | None = None,
        temperature: float = 0.1,
    ) -> PreparedTask:
        self._check_open()
        if type not in ("choice", "boolean", "score"):
            raise ValueError("type must be choice, boolean, or score")
        if not isinstance(options, Mapping) or not 2 <= len(options) <= 256:
            raise ValueError("options must contain 2 to 256 candidates")
        if any(not isinstance(key, str) or not key.strip() for key in options):
            raise ValueError("candidate keys must be non-empty strings")
        if type == "boolean" and set(options) != {"true", "false"}:
            raise ValueError("boolean requires explicit 'true' and 'false' descriptions")
        for name, value, low, high in (
            ("min_similarity", min_similarity, -1, 1),
            ("min_margin", min_margin, 0, 2),
        ):
            if value is not None and (not np.isfinite(value) or not low <= value <= high):
                raise ValueError(f"{name} must be finite and between {low} and {high}")
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        keys = tuple(options)
        values = None
        if type == "score":
            if score_values is None or set(score_values) != set(keys):
                raise ValueError("score_values must match every score candidate key")
            values = tuple(float(score_values[key]) for key in keys)
            if not np.isfinite(values).all() or any(
                a >= b for a, b in zip(values, values[1:], strict=False)
            ):
                raise ValueError(
                    "score_values must be finite and strictly increasing in option order"
                )
        elif score_values is not None:
            raise ValueError("score_values only applies to score tasks")
        states = [
            State(text=value) if isinstance(value, str) else value for value in options.values()
        ]
        if any(not isinstance(state, State) for state in states):
            raise TypeError("candidates must be strings or State objects")
        self.backend.validate(states)
        cache_keys = [state.cache_key(self.backend.identity) for state in states]
        vectors = {key: self._cache.get(key) for key in cache_keys}
        missing = {
            key: state
            for key, state in zip(cache_keys, states, strict=True)
            if vectors[key] is None
        }
        if missing:
            encoded = normalized_vectors(
                self.backend.encode(list(missing.values())),
                rows=len(missing),
                dimension=self.backend.dimension,
            )
            for key, vector in zip(missing, encoded, strict=True):
                self._cache.put(key, vector)
                vectors[key] = vector
        matrix = np.stack([vectors[key] for key in cache_keys])
        matrix.setflags(write=False)
        return PreparedTask(
            self, type, keys, matrix, values, min_similarity, min_margin, temperature
        )

    def decide(self, tasks: Mapping[str, PreparedTask], *, text=None, image=None, audio=None):
        self._check_open()
        if not tasks or any(not isinstance(key, str) or not key for key in tasks):
            raise ValueError("provide a non-empty mapping of task names")
        if any(
            not isinstance(task, PreparedTask) or task._engine is not self
            for task in tasks.values()
        ):
            raise ValueError("all tasks must be prepared by this engine")
        state = State(text=text, image=image, audio=audio)
        self.backend.validate([state])
        query = normalized_vectors(
            self.backend.encode([state]), rows=1, dimension=self.backend.dimension
        )[0]
        matrix = np.concatenate([task._vectors for task in tasks.values()])
        similarities = np.clip(matrix @ query, -1.0, 1.0)
        result = {}
        offset = 0
        for name, task in tasks.items():
            count = len(task.keys)
            result[name] = task._score(similarities[offset : offset + count])
            offset += count
        return result

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("engine is closed")

    def close(self) -> None:
        if self._closed:
            return
        self._cache.close()
        close = getattr(self.backend, "close", None)
        if close is not None:
            close()
        self._closed = True

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *_):
        self.close()
