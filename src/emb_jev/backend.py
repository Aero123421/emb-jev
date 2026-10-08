from __future__ import annotations

import json
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Protocol

import numpy as np

from .types import State, normalized_vectors


class Backend(Protocol):
    identity: str
    dimension: int

    def validate(self, states: Sequence[State]) -> None: ...

    def encode(self, states: Sequence[State]) -> np.ndarray: ...


class EmbeddingGemmaBackend:
    """Official Sentence Transformers pipeline with explicit input length checks.

    Uses preprocessing + forward directly, so each single input has exactly one
    model forward, even when multiple questions share its embedding.
    """

    def __init__(
        self,
        *,
        modalities: Sequence[str] = ("text",),
        model_id: str = "google/embeddinggemma-2",
        revision: str | None = None,
        device: str | None = None,
        dimension: int = 768,
        max_tokens: int = 2048,
        batch_size: int = 16,
        local_files_only: bool = False,
    ):
        selected = frozenset(modalities)
        if not selected or not selected <= {"text", "image", "audio"}:
            raise ValueError("modalities must contain text, image, and/or audio")
        if dimension not in (128, 256, 512, 768):
            raise ValueError("dimension must be 128, 256, 512, or 768")
        if not isinstance(max_tokens, int) or not 1 <= max_tokens <= 8192:
            raise ValueError("max_tokens must be between 1 and 8192")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be positive")
        try:
            import torch
            from huggingface_hub import hf_hub_download
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ImportError(
                "install model dependencies with: pip install 'emb-jev[model]'"
            ) from error

        config = {}
        if "image" not in selected:
            config["vision_config"] = None
        if "audio" not in selected:
            config["audio_config"] = None
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = (
            torch.bfloat16
            if device.startswith("cuda") and torch.cuda.is_bf16_supported()
            else torch.float32
        )
        config_path = hf_hub_download(
            model_id, "config.json", revision=revision, local_files_only=local_files_only
        )
        commit = Path(config_path).parent.name
        self.model = SentenceTransformer(
            model_id,
            revision=commit,
            device=device,
            local_files_only=local_files_only,
            config_kwargs=config,
            model_kwargs={"dtype": dtype},
        )
        self.model.eval()
        self.dimension = dimension
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.modalities = selected | {"text"}
        self.forward_calls = 0
        self.identity = json.dumps(
            {
                "model": model_id,
                "revision": commit,
                "modalities": sorted(selected),
                "dimension": dimension,
                "max_tokens": max_tokens,
                "dtype": str(dtype),
                "device": device,
                "sentence-transformers": version("sentence-transformers"),
                "transformers": version("transformers"),
                "preprocessing": 1,
            },
            sort_keys=True,
        )

    def validate(self, states: Sequence[State]) -> None:
        for state in states:
            for modality in ("text", "image", "audio"):
                if getattr(state, modality) is not None and modality not in self.modalities:
                    raise ValueError(f"{modality} encoder is disabled")

    def encode(self, states: Sequence[State]) -> np.ndarray:
        import torch
        from sentence_transformers.util import batch_to_device

        self.validate(states)
        if not states:
            return np.empty((0, self.dimension), dtype=np.float32)
        chunks = []
        with torch.inference_mode():
            for offset in range(0, len(states), self.batch_size):
                inputs = [
                    state.model_input() for state in states[offset : offset + self.batch_size]
                ]
                features = self.model.preprocess(
                    inputs, prompt="", processing_kwargs={"text": {"truncation": False}}
                )
                lengths = features["attention_mask"].sum(dim=-1)
                if (lengths > self.max_tokens).any():
                    raise ValueError(
                        f"input exceeds max_tokens={self.max_tokens}; "
                        "shorten it or split explicitly"
                    )
                features = batch_to_device(features, self.model.device)
                output = self.model(features)["sentence_embedding"]
                self.forward_calls += 1
                chunks.append(output[:, : self.dimension].float().cpu().numpy())
        return normalized_vectors(
            np.concatenate(chunks), rows=len(states), dimension=self.dimension
        )

    def close(self) -> None:
        self.model = None
