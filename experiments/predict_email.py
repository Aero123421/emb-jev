"""Use the experiment's Jev-labeled example cache without any API calls."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from emb_jev import EmbeddingGemmaBackend, State

from .email_routing import LABELS, MODEL_REVISION, knn_scores, read_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    parser.add_argument("--work", type=Path, default=Path(".cache/experiment"))
    parser.add_argument("--report", type=Path, default=Path("reports/bitext-1000"))
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--method", choices=("auto", "knn", "centroid"), default="auto")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    result = read_json(args.report / "results.json")
    params = result["knn_parameters"]["jev"]
    method = (
        result["selected_local_method"].removeprefix("jev_")
        if args.method == "auto"
        else args.method
    )
    if method == "centroid":
        cache = np.load(args.work / "jev-centroids.npz")
        if cache["revision"].item() != MODEL_REVISION or cache["labels"].tolist() != LABELS:
            raise ValueError("Centroid cache model/label definition mismatch")
        vectors = cache["vectors"]
        present = cache["present"]
    else:
        samples = read_json(args.work / "splits.json")["support"]
        responses = read_json(args.work / "jev-support.json")
        by_id = {row["id"]: row["response"]["answers"]["intent"]["choice"] for row in responses}
        labels = np.array([LABELS.index(by_id[row["id"]]) for row in samples])
        vectors = np.load(args.work / "vectors.npz")["support"]
    backend = EmbeddingGemmaBackend(revision=MODEL_REVISION)
    backend.encode([State(text=args.text)])  # Warm-up excluded from timing.
    elapsed = []
    search_elapsed = []
    for _ in range(args.runs):
        start = time.perf_counter()
        query = backend.encode([State(text=args.text)])
        search_start = time.perf_counter()
        similarities = query @ vectors.T
        if method == "centroid":
            similarities[:, ~present] = -np.inf
            scores = similarities[0]
        else:
            scores = knn_scores(similarities, labels, params["k"], params["temperature"])[0]
        search_elapsed.append((time.perf_counter() - search_start) * 1000)
        elapsed.append((time.perf_counter() - start) * 1000)
    winner = int(scores.argmax())
    margin = float(np.sort(scores)[-1] - np.sort(scores)[-2])
    print(
        json.dumps(
            {
                "intent": LABELS[winner],
                "method": method,
                "score": float(scores[winner]),
                "score_type": "cosine_similarity"
                if method == "centroid"
                else "relative_vote_weight",
                "margin": margin,
                "calibrated_probability": False,
                "api_calls": 0,
                "cache_vector_bytes": vectors.nbytes,
                "runs": args.runs,
                "warm_end_to_end_ms_p50": float(np.quantile(elapsed, 0.5)),
                "warm_end_to_end_ms_p95": float(np.quantile(elapsed, 0.95)),
                "search_ms_p50": float(np.quantile(search_elapsed, 0.5)),
            },
            indent=2,
        )
    )
    backend.close()


if __name__ == "__main__":
    main()
