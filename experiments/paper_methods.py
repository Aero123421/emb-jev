"""Training-free retrieval rules, with all choices made on development data."""

from __future__ import annotations

import numpy as np

from .email_routing import LABELS, make_centroids

K_GRID = (1, 3, 5, 9, 15, 31, 63)
T_GRID = (0.02, 0.05, 0.1, 0.2)
FILTER_GRID = (0.0, 0.5, 0.7, 0.8, 0.9, 0.95)
ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
PRIOR_T_GRID = (0.01, 0.03, 0.07, 0.15)


def softmax(values, temperature):
    scaled = (values - values.max(axis=1, keepdims=True)) / temperature
    weights = np.exp(scaled)
    return weights / weights.sum(axis=1, keepdims=True)


def cache_vote(similarities, distributions, k, temperature, *, indices=None):
    if indices is None:
        indices = np.argsort(-similarities, axis=1, kind="stable")
    indices = indices[:, : min(k, len(distributions))]
    values = np.take_along_axis(similarities, indices, axis=1)
    weights = softmax(values, temperature)
    return np.einsum("nk,nkc->nc", weights, distributions[indices], optimize=True)


def filtered_centroids(vectors, labels, confidences, threshold):
    keep = confidences >= threshold
    # A low-confidence class cannot silently disappear: use its unfiltered mean.
    for label in range(len(LABELS)):
        if not np.any(keep & (labels == label)):
            keep |= labels == label
    centers, present = make_centroids(vectors[keep], labels[keep])
    return centers, present, int(keep.sum())


def similarity_to_centroids(queries, centers, present):
    scores = queries @ centers.T
    scores[:, ~present] = -np.inf
    return scores


def choose(candidate, scores, target, best):
    agreement = float(np.mean(scores.argmax(axis=1) == target))
    # Fixed enumeration order resolves ties, with no test-set tie-breaking.
    if best is None or agreement > best["validation_primary_score"]:
        return {**candidate, "validation_primary_score": agreement}
    return best


def tune(vectors, labels, distributions, confidence, validation, target, descriptions, source):
    similarities = validation @ vectors.T
    indices = np.argsort(-similarities, axis=1, kind="stable")
    one_hot = np.eye(len(LABELS), dtype=np.float32)[labels]
    centers, present = make_centroids(vectors, labels)
    centroid_scores = similarity_to_centroids(validation, centers, present)
    desc_scores = validation @ descriptions.T
    selected = {
        "nearest": {
            "kind": "nearest",
            "validation_primary_score": float(np.mean(labels[indices[:, 0]] == target)),
        },
        "centroid": {
            "kind": "centroid",
            "validation_primary_score": float(np.mean(centroid_scores.argmax(axis=1) == target)),
        },
    }
    for name, distribution in (("weighted_knn", one_hot), ("soft_knn", distributions)):
        if name == "soft_knn" and source == "dataset":
            continue  # Identical to weighted_knn for one-hot dataset labels.
        best = None
        for k in K_GRID:
            for temperature in T_GRID:
                scores = cache_vote(similarities, distribution, k, temperature, indices=indices)
                best = choose(
                    {"kind": name, "k": k, "temperature": temperature}, scores, target, best
                )
        selected[name] = best
    if source == "jev":
        best = None
        for threshold in FILTER_GRID:
            filtered, mask, retained = filtered_centroids(vectors, labels, confidence, threshold)
            scores = similarity_to_centroids(validation, filtered, mask)
            best = choose(
                {
                    "kind": "filtered_centroid",
                    "threshold": threshold,
                    "retained_examples": retained,
                },
                scores,
                target,
                best,
            )
        selected["filtered_centroid"] = best
    parent_name = "soft_knn" if source == "jev" else "weighted_knn"
    parent = selected[parent_name]
    votes = cache_vote(
        similarities, distributions, parent["k"], parent["temperature"], indices=indices
    )
    best = None
    for prior_name, prior_scores in (("centroid", centroid_scores), ("description", desc_scores)):
        for temperature in PRIOR_T_GRID:
            prior = softmax(prior_scores, temperature)
            for alpha in ALPHA_GRID:
                scores = (1 - alpha) * votes + alpha * prior
                best = choose(
                    {
                        "kind": "hybrid",
                        "parent": parent_name,
                        "k": parent["k"],
                        "temperature": parent["temperature"],
                        "prior": prior_name,
                        "prior_temperature": temperature,
                        "alpha": alpha,
                    },
                    scores,
                    target,
                    best,
                )
    selected["hybrid"] = best
    selected["best"] = max(selected, key=lambda key: selected[key]["validation_primary_score"])
    return selected


def predict(
    config,
    vectors,
    labels,
    distributions,
    confidence,
    queries,
    descriptions,
    *,
    similarities=None,
    indices=None,
):
    kind = config["kind"]
    if kind == "centroid":
        centers, present = make_centroids(vectors, labels)
        return similarity_to_centroids(queries, centers, present).argmax(axis=1)
    if kind == "filtered_centroid":
        centers, present, _ = filtered_centroids(vectors, labels, confidence, config["threshold"])
        return similarity_to_centroids(queries, centers, present).argmax(axis=1)
    if similarities is None:
        similarities = queries @ vectors.T
    if kind == "nearest":
        return labels[similarities.argmax(axis=1)]
    if kind == "weighted_knn":
        distribution = np.eye(len(LABELS), dtype=np.float32)[labels]
    else:
        distribution = distributions
    votes = cache_vote(
        similarities, distribution, config["k"], config["temperature"], indices=indices
    )
    if kind in ("weighted_knn", "soft_knn"):
        return votes.argmax(axis=1)
    if kind == "hybrid":
        if config["prior"] == "centroid":
            centers, present = make_centroids(vectors, labels)
            scores = similarity_to_centroids(queries, centers, present)
        else:
            scores = queries @ descriptions.T
        prior = softmax(scores, config["prior_temperature"])
        return ((1 - config["alpha"]) * votes + config["alpha"] * prior).argmax(axis=1)
    raise ValueError(f"Unknown method {kind}")
