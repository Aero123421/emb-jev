import numpy as np
import pytest

from experiments.email_routing import LABELS
from experiments.paper_methods import cache_vote, filtered_centroids, predict
from experiments.paper_protocol import paired_comparison, wilson


def test_soft_teacher_cache_preserves_distribution_instead_of_hardening_it():
    distribution = np.zeros((2, len(LABELS)), dtype=np.float32)
    distribution[0, :2] = [0.51, 0.49]
    distribution[1, :2] = [0.01, 0.99]
    scores = cache_vote(np.array([[0.9, 0.9]]), distribution, 2, 0.1)
    assert scores[0, 0] == pytest.approx(0.26)
    assert scores[0, 1] == pytest.approx(0.74)
    assert scores.sum() == pytest.approx(1)


def test_confidence_filter_keeps_a_low_confidence_only_class_available():
    centers, present, retained = filtered_centroids(
        np.eye(3), np.array([0, 0, 1]), np.array([0.95, 0.4, 0.3]), 0.9
    )
    assert retained == 2
    assert present[:2].tolist() == [True, True]
    assert centers[0].tolist() == [1, 0, 0]
    assert centers[1].tolist() == [0, 0, 1]
    assert not present[2:].any()


def test_hybrid_zero_alpha_is_exactly_the_parent_prediction():
    vectors = np.eye(2)
    labels = np.array([0, 1])
    distributions = np.eye(len(LABELS))[:2]
    descriptions = np.zeros((len(LABELS), 2))
    descriptions[1] = [1, 0]  # Prior opposes the closest example.
    query = np.array([[1, 0]])
    config = {
        "kind": "hybrid",
        "k": 1,
        "temperature": 0.1,
        "alpha": 0,
        "prior": "description",
        "prior_temperature": 0.1,
    }
    prediction = predict(config, vectors, labels, distributions, np.ones(2), query, descriptions)
    assert prediction.tolist() == [0]


def test_paired_difference_counts_disagreement_and_wilson_bounds():
    target = np.array([0, 0, 1, 1])
    baseline = np.array([0, 1, 0, 1])
    candidate = np.array([0, 0, 1, 1])
    comparison = paired_comparison(candidate, baseline, target)
    assert comparison["candidate_only_correct"] == 2
    assert comparison["baseline_only_correct"] == 0
    assert comparison["accuracy_difference"] == 0.5
    low, high = wilson(2, 4)
    assert 0 < low < 0.5 < high < 1
