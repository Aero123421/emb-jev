import asyncio
import json

import httpx
import numpy as np
import pytest

from experiments.email_routing import knn_scores, normalize_text
from experiments.jev import (
    RESERVATION,
    BudgetExceeded,
    Ledger,
    classify,
    fingerprint,
    load_key,
)


def response(choice="a", tokens=100):
    return {
        "model": "jev-1.13.0",
        "answers": {"intent": {"choice": choice, "probabilities": {"a": 0.9, "b": 0.1}}},
        "usage": {"input_tokens": tokens},
    }


def test_budget_cap_counts_unknown_attempts_and_survives_restart(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    ledger = Ledger(path, "0.003")
    attempt = ledger.reserve("first")
    ledger.failure(attempt, "transport_error")
    with pytest.raises(BudgetExceeded):
        ledger.reserve("second")
    ledger.close()
    restarted = Ledger(path, "0.003")
    with pytest.raises(BudgetExceeded):
        restarted.reserve("third")
    assert restarted.summary()["conservative_spend_including_unknown_usd"] == RESERVATION / 1e9
    restarted.close()


def test_success_settles_reservation_and_caches_exact_payload(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite3", "0.003")
    digest = fingerprint({"state": "first", "model": "jev-1.13.0"})
    attempt = ledger.reserve(digest)
    ledger.success(attempt, digest, response(tokens=100))
    assert ledger.cached(digest) == response(tokens=100)
    assert ledger.cached(fingerprint({"state": "second", "model": "jev-1.13.0"})) is None
    ledger.reserve("second")  # Settled usage frees unused reservation.
    assert ledger.summary()["successful_usage_estimated_usd"] == 0.0000042
    ledger.close()


def test_api_replay_does_not_send_another_paid_request(tmp_path):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=response())

    async def run():
        ledger = Ledger(tmp_path / "ledger.sqlite3")
        async with httpx.AsyncClient(
            base_url="https://api.typesafe.ai", transport=httpx.MockTransport(handler)
        ) as client:
            first = await classify(client, ledger, "message", {"a": "A", "b": "B"}, "decide")
            second = await classify(client, ledger, "message", {"a": "A", "b": "B"}, "decide")
        assert first == second
        assert len(calls) == 1
        assert ledger.summary()["successful_calls"] == 1
        ledger.close()

    asyncio.run(run())


def test_budget_prevents_http_and_error_does_not_expose_key(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(401, text="secret-in-server-response")

    async def run():
        ledger = Ledger(tmp_path / "ledger.sqlite3", "0.001")
        async with httpx.AsyncClient(
            base_url="https://api.typesafe.ai", transport=httpx.MockTransport(handler)
        ) as client:
            with pytest.raises(BudgetExceeded):
                await classify(client, ledger, "message", {"a": "A", "b": "B"}, "decide")
            assert not calls
            ledger.limit = 4_000_000_000
            with pytest.raises(RuntimeError) as error:
                await classify(client, ledger, "message", {"a": "A", "b": "B"}, "decide")
            assert "secret" not in str(error.value)
        assert ledger.summary()["successful_calls"] == 0
        ledger.close()

    asyncio.run(run())


def test_key_loader_accepts_raw_and_named_keys_without_logging(tmp_path, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    path = tmp_path / ".env"
    path.write_text("test-token")
    assert load_key(path) == "test-token"
    path.write_text("# header\nTYPESAFE_API_KEY='test-token'\n")
    assert load_key(path) == "test-token"


def test_example_cache_votes_by_label_not_by_arbitrary_candidate_id():
    scores = knn_scores(np.array([[0.90, 0.89, 0.91]]), np.array([0, 0, 1]), 3, 0.1)
    assert scores.argmax(axis=1).item() == 0  # Two close examples beat one closer example.
    assert scores.sum() == pytest.approx(1)
    nearest = knn_scores(np.array([[0.90, 0.89, 0.91]]), np.array([0, 0, 1]), 1, 0.1)
    assert nearest.argmax(axis=1).item() == 1


def test_split_normalization_collapses_case_punctuation_and_placeholder_values():
    assert normalize_text("Cancel ORDER {{Order Number}}!") == normalize_text(
        "cancel order {{1234}}"
    )
