import json

import numpy as np
import pytest

from emb_jev import Engine, State


class FixedBackend:
    dimension = 3
    identity = "test-model@commit-a:preprocessing-1"

    def __init__(self, vectors=None):
        self.calls = []
        self.vectors = vectors or {
            "billing": [1, 0, 0],
            "technical": [0, 1, 0],
            "sales": [0, 0, 1],
            "request": [3, 0, 0],
            "unknown": [1, 1, 1],
        }

    def validate(self, states):
        pass

    def encode(self, states):
        self.calls.append(tuple(states))
        return np.array([self.vectors[state.text] for state in states], dtype=np.float32)


def tasks(engine):
    return {
        "category": engine.prepare(type="choice", options={"b": "billing", "t": "technical"}),
        "financial": engine.prepare(
            type="boolean", options={"true": "billing", "false": "technical"}
        ),
        "score": engine.prepare(
            type="score",
            options={"low": "technical", "high": "billing"},
            score_values={"low": 1, "high": 5},
        ),
    }


def test_all_types_share_one_input_encoding_and_keep_types():
    backend = FixedBackend()
    with Engine(backend=backend) as engine:
        schema = tasks(engine)
        assert len(backend.calls) == 1  # Same descriptions are reused across tasks.
        results = engine.decide(schema, text="request")
        assert len(backend.calls) == 2
        assert len(backend.calls[-1]) == 1
        assert results["category"].value == "b"
        assert results["financial"].value is True
        assert results["score"].value == 5
        assert 1 < results["score"].expected_score < 5
        assert results["category"].similarity == 1
        assert results["category"].margin == 1
        for result in results.values():
            assert sum(result.weights.values()) == pytest.approx(1)
            json.dumps(result.to_dict(), allow_nan=False)


def test_boolean_false_preserves_false_instead_of_null():
    with Engine(backend=FixedBackend()) as engine:
        task = tasks(engine)["financial"]
        result = task.decide(text="technical")
        assert result.value is False
        assert result.status == "decided"


def test_abstention_and_score_suppression():
    with Engine(backend=FixedBackend()) as engine:
        task = engine.prepare(
            type="score",
            options={"a": "billing", "b": "technical"},
            score_values={"a": 1, "b": 2},
            min_margin=0.1,
        )
        result = task.decide(text="unknown")
        assert result.value is None
        assert result.expected_score is None
        assert result.status == "abstained"
        assert result.margin == 0
        assert result.selected_key in ("a", "b")


def test_similarity_threshold_can_reject_even_a_clear_winner():
    backend = FixedBackend({"a": [1, 0, 0], "b": [-1, 0, 0], "q": [0.1, 1, 0]})
    with Engine(backend=backend) as engine:
        task = engine.prepare(type="choice", options={"a": "a", "b": "b"}, min_similarity=0.5)
        assert task.decide(text="q").status == "abstained"


def test_option_order_does_not_change_the_winner_or_scores():
    with Engine(backend=FixedBackend()) as engine:
        a = engine.prepare(type="choice", options={"b": "billing", "t": "technical"})
        b = engine.prepare(type="choice", options={"t": "technical", "b": "billing"})
        result = engine.decide({"a": a, "b": b}, text="request")
        assert result["a"].value == result["b"].value
        assert result["a"].similarities == result["b"].similarities


def test_cache_survives_restart_and_invalidates_on_model_change(tmp_path):
    cache = tmp_path / "cache.sqlite3"
    first = FixedBackend()
    with Engine(backend=first, cache_path=cache) as engine:
        tasks(engine)
    second = FixedBackend()
    with Engine(backend=second, cache_path=cache) as engine:
        schema = tasks(engine)
        assert second.calls == []
        engine.decide(schema, text="request")
        engine.decide(schema, text="request")
        assert len(second.calls) == 2  # Input embeddings are never cached.
    third = FixedBackend()
    third.identity = "test-model@commit-b:preprocessing-1"
    with Engine(backend=third, cache_path=cache) as engine:
        tasks(engine)
        assert len(third.calls) == 1


def test_changed_candidate_text_is_reembedded():
    backend = FixedBackend()
    with Engine(backend=backend) as engine:
        engine.prepare(type="choice", options={"b": "billing", "t": "technical"})
        engine.prepare(type="choice", options={"b": "sales", "t": "technical"})
        assert len(backend.calls) == 2
        assert backend.calls[-1] == (State(text="sales"),)


def test_media_cache_hashes_content_not_filename(tmp_path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    a.write_bytes(b"first")
    b.write_bytes(b"first")
    assert State(image=a).cache_key("model") == State(image=b).cache_key("model")
    before = State(image=a).cache_key("model")
    a.write_bytes(b"second")
    assert State(image=a).cache_key("model") != before
    assert State(image=a).cache_key("model") != State(audio=a).cache_key("model")


@pytest.mark.parametrize(
    "vectors",
    [
        {"billing": [0, 0, 0], "technical": [0, 1, 0]},
        {"billing": [np.nan, 0, 0], "technical": [0, 1, 0]},
        {"billing": [np.inf, 0, 0], "technical": [0, 1, 0]},
        {"billing": [1, 0], "technical": [0, 1]},
    ],
)
def test_invalid_embeddings_fail_before_scoring(vectors):
    with Engine(backend=FixedBackend(vectors)) as engine:
        with pytest.raises(ValueError):
            engine.prepare(type="choice", options={"b": "billing", "t": "technical"})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"type": "other", "options": {"a": "billing", "b": "technical"}},
        {"type": "choice", "options": {"a": "billing"}},
        {"type": "boolean", "options": {"a": "billing", "b": "technical"}},
        {"type": "choice", "options": {"a": "billing", "b": "technical"}, "temperature": 0},
        {"type": "choice", "options": {"a": "billing", "b": "technical"}, "min_margin": np.nan},
        {
            "type": "score",
            "options": {"a": "billing", "b": "technical"},
            "score_values": {"a": 2, "b": 1},
        },
        {"type": "score", "options": {"a": "billing", "b": "technical"}, "score_values": {"a": 1}},
    ],
)
def test_invalid_tasks_do_not_run_model(kwargs):
    backend = FixedBackend()
    with Engine(backend=backend) as engine:
        with pytest.raises(ValueError):
            engine.prepare(**kwargs)
        assert backend.calls == []


def test_engine_ownership_and_closed_state():
    with Engine(backend=FixedBackend()) as a, Engine(backend=FixedBackend()) as b:
        task = tasks(a)["category"]
        with pytest.raises(ValueError, match="this engine"):
            b.decide({"task": task}, text="request")
    a.close()
    with pytest.raises(RuntimeError, match="closed"):
        task.decide(text="request")


def test_mixed_input_positions_are_explicit(tmp_path):
    image = tmp_path / "a.png"
    audio = tmp_path / "a.wav"
    image.touch()
    audio.touch()
    value = State(text="説明", image=image, audio=audio).model_input()
    assert value["text"] == "task: classification | query: 説明 <|image|> <|audio|>"
    assert value["image"] == str(image)
    assert State(image=image).model_input()["text"] == "<|image|>"


def test_missing_inputs_and_files_are_errors(tmp_path):
    with pytest.raises(ValueError):
        State()
    with pytest.raises(ValueError):
        State(text=" ")
    with pytest.raises(ValueError):
        State(text="<|image|>")
    with pytest.raises(FileNotFoundError):
        State(audio=tmp_path / "missing.wav")
