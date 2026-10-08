"""Real-model smoke check. Generated media verify execution, not accuracy."""

import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image

from emb_jev import Engine, State

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        image = folder / "red.png"
        audio = folder / "tone.wav"
        Image.new("RGB", (224, 224), (255, 0, 0)).save(image)
        sf.write(audio, np.sin(2 * np.pi * 440 * np.arange(16000) / 16000) * 0.1, 16000)
        with Engine(modalities=["text", "image", "audio"]) as engine:
            # Text and media candidates can be registered together.
            engine.prepare(
                type="choice",
                options={"text_reference": "赤い画像", "image_reference": State(image=image)},
            )
            parity_inputs = [
                State(text="カードから二重に代金が引き落とされた"),
                State(image=image),
                State(audio=audio),
                State(text="赤色の画像と音", image=image, audio=audio),
            ]
            parity = []
            for state in parity_inputs:
                actual = engine.backend.encode([state])[0]
                reference = engine.backend.model.encode(
                    state.model_input(),
                    prompt="",
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                reference = reference.astype(np.float32)
                reference /= np.linalg.norm(reference)
                cosine = float(actual @ reference)
                parity.append(cosine)
                assert cosine > 0.99999, f"official encode parity mismatch: cosine={cosine}"
            category = engine.prepare(
                type="choice",
                options={
                    "billing": "二重請求や支払いの問い合わせ",
                    "technical": "ログインやアプリの不具合",
                },
            )
            binary = engine.prepare(
                type="boolean",
                options={
                    "true": "支払いについて相談している",
                    "false": "アプリの操作方法について相談している",
                },
            )
            score = engine.prepare(
                type="score",
                options={"low": "丁寧な問い合わせ", "high": "怒っている問い合わせ"},
                score_values={"low": 1, "high": 5},
            )
            tasks = {"category": category, "binary": binary, "score": score}
            calls = []
            hook = engine.backend.model[0].model.language_model.register_forward_hook(
                lambda *_: calls.append(1)
            )
            report = {}
            try:
                for name, inputs in (
                    ("text", {"text": "カードから二重に代金が引き落とされた"}),
                    ("image", {"image": image}),
                    ("audio", {"audio": audio}),
                    ("mixed", {"text": "赤色の画像と音", "image": image, "audio": audio}),
                ):
                    before = len(calls)
                    results = engine.decide(tasks, **inputs)
                    assert len(calls) - before == 1, f"{name}: more than one backbone forward"
                    for result in results.values():
                        assert np.isfinite(list(result.similarities.values())).all()
                        assert abs(sum(result.weights.values()) - 1) < 1e-6
                    report[name] = {key: value.to_dict() for key, value in results.items()}
            finally:
                hook.remove()
            engine.backend.max_tokens = 1
            previous = engine.backend.forward_calls
            try:
                engine.decide(tasks, text="カードから二重に代金が引き落とされた")
            except ValueError as error:
                assert "max_tokens" in str(error)
            else:
                raise AssertionError("oversized input was silently accepted")
            assert engine.backend.forward_calls == previous
            report["observed_model_forward_calls"] = len(calls)
            report["official_encode_parity_cosines"] = parity
            report["note"] = "Synthetic media check execution only; no image/audio accuracy claim."
            print(json.dumps(report, ensure_ascii=False, indent=2))
