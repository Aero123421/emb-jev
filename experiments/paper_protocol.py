"""Fresh 10,000-example holdout, frozen choices, and random baselines.

Stages must run in order. Test labels/predictions are not read by tune().
Paid stages reuse the first experiment's global $4 SQLite spending ledger.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import platform
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from emb_jev import EmbeddingGemmaBackend, State

from . import paper_methods as methods
from .email_routing import (
    DATA_REVISION,
    DATASET,
    DESCRIPTIONS,
    INSTRUCTIONS,
    LABELS,
    MODEL_REVISION,
    normalize_text,
    read_json,
    write_json,
)
from .jev import MODEL, Ledger, label_rows, load_key

SEED = 20261008
SIZES = (500, 1000, 2000)
SUBSET_SEEDS = (0, 42, 2026, 31415, 27182)
THRESHOLD = 0.90
STAGES = (
    "prepare",
    "embed_development",
    "label_development",
    "tune",
    "label_test",
    "embed_test",
    "evaluate",
)


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timestamp():
    return datetime.now(UTC).isoformat()


def prepare(work, report, old_report, source):
    if (report / "frozen.json").exists():
        raise ValueError("Protocol already frozen; preparation cannot overwrite this run")
    work.mkdir(parents=True, exist_ok=True)
    old = read_json(old_report / "manifest.json")
    if sha_file(source) != old["source_sha256"]:
        raise ValueError("Dataset bytes changed from the pinned source")
    old_ids = {index for split in old["split_ids"].values() for index in split}
    assert len(old_ids) == 2500
    with source.open(newline="") as stream:
        raw = list(csv.DictReader(stream))
    groups = {}
    categories = {}
    for index, row in enumerate(raw):
        assert row["intent"] in LABELS
        categories[row["intent"]] = row["category"]
        groups.setdefault(normalize_text(row["instruction"]), []).append(
            {"id": index, "text": row["instruction"], "label": row["intent"]}
        )
    clean = [rows[0] for rows in groups.values() if len({r["label"] for r in rows}) == 1]
    clean.sort(key=lambda row: row["id"])
    old_positions = [i for i, row in enumerate(clean) if row["id"] in old_ids]
    assert len(old_positions) == 2500
    eligible = np.array([i for i, row in enumerate(clean) if row["id"] not in old_ids])
    rng = np.random.default_rng(SEED)
    order = rng.permutation(eligible)
    support = order[:2000].tolist()
    remaining = order[2000:]
    lexical = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2).fit_transform(
        [normalize_text(row["text"]) for row in clean]
    )
    references = support + old_positions
    splits = {"support": support}
    excluded = {}
    maximum = {}
    for name, count in (("validation", 1000), ("test", 10000)):
        selected, rejected, maxima_kept, consumed = [], 0, [], 0
        for offset in range(0, len(remaining), 256):
            block = remaining[offset : offset + 256]
            maxima = (lexical[block] @ lexical[references].T).max(axis=1).toarray().ravel()
            for index, similarity in zip(block, maxima, strict=True):
                consumed += 1
                if similarity >= THRESHOLD:
                    rejected += 1
                else:
                    selected.append(int(index))
                    maxima_kept.append(float(similarity))
                if len(selected) == count:
                    break
            if len(selected) == count:
                break
        if len(selected) != count:
            raise ValueError("Cannot create requested holdout at the exclusion threshold")
        splits[name] = selected
        excluded[name] = rejected
        maximum[name] = max(maxima_kept)
        references.extend(selected)
        remaining = remaining[consumed:]
    rows = {name: [clean[index] for index in indices] for name, indices in splits.items()}
    used = [{r["id"] for r in values} for values in rows.values()]
    assert not (used[0] & used[1] or used[0] & used[2] or used[1] & used[2])
    assert not set.union(*used) & old_ids
    # Test labels stored separately; neither development embedding nor tuning loads them.
    write_json(work / "support.json", rows["support"])
    write_json(work / "validation.json", rows["validation"])
    write_json(work / "test.json", [{"id": r["id"], "text": r["text"]} for r in rows["test"]])
    write_json(
        work / "sealed-targets.json", [{"id": r["id"], "label": r["label"]} for r in rows["test"]]
    )
    subset_orders = []
    for seed in SUBSET_SEEDS:
        order = np.arange(2000) if seed == 0 else np.random.default_rng(seed).permutation(2000)
        subset_orders.append({"seed": seed, "positions": order.tolist()})
    protocol = {
        "created_at_utc": timestamp(),
        "dataset": DATASET,
        "dataset_revision": DATA_REVISION,
        "source_sha256": sha_file(source),
        "source_rows": len(raw),
        "clean_unique_rows": len(clean),
        "old_development_rows_excluded": len(old_ids),
        "seed": SEED,
        "support_sizes": SIZES,
        "subset_orders": subset_orders,
        "split_ids": {n: [r["id"] for r in rs] for n, rs in rows.items()},
        "near_duplicate_threshold": THRESHOLD,
        "near_duplicate_excluded": excluded,
        "maximum_cross_split_lexical_similarity": maximum,
        "categories": categories,
        "labels": LABELS,
        "descriptions": DESCRIPTIONS,
        "instructions": INSTRUCTIONS,
        "embedding_model": "google/embeddinggemma-2",
        "embedding_revision": MODEL_REVISION,
        "jev_model": MODEL,
        "weight_updates": False,
        "primary_metric": "Jev agreement on 10,000 held-out messages",
        "dataset_branch_primary_metric": "Agreement with synthetic dataset reference labels",
        "primary_comparison": (
            "validation-selected Jev method vs hard centroid, 1,000 examples, seed 0"
        ),
        "secondary_metrics": [
            "macro F1",
            "Cohen kappa",
            "category agreement",
            "reference label accuracy",
        ],
        "baselines": [
            "uniform_random",
            "prior_random",
            "majority",
            "description_zero_shot",
            "nearest",
            "centroid",
            "weighted_knn",
        ],
        "improvements": ["confidence_filtered_centroid", "soft_teacher_knn", "hybrid_vote_prior"],
        "search_space": {
            "k": methods.K_GRID,
            "temperature": methods.T_GRID,
            "confidence_threshold": methods.FILTER_GRID,
            "blend_alpha": methods.ALPHA_GRID,
            "prior_temperature": methods.PRIOR_T_GRID,
            "priors": ["centroid", "description"],
        },
        "selection": "Max validation teacher agreement for Jev branch; max validation dataset "
        "label agreement for dataset branch. All ties use fixed enumeration order.",
        "random_baseline": "Fixed seed predictions plus exact expected uniform accuracy 1/27; "
        "100 independent random-classification seeds summarized at final evaluation.",
        "inference": "One frozen embedding forward per incoming message; vector operations only",
        "limitations": [
            "Synthetic English short queries, not real emails",
            "No complete template-family holdout",
            "No assurance that pretrained models never saw this public dataset",
            "Five subsets share the same 2,000-example bank; they are not independent datasets",
        ],
    }
    write_json(report / "protocol.json", protocol)
    print(
        json.dumps(
            {
                "split_sizes": {n: len(rs) for n, rs in rows.items()},
                "near_duplicate_excluded": excluded,
                "maximum_cross_split_similarity": maximum,
            }
        ),
        flush=True,
    )


def embed(work, report, test=False):
    if test and not (report / "frozen.json").exists():
        raise ValueError("Freeze development choices before embedding final test")
    backend = EmbeddingGemmaBackend(revision=MODEL_REVISION, batch_size=32)
    start = time.perf_counter()
    result = {}
    names = ("test",) if test else ("support", "validation")
    for name in names:
        rows = read_json(work / f"{name}.json")
        print(f"Embedding {name}: {len(rows)}", flush=True)
        result[name] = backend.encode([State(text=r["text"]) for r in rows])
    if not test:
        result["descriptions"] = backend.encode([State(text=d) for d in DESCRIPTIONS.values()])
    suffix = "test" if test else "development"
    np.savez_compressed(work / f"vectors-{suffix}.npz", **result)
    write_json(
        report / f"runtime-{suffix}.json",
        {
            "backend": json.loads(backend.identity),
            "seconds": time.perf_counter() - start,
            "forward_calls": backend.forward_calls,
            "python": platform.python_version(),
            "packages": {
                name: version(name)
                for name in (
                    "numpy",
                    "torch",
                    "sentence-transformers",
                    "transformers",
                    "scikit-learn",
                    "scipy",
                )
            },
        },
    )
    backend.close()


def label(work, report, ledger_path, env, budget, concurrency, test=False):
    if test and not (report / "frozen.json").exists():
        raise ValueError("Freeze development choices before labeling final test")
    ledger = Ledger(ledger_path, budget)
    try:
        for name in ("test",) if test else ("support", "validation"):
            print(f"Jev labeling {name}", flush=True)
            responses = asyncio.run(
                label_rows(
                    read_json(work / f"{name}.json"),
                    ledger,
                    DESCRIPTIONS,
                    INSTRUCTIONS,
                    load_key(env),
                    concurrency,
                )
            )
            write_json(work / f"jev-{name}.json", responses)
    finally:
        write_json(report / "spending.json", ledger.summary())
        ledger.close()


def teacher_data(rows, responses):
    by_id = {row["id"]: row["response"] for row in responses}
    labels, distributions, confidence = [], [], []
    for row in rows:
        body = by_id[row["id"]]
        if body["model"] != MODEL:
            raise ValueError("Teacher version drift")
        answer = body["answers"]["intent"]
        labels.append(LABELS.index(answer["choice"]))
        distribution = np.array([answer["probabilities"][key] for key in LABELS], dtype=np.float32)
        if (
            not np.isfinite(distribution).all()
            or (distribution < 0).any()
            or distribution.sum() <= 0
        ):
            raise ValueError("Invalid teacher probability distribution")
        distributions.append(distribution / distribution.sum())
        confidence.append(float(answer["confidence"]))
    return np.array(labels), np.array(distributions), np.array(confidence)


def development(work):
    support = read_json(work / "support.json")
    validation = read_json(work / "validation.json")
    vectors = np.load(work / "vectors-development.npz")
    teacher = teacher_data(support, read_json(work / "jev-support.json"))
    val_teacher = teacher_data(validation, read_json(work / "jev-validation.json"))[0]
    reference = np.array([LABELS.index(r["label"]) for r in support])
    val_reference = np.array([LABELS.index(r["label"]) for r in validation])
    return (
        vectors,
        {
            "jev": (*teacher, val_teacher),
            "dataset": (
                reference,
                np.eye(len(LABELS), dtype=np.float32)[reference],
                np.ones(len(reference)),
                val_reference,
            ),
        },
        val_reference,
        val_teacher,
    )


def tune(work, report):
    if (report / "results.json").exists():
        raise ValueError("Final evaluation exists; do not retune on this holdout")
    if (report / "frozen.json").exists():
        print("Already frozen; reuse choices", flush=True)
        return
    vectors, sources, _, _ = development(work)
    protocol = read_json(report / "protocol.json")
    chosen = []
    for subset in protocol["subset_orders"]:
        order = np.array(subset["positions"])
        for size in SIZES:
            indices = order[:size]
            for source, (labels, distributions, confidence, target) in sources.items():
                settings = methods.tune(
                    vectors["support"][indices],
                    labels[indices],
                    distributions[indices],
                    confidence[indices],
                    vectors["validation"],
                    target,
                    vectors["descriptions"],
                    source,
                )
                chosen.append(
                    {
                        "source": source,
                        "support_size": size,
                        "subset_seed": subset["seed"],
                        "indices": indices.tolist(),
                        "methods": settings,
                    }
                )
                print(
                    json.dumps(
                        {
                            "source": source,
                            "size": size,
                            "seed": subset["seed"],
                            "selected": settings["best"],
                            "validation_score": settings[settings["best"]][
                                "validation_primary_score"
                            ],
                        }
                    ),
                    flush=True,
                )
    write_json(
        report / "frozen.json",
        {
            "frozen_at_utc": timestamp(),
            "protocol_sha256": sha_file(report / "protocol.json"),
            "methods_sha256": sha_file(Path(methods.__file__)),
            "code_sha256": {
                str(path): sha_file(path)
                for path in (
                    Path("experiments/paper_methods.py"),
                    Path("experiments/paper_protocol.py"),
                    Path("experiments/email_routing.py"),
                    Path("src/emb_jev/backend.py"),
                    Path("src/emb_jev/types.py"),
                )
            },
            "input_sha256": {
                name: sha_file(work / name)
                for name in (
                    "support.json",
                    "validation.json",
                    "test.json",
                    "sealed-targets.json",
                    "jev-support.json",
                    "jev-validation.json",
                )
            },
            "development_vectors_sha256": sha_file(work / "vectors-development.npz"),
            "settings": chosen,
        },
    )


def wilson(successes, count):
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [center - half, center + half]


def measure(predictions, teacher, reference, mapping):
    result = {}
    for name, target in (("teacher", teacher), ("dataset", reference)):
        matches = predictions == target
        result[f"{name}_agreement"] = float(matches.mean())
        result[f"{name}_agreement_95ci"] = wilson(int(matches.sum()), len(matches))
        result[f"{name}_macro_f1"] = float(
            f1_score(
                target, predictions, labels=range(len(LABELS)), average="macro", zero_division=0
            )
        )
        result[f"{name}_kappa"] = float(
            cohen_kappa_score(target, predictions, labels=range(len(LABELS)))
        )
        result[f"{name}_category_agreement"] = float(
            accuracy_score(mapping[target], mapping[predictions])
        )
    return result


def paired_comparison(candidate, baseline, target):
    cand_correct, base_correct = candidate == target, baseline == target
    gained = int(np.sum(cand_correct & ~base_correct))
    lost = int(np.sum(~cand_correct & base_correct))
    different = gained + lost
    probability = different / len(target)
    rng = np.random.default_rng(SEED)
    # Bootstrap paired correctness outcomes in aggregate; exactly the same
    # empirical paired distribution as resampling all test rows, without 200MB indices.
    resamples = rng.multinomial(
        len(target), [gained / len(target), lost / len(target), 1 - probability], size=10000
    )
    differences = (resamples[:, 0] - resamples[:, 1]) / len(target)
    return {
        "accuracy_difference": (gained - lost) / len(target),
        "difference_95ci": np.quantile(differences, [0.025, 0.975]).tolist(),
        "candidate_only_correct": gained,
        "baseline_only_correct": lost,
        "mcnemar_exact_p": float(binomtest(gained, different, 0.5).pvalue) if different else 1.0,
        "paired_bootstrap_resamples": 10000,
        "seed": SEED,
    }


def evaluate(work, report):
    frozen = read_json(report / "frozen.json")
    if frozen["protocol_sha256"] != sha_file(report / "protocol.json"):
        raise ValueError("Protocol changed after freeze")
    if frozen["methods_sha256"] != sha_file(Path(methods.__file__)):
        raise ValueError("Prediction methods changed after freeze")
    if frozen["development_vectors_sha256"] != sha_file(work / "vectors-development.npz"):
        raise ValueError("Development embeddings changed after freeze")
    for name, digest in frozen["code_sha256"].items():
        if sha_file(Path(name)) != digest:
            raise ValueError("Prediction/experiment code changed after freeze")
    for name, digest in frozen["input_sha256"].items():
        if sha_file(work / name) != digest:
            raise ValueError("Frozen input changed after freeze")
    protocol = read_json(report / "protocol.json")
    vectors, sources, _, _ = development(work)
    inputs = read_json(work / "test.json")
    teacher = teacher_data(inputs, read_json(work / "jev-test.json"))[0]
    by_id = {r["id"]: r["label"] for r in read_json(work / "sealed-targets.json")}
    reference = np.array([LABELS.index(by_id[row["id"]]) for row in inputs])
    queries = np.load(work / "vectors-test.npz")["test"]
    categories = sorted(set(protocol["categories"].values()))
    mapping = np.array([categories.index(protocol["categories"][key]) for key in LABELS])
    assert len(inputs) == len(queries) == 10000
    uniform = np.random.default_rng(SEED).integers(len(LABELS), size=len(inputs))
    zero_shot = (queries @ vectors["descriptions"].T).argmax(axis=1)
    common = {
        "uniform_random": measure(uniform, teacher, reference, mapping),
        "description_zero_shot": measure(zero_shot, teacher, reference, mapping),
        "jev_direct": measure(teacher, teacher, reference, mapping),
    }
    random_runs = np.array(
        [
            measure(
                np.random.default_rng(seed).integers(len(LABELS), size=len(inputs)),
                teacher,
                reference,
                mapping,
            )["teacher_agreement"]
            for seed in range(100)
        ]
    )
    results, primary_predictions = [], {}
    all_similarities = queries @ vectors["support"].T
    neighbor_cache = {}
    for setting in frozen["settings"]:
        size, source, seed = setting["support_size"], setting["source"], setting["subset_seed"]
        indices = np.array(setting["indices"])
        labels, distributions, confidence, _ = sources[source]
        arguments = (
            vectors["support"][indices],
            labels[indices],
            distributions[indices],
            confidence[indices],
            queries,
            vectors["descriptions"],
        )
        cache_key = (size, seed)
        if cache_key not in neighbor_cache:
            subset_similarities = all_similarities[:, indices]
            neighbor_cache[cache_key] = np.argsort(-subset_similarities, axis=1, kind="stable")[
                :, : max(methods.K_GRID)
            ]
        subset_similarities = all_similarities[:, indices]
        predictions = {}
        for name, config in setting["methods"].items():
            if name == "best":
                continue
            predictions[name] = methods.predict(
                config,
                *arguments,
                similarities=subset_similarities,
                indices=neighbor_cache[cache_key],
            )
        counts = np.bincount(labels[indices], minlength=len(LABELS))
        predictions["majority"] = np.full(len(inputs), counts.argmax())
        predictions["prior_random"] = np.random.default_rng(SEED + size + seed).choice(
            len(LABELS), size=len(inputs), p=counts / counts.sum()
        )
        for name, pred in predictions.items():
            results.append(
                {
                    "source": source,
                    "support_size": size,
                    "subset_seed": seed,
                    "method": name,
                    "selected_on_validation": name == setting["methods"]["best"],
                    **measure(pred, teacher, reference, mapping),
                }
            )
        if seed == 0:
            for name, pred in predictions.items():
                primary_predictions[f"{source}_{size}_{name}"] = pred
            primary_predictions[f"{source}_{size}_selected"] = predictions[
                setting["methods"]["best"]
            ]
    comparisons = []
    for source, target in (("jev", teacher), ("dataset", reference)):
        for size in SIZES:
            prefix = f"{source}_{size}_"
            comparisons.append(
                {
                    "source": source,
                    "size": size,
                    "comparison": "selected_vs_centroid",
                    **paired_comparison(
                        primary_predictions[prefix + "selected"],
                        primary_predictions[prefix + "centroid"],
                        target,
                    ),
                }
            )
    # Holm adjustment across the six predeclared selected-vs-centroid comparisons.
    order = sorted(range(len(comparisons)), key=lambda i: comparisons[i]["mcnemar_exact_p"])
    previous = 0
    for rank, index in enumerate(order):
        previous = max(
            previous, min(1, comparisons[index]["mcnemar_exact_p"] * (len(order) - rank))
        )
        comparisons[index]["holm_adjusted_p"] = previous
    summary = []
    groups = {}
    for row in results:
        groups.setdefault((row["source"], row["support_size"], row["method"]), []).append(row)
    for (source, size, method), rows in groups.items():
        summary.append(
            {
                "source": source,
                "support_size": size,
                "method": method,
                "subset_runs": len(rows),
                **{
                    key + "_mean": float(np.mean([r[key] for r in rows]))
                    for key in ("teacher_agreement", "dataset_agreement")
                },
                **{
                    key + "_std": float(np.std([r[key] for r in rows], ddof=1))
                    for key in ("teacher_agreement", "dataset_agreement")
                },
            }
        )
    final = {
        "evaluated_at_utc": timestamp(),
        "frozen_sha256": sha_file(report / "frozen.json"),
        "test_size": len(inputs),
        "baselines": common,
        "runs": results,
        "subset_summary": summary,
        "paired_comparisons": comparisons,
        "uniform_random_expected_agreement": 1 / len(LABELS),
        "uniform_random_100_runs_teacher_agreement_mean": float(random_runs.mean()),
        "uniform_random_100_runs_teacher_agreement_std": float(random_runs.std(ddof=1)),
    }
    write_json(report / "results.json", final)
    with (report / "metrics.csv").open("w", newline="") as stream:
        fields = [key for key in results[0] if not key.endswith("95ci")]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    with (report / "predictions.csv").open("w", newline="") as stream:
        fields = [
            "source_row_id",
            "dataset_reference",
            "jev_direct",
            "uniform_random",
            "description_zero_shot",
            *primary_predictions,
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(inputs):
            writer.writerow(
                {
                    "source_row_id": row["id"],
                    "dataset_reference": LABELS[reference[index]],
                    "jev_direct": LABELS[teacher[index]],
                    "uniform_random": LABELS[uniform[index]],
                    "description_zero_shot": LABELS[zero_shot[index]],
                    **{name: LABELS[pred[index]] for name, pred in primary_predictions.items()},
                }
            )
    print(
        json.dumps(
            {
                "baselines": common,
                "paired_comparisons": comparisons,
                "main_runs": [r for r in results if r["subset_seed"] == 0],
            },
            indent=2,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--work", type=Path, default=Path(".cache/paper"))
    parser.add_argument("--report", type=Path, default=Path("reports/bitext-paper"))
    parser.add_argument("--old-report", type=Path, default=Path("reports/bitext-1000"))
    parser.add_argument("--source", type=Path, default=Path(".cache/experiment/bitext.csv"))
    parser.add_argument("--ledger", type=Path, default=Path(".cache/experiment/jev-ledger.sqlite3"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--budget-usd", default="4")
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 32:
        parser.error("concurrency must be 1..32")
    for stage in STAGES:
        if args.stage not in (stage, "all"):
            continue
        if stage == "prepare":
            prepare(args.work, args.report, args.old_report, args.source)
        elif stage.startswith("embed_"):
            embed(args.work, args.report, test=stage == "embed_test")
        elif stage.startswith("label_"):
            label(
                args.work,
                args.report,
                args.ledger,
                args.env,
                args.budget_usd,
                args.concurrency,
                test=stage == "label_test",
            )
        elif stage == "tune":
            tune(args.work, args.report)
        else:
            evaluate(args.work, args.report)


if __name__ == "__main__":
    main()
