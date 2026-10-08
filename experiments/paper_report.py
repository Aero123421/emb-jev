# ruff: noqa: E501 -- Markdown tables in the report template require long lines.
"""Render the frozen experiment's results without changing prediction rules."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from .email_routing import LABELS, read_json, write_json

REPORT = Path("reports/bitext-paper")
NAMES = {
    "nearest": "1近傍",
    "centroid": "カテゴリ平均",
    "weighted_knn": "複数近傍の投票",
    "filtered_centroid": "低確信の見本を除く平均",
    "soft_knn": "Jevの全確率を使う近傍",
    "hybrid": "近傍＋平均/説明の混合",
    "majority": "見本の最多カテゴリ",
    "prior_random": "見本の比率でランダム分類",
}


def pct(value):
    return f"{100 * value:.2f}%"


def main():
    result = read_json(REPORT / "results.json")
    frozen = read_json(REPORT / "frozen.json")
    protocol = read_json(REPORT / "protocol.json")
    spending = read_json(REPORT / "spending.json")
    configs = {
        (row["source"], row["support_size"]): row
        for row in frozen["settings"]
        if row["subset_seed"] == 0
    }
    runs = {
        (row["source"], row["support_size"], row["method"]): row
        for row in result["runs"]
        if row["subset_seed"] == 0
    }
    main_table = []
    for source in ("jev", "dataset"):
        for size in protocol["support_sizes"]:
            method = configs[(source, size)]["methods"]["best"]
            base = runs[(source, size, "centroid")]
            improved = runs[(source, size, method)]
            main_table.append(
                f"| {'Jev' if source == 'jev' else '既存ラベル'} | {size:,} | "
                f"{pct(base['teacher_agreement'])} | {pct(improved['teacher_agreement'])} | "
                f"{pct(base['dataset_agreement'])} | {pct(improved['dataset_agreement'])} | "
                f"{NAMES[method]} |"
            )
    details = []
    for source in ("jev", "dataset"):
        for size in protocol["support_sizes"]:
            for method in NAMES:
                if (source, size, method) not in runs:
                    continue
                row = runs[(source, size, method)]
                details.append(
                    f"| {'Jev' if source == 'jev' else '既存ラベル'} | {size:,} | {NAMES[method]} | "
                    f"{pct(row['teacher_agreement'])} | {pct(row['dataset_agreement'])} | "
                    f"{row['teacher_macro_f1']:.4f} | {row['dataset_macro_f1']:.4f} |"
                )
    comparisons = []
    for row in result["paired_comparisons"]:
        low, high = row["difference_95ci"]
        comparisons.append(
            f"| {row['source']} | {row['size']:,} | {100 * row['accuracy_difference']:+.2f} | "
            f"[{100 * low:+.2f}, {100 * high:+.2f}] | {row['candidate_only_correct']} | "
            f"{row['baseline_only_correct']} | {row['holm_adjusted_p']:.3g} |"
        )
    main_config = configs[("jev", 1000)]
    selected = main_config["methods"]["best"]
    main_row = runs[("jev", 1000, selected)]
    main_base = runs[("jev", 1000, "centroid")]
    primary = next(
        row
        for row in result["paired_comparisons"]
        if row["source"] == "jev" and row["size"] == 1000
    )
    baselines = result["baselines"]
    selections = json.dumps(
        [
            {
                "source": source,
                "support_size": size,
                "selected_method": row["methods"]["best"],
                "configuration": row["methods"][row["methods"]["best"]],
            }
            for (source, size), row in configs.items()
        ],
        ensure_ascii=False,
        indent=2,
    )
    summary_rows = []
    for row in result["subset_summary"]:
        if row["method"] in ("centroid", "hybrid"):
            summary_rows.append(
                f"| {row['source']} | {row['support_size']:,} | {NAMES[row['method']]} | "
                f"{100 * row['teacher_agreement_mean']:.2f} ± {100 * row['teacher_agreement_std']:.2f} | "
                f"{100 * row['dataset_agreement_mean']:.2f} ± {100 * row['dataset_agreement_std']:.2f} |"
            )
    text = f"""# 固定EmbeddingによるJev判定の近似：漏洩を防いだ10,000件比較

## 要旨

Jevを見本のラベル付けに使い、その見本をEmbeddingGemma 2で埋め込むと、
新しい入力をJev APIなしでどこまで同じように分類できるかを検証した。
英語の合成問い合わせ27分類で、見本500・1,000・2,000件を比較し、
すべての条件に共通の未使用テスト10,000件を使用した。
**主指標はJev直接判定との一致率**である。
既存データセットラベルとの一致率は別の指標として併記した。

事前指定した主比較（Jev見本1,000件）では、カテゴリ平均の
{pct(main_base["teacher_agreement"])}に対し、調整用データで選んだ{NAMES[selected]}は
{pct(main_row["teacher_agreement"])}で、差は{100 * primary["accuracy_difference"]:+.2f}ポイントだった。
一様ランダム分類のJev一致率は{pct(baselines["uniform_random"]["teacher_agreement"])}。
モデルの重み更新・勾配学習・追加ヘッドの学習は行っていない。

## 1. 研究課題と指標

第一の問いは「Jevが見本を分類し、固定Embeddingと検索で未使用入力のJev判定を再現できるか」。
ここではJevの直接判定を教師の目標とし、一致率・Macro F1・Cohenのκを測定する。
Jev自身が正しいかどうかを測る指標とは区別する。

第二の問いは「既存ラベル付き見本をEmbeddingにするだけでデータセットの分類を再現できるか」。
既存ラベルを持つ同じ見本を使い、データセットラベルとの一致率を測定する。
データセットは合成データであり、そのラベルを人が独立に検証した正解とは扱わない。

## 2. データと分割

データは[Bitext Customer Support](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)。
コミット`{protocol["dataset_revision"]}`のCSV26,872行、27 intent、11 category。
本文は`instruction`だけを使用し、回答文・intent・categoryをモデルへ渡さない。
完全なメールのデータではなく、短い英語の合成問い合わせである。ライセンスはCDLA-Sharing-1.0。

Unicode NFKC、大小文字統一、句読点除去、プレースホルダ置換後の完全重複を除き、
{protocol["clean_unique_rows"]:,}件が残った。
最初の予備実験で結果を見た2,500件は、本実験の見本・調整・テストからすべて除いた。
新しいランダムseed `{protocol["seed"]}`で次の分割を先に固定した。

| 分割 | 件数 | 用途 |
| --- | ---: | --- |
| 見本バンク | 2,000 | Jevラベル版と既存ラベル版の両方で共有 |
| 調整用 | 1,000 | 設定選択。Jev一致率／既存ラベル一致率をそれぞれ目的にする |
| テスト | 10,000 | 最終評価のみ。全条件で共通 |

文字3〜5-gram TF-IDFによるコサイン類似度0.90以上の近似文も分割境界で除外した。
調整用は見本＋予備実験データとの近似文を{protocol["near_duplicate_excluded"]["validation"]:,}件、
テストは見本＋調整用＋予備実験データとの近似文を
{protocol["near_duplicate_excluded"]["test"]:,}件除外した。
最大の交差分割類似度は調整用{protocol["maximum_cross_split_lexical_similarity"]["validation"]:.6f}、
テスト{protocol["maximum_cross_split_lexical_similarity"]["test"]:.6f}。
TF-IDFは分割の検査だけに使い、分類器には使わない。

見本500・1,000件は2,000件バンクの先頭部分を使う入れ子の集合。
抽出のばらつきも確認するため、同じバンクの並べ替え5通りを使用した。
最大2,000件では同じバンクをすべて使うため、5つの独立データセットではない。

## 3. 漏洩防止と事前固定

- 見本・調整・テストの行IDは互いに素で、予備実験2,500件とも交差しない。
- 見本に保存するのは見本の本文Embeddingと、見本のラベルまたはJev確率分布だけ。
  調整用・テストの入力やラベルを見本キャッシュへ追加しない。
- テストの既存ラベルは別ファイルに分離し、設定選択コードは読み込まない。
- テストのJev分類・Embeddingを作る前に、全設定を確定した。
- `[protocol.json](protocol.json)`は分割・指標・探索候補を記録し、
  `[frozen.json](frozen.json)`は採用設定、設定固定時刻、コードと入力のSHA-256を保存する。
- 固定時刻は`{frozen["frozen_at_utc"]}`、最終評価時刻は`{result["evaluated_at_utc"]}`。
  固定後にコードや入力が変わると最終評価コードは停止する。
- テスト結果を見た後の設定変更・候補説明の書き直し・再選択は行っていない。

これは行重複、定義した近似文、設定選択による漏洩を防ぐ設計である。
未知の生成テンプレート全体を分離できた保証や、事前学習にこの公開データが含まれない保証はない。

## 4. モデルと分類方式

Embeddingは[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2)のtext-only構成、
768次元、BF16。コミット`{protocol["embedding_revision"]}`に固定した。
本文と候補説明には公式の対称な分類プレフィックス
`task: classification | query:`を付け、ベクトルを長さ1に正規化する。
Jevは`{protocol["jev_model"]}`、全リクエストで同じ27候補説明と判断指示を使用する。

| 方式 | 仕組み |
| --- | --- |
| 一様ランダム | 27候補を等確率で選ぶ。固定seedと100 seed平均を報告 |
| 比率ランダム | 見本のラベル比率でランダムに選ぶ |
| 最多カテゴリ | 見本で最も多いラベルを全件に返す |
| 説明文のみ | 入力と27候補の説明Embeddingを比較 |
| 1近傍 | 最も近い1件の見本ラベルを返す |
| カテゴリ平均 | ラベルごとに見本ベクトルを平均・正規化し、入力との類似度を比較 |
| 複数近傍投票 | 近いk件にコサイン類似度のsoftmax重みを付けてラベルを投票 |
| 低確信の見本を除く平均 | Jevのconfidenceがしきい値以上の見本で平均。消える分類は未フィルタ平均に戻す |
| 全確率の近傍 | Jevのargmaxだけでなく、見本の全27候補の確率分布を重み付きで平均 |
| 混合 | 近傍の分布と、平均ベクトル／候補説明から得た相対重みを混ぜる |

全確率の近傍は、入力q、見本s_i、Jev分布P_iに対して
`w_i = softmax(cos(E(q), E(s_i))/T)`、`V(q) = Σ w_i P_i`を計算する。
既存ラベル版ではP_iをone-hotとする。
混合方式は`(1−α)V(q) + α softmax(prior_similarity/τ)`を使う。
校正済みの正解確率とは主張しない。入力はモデルに1回通し、残りはベクトル計算だけである。

近傍数`1/3/5/9/15/31/63`、温度`0.02/0.05/0.1/0.2`、
confidenceしきい値`0/0.5/0.7/0.8/0.9/0.95`、
混合比`0/0.25/0.5/0.75/1`、事前重み温度`0.01/0.03/0.07/0.15`を探索した。
見本の件数・並べ替え・ラベル源ごとに、調整用の主指標だけで選択した。
同点は固定された探索順で解決する。
「改善方式」は調整用で選ばれた方式を意味し、テストでの改善を保証する呼び名ではない。

## 5. 主結果：10,000件での比較

まず、全条件共通の基準を示す。

| 基準 | Jevとの一致率 | データセットラベルとの一致率 |
| --- | ---: | ---: |
| 一様ランダム・固定seed | {pct(baselines["uniform_random"]["teacher_agreement"])} | {pct(baselines["uniform_random"]["dataset_agreement"])} |
| 候補説明文のみ | {pct(baselines["description_zero_shot"]["teacher_agreement"])} | {pct(baselines["description_zero_shot"]["dataset_agreement"])} |
| Jev直接判定 | 100.00% | {pct(baselines["jev_direct"]["dataset_agreement"])} |

一様ランダムの期待一致率は1/27 = {pct(result["uniform_random_expected_agreement"])}。
100 seedでのJev一致率は平均{pct(result["uniform_random_100_runs_teacher_agreement_mean"])}、
標準偏差{100 * result["uniform_random_100_runs_teacher_agreement_std"]:.3f}ポイントだった。

| 見本のラベル源 | 見本数 | 平均方式：Jev一致 | 選択方式：Jev一致 | 平均方式：既存ラベル一致 | 選択方式：既存ラベル一致 | 調整用で選んだ方式 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
{chr(10).join(main_table)}

ここは主並べ替えseed 0の結果。Jevラベル版はJev一致率を目的に、
既存ラベル版は既存ラベル一致率を目的に設定を選んでいる。
両列を同じ目的に最適化した結果として解釈しない。

![見本件数・改善方式・ランダム分類の比較](comparison.png)

## 6. 全方式の比較

失敗した改善候補やランダム分類も含め、全方式を掲載する。

| ラベル源 | 見本数 | 方式 | Jev一致率 | 既存ラベル一致率 | Jev Macro F1 | 既存ラベル Macro F1 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
{chr(10).join(details)}

## 7. 不確実性と対比較

各一致率の95% Wilson区間、Cohenのκ、大分類一致率は[results.json](results.json)に保存した。
採用方式とカテゴリ平均を**同じテスト行**で比較し、
差のbootstrap 10,000回の95%区間とexact McNemar検定を求めた。
6つの事前指定した比較についてHolm補正したp値も示す。
Jev源は教師判定を、既存ラベル源はデータセットラベルを評価目標とする。

| ラベル源 | 見本数 | 差（ポイント） | 95%区間 | 選択方式のみ一致 | 平均方式のみ一致 | Holm補正p |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
{chr(10).join(comparisons)}

主比較はJev源・1,000件。その他は副次的比較である。
![カテゴリ平均からの変化と95%区間](improvement.png)

行単位の区間・検定はテスト行の独立性を仮定し、
合成テンプレート由来の相関や未知分野への分布変化は扱っていない。
小さなp値だけで実メールでも同じ改善があるとは解釈しない。

## 8. 見本抽出のばらつき

同じ2,000件バンクの並べ替え5通りの平均 ± 標準偏差（ポイント）。
同じテスト集合を使い、各方式の設定はそれぞれ調整用で選んだ。
ここでは選択の結果によらず、平均方式と混合方式を固定して比較する。

| ラベル源 | 見本数 | 方式 | Jev一致率：平均±SD | 既存ラベル一致率：平均±SD |
| --- | ---: | --- | ---: | ---: |
{chr(10).join(summary_rows)}

最大2,000件では同じ見本集合であるため、このSDは独立な抽出実験の不確実性ではない。

## 9. 再現用ファイルと料金

- [protocol.json](protocol.json): 分割ID、seed、説明文、探索候補、モデルバージョン。
- [frozen.json](frozen.json): テスト前に固定した設定とハッシュ。
- [results.json](results.json) / [metrics.csv](metrics.csv): 全方式・全並べ替えの数値。
- `predictions.csv`（ローカルのみ・Git対象外）: 10,000行のJev判定・既存ラベル・各方式予測。
- [per_class.json](per_class.json): 主並べ替えの各分類ごとの指標と混同行列。
- [runtime-development.json](runtime-development.json) / [runtime-test.json](runtime-test.json): 実行環境。
- [spending.json](spending.json): 予備実験から通算したAPI使用量。

本実験は見本2,000＋調整1,000＋テスト10,000件の13,000件をJevに直接分類させた。
予備実験2,000呼び出しを含めた台帳は{spending["successful_calls"]:,}成功呼び出し、
{spending["input_tokens"]:,} input tokens。
[公式料金](https://docs.typesafe.ai/models)からの通算料金推計は
**${spending["successful_usage_estimated_usd"]:.6f}**。
提供者の請求明細を照合した金額ではない。
各試行前の最大料金予約・不明な応答の予約保持を含め、通算上限$4を維持した。

生データ・生APIレスポンス・Embeddingは`.cache/paper/`、
API台帳は`.cache/experiment/jev-ledger.sqlite3`に保存する。
データセット本体・生レスポンス・ベクトル・行ごとの予測・APIキーはGit対象外。
Gitにはコード、集計結果、図、分割IDと設定の再現用メタデータを含める。
保存済みレスポンスは同じ要求の再実行で再利用する。

```bash
uv pip install --python .venv/bin/python -e '.[model,dev,experiment]'
export HF_HOME="$PWD/.cache/huggingface"
# 元CSVがなければ予備実験のprepareだけでダウンロードする（API呼び出しなし）
.venv/bin/python -m experiments.email_routing --stage prepare
# 既存の固定済み実験を上書きしないため、再現には別のwork/reportを指定する
.venv/bin/python -m experiments.paper_protocol --stage all \\
  --work .cache/reproduction --report reports/reproduction --budget-usd 4
```

`all`のlabel段階は課金される。固定済みの本実験を確認する場合は保存済みファイルを使う。
`--stage prepare/embed_development/label_development/tune/label_test/embed_test/evaluate`で段階実行も可能。
最終評価の再現は固定した設定をそのまま使い、再調整しない。
Jevの再呼び出しが同一結果になる保証はないので、保存済みレスポンスを保持する。

## 10. 限界と論文で主張できる範囲

この結果は、指定した公開データ・教師バージョン・固定Embedding・見本バンクでの判定近似を示す。
「Jevと同じ知能」「確率まで校正された代替」「日本語の実メールでも同じ性能」
「生成テンプレート全体を分けた未知ドメイン評価」とは主張できない。
画像・音声・複数用件のメール・未知カテゴリは本実験の対象外である。
モデル本体は小型化しておらず、量子化も行っていない。
データセットラベルとの一致率とJev判定との一致率は、別々の目標として記述する必要がある。

## 付録：主並べ替えで固定した選択設定

```json
{selections}
```
"""
    (REPORT / "README.md").write_text(text)
    per_class()
    plot(result)


def per_class():
    with (REPORT / "predictions.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for name in rows[0]:
        if name in ("source_row_id", "dataset_reference", "jev_direct"):
            continue
        if not (name.endswith("selected") or name in ("uniform_random", "description_zero_shot")):
            continue
        pred = np.array([LABELS.index(row[name]) for row in rows])
        result[name] = {}
        for target_name in ("jev_direct", "dataset_reference"):
            target = np.array([LABELS.index(row[target_name]) for row in rows])
            result[name][target_name] = {
                "classification_report": classification_report(
                    target,
                    pred,
                    labels=range(len(LABELS)),
                    target_names=LABELS,
                    output_dict=True,
                    zero_division=0,
                ),
                "confusion_matrix": confusion_matrix(
                    target, pred, labels=range(len(LABELS))
                ).tolist(),
            }
    write_json(REPORT / "per_class.json", {"labels": LABELS, "results": result})


def plot(result):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    styles = {
        "centroid": ("#2563eb", "Centroid"),
        "weighted_knn": ("#0d9488", "Hard-label kNN"),
        "soft_knn": ("#f59e0b", "Soft-teacher kNN"),
        "hybrid": ("#dc2626", "Hybrid"),
    }
    for ax, source, target, title in (
        (axes[0], "jev", "teacher", "Match Jev (Jev-labeled examples)"),
        (axes[1], "dataset", "dataset", "Match dataset (dataset-labeled examples)"),
    ):
        for method, (color, label) in styles.items():
            rows = [
                r
                for r in result["subset_summary"]
                if r["source"] == source and r["method"] == method
            ]
            if not rows:
                continue
            rows.sort(key=lambda row: row["support_size"])
            ax.errorbar(
                [r["support_size"] for r in rows],
                [100 * r[f"{target}_agreement_mean"] for r in rows],
                yerr=[100 * r[f"{target}_agreement_std"] for r in rows],
                color=color,
                label=label,
                marker="o",
                capsize=4,
            )
        ax.axhline(
            result["baselines"]["uniform_random"][f"{target}_agreement"] * 100,
            color="#6b7280",
            ls="--",
            label="Uniform random",
        )
        ax.axhline(
            result["baselines"]["description_zero_shot"][f"{target}_agreement"] * 100,
            color="#a855f7",
            ls=":",
            label="Descriptions only",
        )
        ax.set(
            title=title,
            xlabel="Cached examples",
            ylabel="Agreement (%)",
            ylim=(0, 100),
            xticks=[500, 1000, 2000],
        )
        ax.grid(alpha=0.2)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Same 10,000 held-out queries; mean +/- SD of 5 nested subsets")
    fig.savefig(REPORT / "comparison.png", dpi=200)
    fig.savefig(REPORT / "comparison.svg")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
    for ax, source, title, color in (
        (axes[0], "jev", "Gain in matching Jev", "#2563eb"),
        (axes[1], "dataset", "Gain in matching dataset", "#0d9488"),
    ):
        rows = [row for row in result["paired_comparisons"] if row["source"] == source]
        rows.sort(key=lambda row: row["size"])
        gain = np.array([row["accuracy_difference"] * 100 for row in rows])
        low = np.array([row["difference_95ci"][0] * 100 for row in rows])
        high = np.array([row["difference_95ci"][1] * 100 for row in rows])
        ax.errorbar(
            gain,
            range(3),
            xerr=np.array([gain - low, high - gain]),
            fmt="o",
            color=color,
            capsize=4,
        )
        ax.axvline(0, color="#6b7280", ls="--", lw=1)
        ax.set(
            title=title,
            xlabel="Selected method minus centroid (percentage points)",
            yticks=range(3),
            yticklabels=["500 examples", "1,000 examples", "2,000 examples"],
            ylim=(-0.5, 2.5),
            xlim=(-0.6, 3.2),
        )
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Paired bootstrap 95% intervals; same 10,000 test queries")
    fig.savefig(REPORT / "improvement.png", dpi=200)
    fig.savefig(REPORT / "improvement.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
