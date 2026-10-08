# emb-jev

EmbeddingGemma 2を使う、追加学習なし・文章生成なしのローカル判定SDK。
テキスト・画像・音声を説明付き候補と照合し、Choice / Boolean / Scoreを返します。

現在はPythonの初期実装です。モデルの重みを変更せず、候補を事前計算してキャッシュし、
判定時は**入力を1回だけモデルに通して全質問で共有**します。未知の候補は登録時に計算が必要です。

## インストール

Python 3.11以降。リポジトリから:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e '.[model,dev]'
```

`pip install -e '.[model,dev]'`でもインストールできます。
初回はHugging Faceから`google/embeddinggemma-2`をダウンロードします。
保存先をこのリポジトリ内にする場合は`HF_HOME="$PWD/.cache/huggingface"`を設定してください。
ダウンロード後は`local_files_only=True`でオフライン使用できます。

## テキスト判定

```python
from emb_jev import Engine

with Engine(modalities=["text"], cache_path=".cache/candidates.sqlite3") as engine:
    routing = engine.prepare(
        type="choice",
        options={
            "billing": "請求、支払い、二重引き落とし、返金に関する問い合わせ",
            "technical": "アプリの不具合、設定、ログイン障害に関する問い合わせ",
            "sales": "商品購入、見積もり、契約前の相談",
        },
    )
    result = routing.decide(text="同じ商品の代金が二重に引き落とされた")
    print(result.to_dict())
```

## 画像・音声と複数判定

```python
from emb_jev import Engine, State

with Engine(modalities=["text", "image", "audio"]) as engine:
    scene = engine.prepare(
        type="choice",
        options={"food": "料理や食べ物の写真", "landscape": "自然の風景の写真"},
    )
    print(scene.decide(image="photo.jpg").to_dict())

    sound = engine.prepare(
        type="choice",
        options={"speech": "人が話している音声", "music": "楽器や歌による音楽"},
    )
    speech = engine.prepare(
        type="boolean",
        options={"true": "人が言葉を話している音声", "false": "人の発話を含まない環境音や音楽"},
    )
    results = engine.decide({"sound": sound, "speech": speech}, audio="clip.wav")
    print({name: result.to_dict() for name, result in results.items()})

    # 候補は説明文だけでなくローカルの見本画像・見本音声も使えます。
    reference = engine.prepare(
        type="choice",
        options={"sample_a": State(image="a.jpg"), "sample_b": State(image="b.jpg")},
    )
```

`decide(text=..., image=..., audio=...)`で混合入力も可能です。本文→画像→音声の順で
1つの入力に組み立てます。初版は画像1枚・音声1クリップまでで、動画は対象外です。
音声はモデルの公式Processorで読み込み・前処理します。短いクリップから始めてください。
上限を超えた入力は明示的にエラーにします。自動的な区間分割は行いません。

## Scoreと保留

```python
with Engine(modalities=["text"]) as engine:
    score = engine.prepare(
        type="score",
        options={
            "low": "落ち着いた丁寧な問い合わせ",
            "middle": "少し苛立っている、待たされて不満を示している問い合わせ",
            "high": "強い怒りや解約の意向を示している問い合わせ",
        },
        score_values={"low": 1, "middle": 3, "high": 5},
        min_margin=0.05,  # 説明用の値。実際には用途別に検証してください。
    )
    result = score.decide(text="何度連絡しても解決しない。もう解約したい")
```

`score_values`は候補の順番で厳密に増加する値を指定します。`value`は選ばれた評価段階、
`expected_score`は相対重みから計算した連続値です。

出力の意味:

| フィールド | 意味 |
| --- | --- |
| `value` | 選択肢ID / bool / 評価段階の数値。保留なら`null` |
| `status` | `decided`または`abstained` |
| `selected_key` | 保留時も含めた最高スコアの候補ID。採用判断は`status`を見る |
| `similarity` | 最高のコサイン類似度 |
| `margin` | 最高候補と次点の類似度差 |
| `similarities` | 全候補のコサイン類似度 |
| `weights` | `softmax(similarities / temperature)`による相対重み |
| `expected_score` | Scoreのみ。保留なら`null` |

**`weights`は校正された正解確率ではありません。**`expected_score`も検証済みの測定値ではありません。
デフォルトの`temperature=0.1`は重みの鋭さを決めるだけで、判定の順位は変えません。
`min_similarity`・`min_margin`は初期値が`None`です。保留するには用途に合わせて指定してください。
モデルの追加学習をせずとも、用途ごとの評価サンプルで精度・しきい値の妥当性は確認してください。

## 実装と制約

- 公式Sentence Transformers 6.1 / Transformers 5.19のモデルとProcessorを使用します。
- テキストに分類プレフィックスを付けます。任意の質問への回答や論理推論を保証するAPIではありません。
  候補の説明が判定内容を定義します。Booleanの成立・不成立も具体的に記述してください。
- 複数質問は同じ入力Embeddingを使います。質問ごとに入力を再エンコードする方式ではありません。
- モデル入力はデフォルト最大2,048トークン、設定可能範囲は1〜8,192です。
  画像・音声も含めた前処理後の長さを確認し、黙って切り捨てません。
- 全候補の登録処理は`batch_size=16`ごとにモデルを実行します。登録と判定のコストを分けてください。
- 既定のベクトル次元は768です。128 / 256 / 512も選べますがモデル本体は小さくなりません。
- 全部入りは740M、テキストのみ270M、画像込み440M、音声込み570Mのモデル構成です。
  Python版のRAMや速度は実環境で測定してください。量子化バックエンドは未実装です。
- GPUで対応していればBF16、それ以外はFP32を使います。FP16は使用しません。
- キャッシュはモデルの解決済みコミット、実装バージョン、前処理、次元、モダリティ、精度、
  デバイス、候補の内容で識別します。画像・音声は内容のハッシュを使います。
  キャッシュには候補のベクトルを保存し、入力のベクトルは保存しません。
- Engineは1ワーカー / 1スレッドで使ってください。異なるEngineのタスクは混ぜられません。
- MediaPipe Decision Maker 1.1.0のPython配布コードはテキストのAPIで、
  `evaluate_batch`も質問ごとに呼び出します。このため初版はSentence Transformersを直接使っています。

## 動作例と検証

問い合わせ分類で、Jevが付けたラベルを見本ベクトルとして使う実験も実行しました。
未使用テスト10,000件、見本500・1,000・2,000件、調整用1,000件で、重み更新は行っていません。
**Jev見本2,000件の混合方式はJevの直接判定と91.41%一致**しました（ランダム分類3.94%）。
別の比較では、既存ラベルの見本2,000件でデータセットのラベルと96.60%一致しました。
英語の合成問い合わせを使った結果です。日本語・実メール・画像・音声の精度を示す結果ではありません。
データ・漏洩防止・全方式の比較・料金・再現手順は
[本実験レポート](reports/bitext-paper/README.md)を参照してください。
最初のテスト1,000件の[予備実験](reports/bitext-1000/README.md)は本実験と区別しています。

```bash
.venv/bin/python examples/decide.py --text 'カードが二重請求された'
.venv/bin/python examples/decide.py --image photo.jpg
.venv/bin/python examples/decide.py --audio clip.wav
.venv/bin/python examples/benchmark.py --text 'カードが二重請求された' --runs 10
.venv/bin/pytest
.venv/bin/ruff check .
```

通常のテストはモデルをダウンロードせず、数値処理・キャッシュ・共有実行などを検証します。
`examples/smoke.py`は実モデルで日本語テキスト・単色画像・合成音を処理し、
有限な出力、公式encodeとの一致、複数質問の1回のforward、入力上限のエラーを確認します。
画像・音声の分類精度の評価ではありません。[実行結果](docs/validation.md)も参照してください。

参考: [公式モデルカード](https://huggingface.co/google/embeddinggemma-2)、
[公式開発者ガイド](https://developers.googleblog.com/en/embeddinggemma-2-the-developer-guide/)、
[MediaPipe Pythonガイド](https://developers.google.com/edge/mediapipe/solutions/decision/decision_maker/python)。
