# 言語指示で動く片付けロボット（Physical AI 応用1講座 最終課題）

自然言語の指示（日本語・英語）を受けて、カメラ画像から **AI（OWLv2：オープン語彙物体検出）** で物体を見つけ、
**深度画像による3D幾何検証**で認識を補正し、**Franka Emika Panda** アームで物体をトレイへ仕分ける
MuJoCo シミュレーションです。

```
「ボールは全部青いトレイに入れて、黄色いキューブは緑の箱へ。それから残りを色ごとに片付けて」
        │ 言語理解（ルールベース / 任意で LLM）
        ▼
 実行計画: [ボール→青] [黄色いキューブ→緑] [すべて→同じ色]
        │
        ▼  ┌───────────────── 閉ループ（1手ごとに再観測） ─────────────────┐
 観測姿勢 → 俯瞰RGB-D撮影 → OWLv2推論 → 深度で3D復元・形状/色/向きを検証
        → 対象と行き先を決定 → IK + 重力補償で把持 → 把持判定 → 運搬・配置 → 再観測で検証
```

## 主な特徴

| 要素 | 内容 |
| --- | --- |
| AI（推論） | `google/owlv2-base-patch16-ensemble` によるテキスト指定の物体検出（CPU で約2秒/枚） |
| 3D検証 | 深度画像を3D点群に逆投影。高さ・上面の平坦さ・外形で形状を、色相ヒストグラムで色を判定し、OWLv2 のラベル誤りを補正。最小外接矩形でキューブの向きを推定 |
| 言語理解 | 日本語/英語の複合指示を順序付きの計画に分解（「全部」「残り」「色ごと」「〜と〜」等）。`OPENAI_API_KEY` があれば LLM パーサを使用し、失敗時はルールに自動フォールバック |
| 制御 | 減衰最小二乗法 IK（ヌル空間で姿勢維持）＋ min-jerk 軌道＋重力補償フィードフォワード（手先誤差 約6mm → 約1mm） |
| ロバスト性 | 指の開き幅で把持失敗・落下を検知 → 再観測してリトライ。毎手再観測するので人が物体を動かしても追従。行き先のトレイが無い物体は「動かさない」と判断 |

## 定量評価（`python -m pai_tidy.evaluate`）

ランダム配置30シーン（物体6個＋トレイ3個 × 30 = 270 個）での認識精度:

| 指標 | OWLv2 のみ | OWLv2 + 3D検証（本手法） |
| --- | --- | --- |
| ラベル（色＋形状）正解率 | 35.9 % | **100 %** |
| 物体位置誤差（平均 / 最大） | – | 0.47 mm / 0.94 mm |
| キューブ向き誤差（平均） | – | 0.51° |

ランダム配置12エピソード（物体5個、指示3種類）でのタスク成功率: 物体単位 **100 %（60/60）**、全物体が正しく片付いたエピソード **12/12**。
提出動画の2シナリオも 10/10 物体で成功（把持失敗の故障注入1回から自動復帰）。結果は `outputs/eval.json`, `outputs/results.json` に保存されます。

## 動かし方

動作確認環境: Ubuntu 24.04, Python 3.12, CPU のみ（GPU 不要）。

```bash
# 依存関係（GPU なしの場合は CPU 版 PyTorch）
python3 -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
# 画面の無い環境でのオフスクリーン描画と日本語フォント
sudo apt-get install -y libosmesa6 fonts-noto-cjk

# 提出動画（2シナリオ）を outputs/demo.mp4 に生成（CPU 4コアで約15分）
python -m pai_tidy.run --demo

# 任意の指示をランダムなシーンで実行（動画も保存する場合は --video）
python -m pai_tidy.run -i "赤いものは青いトレイに、残りは色ごとに片付けて" --seed 3 -n 5 --video outputs/custom.mp4

# 定量評価
python -m pai_tidy.evaluate --perception 30 --tasks 9
```

初回実行時に OWLv2 の重み（約 600MB）を Hugging Face から自動ダウンロードします。
`MUJOCO_GL` は既定で `osmesa`（ソフトウェア描画）です。GPU 付きのマシンでは `MUJOCO_GL=egl` で高速化できます。

## ディレクトリ構成

```
pai_tidy/
  scene.py       シーン生成（机・物体・トレイ・俯瞰カメラ）
  perception.py  OWLv2 推論 + 深度による3D検証（形状・色・位置・向き）
  language.py    自然言語 → 実行計画（ルール / LLM）
  robot.py       IK・軌道生成・重力補償・グリッパ
  agent.py       閉ループの片付けエージェント（観測→把持→配置→検証→リトライ）
  video.py       動画の合成（字幕・計画・AI知覚ワイプ・タイトルカード）
  run.py         デモ動画生成 / 任意指示の実行
  evaluate.py    定量評価
assets/franka_emika_panda/  MuJoCo Menagerie の Panda モデル（Apache-2.0）
report/          レポート（HTML テンプレート → PDF）
```

## クレジット

- ロボットモデル: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) `franka_emika_panda`（Apache License 2.0、`assets/franka_emika_panda/LICENSE`）
- 物体検出: [OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble)（Minderer et al., 2023, Apache License 2.0）
- 物理シミュレータ: [MuJoCo](https://mujoco.org/)
