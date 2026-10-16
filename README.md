# hpc_rgbd

標準YOLO11nの入力をRGB＋深度の4chへ拡張した物体検出です。
**普段使うのは4本のPythonファイルと、共通設定の `settings.yaml` だけです。**

## 準備

Python 3.13と、CPU/GPUに合うPyTorchを用意し、リポジトリのルートで実行します。

```sh
python -m pip install -r requirements.txt
```

**4chのSUN RGB-Dサンプル12枚を同梱しています。初期設定のまま試せます。**

```yaml
source_dataset: samples/sunrgbd_top5
```

自分のデータを使う場合だけ、settings.yamlのsource_datasetを変更してください。入力元には `images/train`、`images/val`、`labels/train`、`labels/val` が必要です。

## 実行方法

カメラ撮影の仮入口（RealSense想定・実機未確認）もあります。`python 00_capture_rgbd.py` で開き、Spaceで保存、Qで終了します。対応SDKが必要です。[カメラの説明](docs/CAMERA.md)

```sh
python 01_prepare_data.py     # 入力確認・データ整形
python 02_train_model.py      # 4ch学習
python 03_detect_objects.py   # 検出画像とJSONを保存
python 04_evaluate_model.py   # 正解と比較してAP・mAPを計算
```

初期設定は **学習8枚・検証4枚、160px、2 epoch、CPU** の動作確認用です。変更する場所は `settings.yaml` の画像枚数・`epochs`・`image_size`・`batch_size`・`device` です。ゼロ化や欠損補完は行いません。

## 結果の場所（初期設定）

| 段階 | 保存先 |
|---|---|
| 入力・整形 | `work/quickstart/data/`：RGB＋正解枠・深度のプレビューなど |
| 学習 | `runs/quickstart/weights/best.pt`：検出・評価に使うモデル |
| 検出 | `work/quickstart/detections/`：枠付き画像・JSON |
| 評価 | `work/quickstart/evaluation/`：`metrics.json`・クラス別結果 |

準備済みデータが同じなら01は再利用します。別の実験を残す場合は、settings.yamlの各保存先を変更してください。過去のコードは `legacy/`、詳細説明は `docs/` に分けています。

[詳しい実行手順](docs/WORKFLOW.md) · [YOLOの構造](docs/ARCHITECTURE.md) · [その他の機能](docs/FEATURES.md) · [引継ぎ](HANDOVER.md)
