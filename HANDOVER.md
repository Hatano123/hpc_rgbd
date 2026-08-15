# 引継ぎ資料 — YOLORGBD (SUN RGBD Top5 RGB-only)

このドキュメントはこのリポジトリの引継ぎを目的としています。重要なファイル、セットアップ、実行手順、トラブルシューティング、今後のTODOをまとめています。

## 概要
- **目的**: SUN RGBD Top5 データセットの RGB のみを用いた YOLO ベースの学習・評価を行うプロジェクト。
- **主な成果物**: 学習スクリプト、HPC 用ジョブスクリプト、データセットアーカイブ（zip）、学習出力（`runs/`）

＃＃＃最重要項目
qsubで投げてスパコンのGPUを使い訓練します。
また、これはRGBDの学習でRGBを超える精度を出すことを目標にしています。
オープンデータで学習


レポジトリ最重要ファイル
/home/0/y263357/YOLORGBD/train_hpc_top5.py　RGBD訓練スクリプト
/home/0/y263357/YOLORGBD/move_top5.sh　訓練スクリプト実行スクリプト


## リポジトリ主要ファイル
- **README（トレーニング用）**: [README_top5_rgb_training.md](README_top5_rgb_training.md)
- **メイン学習スクリプト**: [train_hpc_top5_rgb.py](train_hpc_top5_rgb.py)
- **HPC ジョブスクリプト**: [train_hpc_top5_rgb.sh](train_hpc_top5_rgb.sh)
- **ローカル実行スクリプト**: [run_top5_rgb_local.sh](run_top5_rgb_local.sh)
- **依存関係**: [requirements.txt](requirements.txt), [requirements_hpc.txt](requirements_hpc.txt)
- **仮想環境セットアップ**: [setup_pyenv_yolo_env.sh](setup_pyenv_yolo_env.sh)
- **データ（ローカル）**: `sunrgbd_top5_rgb.zip`（リポジトリルートに配置）




## 環境と依存関係
- Python 仮想環境: リポジトリ内の `yolo_env` を想定。`./setup_pyenv_yolo_env.sh` で作成。
- 依存パッケージ: `requirements.txt`（軽量）, `requirements_hpc.txt`（HPC 用追加）
- Torch / Ultralytics: HPC 実行時は永続領域にインストールせず、`TMPDIR`（一時領域）にインストールする仕組みを採用しています（`train_hpc_top5_rgb.sh` の方針）。

## データ準備
- リポジトリルートに `sunrgbd_top5_rgb.zip` を置くと、HPC スクリプトが TMPDIR に展開します。
- 展開後、スクリプトは `data.yaml` を検出してパスを書き換えます（`train_hpc_top5_rgb.sh` 内の処理）。

## 実行方法

**HPC（PBS）**

例: デフォルトで 100 エポック

qsub を用いてジョブを投げます。

```
EPOCHS=100 qsub train_hpc_top5_rgb.sh
```

環境変数で上書き可能: `EPOCHS`, `BATCH_SIZE`, `IMG_SIZE`, `DATA_YAML`, `DEVICE` など。

**ローカル実行**

```
./run_top5_rgb_local.sh
# または
python train_hpc_top5_rgb.py --data sunrgbd_top5_rgb/data.yaml --epochs 100
```

## 主要スクリプトの役割
- `train_hpc_top5_rgb.py`: Ultralytics YOLO を用いた学習処理。本プロジェクトでは 4ch の設定に揃えたハイパーパラメータや HPC 向けの安全策（augment get_indexes ガード）を含む。
- `train_hpc_top5_rgb.sh`: PBS 用バッチ。TMPDIR にデータと torch/ultralytics を展開して学習を実行する。ログは TMPDIR に出力。
- `run_top5_rgb_local.sh`: ローカルでの簡易実行用ラッパー。

## 出力と保存先
- 学習結果は `runs/` 以下に保存されます（`project`/`name` に応じてディレクトリが生成される）。

## よくある問題と対処
- zip が見つからない: リポジトリルートに `sunrgbd_top5_rgb.zip` があるか確認。
- `data.yaml` が見つからない: ZIP 展開後に `data.yaml` が存在するか、`train_hpc_top5_rgb.sh` のログ（TMPDIR）を確認。
- GPU メモリ不足: `BATCH_SIZE` を小さくする。
- pip install 周りのエラー: HPC 環境では SSL モジュールやネットワーク制限があるため、`yolo_env` の Python で `import ssl` が失敗しないことを確認。

## 運用上の注意（重要）
- 本リポジトリは「Torch を永続領域に置かない」運用方針になっています。`train_hpc_top5_rgb.sh` は `TMPDIR` に `torch` と `ultralytics` をインストールします。この慣習はユーザープリファレンスに基づいています。
- 一時領域のクリーンアップは `trap` で行われますが、ジョブが異常終了した場合に残るファイルを確認してください。

## 未解決のTODO / 引継ぎポイント
- `requirements_hpc.txt` の最小化とバージョン固定（再現性向上）
- テストデータでの簡易 CI スクリプト作成
- 実験ログのメタデータ（ハイパーパラや git commit）を自動で保存する仕組みの導入

## 連絡先 / 補足
- 質問や追加のドキュメントが要る場合はこのリポジトリの主要開発者に問い合わせてください（連絡先は別途共有）。

---

作成日: 2026-06-09
