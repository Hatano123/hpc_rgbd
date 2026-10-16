# RGB-Only Training for SUN RGBD Top5 Dataset

このディレクトリには、`sunrgbd_top5_rgb.zip` に対してRGBのみで学習するためのプログラムとシェルスクリプトが含まれています。

## ファイル一覧

### 学習プログラム
- **`train_hpc_top5_rgb.py`** - RGB のみで学習するメイン学習プログラム
  - YOLO11n モデルをサポート
  - カスタマイズ可能なエポック数、バッチサイズなど

### HPC 用シェルスクリプト
- **`run_top5_rgb.sh`** - PBS ジョブスケジューラー対応（HPC環境用）
  - 自動的に `sunrgbd_top5_rgb.zip` を抽出
  - TMPDIR に torch をインストール（ユーザー設定に従う）

### ローカル実行用スクリプト
- **`run_top5_rgb_local.sh`** - ローカルマシン用
  - PBS ディレクティブなし
  - 開発やテスト用

### データセット設定
- **`sunrgbd_top5_rgb/data.yaml`** - RGB のみ (channels: 3) に対応した設定

## 使用方法

### 1. HPC での実行（PBS ジョブスケジューラー）

```bash
# デフォルト設定で実行（100 エポック）
qsub run_top5_rgb.sh

# 環境変数でカスタマイズ
EPOCHS=200 BATCH_SIZE=32 IMG_SIZE=640 qsub run_top5_rgb.sh

# 特定のデータセットを指定
DATA_YAML=/path/to/data.yaml EPOCHS=150 qsub run_top5_rgb.sh
```

### 2. ローカルマシンでの実行

```bash
# デフォルト設定で実行
./run_top5_rgb_local.sh

# 環境変数でカスタマイズ
EPOCHS=200 BATCH_SIZE=16 AUGMENT=1 ./run_top5_rgb_local.sh
```

### 3. 直接 Python で実行

```bash
# 基本的な実行
python train_hpc_top5_rgb.py --data sunrgbd_top5_rgb/data.yaml --epochs 100

# 詳細なオプション
python train_hpc_top5_rgb.py \
    --data sunrgbd_top5_rgb/data.yaml \
    --epochs 200 \
    --batch 32 \
    --imgsz 640 \
    --workers 4 \
    --name my_custom_run \
    --device 0 \
    --augment

# オプション一覧を表示
python train_hpc_top5_rgb.py --help
```

## 環境変数

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `DATA_YAML` | `sunrgbd_top5_rgb/data.yaml` | データセット設定ファイルのパス |
| `EPOCHS` | `100` | 学習エポック数 |
| `BATCH_SIZE` | `16` | バッチサイズ |
| `IMG_SIZE` | `640` | 入力画像サイズ |
| `WORKERS` | `4` | データローダーのワーカー数 |
| `RUN_NAME` | `hpc_sunrgbd_top5_rgb_e{EPOCHS}` | 実行の名前 |
| `DEVICE` | `0` | GPU デバイスID（0=GPU、cpu=CPU） |
| `AUGMENT` | `0` | データ拡張を有効化（1=有効） |

## データセット構造

`sunrgbd_top5_rgb.zip` は以下の構造を想定しています：

```
sunrgbd_top5_rgb/
├── images/
│   ├── train/
│   │   ├── 0001.png
│   │   ├── 0002.png
│   │   └── ...
│   ├── val/
│   │   ├── 0101.png
│   │   └── ...
│   └── test/
│       └── ...
├── labels/
│   ├── train/
│   │   ├── 0001.txt
│   │   ├── 0002.txt
│   │   └── ...
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
└── data.yaml (このファイルで上書きされます)
```

**重要**: このスクリプトは `sunrgbd_top5_rgb.zip` から RGB のみのデータを想定しています。ファイルが `channels: 4` (RGBD) の場合は、先に `train_hpc.py` を使用してください。

## 出力

学習結果は `runs/` ディレクトリに保存されます：

```
runs/
└── hpc_sunrgbd_top5_rgb_e100/
    ├── weights/
    │   ├── best.pt
    │   └── last.pt
    ├── results.csv
    ├── confusion_matrix.png
    └── ...
```

## トラブルシューティング

### `sunrgbd_top5_rgb.zip が見つからない`
→ ワークスペースのルートディレクトリに ZIP ファイルが存在することを確認してください

### `data.yaml が見つからない`
→ 自動抽出後に `sunrgbd_top5_rgb/images/` ディレクトリが作成されていることを確認してください

### GPU メモリ不足エラー
→ `BATCH_SIZE` を小さくしてください：
```bash
BATCH_SIZE=8 qsub run_top5_rgb.sh
```

### Python モジュールが見つからない
→ 仮想環境が正しくアクティベートされていることを確認：
```bash
source yolo_env/bin/activate
python -c "import torch, ultralytics"
```

## 参考資料

- 既存の 3ch 学習プログラム: `train_hpc_3ch.py`
- 既存の RGBD 学習プログラム: `train_hpc.py`
- HPC 用 3ch スクリプト: `run_3ch.sh`
