# 引継ぎ資料 — hpc_rgbd

更新日: 2026-10-16（日本時間）

作業先: `C:/Users/chibi/programming/git/hpc_rgbd`

リポジトリ: https://github.com/Hatano123/hpc_rgbd

基点コミット: `7c9b7e5b0da0579f09aa35556dc8598dd0262904`

## 現在の仕様

ユーザーの最終指定は「特殊なゼロ化・欠落・欠損処理をせず、普通に4ch学習できること」です。RGBと深度は画素対応済みを前提に、保存値の単位変換と線形正規化だけを行います。元データの0値は補完しません。機種は不明で、RealSenseと断定していません。

処理順とコマンドはdocs/WORKFLOW.md、残した便利な機能はdocs/FEATURES.mdにまとめています。

YOLO11nのアーキテクチャ、4ch入力への変更、事前学習重みの初期化、出力の意味はdocs/ARCHITECTURE.mdを参照してください。

## まず動かす

```powershell
Set-Location C:/Users/chibi/programming/git/hpc_rgbd
$env:PYTHONUTF8='1'
python -m rgbd_yolo demo --output work/my_simple_demo
```

このPCでは既存の `C:/Users/chibi/programming/git/YOLORGBD/datasets/sunrgbd_top5` から学習8枚と検証4枚をコピーします。入力元は変更しません。別のPCは `--source` が必要です。Gitには新しいサンプル画像・学習重みを追加しません。

結果は指定先のREPORT.mdです。学習用の小さいデータ、RGB＋正解と深度のプレビュー、整形後の配列例、モデル、検出画像、評価JSONを段階ごとに保存します。

## データ仕様

- 4ch TIFF: OpenCVでB,G,R,Dとして読み込み。今回のローカル形式はRGB保存値÷256、深度保存値÷8でmmへ変換。
- 深度上限8000mmでクリップし、線形に正規化。対数・平方根変換はありません。
- RGB＋深度PNGのペア: RGBは8bit強度、深度はmm。TIFF用の倍率は適用しません。
- 保存形式が違う場合はconfigs/のdivisorを合わせます。ToFの旧生成コードにはRGB×257・深度mmという別形式もあるので、既定設定をそのまま適用しないでください。
- チャンネル数・形状・ラベル範囲を検証しますが、実際の画素対応は元の生成仕様と境界の重ね合わせで別途確認が必要です。

## モジュール

| ファイル | 責務 |
|---|---|
| preprocessing.py | 単位換算・線形正規化 |
| dataset.py | 4ch画像とラベルを学習用に読み込む |
| training.py | 4chモデル構築・重み移植・検証 |
| inference.py | Tensor入力と元画像座標での検出結果 |
| evaluation.py | 評価指標とクラス別結果の保存 |
| pipeline.py | 部分データ準備、プレビュー、一連のデモ |
| cli.py | prepare/train/predict/val/demo/inspectの入口 |

新しい重みに前処理設定を保存し、検出と評価で読み戻します。グローバルなクラスの書き換えはしません。

## 旧コードと変更点

元の77ファイルはlegacyへ無改変で保存し、元チェックアウトとの一致を確認しました。全件対応表はdocs/FILE_MAP.mdです。主要な旧学習ファイル名はCLIラッパーとして残しています。関数のimport互換性や全旧引数は維持していません。

以前の途中テストは対数変換・max_fillなどを使っていましたが、現在のコードでは使用しません。過去のテスト用重み・設定は今回の簡略化版の実験結果と混ぜず、新しいdemoで再学習してください。

## HPC

```sh
qsub -v DATA_YAML=/path/data.yaml,PYTHON_CMD=/path/to/prepared/python,DEVICE=0 scripts/hpc/train.pbs
```

Torchを永続領域へ置かず、既存のTMPDIR運用を継続します。PBSスクリプトではネットワークインストールやZIP展開を自動実行しません。データとホストに合うPyTorchを事前準備してください。HPC/実機では今回テストしていません。

## 確認と次の作業

ローカルCPUでの4ch学習、深度への勾配・重み更新、チェックポイントの保存・再ロード、検出、評価を確認します。実行記録はdocs/VALIDATION.mdへ記載します。少量の動作確認なので、RGBより精度が高いとは判断しません。

次は保存倍率・深度単位・実際の画素対応を確定し、その後に全データ・通常解像度でRGBとRGB-Dを同じ条件で比較します。GPU/HPC、複数worker、分散学習、センサ接続は別途確認が必要です。
