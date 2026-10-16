# シンプルな4ch学習の検証

## カメラ仮入口の確認範囲

00_capture_rgbd.pyを追加しました。RealSense機種・SDK・接続機器が未確定のため、撮影・SDKの位置合わせ・ライブ表示は実機未確認です。

同梱TIFFの--replayで、RGB PNG・深度mm PNG・4ch TIFF・メタデータの保存を確認しました。既存の学習済み4chモデルを--modelで接続した推論と、検出画像・JSONの保存も確認済みです。単体テスト10件には、保存TIFFと共通前処理の整合、元の0値の保持、上書き防止、不正形状・NaNの拒否を含めています。

ファイル経路の確認はカメラ実機の動作保証ではありません。詳細はCAMERA.mdを参照してください。

## 同梱4chサンプルでの再現確認

samples/sunrgbd_top5へ学習8枚・検証4枚のTIFFとラベル、設定、画像SHA256一覧を追加しました。settings.yamlの初期値はこのサンプルです。

Gitのステージに入っているファイルだけを別フォルダへ展開し、外部データへのパスがない状態で01～04の4本を引数なしで順に実行しました。データ準備、2 epochの4ch学習、4枚の一括検出、評価がすべて成功しました。単体テストも8件成功しています。

依存Python環境を用意すれば、別のPCでも追加のデータ取得や入力パス変更なしで少量テストを始められます。学習済みのテスト用重みや実行結果はGitに含めていません。

## 番号付きPython入口の確認

2026-10-16、以下の4本をsettings.yamlの初期値のまま、引数なしで順番に実行しました。学習8枚・検証4枚、160px、batch=2、2 epochで全段階が完了しています。

```sh
python 01_prepare_data.py
python 02_train_model.py
python 03_detect_objects.py
python 04_evaluate_model.py
```

01の再実行では、設定・元画像・コピー画像・ラベルを確認して準備データを再利用できることも確認しました。単体テスト7件には、設定を変えた場合とコピー画像を変更した場合に再利用を拒否する確認を含めています。

出力先はwork/quickstart/data、runs/quickstart、work/quickstart/detections、work/quickstart/evaluationです。現在の重みはweights/yolo11n.ptを使います。以下の既存記録は、同じ内部処理の以前の実行結果です。

## 内部処理の確認記録

2026-10-16（日本時間）、Windows / Python 3.13.14 / PyTorch 2.11.0+cpu / Ultralytics 8.4.31 / OpenCV 4.13.0.92。

最終仕様は、人工的なゼロ化・欠落・欠損補完・非線形深度変換を使用しない4ch学習です。過去の途中テストと区別して、以下を再実行しました。

| 確認 | 結果 |
|---|---|
| SUN RGB-Dの少量入力 | 既存ローカルデータから学習8枚・検証4枚をコピー。元データは変更なし |
| 入力・整形の確認 | ラベル、形状、SHA256、クラス数、RGB＋正解枠と深度のプレビューを保存 |
| 4ch学習 | 160px、batch=2、2 epochが完了 |
| 深度への勾配 | 別の計測テストで全8バッチに有限・非ゼロの勾配 |
| 深度重みの更新 | 最大変化量 0.0001359954 |
| 入力形状 | 全8バッチとも[2,4,160,160] |
| モデル保存・再ロード | best.pt/last.ptの4chと前処理設定を確認 |
| 1枚の検出・フォルダ一括検出 | 成功。4枚分の画像・JSON・index.jsonを保存 |
| 評価 | 成功。metrics.json、クラス別SUMMARY.md、評価図を保存 |
| 単体テスト | 7件成功 |
| Python構文検査 | 新パッケージ・scripts・testsで成功 |

今回のdemoは検証4枚・35物体で、5クラスすべてに正解物体が含まれます。mAP50=0.000487、mAP50-95=0.000195でした。少量の短時間テストであり、精度の優位性を示す値ではありません。

## 再実行

```sh
python -m unittest discover -s tests -v
python -m rgbd_yolo demo --source /path/to/sunrgbd_top5 --output work/demo_02
python scripts/smoke_test.py --source /path/to/sunrgbd_top5 --output work/gradient_02
```

現在は同梱のsamples/sunrgbd_top5を初期入力にしています。demoのsource省略時も同梱サンプルを優先します。既存の出力を保護するため、demo/prepareでは空の出力先を使います。

## ローカル結果

- `work/sunrgbd_simple_demo/REPORT.md`: 一連の流れの結果
- `work/sunrgbd_simple_demo/dataset/`: 12枚の小さいデータセット
- `work/sunrgbd_simple_demo/evaluation/metrics.json`: 評価値
- `work/simple_smoke/evidence.json`: 深度勾配・重み更新の証拠

## 未確認

撮影機種、物理的な画素対応、保存倍率の元仕様、全データ・長時間学習・GPU/HPC・複数worker・分散学習・実機接続。自然に含まれる元の0値はそのままで、補完は行いません。
