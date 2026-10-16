# シンプルな4ch学習の検証

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

このPCではdemoのsource省略時に隣のYOLORGBD/datasets/sunrgbd_top5を探します。他の環境はsourceを明示してください。既存の出力を保護するため、demo/prepareでは空の出力先を使います。

## ローカル結果

- `work/sunrgbd_simple_demo/REPORT.md`: 一連の流れの結果
- `work/sunrgbd_simple_demo/dataset/`: 12枚の小さいデータセット
- `work/sunrgbd_simple_demo/evaluation/metrics.json`: 評価値
- `work/simple_smoke/evidence.json`: 深度勾配・重み更新の証拠

## 未確認

撮影機種、物理的な画素対応、保存倍率の元仕様、全データ・長時間学習・GPU/HPC・複数worker・分散学習・実機接続。自然に含まれる元の0値はそのままで、補完は行いません。
