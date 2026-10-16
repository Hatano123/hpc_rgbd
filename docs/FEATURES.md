# 残している機能

| 機能 | 現在の使い方 |
|---|---|
| 普通のRGB-D 4ch学習 | `python 02_train_model.py`、YOLO11n、前処理共通化 |
| RGBのみの比較学習 | `train --mode rgb`、専用の3ch画像データを使用 |
| 少量データでの一通り確認 | `demo` |
| データの部分抽出・チェック | `python 01_prepare_data.py`、枚数・seed指定、入力元は変更しない |
| RGB・深度・正解枠の表示 | prepareでpreview.jpgを保存 |
| 画像1枚／フォルダ一括検出 | `python 03_detect_objects.py`。個別指定は `predict --image`／`--images` |
| RGB画像＋深度PNG入力 | `predict --rgb ... --depth ...`。深度はmm |
| クラス別評価・JSON保存 | `python 04_evaluate_model.py`。個別指定は `val --output` |
| 学習済み重みと前処理設定の保存 | best.pt、last.pt、manifest.json |
| 深度にも学習が行われるかの確認 | scripts/smoke_test.pyで勾配・重み更新を計測 |
| 学習ログ | train --log-file |
| HPC/PBS | scripts/hpc/train.pbs |
| 旧ToF撮影・校正・Web表示・実験コード | legacyに原本保存。現在の経路とは独立、実機動作は未検証 |

ユーザーの指定により、RGB/Depthを意図的にゼロにする機能、欠落条件を作る処理、欠損補完、非線形の深度変換は現在のコードから外しました。旧コードの歴史的な記録はlegacyにありますが、現在のCLIからは実行しません。
