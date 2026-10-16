# 移行メモ

通常の入口は番号付きの4つのPythonファイルです。内部CLIの `python -m rgbd_yolo` も引き続き使用できます。手順はWORKFLOW.mdを参照してください。

| 旧入口 | 現在の入口 |
|---|---|
| train_hpc_top5.py / train_hpc.py | 02_train_model.py（旧ラッパーはlegacy/compatibility） |
| train_hpc_top5_rgb.py / train_hpc_3ch.py | train --mode rgb |
| folder_detect.py | 03_detect_objects.py。またはpredict。--imageまたは--imagesと--model/--outputを明示 |
| Modelmap.py | 04_evaluate_model.py。またはval --output |
| move_top5.sh | scripts/hpc/train.pbsへのラッパー |
| その他の旧コード | legacy/。原本はFILE_MAP.mdの場所に保存 |

## 現在の単純な4ch経路

- RGB保存倍率、深度保存倍率、深度上限の3項目だけを前処理設定として持ちます。
- RGB/Depthのランダムゼロ化、欠落条件、欠損補完、非線形深度変換はありません。
- 不足チャンネルを自動生成せず、形式不正やNaN/Infは例外にします。
- RGB/RGB-Dとも既定をYOLO11n、optimizerをAdamWに統一。旧optimizer=autoとは学習結果が一致するとは限りません。
- グローバルmonkey patchと、epochを共通ファイルへ書く仕組みはありません。
- 新しい重みには前処理設定を保存。以前の途中テストの特殊設定付き重みは、新経路で再学習してください。

旧コードの関数import互換性や全CLI引数は提供していません。撮影・校正・Webサーバなどは履歴参照用の原本を保持していますが、新しい共通経路への実機移植はしていません。

追加のファイル配置変更は[FILE_LAYOUT.md](FILE_LAYOUT.md)を参照してください。
