# ファイル配置の整理

通常の操作で見る入口は、ルートの4つのPythonファイルとsettings.yamlに絞りました。計算処理はrgbd_yolo内で共通化し、入口ごとに学習ロジックを複製していません。

撮影用の仮入口として00_capture_rgbd.pyを追加しています。RealSense向けで実機未確認のため、詳細と制約はCAMERA.mdを参照してください。

| 以前の場所 | 現在の場所・代わりの入口 |
|---|---|
| train_hpc.py、train_hpc_top5.py | 02_train_model.py。旧ラッパーはlegacy/compatibility/ |
| train_hpc_3ch.py、train_hpc_top5_rgb.py | RGB比較は `python -m rgbd_yolo train --mode rgb`。旧ラッパーはlegacy/compatibility/ |
| folder_detect.py | 03_detect_objects.py。旧ラッパーはlegacy/compatibility/ |
| move_top5.sh | HPCの実行入口はscripts/hpc/train.pbs。旧ラッパーはlegacy/compatibility/ |
| requirements_hpc.txt | 共通requirements.txtを使用。旧ファイルはlegacy/compatibility/ |
| requirements-tested-cpu.txt | requirements/cpu.txt |
| yolo11n.pt | weights/yolo11n.pt（現在の初期重み） |
| yolov8n.pt、yolo26n.pt | legacy/pretrained/（過去のモデル） |
| sunrgbd_top5/data.yaml | 旧設定はlegacy/dataset_configs/sunrgbd_top5.yaml。クラス定義はconfigs/class_names.yaml |
| sunrgbd_top5_rgb/ | datasets/rgb_reference/。元のRGB比較用データを保持 |

旧互換ラッパーは参照用です。移動先からの直接実行は想定せず、新しい入口を使ってください。

元の77ファイルの保存先はFILE_MAP.mdにあります。legacy内の元コード・モデルは履歴資料で、現在の番号付きPython実行経路では読みません。

注意: datasets/rgb_referenceのクラス順はchair、picture、cabinet、books、tableです。4ch側のchair、cabinet、picture、books、tableとは異なるので、比較時はクラス名で対応付けてください。ラベルの番号を無条件に同一視しないでください。

RGB比較用データのdata.yamlは、新しいフォルダを基準に相対パスで解決するよう更新しました。元の固定パス付き設定はlegacy/dataset_configs/rgb_reference_original.yamlに保存しています。
