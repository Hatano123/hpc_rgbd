# 引継ぎ資料 (YOLORGBD)

## 目的
RGB+Depth 4ch でのデータ取得・キャリブレーション・データ生成・学習・評価・推論までの手順を一枚にまとめる。上級者向けの簡潔版。

## 環境
- OS: Windows 前提 (ifm3dpy が Windows 想定)
- Python: 3.10–3.11 目安
- 主要パッケージ: ultralytics, opencv-python, numpy, scipy, pyyaml, ifm3dpy
- 推奨インストール例
```bash
pip install ultralytics opencv-python numpy scipy pyyaml ifm3dpy
```
- ToF 接続に ifm3dpy が必須。インストール後、PC と ToF を同一セグメントに置く。

## 重要ファイル
- 取得: [collect_ifm3d.py](../collect_ifm3d.py)
- キャリブレーション: [CalibrationTOF-RGB.py](../CalibrationTOF-RGB.py)
- データ生成: [MakeDatasetHokan.py](../MakeDatasetHokan.py)
- 学習: [train_4ch.py](../train_4ch.py), 拡張版 [train_4ch_Mosaic.py](../train_4ch_Mosaic.py)
- 評価: [Modelmap.py](../Modelmap.py)
- 推論例: [realtime_rgbd_detect10FPS.py](../realtime_rgbd_detect10FPS.py)
- キャリブ成果物: camera_calibration.mat, tof_calibration.mat, extrinsics.npz
- データ記録先: raw_train_dataset_1-15/ (撮影), datasets16.01-11/rgbd_aligned/ (最終学習用 TIFF+labels)

## フォルダ構成 (主要)
```
YOLORGBD/
  docs/
    handover.md
  AnnotationData/
    dataset.yaml
    train/
    valid/
    test/
    README.dataset.txt
  datasets/
    datasets16.01-11/
      rgbd_aligned/
        images/
        labels/
    led_rgb_only/
    dark_rgb_only/
  raw_train_dataset_1-15/
    rgb/
    tof/
  runs/                  # 学習・推論の出力 (Ultralytics 標準)
  camera_calibration.mat # RGB 内部パラメータ
  tof_calibration.mat    # ToF 内部パラメータ
  extrinsics.npz         # 外部パラメータ (R, T)
  train_4ch.py 等スクリプト群
```

## ワークフロー
1) 露光設定: ifmVisionAssistant で ToF の露光時間・回数を事前設定。
2) データ撮影 (手動):
```bash
python collect_ifm3d.py --amp-threshold 0.05 --app2-exposure-us 1000
# Space: 撮影, q: 終了
```
- 出力: raw_train_dataset_1-15/rgb/*.png と tof/*.npz
3) キャリブレーション:
```bash
python CalibrationTOF-RGB.py
# 起動後 モード1: 収集→外部パラメータ計算→投影確認
```
- 必要: camera_calibration.mat (内部パラメータ)。RGB カメラ交換/レンズ変更/解像度変更時は必ず再計測し更新。
- 出力: extrinsics.npz (R, T)。
4) データ生成 (RGB+Depth 4ch TIFF 作成):
```bash
python MakeDatasetHokan.py
```
- 入力: raw_dataset_merged/tof (npz) と CUP11-14/{train,valid,test}
- 出力: datasets16.01-11/rgbd_aligned/{images,labels}
5) 学習 (4ch):
```bash
python train_4ch.py
# data=datasets16.01-11/rgbd_aligned/dataset.yaml, weights=yolov8n.pt
```
- データ拡張: 水平反転確定、RGB ドロップ確率 0.3。深度は 0–5000mm を正規化。
- Mosaic 版を使う場合は train_4ch_Mosaic.py。
6) 評価 / mAP 比較:
```bash
python Modelmap.py --rgbd-model runs/detect/<exp>/weights/best.pt \
  --data datasets16.01-11/rgbd_aligned/dataset.yaml --imgsz 640 --batch 4 --split val
```
7) 実機推論 (例):
```bash
python realtime_rgbd_detect10FPS.py --model runs/detect/<exp>/weights/best.pt \
  --tof-ip 192.168.1.69 --webcam-index 1 --show-depth
```

## データ構造メモ
- raw_train_dataset_1-15/rgb | tof: 撮影直後 (png + npz)
- datasets16.01-11/rgbd_aligned/images: 16bit 4ch TIFF (RGB+Depth 正規化前 0–65535, depth mm)
- datasets16.01-11/rgbd_aligned/labels: YOLO txt
- dataset.yaml: クラスや分割定義。4ch 用は datasets16.01-11/rgbd_aligned/dataset.yaml を指す。

## カメラ変更時の再キャリブ手順
1) 新 RGB カメラで内部キャリブ → camera_calibration.mat 更新。
2) CalibrationTOF-RGB.py モード1で外部パラメータ再計算 → extrinsics.npz 更新。
3) 推論・学習・評価すべてで新しい mat/npz を使用。

## トラブルシュート
- ifm3dpy ImportError: `pip install ifm3dpy` を実行し、Windows の PATH に ifm3d SDK が入っているか確認。
- ToF 取得がタイムアウトする: ネットワーク疎通 (ping)、FG_WAIT_TIMEOUT_MS を長めに設定。
- 学習時にラベル欠損警告: datasets16.01-11/rgbd_aligned/labels 内の欠損ファイルを確認し、必要なら空ファイル追加。
- 推論で深度がゼロになる: extrinsics.npz の整合性、Exposure 設定、Depth 正規化上限 (train_4ch.py の MAX_DEPTH_VAL=5000) を確認。

## 今後の改善メモ
- requirements.txt の固定化 (ultralytics/ifm3dpy バージョン固定)。
- 3ch ベースライン評価の安定化 (README 記載の課題)。
- 推論スクリプトで GUI UI を簡素化し、設定を YAML 化。
