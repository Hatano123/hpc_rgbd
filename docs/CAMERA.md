# 00: カメラ撮影の仮入口

`00_capture_rgbd.py` は **RealSenseを想定した仮実装** です。使用機種は未確定で、実機では確認していません。現在の01～04はそのまま使えます。ifmなど別機種の取得処理は、この00には接続していません。

## カメラで撮影

RealSense本体と対応SDK、Pythonパッケージの `pyrealsense2` が必要です。利用するPython版・OSで対応するSDK環境を用意してください。リポジトリの確認環境であるPython 3.13でのSDK動作は未確認です。通常の学習にこのパッケージは不要です。

```sh
python 00_capture_rgbd.py
```

RGBと深度のプレビューを表示します。**Spaceで1枚保存、QまたはEscで終了**します。既定は640×480・30fpsのストリーム要求で、対応しない機種では設定変更が必要です。これは検出処理の実測FPSではありません。

RGBと深度はSDKのframesetから取得し、`rs.align(rs.stream.color)` で深度をカラー画像へ位置合わせします。SDKのdepth scaleで生の深度値をmmへ換算します。取得と位置合わせのAPIは [RealSense公式例](https://github.com/realsenseai/librealsense/blob/master/wrappers/python/examples/align-depth2color.py) を参照しています。背景除去や穴埋めは実装していません。

## 学習済みモデルでライブ検出

```sh
python 00_capture_rgbd.py --model runs/quickstart/weights/best.pt
```

4chモデルの前処理設定を読み戻し、共通の推論関数へ渡します。Spaceで撮影すると、検出結果の画像・JSONも保存します。02でサンプルを2 epoch学習しただけのモデルでは、実環境の精度は期待できません。

画面なしで指定枚数を保存する場合:

```sh
python 00_capture_rgbd.py --frames 10 --no-preview
```

## 保存内容

既定では `work/captures/日時/` に保存します。`--output フォルダ` でも指定できます。既存ファイルを上書きしないよう、空のフォルダを使用します。

| フォルダ・ファイル | 内容 |
|---|---|
| rgb/*.png | BGRのuint8画像を通常のカラーPNGとして保存 |
| depth_mm/*.png | mm単位に丸めたuint16深度。PNGの上限は65535mm |
| rgbd/*.tiff | BGR×256、深度mm×8の4ch uint16。深度は学習側に合わせ8000mmでクリップ |
| metadata/*.json | 取得時刻、センサの深度単位、カラーの内部パラメータなど |
| preprocess.yaml | 保存TIFFを読むための倍率と深度上限 |
| detections/ | --modelを指定した場合、保存フレームの検出画像・JSON |

元の深度0値は保持します。人工的にRGB/Depthをゼロにする処理、欠損補完、新しい校正の推定は行いません。SDKが提供する位置合わせを利用します。

**撮影画像には正解ラベルがありません。** 自分の撮影データで学習・評価するには、物体の枠を付けてYOLOラベルを作り、images/train・images/valと同名のlabels/train・labels/valへ分割してから、settings.yamlのsource_datasetを指定します。撮影しただけでは01～04用のラベル付きデータにはなりません。

保存済みTIFFを検出するだけなら、正解ラベルは不要です。

```sh
python -m rgbd_yolo predict --model runs/quickstart/weights/best.pt --images work/captures/撮影日時/rgbd --output work/camera_predictions
```

## カメラなしの確認

```sh
python 00_capture_rgbd.py --replay samples/sunrgbd_top5/images/val/sunrgbd_val_00223.tiff
```

`--replay` は上記の保存形式の4ch TIFFを1枚読み、同じ保存経路を通して終了します。GUIやRealSense SDKは使いません。ファイル名は同梱サンプルの実際の名前に合わせてください。保存形式の往復と既存モデルへの入力を確認するためのもので、カメラやSDKの動作確認ではありません。
