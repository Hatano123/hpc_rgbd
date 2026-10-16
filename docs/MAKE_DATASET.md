# 学習・評価用データの作り方

**撮影 → RGB画像に枠を付ける → 画像とラベルを学習・検証へ分ける → settings.yamlを変更する**、の順で準備します。

同梱のサンプルを試すだけなら作成作業は不要です。以下は自分で撮影したデータを使う場合の手順です。

## 1. RGBと深度を撮影する

RealSense向けの仮入口:

```sh
python 00_capture_rgbd.py --output work/captures/session_01
```

Spaceで撮影、Qで終了します。対応SDKが必要で、実機では未確認です。[カメラの説明](CAMERA.md)を参照してください。

```text
work/captures/session_01/
  rgb/frame_000000.png        ラベル付けに使うRGB画像
  depth_mm/frame_000000.png   mm単位の深度
  rgbd/frame_000000.tiff      学習に使う4ch画像
  metadata/frame_000000.json  取得情報
  preprocess.yaml            保存倍率と深度上限
```

同じ番号のRGBと深度は同じ画素位置に対応している必要があります。00ではSDKで深度をカラー画像へ位置合わせします。別のカメラのデータを使う場合も、学習前に位置合わせを済ませてください。同じ縦横サイズというだけでは、画素対応の確認にはなりません。

### 位置合わせ済みRGB PNG・深度PNGがすでにある場合

深度が**mm単位**であることを確認してから、次のように既存の保存関数で4ch TIFFへ変換できます。この処理にカメラSDKは不要です。

```python
import cv2
from rgbd_yolo.camera import save_capture

rgb = cv2.imread("work/input/rgb.png")
depth_mm = cv2.imread("work/input/depth_mm.png", cv2.IMREAD_UNCHANGED)
if rgb is None or depth_mm is None:
    raise FileNotFoundError("RGBまたは深度画像が読み込めません")

save_capture(
    "work/captures/imported", 0, rgb, depth_mm,
    {"source": "aligned RGB and millimetre depth"},
)
```

結果は `work/captures/imported/rgbd/frame_000000.tiff` です。複数画像の場合はindexの0を画像ごとに変えます。深度がm単位なら先に×1000でmmへ変換します。深度を見やすく色付けした画像は、深度値の入力には使用しません。

## 2. RGB画像に物体の枠を付ける

`rgb/*.png` の元画像をアノテーションツールで開き、物体の矩形とクラスを付けて、YOLOの物体検出形式のtxtとして書き出します。4ch TIFFや、RGBと深度を横に並べたプレビューへ直接枠を付ける必要はありません。

ラベルは**元のRGB画像と同じ画素座標**に対応させます。画像の切り抜き・回転などをする場合、深度とラベルにも同じ幾何変換が必要です。

例として、クラスを次の2種類に決めます。

```text
0: chair
1: table
```

`frame_000000.png` のラベルは `frame_000000.txt` です。1物体につき1行書きます。

```text
class_id x_center y_center width height
```

座標4値は画像の幅・高さで割った0～1の値です。

```text
x_center = (x1 + x2) / (2 × 画像幅)
y_center = (y1 + y2) / (2 × 画像高さ)
width    = (x2 - x1) / 画像幅
height   = (y2 - y1) / 画像高さ
```

例: 640×480の画像で、chairの枠が左上(100,120)、右下(300,360)なら:

```text
0 0.3125 0.5 0.3125 0.5
```

対象物が本当にない画像には空のtxtを用意します。まだラベル付けしていない画像を、対象物がない画像として扱わないでください。現在の入力確認では、対応するtxtがない画像はエラーにします。

## 3. 学習用と検証用に分ける

例として100枚を学習80枚・検証20枚へ分けます。撮影回・場所・シーンで分け、ほぼ同じ連続フレームが学習と検証の両方に入らないようにします。

各RGB画像に対応する **4ch TIFFとラベルtxtをセット** で配置します。

```text
work/my_dataset/
  images/
    train/
      session01_000000.tiff
      session01_000001.tiff
    val/
      session02_000000.tiff
  labels/
    train/
      session01_000000.txt
      session01_000001.txt
    val/
      session02_000000.txt
  data.yaml
```

複数の撮影回では同じframe番号があるため、session名などを付けて区別します。リネームするときはTIFFと対応するtxtを同じ名前にしてください。元のRGB PNG・深度PNG・取得情報も、撮影記録として残しておきます。

`data.yaml` の例:

```yaml
train: images/train
val: images/val
nc: 2
names: [chair, table]
channels: 4
```

クラス番号は0から始め、namesの順番とラベルの番号を一致させます。この例のtableは1です。同梱のTop5とはクラス構成が違いますが、モデルの出力クラス数はこのYAMLに合わせます。

絶対パスの `path:` は不要です。imagesとlabelsはこのdata.yamlと同じフォルダを基準に配置します。source_dataset内にdata.yamlがある場合、そのnamesがクラス定義として優先されます。

## 4. 保存形式と設定を合わせる

00の保存関数が作るTIFFは次の形式です。

| チャンネル | 保存値 | 読み戻し |
|---|---|---|
| B,G,R | 8bit強度×256をuint16で保存 | ÷256 |
| D | 深度mm×8をuint16で保存。上限8000mm | ÷8 |

この形式には `configs/sunrgbd_tiff.yaml` を使います。単にuint16へ型変換しただけでは、この保存倍率にはなりません。RGB・深度がすでにそのままの強度値・mm値で入った4ch TIFFなら、`configs/aligned_mm_tiff.yaml` を使います。

`settings.yaml` の例（上で用意した100枚をすべて使用する場合）:

```yaml
source_dataset: work/my_dataset
data_directory: work/my_experiment/data
preprocessing: configs/sunrgbd_tiff.yaml
train_count: 80
val_count: 20
training_directory: runs/my_experiment
detection_directory: work/my_experiment/detections
evaluation_directory: work/my_experiment/evaluation
```

これは変更する項目の抜粋です。その他の設定は既存のsettings.yamlに残してください。train_countとval_countは**使用する枚数**で、実際に用意した各分割の枚数以下にします。初期値の8・4のままなら、その枚数だけ抽出します。

このデータ作成経路では、人工的な欠落や欠損補完は行いません。元データの深度0値はそのままです。

## 5. 準備データを確認して学習する

```sh
python 01_prepare_data.py
```

まず出力先のpreview.jpgを見て、RGB上の正解枠とクラスが適切か、深度が同じシーンに対応しているかを確認します。input_report.jsonで使用枚数・クラス別物体数も確認できます。

その後に:

```sh
python 02_train_model.py
python 03_detect_objects.py
python 04_evaluate_model.py
```

初期の160px・2 epochは動作確認用です。本学習ではsettings.yamlのimage_size・epochsなどを変更してください。データやクラスを変更する場合はdata_directoryを新しくし、以前の実験と混ぜないようにします。

検出だけなら正解ラベルは不要ですが、学習と精度評価には正解ラベルが必要です。詳しい各段階の動作は [WORKFLOW.md](WORKFLOW.md) を参照してください。
