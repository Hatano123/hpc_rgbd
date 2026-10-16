# データ入力から評価まで

## 1. データ入力

画素対応済みの4ch TIFFとYOLO形式ラベルを使用します。

```text
dataset/
  images/train/*.tiff
  images/val/*.tiff
  labels/train/*.txt
  labels/val/*.txt
  data.yaml
```

画像とラベルは同じファイル名の拡張子違いです。ラベル1行は `class_id x_center y_center width height`、座標は0～1です。

入力元には手を加えず、指定枚数を別の場所へコピーします。seedを固定して同じ部分集合を選びます。

```sh
python -m rgbd_yolo prepare --source /path/to/sunrgbd_top5 --output work/subset --train-count 8 --val-count 4 --seed 0
```

元データにdata.yamlがない場合、既定ではリポジトリのTop5クラス定義を使用します。別クラスなら `--classes /path/classes.yaml` を指定してください。

## 2. データ整形

既定のSUN RGB-D TIFFプロファイルで行うことは、以下の単位変換と線形正規化です。

```text
RGB保存値 / 256                         → BGRの8bit強度
Depth保存値 / 8                         → 深度mm
clip(深度mm, 0, 8000) / 8000             → 深度0～1
BGR / 255 と深度0～1を結合               → [B,G,R,D]
学習・推論に渡すTensor                   → [batch,4,height,width]
```

内部の画像配列は0～255、Tensor化する段階で÷255します。前処理はtrain/val/predictで共通です。深度の0値を他の値で埋めたり、チャンネルをランダムにゼロにしたりしません。深度の対数変換もありません。

`prepare` では次の確認用ファイルも出力します。学習はコピーしたTIFFを共通前処理で読み、NPZは確認用です。

| 出力 | 内容 |
|---|---|
| data.yaml | 学習・検証の分割とクラス、channels: 4 |
| preprocess.yaml | 保存倍率と深度上限 |
| preview.jpg | 左: RGBと正解枠、右: 正規化後の深度 |
| sample_preprocessed.npz | 実際の整形済み4ch配列の例 |
| input_report.json | 選択画像、SHA256、枚数、クラス別物体数、設定 |

プレビューの深度の色は表示用で、追加の入力チャンネルではありません。

## 3. 4ch学習

```sh
python -m rgbd_yolo train --data work/subset/data.yaml --config work/subset/preprocess.yaml --epochs 2 --imgsz 160 --batch 2 --name quick --no-plots
```

`runs/quick/weights/best.pt` と `last.pt`、損失などの `results.csv`、設定と実行環境の `manifest.json` が出ます。通常解像度の本学習は `--imgsz 640`、エポック数やbatchを実行環境に合わせて変更します。

RGBだけとの比較は `train --mode rgb`。3ch画像用の別data.yamlを指定します。4ch画像から勝手に深度を削除する処理ではありません。

## 4. 検出

```sh
python -m rgbd_yolo predict --model runs/quick/weights/best.pt --images work/subset/images/val --imgsz 160 --output work/detections
```

画像ごとに `prediction.jpg` と `detections.json`、フォルダ全体に `index.json` を出します。予測の枠、クラス名、信頼度を確認できます。1枚だけなら `--image` を使います。正解ラベルを使用しないのが検出です。

## 5. 評価

```sh
python -m rgbd_yolo val --model runs/quick/weights/best.pt --data work/subset/data.yaml --imgsz 160 --output work/evaluation
```

正解ラベルと検出結果を比較し、Precision、Recall、AP50、AP50-95、mAPを計算します。`metrics.json` とクラス別の `SUMMARY.md`、Ultralyticsの評価図を保存します。分割に正解がないクラスは評価できません。

入力整形が正しいこと、学習できること、精度が良いことはそれぞれ別の確認です。8枚・2 epochで良い精度になることは期待せず、まず各段階がつながるかを確認します。

## 一通り自動実行

```sh
python -m rgbd_yolo demo --source /path/to/sunrgbd_top5 --output work/demo_01
```

途中経過が1/5～5/5で表示され、最後に `work/demo_01/REPORT.md` が出ます。再実行時は `--output work/demo_02` のように新しい出力先を使います。
