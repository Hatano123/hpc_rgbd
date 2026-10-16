# 段階別の実行手順

リポジトリのルートにあるsettings.yamlを編集し、4つのPythonファイルを順に実行します。引数を毎回書く必要はありません。

初期設定では同梱のsamples/sunrgbd_top5（学習8枚・検証4枚）を使います。最初の動作確認では設定変更は不要です。

## 入力データ

撮影からラベル付け・学習用フォルダの作成までの手順は [MAKE_DATASET.md](MAKE_DATASET.md) を参照してください。

```text
source_dataset/
  images/train/*.tiff
  images/val/*.tiff
  labels/train/*.txt
  labels/val/*.txt
  data.yaml              クラス名。ない場合はconfigs/class_names.yamlを使用
```

画像とラベルは同じファイル名の拡張子違いです。ラベル1行は `class_id x_center y_center width height`、座標は0～1です。

## 01: 入力確認・データ整形

```sh
python 01_prepare_data.py
```

settings.yamlで指定した入力元から学習・検証画像をコピーします。元データは変更しません。画像形状・ラベルの形式と範囲・画像数を確認します。

前処理はRGBの保存倍率、深度の保存倍率、深度上限の3項目だけです。今回の形式はRGB保存値÷256、深度保存値÷8→mm、深度を0～8000mmでクリップして線形正規化します。欠損補完や人工的なゼロ化はありません。

内部ではHWC・BGRD・0～255の配列を作り、モデル入力時にBCHW・0～1に変換します。学習と検出・評価で同じ前処理を使います。

出力先の初期値は `work/quickstart/data` です。

| ファイル | 確認すること |
|---|---|
| preview.jpg | RGB＋正解枠と深度を左右に並べた画像 |
| input_report.json | 選択画像、クラス別物体数、設定、SHA256 |
| data.yaml | 学習・検証の分割とクラス、channels: 4 |
| preprocess.yaml | 保存倍率と深度上限 |
| sample_preprocessed.npz | 整形済み配列の確認用例 |

学習はコピーしたTIFFを共通ローダで読みます。NPZは確認用です。01の再実行時は、入力と設定が一致する整形済みデータを確認して再利用します。

## 02: 4ch学習

```sh
python 02_train_model.py
```

01で作ったdata.yamlとpreprocess.yamlを読み込みます。初期重みはweights/yolo11n.ptです。標準YOLO11nの入力層を4chへ拡張し、深度も含めて学習します。

初期の出力先は `runs/quickstart`。`weights/best.pt`、`weights/last.pt`、損失のresults.csv、実行設定のmanifest.jsonを保存します。モデルには前処理設定も保存します。

## 03: 検出

```sh
python 03_detect_objects.py
```

02のbest.ptを読み込み、準備した検証画像を一括検出します。別の重みを使う場合はsettings.yamlのmodel_fileを指定します。

初期の出力先は `work/quickstart/detections`。画像ごとのフォルダにprediction.jpgとdetections.json、全体にindex.jsonを保存します。正解ラベルを使わず予測する段階です。

## 04: 評価

```sh
python 04_evaluate_model.py
```

モデルを検証用画像と正解ラベルで評価します。03の検出JSONを再利用するのではなく、評価用の設定で推論し直して計測します。03は画像の確認、04は指標の計算という独立した段階です。

初期の出力先は `work/quickstart/evaluation`。metrics.json、クラス別SUMMARY.md、評価図を保存します。Precision、Recall、AP50、AP50-95、mAPを確認してください。正解物体がないクラスは評価できません。

## 再実行と別の実験

- 02以降はその段階だけ実行できます。準備データや重みがない場合は、実行すべき前の段階を表示してエラーにします。
- すべてのスクリプトが `--settings 別の設定.yaml` に対応します。相対パスは設定ファイルの場所が基準です。
- データ選択の設定・入力が変わる場合はdata_directoryを変えます。過去のモデルを残す場合はtraining_directoryを変えます。
- 初期値は動作確認用です。8枚・2 epochの結果で精度の良し悪しは判断できません。

## 内部処理を直接呼ぶ場合

従来の `python -m rgbd_yolo prepare/train/predict/val/demo` も残しています。個別の細かい引数が必要な場合だけ使用してください。
