# SUN RGB-D 4chサンプル

学習8枚・検証4枚の4ch TIFFと、同名のYOLO形式ラベルを同梱しています。既存ローカルのSUN RGB-D由来データから抽出した動作確認用サンプルです。

- 画素対応済みのB,G,R,Dとして読み込みます。保存値はRGB×256、Depthはmm×8として扱います。
- 前処理設定はpreprocess.yaml、クラスと分割はdata.yaml、画像のSHA256はmanifest.jsonです。
- 5クラス: chair、cabinet、picture、books、table。
- 元データの0値はそのままです。人工的な欠落・欠損補完はありません。
- 少量データで処理が通るか確認するためのもので、精度比較用ではありません。

リポジトリのルートで、01_prepare_data.pyから04_evaluate_model.pyまで順番に実行してください。初期のsettings.yamlはこのサンプルを使用します。
