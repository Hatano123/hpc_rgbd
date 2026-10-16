import zipfile
import os

def extract_zip(zip_path, extract_to):
    if not os.path.exists(zip_path):
        print(f"Error: {zip_path} が見つかりません。")
        return

    print(f"展開中: {zip_path} ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # 展開先フォルダがない場合は作成
        os.makedirs(extract_to, exist_ok=True)
        zip_ref.extractall(extract_to)
    print(f"完了: {extract_to} に展開されました。")

# 設定
base_path = "/home/0/y263357/YOLORGBD"
code_zip = os.path.join(base_path, "code.zip")
data_zip = os.path.join(base_path, "sunrgbd_tiff.zip")

# 1. コードの展開 (YOLORGBD直下に展開)
extract_zip(code_zip, base_path)

# 2. データの展開 (datasets/sunrgbd フォルダに展開)
# YOLOの学習ではデータを別フォルダにまとめると管理しやすいため
extract_data_to = os.path.join(base_path, "datasets/sunrgbd")
extract_zip(data_zip, extract_data_to)