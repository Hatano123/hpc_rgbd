import cv2
import numpy as np
import glob
import os
import shutil
import scipy.io as sio
from scipy.interpolate import griddata # ★ 補完のために追加

# --- 設定 ---
RAW_TOF_DIR = 'raw_dataset/tof'
ANNOTATION_DIR = 'AnnotationData'
# ★ 保存先を 'datasets3' (uint8 用) に変更
DATASET_ROOT_DIR = 'datasets3/rgbd_aligned'
WEBCAM_PARAMS_PATH = 'camera_calibration.mat'
# ★ 解像度を 640x480 に変更
IMG_WIDTH = 640
IMG_HEIGHT = 480
MAX_DEPTH_M = 1.3 # ★ 深度の正規化上限 (メートル)

def load_calibration_params():
    """キャリブレーションパラメータを設定"""
    try:
        webcam_mat = sio.loadmat(WEBCAM_PARAMS_PATH)
        cam_matrix = webcam_mat['cameraMatrix']
        dist_coeffs = webcam_mat['distCoeffs'].flatten()

        R_matrix = np.array([
            [ 0.07489294, -0.99671694,  0.03076343],
            [ 0.10661718,  0.03867679,  0.99354762],
            [-0.99147558, -0.07112979,  0.10916377]
        ])
        rvec, _ = cv2.Rodrigues(R_matrix)

        tvec = np.array([
            [ -0.02300999 * -1.0],
            [-0.09370957 * 1.05],
            [-0.27187898]
        ])

        print("キャリブレーションファイルを正常に読み込み、固定の外部パラメータを設定しました。")
        return cam_matrix, dist_coeffs, rvec, tvec

    except FileNotFoundError as e:
        print(f"エラー: {WEBCAM_PARAMS_PATH} が見つかりません: {e}")
        return None

def main():
    print("--- 4チャンネル TIFF 学習データ (uint8深度) の作成開始 ---")

    params = load_calibration_params()
    if params is None: return
    cam_matrix, dist_coeffs, rvec, tvec = params

    splits = ['train', 'valid', 'test']
    for split in splits:
        print(f"\n--- '{split}' セットの処理を開始 ---")

        rgb_dir = os.path.join(ANNOTATION_DIR, split, 'images')
        label_dir = os.path.join(ANNOTATION_DIR, split, 'labels')

        output_images_dir = os.path.join(DATASET_ROOT_DIR, 'images', split)
        output_labels_dir = os.path.join(DATASET_ROOT_DIR, 'labels', split)
        os.makedirs(output_images_dir, exist_ok=True)
        os.makedirs(output_labels_dir, exist_ok=True)

        label_files = sorted(glob.glob(os.path.join(label_dir, '*.txt')))
        if not label_files:
            print(f"警告: '{label_dir}' にラベルファイルが見つかりません。スキップします。")
            continue

        total_files = len(label_files)
        print(f"{total_files} 個のファイルを処理します。")

        for i, label_path_src in enumerate(label_files):
            filename_base_long_txt = os.path.basename(label_path_src)
            filename_base_long = filename_base_long_txt.replace('.txt', '')

            # 複雑な名前から、最初の '_' で区切って元の番号部分だけを抜き出す
            filename_base_short = filename_base_long.split('_')[0]

            rgb_path = os.path.join(rgb_dir, filename_base_long + '.jpg')
            tof_path = os.path.join(RAW_TOF_DIR, f"{filename_base_short}.npz")

            if not (os.path.exists(rgb_path) and os.path.exists(tof_path)):
                print(f"警告: ペアデータが見つからないためスキップ: RGB='{rgb_path}', ToF='{tof_path}'")
                continue

            tof_data = np.load(tof_path)
            if 'x' not in tof_data or 'y' not in tof_data or 'z' not in tof_data:
                print(f"警告: ToFファイル '{tof_path}' に必要なキー 'x', 'y', 'z' がありません。スキップします。")
                continue

            x_img, y_img, z_img = tof_data['x'], tof_data['y'], tof_data['z']
            z_img = -z_img
            points_3d = np.stack([x_img.flatten(), y_img.flatten(), z_img.flatten()], axis=-1)
            depth_values = x_img.flatten() # 深度として X (距離) を使用
            valid_mask = depth_values > 0
            points_3d = points_3d[valid_mask]
            depth_values = depth_values[valid_mask]
            if len(points_3d) == 0: continue

            projected_points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, cam_matrix, dist_coeffs)

            # 1. まばらな深度マップ (Sparse map) を作成
            aligned_depth_map = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
            for p, d in zip(projected_points_2d, depth_values):
                u, v = int(p[0][0]), int(p[0][1])
                if 0 <= u < IMG_WIDTH and 0 <= v < IMG_HEIGHT:
                    if aligned_depth_map[v, u] == 0 or d < aligned_depth_map[v, u]:
                        aligned_depth_map[v, u] = d

            # --- ▼▼▼ 2. 深度データの補完処理 (Sparse to Dense) ▼▼▼ ---

            # 有効な深度データ（> 0）を持つピクセルの座標 (v, u) と値 (d) を取得
            v_coords, u_coords = np.where(aligned_depth_map > 0)

            if v_coords.size > 10: # 補完するのに十分な点があるか確認
                known_points = np.stack([v_coords, u_coords], axis=-1)
                known_values = aligned_depth_map[v_coords, u_coords]

                # 補完対象となる全ピクセルのグリッド (v, u) を作成
                v_grid, u_grid = np.mgrid[0:IMG_HEIGHT, 0:IMG_WIDTH]

                # griddata で補完 (method='nearest' で最近傍の値を穴埋め)
                dense_depth_map = griddata(known_points, known_values, (v_grid, u_grid), method='nearest')

                # griddataは補完できなかった領域（画像の範囲外など）をnanで埋めるため、0に戻す
                dense_depth_map[np.isnan(dense_depth_map)] = 0
                aligned_depth_map = dense_depth_map.astype(np.float32)

            # --- ▲▲▲ 補完処理ここまで ▲▲▲ ---

            # 3. RGB画像 (uint8)
            rgb_image = cv2.imread(rgb_path)
            if rgb_image.shape[0] != IMG_HEIGHT or rgb_image.shape[1] != IMG_WIDTH:
                rgb_image = cv2.resize(rgb_image, (IMG_WIDTH, IMG_HEIGHT))

            # --- ▼▼▼ 4. 深度情報を uint8 (0-255) で保存 ▼▼▼ ---

            # 4a. 深度データ (float32, メートル単位) を uint8 (0-255) に正規化・変換
            depth_clipped = np.clip(aligned_depth_map, 0, MAX_DEPTH_M)
            # メートル(0-MAX_DEPTH_M) を 0-255 に正規化
            depth_normalized = (depth_clipped / MAX_DEPTH_M) * 255.0
            depth_uint8 = depth_normalized.astype(np.uint8) # ★ uint8 に変換

            # 4b. 4chに結合
            depth_with_axis = np.expand_dims(depth_uint8, axis=-1)
            # RGB(uint8) + D(uint8) = 4ch (uint8)
            rgbd_image_uint8 = np.concatenate((rgb_image, depth_with_axis), axis=-1)

            # 4c. 8bit 4ch TIFF として保存
            tiff_save_path = os.path.join(output_images_dir, f"{filename_base_short}.tiff")
            cv2.imwrite(tiff_save_path, rgbd_image_uint8) # ★ uint8 で保存

            # --- ▲▲▲ 修正ここまで ▲▲▲ ---

            # 5. ラベルファイルのコピー
            label_path_dst = os.path.join(output_labels_dir, f"{filename_base_short}.txt")
            shutil.copyfile(label_path_src, label_path_dst)

            # print文を修正
            print(f"[{i+1}/{total_files}] 処理完了 ({split}): {filename_base_short}.tiff / .txt (8bit)")

    print("\n--- 全てのファイルの処理が完了しました ---")

if __name__ == '__main__':
    main()