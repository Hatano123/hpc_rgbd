# -*- coding: utf-8 -*-
"""
WebカメラとToFカメラ間の手動クロスキャリブレーション統合スクリプト
（チェックモード、行列直接指定モード機能付き）

■ 実行フロー:
1.  実行するとモード選択が表示されます。
    - '1'を選択: データ収集 -> 計算 -> 評価 の全工程を実行します。
    - '2'を選択: 'extrinsics.npz'を読み込み、評価モードのみを即時実行します。
    - '3'を選択: スクリプト内に直接記述された外部パラメータを使い、評価モードを実行します。
2.  各工程の詳細は、画面の指示に従ってください。
"""
import cv2
import numpy as np
import glob
import os
import socket
import re
import time
import scipy.io as sio
import traceback

# --- 1. 定数と設定 (★★ ユーザーが設定する項目 ★★) ---
CHECKERBOARD = (10, 7)
SQUARE_SIZE = 0.024
SAVE_DIR = 'manual_calib_data'
WEBCAM_PARAMS_PATH = 'camera_calibration.mat'
TOF_PARAMS_PATH = 'tof_calibration.mat'
EXTRINSICS_SAVE_PATH = 'extrinsics.npz'
WEBCAM_INDEX = 1
TOF_IP = "192.168.1.69"
TOF_PORT = 50010
TOF_WIDTH = 176
TOF_HEIGHT = 132

# --- 2. ソケット通信関数 ---
def read_socket_data(s, number_bytes):
    data = bytearray()
    while len(data) < number_bytes:
        part = s.recv(number_bytes - len(data))
        if not part: raise RuntimeError('Socket connection broken')
        data.extend(part)
    return data

def read_pcic_answer(s):
    header = read_socket_data(s, 16)
    ticket = header[0:4]
    match = re.search(rb'L(\d+)', header)
    if not match: raise ValueError(f"Could not find payload length in header: {header.decode('ascii', errors='ignore')}")
    total_payload_length = int(match.group(1))
    full_response_data = read_socket_data(s, total_payload_length)
    return ticket, full_response_data[4:-2]

def wait_for_ticket(s, ticket_str):
    while True:
        read_ticket, answer = read_pcic_answer(s)
        if read_ticket == ticket_str.encode(): return answer

def send_pcic_command(s, cmd, ticket="1000"):
    cmd_bytes = cmd.encode('ascii')
    cmd_len = len(cmd_bytes) + 6
    header_str = f"{ticket}L{cmd_len:09d}\r\n"
    s.sendall(header_str.encode())
    s.sendall(ticket.encode())
    s.sendall(cmd_bytes)
    s.sendall(b"\r\n")
    try:
        return wait_for_ticket(s, ticket)
    except (RuntimeError, ValueError) as e:
        print(f"Error sending command '{cmd}': {e}")
        return None

def connect_to_tof():
    try:
        tof_s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tof_s.connect((TOF_IP, TOF_PORT))
        tof_s.settimeout(5)
        print(f"ToFカメラ ({TOF_IP}:{TOF_PORT}) に接続しました。")
        send_pcic_command(tof_s, "p0")
        return tof_s, True
    except Exception as e:
        print(f"警告: ToFカメラに接続できませんでした: {e}")
        return None, False

# --- 3. TOFデータ処理関数 ---
def get_tof_data(tof_socket):
    DATA_START_OFFSET_BYTES = 57
    def load_image_from_raw(data, dtype, expected_bytes):
        if data is None or len(data) < DATA_START_OFFSET_BYTES: return None
        image_payload = data[DATA_START_OFFSET_BYTES:]
        if len(image_payload) != expected_bytes: return None
        return np.frombuffer(image_payload, dtype=dtype).reshape((TOF_HEIGHT, TOF_WIDTH))
    try:
        x_raw = send_pcic_command(tof_socket, "I06?")
        y_raw = send_pcic_command(tof_socket, "I04?")
        z_raw = send_pcic_command(tof_socket, "I05?")
        ir_raw = send_pcic_command(tof_socket, "I03?")
        if not all((x_raw, y_raw, z_raw, ir_raw)): return None
        expected_bytes = TOF_WIDTH * TOF_HEIGHT * 2
        x_img = load_image_from_raw(x_raw, np.int16, expected_bytes)
        y_img = load_image_from_raw(y_raw, np.int16, expected_bytes)
        z_img = load_image_from_raw(z_raw, np.int16, expected_bytes)
        ir_img = load_image_from_raw(ir_raw, np.uint16, expected_bytes)
        if not all((x_img is not None, y_img is not None, z_img is not None, ir_img is not None)): return None
        x_img = x_img / 1000.0
        y_img = y_img / 1000.0
        z_img = z_img / 1000.0
        return x_img, y_img, z_img, ir_img
    except Exception as e:
        print(f"!!致命的なエラー!! get_tof_data 内で予期せぬエラーが発生しました: {e}")
        traceback.print_exc()
        return None

# --- 4. キャリブレーションのコア処理 ---
def load_intrinsics():
    print("\n--- [ステップ 1/5] 内部パラメータの読み込み ---")
    try:
        webcam_mat = sio.loadmat(WEBCAM_PARAMS_PATH)
        webcam_cam_matrix = webcam_mat['cameraMatrix']
        webcam_dist_coeffs = webcam_mat['distCoeffs'].flatten()
        print("内部パラメータを正常に読み込みました。")
        return webcam_cam_matrix, webcam_dist_coeffs
    except (FileNotFoundError, KeyError) as e:
        print(f"エラー: パラメータファイル '{WEBCAM_PARAMS_PATH}' の読み込みに失敗しました: {e}")
        return None, None

def collect_data():
    print("\n--- [ステップ 2/5] データ収集 ---")
    print("チェッカーボードを両カメラに写し、[s]で保存, [q]で終了")
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened(): print(f"エラー: Webカメラ(インデックス: {WEBCAM_INDEX})を開けません。"); return
    tof_s, tof_ok = connect_to_tof()
    os.makedirs(SAVE_DIR, exist_ok=True)
    img_count = len(glob.glob(os.path.join(SAVE_DIR, 'webcam_*.png')))
    while True:
        ret, frame_web = cap.read()
        if not ret: print("エラー: Webカメラからフレームを取得できません。"); break
        display_tof = np.zeros((TOF_HEIGHT, TOF_WIDTH, 3), dtype=np.uint8)
        tof_data_tuple = get_tof_data(tof_s) if tof_ok else None
        if tof_data_tuple:
            _, _, _, ir_img = tof_data_tuple
            if ir_img is not None:
                display_tof = cv2.normalize(ir_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                display_tof = cv2.cvtColor(display_tof, cv2.COLOR_GRAY2BGR)
        h, w, _ = frame_web.shape
        display_tof_resized = cv2.resize(display_tof, (w, h))
        cv2.imshow('Webcam (Left) - ToF IR (Right) | [s] save, [q] quit', np.hstack((frame_web, display_tof_resized)))
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('s'):
            if tof_data_tuple:
                img_count += 1
                web_path = os.path.join(SAVE_DIR, f'webcam_{img_count:03d}.png')
                cv2.imwrite(web_path, frame_web)
                x_img, y_img, z_img, ir_img = tof_data_tuple
                np.savez(os.path.join(SAVE_DIR, f'tof_{img_count:03d}.npz'), x=x_img, y=y_img, z=z_img, ir=ir_img)
                print(f"保存しました: {web_path} と tof_{img_count:03d}.npz")
            else: print("警告: ToFデータが取得できていないため、保存できません。")
    cap.release()
    if tof_s: tof_s.close()
    cv2.destroyAllWindows()
    print("データ収集を終了しました。")

def calculate_extrinsics(webcam_cam_matrix, webcam_dist_coeffs):
    print("\n--- [ステップ 3/5 & 4/5] 外部パラメータの計算 ---")
    all_object_points, all_image_points = [], []
    image_files = sorted(glob.glob(os.path.join(SAVE_DIR, 'webcam_*.png')))
    tof_files = sorted(glob.glob(os.path.join(SAVE_DIR, 'tof_*.npz')))
    if len(image_files) != len(tof_files) or not image_files:
        print("エラー: Webカメラ画像とToFデータの数が一致しないか、データがありません。")
        return None, None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    successful_pairs = 0
    for i, (img_file, tof_file) in enumerate(zip(image_files, tof_files)):
        img_web = cv2.imread(img_file)
        gray_web = cv2.cvtColor(img_web, cv2.COLOR_BGR2GRAY)
        ret_web, corners_web = cv2.findChessboardCorners(gray_web, CHECKERBOARD, None)
        tof_data = np.load(tof_file)
        ir_img = tof_data['ir']
        ir_gray = cv2.normalize(ir_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        ret_tof, corners_tof = cv2.findChessboardCorners(ir_gray, CHECKERBOARD, None)
        if ret_web and ret_tof:
            successful_pairs += 1
            corners_web_subpix = cv2.cornerSubPix(gray_web, corners_web, (11,11), (-1,-1), criteria)
            corners_tof_subpix = cv2.cornerSubPix(ir_gray, corners_tof, (11,11), (-1,-1), criteria)
            for j in range(len(corners_tof_subpix)):
                tof_corner = corners_tof_subpix[j][0]
                web_corner = corners_web_subpix[j]
                u, v = int(tof_corner[0]), int(tof_corner[1])
                x, y, z = tof_data['x'][v, u], tof_data['y'][v, u], tof_data['z'][v, u]
                y, z = -y, -z # 座標系を補正
                if z > 0.1:
                    all_object_points.append([x, y, z])
                    all_image_points.append(web_corner)
    print(f"\nコーナー検出に成功したペア数: {successful_pairs}")
    print(f"全てのペアから収集した有効な対応点の総数: {len(all_object_points)}")
    MIN_POINTS_FOR_CALCULATION = 20
    if len(all_object_points) < MIN_POINTS_FOR_CALCULATION:
        print(f"エラー: 計算に十分な対応点が得られませんでした。(必要: {MIN_POINTS_FOR_CALCULATION}点, 取得: {len(all_object_points)}点)")
        return None, None
    obj_pts = np.array(all_object_points, dtype=np.float32)
    img_pts = np.array(all_image_points, dtype=np.float32)
    success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, webcam_cam_matrix, webcam_dist_coeffs)
    if success:
        print("\n✅ 外部パラメータ計算成功！")
        np.savez(EXTRINSICS_SAVE_PATH, rvec=rvec, tvec=tvec)
        print(f"結果を {EXTRINSICS_SAVE_PATH} に保存しました。")
        return rvec, tvec
    else:
        print("エラー: solvePnPによる外部パラメータ計算に失敗しました。")
        return None, None

def verify_projection(webcam_cam_matrix, webcam_dist_coeffs, rvec, tvec):
    print("\n--- [ステップ 5/5] 評価モード ---")
    print("ToF点群をWebカメラ映像に投影します。[q]で終了")
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    tof_s, tof_ok = connect_to_tof()
    if not tof_ok: print("ToFカメラに接続できないため、評価をスキップします。"); cap.release(); return
    while True:
        ret, frame = cap.read()
        if not ret: break
        projected_frame = frame.copy() # 元のフレームを汚さないようにコピーを作成
        tof_data_tuple = get_tof_data(tof_s)
        if tof_data_tuple:
            x_img, y_img, z_img, ir_img = tof_data_tuple
            z_img = -z_img # 座標系を補正
            AMPLITUDE_THRESHOLD = 300
            valid_mask_amplitude = (ir_img.flatten() > AMPLITUDE_THRESHOLD)
            points_3d = np.stack((x_img.flatten(), y_img.flatten(), z_img.flatten()), axis=-1)
            valid_mask_distance = points_3d[:, 2] > 0.1
            final_valid_mask = valid_mask_distance & valid_mask_amplitude
            points_3d_valid = points_3d[final_valid_mask]
            if len(points_3d_valid) > 0:
                projected_points_2d, _ = cv2.projectPoints(points_3d_valid, rvec, tvec, webcam_cam_matrix, webcam_dist_coeffs)
                for p in projected_points_2d:
                    u, v = int(p[0][0]), int(p[0][1])
                    if 0 <= u < frame.shape[1] and 0 <= v < frame.shape[0]:
                        cv2.circle(projected_frame, (u, v), 1, (0, 255, 0), -1)
        cv2.imshow('Verification: Projected ToF points on Webcam', projected_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    if tof_s: tof_s.close()
    cv2.destroyAllWindows()

# --- メイン実行部分 ---
if __name__ == "__main__":

    # ★★ モード選択機能 ★★
    print("実行するモードを選択してください:")
    print("  1: キャリブレーションを最初からすべて行う (データ収集 -> 計算 -> 評価)")
    print("  2: 保存済みの 'extrinsics.npz' を使って評価モードのみ実行する")
    print("  3: コードに直接書き込まれた行列を使って評価モードのみ実行する")
    mode = input("選択 (1, 2 or 3): ")

    w_matrix, w_dist = load_intrinsics()
    if w_matrix is None:
        print("Webカメラの内部パラメータが読み込めないため、プログラムを終了します。")

    elif mode == '1':
        # モード1: フルキャリブレーション
        collect_data()
        rvec, tvec = calculate_extrinsics(w_matrix, w_dist)
        if rvec is not None and tvec is not None:
            verify_projection(w_matrix, w_dist, rvec, tvec)

    elif mode == '2':
        # モード2: ファイルから読み込んで評価
        try:
            print(f"\n'{EXTRINSICS_SAVE_PATH}' から外部パラメータを読み込みます。")
            data = np.load(EXTRINSICS_SAVE_PATH)

            # キーの存在をチェックする処理
            if 'rvec' in data and 'tvec' in data:
                rvec = data['rvec']
                tvec = data['tvec']
                print("読み込み成功。評価モードを開始します。")
                verify_projection(w_matrix, w_dist, rvec, tvec)
            else:
                # キーが存在しない場合のエラーメッセージ
                print(f"エラー: {EXTRINSICS_SAVE_PATH} に 'rvec' または 'tvec' が含まれていません。")
                print("このファイルは不完全か、異なる形式の可能性があります。")
                print("モード'1'を実行して、キャリブレーションファイルを再生成してください。")

        except FileNotFoundError:
            print(f"エラー: {EXTRINSICS_SAVE_PATH} が見つかりません。")
            print("先にモード'1'を実行して、キャリブレーションファイルを生成してください。")
        except Exception as e:
            # 念のため、その他の予期せぬエラーも捕捉します
            print(f"ファイルの読み込み中に予期せぬエラーが発生しました: {e}")

    elif mode == '3':
        # ★★ モード3: ここに行列の値を直接入力 ★★
        try:
            print("\nコードに直接記述された外部パラメータを使用します。")

            # ↓↓↓↓ ここにあなたの値を入力してください ↓↓↓↓
            # 指定された3x3の回転行列
            R_matrix = np.array([
                [ 0.07489294, -0.99671694,  0.03076343],
                [ 0.10661718,  0.03867679,  0.99354762],
                [-0.99147558, -0.07112979,  0.10916377]
            ])
            rvec, _ = cv2.Rodrigues(R_matrix)
            tvec = np.array([
                [ -0.02300999 * -2.5],
                [-0.09370957 *1.3],
                [-0.27187898]
            ])
            print("読み込み成功。評価モードを開始します。")
            verify_projection(w_matrix, w_dist, rvec, tvec)

        except Exception as e:
            print(f"エラー: 指定された行列の形式が正しくない可能性があります: {e}")

    else:
        print("無効な選択です。'1', '2'または'3'を入力してください。")

    print("\nプログラムを終了します。")