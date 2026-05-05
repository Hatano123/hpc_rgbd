# -*- coding: utf-8 -*-
"""
WebカメラとToFカメラ間の手動クロスキャリブレーション統合スクリプト
（ifm3dpyライブラリ使用・露光時間自動設定・自動取得・モード2調整 適用バージョン）

■ 修正点 (2025/11/12):
- 単位のバグを修正: ToFのXYZデータを (mm) -> (m) に変換 (/ 1000.0)
- calculate_extrinsics と verify_projection の両方に適用
- Z軸反転 (-z_val) はメートル単位で実行
"""
import cv2
import numpy as np
import glob
import os
import re
import time
import scipy.io as sio
import traceback
import json
import logging

# --- ifm3dpy (aa.py から) のライブラリを追加 ---
try:
    from ifm3dpy.device import Device, O3D
    from ifm3dpy.framegrabber import FrameGrabber, buffer_id
except ImportError:
    print("エラー: ifm3dpy ライブラリが見つかりません。")
    print("コマンドプロンプト（PowerShellなど）で `pip install ifm3dpy` を実行してください。")
    exit(-1)
# ---------------------------------------------

# --- 1. 定数と設定 (★★ ユーザーが設定する項目 ★★) ---
CHECKERBOARD = (10, 7)
SQUARE_SIZE = 0.034
SAVE_DIR = 'manual_calib_data'
WEBCAM_PARAMS_PATH = 'camera_calibration.mat'
TOF_PARAMS_PATH = 'tof_calibration.mat' # (アプローチ2Bでは不要だが念のため残す)
EXTRINSICS_SAVE_PATH = 'extrinsics.npz'
WEBCAM_INDEX = 1
TOF_IP = "192.168.1.69"
APPLY_TVEC_MIRROR = False  # True only if old extrinsics look mirrored in Mode 2

# ★★ (aa.py から) ToFカメラの自動設定値 ★★
NEW_EXPOSURE_TIME_US = 10000 # (us) キャリブレーション用に設定したい露光時間

# ★★ 深度による色付けの範囲と表示点のしきい値を設定 ★★
MIN_DEPTH_VIZ = 0.3  # [m] 深度可視化の最小距離
MAX_DEPTH_VIZ = 1.0  # [m] 深度可視化の最大距離



TOF_WIDTH = 176*2  # (352)
TOF_HEIGHT = 132*2 # (264)

# --- ロガー設定 (aa.py から) ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ---------------------------------

# --- 3. ifm3dpy を使ったToFセットアップとデータ取得 (aa.py ベースに修正) ---
def setup_tof_grabber():
    """
    ToFカメラに接続し、露光時間を設定し、FrameGrabberを準備する。
    ★ 修正点: 高解像度 (Resolution: "1") の アプリ '2' を使用する
    """
    try:
        # --- 1. デバイスに接続し、設定を変更 (aa.py のロジック) ---
        device = Device(TOF_IP)
        device_type = device.who_am_i()

        if device_type == device.device_family.O3D or device_type == device.device_family.O3X:
            o3d = O3D(TOF_IP)
            
            logger.info("Reading full config from O3D/O3X...")
            config = o3d.to_json()

            # --- 1a. アプリケーション '2' の露光時間を変更 ---
            # Try to parse config (to_json may return dict or JSON string). Modify Apps[2].Imager.ExposureTime.
            try:
                cfg = None
                if isinstance(config, str):
                    cfg = json.loads(config)
                elif isinstance(config, dict):
                    cfg = config
                else:
                    # unknown type; try to coerce
                    try:
                        cfg = json.loads(str(config))
                    except Exception:
                        cfg = None

                if cfg is None:
                    logger.warning("Could not parse device config; skipping exposure change.")
                else:
                    apps = cfg.get('Apps') or cfg.get('apps')
                    if apps and len(apps) > 2:
                        app2 = apps[2]
                        imager = app2.get('Imager') or app2.get('imager')
                        if imager is not None:
                            imager['ExposureTime'] = int(NEW_EXPOSURE_TIME_US)
                            logger.info(f"Set Apps[2].Imager.ExposureTime = {NEW_EXPOSURE_TIME_US} µs")
                        else:
                            logger.warning("Apps[2] does not contain 'Imager' field; cannot set exposure.")
                    else:
                        logger.warning("Apps[2] not found in device config; cannot set exposure.")

                    # --- 1b. アクティブなアプリケーションを '2' に設定 ---
                    if 'ActiveApplication' in cfg:
                        cfg['ActiveApplication'] = 2
                    elif 'activeApplication' in cfg:
                        cfg['activeApplication'] = 2
                    else:
                        # best-effort: set top-level key
                        cfg['ActiveApplication'] = 2

                    # --- 1c. 変更した設定全体を書き戻し ---
                    try:
                        logger.info("Writing modified config back to O3D/O3X...")
                        o3d.from_json(cfg)
                        logger.info("O3D/O3X configuration successful.")
                    except Exception as e:
                        logger.error(f"Failed to write config back to device: {e}")
                        traceback.print_exc()

            except Exception as e:
                logger.error(f"Error while attempting to modify device config: {e}")
                traceback.print_exc()

            # --- 2. FrameGrabber を初期化 ---
            logger.info("Starting FrameGrabber...")
            fg = FrameGrabber(device) 
            
            # キャリブレーションに必要なデータ (XYZ点群と振幅画像) を要求
            fg.start([
                buffer_id.XYZ,                  # X, Y, Z の点群データ
                buffer_id.NORM_AMPLITUDE_IMAGE  # 振幅画像 (IRの代わり)
            ])
            
            logger.info(f"ToFカメラ ({TOF_IP}) に接続し、FrameGrabberを開始しました。")
            return device, fg

        else:
            logger.warning(f"This script is for O3D/O3X, but device is {device_type}")
            return None, None

    except Exception as e:
        logger.error(f"!!致命的なエラー!! setup_tof_grabber 内でエラーが発生しました: {e}")
        traceback.print_exc()
        return None, None
    
def get_tof_data(fg: FrameGrabber):
    """
    FrameGrabber からデータを取得し、
    (x_img, y_img, z_img, ir_img) の形式で返す。
    """
    try:
        # フレームを待機 (1500ms タイムアウト)
        [ok, frame] = fg.wait_for_frame().wait_for(1500)

        if not ok:
            logger.warning("Timeout while waiting for ToF frame.")
            return None

        # フレームからXYZバッファを取得 (H, W, 3) のNumPy配列
        xyz_data = frame.get_buffer(buffer_id.XYZ)
        
        # フレームから振幅画像バッファを取得 (H, W) のNumPy配列
        ir_img = frame.get_buffer(buffer_id.NORM_AMPLITUDE_IMAGE)

        # XYZデータを x, y, z に分割
        # ifm3dpy はメートル単位の float で返す... と仮定していたが、
        # ログから「ミリメートル」が返されていることが判明。
        # ★★★ /1000.0 でメートルに変換 ★★★
        x_img = xyz_data[:, :, 0] / 1000.0
        y_img = xyz_data[:, :, 1] / 1000.0
        z_img = xyz_data[:, :, 2] / 1000.0
        
        return x_img, y_img, z_img, ir_img

    except Exception as e:
        logger.error(f"!!致命的なエラー!! get_tof_data 内で予期せぬエラーが発生しました: {e}")
        traceback.print_exc()
        return None

# --- 4. キャリブレーションのコア処理 (ifm3dpy を使うように修正) ---

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

# --- ▼▼▼【改良】collect_data を手動撮影版に置き換え ▼▼▼ ---
def collect_data():
    print("\n--- [ステップ 2/5] データ収集 (手動撮影モード) ---")
    print("チェッカーボードを両カメラに写してください。")
    print("[Space]キーで撮影、[q]で終了します。")

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened(): 
        print(f"エラー: Webカメラ(インデックス: {WEBCAM_INDEX})を開けません。")
        return
    # DEBUG: 現在のカメラ解像度を確認（最初のフレームを一度読み出して shape を表示）
    try:
        ret_check, frame_check = cap.read()
        if ret_check:
            print(f"[DEBUG] Webcam actual frame shape (collect_data): {frame_check.shape} (H,W,CH)")
        else:
            print("[DEBUG] Warning: could not read test frame after opening webcam (collect_data).")
    except Exception as e:
        print(f"[DEBUG] Exception while checking webcam frame in collect_data(): {e}")
        
    # --- 修正: ifm3dpy でセットアップ ---
    device, fg = setup_tof_grabber()
    if not fg:
        print("エラー: ToFカメラのセットアップに失敗しました。データ収集を中止します。")
        cap.release()
        return
    # ---------------------------------

    os.makedirs(SAVE_DIR, exist_ok=True)
    img_count = len(glob.glob(os.path.join(SAVE_DIR, 'webcam_*.png')))

    # create window and mouse-callback based capture button
    window_name = 'Webcam (Left) - ToF IR (Right) | [Space] to capture, [q] quit'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    capture_requested = {'val': False}
    def _on_mouse_collect(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            bx, by, bw, bh = 10, 10, 160, 50
            if bx <= x <= bx + bw and by <= y <= by + bh:
                capture_requested['val'] = True
    cv2.setMouseCallback(window_name, _on_mouse_collect)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    while True:
        # --- 修正: ifm3dpy でToF取得 ---
        tof_data_tuple = get_tof_data(fg)
        ret, frame_web = cap.read()
        
        if not ret: 
            print("エラー: Webカメラからフレームを取得できません。"); break

        frame_web_display = frame_web.copy()
        
        ir_gray = None
        display_tof = None
        if tof_data_tuple:
            x_img, y_img, z_img, ir_img = tof_data_tuple
            ir_gray = cv2.normalize(ir_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            display_tof = cv2.cvtColor(ir_gray, cv2.COLOR_GRAY2BGR)
        else:
            display_tof = np.zeros((TOF_HEIGHT, TOF_WIDTH, 3), dtype=np.uint8)
        
        gray_web = cv2.cvtColor(frame_web, cv2.COLOR_BGR2GRAY)

        # --- 両方のカメラでコーナー検出を試みる ---
        ret_web, corners_web = cv2.findChessboardCorners(gray_web, CHECKERBOARD, None)
        
        ret_tof = False
        if ir_gray is not None:
            ret_tof, corners_tof = cv2.findChessboardCorners(ir_gray, CHECKERBOARD, None)

        # --- コーナー検出のフィードバック表示 ---
        if ret_web:
            cv2.drawChessboardCorners(frame_web_display, CHECKERBOARD, corners_web, ret_web)
        if ret_tof:
            cv2.drawChessboardCorners(display_tof, CHECKERBOARD, corners_tof, ret_tof)

        # --- 表示 ---
        h, w, _ = frame_web_display.shape
        display_tof_resized = cv2.resize(display_tof, (w, h))
        
        # 検出状態が分かりやすいように左右に並べて表示
        combined_view = np.hstack((frame_web_display, display_tof_resized))
        
        # 撮影可能状態を表示
        if ret_web and ret_tof:
            cv2.putText(combined_view, "Ready to capture! Press [Space]", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            status = "Detecting: "
            if ret_web:
                status += "Webcam OK"
            if ret_tof:
                status += " | ToF OK"
            if not ret_web and not ret_tof:
                status += "No checkerboard detected"
            cv2.putText(combined_view, status, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # draw on-screen capture button (top-left)
        bx, by, bw, bh = 10, 10, 160, 50
        cv2.rectangle(combined_view, (bx, by), (bx + bw, by + bh), (0, 128, 255), -1)
        cv2.putText(combined_view, 'CAPTURE', (bx + 10, by + 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        cv2.imshow(window_name, combined_view)

        key = cv2.waitKey(1) & 0xFF

        # Capture on Space key OR on-screen button click
        do_capture = False
        if key == ord(' '):
            do_capture = True
        if capture_requested.get('val', False):
            do_capture = True
            capture_requested['val'] = False

        if do_capture:
            if ret_web and ret_tof and tof_data_tuple:
                print(f"\n撮影 {img_count + 1}:")

                # サブピクセル補正を実行
                corners_web_subpix = cv2.cornerSubPix(gray_web, corners_web, (11,11), (-1,-1), criteria)
                corners_tof_subpix = cv2.cornerSubPix(ir_gray, corners_tof, (11,11), (-1,-1), criteria)

                # データを保存
                img_count += 1
                web_path = os.path.join(SAVE_DIR, f'webcam_{img_count:03d}.png')
                cv2.imwrite(web_path, frame_web)

                # ToFデータを保存
                np.savez(os.path.join(SAVE_DIR, f'tof_{img_count:03d}.npz'), 
                         x=x_img, y=y_img, z=z_img, ir=ir_img)

                print(f"  保存しました: {web_path} と tof_{img_count:03d}.npz")

                # 保存フィードバック（0.5秒間表示）
                feedback_frame = combined_view.copy()
                cv2.putText(feedback_frame, f"Captured #{img_count}!", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                cv2.imshow(window_name, feedback_frame)
                cv2.waitKey(500)
            else:
                print("警告: チェッカーボードが両方のカメラで検出されていません。撮影できません。")

        elif key == ord('q'):
            break

    # --- クリーンアップ ---
    cap.release()
    if fg: 
        fg.stop()
    cv2.destroyAllWindows()
    print(f"\nデータ収集を終了しました。合計 {img_count} 枚を保存しました。")
# --- ▲▲▲ collect_data 置き換えここまで ▲▲▲ ---


# --- ▼▼▼ 自動撮影モード (Mode 4) の追加 ---
def auto_capture_mode(interval_s=3):
    """
    自動撮影モード:
    - ウィンドウにカウントダウン表示を出し、interval_s 秒ごとに自動で撮影・保存します。
    - 'q' キーで終了します。
    """
    print("\n--- 自動撮影モード (Mode 4) ---")
    print(f"{interval_s} 秒ごとに自動撮影します。'q' で終了。")

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"エラー: Webカメラ(インデックス: {WEBCAM_INDEX})を開けません。")
        return 0
    # DEBUG: 現在のカメラ解像度を確認（最初のフレームを一度読み出して shape を表示）
    try:
        ret_check, frame_check = cap.read()
        if ret_check:
            print(f"[DEBUG] Webcam actual frame shape (auto_capture_mode): {frame_check.shape} (H,W,CH)")
        else:
            print("[DEBUG] Warning: could not read test frame after opening webcam (auto_capture_mode).")
    except Exception as e:
        print(f"[DEBUG] Exception while checking webcam frame in auto_capture_mode(): {e}")

    device, fg = setup_tof_grabber()
    if not fg:
        print("エラー: ToFカメラのセットアップに失敗しました。自動撮影を中止します。")
        cap.release()
        return 0

    os.makedirs(SAVE_DIR, exist_ok=True)
    img_count = len(glob.glob(os.path.join(SAVE_DIR, 'webcam_*.png')))

    next_shot = time.monotonic() + interval_s

    # use the same window title and layout as manual capture (Mode 1)
    window_name = 'Webcam (Left) - ToF IR (Right) | [Space] to capture, [q] quit'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    capture_requested = {'val': False}
    def _on_mouse_auto(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            bx, by, bw, bh = 10, 10, 160, 50
            if bx <= x <= bx + bw and by <= y <= by + bh:
                capture_requested['val'] = True
    cv2.setMouseCallback(window_name, _on_mouse_auto)

    # criteria for cornerSubPix (same as manual collect_data)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    try:
        # create a separate large countdown window
        countdown_win = 'Countdown'
        cv2.namedWindow(countdown_win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(countdown_win, 400, 400)

        while True:
            # Get latest ToF and webcam frame
            tof_data_tuple = get_tof_data(fg)
            ret, frame_web = cap.read()
            if not ret:
                print("エラー: Webカメラからフレームを取得できません。自動撮影を終了します。")
                break

            # Prepare ToF visualization (IR) if available
            ir_gray = None
            display_tof = None
            if tof_data_tuple:
                x_img, y_img, z_img, ir_img = tof_data_tuple
                ir_gray = cv2.normalize(ir_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                display_tof = cv2.cvtColor(ir_gray, cv2.COLOR_GRAY2BGR)
            else:
                display_tof = np.zeros((TOF_HEIGHT, TOF_WIDTH, 3), dtype=np.uint8)

            # Prepare corner-drawn side-by-side view matching Mode 1
            h, w = frame_web.shape[:2]
            frame_web_display = frame_web.copy()

            # Checkerboard detection on both views (for readiness)
            gray_web = cv2.cvtColor(frame_web, cv2.COLOR_BGR2GRAY)
            ret_web_cb, corners_web = cv2.findChessboardCorners(gray_web, CHECKERBOARD, None)
            ret_tof_cb = False
            corners_tof = None
            if ir_gray is not None:
                ret_tof_cb, corners_tof = cv2.findChessboardCorners(ir_gray, CHECKERBOARD, None)

            # refine and draw corners like Mode 1
            if ret_web_cb:
                try:
                    corners_web_subpix = cv2.cornerSubPix(gray_web, corners_web, (11,11), (-1,-1), criteria)
                    cv2.drawChessboardCorners(frame_web_display, CHECKERBOARD, corners_web_subpix, ret_web_cb)
                except Exception:
                    cv2.drawChessboardCorners(frame_web_display, CHECKERBOARD, corners_web, ret_web_cb)

            display_tof_draw = display_tof.copy()
            if ret_tof_cb and corners_tof is not None:
                try:
                    corners_tof_subpix = cv2.cornerSubPix(ir_gray, corners_tof, (11,11), (-1,-1), criteria)
                    cv2.drawChessboardCorners(display_tof_draw, CHECKERBOARD, corners_tof_subpix, ret_tof_cb)
                except Exception:
                    cv2.drawChessboardCorners(display_tof_draw, CHECKERBOARD, corners_tof, ret_tof_cb)

            tof_resized = cv2.resize(display_tof_draw, (w, h))
            combined_view = np.hstack((frame_web_display, tof_resized))

            # Display readiness (use same messaging as Mode 1)
            if ret_web_cb and ret_tof_cb:
                cv2.putText(combined_view, "Ready to capture! Press [Space]", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                status = "Detecting: "
                if ret_web_cb:
                    status += "Webcam OK"
                if ret_tof_cb:
                    status += " | ToF OK"
                if not ret_web_cb and not ret_tof_cb:
                    status += "No checkerboard detected"
                cv2.putText(combined_view, status, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Countdown text
            now = time.monotonic()
            remaining = max(0.0, next_shot - now)
            countdown_text = f"Next in: {int(np.ceil(remaining))}s"
            cv2.putText(combined_view, countdown_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # draw capture button
            bx, by, bw, bh = 10, 10, 160, 50
            cv2.rectangle(combined_view, (bx, by), (bx + bw, by + bh), (0, 128, 255), -1)
            cv2.putText(combined_view, 'CAPTURE', (bx + 10, by + 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

            cv2.imshow(window_name, combined_view)

            # Big countdown window
            big = np.zeros((400, 400, 3), dtype=np.uint8)
            big_txt = str(int(np.ceil(remaining)))
            # center the text
            font_scale = 6
            thickness = 8
            ((txt_w, txt_h), _) = cv2.getTextSize(big_txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            org = ((big.shape[1] - txt_w) // 2, (big.shape[0] + txt_h) // 2)
            cv2.putText(big, big_txt, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
            cv2.imshow(countdown_win, big)

            # Capture when time reached AND checkerboard present on both
            if now >= next_shot:
                if tof_data_tuple and ret_web_cb and ret_tof_cb:
                    img_count += 1
                    web_path = os.path.join(SAVE_DIR, f'webcam_{img_count:03d}.png')
                    tof_path = os.path.join(SAVE_DIR, f'tof_{img_count:03d}.npz')

                    cv2.imwrite(web_path, frame_web)
                    np.savez_compressed(tof_path, x=x_img, y=y_img, z=z_img, ir=ir_img)

                    print(f"自動撮影: 保存しました #{img_count} -> {web_path}, {os.path.basename(tof_path)}")

                    # brief visual feedback
                    feedback = combined_view.copy()
                    cv2.putText(feedback, f"SAVED #{img_count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
                    cv2.imshow(window_name, feedback)
                    cv2.imshow(countdown_win, np.zeros((400,400,3), dtype=np.uint8))
                    cv2.waitKey(300)
                else:
                    # skip save and print reason
                    reason = 'ToF missing' if not tof_data_tuple else 'checkerboard not detected on both'
                    print(f"自動撮影: 保存をスキップしました ({reason})")

                next_shot = now + interval_s

            # Manual click-based capture (immediate) when button pressed
            if capture_requested.get('val', False):
                capture_requested['val'] = False
                if tof_data_tuple and ret_web_cb and ret_tof_cb:
                    img_count += 1
                    web_path = os.path.join(SAVE_DIR, f'webcam_{img_count:03d}.png')
                    tof_path = os.path.join(SAVE_DIR, f'tof_{img_count:03d}.npz')

                    cv2.imwrite(web_path, frame_web)
                    np.savez_compressed(tof_path, x=x_img, y=y_img, z=z_img, ir=ir_img)

                    print(f"手動クリック保存: #{img_count} -> {web_path}, {os.path.basename(tof_path)}")
                    feedback = combined_view.copy()
                    cv2.putText(feedback, f"SAVED #{img_count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
                    cv2.imshow(window_name, feedback)
                    cv2.waitKey(300)
                else:
                    print("手動クリック: 保存条件を満たしていないためスキップしました (ToFまたはチェッカーボード検出不足)")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        if fg:
            try:
                fg.stop()
            except Exception:
                pass
        try:
            cv2.destroyWindow(countdown_win)
        except Exception:
            pass
        cv2.destroyAllWindows()

    print(f"\n自動撮影を終了しました。合計 {img_count} 枚を保存しました。")
    return img_count

# --- ▲▲▲ 自動撮影モード ここまで ▲▲▲ ---

# --- ▼▼▼【改良】calculate_extrinsics をバグ修正版に置き換え ▼▼▼ ---
def calculate_extrinsics(cam_mat, dist_coeffs):
    print("\n--- [ステップ 3 改] 外部パラメータ計算 (RANSAC & Unit Check) ---")
    w_files = sorted(glob.glob(os.path.join(SAVE_DIR, 'webcam_*.png')))
    t_files = sorted(glob.glob(os.path.join(SAVE_DIR, 'tof_*.npz')))

    if not w_files or len(w_files) != len(t_files):
        print("データ不足または不一致。")
        return None, None

    per_frame_pairs = []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for wf, tf in zip(w_files, t_files):
        img = cv2.imread(wf)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret_w, corn_w = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        td = np.load(tf)
        # ToFデータの読み込み
        x_raw, y_raw, z_raw = td['x'], td['y'], td['z']
        ir_raw = td['ir']

        # --- 改善点3: 単位の自動判定 (Unit Sniffing) ---
        # Zの中央値が 10.0 より小さければメートル単位、大きければミリとみなす
        median_z = np.median(np.abs(z_raw[z_raw != 0]))
        scale_factor = 1.0
        if median_z > 10.0:
            scale_factor = 1000.0
            # print(f"Info: {os.path.basename(tf)} seems to be in mm. Scaling by 1000.")
        
        x_m = x_raw / scale_factor
        y_m = y_raw / scale_factor
        z_m = z_raw / scale_factor
        # ---------------------------------------------

        ir_g = cv2.normalize(ir_raw, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # --- 改善点1: コーナー検出前の微小なブラー（オプション）---
        # ir_g = cv2.GaussianBlur(ir_g, (3, 3), 0) 

        ret_t, corn_t = cv2.findChessboardCorners(ir_g, CHECKERBOARD, None)

        if ret_w and ret_t:
            corn_w = cv2.cornerSubPix(gray, corn_w, (11,11), (-1,-1), criteria)
            corn_t = cv2.cornerSubPix(ir_g, corn_t, (11,11), (-1,-1), criteria)

            H, W = ir_g.shape
            frame_obj = []
            frame_img = []

            for i in range(len(corn_t)):
                u, v = corn_t[i][0]
                u0, v0 = int(u), int(v)
                if 0 <= u0 < W-1 and 0 <= v0 < H-1:
                    dx, dy = u - u0, v - v0

                    # 補間 (単位統一済みデータを使用)
                    x_val = (1-dx)*(1-dy)*x_m[v0,u0] + dx*(1-dy)*x_m[v0,u0+1] + \
                            (1-dx)*dy*x_m[v0+1,u0] + dx*dy*x_m[v0+1,u0+1]
                    y_val = (1-dx)*(1-dy)*y_m[v0,u0] + dx*(1-dy)*y_m[v0,u0+1] + \
                            (1-dx)*dy*y_m[v0+1,u0] + dx*dy*y_m[v0+1,u0+1]
                    z_val = (1-dx)*(1-dy)*z_m[v0,u0] + dx*(1-dy)*z_m[v0,u0+1] + \
                            (1-dx)*dy*z_m[v0+1,u0] + dx*dy*z_m[v0+1,u0+1]

                    if z_val > 0.1:
                        frame_obj.append([x_val, y_val, z_val])
                        frame_img.append(corn_w[i][0])

            if len(frame_obj) >= 30:
                per_frame_pairs.append((np.array(frame_obj, dtype=np.float32),
                                         np.array(frame_img, dtype=np.float32)))

    if not per_frame_pairs:
        print("有効な対応点が不足しています。")
        return None, None

    # --- フレーム単位の品質チェック ---
    good_obj_sets, good_img_sets = [], []
    kept_frames = 0
    rejected_frames = 0
    for idx, (obj_set, img_set) in enumerate(per_frame_pairs):
        success, rvec_tmp, tvec_tmp = cv2.solvePnP(
            obj_set, img_set, cam_mat, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            rejected_frames += 1
            continue

        proj_tmp, _ = cv2.projectPoints(obj_set, rvec_tmp, tvec_tmp, cam_mat, dist_coeffs)
        err = np.linalg.norm(img_set - proj_tmp.reshape(-1, 2), axis=1)
        mean_err = float(np.mean(err))
        t_norm = float(np.linalg.norm(tvec_tmp))

        if mean_err > 1.5 or t_norm > 0.5:
            rejected_frames += 1
            continue

        good_obj_sets.append(obj_set)
        good_img_sets.append(img_set)
        kept_frames += 1

    print(f"Frame quality gate: kept {kept_frames}, rejected {rejected_frames}")

    if not good_obj_sets:
        print("品質基準を満たすフレームがありません。撮影データを見直してください。")
        return None, None

    obj_pts = np.concatenate(good_obj_sets, axis=0).astype(np.float32)
    img_pts = np.concatenate(good_img_sets, axis=0).astype(np.float32)
    print(f"収集された全点数: {len(obj_pts)}")

    # --- 改善点4: RANSACによる強力な外れ値除去 ---
    # RANSACで初期推定とInlierの選別を行う
    ret, rvec_init, tvec_init, inliers = cv2.solvePnPRansac(
        obj_pts, img_pts, cam_mat, dist_coeffs,
        reprojectionError=3.0,  # 許容誤差(px)。3.0px以上ズレたら外れ値とみなす
        iterationsCount=1000,
        flags=cv2.SOLVEPNP_EPNP
    )

    if not ret:
        print("RANSACによる推定に失敗しました。")
        return None, None

    print(f"RANSAC Inliers: {len(inliers)} / {len(obj_pts)} ({len(inliers)/len(obj_pts)*100:.1f}%)")
    
    # 選別された Inliers だけを使って、高精度な Iterative 法で再計算
    # RANSACの結果はあくまで初期値として利用
    obj_inliers = obj_pts[inliers.flatten()]
    img_inliers = img_pts[inliers.flatten()]

    ret, rvec_final, tvec_final = cv2.solvePnP(
        obj_inliers, img_inliers, cam_mat, dist_coeffs,
        rvec=rvec_init, tvec=tvec_init,
        useExtrinsicGuess=True, # RANSACの結果を初期値として使う
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    # ------------------------------------------------

    # 最終誤差確認
    proj_p, _ = cv2.projectPoints(obj_inliers, rvec_final, tvec_final, cam_mat, dist_coeffs)
    err = np.linalg.norm(img_inliers.reshape(-1,2) - proj_p.reshape(-1,2), axis=1)
    final_err = np.mean(err)

    print(f"★最終平均再投影誤差 (RANSAC後): {final_err:.4f} px")
    
    R, _ = cv2.Rodrigues(rvec_final)
    print("Result R:\n", R)
    print("Result T (m):\n", tvec_final)
    
    np.savez(EXTRINSICS_SAVE_PATH, rvec=rvec_final, tvec=tvec_final, R_matrix=R)
    return rvec_final, tvec_final
    print("\n--- [ステップ 3/5 & 4/5] 外部パラメータの計算 ---")
    all_object_points, all_image_points = [], []
    image_files = sorted(glob.glob(os.path.join(SAVE_DIR, 'webcam_*.png')))
    tof_files = sorted(glob.glob(os.path.join(SAVE_DIR, 'tof_*.npz')))
    
    if len(image_files) != len(tof_files) or not image_files:
        print("エラー: Webカメラ画像とToFデータの数が一致しないか、データがありません。")
        return None, None
        
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    successful_pairs = 0
    
    # --- 1. 元データの収集 ---
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
            
            (TOF_HEIGHT_ACTUAL, TOF_WIDTH_ACTUAL) = ir_gray.shape
            
            for j in range(len(corners_tof_subpix)):
                tof_corner = corners_tof_subpix[j][0]
                web_corner = corners_web_subpix[j]
                
                # --- サブピクセル補間 (変更なし) ---
                u_sub, v_sub = tof_corner[0], tof_corner[1]
                u0, u1, v0, v1 = int(np.floor(u_sub)), int(np.floor(u_sub) + 1), int(np.floor(v_sub)), int(np.floor(v_sub) + 1)

                if not (0 <= u0 and u1 < TOF_WIDTH_ACTUAL and 0 <= v0 and v1 < TOF_HEIGHT_ACTUAL):
                    continue 

                wx1, wx0 = u_sub - u0, 1.0 - (u_sub - u0)
                wy1, wy0 = v_sub - v0, 1.0 - (v_sub - v0)
                
                # (x_val, y_val, z_val) は mm 単位
                x_val = (wx0 * wy0 * tof_data['x'][v0, u0] + wx1 * wy0 * tof_data['x'][v0, u1] + 
                         wx0 * wy1 * tof_data['x'][v1, u0] + wx1 * wy1 * tof_data['x'][v1, u1])
                y_val = (wx0 * wy0 * tof_data['y'][v0, u0] + wx1 * wy0 * tof_data['y'][v0, u1] + 
                         wx0 * wy1 * tof_data['y'][v1, u0] + wx1 * wy1 * tof_data['y'][v1, u1])
                z_val = (wx0 * wy0 * tof_data['z'][v0, u0] + wx1 * wy0 * tof_data['z'][v0, u1] + 
                         wx0 * wy1 * tof_data['z'][v1, u0] + wx1 * wy1 * tof_data['z'][v1, u1])
                # --- 補間ここまで ---

                # --- 単位判定: 50より大きい値はミリメートル想定 → メートルへ変換 ---
                if abs(z_val) > 50.0 or abs(y_val) > 50.0 or abs(x_val) > 50.0:
                    x_m = x_val / 1000.0
                    y_m = y_val / 1000.0
                    z_m = z_val / 1000.0
                else:
                    x_m = x_val
                    y_m = y_val
                    z_m = z_val

                # ★★★ 修正: Z軸は反転させず、そのまま利用 ★★★
                x_obj = x_m
                y_obj = y_m
                z_obj = z_m 
                # ★★★★★★★★★★★★★★★★★★★★★

                # 正のZを利用したフィルタリング (メートル単位)
                if np.isfinite(z_obj) and z_obj > 0.1: # 10cm以上離れている
                    all_object_points.append([x_obj, y_obj, z_obj])
                    all_image_points.append(web_corner)
                        
    # --- 2. solvePnPの実行 (総当たりロジックを削除) ---
    print(f"\nコーナー検出に成功したペア数: {successful_pairs}")
    
    if len(all_object_points) < 20:
        print(f"エラー: 計算に十分な対応点が得られませんでした。(取得: {len(all_object_points)}点)")
        return None, None
        
    obj_pts = np.array(all_object_points, dtype=np.float32)
    img_pts = np.array(all_image_points, dtype=np.float32)
    print(f"全てのペアから収集した有効な対応点の総数: {len(obj_pts)}")

    # --- 2b. 事前トリミング: 仮推定で外れ値除去 ---
    print("\n一次solvePnPで仮の外部パラメータを推定し外れ値除去します...")
    prelim_success, prelim_rvec, prelim_tvec = cv2.solvePnP(
        obj_pts, img_pts, webcam_cam_matrix, webcam_dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not prelim_success:
        print("エラー: 一次 solvePnP に失敗しました。")
        return None, None

    reproject_prelim, _ = cv2.projectPoints(obj_pts, prelim_rvec, prelim_tvec,
                                            webcam_cam_matrix, webcam_dist_coeffs)
    diff_prelim = img_pts.reshape(-1,2) - reproject_prelim.reshape(-1,2)
    errors_prelim = np.sqrt(diff_prelim[:,0]**2 + diff_prelim[:,1]**2)
    mean_pre = np.mean(errors_prelim)
    med_pre = np.median(errors_prelim)
    print(f"一次推定 誤差: mean={mean_pre:.3f}px median={med_pre:.3f}px")

    # しきい値: 動的 (median * 2.5) を上限にしつつ最大 3.0px
    dynamic_thresh = min(3.0, med_pre * 2.5)
    if dynamic_thresh < 1.2: # あまりに低すぎる場合は下限を設ける
        dynamic_thresh = 1.2
    keep_mask = errors_prelim <= dynamic_thresh
    kept_count = int(np.count_nonzero(keep_mask))
    total_count = len(errors_prelim)
    print(f"外れ値トリミング: しきい値={dynamic_thresh:.2f}px 保持={kept_count}/{total_count} ({100*kept_count/total_count:.1f}%)")

    # 最低保持数を確保 (失敗回避) - 例えば 50点未満ならトリミング不適用でそのまま
    if kept_count < 50:
        print("保持点が少ないためトリミングをスキップし全点使用します。")
        trimmed_obj = obj_pts
        trimmed_img = img_pts
    else:
        trimmed_obj = obj_pts[keep_mask]
        trimmed_img = img_pts[keep_mask]

    # --- 2c. 本番 solvePnP （トリミング後） ---
    success, rvec_final, tvec_final = cv2.solvePnP(
        trimmed_obj, trimmed_img, webcam_cam_matrix, webcam_dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        print("\nエラー: トリミング後 solvePnP の計算に失敗しました。")
        return None, None

    # --- 3. 結果の出力と保存 ---
    print("\n✅ 外部パラメータ計算成功！（軽量トリミング適用）")
    
    # --- 再投影誤差の計算 ---
    reprojected_img_pts, _ = cv2.projectPoints(
        obj_pts, rvec_final, tvec_final, webcam_cam_matrix, webcam_dist_coeffs
    )
    error_diff = img_pts.reshape(-1, 2) - reprojected_img_pts.reshape(-1, 2)
    mean_error_final = np.mean(np.sqrt(error_diff[:, 0]**2 + error_diff[:, 1]**2))

    print(f"----------------------------------------")
    print(f"📊 平均再投影誤差 (Mean Reprojection Error): {mean_error_final:.4f} ピクセル")
    
    if mean_error_final <= 1.0: print("   (非常に良好なキャリブレーションです)")
    elif mean_error_final <= 2.0: print("   (良好なキャリブレーションです)")
    else: print(f"   (誤差が大きめです。データ収集のやり直しを推奨します)")
    print(f"----------------------------------------")
    
    R_matrix, _ = cv2.Rodrigues(rvec_final)
    print("計算された外部パラメータ:")
    # 出力: 回転ベクトル (Rodrigues 表現)、回転行列、並進ベクトル
    print("Rotation Vector (rvec) [Rodrigues representation]:")
    print(rvec_final.reshape(-1))
    print("\nRotation Matrix (R_matrix):")
    print(R_matrix)
    print("\nTranslation Vector (tvec) [Meters]:") # 単位がメートルであることを明記
    print(tvec_final)
    print("----------------------------------------")
    
    # 保存時には rvec, tvec に加えて回転行列も保存しておくと復元が容易です
    np.savez(EXTRINSICS_SAVE_PATH, rvec=rvec_final, R_matrix=R_matrix, tvec=tvec_final)
    print(f"結果を {EXTRINSICS_SAVE_PATH} に保存しました。 (rvec, R_matrix, tvec)")
    
    # Print copy-pasteable numpy code for quick use in mode 3 or other scripts
    try:
        _print_extrinsics_as_code(R_matrix, tvec_final)
    except Exception:
        pass

    return rvec_final, tvec_final
# --- ▲▲▲ calculate_extrinsics 置き換えここまで ▲▲▲ ---


def _print_extrinsics_as_code(R, tvec):
    """Print R and tvec as a copy-pasteable numpy code block to the terminal."""
    if R is None or tvec is None:
        return
    R = np.asarray(R, dtype=float)
    t = np.asarray(tvec, dtype=float).reshape((3,))

    print("\n# --- copy-pasteable extrinsics (numpy) ---")
    print("R_matrix = np.array([")
    for row in R:
        vals = ", ".join([f"{v:.8f}" for v in row])
        print(f"    [{vals}],")
    print("])")
    print("# compute Rodrigues rotation vector from the matrix")
    print("rvec, _ = cv2.Rodrigues(R_matrix)")
    print("\n# keep existing translation vector (meters)")
    print("tvec = np.array([")
    for v in t:
        print(f"    [{v:.8f}],")
    print("])\n")


# --- ▼▼▼【改良】verify_projection をバグ修正版に置き換え ▼▼▼ ---
def verify_projection(webcam_cam_matrix, webcam_dist_coeffs, rvec, tvec):
    print("\n--- [ステップ 5/5] 評価モード (距離フィルタ調整機能付き) ---")
    print("ToF点群をWebカメラ映像に投影します。[q]で終了")
    print(f"INFO: 点群は深度に応じて色付けされます (近:{MIN_DEPTH_VIZ}m [赤] -> 遠:{MAX_DEPTH_VIZ}m [青])")
    print("INFO: スライダーでX軸フィルタ範囲を調整できます")
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    
    # --- 修正: ifm3dpy でセットアップ ---
    device, fg = setup_tof_grabber()
    if not fg:
        print("ToFカメラに接続できないため、評価をスキップします。")
        cap.release()
        return
    # ---------------------------------

    # DEBUG: 現在のカメラ解像度を確認（最初のフレームを一度読み出して shape を表示）
    try:
        ret_check, frame_check = cap.read()
        if ret_check:
            print(f"[DEBUG] Webcam actual frame shape (verify_projection): {frame_check.shape} (H,W,CH)")
        else:
            print("[DEBUG] Warning: could not read test frame after opening webcam (verify_projection).")
    except Exception as e:
        print(f"[DEBUG] Exception while checking webcam frame in verify_projection(): {e}")

    # --- スライダー用の変数 (0.01m単位で調整) ---
    z_min_cm = 30  # 0.3m * 100
    z_max_cm = 100 # 1.0m * 100
    
    # ウィンドウとスライダーを作成
    window_name = 'Verification: Projected ToF points on Webcam'
    cv2.namedWindow(window_name)
    cv2.createTrackbar('Z Min [cm]', window_name, z_min_cm, 300, lambda x: None)
    cv2.createTrackbar('Z Max [cm]', window_name, z_max_cm, 300, lambda x: None)

    while True:
        # --- スライダーから現在の値を取得 ---
        z_min_cm = cv2.getTrackbarPos('Z Min [cm]', window_name)
        z_max_cm = cv2.getTrackbarPos('Z Max [cm]', window_name)
        
        z_min_m = z_min_cm / 100.0
        z_max_m = z_max_cm / 100.0
        
        # 最小値が最大値を超えないように調整
        if z_min_m > z_max_m:
            z_min_m, z_max_m = z_max_m, z_min_m
        
        # --- 修正: ifm3dpy でToF取得 ---
        tof_data_tuple = get_tof_data(fg)
        
        ret, frame = cap.read()
        if not ret: break
        


        projected_frame = frame.copy()
        
        # プレースホルダ (ToFが取得できなかった場合用)
        depth_color_map = np.zeros((TOF_HEIGHT, TOF_WIDTH, 3), dtype=np.uint8)

        if tof_data_tuple:
            x_img, y_img, z_img, ir_img = tof_data_tuple # 既にメートル単位
            
            # --- 修正: 取得したToFデータのサイズを取得 ---
            (TOF_HEIGHT_ACTUAL, TOF_WIDTH_ACTUAL) = ir_img.shape
            depth_color_map = np.zeros((TOF_HEIGHT_ACTUAL, TOF_WIDTH_ACTUAL, 3), dtype=np.uint8)
            # -------------------------------------------

            # ★★★ get_tof_data()で既にメートル変換済み。ここでは不要 ★★★
            x_img_m = x_img
            y_img_m = y_img
            z_img_m = z_img
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            
            # --- 1. ToF深度マップのウィンドウを作成 (フィルタ無効) ---
            depth_map_viz = z_img_m.copy()
            depth_range = MAX_DEPTH_VIZ - MIN_DEPTH_VIZ
            if depth_range <= 0: depth_range = 1.0
            
            # Z値 (例: 0.5) を [0, 1] の範囲に正規化
            normalized_depth = (depth_map_viz - MIN_DEPTH_VIZ) / depth_range
            normalized_depth_clipped = np.clip(normalized_depth, 0, 1)
            
            depth_gray_8u = (normalized_depth_clipped * 255).astype(np.uint8)
            depth_color_map = cv2.applyColorMap(depth_gray_8u, cv2.COLORMAP_JET)

            # --- 2. Webカメラに点群を投影 (Z軸のみスライダーでフィルタ) ---
            points_3d = np.stack((x_img_m.flatten(), y_img_m.flatten(), z_img_m.flatten()), axis=-1)

            z_values = points_3d[:, 2]
            
            # Z軸のみでフィルタ（正のZ値）
            mask = (z_values >= z_min_m) & (z_values <= z_max_m)
            filtered_points_3d = points_3d[mask]
            
            filtered_z_values = filtered_points_3d[:, 2]

            if len(filtered_points_3d) > 0:
                projected_points_2d, _ = cv2.projectPoints(filtered_points_3d, rvec, tvec, webcam_cam_matrix, webcam_dist_coeffs)
                
                # Z値を [0, 1] に正規化して色付け
                norm_depth = (filtered_z_values - MIN_DEPTH_VIZ) / depth_range
                norm_depth_clipped = np.clip(norm_depth, 0, 1)
                gray_8u = (norm_depth_clipped * 255).astype(np.uint8)
                point_colors = cv2.applyColorMap(gray_8u, cv2.COLORMAP_JET)

                for i, p in enumerate(projected_points_2d):
                    u, v = int(p[0][0]), int(p[0][1])
                    if 0 <= u < projected_frame.shape[1] and 0 <= v < projected_frame.shape[0]:
                        color = tuple(point_colors[i][0].tolist())
                        cv2.circle(projected_frame, (u, v), 1, color, -1)
        
        # フィルタ範囲とポイント数を画面に表示
        filter_text1 = f"Z Depth Filter: {z_min_m:.2f}m - {z_max_m:.2f}m"
        filter_text2 = f"Points: {len(filtered_points_3d) if tof_data_tuple else 0}"
        cv2.putText(projected_frame, filter_text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(projected_frame, filter_text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
        cv2.imshow(window_name, projected_frame)
        cv2.imshow('ToF Depth Map', depth_color_map)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- 修正: クリーンアップ ---
    cap.release()
    if fg: 
        fg.stop()
    cv2.destroyAllWindows()
# --- ▲▲▲ verify_projection 置き換えここまで ▲▲▲ ---

# --- メイン実行部分 (修正なし) ---
if __name__ == "__main__":
    print("実行するモードを選択してください:")
    print(" 1: キャリブレーションを最初からすべて行う (データ収集 -> 計算 -> 評価)")
    print(" 2: 保存済みの 'extrinsics.npz' を使って評価モードのみ実行する (モード3相当の調整を適用)")
    print(" 3: コードに直接書き込まれた行列を使って評価モードのみ実行する")
    print(" 4: 自動撮影モード (ウィンドウにカウントダウン、指定間隔で自動保存)")
    mode = input("選択 (1, 2, 3 or 4): ")

    w_matrix, w_dist = load_intrinsics()
    if w_matrix is None:
        print("Webカメラの内部パラメータが読み込めないため、プログラムを終了します。")

    elif mode == '1':
        collect_data()
        rvec, tvec = calculate_extrinsics(w_matrix, w_dist)
        if rvec is not None and tvec is not None:
            print("\n計算直後のパラメータで評価モードを開始します。")
            print("微調整が必要な場合は、モード3で値を調整するか、モード2の調整ロジックを変更してください。")
            verify_projection(w_matrix, w_dist, rvec, tvec)

    elif mode == '2':
        try:
            print(f"\n'{EXTRINSICS_SAVE_PATH}' から外部パラメータを読み込みます。")
            data = np.load(EXTRINSICS_SAVE_PATH)
            
            if 'rvec' in data and 'tvec' in data:
                rvec = data['rvec']
                tvec_raw = data['tvec'] 

                # 回転行列が保存されていればそれを使い、なければ rvec から算出
                if 'R_matrix' in data:
                    R_loaded = data['R_matrix']
                else:
                    R_loaded, _ = cv2.Rodrigues(rvec)

                print("読み込み成功。外部パラメータ (調整前):")
                print("Rotation Vector (rvec):")
                print(rvec.reshape(-1))
                print("Rotation Matrix (R):")
                print(R_loaded)
                print("Translation Vector (tvec) [Meters]:")
                print(tvec_raw)

                print("\n読み込んだtvecに手動調整（モード3相当）を適用します。")
                tvec_adjusted = tvec_raw.copy()
                if APPLY_TVEC_MIRROR:
                    tvec_adjusted[0][0] = tvec_raw[0][0] * -1.0
                    tvec_adjusted[1][0] = tvec_raw[1][0]
                else:
                    print("INFO: APPLY_TVEC_MIRROR=False のため、生のtvecを使用しています。")

                print(f"調整後 tvec:")
                print(tvec_adjusted)

                print("\n読み込み＆調整成功。評価モードを開始します。")
                # also print a copy-pasteable numpy block for convenience
                try:
                    _print_extrinsics_as_code(R_loaded, tvec_adjusted)
                except Exception:
                    pass
                verify_projection(w_matrix, w_dist, rvec, tvec_adjusted) 
            
            else:
                print(f"エラー: {EXTRINSICS_SAVE_PATH} に 'rvec' または 'tvec' が含まれていません。")
        
        except FileNotFoundError:
            print(f"エラー: {EXTRINSICS_SAVE_PATH} が見つかりません。")
            print("先にモード'1'を実行して、キャリブレーションファイルを生成してください。")
        except Exception as e:
            print(f"ファイルの読み込み中、または調整中に予期せぬエラーが発生しました: {e}")
            traceback.print_exc()

    elif mode == '3':
        try:
            print("\nコードに直接記述された外部パラメータを使用します。")
            # ↓↓↓↓ ここに 'mode 1' で出力された値をコピー＆ペーストして調整してください ↓↓↓↓
            # Rotation matrix taken from printed output (copy-pasteable)
            R_matrix = np.array([
                [ 0.99951790,  0.02917936,  0.01060768],
                [-0.02924543,  0.99955348,  0.00612796],
                [-0.01042413, -0.00643524,  0.99992496],
            ])
            # compute Rodrigues rotation vector from the matrix
            rvec, _ = cv2.Rodrigues(R_matrix)

            # keep existing translation vector (meters)
            tvec = np.array([
                [-0.07112388],
                [ 0.04808378],
                [ 0.03565554],
            ])
            # ↑↑↑↑ この行列を使って投影位置を調整してください ↑↑↑↑
            print("読み込み成功。使用する外部パラメータ:")
            print("Rotation Matrix (R_matrix):")
            print(R_matrix)
            print("Rotation Vector (rvec):")
            print(rvec.reshape(-1))
            print("Translation Vector (tvec) [Meters]:")
            print(tvec)
            print("評価モードを開始します。")
            verify_projection(w_matrix, w_dist, rvec, tvec)
        except Exception as e:
            print(f"エラー: 指定された行列の形式が正しくない可能性があります: {e}")

    elif mode == '4':
        # 自動撮影モード: interval 3秒で自動保存
        saved = auto_capture_mode(interval_s=6)
        if saved > 0:
            print("自動撮影でデータが保存されました。外部パラメータを計算します...")
            rvec, tvec = calculate_extrinsics(w_matrix, w_dist)
            if rvec is not None and tvec is not None:
                verify_projection(w_matrix, w_dist, rvec, tvec)
        else:
            print("自動撮影でデータが保存されなかったため、終了します。")

    else:
        print("無効な選択です。'1', '2'または'3'を入力してください。")

    print("\nプログラムを終了します。")