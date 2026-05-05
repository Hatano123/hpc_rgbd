"""Simple real-time RGB detection using a pretrained 3-channel YOLO model."""
from __future__ import annotations

import argparse
import time
import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolov8n.pt", help="Path to a 3ch pretrained YOLO model")
    parser.add_argument("--source", default=0, help="Camera index or video path (default webcam 0)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", default=None, help="cuda device string or cpu")
    parser.add_argument("--show-fps", action="store_true", help="Overlay FPS on the frame")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {args.source}")

    cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            source=frame,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )
        plotted = results[0].plot()

        if args.show_fps:
            now = time.time()
            fps = 1.0 / max(1e-6, (now - prev_time))
            prev_time = now
            cv2.putText(plotted, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("Detections", plotted)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
