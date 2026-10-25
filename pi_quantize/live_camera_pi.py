"""
Raspberry Pi Real-Time Camera Digit Recognition
================================================
Captures live frames from a USB webcam or Raspberry Pi Camera Module (CSI),
extracts handwritten digits inside an interactive Region-Of-Interest (ROI),
applies center-of-mass centering & thresholding, and runs real-time INT8 inference.

Usage:
  python live_camera_pi.py --camera 0
  python live_camera_pi.py --model ../quantized_model_int8.onnx
"""

import argparse
import os
import sys
import time
from typing import Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    print("Notice: OpenCV (cv2) is required for camera streaming. Install: pip install opencv-python")

from inference_pi import PiEdgeClassifier


def extract_and_preprocess_roi(roi_gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Binarizes ROI, isolates largest contour (digit stroke), computes center of mass,
    and scales into 28x28 grayscale matching the MNIST distribution.
    """
    # 1. Otsu thresholding or adaptive threshold
    blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0)
    # MNIST has black background (0) and white stroke (255)
    # Detect if paper is white with dark ink:
    mean_val = np.mean(blurred)
    if mean_val > 127:
        # Dark ink on bright paper -> invert
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        # White ink on dark background
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        blank = np.zeros((28, 28), dtype=np.float32)
        blank = (blank - 0.1307) / 0.3081
        return blank.reshape(1, 1, 28, 28), thresh

    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Filter out tiny noise
    if w < 10 or h < 10:
        blank = np.zeros((28, 28), dtype=np.float32)
        blank = (blank - 0.1307) / 0.3081
        return blank.reshape(1, 1, 28, 28), thresh

    digit_crop = thresh[y : y + h, x : x + w]

    # Resize preserving aspect ratio to fit inside 20x20 box (MNIST standard)
    if w > h:
        new_w = 20
        new_h = max(int(h * (20.0 / w)), 1)
    else:
        new_h = 20
        new_w = max(int(w * (20.0 / h)), 1)

    resized_digit = cv2.resize(digit_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Place in 28x28 canvas using center-of-mass
    canvas = np.zeros((28, 28), dtype=np.uint8)
    pad_top = (28 - new_h) // 2
    pad_left = (28 - new_w) // 2
    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized_digit

    # Shift by center of mass (standard LeCun MNIST protocol)
    moments = cv2.moments(canvas)
    if moments["m00"] > 0:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        dx = 14.0 - cx
        dy = 14.0 - cy
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        canvas = cv2.warpAffine(canvas, M, (28, 28), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Normalize
    norm = canvas.astype(np.float32) / 255.0
    norm = (norm - 0.1307) / 0.3081
    tensor = norm.reshape(1, 1, 28, 28)
    return tensor, canvas


def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi Camera Live MNIST Recognition")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--model", type=str, default=None, help="Path to ONNX model")
    parser.add_argument("--width", type=int, default=640, help="Camera frame width")
    parser.add_argument("--height", type=int, default=480, help="Camera frame height")
    args = parser.parse_args()

    if "cv2" not in sys.modules:
        print("Error: opencv-python is not installed.")
        sys.exit(1)

    print("Initializing PiEdgeClassifier...")
    classifier = PiEdgeClassifier(model_path=args.model)
    print(f"Model loaded: {os.path.basename(classifier.model_path)}")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f"Error: Unable to open camera {args.camera}")
        sys.exit(1)

    print("\nLive camera running. Press 'q' to exit.")
    roi_size = 200

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        # Center ROI box
        rx1 = (w - roi_size) // 2
        ry1 = (h - roi_size) // 2
        rx2 = rx1 + roi_size
        ry2 = ry1 + roi_size

        roi = frame[ry1:ry2, rx1:rx2]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        tensor, proc_preview = extract_and_preprocess_roi(roi_gray)

        # Run inference
        t0 = time.perf_counter()
        logits = classifier.session.run(None, {classifier.input_name: tensor})[0][0]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        pred_digit = int(np.argmax(probs))
        confidence = probs[pred_digit] * 100.0

        # Draw HUD overlays
        color = (0, 255, 0) if confidence > 80 else (0, 165, 255)
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, 2)
        cv2.putText(
            frame,
            f"Digit: {pred_digit} ({confidence:.1f}%)",
            (rx1, ry1 - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
        )
        cv2.putText(
            frame,
            f"Latency: {latency_ms:.2f} ms ({1000.0/max(latency_ms, 0.1):.0f} FPS)",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            "Place digit inside center green box | Press 'q' to quit",
            (15, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        # Show 28x28 debug preview in corner
        preview_bgr = cv2.cvtColor(cv2.resize(proc_preview, (84, 84), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
        frame[10:94, w - 94 : w - 10] = preview_bgr
        cv2.rectangle(frame, (w - 95, 9), (w - 9, 95), (255, 255, 255), 1)

        cv2.imshow("Raspberry Pi Edge AI - Digit Recognizer", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
