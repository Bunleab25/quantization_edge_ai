"""
Raspberry Pi Production Inference Module
=======================================
Provides an optimized edge classifier class (PiEdgeClassifier) and CLI
for low-latency single-image and batch inference on Raspberry Pi.

Usage:
  # CLI: Predict single image
  python inference_pi.py --image ../test_images/7/img_00000.png

  # Python API:
  from pi_quantize.inference_pi import PiEdgeClassifier
  classifier = PiEdgeClassifier()
  digit, conf, probs = classifier.predict("test.png")
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except ImportError:
    print("Error: onnxruntime is required. Install via: pip install onnxruntime")
    sys.exit(1)


class PiEdgeClassifier:
    """
    High-performance edge inference session optimized for Raspberry Pi
    ARM Cortex-A72 / Cortex-A76 cores.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_threads: Optional[int] = None,
    ) -> None:
        if model_path is None:
            # Look for quantized model in common locations
            candidates = [
                "quantized_model_int8.onnx",
                os.path.join("..", "quantized_model_int8.onnx"),
                os.path.join(os.path.dirname(__file__), "..", "quantized_model_int8.onnx"),
                os.path.join(os.path.dirname(__file__), "quantized_model_int8.onnx"),
            ]
            for c in candidates:
                if os.path.isfile(c):
                    model_path = os.path.abspath(c)
                    break

        if model_path is None or not os.path.isfile(model_path):
            raise FileNotFoundError("quantized_model_int8.onnx could not be found.")

        self.model_path = model_path
        self.sess_options = ort.SessionOptions()
        self.sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        cpu_threads = num_threads or (os.cpu_count() or 4)
        self.sess_options.intra_op_num_threads = cpu_threads
        self.sess_options.inter_op_num_threads = 1

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=self.sess_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

        # Warmup session with dummy tensor
        dummy = np.zeros((1, 1, 28, 28), dtype=np.float32)
        self.session.run(None, {self.input_name: dummy})

    def preprocess_image(self, img_input: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Prepares 28x28 grayscale image normalized per MNIST dataset standard:
        (x / 255.0 - 0.1307) / 0.3081
        """
        if isinstance(img_input, str):
            if not os.path.isfile(img_input):
                raise FileNotFoundError(f"Image not found: {img_input}")
            img = Image.open(img_input).convert("L")
        elif isinstance(img_input, Image.Image):
            img = img_input.convert("L")
        elif isinstance(img_input, np.ndarray):
            if img_input.ndim == 2:
                img = Image.fromarray(img_input.astype(np.uint8), mode="L")
            else:
                img = Image.fromarray(img_input.astype(np.uint8)).convert("L")
        else:
            raise TypeError("Unsupported image input type.")

        if img.size != (28, 28):
            img = img.resize((28, 28), Image.Resampling.BILINEAR)

        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - 0.1307) / 0.3081
        tensor = arr.reshape(1, 1, 28, 28)
        return tensor

    def predict(
        self, img_input: Union[str, Image.Image, np.ndarray]
    ) -> Tuple[int, float, List[float], float]:
        """
        Runs inference on an image.
        
        Returns:
            Tuple of (predicted_digit, confidence_pct, all_probabilities, latency_ms)
        """
        tensor = self.preprocess_image(img_input)

        t0 = time.perf_counter()
        logits = self.session.run(None, {self.input_name: tensor})[0][0]
        dt_ms = (time.perf_counter() - t0) * 1000.0

        # Numerically stable softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = (exp_logits / np.sum(exp_logits)).tolist()

        predicted_digit = int(np.argmax(probs))
        confidence = probs[predicted_digit] * 100.0

        return predicted_digit, confidence, probs, dt_ms


def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi Edge Inference Classifier")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default=None, help="Path to ONNX model (optional)")
    parser.add_argument("--threads", type=int, default=None, help="Inference threads (optional)")
    args = parser.parse_args()

    classifier = PiEdgeClassifier(model_path=args.model, num_threads=args.threads)
    digit, conf, probs, dt_ms = classifier.predict(args.image)

    print("=" * 55)
    print("  RASPBERRY PI EDGE INFERENCE RESULT")
    print("=" * 55)
    print(f"  Input Image:       {args.image}")
    print(f"  Model Loaded:      {os.path.basename(classifier.model_path)}")
    print(f"  Predicted Digit:   {digit}")
    print(f"  Confidence:        {conf:.2f}%")
    print(f"  Inference Latency: {dt_ms:.3f} ms")
    print("-" * 55)
    print("  Top-3 Probabilities:")
    top3_indices = np.argsort(probs)[::-1][:3]
    for rank, idx in enumerate(top3_indices, start=1):
        print(f"    {rank}. Digit '{idx}': {probs[idx] * 100:.2f}%")
    print("=" * 55)


if __name__ == "__main__":
    main()
