"""
Automated Pipeline Verification Test Suite
==========================================
Validates ONNX model integrity, numerical scales, layer sensitivity metrics,
test dataset isolation, and edge classifier execution.
"""

import json
import os
import unittest
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class TestEdgeQuantizationPipeline(unittest.TestCase):
    def setUp(self):
        self.int8_path = os.path.join(REPO_ROOT, "quantized_model_int8.onnx")
        self.fp32_path = os.path.join(REPO_ROOT, "model_fp32.onnx")
        self.scales_path = os.path.join(REPO_ROOT, "quantization_scales_zp.json")
        self.audit_path = os.path.join(REPO_ROOT, "quantization_audit.json")
        self.data_npz_path = os.path.join(REPO_ROOT, "data", "mnist_test_10k.npz")
        self.esp_model_data = os.path.join(REPO_ROOT, "esp_quantize", "firmware", "main", "model_data.h")

    def test_model_files_exist(self):
        self.assertTrue(os.path.isfile(self.int8_path), "quantized_model_int8.onnx missing")
        self.assertTrue(os.path.isfile(self.fp32_path), "model_fp32.onnx missing")
        self.assertTrue(os.path.isfile(self.scales_path), "quantization_scales_zp.json missing")
        self.assertTrue(os.path.isfile(self.audit_path), "quantization_audit.json missing")
        self.assertTrue(os.path.isfile(self.data_npz_path), "mnist_test_10k.npz missing")
        self.assertTrue(os.path.isfile(self.esp_model_data), "esp_quantize model_data.h missing")

    @unittest.skipIf(ort is None, "onnxruntime not installed")
    def test_onnx_models_loadable(self):
        # INT8 model session
        sess_int8 = ort.InferenceSession(self.int8_path, providers=["CPUExecutionProvider"])
        inp_int8 = sess_int8.get_inputs()[0]
        out_int8 = sess_int8.get_outputs()[0]
        self.assertEqual(inp_int8.shape, [1, 1, 28, 28])
        self.assertEqual(out_int8.shape, [1, 10])

        # FP32 model session
        sess_fp32 = ort.InferenceSession(self.fp32_path, providers=["CPUExecutionProvider"])
        inp_fp32 = sess_fp32.get_inputs()[0]
        out_fp32 = sess_fp32.get_outputs()[0]
        self.assertEqual(inp_fp32.shape, [1, 1, 28, 28])
        self.assertEqual(out_fp32.shape, [1, 10])

    def test_quantization_scales_validity(self):
        with open(self.scales_path, "r", encoding="utf-8") as f:
            scales = json.load(f)
        self.assertIn("quant", scales)
        self.assertIn("activation_scale", scales["quant"])
        self.assertIn("activation_zero_point", scales["quant"])
        self.assertAlmostEqual(scales["quant"]["activation_scale"], 0.02554, places=4)
        self.assertEqual(scales["quant"]["activation_zero_point"], 17)

    def test_quantization_audit_sqnr(self):
        with open(self.audit_path, "r", encoding="utf-8") as f:
            audit = json.load(f)
        self.assertGreater(len(audit), 0)
        for layer, metrics in audit.items():
            self.assertIn("sqnr_db", metrics)
            self.assertIn("cosine_similarity", metrics)
            # Standard quantization threshold
            self.assertGreaterEqual(metrics["sqnr_db"], 30.0, f"Layer {layer} SQNR below 30 dB")
            self.assertGreaterEqual(metrics["cosine_similarity"], 0.995, f"Layer {layer} cosine similarity low")

    def test_dataset_integrity(self):
        data = np.load(self.data_npz_path)
        images = data["images"]
        labels = data["labels"]
        self.assertEqual(images.shape, (10000, 28, 28))
        self.assertEqual(labels.shape, (10000,))
        self.assertEqual(images.dtype, np.uint8)

    @unittest.skipIf(ort is None, "onnxruntime not installed")
    def test_pi_edge_classifier(self):
        from pi_quantize.inference_pi import PiEdgeClassifier
        classifier = PiEdgeClassifier(model_path=self.int8_path)
        dummy_sample = os.path.join(REPO_ROOT, "test_images", "7", "img_00000.png")
        if os.path.isfile(dummy_sample):
            digit, conf, probs, dt_ms = classifier.predict(dummy_sample)
            self.assertEqual(digit, 7)
            self.assertGreater(conf, 90.0)
            self.assertLess(dt_ms, 50.0)


if __name__ == "__main__":
    unittest.main()
