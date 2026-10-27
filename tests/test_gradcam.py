"""
Test suite for Grad-CAM Explainable AI (XAI) and Model Inference.
Validates end-to-end model loading, forward pass inference, and
Grad-CAM heatmap/overlay generation using the trained CNN model.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path so modules can be imported from any working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Silence TensorFlow verbose logs during test execution
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import gradcam

MODEL_PATH = PROJECT_ROOT / "model" / "brain_tumor_model.keras"


def test_gradcam_pipeline():
    print(f"=== 1. Checking Model Path: {MODEL_PATH} ===")
    assert MODEL_PATH.exists(), f"Model file not found at: {MODEL_PATH}"
    print("Model file exists.")

    print("=== 2. Loading Keras Model ===")
    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    assert model is not None, "Model failed to load."
    layer_names = [l.name for l in model.layers]
    print(f"Model layers: {layer_names}")
    assert "conv2d_2" in layer_names, f"'conv2d_2' not in model layers: {layer_names}"
    print("Model loaded successfully.")

    print("=== 3. Creating Synthetic Test MRI Scan ===")
    rng = np.random.default_rng(42)
    synthetic_arr = rng.integers(40, 220, (224, 224, 3), dtype=np.uint8)
    test_pil = Image.fromarray(synthetic_arr)
    assert test_pil.size == (224, 224), "Test image size mismatch"
    print(f"Test image shape: {test_pil.size}")

    print("=== 4. Preprocessing Input ===")
    img = ImageOps.exif_transpose(test_pil).convert("RGB")
    resized = img.resize((224, 224))
    img_array = np.asarray(resized, dtype=np.float32) / 255.0
    input_tensor = np.expand_dims(img_array, axis=0)
    assert input_tensor.shape == (1, 224, 224, 3), f"Bad input shape: {input_tensor.shape}"
    print("Preprocessing completed successfully.")

    print("=== 5. Running Model Inference ===")
    raw_pred = float(np.asarray(model.predict(input_tensor, verbose=0)).reshape(-1)[0])
    result = "Tumor Detected" if raw_pred >= 0.5 else "No Tumor Detected"
    confidence = (raw_pred if raw_pred >= 0.5 else 1.0 - raw_pred) * 100.0
    print(f"Prediction Result: {result} ({confidence:.2f}% confidence, raw score: {raw_pred:.4f})")

    print("=== 6. Generating Grad-CAM Heatmap & Overlay ===")
    gc_result = gradcam.generate_gradcam(
        model=model,
        original_pil=test_pil,
        img_array=input_tensor,
        target_layer_name="conv2d_2",
        alpha=0.45,
    )

    assert gc_result["error"] is None, f"Grad-CAM generation failed with error: {gc_result['error']}"
    assert gc_result["heatmap_pil"] is not None, "Heatmap PIL image is None"
    assert gc_result["overlay_pil"] is not None, "Overlay PIL image is None"

    heatmap = gc_result["heatmap_pil"]
    overlay = gc_result["overlay_pil"]
    print(f"Heatmap dimensions: {heatmap.size}, mode: {heatmap.mode}")
    print(f"Overlay dimensions: {overlay.size}, mode: {overlay.mode}")

    print("\n===============================")
    print("[SUCCESS] ALL TESTS PASSED SUCCESSFULLY")
    print("===============================")


if __name__ == "__main__":
    test_gradcam_pipeline()
