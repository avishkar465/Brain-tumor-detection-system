import io
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt


def _get_gradcam_heatmap(model, img_array, target_layer_name):
    """
    Compute Grad-CAM heatmap using a Keras 3 compatible layer-by-layer approach.

    Instead of building a sub-model (which fails on loaded Sequential models in Keras 3),
    we manually iterate the model layers inside a GradientTape. We capture the
    conv layer output and watch it, then compute gradients of the class score
    with respect to that intermediate tensor.

    Args:
        model:             Loaded Keras model.
        img_array:         Preprocessed batch tensor (1, H, W, C), values in [0, 1].
        target_layer_name: Name of the target convolutional layer.

    Returns:
        heatmap: float32 ndarray in [0, 1], shape = (h, w).
    """
    # Warm-up call: needed so Keras 3 initialises layer states
    dummy = tf.zeros_like(img_array)
    _ = model(dummy, training=False)

    # Verify the target layer exists
    layer_names = [l.name for l in model.layers]
    if target_layer_name not in layer_names:
        raise ValueError(
            f"Layer '{target_layer_name}' not found. Available: {layer_names}"
        )

    input_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        # Manual layer-by-layer forward pass
        x = input_tensor
        conv_out = None
        for layer in model.layers:
            x = layer(x, training=False)
            if layer.name == target_layer_name:
                conv_out = x          # capture conv activations
                tape.watch(conv_out)  # register them for gradient tracking
        preds = x
        # Binary sigmoid: raw output is the class score
        class_score = preds[:, 0]

    if conv_out is None:
        raise RuntimeError(f"Layer '{target_layer_name}' was not reached during forward pass.")

    # Gradients of class score w.r.t. conv layer activations; shape (1, h, w, C)
    grads = tape.gradient(class_score, conv_out)

    if grads is None:
        raise RuntimeError(
            "Gradient computation returned None. "
            "Ensure the target layer is in the computational graph."
        )

    # Global-average-pool gradients -> per-channel importance; shape (C,)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature maps by importance; shape (h, w, C)
    weighted = conv_out[0] * pooled_grads[tf.newaxis, tf.newaxis, :]

    # Sum over channels + ReLU (only positive contributions); shape (h, w)
    heatmap = tf.reduce_sum(weighted, axis=-1)
    heatmap = tf.nn.relu(heatmap).numpy()

    # Normalize to [0, 1]
    hmax = heatmap.max()
    if hmax > 0:
        heatmap = heatmap / hmax

    return heatmap.astype(np.float32)


def _get_colormap(name="jet"):
    """Get colormap supporting modern Matplotlib (>=3.7, 3.8, 3.9+) and legacy versions."""
    if hasattr(mpl, "colormaps"):
        return mpl.colormaps[name]
    return plt.get_cmap(name)


def _colorize_heatmap(heatmap):
    """Apply JET colormap to a [0,1] heatmap; return RGB PIL Image."""
    colormap = _get_colormap("jet")
    colored_rgba = colormap(heatmap)
    colored_rgb = (colored_rgba[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(colored_rgb, mode="RGB")


def _overlay_heatmap(original_pil, heatmap_pil, alpha=0.45):
    """Alpha-blend the colorized heatmap over the original MRI image."""
    orig_rgb = original_pil.convert("RGB")
    heat_rgb = heatmap_pil.resize(orig_rgb.size, Image.LANCZOS)
    orig_arr = np.array(orig_rgb, dtype=np.float32)
    heat_arr = np.array(heat_rgb, dtype=np.float32)
    blended = (orig_arr * (1.0 - alpha) + heat_arr * alpha).clip(0, 255).astype(np.uint8)
    return Image.fromarray(blended, mode="RGB")


def generate_gradcam(model, original_pil, img_array, target_layer_name="conv2d_2", alpha=0.45):
    """
    Full Grad-CAM pipeline: compute heatmap, colorize, overlay on original image.

    Compatible with Keras 3 loaded Sequential models.

    Args:
        model:             Trained Keras model.
        original_pil:      Original (un-preprocessed) PIL Image.
        img_array:         Preprocessed batch tensor (1, 224, 224, 3), values in [0, 1].
        target_layer_name: Conv layer to visualize. Default: "conv2d_2" (last conv layer).
        alpha:             Heatmap opacity fraction (0 = invisible, 1 = fully opaque).

    Returns:
        dict with keys:
            "heatmap_pil" -- Pure JET colorized heatmap (PIL Image, RGB)
            "overlay_pil" -- Original MRI with heatmap blended on top (PIL Image, RGB)
            "error"       -- str error message, or None on success
    """
    try:
        heatmap = _get_gradcam_heatmap(model, img_array, target_layer_name)
        heatmap_pil = _colorize_heatmap(heatmap)
        orig_rgb = original_pil.convert("RGB")
        heatmap_resized = heatmap_pil.resize(orig_rgb.size, Image.LANCZOS)
        overlay_pil = _overlay_heatmap(orig_rgb, heatmap_resized, alpha=alpha)
        return {"heatmap_pil": heatmap_pil, "overlay_pil": overlay_pil, "error": None}
    except Exception as exc:
        return {"heatmap_pil": None, "overlay_pil": None, "error": str(exc)}


def pil_to_bytes(pil_image, fmt="PNG"):
    """Convert a PIL Image to raw bytes (for Streamlit st.image or PDF embedding)."""
    buf = io.BytesIO()
    pil_image.save(buf, format=fmt)
    buf.seek(0)
    return buf.getvalue()
