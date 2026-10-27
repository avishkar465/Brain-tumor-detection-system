"""
Brain Tumor Detection System
A simple, clean, and modern Streamlit application for brain MRI scan analysis.

Workflow:
1. Enter Patient Details (Name, Age, City, Mobile Number)
2. Upload MRI (JPG, JPEG, PNG)
3. Detect Tumor (Inference with pre-trained CNN model)
4. Download PDF Report (Complete summary with image & medical disclaimer)
"""

import os
import sys
import io
from pathlib import Path
from datetime import datetime

# Silence TensorFlow verbose logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

import pdf_report
import gradcam

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "brain_tumor_model.keras"
IMG_SIZE = (224, 224)

# -----------------------------------------------------------------------------
# Page Configuration & Dark Blue/Cyan Theme Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor Detection System",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    /* Global Page Styling */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #0d1e36 0%, #080f1d 75%, #050a14 100%);
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Hide default Streamlit sidebar toggle & footer for a clean view */
    [data-testid="stSidebarNav"] { display: none; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* Header Container */
    .header-card {
        background: linear-gradient(135deg, #0c1c33 0%, #0f2b48 50%, #081628 100%);
        border: 1px solid #0284c7;
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 4px 24px rgba(2, 132, 199, 0.15);
        text-align: center;
    }

    .header-title {
        color: #38bdf8;
        font-size: 30px;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
    }

    .header-subtitle {
        color: #94a3b8;
        font-size: 14px;
        margin-bottom: 16px;
    }

    /* Workflow Stepper Bar */
    .step-bar {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
        font-size: 12px;
        font-weight: 600;
    }

    .step-badge {
        background: rgba(14, 165, 233, 0.12);
        color: #38bdf8;
        border: 1px solid #0284c7;
        padding: 5px 12px;
        border-radius: 20px;
    }

    .step-arrow {
        color: #0284c7;
        font-weight: 800;
    }

    /* Section Cards */
    .section-card {
        background: #0b172a;
        border: 1px solid #1e3a5f;
        border-radius: 14px;
        padding: 22px 24px;
        margin-bottom: 22px;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.25);
    }

    .section-title {
        color: #38bdf8;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Form Labels & Inputs */
    label[data-testid="stWidgetLabel"] p {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
        font-size: 13.5px !important;
    }

    div[data-baseweb="input"] {
        background-color: #07101e !important;
        border: 1px solid #1e3a5f !important;
        border-radius: 10px !important;
        color: #f8fafc !important;
    }

    div[data-baseweb="input"]:focus-within {
        border-color: #0284c7 !important;
        box-shadow: 0 0 0 1px #0284c7 !important;
    }

    /* File Uploader */
    [data-testid="stFileUploader"] {
        background-color: #07101e;
        border: 1px dashed #0284c7;
        border-radius: 12px;
        padding: 14px;
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        color: #ffffff !important;
        border: 1px solid #38bdf8 !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        letter-spacing: 0.3px !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.3) !important;
    }

    .stButton button:hover {
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important;
        border-color: #7dd3fc !important;
        box-shadow: 0 6px 20px rgba(14, 165, 233, 0.45) !important;
        transform: translateY(-1px);
    }

    .stDownloadButton button {
        background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%) !important;
        color: #ffffff !important;
        border: 1px solid #2dd4bf !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 14px rgba(13, 148, 136, 0.3) !important;
    }

    .stDownloadButton button:hover {
        background: linear-gradient(135deg, #14b8a6 0%, #0d9488 100%) !important;
        border-color: #5eead4 !important;
        box-shadow: 0 6px 20px rgba(20, 184, 166, 0.45) !important;
        transform: translateY(-1px);
    }

    /* Result Badges */
    .result-badge-tumor {
        background: rgba(239, 68, 68, 0.15);
        border: 2px solid #ef4444;
        color: #f87171;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        font-size: 22px;
        font-weight: 800;
        margin-bottom: 14px;
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.2);
    }

    .result-badge-notumor {
        background: rgba(16, 185, 129, 0.15);
        border: 2px solid #10b981;
        color: #34d399;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        font-size: 22px;
        font-weight: 800;
        margin-bottom: 14px;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.2);
    }

    .confidence-box {
        background: #07101e;
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
        margin-bottom: 16px;
    }

    .confidence-label {
        font-size: 12px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
    }

    .confidence-val {
        font-size: 26px;
        font-weight: 800;
        color: #38bdf8;
    }

    /* Disclaimer Card */
    .disclaimer-card {
        background: rgba(120, 53, 15, 0.12);
        border: 1px solid #d97706;
        border-radius: 10px;
        padding: 12px 16px;
        color: #fbbf24;
        font-size: 12.5px;
        line-height: 1.5;
        margin-top: 14px;
    }

    /* Grad-CAM / XAI Section */
    .xai-card {
        background: linear-gradient(135deg, #0a1628 0%, #0d1f3a 50%, #071220 100%);
        border: 1px solid #7c3aed;
        border-radius: 14px;
        padding: 22px 24px;
        margin-bottom: 22px;
        box-shadow: 0 2px 20px rgba(124, 58, 237, 0.15);
    }

    .xai-title {
        color: #a78bfa;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .xai-legend {
        background: rgba(124, 58, 237, 0.08);
        border: 1px solid #7c3aed;
        border-radius: 10px;
        padding: 14px 16px;
        margin-top: 16px;
        font-size: 13px;
        color: #c4b5fd;
        line-height: 1.6;
    }

    .xai-legend b { color: #ddd6fe; }

    .xai-error {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid #ef4444;
        border-radius: 10px;
        padding: 12px 16px;
        color: #f87171;
        font-size: 13px;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Load Model (Cached in memory)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load the pre-trained Keras model without compiling."""
    if not MODEL_PATH.exists():
        return None
    return tf.keras.models.load_model(str(MODEL_PATH), compile=False)


# Check model existence
if not MODEL_PATH.exists():
    st.error(f"❌ Model file not found at: {MODEL_PATH}")
    st.info("Please make sure 'model/brain_tumor_model.keras' exists before running.")
    st.stop()

model = load_model()
if model is None:
    st.error("❌ Failed to load the deep learning model.")
    st.stop()


# -----------------------------------------------------------------------------
# Preprocessing and Prediction Function
# -----------------------------------------------------------------------------
def predict_image(pil_image: Image.Image):
    """
    Preprocess image identically to training pipeline and run inference:
    - RGB conversion
    - Resize to (224, 224)
    - Rescale pixel values / 255.0
    - Model predict (Sigmoid output)
    - Output: 'Tumor Detected' if prob >= 0.5 else 'No Tumor Detected'

    Returns:
        result_text (str), confidence (float), raw_pred (float), input_tensor (ndarray)
    """
    # Fix orientation and convert to RGB
    img = ImageOps.exif_transpose(pil_image).convert("RGB")
    # Resize to model input shape
    resized_img = img.resize(IMG_SIZE)
    # Normalize to [0, 1]
    img_array = np.asarray(resized_img, dtype=np.float32) / 255.0
    # Add batch dimension -> (1, 224, 224, 3)
    input_tensor = np.expand_dims(img_array, axis=0)

    # Predict
    raw_output = model.predict(input_tensor, verbose=0)
    raw_pred = float(np.asarray(raw_output).reshape(-1)[0])

    if raw_pred >= 0.5:
        result_text = "Tumor Detected"
        confidence = raw_pred * 100.0
    else:
        result_text = "No Tumor Detected"
        confidence = (1.0 - raw_pred) * 100.0

    # Return input_tensor so Grad-CAM can reuse the same preprocessed array
    return result_text, confidence, raw_pred, input_tensor


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="header-card">
        <div class="header-title">🧠 Brain Tumor Detection System</div>
        <div class="header-subtitle">Deep Learning MRI Diagnostic Screening Dashboard</div>
        <div class="step-bar">
            <span class="step-badge">1. Patient Details</span>
            <span class="step-arrow">➔</span>
            <span class="step-badge">2. Upload MRI</span>
            <span class="step-arrow">➔</span>
            <span class="step-badge">3. Detect Tumor</span>
            <span class="step-arrow">➔</span>
            <span class="step-badge">4. Grad-CAM XAI</span>
            <span class="step-arrow">➔</span>
            <span class="step-badge">5. Download Report</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Step 1: Enter Patient Details
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="section-card">
        <div class="section-title">📋 Step 1: Enter Patient Details</div>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)
with col1:
    patient_name = st.text_input(
        "Patient Name",
        placeholder="Enter full name (e.g. John Doe)",
        key="patient_name",
    )
    patient_age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=st.session_state.get("patient_age", 35),
        step=1,
        key="patient_age",
    )

with col2:
    patient_city = st.text_input(
        "City",
        placeholder="Enter city (e.g. New York)",
        key="patient_city",
    )
    patient_mobile = st.text_input(
        "Mobile Number",
        placeholder="Enter mobile number (e.g. +1 555-0199)",
        key="patient_mobile",
    )

st.write("")

# -----------------------------------------------------------------------------
# Step 2: Upload MRI
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="section-card">
        <div class="section-title">📤 Step 2: Upload MRI</div>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Choose a Brain MRI scan image (JPG, JPEG, or PNG)",
    type=["jpg", "jpeg", "png"],
    help="Upload an axial brain MRI scan image for automated analysis",
    key="mri_file",
)

uploaded_pil_image = None
if uploaded_file is not None:
    try:
        uploaded_pil_image = Image.open(uploaded_file)
        # Preview image
        preview_col1, preview_col2 = st.columns([1, 2])
        with preview_col1:
            st.image(uploaded_pil_image, caption="Uploaded MRI Preview", width=220)
        with preview_col2:
            st.markdown(
                f"""
                <div style="background: #07101e; border: 1px solid #1e3a5f; border-radius: 10px; padding: 14px; margin-top: 10px;">
                    <div style="color: #94a3b8; font-size: 13px;"><b>Filename:</b> {uploaded_file.name}</div>
                    <div style="color: #94a3b8; font-size: 13px; margin-top: 4px;"><b>Original Size:</b> {uploaded_pil_image.width} × {uploaded_pil_image.height} px</div>
                    <div style="color: #94a3b8; font-size: 13px; margin-top: 4px;"><b>Format:</b> {uploaded_pil_image.format or 'Image'}</div>
                    <div style="color: #38bdf8; font-size: 12.5px; margin-top: 8px;">✓ Ready for deep learning inference</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    except Exception as e:
        st.error(f"⚠️ Could not load image file: {e}")
        uploaded_pil_image = None

st.write("")

# -----------------------------------------------------------------------------
# Step 3: Get Result
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="section-card">
        <div class="section-title">🔍 Step 3: Get Result</div>
    </div>
    """,
    unsafe_allow_html=True,
)

detect_clicked = st.button("🔬 Detect Tumor", type="primary", use_container_width=True)

if detect_clicked:
    # Validation of fields
    errors = []
    if not patient_name.strip():
        errors.append("Patient Name is required.")
    if not patient_city.strip():
        errors.append("City is required.")
    if not patient_mobile.strip():
        errors.append("Mobile Number is required.")
    if uploaded_file is None or uploaded_pil_image is None:
        errors.append("Please upload a valid MRI scan image (JPG, JPEG, or PNG).")

    if errors:
        for err in errors:
            st.warning(f"⚠️ {err}")
    else:
        # Run inference with loading spinner
        with st.spinner("Analyzing MRI scan with deep learning model..."):
            try:
                pred_result, confidence, raw_score, input_tensor = predict_image(uploaded_pil_image)
                scan_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # --- Grad-CAM ---------------------------------------------------
                gradcam_result = gradcam.generate_gradcam(
                    model=model,
                    original_pil=uploaded_pil_image,
                    img_array=input_tensor,
                    target_layer_name="conv2d_2",
                    alpha=0.45,
                )
                # ----------------------------------------------------------------

                # Generate PDF report in memory
                # Include overlay image in PDF if Grad-CAM succeeded
                overlay_for_pdf = gradcam_result["overlay_pil"] if gradcam_result["error"] is None else None
                pdf_bytes = pdf_report.generate_pdf_report(
                    patient_name=patient_name.strip(),
                    age=int(patient_age),
                    city=patient_city.strip(),
                    mobile_number=patient_mobile.strip(),
                    scan_datetime=scan_timestamp,
                    mri_image=uploaded_pil_image,
                    prediction_result=pred_result,
                    confidence=confidence,
                    gradcam_overlay=overlay_for_pdf,
                )

                # Store result in session state to persist across reruns
                st.session_state["analysis_result"] = {
                    "patient_name": patient_name.strip(),
                    "age": int(patient_age),
                    "city": patient_city.strip(),
                    "mobile_number": patient_mobile.strip(),
                    "prediction_result": pred_result,
                    "confidence": confidence,
                    "scan_datetime": scan_timestamp,
                    "pdf_bytes": pdf_bytes,
                    "image": uploaded_pil_image,
                    "filename": uploaded_file.name,
                    "gradcam_heatmap": gradcam_result["heatmap_pil"],
                    "gradcam_overlay": gradcam_result["overlay_pil"],
                    "gradcam_error": gradcam_result["error"],
                }
            except Exception as e:
                st.error(f"❌ Error during model prediction: {str(e)}")

# -----------------------------------------------------------------------------
# Display Result & Step 4: Download Report
# -----------------------------------------------------------------------------
if "analysis_result" in st.session_state:
    res = st.session_state["analysis_result"]

    st.write("")
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">📊 Analysis Findings</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    res_col1, res_col2 = st.columns([1, 1.2])

    with res_col1:
        st.image(res["image"], caption="Scanned Brain MRI", use_container_width=True)

    with res_col2:
        # Badge
        if res["prediction_result"] == "Tumor Detected":
            st.markdown(
                '<div class="result-badge-tumor">⚠️ Tumor Detected</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result-badge-notumor">✅ No Tumor Detected</div>',
                unsafe_allow_html=True,
            )

        # Confidence Box
        st.markdown(
            f"""
            <div class="confidence-box">
                <div class="confidence-label">Model Confidence</div>
                <div class="confidence-val">{res['confidence']:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Patient & Scan summary
        st.markdown(
            f"""
            <div style="background: #07101e; border: 1px solid #1e3a5f; border-radius: 10px; padding: 12px 16px; margin-bottom: 16px; font-size: 13px;">
                <div style="color: #94a3b8;"><b>Patient:</b> <span style="color: #f8fafc;">{res['patient_name']} ({res['age']} yrs)</span></div>
                <div style="color: #94a3b8; margin-top: 4px;"><b>City:</b> <span style="color: #f8fafc;">{res['city']}</span> | <b>Mobile:</b> <span style="color: #f8fafc;">{res['mobile_number']}</span></div>
                <div style="color: #94a3b8; margin-top: 4px;"><b>Scan Time:</b> <span style="color: #f8fafc;">{res['scan_datetime']}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------------------------------
    # Step 4: Grad-CAM XAI Visualization
    # -----------------------------------------------------------------------------
    st.write("")
    st.markdown(
        """
        <div class="xai-card">
            <div class="xai-title">🔬 Step 4: Explainable AI — Grad-CAM Heatmap</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if res.get("gradcam_error"):
        st.markdown(
            f'<div class="xai-error">⚠️ Grad-CAM could not be generated: {res["gradcam_error"]}</div>',
            unsafe_allow_html=True,
        )
    elif res.get("gradcam_heatmap") is not None:
        xai_col1, xai_col2, xai_col3 = st.columns(3)
        with xai_col1:
            st.image(
                res["image"],
                caption="Original MRI Scan",
                use_container_width=True,
            )
        with xai_col2:
            st.image(
                res["gradcam_heatmap"],
                caption="Grad-CAM Heatmap (JET)",
                use_container_width=True,
            )
        with xai_col3:
            st.image(
                res["gradcam_overlay"],
                caption="MRI + Heatmap Overlay",
                use_container_width=True,
            )

        st.markdown(
            """
            <div class="xai-legend">
                <b>How to read the Grad-CAM heatmap:</b><br>
                The heatmap is generated from the gradients of the model's last convolutional layer
                (<code>conv2d_2</code>). It shows <i>which regions of the MRI scan influenced the prediction most</i>.<br><br>
                🔴 <b>Red / warm areas</b> — highest importance (model focused here most)<br>
                🟡 <b>Yellow / green areas</b> — moderate importance<br>
                🔵 <b>Blue / cold areas</b> — low or no influence on the prediction<br><br>
                In a positive "Tumor Detected" scan, warm colors typically highlight the suspected
                lesion region. This visualization does <b>not</b> replace clinical diagnosis.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------------------------------
    # Step 5: Download Report
    # -----------------------------------------------------------------------------
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">📥 Step 5: Download Report</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Clean patient name for filename
    safe_name = "".join(c for c in res["patient_name"] if c.isalnum() or c in (" ", "_", "-")).rstrip()
    safe_name = safe_name.replace(" ", "_") or "Patient"
    report_filename = f"Brain_Tumor_Report_{safe_name}.pdf"

    st.download_button(
        label="📄 Download PDF Report",
        data=res["pdf_bytes"],
        file_name=report_filename,
        mime="application/pdf",
        type="primary",
        use_container_width=True,
    )

    st.markdown(
        """
        <div class="disclaimer-card">
            <b>Medical Disclaimer:</b> This report is generated by an AI model for educational purposes only.
            It is not a medical diagnosis. Please consult a qualified medical professional.
        </div>
        """,
        unsafe_allow_html=True,
    )
