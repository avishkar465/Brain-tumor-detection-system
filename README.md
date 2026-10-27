# 🧠 Brain Tumor Detection System

A clean, modern, and explainable deep learning web application built with **TensorFlow / Keras**, **Streamlit**, and **ReportLab**. The system performs binary classification on axial brain MRI scans (identifying **Tumor Detected** vs. **No Tumor Detected**), visualizes focal diagnostic regions using **Grad-CAM (Gradient-weighted Class Activation Mapping)**, and generates a clinical-grade PDF screening summary for download.

---

## 📌 Project Overview

Brain tumors are life-threatening conditions where early, accurate screening can dramatically impact clinical outcomes. This project provides an automated, transparent AI screening dashboard designed to assist educational research and demonstrate computer vision in medical diagnostics:

1. **Patient Intake**: Collects patient metadata (Name, Age, City, Contact).
2. **Automated Inference**: Employs a trained 3-block Convolutional Neural Network (CNN) to predict the presence of a tumor with calibrated confidence.
3. **Explainable AI (Grad-CAM)**: Generates a JET-colormap heatmap and alpha-blended overlay directly from the final convolutional layer (`conv2d_2`) to show which regions drove the model's prediction.
4. **Diagnostic PDF Generation**: Produces a standardized, single-page A4 PDF screening summary embedding patient details, original scan, detection findings, and Grad-CAM explainability visuals.

---

## ✨ Features

- **Dark Blue / Cyan Aesthetic UI**: Clean, responsive layout with intuitive 5-step visual workflow indicators.
- **Deep Learning Inference**: Preprocessed RGB inputs (`224 × 224 × 3`) evaluated against a binary sigmoid CNN model.
- **Explainable AI (Grad-CAM)**: Keras 3 compatible layer-by-layer gradient tracking providing transparent visual explanation (Original MRI, JET Heatmap, and Alpha-Blended Overlay).
- **Publication-Ready PDF Reports**: Dynamically generated via ReportLab with patient details, timestamp, detection badge, confidence score, embedded scan images, and medical disclaimers.
- **Automated Test Suite**: Standalone test runner (`tests/test_gradcam.py`) verifying model integrity, forward inference, and Grad-CAM generation.
- **Git LFS Configured**: Built-in support for tracking large model weights (`.keras`, `.h5`) without exceeding GitHub file-size limits.

---

## 🛠️ Technologies Used

| Category | Technology | Purpose |
| :--- | :--- | :--- |
| **Deep Learning** | TensorFlow 2.x & Keras 3 | CNN model loading, forward inference, and gradient tracking |
| **Web Interface** | Streamlit | Interactive, stateful web dashboard and file handling |
| **Explainable AI** | Matplotlib & NumPy | JET colormap generation, array normalization, and tensor math |
| **Image Processing** | Pillow (PIL) | EXIF transposition, format conversion, and image resizing |
| **Document Generation** | ReportLab | Programmatic single-page A4 PDF screening report rendering |
| **Large File Management**| Git LFS | Versioning large model weights (`128 MB`) on GitHub |

---

## 📁 Project Folder Structure

```text
brain_Tumor_Detection/
├── .gitattributes          # Git Large File Storage (LFS) configuration
├── .gitignore              # Ignores virtual environments, caches, secrets, and reports
├── app.py                  # Main Streamlit web application & UI pipeline
├── gradcam.py              # Grad-CAM heatmap & alpha-blended overlay engine
├── pdf_report.py           # ReportLab medical screening report generator
├── requirements.txt        # Production dependency specifications
├── README.md               # Comprehensive project documentation
├── model/
│   └── brain_tumor_model.keras  # Trained CNN model weights (127.86 MB)
└── tests/
    └── test_gradcam.py     # End-to-end inference and Grad-CAM test suite
```

---

## 💻 Prerequisites & Python Version

- **Operating System**: Windows 10/11, macOS, or Linux.
- **Python Version**: Python `3.11` is recommended for optimal TensorFlow and Keras 3 compatibility.
- **Git** & **Git LFS** (optional, recommended for cloning/pushing model weights).

---

## 🚀 Installation & Setup (Windows)

### 1. Clone or Open the Repository

Open PowerShell or Command Prompt:

```powershell
cd D:\files\brain_Tumor_Detection
```

### 2. Virtual Environment Setup

> **Note**: If you already have a functional virtual environment (such as `venv`), you **do not** need to recreate it. Simply activate it in Step 3.

To create a fresh virtual environment:

```powershell
python -m venv venv
```

### 3. Activate the Virtual Environment

In Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

*(If running Command Prompt, use `venv\Scripts\activate.bat`)*

### 4. Install Dependencies

Install the verified project dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 🧪 Running the Tests

Before launching the web interface, you can verify that the trained model, image preprocessing, and Grad-CAM pipeline are functioning properly:

```powershell
python tests/test_gradcam.py
```

**Expected Output:**
```text
=== 1. Checking Model Path ===
Model file exists.
=== 2. Loading Keras Model ===
Model layers: ['conv2d', 'max_pooling2d', 'conv2d_1', 'max_pooling2d_1', 'conv2d_2', 'max_pooling2d_2', 'flatten', 'dense', 'dropout', 'dense_1']
Model loaded successfully.
=== 3. Creating Synthetic Test MRI Scan ===
Test image shape: (224, 224)
=== 4. Preprocessing Input ===
Preprocessing completed successfully.
=== 5. Running Model Inference ===
Prediction Result: Tumor Detected (... confidence)
=== 6. Generating Grad-CAM Heatmap & Overlay ===
Heatmap dimensions: (52, 52), mode: RGB
Overlay dimensions: (224, 224), mode: RGB
===============================
[SUCCESS] ALL TESTS PASSED SUCCESSFULLY
===============================
```

---

## 🌐 Running the Streamlit Application

Start the Streamlit development server:

```powershell
python -m streamlit run app.py
```

Once started, the application will automatically open in your default browser at:
`http://localhost:8501`

---

## 📖 Step-by-Step User Workflow

1. **Step 1: Patient Details**
   - Enter Patient Name, Age, City, and Mobile Number.
2. **Step 2: Upload MRI**
   - Upload an axial brain MRI scan image (`.jpg`, `.jpeg`, or `.png`).
   - A live thumbnail preview displays image dimensions and format confirmation.
3. **Step 3: Detect Tumor**
   - Click the **🔬 Detect Tumor** button.
   - The system validates form inputs, preprocesses the scan, runs model inference, generates the Grad-CAM visualization, and prepares the PDF in memory.
4. **Step 4: Explainable AI (Grad-CAM Visualization)**
   - Review the 3 side-by-side panels:
     - **Original MRI Scan**: The raw uploaded scan.
     - **Grad-CAM Heatmap (JET)**: Highlights the specific activation focal points of convolutional layer `conv2d_2`.
     - **MRI + Heatmap Overlay**: 45% alpha-blended composite clearly demarcating suspect regions.
   - Read the color interpretation guide:
     - 🔴 **Red / Warm Areas**: Highest influence on the model's decision.
     - 🟡 **Yellow / Green Areas**: Moderate importance.
     - 🔵 **Blue / Cool Areas**: Low or no influence.
5. **Step 5: Download Report**
   - Click **📄 Download PDF Report** to download a finalized, named screening report (e.g., `Brain_Tumor_Report_John_Doe.pdf`).

---

## 🧠 Model Architecture & Weights

The pre-trained model is located at:
`model/brain_tumor_model.keras`

### Architecture Summary

```text
Layer (type)              Output Shape         Param #
=========================================================
InputLayer                (None, 224, 224, 3)  0
Conv2D (32, 3x3, relu)    (None, 222, 222, 32) 896
MaxPooling2D (2x2)        (None, 111, 111, 32) 0
Conv2D (64, 3x3, relu)    (None, 109, 109, 64) 18,496
MaxPooling2D (2x2)        (None, 54, 54, 64)   0
Conv2D (128, 3x3, relu)   (None, 52, 52, 128)  73,856   <-- Grad-CAM Target (conv2d_2)
MaxPooling2D (2x2)        (None, 26, 26, 128)  0
Flatten                   (None, 86528)        0
Dense (128, relu)         (None, 128)          11,075,712
Dropout (0.5)             (None, 128)          0
Dense (1, sigmoid)        (None, 1)            129
=========================================================
Total params: 11,169,089 (42.61 MB float32 weights, ~128 MB model bundle)
```

### Class Mapping
- Class `0`: **No Tumor Detected** (Sigmoid score `< 0.5`)
- Class `1`: **Tumor Detected** (Sigmoid score `≥ 0.5`)

---

## 📦 GitHub Upload & Large File Storage (Git LFS)

The model weights file (`brain_tumor_model.keras`) is **127.86 MB**, which exceeds GitHub's default 100 MB per-file push limit.

The repository includes a pre-configured `.gitattributes` file. To upload the repository to GitHub:

### Option A: Using Git LFS (Recommended)

1. Download and install [Git LFS](https://git-lfs.github.com/).
2. Run the following commands in the project folder:
   ```powershell
   git lfs install
   git add .
   git commit -m "Initial commit: Brain Tumor Detection System with Grad-CAM and PDF reporting"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```

### Option B: Hosting Model via GitHub Releases

If you prefer not to use Git LFS quota:
1. Add `model/*.keras` to `.gitignore`.
2. Push the source code to GitHub.
3. Create a GitHub Release in your repository and attach `brain_tumor_model.keras`.
4. Provide the direct download link in this section so users can place it in `model/`.

---

## 🔧 Troubleshooting Common Errors

### 1. `module 'matplotlib.cm' has no attribute 'get_cmap'`
- **Cause**: Matplotlib 3.9+ removed `matplotlib.cm.get_cmap()`.
- **Solution**: Handled automatically in `gradcam.py` via `_get_colormap()` which routes to `matplotlib.colormaps["jet"]`.

### 2. `Execution of scripts is disabled on this system` (PowerShell)
- **Cause**: Windows PowerShell execution policy restriction.
- **Solution**: Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` in PowerShell, then re-run `venv\Scripts\Activate.ps1`.

### 3. Model File Not Found
- **Cause**: `brain_tumor_model.keras` was moved or not downloaded.
- **Solution**: Ensure `model/brain_tumor_model.keras` exists in the `model/` directory relative to `app.py`.

---

## ⚖️ Medical Disclaimer

> **IMPORTANT**: This software application is developed **strictly for educational, academic, and portfolio demonstration purposes**. It does **NOT** constitute medical software, clinical advice, or a formal diagnostic tool. Predictions made by this deep learning model must never be used to make healthcare or treatment decisions. Always consult a licensed medical professional or radiologist for clinical evaluation.
