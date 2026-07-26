<p align="center">
  <img src="assets/logo.png" width="700">
</p>

# 🩺 MedImage Diagnosis: Brain & Liver Tumor Detection

<a>
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python">
</a>
<a>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow">
</a>

<p align="center">

  <a href="https://machine-learning-pythonproyectofinalvmelissahm-h2zooaso9fgqr6z.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀%20Demo-Streamlit-red?style=for-the-badge">
  </a>

  <a href="https://github.com/melissahm/machine-learning-python_proyecto_final_vmelissahm">
    <img src="https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github">
  </a>
</p>

<p align="center">
Deep Learning application for the analysis of brain MRI scans and abdominal CT images.
</p>

A **Deep Learning** system for the automatic detection of tumors in brain and liver medical images.

The application integrates two independently developed modules:

- 🧠 Binary brain tumor classification using magnetic resonance imaging (MRI).
- 🟤 Liver tumor detection through a cascaded architecture based on U-Net segmentation and a CNN classifier.

The project is presented through a web application built with **Streamlit**, allowing users to run interactive inference on medical images.

<p align="center">
  <img src="assets/screenshots/home_streamlit.png" width="950">
</p>

## 📑 Table of Contents

- Introduction
- Features
- Application Preview
- Brain Model Pipeline
- Liver Model Pipeline
- Datasets
- Performance Summary
- Technologies
- Installation and Execution
- Model Download
- Project Structure
- Limitations
- Future Work
- Medical Disclaimer
- Author

## Introduction

Medical imaging is one of the main tools used in healthcare for the detection and monitoring of multiple diseases. However, both magnetic resonance imaging and computed tomography studies may contain hundreds of two-dimensional slices that must be manually reviewed by specialists.

This project explores the use of **Deep Learning** techniques to support the automatic identification of tumors in two different medical imaging scenarios:

- Brain tumors using magnetic resonance imaging (MRI).
- Liver tumors using computed tomography (CT).

Two independent pipelines were developed and later integrated into a Streamlit web application.

## Features

- Binary classification of brain MRI scans.
- Automatic liver segmentation using U-Net.
- Binary classification of liver tumors.
- Web application developed with Streamlit.
- Automatic model download from Google Drive.
- Real-time prediction on medical images.

## 🖥️ Application Preview

### 🧠 Brain Tumor Classifier

Main interface of the classification module.

<p align="center">
  <img src="assets/screenshots/streamlit_cerebro.png" width="900">
</p>

Example prediction result.

<p align="center">
  <img src="assets/screenshots/streamlit_cerebro_result.png" width="900">
</p>

### 🟤 Liver Analysis

Main interface of the liver analysis module.

<p align="center">
  <img
    src="assets/screenshots/streamlit_higado.png"
    width="900"
    alt="Liver analysis module interface"
  >
</p>

Liver region segmentation.

<p align="center">
  <img
    src="assets/screenshots/streamlit_higado_segmentacion.png"
    width="900"
    alt="Automatic liver segmentation and region of interest extraction"
  >
</p>

Example prediction result.

<p align="center">
  <img
    src="assets/screenshots/streamlit_higado_result.png"
    width="900"
    alt="Liver tumor classification result"
  >
</p>

## 🧠 Brain Model Pipeline

The model receives a preprocessed two-dimensional slice from a brain MRI scan as input.

During inference, the CNN estimates the probability that the image belongs to one of the following two classes:

- Tumor
- No tumor

<p align="center">
  <img src="assets/diagrams/diagrama_cerebro.png" width="850">
</p>

## 🟤 Liver Model Pipeline

The liver module implements a cascaded architecture composed of two Deep Learning models.

First, a U-Net automatically segments the liver in the CT image. The isolated liver region is then passed to a binary CNN classifier that estimates the presence or absence of a tumor.

<p align="center">
  <img src="assets/diagrams/diagrama_higado.png" width="950">
</p>

---

## 📚 Datasets

### 🧠 Brain MRI Dataset

The original dataset contains approximately **7,200 brain MRI images**, distributed across four categories:

- Glioma
- Meningioma
- Pituitary tumor
- No tumor

To transform the task into a binary classification problem, the original classes were grouped as follows:

```text
Tumor    = glioma + meningioma + pituitary tumor
No tumor = no tumor
```

To work with balanced classes, a subset of **3,600 images** was selected:

| Split | No tumor | Tumor | Total |
|---|---:|---:|---:|
| Training | 1,400 | 1,400 | 2,800 |
| Test | 400 | 400 | 800 |
| **Total used** | **1,800** | **1,800** | **3,600** |

The `Tumor` class includes samples from the three original tumor categories while maintaining a balanced representation:

- Training: 466 glioma, 466 meningioma, and 468 pituitary tumor images.
- Test: 133 glioma, 133 meningioma, and 134 pituitary tumor images.

---

### 🟤 Liver CT Dataset

The liver module uses abdominal computed tomography images obtained from volumetric studies.

The dataset contains:

| Item | Quantity |
|---|---:|
| Patients or volumes | 131 |
| Original CT images | 58,638 |
| Liver masks | 58,638 |
| Tumor masks | 58,638 |

Each volumetric study is composed of multiple two-dimensional axial slices. To prevent data leakage, the training and validation split was performed at patient level.

#### Data Used by the V4B Liver Classifier

| Split | No tumor | Tumor | Total |
|---|---:|---:|---:|
| Training | 7,396 | 4,368 | 11,764 |
| Validation | 1,762 | 1,285 | 3,047 |
| **Training + validation** | **9,158** | **5,653** | **14,811** |

An additional **4,332 test images** were used, resulting in a total of **19,143 images** considered during the development and evaluation of the pipeline.

There are no repeated patients between the training and validation sets.

#### Data Used by the Segmentation Model

| Split | Images |
|---|---:|
| Training | 12,157 |
| Validation | 2,654 |
| Test | 4,332 |
| **Total** | **19,143** |

---

## 📊 Performance Summary

### Main Metrics

| Module | Main metric | Result |
|---|---|---:|
| Brain classifier | Accuracy | **0.92** |
| U-Net liver segmentation | Dice Score | **0.92** |
| V4B liver classifier | Accuracy | **0.67** |

> These metrics correspond to academic models developed for educational purposes and do not represent clinical validation.

---

### 🧠 Brain Classifier Evaluation

The binary brain model achieved **92% accuracy** on a balanced test set of 800 images.

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| No tumor | 0.88 | 0.96 | 0.92 | 400 |
| Tumor | 0.96 | 0.86 | 0.91 | 400 |
| **Weighted average** | **0.92** | **0.92** | **0.91** | **800** |

The results show a strong ability to distinguish between images with tumor presence and images without evidence of a tumor.

---

### 🟤 Liver Segmentation Evaluation

The U-Net model achieved a **Dice Score of 0.92** for automatic liver segmentation.

The segmentation stage makes it possible to:

1. Locate the liver region.
2. Generate a binary mask.
3. Extract a region of interest (ROI).
4. Send only the liver region to the tumor classifier.

The final pipeline output depends on segmentation quality, since an inaccurate mask may affect both the crop and the subsequent classification.

---

### 🟤 V4B Liver Classifier Evaluation

The liver classifier achieved **67% accuracy**.

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| No tumor | 0.79 | 0.68 | 0.73 | 2,871 |
| Tumor | 0.50 | 0.64 | 0.56 | 1,461 |
| **Weighted average** | **0.69** | **0.67** | **0.67** | **4,332** |

The model performs better overall on the `No tumor` class, while tumor detection remains the main area for improvement.

---

## 🛠️ Technologies

### Language and Data Analysis

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

### Deep Learning and Image Processing

- TensorFlow
- Keras
- OpenCV
- Pillow
- Convolutional Neural Networks
- U-Net
- Semantic segmentation

### Application and Deployment

- Streamlit
- Google Drive
- gdown
- Git
- GitHub
- GitHub Codespaces

---

## 🚀 Installation and Execution

### 1. Clone the Repository

```bash
git clone https://github.com/melissahm/machine-learning-python_proyecto_final_vmelissahm.git
cd machine-learning-python_proyecto_final_vmelissahm
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

#### Windows

```bash
.venv\Scripts\activate
```

#### Linux or macOS

```bash
source .venv/bin/activate
```

### 3. Install the Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run src/app.py
```

The application will open automatically in your browser.

---

## 🤖 Model Download

The model files are not stored directly in the repository because of their size.

They are downloaded only the first time the application is executed. The models are then stored locally to prevent unnecessary repeated downloads.

The pipeline includes:

- Binary brain classifier.
- U-Net liver segmentation model.
- V4B liver classifier.

Model downloads are managed by the loading modules included in the project, so users do not need to download the files manually.

---

## 📂 Project Structure

```text
.
├── assets/
│   ├── diagrams/
│   │   ├── diagrama_cerebro.png
│   │   └── diagrama_higado.png
│   ├── screenshots/
│   │   ├── home_streamlit.png
│   │   ├── streamlit_cerebro.png
│   │   ├── streamlit_cerebro_result.png
│   │   ├── streamlit_higado.png
│   │   ├── streamlit_higado_carga.png
│   │   ├── streamlit_higado_segmentacion.png
│   │   └── streamlit_higado_result.png
│   ├── brain.png
│   ├── liver.png
│   └── logo.png
├── data/
│   ├── interim/
│   ├── processed/
│   ├── raw/
│   └── test_images/
├── models/
├── src/
│   ├── organ_apps/
│   │   ├── cerebro_app.py
│   │   └── higado_app.py
│   ├── utils/
│   └── app.py
├── .gitignore
├── README.es.md
├── README.md
└── requirements.txt
```

### Main Components

- `src/app.py`: entry point for the Streamlit application.
- `src/organ_apps/cerebro_app.py`: interface and inference logic for the brain module.
- `src/organ_apps/higado_app.py`: segmentation and inference logic for the liver module.
- `src/utils/`: helper functions for model loading and image preprocessing.
- `assets/`: logo, diagrams, and screenshots used in the documentation.
- `data/test_images/`: example images for testing the application.
- `models/`: directory used to store downloaded models.

---

## ⚠️ Limitations

### Brain Model

- Performs binary classification and does not identify the specific tumor type.
- Analyzes two-dimensional slices independently.
- Does not spatially locate the tumor.
- Performance may vary with images from different hospitals, scanners, or MRI protocols.

### Liver Model

- Classification depends on the quality of the previous segmentation stage.
- Slices located at the edges of a volume may contain a very small liver region or no liver at all.
- Liver lesions may vary considerably in shape, size, and density.
- The V4B classifier still requires improvement before being considered for more demanding applications.

---

## 🔮 Future Work

### Brain

- Extend the model to multiclass classification:
  - Glioma.
  - Meningioma.
  - Pituitary tumor.
  - No tumor.
- Train with images from different hospitals and MRI scanners.
- Add interpretability techniques such as Grad-CAM and heatmaps.
- Evaluate architectures capable of using three-dimensional information.

### Liver

- Include a larger number of patients.
- Optimize both the segmentation model and the binary classifier.
- Evaluate more complex liver lesions.
- Improve performance on edge slices.
- Analyze complete volumes instead of independent 2D slices.

---

## 🩺 Medical Disclaimer

This project is an academic prototype developed for educational purposes.

**It is not a certified medical device and must not be used to diagnose diseases, recommend treatments, or make clinical decisions.**

The outputs should only be interpreted as experimental predictions generated by Deep Learning models.

---

## 👩‍💻 Author

**Melissa Huamán**

Data Scientist

<p align="left">
  <a href="https://github.com/melissahm">
    <img src="https://img.shields.io/badge/GitHub-melissahm-black?style=for-the-badge&logo=github">
  </a>
</p>

<p align="right">
  <a href="README.es.md">Versión en español</a>
</p>
