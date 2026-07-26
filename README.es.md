<p align="center">
  <img src="assets/logo.png" width="700">
</p>

# 🩺 MedImage Diagnosis: Brain & Liver Tumor Detection

Sistema desarrollado mediante **Deep Learning** para la detección automática de tumores en imágenes médicas.

El proyecto integra dos módulos independientes:

- 🧠 **Clasificación binaria de tumores cerebrales** a partir de resonancias magnéticas (RM).
- 🟤 **Detección de tumores hepáticos** mediante una arquitectura en cascada basada en **segmentación U-Net** y una **CNN** para clasificación.

La aplicación ha sido desarrollada como proyecto final del Bootcamp de Data Science & Machine Learning de 4Geeks Academy y cuenta con una interfaz web desarrollada en **Streamlit**, que permite realizar predicciones directamente sobre imágenes médicas.

<p align="center">

<a href="https://machine-learning-pythonproyectofinalvmelissahm-h2zooaso9fgqr6z.streamlit.app/">
<img src="https://img.shields.io/badge/🚀%20Demo-Streamlit-red?style=for-the-badge">
</a>

<a href="https://github.com/melissahm/machine-learning-python_proyecto_final_vmelissahm">
<img src="https://img.shields.io/badge/GitHub-Repositorio-black?style=for-the-badge&logo=github">
</a>

</p>

## 📑 Índice

- Introducción
- Características
- Vista previa
- Arquitectura del proyecto
- Clasificador de tumor cerebral
- Detección de tumor hepático
- Dataset
- Resultados
- Tecnologías utilizadas
- Instalación
- Estructura del proyecto
- Trabajo futuro
- Autor

## Introducción

El diagnóstico por imagen constituye una de las principales herramientas utilizadas en medicina para la detección y seguimiento de múltiples enfermedades. Sin embargo, tanto las resonancias magnéticas como las tomografías computarizadas generan estudios compuestos por cientos de cortes bidimensionales que deben ser analizados manualmente por especialistas.

Este proyecto explora el uso de técnicas de **Deep Learning** para asistir en la identificación automática de tumores en dos escenarios clínicos distintos:

- Tumores cerebrales mediante resonancias magnéticas (RM).
- Tumores hepáticos mediante tomografías computarizadas (TC).

Para ello se desarrollaron dos pipelines independientes que posteriormente fueron integrados en una aplicación web desarrollada con Streamlit.

## Características

- Clasificación binaria de resonancias magnéticas cerebrales.
- Segmentación automática del hígado mediante U-Net.
- Clasificación binaria de tumores hepáticos.
- Aplicación web desarrollada con Streamlit.
- Descarga automática de modelos desde Google Drive.
- Predicción en tiempo real sobre imágenes médicas.

## 🖥️ Vista previa

### Página principal

<p align="center">
  <img src="assets/screenshots/home_streamlit.png" width="900">
</p>

### 🧠 Clasificador de tumor cerebral

<p align="center">
  <img src="assets/screenshots/streamlit_cerebro.png" width="900">
</p>

<p align="center">
  <img src="assets/screenshots/streamlit_cerebro_result.png" width="900">
</p>