# diabetic-retinopathy-cnn
A custom Convolutional Neural Network (CNN) model built with TensorFlow/Keras to classify diabetic retinopathy stages from retinal fundus images.
# Classification of Diabetic Retinopathy Stages Using a Custom CNN Architecture

This repository contains the end-to-end deep learning framework developed to automatically detect and classify diabetic retinopathy (DR) severity levels from retinal fundus photographs using the APTOS 2019 dataset.

## Project Overview
Diabetic retinopathy is a leading cause of blindness worldwide. This project implements a sequential Convolutional Neural Network (CNN) in TensorFlow/Keras to streamline the diagnostic pipeline, mapping images into five clinical severity levels:
- **Class 0:** No DR
- **Class 1:** Mild
- **Class 2:** Moderate
- **Class 3:** Severe
- **Class 4:** Proliferative DR

## Tech Stack & Dependencies
- **Language:** Python 3.10.x
- **Deep Learning Framework:** TensorFlow 2.18.0 / Keras
- **Image Processing:** OpenCV (`opencv-python==4.10.0.82`)
- **Data Analytics:** NumPy (`==1.26.4`), Pandas, Scikit-Learn
- **Visualization:** Matplotlib, Seaborn

## Model Architecture
The network is structured as a custom sequential deep learning pipeline:
1. **Input Layer:** Resized to $224 \times 224 \times 3$ pixels and normalized to $[0, 1]$.
2. **Convolutional Layers:** 3 consecutive Conv2D layers with progressive filter sizes (32, 64, 128) using ReLU activation.
3. **Pooling Layers:** MaxPooling2D layers following each convolution to reduce spatial dimensions.
4. **Regularization:** A Dropout layer ($rate = 0.5$) implemented before the final classification to mitigate overfitting.
5. **Output Layer:** Dense layer with 5 units utilizing Softmax activation for categorical probability distribution.

## Optimization & Training Parameters
- **Loss Function:** Categorical Cross-Entropy
- **Optimizer:** Adam ($\alpha = 0.0001$)
- **Batch Size:** 32
- **Epochs:** 20
- **Validation Split:** 20% of the dataset (`stratify=y` for balanced class distribution)

## Repository Structure
- `Untitled2.ipynb`: Complete Jupyter Notebook containing data preprocessing, train-test split, model building, execution logs, and evaluation figures.
- `README.md`: Project documentation and technical specification summary.

## How to Replicate
1. Download the **APTOS 2019 Blindness Detection** dataset.
2. Ensure all packages listed in the dependencies are installed.
3. Update the local file path variable `data_dir` inside the script to point to your dataset directory.
4. Run all cells sequentially in `Untitled2.ipynb`.
