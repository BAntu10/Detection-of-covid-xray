# X-ray Image Classification: COVID-19, Pneumonia, and Normal Chest X-rays

## Introduction

This repository contains a deep learning model developed by Prakhar Biswas to classify chest X-ray images into three categories: COVID-19, Pneumonia, and Normal. The model aims to assist healthcare professionals by providing an automated and accurate method to differentiate between these conditions using X-ray images.

## Features

- **Multi-class Classification**: Classifies X-ray images into COVID-19, Pneumonia, and Normal categories.
- **Pre-trained Model**: Utilizes a pre-trained Convolutional Neural Network (CNN) for transfer learning to achieve high accuracy.
- **Easy to Use**: Simple interface for running predictions on new X-ray images.
- **Data Augmentation**: Includes data augmentation techniques to enhance model robustness.

## Requirements

To run the project, you need to have the following dependencies installed:

- Python 3.8+
- TensorFlow 2.4+
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- OpenCV
- ydata-profiling

You can install the required packages using `pip`:

```bash
pip install tensorflow keras numpy pandas matplotlib scikit-learn opencv-python ydata-profiling
