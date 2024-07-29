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
```
Dataset
The dataset used for training and testing the model comprises chest X-ray images labeled as COVID-19, Pneumonia, and Normal. You can download the dataset from Kaggle or other relevant sources
Usage
Clone the Repository:

```bash
git clone https://github.com/BAntu10/Detection-of-covid-xray.git
cd xray-classification
```


You can train the model by running the train.py script:

Results
The model achieves high accuracy on the test set, with detailed performance metrics available in the results directory.

Contributing
Contributions are welcome! If you have any suggestions, bug reports, or feature requests, feel free to open an issue or submit a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgements
Kaggle for providing the dataset.
TensorFlow and Keras teams for their powerful deep learning libraries.
Contact
For any questions or inquiries, please contact Prakhar Biswas at prakhar@example.com.



This `README.md` provides a comprehensive overview of the repository, including installation inst
