# Lung_Cancer_Detection_Ensemble

This project implements a lung cancer detection system using ensemble deep learning techniques. It utilizes three different convolutional neural networks (CNNs) to classify lung images into Normal, Malignant, and Benign categories, achieving high accuracy and robust performance.

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributors)
- [License](#license)

## Features
- Ensemble model combining three CNN architectures for improved accuracy.
- Image preprocessing and augmentation to enhance training.
- Comprehensive evaluation metrics, including confusion matrix and classification report.
- User-friendly visualization of results and performance metrics.

## Prerequisites
Before running the project, ensure you have the following installed:
- Python 3.x
- TensorFlow
- Keras
- OpenCV
- Matplotlib
- Seaborn
- Scikit-learn
- Pandas

You can install them by running:
```bash
pip install tensorflow keras opencv-python matplotlib seaborn scikit-learn pandas
```
## Installation
Clone this repository:
```
git clone https://github.com/yourusername/Lung_Cancer_Detection_Ensemble.git
```
Navigate into the project directory:
```
cd Lung_Cancer_Detection_Ensemble
```
## How to Run
1. Ensure your dataset is prepared and organized into folders for training and testing.
2. Open the Jupyter Notebook or Python script file.
3. Run the code cells or the script sequentially to train the model and evaluate its performance.
## Usage
1. Place your lung images in the appropriate folders (Normal, Malignant, Benign).
2. Adjust the paths in the code to point to your dataset location.
3. Execute the code to train the model and make predictions.
## Model Architecture
The ensemble model consists of three convolutional neural networks with varying architectures. Each model is trained independently, and their predictions are combined to improve accuracy and reduce overfitting.

## Results
The performance of the ensemble model is evaluated using:

+ Accuracy
+ Confusion Matrix
+ Classification Report
You can visualize the results using Matplotlib and Seaborn.

## Contributors
- [Abdullah Hassan](https://github.com/abdullahhassan) - Contributions and support in developing the project.
