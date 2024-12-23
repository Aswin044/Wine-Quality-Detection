# Wine Quality Detection

This project focuses on predicting the quality of wine using machine learning techniques. The dataset contains physicochemical attributes of wines, and the target variable represents wine quality on a scale. By applying algorithms like K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and AdaBoost, and fine-tuning their hyperparameters, this project aims to achieve an accuracy of 80% or higher.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Machine Learning Models](#machine-learning-models)
4. [Hyperparameter Tuning](#hyperparameter-tuning)
5. [Results](#results)
6. [Usage](#usage)
7. [Conclusion](#conclusion)

## Introduction

Wine quality detection is an essential task for winemakers to ensure customer satisfaction and optimize production. Machine learning provides a data-driven approach to predict wine quality based on its physicochemical properties. This project explores various algorithms and aims to build an accurate and reliable prediction model.

## Dataset

The dataset used for this project is sourced from the UCI Machine Learning Repository, consisting of red and white wine samples. Key features include:

- **Fixed Acidity**
- **Volatile Acidity**
- **Citric Acid**
- **Residual Sugar**
- **Chlorides**
- **Free Sulfur Dioxide**
- **Total Sulfur Dioxide**
- **Density**
- **pH**
- **Sulphates**
- **Alcohol**
- **Quality** (Target variable: a score from 0 to 10)

### Data Preprocessing

- Handled missing values and outliers.
- Normalized/standardized features to improve model performance.
- Split dataset into training and testing sets (e.g., 80-20 split).

## Machine Learning Models

The following models were implemented:

1. **K-Nearest Neighbors (KNN):**
   - Used for its simplicity and effectiveness in classification problems.

2. **Support Vector Machine (SVM):**
   - Applied to separate wine quality levels with a robust margin.

3. **AdaBoost:**
   - Employed to combine weak classifiers into a strong ensemble model.

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Hyperparameter Tuning

Hyperparameter optimization was conducted using:

- **Grid Search CV:** Systematic search over predefined hyperparameter values.
- **Random Search CV:** Random sampling of hyperparameter space.

Key parameters tuned:

- KNN: Number of neighbors, distance metric.
- SVM: Kernel type, C (regularization), gamma.
- AdaBoost: Number of estimators, learning rate.

## Results

The tuned models achieved the following accuracies on the test set:

| Model       | Accuracy |
|-------------|----------|
| KNN         | 81.2%    |
| SVM         | 82.5%    |
| AdaBoost    | 84.3%    |

## Usage

### Prerequisites

- Python 3.7+
- Required libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`

### Installation

Clone the repository and install dependencies:
```bash
$ git clone https://github.com/your-repo/wine-quality-detection.git
$ cd wine-quality-detection
$ pip install -r requirements.txt
```

##.

