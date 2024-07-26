# Fraud Detection with Machine Learning and Streamlit Dashboard

This project demonstrates a comprehensive workflow for fraud detection using machine learning techniques. The workflow includes data preprocessing, model training with hyperparameter tuning, and creating an interactive Streamlit dashboard.

## Table of Contents

- [Introduction](#introduction)
- [Workflow](#workflow)
- [Installation](#installation)
- [Usage](#usage)
- [Streamlit Dashboard](#streamlit-dashboard)

## Introduction

The goal of this project is to build a machine learning model to detect fraudulent transactions and create an interactive dashboard for predictions and visualization. The key steps include:

1. Reading and preprocessing data
2. Imputing missing values
3. Detecting and removing outliers
4. Applying PCA for dimensionality reduction
5. Encoding categorical features
6. Splitting data into training and testing sets
7. Training multiple machine learning models using GridSearchCV
8. Selecting the best performing model
9. Balancing the dataset
10. Creating a Streamlit dashboard

## Workflow

1. **Reading the CSV File**: Load the dataset using pandas.
2. **Null Imputation**: Handle missing values appropriately.
3. **Outlier Detection and Removal**: Identify and remove outliers to clean the data.
4. **Unwanted Column Removal**: Drop unnecessary columns.
5. **PCA for Dimensionality Reduction**: Apply Principal Component Analysis to reduce feature dimensionality.
6. **Encoding Object Columns**: Encode categorical variables using LabelEncoder.
7. **Train-Test Split**: Split the dataset into training and testing sets.
8. **Model Training with GridSearchCV**: Train multiple models and tune hyperparameters using GridSearchCV.
9. **Model Selection**: Select the best performing model based on accuracy.
10. **Dataset Balancing**: Ensure the dataset is balanced to improve model performance.
11. **Streamlit Dashboard**: Create an interactive dashboard to visualize results and make predictions.

## Installation

To run this project, you need Python and the following libraries installed:

- pandas
- numpy
- scikit-learn
- matplotlib
- streamlit

Install the required libraries using pip:

```bash
pip install pandas numpy scikit-learn matplotlib streamlit
