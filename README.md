# Diabetes Prediction ML Application

##  Project Overview

This project is a Machine Learning application designed to predict whether a patient is likely to have diabetes based on selected medical and demographic features.

Two Machine Learning algorithms are used:

- Logistic Regression
- Random Forest Classifier

The final application provides predictions from both models through an interactive Gradio-based GUI.

---

##  Objectives

The main objectives of this project are:

- Analyze the diabetes dataset
- Clean and prepare the data
- Handle invalid/missing values
- Perform feature engineering
- Visualize important patterns
- Select important features
- Handle class imbalance using SMOTE
- Train Logistic Regression and Random Forest models
- Perform hyperparameter tuning
- Evaluate and compare both models
- Analyze ROC-AUC performance
- Create a user-friendly prediction application

---

##  Dataset

The project uses the **Pima Indians Diabetes Dataset**.

The dataset contains information related to patients such as:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age
- Outcome

The target variable is:

- `0` → No Diabetes
- `1` → Diabetes

---

##  Machine Learning Workflow

```text
Dataset
   ↓
Raw Data Analysis
   ↓
Data Preparation
   ↓
Data Wrangling
   ↓
Feature Engineering
   ↓
Data Visualization
   ↓
Feature Selection
   ↓
Train-Test Split
   ↓
Data Imputation
   ↓
SMOTE Data Balancing
   ↓
Logistic Regression
   ↓
Hyperparameter Tuning
   ↓
Random Forest
   ↓
Hyperparameter Tuning
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
ROC-AUC Analysis
   ↓
Feature Importance
   ↓
Model Saving
   ↓
GUI Application
