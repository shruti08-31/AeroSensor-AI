# AeroSensor AI

## Smart Aircraft Engine Health Monitoring using Machine Learning and Deep Learning

---

## Overview

AeroSensor AI is an intelligent predictive maintenance system designed to monitor aircraft engine health using Machine Learning and Deep Learning techniques.

The project predicts the **Remaining Useful Life (RUL)** of aircraft engines and performs **failure classification** using multivariate time-series sensor data from the NASA C-MAPSS Turbofan Engine Degradation dataset.

Unlike traditional maintenance approaches that rely on fixed schedules or post-failure repairs, AeroSensor AI enables condition-based maintenance by identifying degradation patterns early, reducing maintenance costs, minimizing unexpected failures, and improving operational safety.

---

# Motivation

Modern aircraft engines continuously generate large volumes of sensor data during operation. These sensors monitor critical engine parameters such as temperature, pressure, fuel flow, compressor performance, and rotational speed.

Extracting meaningful degradation patterns from this high-dimensional time-series data is a challenging task. Traditional maintenance strategies often result in unnecessary maintenance or unexpected engine failures.

This project applies Artificial Intelligence techniques to predict engine degradation accurately and estimate the Remaining Useful Life before failure occurs.

---

# Problem Statement

Aircraft engines generate large volumes of multivariate time-series sensor data during operation.

The objective of this project is to develop an intelligent system capable of learning degradation patterns from historical sensor measurements and predicting the Remaining Useful Life (RUL) of an engine.

Formally,

> Given the historical multivariate sensor readings of an aircraft engine up to cycle *t*, predict the number of remaining operational cycles before the engine fails.

This is formulated as a supervised regression problem.

### Input

- Engine ID
- Cycle Number
- Three Operational Settings
- Twenty-One Sensor Measurements

### Output

- Remaining Useful Life (RUL)

### Key Challenges

- Learning temporal degradation patterns
- Handling noisy sensor measurements
- High-dimensional multivariate data
- Variable degradation rates across engines
- Feature redundancy

---

# System Architecture

The AeroSensor AI framework follows a multi-stage machine learning pipeline.

```
NASA C-MAPSS Dataset
          │
          ▼
 Data Acquisition
          │
          ▼
 Data Preprocessing
          │
          ▼
 Feature Engineering
          │
          ▼
 Feature Selection
          │
          ▼
 Machine Learning Models
          │
          ▼
 Deep Learning Models
          │
          ▼
 Model Evaluation
          │
          ▼
 Remaining Useful Life Prediction
          │
          ▼
 Failure Classification
```

---

# Pipeline

## 1. Data Acquisition

Raw sensor data is collected from the NASA C-MAPSS Turbofan Engine Degradation dataset.

The dataset contains run-to-failure trajectories of aircraft engines operating under different degradation conditions.

---

## 2. Data Preprocessing

The preprocessing stage includes:

- Removal of constant sensor features
- Handling missing values
- Feature normalization
- Sorting engine cycles
- Remaining Useful Life calculation
- Piecewise RUL target generation

These steps improve data quality and model stability.

---

## 3. Feature Engineering

Temporal information is extracted using:

- Rolling Mean
- Rolling Standard Deviation
- Exponential Moving Average (EMA)
- Sliding Window Sequences
- Lag Features

These engineered features help capture degradation trends more effectively.

---

## 4. Feature Selection

Random Forest feature importance is used to identify the most informative sensor measurements.

Benefits include:

- Reduced dimensionality
- Lower computational cost
- Reduced overfitting
- Improved prediction accuracy

---

## 5. Model Training

Multiple Machine Learning and Deep Learning models are trained and compared.

### Machine Learning Models

- Linear Regression
- Random Forest
- XGBoost
- K-Nearest Neighbors (KNN)

### Deep Learning Models

- Long Short-Term Memory (LSTM)
- Recurrent Neural Network (RNN)
- Convolutional Neural Network (CNN)
- CNN-SVM Hybrid Model

---

## 6. Model Evaluation

The trained models are evaluated using standard regression metrics:

- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score
- Mean Absolute Percentage Error (MAPE)

The best-performing model is selected based on predictive accuracy and generalization capability.

---

# Dataset

## NASA C-MAPSS Turbofan Engine Degradation Dataset

This project uses the NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) dataset, one of the most widely used benchmark datasets for Remaining Useful Life prediction.

The dataset simulates degradation trajectories of turbofan engines until complete failure under different operating conditions and fault modes.

---

## Dataset Structure

The dataset consists of four subsets.

| Dataset | Operating Conditions | Fault Modes |
|----------|----------------------|-------------|
| FD001 | Single | Single |
| FD002 | Multiple | Single |
| FD003 | Single | Multiple |
| FD004 | Multiple | Multiple |

This project focuses on **FD001**, which contains:

- 100 Training Engines
- 100 Testing Engines
- Single Operating Condition
- High Pressure Compressor (HPC) Degradation

Each record contains:

- Engine ID
- Cycle Number
- Three Operational Settings
- Twenty-One Sensor Measurements

---

# Models Implemented

## Regression Models

- Linear Regression
- Random Forest
- XGBoost
- LSTM

## Failure Classification Models

- Binary Classification using CNN
- Binary Classification using CNN-SVM
- Binary and Multiclass Classification using RNN
- Binary and Multiclass Classification using 1D CNN

---

# Key Features

- Remaining Useful Life Prediction
- Binary Failure Classification
- Multiclass Engine Health Classification
- Time-Series Feature Engineering
- Random Forest Feature Selection
- Machine Learning and Deep Learning Model Comparison
- Predictive Maintenance Framework

---

# Applications

The framework can be applied to several industrial domains, including:

- Aircraft Engine Health Monitoring
- Predictive Maintenance
- Industrial Machinery Monitoring
- Smart Manufacturing
- Railway Maintenance
- Automotive Diagnostics
- IoT-Based Asset Monitoring

---

# Future Work

Future enhancements include:

- Transformer-based architectures
- Explainable AI using SHAP and LIME
- Real-time prediction pipeline
- Edge AI deployment
- Multi-condition dataset support (FD002–FD004)
- Cloud deployment and dashboard integration

---

# Authors

**Shruti Prasad**  
B.Tech Artificial Intelligence & Data Science  
Gati Shakti Vishwavidyalaya, Vadodara

**Shreya Singh**  
B.Tech Artificial Intelligence & Data Science  
Gati Shakti Vishwavidyalaya, Vadodara

---

# Acknowledgement

The authors sincerely thank **Dr. Vipul Kumar Mishra** for his continuous guidance, valuable suggestions, and support throughout the development of this project.
