# ✈️ AeroSensor AI

**AeroSensor AI** is a machine learning project focused on **aircraft engine health monitoring** using sensor data. It uses the NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) turbofan engine degradation dataset to solve two closely related predictive maintenance problems:

1. **Remaining Useful Life (RUL) Prediction** — estimating how many operational cycles an engine has left before failure.
2. **Failure Prediction / Classification** — predicting whether an engine is close to failure (binary/multiclass classification) based on live sensor readings.

The goal is to demonstrate how data analytics, feature engineering, and a range of machine learning / deep learning models can be applied to real-world predictive maintenance problems in aviation.

---

## 📌 Project Overview

Aircraft engines are fitted with dozens of sensors that continuously record operational parameters (temperature, pressure, fan speed, etc.). Over time, engine performance degrades until failure. Being able to predict **when** an engine will fail — and **how much life it has left** — helps airlines schedule maintenance proactively, reduce downtime, and improve safety.

This project walks through the full pipeline:

- Cleaning and wrangling raw multivariate time-series sensor data
- Exploratory data analysis and feature selection/importance studies
- Building multiple models for **RUL regression**
- Building multiple models for **failure classification**
- Comparing model performance across classical ML and deep learning approaches

---

## 🗂️ Repository Structure

```
AeroSensor-AI/
│
├── Dataset/
│   ├── train_FD001.txt          # Raw NASA C-MAPSS training data (FD001 subset)
│   ├── RUL_FD001.txt            # True Remaining Useful Life values for the test set
│   └── train_cleaned_data.csv   # Cleaned/processed dataset used across notebooks
│
├── Data_analysis/
│   ├── Data_Wrangling.ipynb         # Loading raw data, naming columns, computing RUL
│   ├── feature selection.ipynb      # Studying the impact of feature selection on model performance
│   └── Feature Importance.ipynb     # Perturbation, missing-value, and permutation importance analysis
│
├── RUL Prediction/
│   ├── Linear_Regression.ipynb                  # Baseline linear regression model
│   ├── RandomRegressor_Prediction_RUL.ipynb     # Random Forest Regressor
│   ├── RUL_Prediction_Using_GB_XGBoost..ipynb   # Gradient Boosting + XGBoost (feature selection via RF)
│   ├── Exponential Degradation Model.ipynb      # Degradation-curve based RUL estimation
│   ├── Similarity-based RUL Prediction.ipynb    # Similarity/nearest-neighbor based RUL estimation
│   └── LSTM RUL Prediction.ipynb                # Sequence-based deep learning (LSTM) regression
│
├── Failure Prediction Classification/
│   ├── LSTM - binary classification.ipynb                 # LSTM for binary failure prediction
│   ├── RNN - binary and multiclass classification.ipynb   # RNN for binary/multiclass classification
│   ├── 1D CNN - binary and multiclass classification.ipynb# 1D CNN for time-series classification
│   └── CNN-SVM binary classification.ipynb                # Hybrid CNN feature extractor + SVM classifier
│
└── README.md
```

---

## 📊 Dataset

This project uses the **NASA Turbofan Engine Degradation Simulation Dataset (C-MAPSS, FD001 subset)**.

Each row in `train_FD001.txt` represents one operational cycle of one engine and contains:

- `UnitNumber` — engine ID
- `Cycle` — operational cycle number
- `OpSet1`, `OpSet2`, `OpSet3` — operational settings
- `SensorMeasure1` … `SensorMeasure21` — 21 sensor readings (temperature, pressure, speed, etc.)

The **Remaining Useful Life (RUL)** target is derived by subtracting the current cycle from the maximum cycle observed for that engine (i.e., cycles remaining until failure).

---

## 🧠 Models Implemented

### Remaining Useful Life (Regression)
| Approach | Notebook |
|---|---|
| Linear Regression | `RUL Prediction/Linear_Regression.ipynb` |
| Random Forest Regressor | `RUL Prediction/RandomRegressor_Prediction_RUL.ipynb` |
| Gradient Boosting + XGBoost | `RUL Prediction/RUL_Prediction_Using_GB_XGBoost..ipynb` |
| Exponential Degradation Model | `RUL Prediction/Exponential Degradation Model.ipynb` |
| Similarity-based Estimation | `RUL Prediction/Similarity-based RUL Prediction.ipynb` |
| LSTM (Deep Learning) | `RUL Prediction/LSTM RUL Prediction.ipynb` |

### Failure Prediction (Classification)
| Approach | Notebook |
|---|---|
| LSTM (Binary) | `Failure Prediction Classification/LSTM - binary classification.ipynb` |
| RNN (Binary & Multiclass) | `Failure Prediction Classification/RNN - binary and multiclass classification.ipynb` |
| 1D CNN (Binary & Multiclass) | `Failure Prediction Classification/1D CNN - binary and multiclass classification.ipynb` |
| CNN-SVM Hybrid | `Failure Prediction Classification/CNN-SVM binary classification.ipynb` |

---

## ⚙️ Tech Stack

- **Language:** Python
- **Data Handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Classical ML:** scikit-learn (Linear Regression, Random Forest, Gradient Boosting, SVM), XGBoost
- **Deep Learning:** TensorFlow / Keras (LSTM, RNN, 1D CNN)
- **Environment:** Jupyter Notebook / Anaconda

---

## 🚀 How to Run This Project

Follow these steps to set up and run the project on your own machine:

### 1. Clone the repository
```bash
git clone https://github.com/shruti08-31/AeroSensor-AI.git
cd AeroSensor-AI
```

### 2. Set up a Python environment
It's recommended to use a virtual environment (via `venv` or Anaconda):

```bash
# Using venv
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# OR using Anaconda
conda create -n aerosensor python=3.10
conda activate aerosensor
```

### 3. Install the required libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow jupyter
```

### 4. Update file paths (if needed)
Some notebooks reference a local working directory (e.g. `os.chdir("C:/Data/aircraft/")`). Update this path — or simply remove it — so it points to the `Dataset/` folder in this repository on your machine.

### 5. Run the notebooks in order
For the clearest walkthrough, open and run the notebooks in this sequence:

1. **`Data_analysis/Data_Wrangling.ipynb`** — load raw data, assign column names, compute RUL
2. **`Data_analysis/feature selection.ipynb`** and **`Data_analysis/Feature Importance.ipynb`** — explore which sensors matter most
3. Any notebook inside **`RUL Prediction/`** — to train and evaluate a Remaining Useful Life model of your choice
4. Any notebook inside **`Failure Prediction Classification/`** — to train and evaluate a failure classification model of your choice

Launch Jupyter with:
```bash
jupyter notebook
```
then open the notebooks from the browser interface and run all cells (`Cell` → `Run All`).

### 6. Explore and compare results
Each notebook prints/plots evaluation metrics (MAE, RMSE, R² for regression; accuracy, precision/recall, confusion matrix for classification) so you can compare approaches directly.

---

## 📈 Future Improvements

- Combine the multiple FD00X subsets (FD002–FD004) of the C-MAPSS dataset for a more generalized model
- Package the best-performing model into a deployable API or dashboard (e.g., using Flask/FastAPI + Power BI)
- Add automated hyperparameter tuning (GridSearch/Optuna) across all models
- Build a unified pipeline script instead of separate notebooks

---

## 👩‍💻 About Me

**Shruti Prasad**
B.Tech in Artificial Intelligence & Data Science
Gati Shakti Vishwavidyalaya, Vadodara, Gujarat, India

I am passionate about Data Analytics, Machine Learning, Business Intelligence, and AI-driven solutions. I enjoy building end-to-end data projects that combine Python, SQL, Power BI, and machine learning to solve real-world business problems.

### Connect with Me
- 💼 LinkedIn: [linkedin.com/in/shruti-prasad-35123636b](https://www.linkedin.com/in/shruti-prasad-35123636b/)
- 💻 GitHub: [github.com/shruti08-31](https://github.com/shruti08-31)

If you found this project useful or have any suggestions, feel free to connect with me or open an issue in this repository.
