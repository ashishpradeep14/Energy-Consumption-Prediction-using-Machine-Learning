# ⚡ Energy Consumption Prediction with Machine Learning and Deep Learning
This project aims to predict appliance energy consumption in a smart home environment using a variety of machine learning models including Random Forest, XGBoost, SVR, LSTM, ARIMA, SARIMA, and AdaBoost. The data includes environmental and weather conditions, enabling precise modeling of real-world scenarios.

## 📊 Project Features
### 📁 Data Cleaning & Preprocessing

### 🏗️ Feature Engineering: Rolling means, ratios, interactions, seasonal features, lagging, statistical aggregates, etc.

### 📈 Exploratory Data Analysis: Correlation heatmaps, KDE plots, boxplots, scatter plots.

### 🤖 Modeling Techniques:

Linear Regression

Decision Tree, Random Forest

XGBoost, AdaBoost

SVR (Support Vector Regression)

LSTM (Deep Learning)

ARIMA & SARIMA (Time Series Forecasting)

K-Nearest Neighbors (KNN)

## 🧪 Evaluation Metrics: MAE, RMSE, R²

## 🔁 Hyperparameter Tuning: Grid search for Random Forest

## 🧠 Custom Pipeline: Modularized preprocessing with scikit-learn pipelines

## 📦 Model Deployment: Serialized model with joblib and prediction function for live input

## 🗂️ File Structure
plaintext
Copy
Edit
📦 energy-consumption-predictor
 ┣ 📜 energy_consumption.py   # Full notebook converted to .py script
 ┣ 📄 README.md               # Project overview
 ┣ 📦 models/
 ┃ ┗ 📄 random_forest_model.pkl
 ┃ ┗ 📄 preprocessing_pipeline.pkl
 ┗ 📊 data/
   ┗ 📄 energydata_complete.csv  # (Not included for privacy)
## 📌 Requirements
Install dependencies:

bash
Copy
Edit
pip install pandas numpy scikit-learn matplotlib seaborn xgboost tensorflow statsmodels joblib
## 🧪 How to Run
Clone the repo:

bash
Copy
Edit
git clone https://github.com/your-username/energy-consumption-predictor.git
cd energy-consumption-predictor
Place the dataset (energydata_complete.csv) in the appropriate directory.

Run the script:

bash
Copy
Edit
python energy_consumption.py
(Optional) Use the model for prediction:

python
Copy
Edit
from energy_consumption import predict_energy
prediction = predict_energy(user_input_dict, X_train)
## 📊 Sample Result (Model Comparison)
Model	MAE	RMSE	R²
Linear Regression	60.43	92.15	0.57
Random Forest	45.11	68.70	0.78
XGBoost	43.88	66.90	0.79
LSTM	49.34	73.28	0.74
SARIMA	59.22	88.16	0.59
AdaBoost	54.21	81.33	0.65

## 🧠 Example Prediction Input
python
Copy
Edit
user_input = {
  "T1": 19.0, "RH_1": 47.0, ..., "Visibility": 40.0, "Tdewpoint": 10.0,
  "rv1": 0.5, "rv2": 0.5
}
python
Copy
Edit
predict_energy_live(user_input, rf_model, X_train)
## 🔮 Future Improvements
Integrate real-time energy monitoring dashboard

Use deep learning with temporal convolution or attention models

Support multi-step forecasting

Build REST API with Flask/FastAPI

📜 License
MIT License
