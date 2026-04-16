# MLOps Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20API-lightgrey)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Random%20Forest-orange)
![MLflow](https://img.shields.io/badge/MLflow-Model%20Registry-blue)

An end-to-end Machine Learning Operations (MLOps) pipeline designed to detect fraudulent applications for a government agency. This system features a dynamic Flask API, token-based security, a local data lake, statistical data drift monitoring, and an automated Champion vs. Challenger retraining pipeline managed by MLflow.

## Key Features

* **Interactive Web Frontend:** A Bootstrap-styled HTML5 interface with built-in data validation for users to submit applications.
* **REST API:** A Flask backend.
* **Continuous Learning (Data Lake):** All incoming API requests are logged in real-time to a local JSONL Data Lake to capture future training data.
* **Statistical Drift Monitoring:** Utilizes the Kolmogorov-Smirnov (KS) Two-Sample Test to monitor economic shifts in incoming data against the baseline training distributions.
* **Automated Retraining Pipeline:** If drift is detected, the system automatically triggers a retraining pipeline using Scikit-Learn's `RandomForestClassifier`.
* **Dynamic Model Registry:** Powered by an MLflow SQLite database, ensuring the API always loads the latest verified `.joblib` model at server startup without hardcoding.

---

## Project Structure

```text
Fraud_Detection_Project/
│
├── ⚙️ Environment & Setup
│   ├── requirements.txt                    # Project dependencies
│   └── README.md                           # Project documentation
│
├── 🧠 Core MLOps Pipeline
│   ├── model.py                            # Initial baseline Random Forest training
│   ├── batch_preprocessing.py              # Simulates human audits and splits data
│   ├── drift_monitor.py                    # SciPy statistical drift monitor (KS-Test)
│   ├── retrain_pipeline.py                 # Champion vs. Challenger training & evaluation
│   └── simulate_time_travel.py             # Orchestration script testing 8-months of data
│
├── 🌐 Production API & Frontend
│   ├── app.py                              # Secure Flask server & dynamic MLflow loading
│   ├── test_api.py                         # Authenticated Python script to test the backend
│   └── templates/
│       └── index.html                      # HTML/Bootstrap web frontend
│
├── 🗄️ Databases & Model Registry
│   ├── mlflow.db                           # SQLite database acting as the MLflow Registry
│   ├── mlruns/                             # MLflow artifact storage
│   ├── preprocessor.joblib                 # Serialized data transformation pipeline
│   └── fraud_model.joblib                  # Fallback serialized model
│
└── 📊 Data Lake & Feature Store
    ├── local_data_lake_raw.jsonl           # Local Data Lake capturing JSON API requests
    ├── feature_store_train_LATEST.csv      # Processed training data
    └── feature_store_test_LATEST.csv       # Processed testing data
```

---
## Data

The original data to train the model was drawn from Kaggle and can be accessed via the following link: https://www.kaggle.com/datasets/prajwaldongre/loan-application-and-transaction-fraud-detection?resource=download

___
## Quick Start Guide
1. Install Dependencies
Ensure you have Python 3.9+ installed. Install the required packages:

```
pip install -r requirements.txt
```

2. Start the MLflow Tracking Server
To view model versions, metrics, and the Champion/Challenger supervisor tags, start the MLflow UI on Port 5000:

```
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Access the dashboard at: ```http://127.0.0.1:5000```


3. Start the API & Web App
Open a new terminal window and start the secure Flask server on Port 5000:

```
python app.py
```
Access the web application at: ```http://127.0.0.1:5000```

---
## Testing the Pipeline
### Test the Production API
You can test the secure ```/predict``` endpoint by running the included test script. Ensure your Flask server is running first.

```
python test_api.py
```
(Note: This script uses a hardcoded API key (boba_fett) to bypass the security middleware).
