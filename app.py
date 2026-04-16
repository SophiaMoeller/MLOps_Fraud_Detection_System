from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from functools import wraps
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import json
from datetime import datetime

app = Flask(__name__)

AUTHORIZED_STAFF_KEYS = {
    "boba_fett": "Junior Analyst",
    "jango_fett": "MLOps Supervisor"
}


def require_api_key(f):

    @wraps(f)
    def decorated_function(*args, **kwargs):
        provided_key = request.headers.get('x-api-key')

        if provided_key and provided_key in AUTHORIZED_STAFF_KEYS:
            staff_role = AUTHORIZED_STAFF_KEYS[provided_key]
            print(f"[SECURITY] Access granted to: {staff_role}")
            return f(*args, **kwargs)
        else:
            print(f"[SECURITY ALERT] Blocked unauthorized access attempt!")
            return jsonify({
                "status": "error",
                "message": "Access Denied: Missing or Invalid API Key."
            }), 401

    return decorated_function

#Load artifacts at server start
print("Loading preprocessor...")
preprocessor = joblib.load('preprocessor.joblib')

print("Connecting to MLflow Model Registry...")
mlflow.set_tracking_uri("sqlite:///mlflow.db")
model_name = "fraud_detection_model"

try:
    #Ask the registry for the latest version
    client = MlflowClient()
    latest_versions = client.get_latest_versions(name=model_name)

    if not latest_versions:
        raise ValueError("No models found in registry.")

    latest_version = latest_versions[-1].version
    model_uri = f"models:/{model_name}/{latest_version}"

    print(f"--- Successfully loaded {model_name} (Version {latest_version}) ---")
    model = mlflow.sklearn.load_model(model_uri)

except Exception as e:
    print(f"[WARNING] Could not connect to MLflow: {e}")
    print("Falling back to local static fraud_model.joblib file...")
    model = joblib.load('fraud_model.joblib')


def engineer_features(data):
    df_engineered = data.copy()
    if 'application_date' in df_engineered.columns:
        df_engineered['application_date'] = pd.to_datetime(df_engineered['application_date'])
        df_engineered['app_year'] = df_engineered['application_date'].dt.year
        df_engineered['app_month'] = df_engineered['application_date'].dt.month
        df_engineered['app_day'] = df_engineered['application_date'].dt.day
        df_engineered = df_engineered.drop('application_date', axis=1)

    if 'loan_amount_requested' in df_engineered.columns and 'monthly_income' in df_engineered.columns:
        df_engineered['income_to_aid_ratio'] = df_engineered['loan_amount_requested'] / (
                    df_engineered['monthly_income'] + 1e-5)
    return df_engineered

# Serve the web frontend
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

#Define API endpoint
@app.route('/predict', methods=['POST'])
#@require_api_key
def predict_fraud():
    try:
        json_data = request.get_json()

        #Add real-time timestamp
        lake_record = json_data.copy()
        lake_record['server_timestamp'] = datetime.now().isoformat()

        #Append it to a local JSONL file
        with open("local_data_lake_raw.jsonl", "a") as f:
            f.write(json.dumps(lake_record) + "\n")

        df_incoming = pd.DataFrame([json_data])

        df_engineered = engineer_features(df_incoming)

        X_processed_array = preprocessor.transform(df_engineered)

        feature_names = preprocessor.get_feature_names_out()
        X_processed_df = pd.DataFrame(X_processed_array, columns=feature_names)

        prediction = model.predict(X_processed_df)
        fraud_probability = model.predict_proba(X_processed_df)[0][1]

        pred_value = int(prediction[0])
        prob_value = float(fraud_probability)

        result_label = "Fraud" if pred_value == 1 else "Not Fraud"

        return jsonify({
            "status": "success",
            "prediction": result_label,
            "fraud_probability": f"{prob_value * 100:.1f}%",
            "flagged_for_review": bool(fraud_probability > 0.01)
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


# Start the Flask Server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)