import pandas as pd
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

#Setup MLflow tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Fraud_Detection_Retraining")

print("--- Starting Automated Retraining Pipeline ---")

#Load the LATEST data
df_train = pd.read_csv('feature_store_train_LATEST.csv')
df_test = pd.read_csv('feature_store_test_LATEST.csv')

target = 'fraud_flag'
X_train = df_train.drop(target, axis=1)
y_train = df_train[target]
X_test = df_test.drop(target, axis=1)
y_test = df_test[target]

model_name = "fraud_detection_model"

#Start the MLflow run
with mlflow.start_run(run_name="Automated_Drift_Retrain"):
    n_estimators = 100
    class_weight = 'balanced'

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("class_weight", class_weight)

    #Train the model
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, class_weight=class_weight)
    rf_model.fit(X_train, y_train)

    #Evaluate
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    y_pred_hard = (y_pred_proba >= 0.10).astype(int)

    precision = precision_score(y_test, y_pred_hard, zero_division=0)
    recall = recall_score(y_test, y_pred_hard, zero_division=0)
    f1 = f1_score(y_test, y_pred_hard, zero_division=0)

    #Log metrics to MLflow
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    client = MlflowClient()

    try:
        #Search for the most recently registered version of our model
        latest_versions = client.get_latest_versions(name=model_name)

        if latest_versions:
            champion_version = latest_versions[-1].version
            champion_uri = f"models:/{model_name}/{champion_version}"

            print(f"\n--- Fetching previous Champion Model (Version {champion_version}) for comparison ---")
            champion_model = mlflow.sklearn.load_model(champion_uri)

            #Test the old model on the drifted test data
            champ_pred_proba = champion_model.predict_proba(X_test)[:, 1]
            champ_pred_hard = (champ_pred_proba >= 0.10).astype(int)
            champ_f1 = f1_score(y_test, champ_pred_hard, zero_division=0)

            #Print the Supervisor Report
            print("\n" + "=" * 50)
            print("SUPERVISOR MODEL COMPARISON REPORT")
            print("=" * 50)
            print(f"Old Model (V{champion_version}) F1 Score:  {champ_f1:.4f}")
            print(f"New Model F1 Score: {new_f1:.4f}")
            print("-" * 50)

            if f1 > champ_f1:
                print("RESULT: The New Model adapted to the drift and PERFORMS BETTER!")
                mlflow.set_tag("supervisor_status", "Approved - Upgrade Recommended")
            else:
                print("RESULT: The New Model is WORSE or EQUAL. Supervisor manual review required.")
                mlflow.set_tag("supervisor_status", "Warning - Degradation Detected")
            print("=" * 50 + "\n")

            #Log the comparison metrics to MLflow
            mlflow.log_metric("previous_model_f1", champ_f1)
            mlflow.log_metric("f1_improvement", f1 - champ_f1)

    except Exception as e:
        print(f"\n--- No previous model found in registry to compare against. This must be the first run! ---")

    #Model Registry
    mlflow.sklearn.log_model(
        sk_model=rf_model,
        artifact_path="random_forest_model",
        registered_model_name=model_name
    )

print(f"--- Retraining Complete! New model registered with F1: {f1:.4f} ---")
print("To view the Model Registry, run this in your terminal: mlflow ui --backend-store-uri sqlite:///mlflow.db")