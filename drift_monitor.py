import pandas as pd
from scipy.stats import ks_2samp
import subprocess

#Load reference data
reference_data = pd.read_csv('feature_store_train.csv')

#Load current (simulated) data
current_data = pd.read_csv('new_api_data_processed.csv')

#Define numerical features to monitor
features_to_monitor = [
    'num__monthly_income',
    'num__interest_rate_offered',
    'num__applicant_age',
    'num__cibil_score',
    'num__income_to_aid_ratio',
    'num__loan_amount_requested',
    'num__debt_to_income_ratio',
    'num__number_of_dependents',
    'num__loan_tenure_months',
    'num__existing_emis_monthly'
]

#Statistical test function to detect drift
def check_data_drift(ref_df, curr_df, features, threshold=0.05):
    print("--- Initiating Data Drift Check ---")
    drifted_features = 0

    for feature in features:
        stat, p_value = ks_2samp(ref_df[feature].dropna(), curr_df[feature].dropna())

        if p_value < threshold:
            print(f"[ALERT] Drift detected in '{feature}' (p-value: {p_value:.4f})")
            drifted_features += 1
        else:
            print(f"[OK] No drift in '{feature}' (p-value: {p_value:.4f})")

    return drifted_features


#Execute the check
drift_count = check_data_drift(reference_data, current_data, features_to_monitor)

#Trigger logic
if drift_count >= 2:
    print("\n[WARNING] Significant overall data drift detected!")
    print("Triggering automated retraining pipeline...")

    subprocess.run(["python", "retrain_pipeline.py"])
else:
    print("\n[SUCCESS] Data distribution is stable. No retraining required.")