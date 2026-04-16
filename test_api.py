import requests
import json

# The URL of your local Flask API
url = 'http://127.0.0.1:5000/predict'

# Simulating JSON payload the web frontend would send
mock_application_data = {
    "application_date": "2026-04-12",
    "loan_amount_requested": 450000,
    "debt_to_income_ratio": 14.8,
    "gender": "Male",
    "number_of_dependents": 4,
    "purpose_of_loan": "Medical Emergency",
    "interest_rate_offered": 14.5,
    "loan_tenure_months": 12,
    "applicant_age": 22,
    "employment_status": "Unemployed",
    "existing_emis_monthly": 12000.00,
    "loan_type": "Personal Loan",
    "property_ownership_status": "Rented",
    "cibil_score": 450,
    "monthly_income": 15000
}

# Create a dictionary for the headers holding our secret key
auth_headers = {
    'x-api-key': 'boba_fett'
}

print("--- Sending Application to API ---")

try:
    # Send the POST request to the API
    response = requests.post(url, json=mock_application_data, headers=auth_headers)

    if response.status_code == 200:
        print("\n[SUCCESS] API responded with:")
        print(json.dumps(response.json(), indent=4))
    else:
        print(f"\n[ERROR] API returned status code: {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"Failed to connect to API: {e}")