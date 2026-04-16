import pandas as pd
import numpy as np
import calendar
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)


def generate_monthly_data(year, month, is_drifting=False):
    num_days = calendar.monthrange(year, month)[1]
    num_applications = np.random.randint(800, 1200)

    start_date = datetime(year, month, 1)
    random_days = np.random.randint(0, num_days, num_applications)
    dates = [start_date + timedelta(days=int(d)) for d in random_days]

    if is_drifting:
        income_mean = 34000
        debt_ratio_mean = 14.5
        cibil_mean = 630
        interest_mean = 13.5
    else:
        income_mean = 51000
        debt_ratio_mean = 8.57
        cibil_mean = 699
        interest_mean = 10.53

    # Generate the simulated data
    data = {
        "application_date": [d.strftime('%Y-%m-%d') for d in dates],

        # Categorical columns
        "gender": np.random.choice(["Male", "Female"], num_applications, p=[0.55, 0.45]),
        "purpose_of_loan": np.random.choice(["Home Renovation", "Debt Consolidation", "Medical Emergency", "Education", "Business Expansion", "Vehicle Purchase", "Wedding"],
                                            num_applications),
        "employment_status": np.random.choice(["Salaried", "Self-Employed", "Unemployed", "Business Owner", "Retired", "Student"], num_applications),
        "loan_type": np.random.choice(["Personal Loan", "Home Loan", "Education Loan", "Car Loan", "Business Loan"], num_applications),
        "property_ownership_status": np.random.choice(["Owned", "Rented", "Jointly Owned"], num_applications),

        # Numerical columns (These will shift if drift is triggered!)
        "monthly_income": np.clip(np.random.normal(income_mean, 15000, num_applications), 10000, 200000),
        "debt_to_income_ratio": np.clip(np.random.normal(debt_ratio_mean, 15, num_applications), 0, 150),
        "cibil_score": np.clip(np.random.normal(cibil_mean, 80, num_applications), 400, 900).astype(int),
        "interest_rate_offered": np.clip(np.random.normal(interest_mean, 2.5, num_applications), 5.0, 20.0),

        # Other numericals
        "loan_amount_requested": np.clip(np.random.normal(150000, 50000, num_applications), 100000, 2000000),
        "number_of_dependents": np.random.randint(0, 5, num_applications),
        "loan_tenure_months": np.random.choice([12, 24, 36, 48, 60, 120, 360], num_applications),
        "applicant_age": np.clip(np.random.normal(38, 10, num_applications), 21, 65).astype(int),
        "existing_emis_monthly": np.clip(np.random.normal(5000, 3000, num_applications), 0, 20000)
    }

    df = pd.DataFrame(data)

    # Save to CSV
    filename = f"api_data_{year}_{month:02d}.csv"
    df.to_csv(filename, index=False)

    status = "DRIFTED" if is_drifting else "Normal"
    print(f"Generated {filename} ({num_applications} records) - Status: {status}")


# Generate a full year of data
print("--- Generating MLOps Test Data (Year: 2026) ---")
for month in range(1, 13):
    apply_drift = True if month >= 7 else False
    generate_monthly_data(2026, month, is_drifting=apply_drift)

print("\n--- Test Data Generation Complete! ---")