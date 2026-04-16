import os
import subprocess
import time

def ensure_monthly_files_exist():
    print("Checking for monthly data files...")
    for month in range(1, 9):
        filename = f"api_data_2026_{month:02d}.csv"
        if not os.path.exists(filename):
            print(f"Creating dummy file for {filename} (just for testing the loop)...")
            # If you have your original data generation script, it's better to use that!
            # For now, we'll just copy the latest feature store if a month is missing
            if os.path.exists('feature_store_train_LATEST.csv'):
                 import shutil
                 shutil.copy('feature_store_train_LATEST.csv', filename)
            else:
                 with open(filename, 'w') as f:
                     f.write("dummy,data\n1,2\n") # Fallback dummy

def run_monthly_simulation():
    ensure_monthly_files_exist()

    print("\n" + "=" * 50)
    print("🚀 STARTING MLOPS TIME TRAVEL SIMULATION 🚀")
    print("=" * 50)

    # Loop through January (01) to August (08)
    for month in range(1, 9):
        month_str = f"{month:02d}"
        filename = f"api_data_2026_{month_str}.csv"

        print(f"\n📅 --- FAST FORWARDING TO 2026-{month_str} ---")

        if not os.path.exists(filename):
            print(f"[SKIP] {filename} not found.")
            continue

        # 1. Run the Batch Preprocessor
        print(f"⚙️  Processing {filename}...")
        try:
            subprocess.run(["python", "batch_preprocessing.py", filename], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error processing data:\n{e.stderr.decode()}")
            continue

        # 2. Run the Drift Monitor
        print(f"🔎 Running SciPy Drift Monitor...")
        result = subprocess.run(["python", "drift_monitor.py"], capture_output=True, text=True)

        # 3. Analyze the Output
        output = result.stdout

        if "[ALERT]" in output or "Drift detected" in output:
            print(f"🚨🚨 DRIFT DETECTED IN MONTH {month_str}! 🚨🚨")
            print("The retraining pipeline should have automatically been triggered.")
            print("Time travel paused. Please check MLflow UI for the new Champion vs Challenger results!")
            break  # Stop the simulation because the architecture did its job!
        else:
            print(f"✅ Data is stable. No drift detected. Moving to next month...")

        time.sleep(1)  # Pause for 1 second so you can read the terminal output


if __name__ == "__main__":
    run_monthly_simulation()