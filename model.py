import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, \
    ConfusionMatrixDisplay
import joblib

# Load the feature store CSV
df_train = pd.read_csv('feature_store_train.csv')
df_test = pd.read_csv('feature_store_test.csv')

# Split back into X and y
target = 'fraud_flag'
X_train_processed = df_train.drop(target, axis=1)
y_train = df_train[target]

X_test_processed = df_test.drop(target, axis=1)
y_test = df_test[target]

#Initialize and train model
print("\n--- Training the Random Forest Baseline ---")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train_processed, y_train)

#Predict fraud probability
y_pred_proba = rf_model.predict_proba(X_test_processed)[:, 1]

custom_threshold = 0.10
y_pred_hard = (y_pred_proba >= custom_threshold).astype(int)


#Statistical evaluation
print(f"\n--- Evaluation Metrics (Threshold: {custom_threshold}) ---")
print(f"Precision: {precision_score(y_test, y_pred_hard):.4f} (How many flagged were actually fraud?)")
print(f"Recall:    {recall_score(y_test, y_pred_hard):.4f} (How much of the total fraud did we catch?)")
print(f"F1-Score:  {f1_score(y_test, y_pred_hard):.4f} (Harmonic mean of Precision and Recall)")

print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred_hard)
print(pd.DataFrame(cm,
                   index=['Actual Legitimate (0)', 'Actual Fraud (1)'],
                   columns=['Predicted Legitimate (0)', 'Predicted Fraud (1)']))

#Confusion matrix graph
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Legitimate', 'Fraud'])

fig, ax = plt.subplots(figsize=(8, 6))

disp.plot(cmap='Blues', ax=ax, values_format='d')

plt.title(f'Confusion Matrix (Threshold: {custom_threshold})', fontsize=16, fontweight='bold', pad=15)
plt.tight_layout()

plt.savefig('presentation_confusion_matrix.png', dpi=300)
print("-> Graph saved as 'presentation_confusion_matrix.png'")

plt.show()

joblib.dump(rf_model, 'fraud_model.joblib')

'''Feature importance check

importances = rf_model.feature_importances_
feature_names = X_train_processed.columns

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

print("\n--- Top 5 Most Important Features ---")
print(feature_importance_df.sort_values(by='Importance', ascending=False).head(5).to_string(index=False))


y_pred_proba = rf_model.predict_proba(X_test_processed)[:, 1]

print(f"\nHighest predicted fraud probability: {y_pred_proba.max():.4f}")
print(f"Average predicted fraud probability: {y_pred_proba.mean():.4f}")'''
