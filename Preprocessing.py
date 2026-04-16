import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

"""#Convert csv file into JSON file to mimic data coming from online application forms
df = pd.read_csv('C:/Users/Admin/OneDrive/OneDrive - IU International University of Applied Sciences/IU/Project from Model to Production/loan_applications.csv')

json_data = df.to_json(orient='records', indent=4)

open('applications.json', 'w') as f:
    f.write(json_data)"""


#Raw storage
def load_raw_data(file_path):
    print("--- Loading Raw Data from Data Lake ---")
    return pd.read_csv(file_path)

#Feature engineering
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


#Descriptive statistics and validation
def perform_initial_checks(df, target_column):
    print("\n--- Descriptive Statistics ---")
    print(df.describe())

    print("\n--- Data Validation: Missing Values ---")
    print(df.isnull().sum())

    print("\n--- Class Imbalance Check ---")
    imbalance = df[target_column].value_counts(normalize=True) * 100
    print(f"Distribution:\n{imbalance}")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))

    ax = sns.countplot(
        data=df,
        x=target_column,
        hue=target_column,
        legend=False,
        palette={0: '#4C72B0', 1: '#C44E52', '0': '#4C72B0', '1': '#C44E52'}
    )
    plt.title('Class Imbalance: Fraud vs. Legitimate Applications', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Fraud Flag (0 = Legitimate, 1 = Fraud)', fontsize=12)
    plt.ylabel('Number of Applications', fontsize=12)

    total_applications = len(df)
    for p in ax.patches:
        height = p.get_height()
        percentage = f'{(100 * height / total_applications):.1f}%'

        ax.annotate(percentage,
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold',
                    xytext=(0, 5), textcoords='offset points')

    plt.tight_layout()

    plt.savefig('presentation_class_imbalance.png', dpi=300)
    print("-> Graph saved as 'presentation_class_imbalance.png'")

    plt.show()

    return imbalance


#ETL Pipeline: Preprocessing & Encoding
def build_preprocessing_pipeline(df, numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor


df = load_raw_data('C:/Users/Admin/OneDrive/OneDrive - IU International University of Applied Sciences/IU/Project from Model to Production/loan_applications.csv')

df = engineer_features(df)

columns_to_drop = ['application_id', 'customer_id', 'residential_address', 'fraud_type', 'loan_status']
df = df.drop(columns_to_drop, axis=1, errors='ignore')

target = 'fraud_flag'
numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32']).columns.drop(target).tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

perform_initial_checks(df, target)

X = df.drop(target, axis=1)
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = build_preprocessing_pipeline(df, numeric_cols, categorical_cols)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()
feature_store_df = pd.DataFrame(X_train_processed, columns=feature_names)

feature_store_df['fraud_flag'] = y_train.values

feature_store_df.to_csv('feature_store_train.csv', index=False)


feature_store_test_df = pd.DataFrame(X_test_processed, columns=feature_names)
feature_store_test_df['fraud_flag'] = y_test.values
feature_store_test_df.to_csv('feature_store_test.csv', index=False)
print("--- Train and Test Features successfully stored in CSVs ---")

joblib.dump(preprocessor, 'preprocessor.joblib')