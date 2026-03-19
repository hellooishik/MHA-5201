import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df


def clean_data(df):
    # Drop ID column
    df = df.drop("customerID", axis=1)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

    # Fill numeric missing values
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Fill categorical missing values
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


def encode_data(df):
    # Convert target column
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # One-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    return df


def split_data(df):
    from sklearn.model_selection import train_test_split

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    return train_test_split(X, y, test_size=0.2, random_state=42)