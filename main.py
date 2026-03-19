from src.data_preprocessing import load_data, clean_data, encode_data, split_data
from src.train_model import train_models, save_models
from src.evaluate import evaluate_models

def run_pipeline():
    # Load data
    df = load_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # Process data
    df = clean_data(df)
    df = encode_data(df)

    # Split
    X_train, X_test, y_train, y_test = split_data(df)

    # Train
    models = train_models(X_train, y_train)

    # Evaluate
    evaluate_models(models, X_test, y_test)

    # Save
    save_models(models)

if __name__ == "__main__":
    run_pipeline()