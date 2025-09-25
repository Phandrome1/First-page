import pandas as pd
import joblib
import os
from src.data_preprocessing import load_data, preprocess_data

def generate_submission(model=None, X_test=None, test_passenger_ids=None, output_path="submission/submission.csv"):
    # If model is None, load the saved one
    if model is None:
        model = joblib.load(os.path.join("models", "best_model.pkl"))

    # If no test data provided, reload
    if X_test is None or test_passenger_ids is None:
        train_df, test_df = load_data()
        _, _, X_test, test_passenger_ids = preprocess_data(train_df, test_df)

    # Predict on test set
    predictions = model.predict(X_test)

    # Create submission DataFrame
    submission = pd.DataFrame({
        "PassengerId": test_passenger_ids,
        "Survived": predictions
    })

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save CSV
    submission.to_csv(output_path, index=False)
    print(f" Submission file saved at {output_path}")
