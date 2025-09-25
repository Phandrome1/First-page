from src.data_preprocessing import load_data, preprocess_data
from src.train_model import train_and_evaluate
from src.predict import generate_submission
import os

def main():
    print(" Titanic ML Pipeline started...")

    # Step 1: Load
    train_df, test_df = load_data("data/train.csv", "data/test.csv")
    print(" Data loaded successfully!")

    # Step 2: Preprocess
    X, y, X_test, test_passenger_ids = preprocess_data(train_df, test_df)
    print(" Preprocessing completed!")

    # Step 3: Train & Save
    best_model, X_test, test_passenger_ids = train_and_evaluate()
    print(" Model trained & saved!")

    # Step 4: Predict & Save Submission
    os.makedirs("submission", exist_ok=True)
    generate_submission(best_model, X_test, test_passenger_ids, output_path="submission/submission.csv")
    print(" Submission file created at submission/submission.csv")

    print(" Pipeline finished successfully!")

if __name__ == "__main__":
    main()
