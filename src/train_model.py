import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

from src.data_preprocessing import load_data, preprocess_data, build_preprocessor


def train_and_evaluate():
    # Load datasets
    train_df, test_df = load_data()
    X, y, X_test, test_passenger_ids = preprocess_data(train_df, test_df)

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Models to compare
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    best_model = None
    best_score = 0

    # Train each model
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        print(f"\n{name}")
        print("Accuracy:", acc)
        print(classification_report(y_val, y_pred))

        if acc > best_score:
            best_score = acc
            best_model = pipeline

    #  Save the best model inside project_root/models/
    model_path = os.path.join("models", "best_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"\n Best model saved at {model_path} with accuracy {best_score:.4f}")

    return best_model, X_test, test_passenger_ids

