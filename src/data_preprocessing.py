import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_data(train_path="data/train.csv", test_path="data/test.csv"):
    """Load train and test datasets."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def feature_engineering(df):
    """Add new features like FamilySize and IsAlone."""
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df

def preprocess_data(train_df, test_df):
    """Clean and preprocess Titanic dataset."""
    # Drop irrelevant columns
    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
    train_df = train_df.drop(columns=drop_cols, errors="ignore")
    test_passenger_ids = test_df["PassengerId"]  # save for submission
    test_df = test_df.drop(columns=drop_cols, errors="ignore")

    # Handle missing values (no inplace)
    train_df["Age"] = train_df["Age"].fillna(train_df["Age"].median())
    test_df["Age"] = test_df["Age"].fillna(test_df["Age"].median())

    train_df["Embarked"] = train_df["Embarked"].fillna(train_df["Embarked"].mode()[0])
    test_df["Embarked"] = test_df["Embarked"].fillna(test_df["Embarked"].mode()[0])

    test_df["Fare"] = test_df["Fare"].fillna(test_df["Fare"].median())

    # Feature engineering
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)

    # Define features and target
    X = train_df.drop("Survived", axis=1)
    y = train_df["Survived"]

    return X, y, test_df, test_passenger_ids


def build_preprocessor():
    """Build preprocessing pipeline (imputation, encoding, scaling)."""
    numeric_features = ["Age", "Fare", "FamilySize", "SibSp", "Parch", "Pclass"]
    categorical_features = ["Sex", "Embarked"]

    # Pipelines
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor
