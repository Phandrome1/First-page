# Titanic Survival Prediction - Machine Learning Pipeline

This project builds a **Machine Learning pipeline** to predict passenger survival on the Titanic dataset.  
It demonstrates end-to-end ML workflow including **data preprocessing, model training, evaluation, and prediction**.

---

## 📂 Project Structure


---

## 🛠️ Libraries Used

- **pandas** – Data handling and analysis  
- **numpy** – Numerical computations  
- **scikit-learn** – Machine Learning models and utilities  
  - `train_test_split`  
  - `Pipeline`, `ColumnTransformer`  
  - `StandardScaler`, `OneHotEncoder`, `SimpleImputer`  
  - `LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`  
  - `accuracy_score`, `classification_report`  
- **joblib** – Model saving and loading  
- **os** – File system operations  

---

## ⚙️ Workflow

1. **Data Preprocessing (`data_preprocessing.py`)**
   - Load Titanic dataset
   - Handle missing values using `SimpleImputer`
   - Encode categorical variables with `OneHotEncoder`
   - Scale numerical features with `StandardScaler`
   - Build preprocessing pipeline with `ColumnTransformer`

2. **Model Training (`train_model.py`)**
   - Train multiple classifiers (`LogisticRegression`, `RandomForest`, `GradientBoosting`)
   - Evaluate models using accuracy and classification report
   - Save the best-performing model with `joblib`

3. **Prediction (`predict.py`)**
   - Load trained model
   - Generate predictions on test dataset
   - Create a submission file

4. **Run Pipeline (`run_pipeline.py`)**
   - Automates the full pipeline from preprocessing to prediction

---

## 🚀 How to Run

1. Clone this repository:

```bash
git clone https://github.com/Phandrome1/ml_pipe_project.git
cd ml_pipe_project

Install dependencies:
pip install -r requirements.txt

Run the pipeline:
python src/run_pipeline.py
