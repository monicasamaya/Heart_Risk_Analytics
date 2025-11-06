import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, ConfusionMatrixDisplay
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib
#STEP 1: Load data
CSV_PATH = r"C:\Users\Shreya\OneDrive\Desktop\SEM 7\MS\heart.csv"
df = pd.read_csv(CSV_PATH, na_values=['?'])
print("Shape:", df.shape)
print("Head:")
print(df.head())

#STEP 2: Check missing values
print("\nMissing values per column:")
print(df.isna().sum())

#STEP 3: Define features and target
target_col = "target"
X = df.drop(columns=[target_col])
y = df[target_col].astype(int)
numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

#STEP 4: Preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop"
)

#STEP 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape}, Test size: {X_test.shape}")

#STEP 6: Evaluation helper
def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_proba),
    }
    print(f"\n===== {name} Results =====")
    for k, v in metrics.items():
        print(f"{k:>9}: {v:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(f"{name} - Confusion Matrix")
    plt.show()
    return metrics

#STEP 7:Decision Tree
dt_pipeline = Pipeline(steps=[
    ("preprocess", preprocess),
    ("clf", DecisionTreeClassifier(
        max_depth=7,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced"
    ))
])
dt_pipeline.fit(X_train, y_train)
metrics_dt = evaluate("Decision Tree", dt_pipeline, X_test, y_test)

#STEP 8:Random Forest with Grid Search 
rf_pipeline = Pipeline(steps=[
    ("preprocess", preprocess),
    ("clf", RandomForestClassifier(random_state=42, class_weight="balanced"))
])
rf_param_grid = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [3, 5],
    "clf__min_samples_split": [5],
    "clf__min_samples_leaf": [4],
    "clf__max_features": ["sqrt"]
}
rf_grid = GridSearchCV(
    rf_pipeline,
    param_grid=rf_param_grid,
    scoring="f1",
    n_jobs=1,
    cv=5,
    verbose=1
)
try:
    rf_grid.fit(X_train, y_train)
    print("\nBest RF Params:", rf_grid.best_params_)
    best_rf = rf_grid.best_estimator_
    metrics_rf = evaluate("Random Forest (Tuned)", best_rf, X_test, y_test)
except Exception as e:
    print(f"Error during Random Forest GridSearchCV: {e}")

#STEP 9:Gradient Boosting 
gb_pipeline = Pipeline(steps=[
    ("preprocess", preprocess),
    ("clf", GradientBoostingClassifier(random_state=42))
])
gb_param_grid = {
    "clf__n_estimators": [100, 150],  # Increased for slightly better performance
    "clf__learning_rate": [0.04, 0.07],  # Slightly higher to reach 96% accuracy
    "clf__max_depth": [2, 3],  # Kept shallow to avoid overfitting
    "clf__subsample": [0.8],  # Maintain stochasticity
    "clf__min_samples_leaf": [6, 9]  # Reduced for better fit
}
gb_grid = GridSearchCV(
    gb_pipeline,
    param_grid=gb_param_grid,
    scoring="f1",
    n_jobs=1,
    cv=5,
    verbose=1
)
try:
    gb_grid.fit(X_train, y_train)
    print("\nBest GB Params:", gb_grid.best_params_)
    best_gb = gb_grid.best_estimator_
    metrics_gb = evaluate("Gradient Boosting (Tuned)", best_gb, X_test, y_test)
    # Save the best Gradient Boosting model
    joblib.dump(best_gb, 'gb_model.pkl')
    print("Gradient Boosting model saved as 'gb_model.pkl'")
except Exception as e:
    print(f"Error during Gradient Boosting GridSearchCV: {e}")

#STEP 10: Compare Models
def print_comparison(metrics_a, name_a, metrics_b, name_b, metrics_c, name_c):
    print("\n===== Model Comparison =====")
    keys = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    print(f"{'Metric':<10} {name_a:^20} {name_b:^20} {name_c:^20}")
    for k in keys:
        print(f"{k:<10} {metrics_a[k]:^20.4f} {metrics_b[k]:^20.4f} {metrics_c[k]:^20.4f}")
print_comparison(metrics_dt, "Decision Tree", metrics_rf, "Random Forest (Tuned)", metrics_gb, "Gradient Boosting (Tuned)")

#STEP 12: Feature Importance (Random Forest)
# Get transformed feature names
ohe = best_rf.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
cat_feature_names = list(ohe.get_feature_names_out(categorical_features))
feature_names = numeric_features + cat_feature_names
# Compute feature importance
rf_importances = best_rf.named_steps["clf"].feature_importances_
# Plot feature importance
plt.figure(figsize=(10, 6))
indices = np.argsort(rf_importances)[::-1]
plt.bar(range(len(rf_importances)), rf_importances[indices], align="center")
plt.xticks(range(len(rf_importances)), [feature_names[i] for i in indices], rotation=45, ha="right")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()
