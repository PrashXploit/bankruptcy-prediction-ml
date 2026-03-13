import os
import pickle
from typing import Dict, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC


# Path to the dataset (update if your file name is different)
DATA_PATH = os.path.join("data", "company_bankruptcy_prediction.csv")

# Output directories for models and plots
MODELS_DIR = "models"
PLOTS_DIR = "notebooks"  # reuse notebooks folder to store generated figures

# If True, run quick hyperparameter tuning for stronger performance
ENABLE_TUNING = True


def load_dataset(path: str) -> pd.DataFrame:
    """Load the Taiwan Company Bankruptcy Prediction dataset from CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. "
            "Please place the CSV file in the 'data/' folder and update DATA_PATH if needed."
        )
    df = pd.read_csv(path)
    if "Bankrupt?" not in df.columns:
        raise ValueError(
            "Target column 'Bankrupt?' not found in the dataset. "
            "Please ensure the column name matches exactly."
        )
    return df


def basic_eda(df: pd.DataFrame) -> None:
    """Perform basic EDA: info, describe, class distribution, and correlation heatmap."""
    print("\n===== Dataset Info =====")
    print(df.info())

    print("\n===== Dataset Description (first 5 numeric columns) =====")
    print(df.describe().iloc[:, :5])

    # Class distribution
    print("\n===== Class Distribution (Bankrupt?) =====")
    class_counts = df["Bankrupt?"].value_counts()
    print(class_counts)

    plt.figure(figsize=(6, 4))
    sns.countplot(x="Bankrupt?", data=df)
    plt.title("Class Distribution (Bankrupt?)")
    plt.xlabel("Bankrupt? (0 = No, 1 = Yes)")
    plt.ylabel("Count")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    class_dist_path = os.path.join(PLOTS_DIR, "class_distribution.png")
    plt.savefig(class_dist_path, bbox_inches="tight")
    plt.close()
    print(f"Saved class distribution plot to: {class_dist_path}")

    # Correlation heatmap for a subset of features (to keep it readable)
    plt.figure(figsize=(12, 10))
    # Take up to first 15 numeric columns including target for visualization
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    subset_cols = numeric_cols[:15]
    corr = df[subset_cols].corr()
    sns.heatmap(corr, annot=False, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap (subset of numeric features)")
    corr_path = os.path.join(PLOTS_DIR, "correlation_heatmap.png")
    plt.savefig(corr_path, bbox_inches="tight")
    plt.close()
    print(f"Saved correlation heatmap to: {corr_path}")


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate features (X) and target (y)."""
    X = df.drop(columns=["Bankrupt?"])
    y = df["Bankrupt?"]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a preprocessing transformer:
    - Impute missing numeric values with median
    - Scale numeric features
    """
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
        ]
    )
    return preprocessor


def get_models() -> Dict[str, object]:
    """Create a dictionary of candidate models."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, solver="lbfgs", class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
            max_depth=None,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=400,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "SVM (RBF Kernel)": SVC(
            kernel="rbf",
            probability=True,
            random_state=42,
            class_weight="balanced",
        ),
    }
    return models


def build_pipelines(
    preprocessor: ColumnTransformer, models: Dict[str, object]
) -> Dict[str, Pipeline]:
    """
    Wrap each model in a Pipeline that includes preprocessing.
    This ensures the same preprocessing is applied during training and inference.
    """
    pipelines: Dict[str, Pipeline] = {}
    for name, model in models.items():
        pipelines[name] = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
    return pipelines


def maybe_tune_model(
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """
    Optional light hyperparameter tuning using GridSearchCV for select models.
    Kept intentionally small for beginner-friendliness and reasonable runtime.
    """
    if not ENABLE_TUNING:
        return pipeline

    param_grid: Dict[str, list[Any]] | None = None

    if name == "Random Forest":
        param_grid = {
            "model__n_estimators": [300, 600],
            "model__max_depth": [None, 12, 20],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
            "model__max_features": ["sqrt", 0.5],
        }
    elif name == "Extra Trees":
        param_grid = {
            "model__n_estimators": [400, 800],
            "model__max_depth": [None, 12, 20],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
            "model__max_features": ["sqrt", 0.5],
        }
    elif name == "SVM (RBF Kernel)":
        param_grid = {
            "model__C": [0.5, 1.0, 2.0],
            "model__gamma": ["scale", 0.05, 0.1],
        }
    elif name == "Logistic Regression":
        param_grid = {
            "model__C": [0.5, 1.0, 2.0],
        }

    if not param_grid:
        return pipeline

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    print(f"Best params for {name}: {search.best_params_}")
    return search.best_estimator_


def evaluate_models(
    pipelines: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Any], str, Pipeline]:
    """
    Train and evaluate each model pipeline.
    Returns:
    - accuracies dict
    - f1 (macro) dict
    - detailed results dict (confusion matrices + reports)
    - name of best model
    - best model pipeline
    """
    accuracies: Dict[str, float] = {}
    f1s: Dict[str, float] = {}
    detailed: Dict[str, Any] = {}
    best_model_name = None
    best_model_pipeline: Pipeline | None = None
    best_accuracy = -1.0

    for name, pipeline in pipelines.items():
        print(f"\n===== Training: {name} =====")
        tuned = maybe_tune_model(name, pipeline, X_train, y_train)
        tuned.fit(X_train, y_train)

        y_pred = tuned.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1m = f1_score(y_test, y_pred, average="macro")
        accuracies[name] = acc
        f1s[name] = f1m

        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        detailed[name] = {"confusion_matrix": cm, "classification_report": report}

        print(f"Accuracy: {acc:.4f}")
        print(f"F1 (macro): {f1m:.4f}")
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_model_pipeline = tuned

    if best_model_name is None or best_model_pipeline is None:
        raise RuntimeError("No best model found. Check model training.")

    print("\n===== Model Accuracies =====")
    for name, acc in accuracies.items():
        print(f"{name}: {acc:.4f} (F1-macro: {f1s[name]:.4f})")

    print(f"\nBest model: {best_model_name} with accuracy {best_accuracy:.4f}")

    # Plot model performance comparison
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.figure(figsize=(8, 5))
    model_names = list(accuracies.keys())
    model_accs = list(accuracies.values())
    sns.barplot(x=model_names, y=model_accs, palette="viridis")
    plt.xticks(rotation=20)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Model Performance Comparison (Accuracy)")
    perf_path = os.path.join(PLOTS_DIR, "model_performance.png")
    plt.savefig(perf_path, bbox_inches="tight")
    plt.close()
    print(f"Saved model performance plot to: {perf_path}")

    return accuracies, f1s, detailed, best_model_name, best_model_pipeline


def plot_feature_importance(
    best_model_name: str,
    best_pipeline: Pipeline,
    feature_names: list[str],
) -> None:
    """
    Plot feature importance for the best model, if supported.
    For tree-based models, use feature_importances_.
    For logistic regression, use absolute value of coefficients.
    """
    model = best_pipeline.named_steps["model"]
    importances = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])

    if importances is None:
        print(
            f"Feature importance not available for model '{best_model_name}'."
        )
        return

    # Align number of features (after preprocessing) with original names if possible.
    # Here we assume only numeric features were used and ColumnTransformer
    # applied them in the same order.
    if len(importances) != len(feature_names):
        # Fallback: just plot top 20 importances without specific labels
        print(
            "Number of importances does not match number of original features. "
            "Plotting top importances without detailed labels."
        )
        sorted_idx = np.argsort(importances)[::-1][:20]
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(sorted_idx)), importances[sorted_idx])
        plt.title(f"Feature Importance (Top {len(sorted_idx)}) - {best_model_name}")
        plt.xlabel("Feature Index (ordered by importance)")
        plt.ylabel("Importance")
    else:
        # Plot top 20 features with names
        sorted_idx = np.argsort(importances)[::-1][:20]
        top_features = [feature_names[i] for i in sorted_idx]
        top_importances = importances[sorted_idx]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_importances, y=top_features, orient="h", palette="mako")
        plt.title(f"Top 20 Feature Importances - {best_model_name}")
        plt.xlabel("Importance")
        plt.ylabel("Feature")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    feat_path = os.path.join(PLOTS_DIR, "feature_importance.png")
    plt.tight_layout()
    plt.savefig(feat_path, bbox_inches="tight")
    plt.close()
    print(f"Saved feature importance plot to: {feat_path}")


def save_best_model(best_pipeline: Pipeline, path: str) -> None:
    """Save the best model pipeline to disk using pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(best_pipeline, f)
    print(f"\nSaved best model to: {path}")


def save_model_bundle(
    path: str,
    *,
    pipeline: Pipeline,
    feature_names: list[str],
    best_model_name: str,
    accuracies: Dict[str, float],
    f1s: Dict[str, float],
    detailed: Dict[str, Any],
) -> None:
    """
    Save a bundle that contains:
    - trained pipeline
    - feature names
    - model comparison metrics
    This makes the Streamlit dashboard richer and more reliable.
    """
    bundle = {
        "pipeline": pipeline,
        "feature_names": feature_names,
        "best_model_name": best_model_name,
        "accuracies": accuracies,
        "f1_macro": f1s,
        "detailed": detailed,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nSaved model bundle to: {path}")

def main() -> None:
    print("===== Bankruptcy Prediction using Machine Learning =====")
    print(f"Loading dataset from: {DATA_PATH}")

    df = load_dataset(DATA_PATH)

    # EDA
    basic_eda(df)

    # Features and target
    X, y = split_features_target(df)
    feature_names = X.columns.tolist()

    # Train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print(
        f"\nTrain size: {X_train.shape[0]} samples, "
        f"Test size: {X_test.shape[0]} samples"
    )

    # Preprocessing and models
    preprocessor = build_preprocessor(X_train)
    models = get_models()
    pipelines = build_pipelines(preprocessor, models)

    # Train, evaluate, and select best model
    accuracies, f1s, detailed, best_model_name, best_model_pipeline = evaluate_models(
        pipelines,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # Feature importance visualization
    plot_feature_importance(best_model_name, best_model_pipeline, feature_names)

    # Save best model
    best_model_path = os.path.join(MODELS_DIR, "bankruptcy_model.pkl")
    # Save bundle (preferred) so the dashboard has access to feature names + metrics.
    # We keep the filename as requested.
    save_model_bundle(
        best_model_path,
        pipeline=best_model_pipeline,
        feature_names=feature_names,
        best_model_name=best_model_name,
        accuracies=accuracies,
        f1s=f1s,
        detailed=detailed,
    )

    print("\nAll done!")


if __name__ == "__main__":
    main()

