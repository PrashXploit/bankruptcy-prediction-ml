import os
import pickle
from typing import Optional, Any, Dict, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st


MODELS_DIR = os.path.join("models")
MODEL_PATH = os.path.join(MODELS_DIR, "bankruptcy_model.pkl")


@st.cache_resource
def load_model_bundle(path: str) -> Optional[Dict[str, Any]]:
    """Load the trained model bundle (or legacy pipeline) from disk."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        obj = pickle.load(f)

    # New format: dict bundle
    if isinstance(obj, dict) and "pipeline" in obj:
        return obj

    # Legacy format: pipeline only
    # Normalize feature_names into a plain Python list (avoids numpy truthiness issues)
    fn = getattr(obj, "feature_names_in_", None)
    if fn is not None and not isinstance(fn, list):
        try:
            fn = list(fn)
        except Exception:
            fn = None

    return {
        "pipeline": obj,
        "feature_names": fn,
        "best_model_name": "Best Model",
        "accuracies": None,
        "f1_macro": None,
        "detailed": None,
    }


def extract_importances(
    pipeline: object, feature_names: Optional[list[str]]
) -> Optional[Tuple[list[str], np.ndarray]]:
    """Extract feature importances (tree) or coefficients (logistic regression) if available."""
    try:
        model = pipeline.named_steps["model"]
    except Exception:
        return None

    importances = None
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        importances = np.abs(np.asarray(model.coef_[0], dtype=float))

    if importances is None:
        return None

    # feature_names might be a numpy array in some sklearn versions; avoid boolean checks.
    if feature_names is None or len(feature_names) == 0 or len(feature_names) != len(importances):
        feature_names = [f"Feature {i}" for i in range(len(importances))]

    return list(feature_names), importances


def manual_input_interface(pipeline: object, feature_names: Optional[list[str]]) -> None:
    """
    Simple interface where the user can type values for each feature.
    The form is built dynamically from the model's expected feature names.
    """
    st.subheader("Manual Input (Single Company)")
    st.write(
        "Enter the financial indicators below to get a bankruptcy prediction "
        "for a single company."
    )

    if feature_names is None:
        st.info(
            "Could not automatically detect feature names from the model. "
            "Manual input interface is disabled. You can still use the CSV upload mode."
        )
        return

    # Create input widgets for each feature
    st.markdown("**Enter feature values**")
    cols = st.columns(2)
    inputs = {}
    for i, feature in enumerate(feature_names):
        with cols[i % 2]:
            # Use number_input with a wide default range
            value = st.number_input(
                feature,
                value=0.0,
                step=0.1,
                format="%.4f",
            )
            inputs[feature] = value

    if st.button("Predict Bankruptcy (Single Company)"):
        # Build a single-row DataFrame in the same column order
        row = pd.DataFrame([[inputs[f] for f in feature_names]], columns=feature_names)
        try:
            pred = pipeline.predict(row)[0]
            label = "Bankrupt" if int(pred) == 1 else "Not Bankrupt"

            proba_text = ""
            if hasattr(pipeline, "predict_proba"):
                proba = pipeline.predict_proba(row)[0]
                proba_text = (
                    f"\n\nProbability of **Not Bankrupt**: `{proba[0]:.4f}`  \n"
                    f"Probability of **Bankrupt**: `{proba[1]:.4f}`"
                )

            st.success(
                f"Prediction: **{label}** (class {int(pred)}){proba_text}"
            )
        except Exception as e:
            st.error(
                "Error while making prediction with the entered values. "
                "Please check that the inputs are valid."
            )
            st.text(str(e))


def dataset_analysis_interface() -> None:
    """
    Interface to upload any company dataset CSV and perform simple analysis:
    - show head, shape, columns
    - basic statistics
    - class distribution if 'Bankrupt?' column exists
    - correlation heatmap for numeric features
    """
    st.subheader("Dataset Analysis (EDA)")
    st.write(
        "Upload a company dataset CSV to quickly explore it. "
        "If it contains a `Bankrupt?` column, the class distribution will also be shown."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV file for analysis", type=["csv"], key="eda_uploader"
    )

    if uploaded_file is None:
        return

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the uploaded file: {e}")
        return

    if df.empty:
        st.warning("The uploaded CSV file is empty.")
        return

    st.markdown("**First 5 rows**")
    st.dataframe(df.head())

    st.markdown("**Shape**")
    st.write(f"{df.shape[0]} rows × {df.shape[1]} columns")

    st.markdown("**Columns**")
    st.write(list(df.columns))

    st.markdown("**Basic statistics (numeric columns)**")
    st.write(df.describe())

    # Class distribution if target column present
    if "Bankrupt?" in df.columns:
        st.markdown("**Class distribution (Bankrupt?)**")
        class_counts = df["Bankrupt?"].value_counts()
        st.write(class_counts)

        fig, ax = plt.subplots(figsize=(4, 3))
        sns.countplot(x="Bankrupt?", data=df, ax=ax)
        ax.set_title("Class Distribution (Bankrupt?)")
        ax.set_xlabel("Bankrupt? (0 = No, 1 = Yes)")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    # Correlation heatmap for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        st.markdown("**Correlation heatmap (numeric features)**")
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)
    else:
        st.info(
            "Not enough numeric columns to plot a correlation heatmap."
        )


def prediction_csv_interface(
    pipeline: object, feature_names: Optional[list[str]]
) -> None:
    st.subheader("Predict from CSV")
    st.markdown(
        """
        Upload a CSV containing **only feature columns** (no `Bankrupt?`) to get predictions.

        Tip: If your CSV also includes a `Bankrupt?` column, the app will show evaluation metrics too.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload CSV file with company financial data",
        type=["csv"],
        key="predict_uploader",
    )
    if uploaded_file is None:
        return

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the uploaded file: {e}")
        return

    if df.empty:
        st.warning("The uploaded CSV file is empty.")
        return

    st.write("Preview:")
    st.dataframe(df.head())

    y_true = None
    if "Bankrupt?" in df.columns:
        y_true = df["Bankrupt?"].astype(int)
        X = df.drop(columns=["Bankrupt?"])
    else:
        X = df

    # If we know expected features, align columns (helps avoid user column order issues)
    if feature_names is not None:
        missing = [c for c in feature_names if c not in X.columns]
        extra = [c for c in X.columns if c not in feature_names]
        if missing:
            st.error(
                "Your CSV is missing required columns:\n\n" + "\n".join(f"- {m}" for m in missing)
            )
            return
        if extra:
            st.warning(
                "Your CSV has extra columns that will be ignored:\n\n"
                + "\n".join(f"- {c}" for c in extra)
            )
        X = X[feature_names]

    try:
        preds = pipeline.predict(X)
        proba = pipeline.predict_proba(X) if hasattr(pipeline, "predict_proba") else None
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return

    out = X.copy()
    out["Prediction"] = preds
    out["Prediction Label"] = pd.Series(preds).map({0: "Not Bankrupt", 1: "Bankrupt"}).values
    if proba is not None:
        out["Prob_Not_Bankrupt"] = proba[:, 0]
        out["Prob_Bankrupt"] = proba[:, 1]

    st.subheader("Predictions")
    st.dataframe(out)

    csv_download = out.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download predictions as CSV",
        data=csv_download,
        file_name="bankruptcy_predictions.csv",
        mime="text/csv",
    )

    if y_true is not None:
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

        st.subheader("Evaluation (because `Bankrupt?` was provided)")
        acc = accuracy_score(y_true, preds)
        st.metric("Accuracy", f"{acc:.4f}")

        cm = confusion_matrix(y_true, preds)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.markdown("**Classification Report**")
        st.code(classification_report(y_true, preds), language="text")


def model_insights_interface(bundle: Dict[str, Any]) -> None:
    st.subheader("Model Insights")
    st.caption(
        "These insights come from the **trained model** saved in `models/bankruptcy_model.pkl` "
        "(no CSV upload is required)."
    )
    pipeline = bundle["pipeline"]
    feature_names = bundle.get("feature_names")
    best_name = bundle.get("best_model_name", "Best Model")

    st.markdown(f"**Selected model:** `{best_name}`")

    accs = bundle.get("accuracies")
    f1s = bundle.get("f1_macro")
    show_comparison = st.checkbox(
        "Show model comparison (accuracy / F1)",
        value=False,
        help="Shows metrics saved during the last training run.",
    )
    if show_comparison:
        if isinstance(accs, dict):
            st.markdown("**Model comparison (from training run)**")
            comp = pd.DataFrame(
                {
                    "Accuracy": pd.Series(accs),
                    "F1 (macro)": pd.Series(f1s) if isinstance(f1s, dict) else np.nan,
                }
            ).reset_index().rename(columns={"index": "Model"})
            st.dataframe(comp, use_container_width=True)

            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(data=comp, x="Model", y="Accuracy", ax=ax, palette="viridis")
            ax.set_ylim(0, 1)
            ax.set_title("Accuracy Comparison")
            ax.tick_params(axis="x", rotation=20)
            st.pyplot(fig)
        else:
            st.info(
                "Model comparison metrics were not found in the saved file. "
                "Re-run `python main.py` to regenerate them."
            )

    show_importance = st.checkbox(
        "Show feature importance (top 20)",
        value=False,
        help="Shows feature importance/coefficient magnitude from the trained model.",
    )
    if show_importance:
        extracted = extract_importances(pipeline, feature_names)
        if extracted is None:
            st.info("Feature importance is not available for the selected model.")
            return

        fn, imps = extracted
        order = np.argsort(imps)[::-1][:20]
        top = pd.DataFrame({"Feature": np.array(fn)[order], "Importance": imps[order]})

        st.markdown("**Top 20 feature importances**")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(data=top, x="Importance", y="Feature", ax=ax, palette="mako")
        ax.set_title("Top 20 Feature Importances")
        st.pyplot(fig)


def main() -> None:
    st.set_page_config(page_title="Bankruptcy Prediction Dashboard", layout="wide")
    st.title("Bankruptcy Prediction Dashboard")
    st.write(
        "This app uses a machine learning model trained on the "
        "Taiwan Company Bankruptcy Prediction dataset."
    )

    bundle = load_model_bundle(MODEL_PATH)
    if bundle is None:
        st.error(
            f"Trained model not found at `{MODEL_PATH}`. "
            "Please run `python main.py` first to train and save the model."
        )
        return

    pipeline = bundle["pipeline"]
    feature_names = bundle.get("feature_names")

    tabs = st.tabs(["Predict", "Analyze Dataset", "Model Insights"])
    with tabs[0]:
        col1, col2 = st.columns([1, 1])
        with col1:
            show_manual = st.checkbox(
                "Enable manual input form (single company)",
                value=False,
                help="Shows a large form (one field per feature).",
            )
            if show_manual:
                manual_input_interface(pipeline, feature_names)
            else:
                st.info(
                    "Manual input is hidden to keep the UI clean. "
                    "Enable it using the checkbox above."
                )
        with col2:
            prediction_csv_interface(pipeline, feature_names)

    with tabs[1]:
        dataset_analysis_interface()

    with tabs[2]:
        model_insights_interface(bundle)


if __name__ == "__main__":
    main()

