import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)

# ==================================================
# CONFIG
# ==================================================
SVM_SAMPLE_RATIO = 0.20
DECISION_THRESHOLD = 0.5

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide"
)

st.title("💳 Fraud Detection Dashboard")
st.caption("Comparative Analysis of Machine Learning Models for Fraud Detection")

# ==================================================
# LOAD MODELS & SCALER
# ==================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
@st.cache_resource
def load_models_and_scaler():
    models = {
        "Logistic Regression": joblib.load(os.path.join(current_dir, "logistic_model.pkl")),
        "Decision Tree": joblib.load(os.path.join(current_dir, "dt_model.pkl")),
        "Random Forest": joblib.load(os.path.join(current_dir, "rf_model.pkl")),
        "SVM": joblib.load(os.path.join(current_dir, "svm_model.pkl")),
        "XGBoost": joblib.load(os.path.join(current_dir, "xgb_model.pkl")),
    }
    scaler = joblib.load(os.path.join(current_dir, "scaler.pkl"))
    return models, scaler

models, scaler = load_models_and_scaler()

# ==================================================
# LOAD TEST DATA
# ==================================================
@st.cache_data
def load_test_data():
    X_test_scaled = joblib.load(os.path.join(current_dir, "X_test_scaled.pkl"))
    y_test = joblib.load(os.path.join(current_dir, "y_test.pkl"))
    return X_test_scaled, y_test

# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.header("⚙️ Settings")
selected_model_name = st.sidebar.selectbox(
    "Select Model",
    list(models.keys())
)
selected_model = models[selected_model_name]

# ==================================================
# TABS
# ==================================================
tab_overview, tab_model = st.tabs(
    ["📊 Overview & Comparison", f"🧠 {selected_model_name}"]
)

# ==================================================
# TAB 1 — OVERVIEW & COMPARISON
# ==================================================
with tab_overview:
    st.subheader("📈 Model Performance Comparison (Test Set)")

    if st.button("🚀 Load Comparison Metrics & ROC"):
        with st.spinner("Evaluating models..."):
            X_test_scaled, y_test = load_test_data()
            metrics = []

            fig, ax = plt.subplots(figsize=(7, 6))

            for name, model in models.items():
                # --- SVM sampling ---
                if name == "SVM":
                    sample_idx = np.random.choice(
                        len(X_test_scaled),
                        size=int(SVM_SAMPLE_RATIO * len(X_test_scaled)),
                        replace=False
                    )
                    X_eval = X_test_scaled[sample_idx]
                    y_eval = y_test.iloc[sample_idx]
                else:
                    X_eval = X_test_scaled
                    y_eval = y_test

                y_prob = model.predict_proba(X_eval)[:, 1]
                y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)

                metrics.append({
                    "Model": name,
                    "Accuracy": accuracy_score(y_eval, y_pred),
                    "Precision": precision_score(y_eval, y_pred),
                    "Recall": recall_score(y_eval, y_pred),
                    "F1 Score": f1_score(y_eval, y_pred),
                    "ROC-AUC": roc_auc_score(y_eval, y_prob)
                })

                fpr, tpr, _ = roc_curve(y_eval, y_prob)
                ax.plot(fpr, tpr, label=name)

            ax.plot([0, 1], [0, 1], linestyle="--")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve Comparison")
            ax.legend()
            st.pyplot(fig)

            # ==========================
            # METRICS + RANKING
            # ==========================
            df = pd.DataFrame(metrics)
            df = df.sort_values("F1 Score", ascending=False)
            df.insert(0, "Rank", range(1, len(df) + 1))

            st.subheader("🏆 Model Ranking (by F1 Score)")
            numeric_cols = df.select_dtypes(include="number").columns
            st.dataframe(
                df.style.format({col: "{:.3f}" for col in numeric_cols})
            )

# ==================================================
# TAB 2 — SELECTED MODEL DETAILS
# ==================================================
with tab_model:
    st.subheader(f"📌 {selected_model_name} – Test Set Evaluation")

    if st.button(f"🔍 Load {selected_model_name} Evaluation"):
        with st.spinner("Running model evaluation..."):
            X_test_scaled, y_test = load_test_data()

            if selected_model_name == "SVM":
                sample_idx = np.random.choice(
                    len(X_test_scaled),
                    size=int(SVM_SAMPLE_RATIO * len(X_test_scaled)),
                    replace=False
                )
                X_eval = X_test_scaled[sample_idx]
                y_eval = y_test.iloc[sample_idx]
            else:
                X_eval = X_test_scaled
                y_eval = y_test

            y_prob = selected_model.predict_proba(X_eval)[:, 1]
            y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Accuracy", f"{accuracy_score(y_eval, y_pred):.3f}")
            col2.metric("Precision", f"{precision_score(y_eval, y_pred):.3f}")
            col3.metric("Recall", f"{recall_score(y_eval, y_pred):.3f}")
            col4.metric("F1 Score", f"{f1_score(y_eval, y_pred):.3f}")
            col5.metric("ROC-AUC", f"{roc_auc_score(y_eval, y_prob):.3f}")

            st.divider()

            col_l, col_r = st.columns(2)

            with col_l:
                cm = confusion_matrix(y_eval, y_pred)
                fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
                ax_cm.set_title("Confusion Matrix")
                st.pyplot(fig_cm)

            with col_r:
                fpr, tpr, _ = roc_curve(y_eval, y_prob)
                fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
                ax_roc.plot(fpr, tpr, label="ROC Curve")
                ax_roc.plot([0, 1], [0, 1], linestyle="--")
                ax_roc.legend()
                ax_roc.set_title("ROC Curve")
                st.pyplot(fig_roc)

    st.divider()

    # ==================================================
    # UPLOAD CSV
    # ==================================================
    st.subheader("📂 Upload Transactions CSV")

    uploaded_file = st.file_uploader(
        "Upload CSV (with or without Class column)",
        type="csv"
    )

    if uploaded_file:
        with st.spinner("Processing uploaded file..."):
            data = pd.read_csv(uploaded_file)

            st.info(f"Rows: {data.shape[0]} | Columns: {data.shape[1]}")

            if "Class" in data.columns:
                y_true = data["Class"]
                X_input = data.drop(columns=["Class"])
                st.success("Target column detected: Class")
            else:
                X_input = data.copy()
                y_true = None

            X_scaled = scaler.transform(X_input)

            y_prob = selected_model.predict_proba(X_scaled)[:, 1]
            y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)

            result_df = pd.DataFrame({
                "Predicted Class": y_pred,
                "Fraud Probability": y_prob
            })

            st.dataframe(result_df)

            if y_true is not None:
                st.subheader("📊 Evaluation on Uploaded Data")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.3f}")
                c2.metric("Precision", f"{precision_score(y_true, y_pred):.3f}")
                c3.metric("Recall", f"{recall_score(y_true, y_pred):.3f}")
                c4.metric("F1 Score", f"{f1_score(y_true, y_pred):.3f}")
