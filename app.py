import os
import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="IMS Bearing Fault Diagnosis",
    page_icon="⚙️",
    layout="wide"
)


# --------------------------------------------------
# File paths
# --------------------------------------------------
MODEL_PATH = "models/best_bearing_model.joblib"
SCALER_PATH = "models/scaler.joblib"
FEATURES_PATH = "models/feature_columns.joblib"
CONFUSION_MATRIX_PATH = "plots/model_confusion_matrices.png"


# --------------------------------------------------
# Load model files
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    required_files = [
        MODEL_PATH,
        SCALER_PATH,
        FEATURES_PATH
    ]

    missing_files = [
        file_path
        for file_path in required_files
        if not os.path.exists(file_path)
    ]

    if missing_files:
        raise FileNotFoundError(
            f"Missing model files: {missing_files}"
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURES_PATH)

    return model, scaler, feature_columns


try:
    model, scaler, feature_columns = load_artifacts()
except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()


# --------------------------------------------------
# Prediction function
# --------------------------------------------------
def predict_bearing_state(input_df):
    """
    Predict bearing state.

    0 = Healthy
    1 = Faulty / Degraded
    """

    input_features = input_df[feature_columns].copy()

    # Convert all features into numeric values
    input_features = input_features.apply(
        pd.to_numeric,
        errors="coerce"
    )

    if input_features.isnull().any().any():
        null_columns = input_features.columns[
            input_features.isnull().any()
        ].tolist()

        raise ValueError(
            f"Invalid or missing values found in: {null_columns}"
        )

    scaled_features = scaler.transform(input_features)
    predictions = model.predict(scaled_features)

    result_df = input_df.copy()
    result_df["prediction"] = predictions

    result_df["bearing_state"] = result_df["prediction"].map({
        0: "Healthy",
        1: "Faulty / Degraded"
    })

    # Probability available for Logistic Regression,
    # KNN, SVM probability=True and Random Forest
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(scaled_features)

        result_df["healthy_probability"] = probabilities[:, 0]
        result_df["fault_probability"] = probabilities[:, 1]

    return result_df


# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("⚙️ IMS Bearing Fault Diagnosis System")

st.markdown(
    """
    Upload extracted bearing features to classify each observation as
    **Healthy** or **Faulty / Degraded**.
    """
)

with st.expander("Model information"):
    st.write("Model:", type(model).__name__)
    st.write("Number of required features:", len(feature_columns))
    st.write("Classes: `0 = Healthy`, `1 = Faulty / Degraded`")


# --------------------------------------------------
# Tabs
# --------------------------------------------------
upload_tab, manual_tab, model_tab = st.tabs([
    "Upload Feature CSV",
    "Manual Prediction",
    "Model Evaluation"
])


# ==================================================
# TAB 1: CSV upload
# ==================================================
with upload_tab:
    st.subheader("Upload bearing feature CSV")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)

            st.success(
                f"Loaded {uploaded_df.shape[0]} rows and "
                f"{uploaded_df.shape[1]} columns."
            )

            st.write("Uploaded data preview")
            st.dataframe(
                uploaded_df.head(),
                width="content"
            )

            missing_columns = [
                column
                for column in feature_columns
                if column not in uploaded_df.columns
            ]

            if missing_columns:
                st.error(
                    "The uploaded CSV is missing required columns:"
                )
                st.code("\n".join(missing_columns))

            else:
                if st.button(
                    "Run Fault Diagnosis",
                    type="primary",
                    width="content"
                ):
                    with st.spinner("Analysing bearing data..."):
                        result_df = predict_bearing_state(
                            uploaded_df
                        )

                    st.success("Prediction completed successfully.")

                    total_samples = len(result_df)
                    healthy_count = (
                        result_df["prediction"] == 0
                    ).sum()
                    faulty_count = (
                        result_df["prediction"] == 1
                    ).sum()

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Total samples",
                        total_samples
                    )

                    col2.metric(
                        "Healthy samples",
                        int(healthy_count)
                    )

                    col3.metric(
                        "Faulty samples",
                        int(faulty_count)
                    )

                    st.subheader("Prediction results")

                    display_columns = [
                        column
                        for column in [
                            "file_index",
                            "bearing_state",
                            "healthy_probability",
                            "fault_probability"
                        ]
                        if column in result_df.columns
                    ]

                    st.dataframe(
                        result_df[display_columns],
                        width="content"
                    )

                    if "fault_probability" in result_df.columns:
                        st.subheader(
                            "Fault probability across observations"
                        )

                        chart_df = result_df[
                            ["fault_probability"]
                        ].copy()

                        st.line_chart(chart_df)

                    csv_data = result_df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="Download prediction results",
                        data=csv_data,
                        file_name="bearing_predictions.csv",
                        mime="text/csv",
                        width="content"
                    )

        except Exception as error:
            st.error(f"Unable to process CSV: {error}")

    else:
        st.info(
            "Upload a CSV containing all the model feature columns."
        )

        with st.expander("View required columns"):
            st.code("\n".join(feature_columns))


# ==================================================
# TAB 2: Manual input
# ==================================================
with manual_tab:
    st.subheader("Enter one bearing observation")

    st.info(
        "Enter the extracted statistical values for all four bearings."
    )

    manual_values = {}

    for bearing_number in range(1, 5):
        with st.expander(
            f"Bearing {bearing_number}",
            expanded=(bearing_number == 1)
        ):
            col1, col2 = st.columns(2)

            bearing_features = [
                f"B{bearing_number}_mean",
                f"B{bearing_number}_std",
                f"B{bearing_number}_rms",
                f"B{bearing_number}_peak",
                f"B{bearing_number}_kurtosis",
                f"B{bearing_number}_skew",
                f"B{bearing_number}_crest_factor",
                f"B{bearing_number}_shape_factor"
            ]

            for index, feature in enumerate(bearing_features):
                selected_column = (
                    col1 if index % 2 == 0 else col2
                )

                manual_values[feature] = (
                    selected_column.number_input(
                        feature,
                        value=0.0,
                        format="%.6f",
                        key=f"manual_{feature}"
                    )
                )

    if st.button(
        "Predict Bearing Condition",
        type="primary",
        width="content",
        key="manual_predict"
    ):
        try:
            manual_df = pd.DataFrame(
                [manual_values],
                columns=feature_columns
            )

            result = predict_bearing_state(manual_df)

            prediction = int(
                result.loc[0, "prediction"]
            )

            state = result.loc[0, "bearing_state"]

            if prediction == 0:
                st.success(
                    f"Prediction: {state}"
                )
            else:
                st.error(
                    f"Prediction: {state}"
                )

            if "fault_probability" in result.columns:
                fault_probability = float(
                    result.loc[0, "fault_probability"]
                )

                healthy_probability = float(
                    result.loc[0, "healthy_probability"]
                )

                col1, col2 = st.columns(2)

                col1.metric(
                    "Healthy probability",
                    f"{healthy_probability:.2%}"
                )

                col2.metric(
                    "Fault probability",
                    f"{fault_probability:.2%}"
                )

                st.progress(
                    min(max(fault_probability, 0.0), 1.0),
                    text=(
                        f"Estimated fault probability: "
                        f"{fault_probability:.2%}"
                    )
                )

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# ==================================================
# TAB 3: Evaluation image
# ==================================================
with model_tab:
    st.subheader("Model evaluation")

    if os.path.exists(CONFUSION_MATRIX_PATH):
        st.image(
            CONFUSION_MATRIX_PATH,
            caption="Confusion matrix comparison",
           width="content"
        )
    else:
        st.warning(
            "Confusion matrix image was not found in the plots folder."
        )

    st.write("Selected best model:", type(model).__name__)

    st.warning(
        """
        Your current labels are created using `file_index >= 700`.
        Therefore, the model learns your manually defined healthy/faulty
        boundary. This should be replaced with verified bearing-failure
        labels before treating the application as a reliable diagnostic tool.
        """
    )