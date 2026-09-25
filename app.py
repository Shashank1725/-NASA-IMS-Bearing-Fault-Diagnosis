import os
import uuid

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(
    page_title="Bearing Intelligence | NASA IMS",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "models/best_bearing_model.joblib"
SCALER_PATH = "models/scaler.joblib"
FEATURES_PATH = "models/feature_columns.joblib"
CONFUSION_MATRIX_PATH = "plots/model_confusion_matrices.png"

# A lightweight Three.js canvas: particles only, no imported 3D assets.
THREE_BACKGROUND = """
<!doctype html>
<html>
<head><style>
html, body, #scene { margin:0; width:100%; height:100%; overflow:hidden; background:#07111f; }
#scene { position:fixed; inset:0; }
</style></head>
<body><div id="scene"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const host = document.getElementById('scene');
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 1, 2000);
camera.position.z = 520;
const renderer = new THREE.WebGLRenderer({antialias:true, alpha:true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight); host.appendChild(renderer.domElement);
const count = 850, positions = new Float32Array(count * 3), colors = new Float32Array(count * 3);
for (let i=0; i<count; i++) {
  positions[i*3] = (Math.random()-0.5)*1100;
  positions[i*3+1] = (Math.random()-0.5)*700;
  positions[i*3+2] = (Math.random()-0.5)*900;
  const c = new THREE.Color(Math.random() > .55 ? '#1dd6c1' : '#4f7cff');
  colors[i*3] = c.r; colors[i*3+1] = c.g; colors[i*3+2] = c.b;
}
const geometry = new THREE.BufferGeometry();
geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
const material = new THREE.PointsMaterial({size:2.2, vertexColors:true, transparent:true, opacity:.5});
const particles = new THREE.Points(geometry, material); scene.add(particles);
function resize(){ camera.aspect=innerWidth/innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth,innerHeight); }
addEventListener('resize', resize);
function animate(){ requestAnimationFrame(animate); particles.rotation.y += .00035; particles.rotation.x += .00008; renderer.render(scene,camera); }
animate();
</script></body></html>
"""


@st.cache_resource
def load_artifacts():
    required = [MODEL_PATH, SCALER_PATH, FEATURES_PATH]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing model files: {missing}")
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH), joblib.load(FEATURES_PATH)


try:
    model, scaler, feature_columns = load_artifacts()
except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()


def predict_bearing_state(input_df):
    features = input_df[feature_columns].apply(pd.to_numeric, errors="coerce")
    if features.isnull().any().any():
        bad = features.columns[features.isnull().any()].tolist()
        raise ValueError(f"Invalid or missing values found in: {bad}")

    scaled = scaler.transform(features)
    predictions = model.predict(scaled)
    result = input_df.copy()
    result["prediction"] = predictions
    result["bearing_state"] = result["prediction"].map({0: "Healthy", 1: "Faulty / Degraded"}).fillna(result["prediction"].astype(str))

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(scaled)
        classes = list(getattr(model, "classes_", [0, 1]))
        for label, column in [(0, "healthy_probability"), (1, "fault_probability")]:
            if label in classes:
                result[column] = probabilities[:, classes.index(label)]
    return result


def numeric_columns(frame):
    return frame.select_dtypes(include="number").columns.tolist()


def dataset_explorer(frame, title="Live dataset explorer"):
    """Build charts from the uploaded frame, never from a hard-coded series."""
    st.subheader(title)
    numeric = numeric_columns(frame)
    if not numeric:
        st.info("This dataset has no numeric columns to chart yet.")
        return

    controls = st.columns([1.2, 1.2, 1])
    x_options = ["Row index"] + frame.columns.tolist()
    x_choice = controls[0].selectbox("X-axis", x_options, key=f"x_{id(frame)}")
    y_choice = controls[1].selectbox("Signal / feature", numeric, key=f"y_{id(frame)}")
    chart_type = controls[2].selectbox("Chart", ["Trend", "Distribution", "Correlation"], key=f"chart_{id(frame)}")

    chart_frame = frame.copy().reset_index(drop=True)
    x = chart_frame.index if x_choice == "Row index" else chart_frame[x_choice]
    if chart_type == "Trend":
        plot = px.line(x=x, y=chart_frame[y_choice], labels={"x": x_choice, "y": y_choice}, template="plotly_dark")
        plot.update_traces(line_color="#1dd6c1", line_width=2)
        plot.update_layout(height=390, margin=dict(l=10, r=10, t=25, b=10), hovermode="x unified")
    elif chart_type == "Distribution":
        plot = px.histogram(chart_frame, x=y_choice, nbins=35, marginal="box", template="plotly_dark", color_discrete_sequence=["#4f7cff"])
        plot.update_layout(height=390, margin=dict(l=10, r=10, t=25, b=10))
    else:
        corr = chart_frame[numeric].corr()
        plot = px.imshow(corr, color_continuous_scale="Tealgrn", zmin=-1, zmax=1, aspect="auto", template="plotly_dark")
        plot.update_layout(height=390, margin=dict(l=10, r=10, t=25, b=10))
    st.plotly_chart(plot, use_container_width=True, key=f"plot_{uuid.uuid4()}")


st.markdown("""<style>
.stApp { background: radial-gradient(circle at 10% 0%, #102c4a 0, #07111f 42%, #050b14 100%); }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; position: relative; z-index: 1; }
[data-testid="stMetric"] { background: rgba(13, 32, 54, .76); border: 1px solid rgba(29,214,193,.22); padding: 14px; border-radius: 14px; }
</style>""", unsafe_allow_html=True)
components.html(THREE_BACKGROUND, height=250, scrolling=False)

st.title("⚙️ Bearing Intelligence")
st.caption("Dataset-driven diagnostics for vibration health monitoring")

with st.sidebar:
    st.header("System status")
    st.success("Model ready")
    st.write(f"**Estimator:** {type(model).__name__}")
    st.write(f"**Required features:** {len(feature_columns)}")
    st.divider()
    st.caption("Charts are generated from the current upload. No static plot is reused.")

upload_tab, manual_tab, model_tab = st.tabs(["📁 Explore & diagnose", "🧪 Manual prediction", "📊 Model evaluation"])

with upload_tab:
    uploaded_file = st.file_uploader("Upload any CSV with model features", type=["csv"], help="The explorer accepts any CSV. Diagnosis requires the trained feature columns.")
    if uploaded_file is None:
        st.info("Upload a feature CSV to generate live charts and predictions.")
        with st.expander("Required columns for diagnosis"):
            st.code("\n".join(feature_columns))
    else:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded **{uploaded_df.shape[0]:,} rows** × **{uploaded_df.shape[1]} columns**")
            dataset_explorer(uploaded_df)
            with st.expander("Preview uploaded data"):
                st.dataframe(uploaded_df.head(100), use_container_width=True)

            missing = [column for column in feature_columns if column not in uploaded_df.columns]
            if missing:
                st.warning("The explorer is available, but diagnosis needs these missing model columns:")
                st.code("\n".join(missing))
            elif st.button("Run fault diagnosis", type="primary", use_container_width=True):
                with st.spinner("Analysing uploaded observations..."):
                    st.session_state["result_df"] = predict_bearing_state(uploaded_df)

            result_df = st.session_state.get("result_df")
            if result_df is not None:
                st.divider()
                st.subheader("Diagnosis results")
                healthy = int((result_df["prediction"] == 0).sum())
                faulty = int((result_df["prediction"] == 1).sum())
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Observations", f"{len(result_df):,}")
                m2.metric("Healthy", f"{healthy:,}")
                m3.metric("Faulty / degraded", f"{faulty:,}")
                m4.metric("Fault rate", f"{faulty / max(len(result_df), 1):.1%}")

                display = [c for c in ["file_index", "timestamp", "bearing_state", "healthy_probability", "fault_probability"] if c in result_df.columns]
                st.dataframe(result_df[display], use_container_width=True)
                if "fault_probability" in result_df:
                    probability = px.line(result_df, y="fault_probability", color="bearing_state", color_discrete_map={"Healthy": "#1dd6c1", "Faulty / Degraded": "#ff647c"}, template="plotly_dark")
                    probability.update_layout(height=390, xaxis_title="Observation", yaxis_title="Fault probability", yaxis=dict(range=[0, 1]), margin=dict(l=10, r=10, t=25, b=10))
                    st.plotly_chart(probability, use_container_width=True)
                st.download_button("Download prediction results", result_df.to_csv(index=False).encode("utf-8"), "bearing_predictions.csv", "text/csv")
        except Exception as error:
            st.error(f"Unable to process CSV: {error}")

with manual_tab:
    st.subheader("Enter one bearing observation")
    manual_values = {}
    cols = st.columns(2)
    for index, feature in enumerate(feature_columns):
        manual_values[feature] = cols[index % 2].number_input(feature, value=0.0, format="%.6f", key=f"manual_{feature}")
    if st.button("Predict bearing condition", type="primary"):
        try:
            result = predict_bearing_state(pd.DataFrame([manual_values], columns=feature_columns))
            state = result.loc[0, "bearing_state"]
            (st.success if state == "Healthy" else st.error)(f"Prediction: {state}")
            if "fault_probability" in result:
                st.progress(float(result.loc[0, "fault_probability"]), text=f"Estimated fault probability: {result.loc[0, 'fault_probability']:.2%}")
        except Exception as error:
            st.error(f"Prediction failed: {error}")

with model_tab:
    st.subheader("Model evaluation")
    if os.path.exists(CONFUSION_MATRIX_PATH):
        st.image(CONFUSION_MATRIX_PATH, caption="Confusion matrix comparison", use_container_width=True)
    else:
        st.info("No confusion matrix image was found in the plots folder.")
    st.write(f"Selected best model: **{type(model).__name__}**")
    st.warning("The current labels were created with a file-index boundary. Use verified failure labels before treating predictions as a production diagnosis.")
""