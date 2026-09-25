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

# Animated Three.js canvas using generated particles only; no 3D assets are imported.
THREE_BACKGROUND = """
<!doctype html><html><head><style>
html,body,#scene{margin:0;width:100%;height:100%;overflow:hidden;background:#050b14}
#scene{position:fixed;inset:0}
</style></head><body><div id="scene"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const host=document.getElementById('scene'), scene=new THREE.Scene();
const camera=new THREE.PerspectiveCamera(55,innerWidth/innerHeight,1,1800); camera.position.z=520;
const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2)); renderer.setSize(innerWidth,innerHeight); host.appendChild(renderer.domElement);
const count=700, positions=new Float32Array(count*3), colors=new Float32Array(count*3), speeds=[];
for(let i=0;i<count;i++){positions[i*3]=(Math.random()-.5)*1200;positions[i*3+1]=(Math.random()-.5)*720;positions[i*3+2]=(Math.random()-.5)*1000;speeds.push(.0003+Math.random()*.0007);const c=new THREE.Color(Math.random()>.52?'#1dd6c1':'#4f7cff');colors[i*3]=c.r;colors[i*3+1]=c.g;colors[i*3+2]=c.b;}
const geometry=new THREE.BufferGeometry(); geometry.setAttribute('position',new THREE.BufferAttribute(positions,3)); geometry.setAttribute('color',new THREE.BufferAttribute(colors,3));
const material=new THREE.PointsMaterial({size:2,vertexColors:true,transparent:true,opacity:.42,blending:THREE.AdditiveBlending});
const particles=new THREE.Points(geometry,material); scene.add(particles);
const clock=new THREE.Clock(); let targetX=0,targetY=0;
addEventListener('pointermove',e=>{targetX=(e.clientX/innerWidth-.5)*.14;targetY=(e.clientY/innerHeight-.5)*.08});
addEventListener('resize',()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight)});
function animate(){requestAnimationFrame(animate);const t=clock.getElapsedTime();particles.rotation.y+=.00022;particles.rotation.x+=.00006;particles.position.x+=(targetX-particles.position.x)*.018;particles.position.y+=(-targetY-particles.position.y)*.018;material.opacity=.34+Math.sin(t*.55)*.06;renderer.render(scene,camera)} animate();
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


def gauge(value, title="Fault probability"):
    value = max(0.0, min(1.0, float(value)))
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={"suffix": "%", "font": {"size": 34, "color": "#f4f8ff"}},
        title={"text": title, "font": {"size": 16, "color": "#b8c9df"}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%", "tickcolor": "#8da3bf", "tickfont": {"color": "#8da3bf"}},
            "bar": {"color": "#ff647c", "thickness": .24},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 35], "color": "rgba(29,214,193,.30)"},
                {"range": [35, 70], "color": "rgba(255,193,93,.28)"},
                {"range": [70, 100], "color": "rgba(255,100,124,.28)"},
            ],
            "threshold": {"line": {"color": "#ffffff", "width": 4}, "thickness": .78, "value": 50},
        },
    ))
    fig.update_layout(height=260, margin=dict(l=18, r=18, t=48, b=8), paper_bgcolor="rgba(0,0,0,0)", font_color="#f4f8ff")
    return fig


def analytics(frame):
    numeric = numeric_columns(frame)
    if not numeric:
        st.info("No numeric columns are available for advanced analytics.")
        return
    valid = frame[numeric].apply(pd.to_numeric, errors="coerce")
    summary = pd.DataFrame({
        "Feature": numeric,
        "Missing": valid.isna().sum().values,
        "Unique": valid.nunique().values,
        "Mean": valid.mean().values,
        "Std dev": valid.std().values,
        "Min": valid.min().values,
        "Max": valid.max().values,
    }).sort_values("Std dev", ascending=False)
    st.dataframe(summary.style.format({"Mean": "{:.4g}", "Std dev": "{:.4g}", "Min": "{:.4g}", "Max": "{:.4g}"}), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        variability = summary.head(min(12, len(summary))).sort_values("Std dev")
        fig = px.bar(variability, x="Std dev", y="Feature", orientation="h", title="Most variable signals", template="plotly_dark", color="Std dev", color_continuous_scale="Tealgrn")
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=45, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        missing = summary[summary["Missing"] > 0].sort_values("Missing", ascending=False)
        if missing.empty:
            st.success("Data quality check: no missing numeric values detected.")
        else:
            fig = px.bar(missing, x="Feature", y="Missing", title="Missing numeric values", template="plotly_dark", color="Missing", color_continuous_scale="Sunset")
            fig.update_layout(height=400, margin=dict(l=10, r=10, t=45, b=10), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)


def dataset_explorer(frame):
    st.subheader("Live dataset explorer")
    numeric = numeric_columns(frame)
    if not numeric:
        st.info("This dataset has no numeric columns to chart yet.")
        return
    controls = st.columns([1.2, 1.2, 1, 1])
    x_options = ["Row index"] + frame.columns.tolist()
    x_choice = controls[0].selectbox("X-axis", x_options, key=f"x_{id(frame)}")
    y_choice = controls[1].selectbox("Feature", numeric, key=f"y_{id(frame)}")
    chart_type = controls[2].selectbox("Chart", ["Trend", "Distribution", "Correlation"], key=f"chart_{id(frame)}")
    rolling = controls[3].slider("Smoothing", 1, 40, 1, key=f"smooth_{id(frame)}")
    chart_frame = frame.copy().reset_index(drop=True)
    if rolling > 1 and chart_type == "Trend":
        chart_frame[y_choice] = pd.to_numeric(chart_frame[y_choice], errors="coerce").rolling(rolling, min_periods=1).mean()
    if chart_type == "Trend":
        x = chart_frame.index if x_choice == "Row index" else chart_frame[x_choice]
        fig = px.line(x=x, y=chart_frame[y_choice], labels={"x": x_choice, "y": y_choice}, template="plotly_dark")
        fig.update_traces(line_color="#1dd6c1", line_width=2)
    elif chart_type == "Distribution":
        fig = px.histogram(chart_frame, x=y_choice, nbins=35, marginal="box", template="plotly_dark", color_discrete_sequence=["#4f7cff"])
    else:
        fig = px.imshow(chart_frame[numeric].corr(), color_continuous_scale="Tealgrn", zmin=-1, zmax=1, aspect="auto", template="plotly_dark")
    fig.update_layout(height=390, margin=dict(l=10, r=10, t=25, b=10), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True, key=f"plot_{uuid.uuid4()}")


st.markdown("""<style>
.stApp{background:radial-gradient(circle at 10% 0%,#102c4a 0,#07111f 42%,#050b14 100%)}
[data-testid="stHeader"]{background:transparent}.block-container{padding-top:1.5rem;position:relative;z-index:1}
[data-testid="stMetric"]{background:rgba(13,32,54,.78);border:1px solid rgba(29,214,193,.22);padding:14px;border-radius:14px}
.dashboard-card{background:linear-gradient(135deg,rgba(17,43,72,.92),rgba(8,22,39,.92));border:1px solid rgba(118,158,201,.18);border-radius:18px;padding:20px 22px;min-height:100px;box-shadow:0 14px 36px rgba(0,0,0,.16)}
.card-label{color:#8da3bf;font-size:.78rem;text-transform:uppercase;letter-spacing:.11em}.card-value{color:#f4f8ff;font-size:2rem;font-weight:700;margin-top:7px}.card-help{color:#9fb2c9;font-size:.82rem;margin-top:4px}
</style>""", unsafe_allow_html=True)
components.html(THREE_BACKGROUND, height=260, scrolling=False)
st.title("⚙️ Bearing Intelligence")
st.caption("Real-time health monitoring for uploaded vibration datasets")

with st.sidebar:
    st.header("System status")
    st.success("Model ready")
    st.write(f"**Estimator:** {type(model).__name__}")
    st.write(f"**Required features:** {len(feature_columns)}")
    st.divider()
    st.caption("Charts, analytics, and health cards are recalculated from the current upload.")

upload_tab, manual_tab, model_tab = st.tabs(["📁 Explore & diagnose", "🧪 Manual prediction", "📊 Model evaluation"])

with upload_tab:
    uploaded_file = st.file_uploader("Upload any CSV with model features", type=["csv"], help="Exploration works with any CSV. Diagnosis additionally requires the trained feature columns.")
    if uploaded_file is None:
        st.info("Upload a feature CSV to populate the live health dashboard.")
        with st.expander("Required columns for diagnosis"):
            st.code("\n".join(feature_columns))
    else:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            file_token = f"{uploaded_file.name}:{uploaded_file.size}"
            if st.session_state.get("file_token") != file_token:
                st.session_state["file_token"] = file_token
                st.session_state.pop("result_df", None)
            numeric = numeric_columns(uploaded_df)
            st.success(f"Loaded **{uploaded_df.shape[0]:,} rows** × **{uploaded_df.shape[1]} columns**")
            st.markdown("### Dataset health overview")
            quality = uploaded_df[numeric].isna().sum().sum() if numeric else 0
            unique_ratio = uploaded_df.nunique().mean() / max(len(uploaded_df), 1) if len(uploaded_df) else 0
            cards = [
                ("Rows analysed", f"{len(uploaded_df):,}", "observations in upload"),
                ("Signals", f"{len(numeric):,}", "numeric features detected"),
                ("Data quality", "Clean" if quality == 0 else f"{quality:,} missing", "numeric values checked"),
                ("Diversity", f"{unique_ratio:.1%}", "average unique ratio"),
            ]
            cols = st.columns(4)
            for col, (label, value, help_text) in zip(cols, cards):
                col.markdown(f'<div class="dashboard-card"><div class="card-label">{label}</div><div class="card-value">{value}</div><div class="card-help">{help_text}</div></div>', unsafe_allow_html=True)

            dataset_explorer(uploaded_df)
            with st.expander("Advanced dataset analytics", expanded=True):
                analytics(uploaded_df)
            with st.expander("Preview uploaded data"):
                st.dataframe(uploaded_df.head(100), use_container_width=True)

            missing = [column for column in feature_columns if column not in uploaded_df.columns]
            if missing:
                st.warning("Exploration is available, but diagnosis needs these missing model columns:")
                st.code("\n".join(missing))
            elif st.button("Run fault diagnosis", type="primary", use_container_width=True):
                with st.spinner("Analysing uploaded observations..."):
                    st.session_state["result_df"] = predict_bearing_state(uploaded_df)

            result_df = st.session_state.get("result_df")
            if result_df is not None:
                st.divider()
                st.subheader("Health dashboard")
                healthy = int((result_df["prediction"] == 0).sum())
                faulty = int((result_df["prediction"] == 1).sum())
                rate = faulty / max(len(result_df), 1)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Observations", f"{len(result_df):,}")
                m2.metric("Healthy", f"{healthy:,}")
                m3.metric("Faulty / degraded", f"{faulty:,}")
                m4.metric("Fault rate", f"{rate:.1%}")
                if "fault_probability" in result_df:
                    g1, g2 = st.columns([1, 1.6])
                    with g1:
                        gauge(result_df["fault_probability"].mean())
                        st.caption("Mean predicted risk across uploaded observations")
                    with g2:
                        probability = px.line(result_df, y="fault_probability", color="bearing_state", color_discrete_map={"Healthy": "#1dd6c1", "Faulty / Degraded": "#ff647c"}, template="plotly_dark")
                        probability.update_layout(height=260, xaxis_title="Observation", yaxis_title="Fault probability", yaxis=dict(range=[0, 1]), margin=dict(l=10, r=10, t=25, b=10), legend_title="State")
                        st.plotly_chart(probability, use_container_width=True)
                display = [c for c in ["file_index", "timestamp", "bearing_state", "healthy_probability", "fault_probability"] if c in result_df.columns]
                st.dataframe(result_df[display], use_container_width=True)
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
                st.plotly_chart(gauge(result.loc[0, "fault_probability"], "Manual prediction risk"), use_container_width=True)
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
