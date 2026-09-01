import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DiabetesML — Screening Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Lora:ital,wght@0,500;0,600;1,400&display=swap');

/* App shell */
.stApp                          { background: #F8FAFC !important; font-family: 'Inter',sans-serif !important; }
[data-testid="stAppViewContainer"] { background: #F8FAFC !important; }
[data-testid="stHeader"]        { background: transparent !important; }
#MainMenu, footer, header       { visibility: hidden; }
html, body, [class*="css"]      { font-family: 'Inter', sans-serif !important; }

/* Expander — never touch <details>/<summary> directly */
[data-testid="stExpander"] {
  background: #FFFFFF !important; border: 1px solid #E2E8F0 !important;
  border-radius: 12px !important; overflow: hidden !important;
}
[data-testid="stExpander"] > div:first-child { background: #FFFFFF !important; border-radius: 12px 12px 0 0 !important; }
[data-testid="stExpander"] summary,
[data-testid="stExpander"] [data-baseweb="accordion"] summary { padding: 14px 18px !important; background: #FFFFFF !important; }
[data-testid="stExpander"] summary p,
[data-testid="stExpanderToggleIcon"] ~ * p { color: #0F172A !important; font-size: 0.95rem !important; font-weight: 600 !important; }
[data-testid="stExpanderToggleIcon"] svg { color: #64748B !important; }
[data-testid="stExpander"] [data-testid="stExpanderDetails"] { padding: 4px 18px 18px !important; background: #FFFFFF !important; }

/* Markdown */
.stMarkdown p, .stMarkdown span, .stMarkdown li { color: #0F172A !important; }
.stMarkdown h1,.stMarkdown h2,.stMarkdown h3,.stMarkdown h4 { color: #0F172A !important; }
.stMarkdown strong { color: #0F172A !important; }

/* Widget labels */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span,
[data-testid="stWidgetLabel"] label { color: #0F172A !important; font-weight: 500 !important; }

/* Slider */
[data-testid="stSlider"] > div > div > div > div { background: #2563EB !important; }
[data-baseweb="slider"] [role="slider"] { background: #2563EB !important; border-color: #2563EB !important; }

/* Number input */
[data-testid="stNumberInput"] input {
  color: #0F172A !important; background: #FFFFFF !important;
  border: 1.5px solid #CBD5E1 !important; border-radius: 8px !important;
}
[data-testid="stNumberInputContainer"] button { color: #475569 !important; }

/* Tabs */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
  border-bottom: 2px solid #E2E8F0 !important; background: transparent !important; gap: 0 !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
  background: transparent !important; color: #94A3B8 !important;
  font-weight: 500 !important; font-size: 0.92rem !important;
  border-bottom: 3px solid transparent !important;
  padding: 10px 22px !important; margin-bottom: -2px !important;
}
[data-testid="stTabs"] [aria-selected="true"] { color: #2563EB !important; border-bottom-color: #2563EB !important; }
[data-testid="stTabs"] [data-baseweb="tab"] p,
[data-testid="stTabs"] [data-baseweb="tab"] span { color: inherit !important; }

/* Metrics */
[data-testid="metric-container"] {
  background: #FFFFFF !important; border: 1px solid #E2E8F0 !important;
  border-radius: 12px !important; padding: 18px 20px !important;
}
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] span { color: #475569 !important; font-size: 0.82rem !important; }
[data-testid="stMetricValue"] > div { color: #2563EB !important; font-weight: 700 !important; font-size: 1.8rem !important; }
[data-testid="stMetricDelta"] span  { font-size: 0.78rem !important; }

/* Button */
.stButton > button {
  background: #2563EB !important; color: #FFFFFF !important;
  font-weight: 600 !important; font-size: 1rem !important;
  border: none !important; border-radius: 10px !important;
  padding: 0.85rem 2rem !important; width: 100% !important;
  box-shadow: 0 2px 12px rgba(37,99,235,0.25) !important;
  transition: all 0.2s ease !important;
}
.stButton > button:hover { background: #1D4ED8 !important; transform: translateY(-1px) !important; }
.stButton > button p, .stButton > button span { color: #FFFFFF !important; }

/* Alert */
[data-testid="stAlert"] p { color: #0F172A !important; }

/* HR */
hr { border-color: #E2E8F0 !important; }
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("diabetes_model.pkl")

try:
    model = load_model()
    MODEL_OK = True
except Exception as e:
    MODEL_OK = False
    MODEL_ERR = str(e)

# ── HELPERS ───────────────────────────────────────────────────────────────────
def prediction_style(pred, prob):
    """Return (label, color, bg, border_color) based on model prediction."""
    if pred == 1:
        return "Higher Likelihood of Diabetes", "#B91C1C", "#FEF2F2", "#FCA5A5"
    else:
        return "Lower Likelihood of Diabetes",  "#15803D", "#F0FDF4", "#86EFAC"

def donut_chart(prob_positive):
    """Donut chart showing probability of higher likelihood."""
    pct = round(prob_positive * 100, 1)
    color = "#B91C1C" if prob_positive >= 0.5 else "#15803D"
    fig = go.Figure(go.Pie(
        values=[pct, 100 - pct], hole=0.72,
        marker=dict(colors=[color, "#E2E8F0"]),
        textinfo="none", hoverinfo="skip",
    ))
    fig.add_annotation(
        text=f"<b>{pct}%</b>", x=0.5, y=0.5,
        font=dict(size=32, color=color, family="Inter"),
        showarrow=False,
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        height=215,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def radar_chart(vals):
    """Normalised radar chart of the 8 input features."""
    ranges = [(0,17),(44,200),(24,122),(0,74),(0,416),(18,54.2),(0.078,2.42),(21,81)]
    normed = [round((v - lo) / (hi - lo) * 10, 2) for v, (lo, hi) in zip(vals, ranges)]
    labels = ["Pregnancies","Glucose","Blood\nPressure","Skin\nThickness",
              "Insulin","BMI","Pedigree","Age"]
    fig = go.Figure(go.Scatterpolar(
        r=normed + [normed[0]], theta=labels + [labels[0]],
        fill="toself", fillcolor="rgba(37,99,235,0.10)",
        line=dict(color="#2563EB", width=2.5),
        marker=dict(size=6, color="#2563EB"),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0,10],
                            tickfont=dict(size=9, color="#94A3B8"),
                            gridcolor="#CBD5E1"),
            angularaxis=dict(tickfont=dict(size=11, color="#334155"),
                             gridcolor="#CBD5E1"),
        ),
        showlegend=False, height=280,
        margin=dict(l=30, r=30, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def bar_chart():
    """Dataset class distribution bar chart."""
    fig = go.Figure(go.Bar(
        x=["No Diabetes (0)", "Diabetes (1)"], y=[500, 268],
        marker=dict(color=["#2563EB", "#DC2626"], line=dict(width=0)),
        text=["500  (65.1%)", "268  (34.9%)"],
        textposition="outside",
        textfont=dict(size=13, color="#0F172A"),
    ))
    fig.update_layout(
        height=215,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, tickfont=dict(color="#475569", size=12)),
        yaxis=dict(showgrid=True, gridcolor="#E2E8F0",
                   tickfont=dict(color="#94A3B8"), range=[0, 630]),
        showlegend=False,
    )
    return fig

def metric_tile(col, name, val, unit, low_thresh, high_thresh):
    """Render a colour-coded metric tile with neutral range labels."""
    if val <= low_thresh:
        ic, ib, it, status = "✅", "#F0FDF4", "#15803D", "Within typical range"
    elif val <= high_thresh:
        ic, ib, it, status = "⚠️", "#FFFBEB", "#B45309", "Elevated"
    else:
        ic, ib, it, status = "🔴", "#FEF2F2", "#B91C1C", "Outside typical range"

    col.markdown(f"""
    <div style="background:{ib};border:1.5px solid {it}40;border-radius:12px;
                padding:16px 18px;box-shadow:0 1px 4px rgba(0,0,0,0.04);">
      <p style="margin:0 0 4px;font-size:0.78rem;color:#64748B;font-weight:500;">{ic} {name}</p>
      <p style="margin:0;font-size:1.5rem;font-weight:700;color:{it};line-height:1.2;">
        {val:.1f}<span style="font-size:0.8rem;font-weight:400;"> {unit}</span>
      </p>
      <p style="margin:5px 0 0;font-size:0.75rem;color:{it};font-weight:600;">{status}</p>
      <p style="margin:2px 0 0;font-size:0.7rem;color:#94A3B8;">Informational only</p>
    </div>
    """, unsafe_allow_html=True)

# ── NAV BAR ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:#FFFFFF;border-bottom:1px solid #E2E8F0;padding:14px 32px;
            display:flex;align-items:center;justify-content:space-between;
            margin-bottom:24px;border-radius:0 0 12px 12px;
            box-shadow:0 1px 4px rgba(0,0,0,0.06);">
  <div style="display:flex;align-items:center;gap:12px;">
    <div style="background:#EFF6FF;width:40px;height:40px;border-radius:10px;
                display:flex;align-items:center;justify-content:center;font-size:1.35rem;">🩺</div>
    <div>
      <span style="font-family:'Lora',serif;font-size:1.25rem;font-weight:600;color:#0F172A;">
        DiabetesML <span style="color:#2563EB;">Dashboard</span>
      </span>
      <span style="margin-left:12px;font-size:0.7rem;color:#94A3B8;
                   letter-spacing:0.1em;text-transform:uppercase;font-weight:500;">
        ML Screening Tool
      </span>
    </div>
  </div>
  <div style="display:flex;gap:8px;align-items:center;">
    <span style="background:#F0FDF4;color:#15803D;border:1px solid #86EFAC;
                 border-radius:20px;font-size:0.74rem;font-weight:600;padding:4px 13px;">
      ● ML Powered
    </span>
    <span style="background:#EFF6FF;color:#1D4ED8;border:1px solid #93C5FD;
                 border-radius:20px;font-size:0.74rem;font-weight:600;padding:4px 13px;">
      GaussianNB · 768 Samples
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

if not MODEL_OK:
    st.error(f"Place **diabetes_model.pkl** in the same folder as this script.\n\n`{MODEL_ERR}`")
    st.stop()

# ── DISCLAIMER BANNER ─────────────────────────────────────────────────────────
st.markdown("""
<div style="background:#FFFBEB;border:1px solid #FCD34D;border-left:4px solid #F59E0B;
            border-radius:10px;padding:13px 20px;margin-bottom:22px;
            display:flex;align-items:flex-start;gap:12px;">
  <span style="font-size:1.1rem;margin-top:1px;flex-shrink:0;">⚠️</span>
  <span style="font-size:0.86rem;color:#78350F;line-height:1.6;">
    <strong style="color:#92400E;">Medical Disclaimer:</strong>
    This application is an educational Machine Learning project and does not provide a medical diagnosis.
    Predictions should not replace professional medical advice, diagnosis, or treatment.
    Please consult a qualified healthcare professional for medical concerns.
  </span>
</div>
""", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "  🔬  Patient Screening  ",
    "  📊  Model Insights  ",
    "  📋  About  ",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — PATIENT SCREENING
# ═════════════════════════════════════════════════════════════════════════════
with tab1:

    st.markdown("""
    <h2 style="font-family:'Lora',serif;font-size:1.5rem;font-weight:600;
               color:#0F172A;margin:8px 0 4px;">Health Parameter Inputs</h2>
    <p style="color:#64748B;font-size:0.92rem;margin:0 0 22px;">
      Adjust the clinical values below. The screening result updates automatically.
    </p>
    """, unsafe_allow_html=True)

    # ── ROW 1 — inputs ────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown('<p style="color:#0F172A;font-weight:600;font-size:0.88rem;margin:0 0 6px;">🤰 Pregnancies</p>', unsafe_allow_html=True)
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=17, value=3, step=1, label_visibility="collapsed")
        st.markdown('<p style="color:#64748B;font-size:0.78rem;margin:4px 0 0;">Range: 0 – 17 &nbsp;|&nbsp; Can be zero</p>', unsafe_allow_html=True)

    with c2:
        st.markdown('<p style="color:#0F172A;font-weight:600;font-size:0.88rem;margin:0 0 6px;">🩸 Glucose (mg/dL)</p>', unsafe_allow_html=True)
        glucose = st.slider("Glucose", min_value=44, max_value=200, value=120, step=1, label_visibility="collapsed")
        st.markdown(f'<p style="color:#64748B;font-size:0.78rem;margin:4px 0 0;">Value: <b style="color:#0F172A;">{glucose}</b> &nbsp;|&nbsp; Dataset range: 44–200</p>', unsafe_allow_html=True)

    with c3:
        st.markdown('<p style="color:#0F172A;font-weight:600;font-size:0.88rem;margin:0 0 6px;">💉 Blood Pressure (mmHg)</p>', unsafe_allow_html=True)
        blood_pressure = st.slider("BP", min_value=24, max_value=122, value=70, step=1, label_visibility="collapsed")
        st.markdown(f'<p style="color:#64748B;font-size:0.78rem;margin:4px 0 0;">Value: <b style="color:#0F172A;">{blood_pressure}</b> &nbsp;|&nbsp; Dataset range: 24–122</p>', unsafe_allow_html=True)

    with c4:
        st.markdown('<p style="color:#0F172A;font-weight:600;font-size:0.88rem;margin:0 0 6px;">📏 Skin Thickness (mm)</p>', unsafe_allow_html=True)
        skin_thickness = st.slider("Skin", min_value=1, max_value=74, value=20, step=1, label_visibility="collapsed")
        st.markdown(f'<p style="color:#64748B;font-size:0.78rem;margin:4px 0 0;">Value: <b style="color:#0F172A;">{skin_thickness}</b> mm &nbsp;|&nbsp; Dataset range: 1–74</p>', unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── ROW 2 — inputs ────────────────────────────────────────────────────────
    c5, c6, c7, c8 = st.columns(4)

    with c5:
        st.markdown('<p style="color:#0F172A;font-weight:600;font-size:0.88rem;margin:0 0 6px;">💊 Insulin (µU/mL)</p>', unsafe_allow_html=True)
        insulin = st.slider("Insulin", min_value=1, max_value=416, value=80, step=1, label_visibility="collapsed")
        st.markdown(f'<p style="color:#64748B;font-size:0.78rem;margin:4px 0 0;">Value: <b style="color:#0F172A;">{insulin}</b> &nbsp;|&nbsp; Dataset range: 1–416</p>', unsafe_allow_html=True)

    with c6:
        st.markdown('<p style="color:#0F172A;font-weight:600;font-size:0.88rem;margin:0 0 6px;">⚖️ BMI (kg/m²)</p>', unsafe_allow_html=True)
        bmi = st.slider("BMI", min_value=18.0, max_value=54.2, value=27.0, step=0.1, label_visibility="collapsed")
        bmi_cat = "Normal weight" if bmi < 25 else "Overweight" if bmi < 30 else "Obese range"
        st.markdown(f'<p style="color:#64748B;font-size:0.78rem;margin:4px 0 0;">Value: <b style="color:#0F172A;">{bmi:.1f}</b> &nbsp;|&nbsp; {bmi_cat}</p>', unsafe_allow_html=True)

    with c7:
        st.markdown('<p style="color:#0F172A;font-weight:600;font-size:0.88rem;margin:0 0 6px;">🧬 Diabetes Pedigree Function</p>', unsafe_allow_html=True)
        dpf = st.slider("DPF", min_value=0.078, max_value=2.420, value=0.470, step=0.001, format="%.3f", label_visibility="collapsed")
        st.markdown(f'<p style="color:#64748B;font-size:0.78rem;margin:4px 0 0;">Value: <b style="color:#0F172A;">{dpf:.3f}</b> &nbsp;|&nbsp; Genetic risk proxy</p>', unsafe_allow_html=True)

    with c8:
        st.markdown('<p style="color:#0F172A;font-weight:600;font-size:0.88rem;margin:0 0 6px;">🎂 Age (years)</p>', unsafe_allow_html=True)
        age = st.number_input("Age", min_value=21, max_value=81, value=33, step=1, label_visibility="collapsed")
        st.markdown('<p style="color:#64748B;font-size:0.78rem;margin:4px 0 0;">Range: 21 – 81 years</p>', unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── PREDICT BUTTON ─────────────────────────────────────────────────────────
    _, btn_col, _ = st.columns([2, 1.5, 2])
    with btn_col:
        st.button("🔬  Run Screening")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── PREDICT using named DataFrame columns (matching training data) ──────────
    features_df = pd.DataFrame([{
        "Pregnancies":              pregnancies,
        "Glucose":                  glucose,
        "BloodPressure":            blood_pressure,
        "SkinThickness":            skin_thickness,
        "Insulin":                  insulin,
        "BMI":                      bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age":                      age,
    }])

    prediction   = model.predict(features_df)[0]
    probabilities = model.predict_proba(features_df)[0]   # [prob_class0, prob_class1]
    prob_positive = probabilities[1]
    prob_negative = probabilities[0]

    result_label, res_color, res_bg, res_border = prediction_style(prediction, prob_positive)
    icon = "🔴" if prediction == 1 else "🟢"

    # ── RESULT SECTION ─────────────────────────────────────────────────────────
    st.markdown('<hr style="border:none;border-top:1px solid #E2E8F0;margin:4px 0 20px;">', unsafe_allow_html=True)
    st.markdown('<h3 style="font-family:\'Lora\',serif;font-size:1.2rem;font-weight:600;color:#0F172A;margin:0 0 18px;">Screening Result</h3>', unsafe_allow_html=True)

    rl, rm, rr = st.columns([1.2, 1, 1.8])

    with rl:
        st.markdown(f"""
        <div style="background:{res_bg};border:2px solid {res_border};border-radius:16px;
                    padding:28px 20px;text-align:center;box-shadow:0 2px 10px rgba(0,0,0,0.05);">
          <div style="font-size:2.4rem;margin-bottom:10px;">{icon}</div>
          <p style="font-family:'Lora',serif;font-size:1.1rem;font-weight:600;
                    color:{res_color};margin:0 0 12px;line-height:1.4;">{result_label}</p>
          <div style="display:flex;flex-direction:column;gap:8px;margin-top:14px;">
            <div style="background:rgba(255,255,255,0.7);border-radius:8px;padding:8px 12px;">
              <p style="margin:0;font-size:0.75rem;color:#64748B;">Prob. Lower Likelihood</p>
              <p style="margin:0;font-size:1.1rem;font-weight:700;color:#15803D;">{prob_negative*100:.1f}%</p>
            </div>
            <div style="background:rgba(255,255,255,0.7);border-radius:8px;padding:8px 12px;">
              <p style="margin:0;font-size:0.75rem;color:#64748B;">Prob. Higher Likelihood</p>
              <p style="margin:0;font-size:1.1rem;font-weight:700;color:#B91C1C;">{prob_positive*100:.1f}%</p>
            </div>
          </div>
          <p style="color:#64748B;font-size:0.78rem;margin:14px 0 0;line-height:1.6;">
            GaussianNB classification model<br>Pima Indians Diabetes Dataset
          </p>
        </div>
        """, unsafe_allow_html=True)

    with rm:
        st.markdown('<p style="font-weight:600;color:#0F172A;font-size:0.9rem;margin-bottom:4px;">Risk Probability</p>', unsafe_allow_html=True)
        st.plotly_chart(donut_chart(prob_positive), use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown('<p style="font-size:0.72rem;color:#94A3B8;text-align:center;margin-top:-8px;">Probability of Higher Likelihood</p>', unsafe_allow_html=True)

    with rr:
        st.markdown('<p style="font-weight:600;color:#0F172A;font-size:0.9rem;margin-bottom:4px;">Clinical Profile Radar</p>', unsafe_allow_html=True)
        features_list = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
        st.plotly_chart(radar_chart(features_list), use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown('<p style="font-size:0.72rem;color:#94A3B8;text-align:center;margin-top:-8px;">Normalised feature profile — for visualisation only</p>', unsafe_allow_html=True)

    # ── METRIC TILES ───────────────────────────────────────────────────────────
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    tile_cols = st.columns(4)
    metric_tile(tile_cols[0], "Glucose",        glucose,        "mg/dL", 100, 125)
    metric_tile(tile_cols[1], "Blood Pressure", blood_pressure, "mmHg",   80,  90)
    metric_tile(tile_cols[2], "BMI",            bmi,            "kg/m²",  25,  30)
    metric_tile(tile_cols[3], "Insulin",        insulin,        "µU/mL",  25, 200)
    st.markdown('<p style="font-size:0.72rem;color:#94A3B8;margin-top:6px;">* Thresholds shown are general informational references only — not a clinical diagnosis.</p>', unsafe_allow_html=True)

    # ── GUIDANCE SECTION ───────────────────────────────────────────────────────
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    with st.expander("Guidance & Next Steps", expanded=(prediction == 1)):
        if prediction == 1:
            guidance = [
                ("💬 Speak to a professional",
                 "Consider discussing this screening result with a qualified healthcare professional."),
                ("🥗 Healthy lifestyle",
                 "Maintain healthy lifestyle habits including a balanced diet and regular physical activity."),
                ("📅 Routine check-ups",
                 "Regular health check-ups allow for ongoing monitoring of relevant health parameters."),
                ("📌 Important note",
                 "This is a machine learning screening result, not a medical diagnosis. It is based on statistical patterns in the Pima Indians Diabetes Dataset."),
            ]
        else:
            guidance = [
                ("✅ Lower predicted likelihood",
                 "This result indicates a lower predicted likelihood based on the model inputs."),
                ("🥗 Keep up healthy habits",
                 "Continue maintaining healthy lifestyle habits and consider routine health check-ups."),
                ("📅 Routine screening",
                 "Periodic health assessments are generally recommended as part of preventive care."),
                ("📌 Important note",
                 "This is a machine learning screening result, not a medical diagnosis. It is based on statistical patterns in the Pima Indians Diabetes Dataset."),
            ]
        rc1, rc2 = st.columns(2)
        for i, (title_text, body_text) in enumerate(guidance):
            [rc1, rc2][i % 2].markdown(f"""
            <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-left:3px solid #2563EB;
                        border-radius:10px;padding:16px 18px;margin-bottom:10px;">
              <p style="margin:0 0 5px;font-weight:700;color:#0F172A;font-size:0.91rem;">{title_text}</p>
              <p style="margin:0;color:#475569;font-size:0.84rem;line-height:1.65;">{body_text}</p>
            </div>
            """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL INSIGHTS
# ═════════════════════════════════════════════════════════════════════════════
with tab2:

    st.markdown("""
    <h2 style="font-family:'Lora',serif;font-size:1.5rem;font-weight:600;
               color:#0F172A;margin:8px 0 4px;">Model Performance Insights</h2>
    <p style="color:#64748B;font-size:0.92rem;margin:0 0 22px;">
      Key metrics and dataset facts from the training notebook.
    </p>
    """, unsafe_allow_html=True)

    # Metric cards — corrected values
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Training Samples", "768",       "Pima Indians Dataset")
    m2.metric("Test Accuracy",    "72.7%",     "on held-out test set")
    m3.metric("Algorithm",        "GaussianNB","Gaussian Naive Bayes")
    m4.metric("Input Features",   "8",         "Clinical parameters")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    ca, cb = st.columns(2)

    with ca:
        # Performance breakdown
        st.markdown('<h4 style="color:#0F172A;font-size:0.98rem;font-weight:700;margin-bottom:12px;">Positive Class Performance</h4>', unsafe_allow_html=True)
        for metric_name, value, note in [
            ("Precision",  "74%", "Of predicted positives, 74% are true positives"),
            ("Recall",     "80%", "Correctly identified 80% of actual positive cases"),
            ("F1 Score",   "77%", "Harmonic mean of Precision and Recall"),
            ("Accuracy",  "72.7%","Overall correct predictions on the test set"),
        ]:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:10px 14px;margin-bottom:6px;
                        background:#FFFFFF;border:1px solid #E2E8F0;border-radius:8px;">
              <div>
                <span style="font-weight:600;color:#1E293B;font-size:0.86rem;">{metric_name}</span>
                <p style="margin:2px 0 0;color:#94A3B8;font-size:0.75rem;">{note}</p>
              </div>
              <span style="background:#EFF6FF;color:#1D4ED8;border:1px solid #BFDBFE;
                           border-radius:6px;padding:3px 12px;font-size:0.82rem;font-weight:700;">
                {value}
              </span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        # Confusion matrix
        st.markdown('<h4 style="color:#0F172A;font-size:0.98rem;font-weight:700;margin-bottom:10px;">Confusion Matrix (Test Set)</h4>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;padding:16px;">
          <table style="width:100%;border-collapse:collapse;font-size:0.84rem;">
            <thead>
              <tr>
                <th style="padding:8px;color:#94A3B8;font-weight:500;text-align:left;"></th>
                <th style="padding:8px;color:#15803D;font-weight:600;text-align:center;">Predicted: 0</th>
                <th style="padding:8px;color:#B91C1C;font-weight:600;text-align:center;">Predicted: 1</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td style="padding:8px;color:#15803D;font-weight:600;">Actual: 0</td>
                <td style="padding:10px;background:#F0FDF4;color:#15803D;font-weight:700;text-align:center;border-radius:6px;">42</td>
                <td style="padding:10px;background:#FEF2F2;color:#B91C1C;font-weight:700;text-align:center;border-radius:6px;">24</td>
              </tr>
              <tr>
                <td style="padding:8px;color:#B91C1C;font-weight:600;">Actual: 1</td>
                <td style="padding:10px;background:#FEF2F2;color:#B91C1C;font-weight:700;text-align:center;border-radius:6px;">18</td>
                <td style="padding:10px;background:#F0FDF4;color:#15803D;font-weight:700;text-align:center;border-radius:6px;">70</td>
              </tr>
            </tbody>
          </table>
          <p style="color:#94A3B8;font-size:0.74rem;margin:10px 0 0;">Test set: 154 samples (20% of 768)</p>
        </div>
        """, unsafe_allow_html=True)

    with cb:
        st.markdown('<h4 style="color:#0F172A;font-size:0.98rem;font-weight:700;margin-bottom:12px;">Dataset Class Distribution</h4>', unsafe_allow_html=True)
        st.plotly_chart(bar_chart(), use_container_width=True, config={"displayModeBar": False})

        st.markdown('<h4 style="color:#0F172A;font-size:0.98rem;font-weight:700;margin:14px 0 10px;">ML Pipeline</h4>', unsafe_allow_html=True)
        for step_title, step_desc in [
            ("1 · Load & Explore",   "768 samples · 8 features"),
            ("2 · Data Cleaning",    "Invalid zeros → NaN → median imputation for Glucose, BP, Skin, Insulin, BMI"),
            ("3 · Train/Test Split", "80% training · 20% testing · stratified split"),
            ("4 · Model Comparison", "Multiple classification algorithms evaluated"),
            ("5 · GaussianNB",       "Selected as the final model — no feature scaling required"),
            ("6 · Export",           "joblib.dump → diabetes_model.pkl"),
        ]:
            st.markdown(f"""
            <div style="display:flex;gap:12px;align-items:flex-start;padding:9px 14px;margin-bottom:5px;
                        background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;">
              <span style="background:#2563EB;color:#FFFFFF;border-radius:6px;padding:3px 10px;
                           font-size:0.72rem;font-weight:700;white-space:nowrap;flex-shrink:0;">{step_title}</span>
              <span style="color:#334155;font-size:0.84rem;line-height:1.5;">{step_desc}</span>
            </div>
            """, unsafe_allow_html=True)

    # Model notes
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:12px;padding:18px 22px;">
      <p style="color:#1E40AF;font-size:0.87rem;line-height:1.75;margin:0;">
        <strong style="color:#1E3A8A;">Model selection note:</strong>
        Gaussian Naive Bayes was selected after comparing multiple classification models.
        It provided a balanced performance with strong recall for identifying positive cases (80%)
        while maintaining competitive overall accuracy (72.7%). This makes it well-suited for
        a screening context where missing true positive cases is more costly than false alarms.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Feature reference table
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown('<h4 style="color:#0F172A;font-size:0.98rem;font-weight:700;margin-bottom:12px;">Feature Reference Ranges (Dataset)</h4>', unsafe_allow_html=True)
    for feat, lo, hi in [
        ("Pregnancies",                 "0",     "17"),
        ("Glucose (mg/dL)",             "44",    "200"),
        ("Blood Pressure (mmHg)",       "24",    "122"),
        ("Skin Thickness (mm)",         "1",     "74"),
        ("Insulin (µU/mL)",             "1",     "416"),
        ("BMI (kg/m²)",                 "18.0",  "54.2"),
        ("Diabetes Pedigree Function",  "0.078", "2.420"),
        ("Age (years)",                 "21",    "81"),
    ]:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
                    padding:9px 14px;margin-bottom:5px;
                    background:#FFFFFF;border:1px solid #E2E8F0;border-radius:8px;">
          <span style="font-weight:500;color:#1E293B;font-size:0.84rem;">{feat}</span>
          <span style="color:#94A3B8;font-size:0.8rem;">Dataset range: {lo} – {hi}</span>
        </div>
        """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    a1, a2 = st.columns([3, 2])

    with a1:
        st.markdown("""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:14px;
                    padding:28px;box-shadow:0 1px 6px rgba(0,0,0,0.05);margin-bottom:14px;">
          <h3 style="font-family:'Lora',serif;font-size:1.25rem;color:#0F172A;margin-top:0;">
            About This Project
          </h3>
          <p style="color:#475569;line-height:1.75;font-size:0.9rem;margin-bottom:12px;">
            This project demonstrates an end-to-end Machine Learning classification workflow,
            from data preprocessing and model comparison to deployment using Streamlit.
            It was built on the
            <strong style="color:#0F172A;">Pima Indians Diabetes Dataset</strong> — a
            well-known benchmark dataset from the National Institute of Diabetes and
            Digestive and Kidney Diseases (NIDDK), containing 768 records across 8 clinical features.
          </p>
          <p style="color:#475569;line-height:1.75;font-size:0.9rem;margin-bottom:0;">
            The final model is <strong style="color:#0F172A;">Gaussian Naive Bayes (GaussianNB)</strong>,
            selected after comparing multiple classifiers. It outputs calibrated probabilities,
            making it suitable for a screening context where understanding confidence levels
            matters alongside the prediction itself.
          </p>
        </div>
        """, unsafe_allow_html=True)

        # ML Workflow
        st.markdown('<h4 style="color:#0F172A;font-size:0.98rem;font-weight:700;margin-bottom:10px;">Machine Learning Workflow</h4>', unsafe_allow_html=True)
        for step_title, desc in [
            ("Data Loading & EDA",       "Explored 768 samples across 8 clinical features and the binary target variable."),
            ("Data Cleaning",            "Invalid zero values for Glucose, Blood Pressure, Skin Thickness, Insulin, and BMI were replaced with NaN and imputed using column medians."),
            ("Train / Test Split",       "Stratified 80/20 split to preserve class balance across training and test sets."),
            ("Model Comparison",         "Multiple classification algorithms were evaluated before selecting the final model."),
            ("GaussianNB — Final Model", "Gaussian Naive Bayes selected for its strong recall on the positive class and overall competitive performance."),
            ("Deployment",               "Model exported with Joblib and deployed as an interactive Streamlit dashboard."),
        ]:
            st.markdown(f"""
            <div style="display:flex;gap:12px;align-items:flex-start;padding:11px 16px;
                        margin-bottom:7px;background:#F8FAFC;border:1px solid #E2E8F0;border-radius:9px;">
              <span style="font-weight:700;color:#0F172A;min-width:190px;font-size:0.87rem;
                           flex-shrink:0;">{step_title}</span>
              <span style="color:#475569;font-size:0.84rem;line-height:1.55;">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

    with a2:
        # Tech stack
        st.markdown("""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:14px;
                    padding:24px;margin-bottom:14px;box-shadow:0 1px 6px rgba(0,0,0,0.05);">
          <h4 style="color:#0F172A;margin-top:0;font-family:'Lora',serif;font-size:1.05rem;">
            Technologies Used
          </h4>
        """, unsafe_allow_html=True)

        for name, role in [
            ("🐍 Python 3.x",       "Core language"),
            ("🐼 Pandas",           "Data processing"),
            ("🔢 NumPy",            "Numerical operations"),
            ("🤖 Scikit-learn",     "ML algorithms & evaluation"),
            ("🧮 GaussianNB",       "Final classification model"),
            ("🗂️ Joblib",           "Model serialisation"),
            ("🌐 Streamlit",        "Web interface"),
            ("📈 Plotly",           "Interactive charts"),
        ]:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:8px 12px;margin-bottom:5px;background:#F8FAFC;border-radius:8px;">
              <span style="font-weight:600;color:#0F172A;font-size:0.85rem;">{name}</span>
              <span style="color:#64748B;font-size:0.82rem;">{role}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Dataset reference
        st.markdown("""
        <div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:14px;padding:22px;">
          <h4 style="color:#1D4ED8;margin-top:0;font-size:0.95rem;font-weight:700;">
            📚 Dataset Reference
          </h4>
          <p style="color:#1E40AF;font-size:0.84rem;line-height:1.75;margin:0;">
            <strong style="color:#1E3A8A;">Pima Indians Diabetes Dataset</strong><br>
            National Institute of Diabetes &amp;<br>Digestive &amp; Kidney Diseases<br><br>
            768 female patients · Ages 21–81<br>
            8 clinical features · Binary outcome<br><br>
            <span style="color:#15803D;font-weight:600;">● Class 0 (No Diabetes): 500 (65.1%)</span><br>
            <span style="color:#B91C1C;font-weight:600;">● Class 1 (Diabetes): 268 (34.9%)</span>
          </p>
        </div>
        """, unsafe_allow_html=True)

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="border-top:1px solid #E2E8F0;padding-top:16px;
            display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
  <span style="color:#94A3B8;font-size:0.78rem;">
    Educational ML project · Not a medical diagnostic tool
  </span>
  <span style="color:#94A3B8;font-size:0.78rem;">
    Streamlit · GaussianNB · Pima Indians Dataset · 768 Samples
  </span>
</div>
""", unsafe_allow_html=True)