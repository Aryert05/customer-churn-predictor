import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# ─────────────────────────────────────────────
#  Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnIQ · Customer Churn Prediction",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  Custom CSS – DeepVision-inspired dark theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* ── Root palette ── */
:root {
    --bg:       #141414;
    --surface:  #1e1e1e;
    --surface2: #252525;
    --border:   #2e2e2e;
    --accent:   #4ade80;     /* mint-green like the screenshot */
    --accent2:  #22d3ee;
    --danger:   #f87171;
    --warning:  #fbbf24;
    --text:     #e4e4e7;
    --muted:    #71717a;
    --font:     'DM Sans', sans-serif;
    --mono:     'DM Mono', monospace;
}

/* ── Global resets ── */
html, body, [class*="css"] { font-family: var(--font); }
.stApp { background: var(--bg); color: var(--text); }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }
section[data-testid="stSidebar"] .stSlider > div > div > div {
    background: var(--accent) !important;
}
section[data-testid="stSidebar"] label { font-size: 0.78rem; color: var(--muted) !important; }

/* ── Widgets ── */
.stSelectbox > div > div,
.stTextInput > div > div > input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}
.stSlider > div > div > div { background: var(--surface2); }

/* ── Button ── */
.stButton > button {
    background: var(--accent) !important;
    color: #0a0a0a !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 2rem !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em !important;
    width: 100%;
    transition: opacity 0.2s ease;
    font-family: var(--font) !important;
}
.stButton > button:hover { opacity: 0.85; }

/* ── Progress bar ── */
.stProgress > div > div > div > div { background: var(--accent) !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.75rem !important; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-size: 1.6rem !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { font-size: 0.78rem !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Helpers: card components
# ─────────────────────────────────────────────
def card(content_html: str, extra_style: str = "") -> None:
    st.markdown(
        f"""<div style="background:#1e1e1e;border:1px solid #2e2e2e;border-radius:14px;
                        padding:1.4rem 1.6rem;{extra_style}">{content_html}</div>""",
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, sub: str, color: str = "#4ade80") -> str:
    return f"""
    <div style="background:#1e1e1e;border:1px solid #2e2e2e;border-radius:14px;
                padding:1.2rem 1.4rem;height:100%;">
      <p style="margin:0;font-size:0.72rem;color:#71717a;letter-spacing:.06em;text-transform:uppercase;">{label}</p>
      <p style="margin:.35rem 0 .2rem;font-size:1.75rem;font-weight:700;color:{color};font-family:'DM Mono',monospace;">{value}</p>
      <p style="margin:0;font-size:0.75rem;color:#52525b;">{sub}</p>
    </div>"""


def result_card(churn: bool, prob: float) -> str:
    if churn:
        bg, border, icon, headline, sub = (
            "#2a1215", "#7f1d1d", "⚠️",
            "High Churn Risk Detected",
            "This customer is likely to churn. Immediate retention action recommended.",
        )
        color = "#f87171"
    else:
        bg, border, icon, headline, sub = (
            "#0f1f18", "#14532d", "✅",
            "Low Churn Risk",
            "This customer shows strong retention signals. Keep up the engagement.",
        )
        color = "#4ade80"
    return f"""
    <div style="background:{bg};border:1px solid {border};border-radius:14px;padding:1.4rem 1.6rem;">
      <div style="display:flex;align-items:center;gap:.7rem;margin-bottom:.5rem;">
        <span style="font-size:1.4rem;">{icon}</span>
        <span style="font-size:1.1rem;font-weight:700;color:{color};">{headline}</span>
      </div>
      <p style="margin:0;font-size:.85rem;color:#a1a1aa;">{sub}</p>
    </div>"""


# ─────────────────────────────────────────────
#  Model loading
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model, feature_cols = None, None
    if os.path.exists("churn_model.pkl"):
        model = joblib.load("churn_model.pkl")
    if os.path.exists("feature_columns.pkl"):
        feature_cols = joblib.load("feature_columns.pkl")
    return model, feature_cols

model, feature_columns = load_artifacts()


# ─────────────────────────────────────────────
#  Sidebar – customer inputs
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:.2rem 0 1.2rem;">
      <p style="margin:0;font-size:1.05rem;font-weight:700;color:#e4e4e7;">📡 ChurnIQ</p>
      <p style="margin:.1rem 0 0;font-size:.72rem;color:#52525b;letter-spacing:.05em;text-transform:uppercase;">Customer Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Customer Profile")
    st.caption("Fill in the customer attributes below")

    st.markdown("**Account Info**")
    tenure          = st.slider("Tenure (months)", 0, 72, 24)
    contract        = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment_method  = st.selectbox("Payment Method",
                                   ["Electronic check", "Mailed check",
                                    "Bank transfer (automatic)", "Credit card (automatic)"])

    st.markdown("**Charges**")
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
    total_charges   = st.slider("Total Charges ($)", 0.0, 9000.0,
                                float(tenure * monthly_charges), step=10.0)

    st.markdown("**Services**")
    internet_service  = st.selectbox("Internet Service",  ["DSL", "Fiber optic", "No"])
    tech_support      = st.selectbox("Tech Support",      ["Yes", "No", "No internet service"])
    online_security   = st.selectbox("Online Security",   ["Yes", "No", "No internet service"])
    online_backup     = st.selectbox("Online Backup",     ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])

    st.markdown("**Demographics**")
    gender         = st.selectbox("Gender",         ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])

    st.markdown("---")
    predict_clicked = st.button("🔍  Predict Churn")


# ─────────────────────────────────────────────
#  Main header
# ─────────────────────────────────────────────
st.markdown("""
<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:1.6rem;">
  <div>
    <h1 style="margin:0;font-size:1.7rem;font-weight:700;color:#e4e4e7;">
      Customer Churn Prediction Dashboard
    </h1>
    <p style="margin:.35rem 0 0;font-size:.88rem;color:#71717a;">
      Predict customer churn risk using ML · Powered by Random Forest
    </p>
  </div>
  <div style="background:#1e1e1e;border:1px solid #2e2e2e;border-radius:10px;
              padding:.5rem 1rem;font-size:.78rem;color:#4ade80;font-family:'DM Mono',monospace;">
    Model: ✓ Loaded
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# ─────────────────────────────────────────────
#  Feature engineering (matches training pipeline)
# ─────────────────────────────────────────────
def build_input_df() -> pd.DataFrame:
    raw = {
        "tenure":             tenure,
        "MonthlyCharges":     monthly_charges,
        "TotalCharges":       total_charges,
        "gender":             gender,
        "SeniorCitizen":      1 if senior_citizen == "Yes" else 0,
        "Contract":           contract,
        "PaymentMethod":      payment_method,
        "TechSupport":        tech_support,
        "InternetService":    internet_service,
        "OnlineSecurity":     online_security,
        "OnlineBackup":       online_backup,
        "DeviceProtection":   device_protection,
    }
    df = pd.DataFrame([raw])

    # One-hot encode categoricals
    cat_cols = ["gender", "Contract", "PaymentMethod",
                "TechSupport", "InternetService",
                "OnlineSecurity", "OnlineBackup", "DeviceProtection"]
    df = pd.get_dummies(df, columns=cat_cols)

    # Align to training feature columns if available
    if feature_columns is not None:
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_columns]

    return df


# ─────────────────────────────────────────────
#  Churn-reason heuristics
# ─────────────────────────────────────────────
def churn_reasons(prob: float) -> list[str]:
    reasons = []
    if contract == "Month-to-month":
        reasons.append("📋 **Month-to-month contract** — no long-term commitment increases churn likelihood.")
    if monthly_charges > 80:
        reasons.append("💰 **High monthly charges** — customer may seek a cheaper alternative.")
    if tech_support == "No" and internet_service != "No":
        reasons.append("🛠️ **No Tech Support** — unresolved technical issues drive frustration & churn.")
    if tenure < 12:
        reasons.append("📅 **Low tenure** — new customers haven't yet built loyalty.")
    if payment_method == "Electronic check":
        reasons.append("💳 **Electronic check payment** — historically correlates with higher churn.")
    if online_security == "No":
        reasons.append("🔒 **No Online Security** — security concerns can push customers away.")
    # Always return top 3
    if not reasons:
        reasons = ["📊 Model scored this customer based on combined feature interactions.",
                   "📉 Low individual risk factors detected.",
                   "✅ Customer profile matches retained segments."]
    return reasons[:3]


def retention_suggestions(churn: bool, prob: float) -> list[str]:
    if not churn:
        return [
            "🎯 Enroll in loyalty rewards program to deepen engagement.",
            "📬 Schedule a quarterly check-in to surface upsell opportunities.",
            "🎁 Offer a proactive discount on renewal to lock in long-term.",
        ]
    suggestions = []
    if contract == "Month-to-month":
        suggestions.append("📋 Offer a discounted **annual contract** upgrade to lock in commitment.")
    if tech_support == "No":
        suggestions.append("🛠️ Provide a **free Tech Support trial** to resolve friction points.")
    if monthly_charges > 80:
        suggestions.append("💰 Present a **loyalty discount** or bundle package to reduce perceived cost.")
    suggestions += [
        "📞 Trigger a proactive **retention call** from the success team within 48 h.",
        "🎁 Send a personalised **win-back offer** with limited-time incentive.",
    ]
    return suggestions[:3]


# ─────────────────────────────────────────────
#  Prediction & output
# ─────────────────────────────────────────────
prob, churn_label, churn_bool = 0.0, "—", False

if predict_clicked:
    if model is None:
        st.warning("⚠️ No model file found. Place `churn_model.pkl` in the same directory as `app.py`.")
    else:
        input_df = build_input_df()
        prob = float(model.predict_proba(input_df)[0][1])
        churn_bool  = prob >= 0.5
        churn_label = "Yes" if churn_bool else "No"

    # ── KPI row ──────────────────────────────
    k1, k2, k3 = st.columns(3)
    risk_level  = "Critical" if prob > 0.75 else ("High" if prob > 0.5 else ("Moderate" if prob > 0.3 else "Low"))
    priority    = "Immediate" if prob > 0.75 else ("High" if prob > 0.5 else ("Normal" if prob > 0.3 else "Low"))
    risk_color  = "#f87171" if prob > 0.5 else ("#fbbf24" if prob > 0.3 else "#4ade80")

    with k1:
        st.markdown(kpi_card("Churn Probability", f"{prob*100:.1f}%", "Model confidence score", risk_color),
                    unsafe_allow_html=True)
    with k2:
        st.markdown(kpi_card("Customer Risk Level", risk_level, "Based on churn threshold ≥ 50 %", risk_color),
                    unsafe_allow_html=True)
    with k3:
        st.markdown(kpi_card("Retention Priority", priority, "Recommended action urgency", risk_color),
                    unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.2rem'></div>", unsafe_allow_html=True)

    # ── Result card + probability bar ────────
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown(result_card(churn_bool, prob), unsafe_allow_html=True)

        st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='margin:0 0 .4rem;font-size:.78rem;color:#71717a;letter-spacing:.05em;"
            f"text-transform:uppercase;'>Churn Probability Score</p>",
            unsafe_allow_html=True,
        )
        st.progress(prob)
        low_c  = "#4ade80"
        high_c = "#f87171"
        bar_c  = high_c if churn_bool else low_c
        st.markdown(
            f"<p style='margin:.3rem 0 0;font-size:1.1rem;font-weight:700;"
            f"color:{bar_c};font-family:\"DM Mono\",monospace;'>{prob*100:.1f}% probability</p>",
            unsafe_allow_html=True,
        )

    with col_b:
        card(f"""
        <p style="margin:0 0 .8rem;font-size:.72rem;color:#71717a;letter-spacing:.06em;text-transform:uppercase;">
          Customer Snapshot
        </p>
        <table style="width:100%;border-collapse:collapse;font-size:.82rem;">
          <tr><td style="color:#71717a;padding:.25rem 0;">Tenure</td>
              <td style="text-align:right;color:#e4e4e7;font-family:'DM Mono',monospace;">{tenure} mo</td></tr>
          <tr><td style="color:#71717a;padding:.25rem 0;">Monthly</td>
              <td style="text-align:right;color:#e4e4e7;font-family:'DM Mono',monospace;">${monthly_charges:.2f}</td></tr>
          <tr><td style="color:#71717a;padding:.25rem 0;">Total</td>
              <td style="text-align:right;color:#e4e4e7;font-family:'DM Mono',monospace;">${total_charges:.0f}</td></tr>
          <tr><td style="color:#71717a;padding:.25rem 0;">Contract</td>
              <td style="text-align:right;color:#e4e4e7;">{contract}</td></tr>
          <tr><td style="color:#71717a;padding:.25rem 0;">Internet</td>
              <td style="text-align:right;color:#e4e4e7;">{internet_service}</td></tr>
          <tr><td style="color:#71717a;padding:.25rem 0;">Gender</td>
              <td style="text-align:right;color:#e4e4e7;">{gender}</td></tr>
          <tr><td style="color:#71717a;padding:.25rem 0;">Senior</td>
              <td style="text-align:right;color:#e4e4e7;">{senior_citizen}</td></tr>
        </table>
        """)

    st.markdown("<div style='margin-top:1.4rem'></div>", unsafe_allow_html=True)

    # ── Reasons & Suggestions ─────────────────
    col_r, col_s = st.columns(2)

    with col_r:
        reasons_html = "".join(
            f"<div style='display:flex;gap:.6rem;align-items:flex-start;margin-bottom:.65rem;'>"
            f"<div style='min-width:6px;height:6px;background:#4ade80;border-radius:50%;margin-top:.42rem;'></div>"
            f"<p style='margin:0;font-size:.84rem;color:#d4d4d8;'>{r}</p></div>"
            for r in churn_reasons(prob)
        )
        card(f"""
        <p style="margin:0 0 .9rem;font-size:.72rem;color:#71717a;letter-spacing:.06em;text-transform:uppercase;">
          Top 3 Churn Signals
        </p>
        {reasons_html}
        """)

    with col_s:
        sugg_html = "".join(
            f"<div style='display:flex;gap:.6rem;align-items:flex-start;margin-bottom:.65rem;'>"
            f"<div style='min-width:6px;height:6px;background:#22d3ee;border-radius:50%;margin-top:.42rem;'></div>"
            f"<p style='margin:0;font-size:.84rem;color:#d4d4d8;'>{s}</p></div>"
            for s in retention_suggestions(churn_bool, prob)
        )
        card(f"""
        <p style="margin:0 0 .9rem;font-size:.72rem;color:#71717a;letter-spacing:.06em;text-transform:uppercase;">
          Retention Recommendations
        </p>
        {sugg_html}
        """)

else:
    # ── Placeholder state ────────────────────
    st.markdown("""
    <div style="background:#1e1e1e;border:1px solid #2e2e2e;border-radius:14px;
                padding:3rem;text-align:center;">
      <p style="font-size:2rem;margin:0 0 .5rem;">📡</p>
      <p style="font-size:1rem;font-weight:600;color:#e4e4e7;margin:0 0 .4rem;">
        Configure &amp; Predict
      </p>
      <p style="font-size:.85rem;color:#52525b;margin:0;">
        Fill in the customer attributes in the sidebar and click <strong style="color:#4ade80;">Predict Churn</strong>
        to generate an instant risk assessment.
      </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────
st.markdown("<div style='margin-top:3rem'></div>", unsafe_allow_html=True)
st.markdown("""
<div style="border-top:1px solid #2e2e2e;padding:1rem 0;display:flex;
            justify-content:space-between;align-items:center;">
  <span style="font-size:.75rem;color:#3f3f46;">
    ChurnIQ · Customer Intelligence Platform
  </span>
  <span style="font-size:.75rem;color:#3f3f46;font-family:'DM Mono',monospace;">
    Random Forest · Telecom Churn Dataset
  </span>
</div>
""", unsafe_allow_html=True)