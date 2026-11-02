from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from utils.recommendations import FEATURE_NAME_MAP, RECOMMENDATION_MAP


BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "xgboost.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")
explainer = shap.Explainer(model)

DISPLAY_FEATURE_NAME_MAP = {
    **FEATURE_NAME_MAP,
    "weight_lb": "Weight (kg)",
}
CLASSIFICATION_THRESHOLD = 0.50

st.set_page_config(
    page_title="CardioCare AI | Cardiovascular Assessment",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --cc-navy: #123247;
        --cc-blue: #176b87;
        --cc-teal: #159b98;
        --cc-teal-dark: #0d7778;
        --cc-ink: #203747;
        --cc-muted: #637887;
        --cc-line: #dce8ed;
        --cc-surface: #ffffff;
        --cc-canvas: #f3f8fa;
    }

    .stApp,
    [data-testid="stAppViewContainer"] {
        background: var(--cc-canvas);
        color: var(--cc-ink);
    }
    html, body, [class*="css"] {
        font-family: "Inter", "Segoe UI", -apple-system, BlinkMacSystemFont, sans-serif;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMainBlockContainer"] {
        max-width: 1360px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fbfc 0%, #eef5f7 100%);
        border-right: 1px solid var(--cc-line);
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.3rem; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--cc-muted);
    }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 { color: var(--cc-navy); }

    .cc-brand {
        display: flex;
        align-items: center;
        gap: .72rem;
        margin: 0 0 1.5rem;
        color: var(--cc-navy);
        font-size: 1.12rem;
        font-weight: 750;
        letter-spacing: -.025em;
    }
    .cc-brand-mark {
        display: grid;
        width: 2.5rem;
        height: 2.5rem;
        place-items: center;
        border-radius: .85rem;
        background: linear-gradient(145deg, var(--cc-blue), var(--cc-teal));
        box-shadow: 0 7px 18px rgba(21, 155, 152, .2);
        color: white;
        font-size: 1.2rem;
    }
    .cc-brand small {
        display: block;
        margin-top: .1rem;
        color: var(--cc-muted);
        font-size: .7rem;
        font-weight: 550;
        letter-spacing: .08em;
        text-transform: uppercase;
    }
    .cc-hero { margin: .1rem 0 1.65rem; }
    .cc-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: .4rem;
        margin-bottom: .75rem;
        color: var(--cc-teal-dark);
        font-size: .76rem;
        font-weight: 750;
        letter-spacing: .12em;
        text-transform: uppercase;
    }
    .cc-hero h1 {
        margin: 0;
        color: var(--cc-navy);
        font-size: clamp(2rem, 4vw, 3.15rem);
        font-weight: 750;
        letter-spacing: -.055em;
        line-height: 1.08;
    }
    .cc-hero p {
        max-width: 760px;
        margin: .8rem 0 0;
        color: var(--cc-muted);
        font-size: 1rem;
        line-height: 1.65;
    }
    .cc-panel-title {
        margin: .35rem 0 .85rem;
        color: var(--cc-navy);
        font-size: 1.35rem;
        font-weight: 720;
        letter-spacing: -.025em;
    }
    .cc-panel-copy {
        margin: -.4rem 0 1rem;
        color: var(--cc-muted);
        font-size: .92rem;
        line-height: 1.5;
    }
    .cc-section-heading {
        margin: 1.8rem 0 .2rem;
        color: var(--cc-navy);
        font-size: 1.28rem;
        font-weight: 720;
        letter-spacing: -.025em;
    }
    .cc-section-copy {
        margin: 0 0 .9rem;
        color: var(--cc-muted);
        font-size: .9rem;
    }
    .cc-result-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin: .2rem 0 1.1rem;
    }
    .cc-summary-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(125px, 1fr));
        gap: .85rem 1rem;
    }
    .cc-summary-item {
        min-width: 0;
        padding: .2rem .25rem;
    }
    .cc-summary-label {
        display: block;
        margin-bottom: .25rem;
        color: var(--cc-muted);
        font-size: .74rem;
        font-weight: 700;
        letter-spacing: .06em;
        text-transform: uppercase;
    }
    .cc-summary-value {
        display: block;
        color: var(--cc-navy);
        font-size: 1.03rem;
        font-weight: 720;
        line-height: 1.35;
        overflow-wrap: anywhere;
    }
    .cc-result-card {
        min-height: 142px;
        padding: 1.25rem 1.4rem;
        border: 1px solid var(--cc-line);
        border-radius: 1.1rem;
        background: var(--cc-surface);
        box-shadow: 0 9px 28px rgba(21, 66, 84, .055);
    }
    .cc-result-card.risk-high {
        border-top: 4px solid #d68157;
    }
    .cc-result-card.risk-low {
        border-top: 4px solid var(--cc-teal);
    }
    .cc-card-label {
        color: var(--cc-muted);
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .09em;
        text-transform: uppercase;
    }
    .cc-risk-value {
        margin: .55rem 0 .3rem;
        color: var(--cc-navy);
        font-size: 1.55rem;
        font-weight: 760;
        letter-spacing: -.035em;
    }
    .cc-probability-value {
        margin: .35rem 0 .15rem;
        color: var(--cc-teal-dark);
        font-size: 2.1rem;
        font-weight: 760;
        letter-spacing: -.05em;
    }
    .cc-card-note {
        color: var(--cc-muted);
        font-size: .82rem;
        line-height: 1.4;
    }
    .cc-progress-track {
        height: .42rem;
        margin-top: .85rem;
        overflow: hidden;
        border-radius: 999px;
        background: #e9f0f3;
    }
    .cc-progress-fill {
        height: 100%;
        border-radius: inherit;
        background: linear-gradient(90deg, var(--cc-blue), var(--cc-teal));
    }
    .cc-driver-index {
        color: var(--cc-teal-dark);
        font-size: .7rem;
        font-weight: 750;
        letter-spacing: .1em;
        text-transform: uppercase;
    }
    .cc-impact {
        display: inline-block;
        margin: .25rem 0 .7rem;
        padding: .28rem .55rem;
        border-radius: 999px;
        background: #e8f5f4;
        color: var(--cc-teal-dark);
        font-size: .78rem;
        font-weight: 700;
    }
    .cc-impact-high {
        background: #fff1e9;
        color: #a95130;
    }
    .cc-chart-card {
        height: 100%;
        padding: .95rem 1rem .5rem;
        border: 1px solid var(--cc-line);
        border-radius: 1rem;
        background: var(--cc-surface);
        box-shadow: 0 8px 25px rgba(21, 66, 84, .045);
    }
    .cc-chart-title {
        margin: .1rem 0 .25rem;
        color: var(--cc-navy);
        font-size: 1rem;
        font-weight: 700;
    }
    .cc-chart-caption {
        margin: 0 0 .4rem;
        color: var(--cc-muted);
        font-size: .8rem;
    }

    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input,
    [data-baseweb="select"] > div {
        min-height: 2.8rem;
        border-color: #d4e2e8;
        border-radius: .72rem;
        background-color: white;
        color: var(--cc-ink);
    }
    [data-testid="stNumberInput"] input:focus,
    [data-testid="stTextInput"] input:focus {
        border-color: var(--cc-teal);
        box-shadow: 0 0 0 2px rgba(21, 155, 152, .12);
    }
    [data-testid="stWidgetLabel"] p {
        color: #344f60;
        font-size: .84rem;
        font-weight: 650;
    }
    [data-testid="stRadio"] [role="radiogroup"] { gap: .4rem; }
    [data-testid="stRadio"] label { color: #344f60; }

    .stButton > button {
        min-height: 3rem;
        border: 0;
        border-radius: .78rem;
        background: linear-gradient(105deg, var(--cc-blue), var(--cc-teal));
        box-shadow: 0 8px 18px rgba(21, 115, 132, .19);
        color: white;
        font-weight: 720;
        transition: transform .16s ease, box-shadow .16s ease, filter .16s ease;
    }
    .stButton > button:hover {
        border: 0;
        box-shadow: 0 11px 22px rgba(21, 115, 132, .25);
        color: white;
        filter: brightness(1.04);
        transform: translateY(-1px);
    }
    .stButton > button:focus-visible {
        outline: 3px solid rgba(21, 155, 152, .35);
        outline-offset: 2px;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--cc-line);
        border-radius: 1rem;
        background: white;
        box-shadow: 0 7px 22px rgba(21, 66, 84, .04);
    }
    [data-testid="stAlert"] { border-radius: .85rem; }
    @media (max-width: 760px) {
        [data-testid="stMainBlockContainer"] {
            padding: 1.15rem 1rem 2rem;
        }
        .cc-hero { margin-bottom: 1.2rem; }
        .cc-result-grid { grid-template-columns: 1fr; gap: .7rem; }
        .cc-result-card { min-height: auto; padding: 1rem 1.1rem; }
        .cc-section-heading { margin-top: 1.3rem; }
        [data-testid="stSidebar"] { min-width: 17rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """
    <div class="cc-brand">
        <span class="cc-brand-mark">♥</span>
        <span>CardioCare AI<small>Personal health insights</small></span>
    </div>
    <div class="cc-panel-title">Patient details</div>
    <div class="cc-panel-copy">Enter the measurements below to create a cardiovascular assessment.</div>
    """,
    unsafe_allow_html=True,
)

input_data = {
    "age_years": st.sidebar.number_input(
        "Age (years)", min_value=18, max_value=100, value=50
    ),
    "height": st.sidebar.number_input(
        "Height (cm)", min_value=120, max_value=240, value=170
    ),
    "weight_kg": st.sidebar.number_input(
        "Weight (kg)", min_value=30, max_value=180, value=70
    ),
    "ap_hi": st.sidebar.number_input(
        "Systolic blood pressure (mmHg)", min_value=80, max_value=250, value=120
    ),
    "ap_lo": st.sidebar.number_input(
        "Diastolic blood pressure (mmHg)", min_value=50, max_value=150, value=80
    ),
    "cholesterol": st.sidebar.selectbox(
        "Cholesterol level",
        [1, 2, 3],
        format_func=lambda x: [
            "Normal (<200 mg/dL)",
            "Elevated (200–239 mg/dL)",
            "High (>240 mg/dL)",
        ][x - 1],
    ),
    "gluc": st.sidebar.selectbox(
        "Glucose level",
        [1, 2, 3],
        format_func=lambda x: [
            "Normal (<100 mg/dL)",
            "Elevated (100–125 mg/dL)",
            "High (>126 mg/dL)",
        ][x - 1],
    ),
    "smoke": st.sidebar.radio(
        "Do you smoke?", [0, 1], format_func=lambda x: "Yes" if x else "No"
    ),
    "alco": st.sidebar.radio(
        "Do you consume alcohol?", [0, 1], format_func=lambda x: "Yes" if x else "No"
    ),
    "active": st.sidebar.radio(
        "Are you regularly physically active?",
        [0, 1],
        format_func=lambda x: "Yes" if x else "No",
    ),
}

patient_gender = st.sidebar.selectbox(
    "Gender (optional, summary only)",
    ["Prefer not to say", "Female", "Male"],
)
st.sidebar.caption("Measurements are used for this assessment only.")
st.sidebar.caption("Gender is shown in the summary and is not used by the model.")

weight_kg = input_data.pop("weight_kg")
height_m = input_data["height"] / 100
input_data["bmi"] = weight_kg / (height_m**2)
input_data["weight_lb"] = weight_kg * 2.20462

input_df = pd.DataFrame([input_data])
expected_features = model.get_booster().feature_names
input_df = input_df[expected_features]
model_input = pd.DataFrame(
    scaler.transform(input_df), columns=expected_features, index=input_df.index
)
input_df_display = model_input.rename(columns=DISPLAY_FEATURE_NAME_MAP)

st.markdown(
    """
    <div class="cc-hero">
        <div class="cc-eyebrow">🫀 CardioCare AI · Cardiovascular health</div>
        <h1>A clearer view of your heart health.</h1>
        <p>Review a personalized model assessment and the factors that influenced it. Use these insights to support a conversation with a qualified healthcare professional.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.button("Run cardiovascular assessment", use_container_width=True):
    prediction = model.predict(model_input)[0]
    probability = model.predict_proba(model_input)[0][1]
    result_label = "CVD-positive prediction" if prediction == 1 else "CVD-negative prediction"
    result_class = "risk-high" if prediction == 1 else "risk-low"
    result_note = (
        f"At or above the {CLASSIFICATION_THRESHOLD:.2f} classification threshold."
        if prediction == 1
        else f"Below the {CLASSIFICATION_THRESHOLD:.2f} classification threshold."
    )

    st.markdown(
        f"""
        <div class="cc-result-grid">
            <div class="cc-result-card {result_class}">
                <div class="cc-card-label">Assessment result</div>
                <div class="cc-risk-value">{result_label}</div>
                <div class="cc-card-note">{result_note}</div>
            </div>
            <div class="cc-result-card">
                <div class="cc-card-label">Probability of the dataset label</div>
                <div class="cc-probability-value">{probability:.1%}</div>
                <div class="cc-card-note">Classification threshold: {CLASSIFICATION_THRESHOLD:.2f}. A model prediction, not a diagnosis.</div>
                <div class="cc-progress-track" role="progressbar" aria-valuenow="{probability * 100:.1f}" aria-valuemin="0" aria-valuemax="100">
                    <div class="cc-progress-fill" style="width: {probability * 100:.1f}%"></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="cc-section-heading">Patient assessment summary</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="cc-section-copy">These are the submitted values used for this assessment.</div>',
        unsafe_allow_html=True,
    )
    summary_items = [
        ("Age", f"{input_data['age_years']} years"),
        ("Gender", patient_gender),
        ("Height", f"{input_data['height']} cm"),
        ("Weight", f"{weight_kg:g} kg"),
        ("BMI", f"{input_data['bmi']:.1f}"),
        ("Blood pressure", f"{input_data['ap_hi']}/{input_data['ap_lo']} mmHg"),
    ]
    summary_html = '<div class="cc-summary-grid">' + "".join(
        f'<div class="cc-summary-item"><span class="cc-summary-label">{label}</span>'
        f'<span class="cc-summary-value">{value}</span></div>'
        for label, value in summary_items
    ) + "</div>"
    with st.container(border=True):
        st.markdown(summary_html, unsafe_allow_html=True)

    shap_values = explainer(model_input)
    shap_values.feature_names = [
        DISPLAY_FEATURE_NAME_MAP.get(feature, feature) for feature in expected_features
    ]

    st.markdown('<div class="cc-section-heading">Prediction explanation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="cc-section-copy">Explore the strongest factors behind this individual model score.</div>',
        unsafe_allow_html=True,
    )
    chart_summary, chart_waterfall = st.columns(2, gap="medium")

    with chart_summary:
        with st.container(border=True):
            st.markdown('<div class="cc-chart-title">Top feature impact</div>', unsafe_allow_html=True)
            st.markdown('<div class="cc-chart-caption">Per-patient feature impact; this is not a population-wide ranking.</div>', unsafe_allow_html=True)
            fig_summary, _ = plt.subplots(figsize=(8, 5))
            shap.summary_plot(
                shap_values,
                input_df_display,
                plot_type="bar",
                color="#159b98",
                max_display=8,
                show=False,
            )
            plt.tight_layout()
            st.pyplot(fig_summary, width="stretch")
            plt.close(fig_summary)

    with chart_waterfall:
        with st.container(border=True):
            st.markdown('<div class="cc-chart-title">How features move the score</div>', unsafe_allow_html=True)
            st.markdown('<div class="cc-chart-caption">Values are log-odds, not percentages. A SHAP value of −0.63 is not a 63% probability decrease.</div>', unsafe_allow_html=True)
            plt.figure(figsize=(8, 5))
            shap.plots.waterfall(shap_values[0], max_display=8, show=False)
            fig_waterfall = plt.gcf()
            fig_waterfall.tight_layout()
            st.pyplot(fig_waterfall, width="stretch")
            plt.close(fig_waterfall)

    st.markdown('<div class="cc-section-heading">Risk breakdown</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="cc-section-copy">Treatment Recommendations and Lifestyle Guidance are shown for the most influential factors.</div>',
        unsafe_allow_html=True,
    )

    shap_dict = dict(zip(input_df.columns, shap_values[0].values))
    sorted_features = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    recommendation_columns = st.columns(2, gap="medium")

    category_labels = {
        "cholesterol": ["Normal", "Elevated", "High"],
        "gluc": ["Normal", "Elevated", "High"],
        "smoke": ["No", "Yes"],
        "alco": ["No", "Yes"],
        "active": ["No", "Yes"],
    }

    for index, (feature, value) in enumerate(sorted_features[:3], start=1):
        display_name = DISPLAY_FEATURE_NAME_MAP.get(feature, feature)
        recommendation_name = FEATURE_NAME_MAP.get(feature, feature)
        rec = RECOMMENDATION_MAP.get(recommendation_name, {})
        raw_value = input_df[feature].iloc[0]
        if feature == "weight_lb":
            patient_value = f"{raw_value / 2.20462:.1f} kg"
        elif feature == "bmi":
            patient_value = f"{raw_value:.1f}"
        elif feature in category_labels:
            patient_value = category_labels[feature][int(raw_value) - (1 if feature in ("cholesterol", "gluc") else 0)]
        else:
            patient_value = f"{raw_value:g}"

        impact_direction = "raises" if value > 0 else "lowers"
        impact_class = "cc-impact cc-impact-high" if value > 0 else "cc-impact"

        with recommendation_columns[(index - 1) % 2]:
            with st.container(border=True):
                st.markdown(f'<div class="cc-driver-index">Key factor {index}</div>', unsafe_allow_html=True)
                st.markdown(f"#### {display_name}")
                st.markdown(
                    f'<span class="{impact_class}">{impact_direction.title()} model score · SHAP {value:+.3f}</span>',
                    unsafe_allow_html=True,
                )
                st.caption(f"Patient input: {patient_value}")
                if feature in {"ap_hi", "ap_lo"}:
                    st.markdown(
                        "**Clinical context** WHO defines hypertension in adults using systolic readings of at least 140 mmHg and/or diastolic readings of at least 90 mmHg on measurements taken on two different days. One input here cannot diagnose or rule out hypertension. [WHO guidance](https://www.who.int/india/health-topics/hypertension)."
                    )
                if feature == "age_years":
                    st.markdown(
                        "**Model interpretation** Age is not modifiable. This SHAP value shows how age moved the model score relative to its baseline; it does not measure how well anyone is managing their health."
                    )
                    if value > 0:
                        st.markdown(f"**Treatment Recommendations**  \n{rec.get('treatment_advice', 'N/A')}")
                        st.markdown(f"**Lifestyle Guidance**  \n{rec.get('lifestyle_guidance', 'N/A')}")
                        st.markdown(f"**Target range**  \n{rec.get('target', 'N/A')}")
                    else:
                        st.markdown("**Treatment Recommendations** This model contribution alone does not determine whether treatment is needed.")
                        st.markdown("**Lifestyle Guidance** Focus on modifiable health factors and routine, clinician-guided preventive care.")
                elif value > 0:
                    st.markdown(f"**Why it matters**  \n{rec.get('risk_reason', 'N/A')}")
                    st.markdown(f"**Treatment Recommendations**  \n{rec.get('treatment_advice', 'N/A')}")
                    st.markdown(f"**Lifestyle Guidance**  \n{rec.get('lifestyle_guidance', 'N/A')}")
                    st.markdown(f"**Target range**  \n{rec.get('target', 'N/A')}")
                elif feature in {"ap_hi", "ap_lo"}:
                    st.markdown("**Model interpretation** This feature lowered the model score; that attribution is not a clinical classification of this blood pressure reading.")
                    st.markdown("**Lifestyle Guidance** If you are concerned about your blood pressure, discuss properly repeated measurements with a healthcare professional.")
                else:
                    st.markdown(f"**Lifestyle Guidance**  \n{rec.get('no_risk_recommendation', 'Looks good!')}")

    st.caption("Model outputs are estimates based on the training dataset and are not a diagnosis or a substitute for medical care.")
