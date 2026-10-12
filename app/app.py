
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
REFERENCE_YEAR = 2021

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------- CUSTOM DARK DASHBOARD THEME ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #080d17;
    --panel: #0c1422;
    --field: #101b2c;
    --border: #1d2c43;
    --blue: #2875ff;
    --text: #f1f5ff;
    --muted: #a8badb;
    --green: #48e0bd;
}

.stApp {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

header[data-testid="stHeader"] {
    background: rgba(8, 13, 23, 0.95);
}

#MainMenu, footer {
    visibility: hidden;
}

.block-container {
    max-width: 1500px;
    padding: 1.5rem 2rem 3rem;
}

h1, h2, h3, p, label, span {
    font-family: 'Inter', sans-serif;
}

h1 {
    color: var(--text) !important;
    font-weight: 800 !important;
    letter-spacing: -1px;
}

h2, h3 {
    color: var(--text) !important;
}

p, .stCaption {
    color: var(--muted);
}

/* Header */
.hero {
    display: flex;
    align-items: center;
    gap: 20px;
    padding: 5px 5px 24px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 22px;
}

.hero-icon {
    font-size: 55px;
    line-height: 1;
}

.hero-subtitle {
    color: #b6c8e9;
    font-size: 15px;
    margin-top: -4px;
}

/* Dashboard panels */
.panel {
    background: linear-gradient(145deg, #0d1625, #0a111d);
    border: 1px solid var(--border);
    border-radius: 13px;
    padding: 22px;
    margin-bottom: 20px;
}

.panel-heading {
    display: flex;
    align-items: center;
    gap: 12px;
    color: #f3f6ff;
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 20px;
}

.panel-icon {
    font-size: 22px;
    color: #4c94ff;
}

/* Input labels */
.stSelectbox label,
.stNumberInput label {
    color: #c4d3ee !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

/* Text and number inputs */
.stNumberInput input {
    background: var(--field) !important;
    color: white !important;
    border: 1px solid #344764 !important;
    border-radius: 9px !important;
    min-height: 42px;
}

.stSelectbox [data-baseweb="select"] > div {
    background: var(--field) !important;
    color: white !important;
    border: 1px solid #344764 !important;
    border-radius: 9px !important;
    min-height: 42px;
}

.stSelectbox [data-baseweb="select"] * {
    color: white;
}

.stNumberInput button {
    background: var(--field) !important;
    color: #a8c5ff !important;
    border-color: #344764 !important;
}

/* Primary button */
.stButton > button[kind="primary"],
.stButton > button {
    background: linear-gradient(100deg, #1767f8, #287aff) !important;
    color: white !important;
    border: none !important;
    border-radius: 9px !important;
    min-height: 45px;
    font-weight: 700 !important;
    transition: 0.2s ease;
}

.stButton > button:hover {
    background: #4389ff !important;
    box-shadow: 0 4px 20px #1767f844;
    transform: translateY(-1px);
}

/* Decorative car image */
.car-image-container {
    background: linear-gradient(145deg, #111f34, #0a1220);
    border: 1px solid #263a56;
    border-radius: 12px;
    padding: 12px;
    margin-bottom: 18px;
}

.car-image {
    width: 100%;
    height: 230px;
    object-fit: cover;
    border-radius: 9px;
    display: block;
}

.car-card-title {
    color: #f1f5ff;
    font-size: 19px;
    font-weight: 700;
    margin-top: 18px;
}

.car-card-description {
    color: #a8badb;
    font-size: 13px;
    line-height: 1.8;
    margin-top: 8px;
}

.feature-tag {
    display: inline-block;
    padding: 6px 10px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
    margin: 12px 5px 2px 0;
    background: #172b46;
    color: #8db8ff;
}

.feature-tag-green {
    background: #12352f;
    color: #48e0bd;
}

/* Result card */
.price-card {
    background: linear-gradient(115deg, #092522, #0b1922);
    border: 1px solid #11b99c;
    border-radius: 13px;
    padding: 24px 28px;
    min-height: 150px;
    margin-top: 2px;
}

.price-heading {
    font-size: 18px;
    font-weight: 700;
    color: #e8fff9;
    margin-bottom: 14px;
}

.price-value {
    color: var(--green);
    font-size: 36px;
    font-weight: 800;
    letter-spacing: -1px;
    margin-bottom: 8px;
}

.price-range {
    color: #d3e3fa;
    font-size: 14px;
    line-height: 1.9;
}

.info-card {
    background: linear-gradient(140deg, #111c30, #0d1523);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 22px;
    margin-top: 8px;
    color: #c8d6ee;
    font-size: 13px;
    line-height: 1.9;
}

.info-heading {
    color: #f2f6ff;
    font-weight: 700;
    margin-bottom: 8px;
}

div[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Widget spacing */
div[data-testid="stVerticalBlock"] {
    gap: 0.7rem;
}

hr {
    border-color: var(--border);
}

/* Mobile layout */
@media (max-width: 768px) {
    .block-container {
        padding: 1rem 1rem 2rem;
    }

    .hero {
        gap: 12px;
    }

    .hero-icon {
        font-size: 40px;
    }

    .hero-subtitle {
        font-size: 13px;
        line-height: 1.6;
    }

    .price-value {
        font-size: 29px;
    }

    .car-image {
        height: 190px;
    }
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_assets():
    model = joblib.load(ROOT / "models" / "model.joblib")

    meta = json.loads(
        (ROOT / "models" / "meta.json").read_text(
            encoding="utf-8"
        )
    )

    return model, meta


def lakh(value):
    return f"₹{value / 100000:.2f} Lakh"


try:
    model, meta = load_assets()

except Exception as exc:
    st.error(
        "Could not load the trained model or metadata. "
        "Check that models/model.joblib and models/meta.json exist."
    )
    st.exception(exc)
    st.stop()


defaults = meta["defaults"]



# ---------------- DASHBOARD HEADER ----------------
st.markdown("""
<div class="hero">
    <div class="hero-icon">🚘</div>
    <div>
        <h1 style="font-size:34px; margin:0;">
            Used Car Price Predictor
        </h1>
        <div class="hero-subtitle">
            Estimate the resale value of a used car in India,
            using a model trained on CarDekho listings.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)




# ---------------- MAIN DASHBOARD ----------------
details_col, image_col = st.columns(
    [2.05, 1.12],
    gap="medium"
)


# ---------------- CAR DETAILS ----------------
with details_col:

    st.markdown("""
    <div class="panel">
        <div class="panel-heading">
            <span class="panel-icon">⚙</span>
            Car Details
        </div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns(2, gap="medium")

    with left:

        brands = meta["brands"]

        preferred_brand = (
            "Maruti" if "Maruti" in brands else brands[0]
        )

        brand = st.selectbox(
            "Brand",
            brands,
            index=brands.index(preferred_brand),
            key="brand",
        )

        models = meta["models_by_brand"][brand]

        car_model = st.selectbox(
            "Model",
            models,
            index=models.index("Swift") if "Swift" in models else 0,
            key="car_model",
        )

        year = st.number_input(
            "Year of manufacture",
            min_value=1995,
            max_value=2020,
            value=2015,
            step=1,
        )

        km_driven = st.number_input(
            "Kilometres driven",
            min_value=0,
            max_value=500000,
            value=50000,
            step=1000,
        )

        fuel = st.selectbox(
            "Fuel",
            meta["fuels"],
        )

        transmission = st.selectbox(
            "Transmission",
            meta["transmissions"],
        )

        owner = st.selectbox(
            "Owner",
            meta["owners"],
        )

    with right:

        seller_type = st.selectbox(
            "Seller type",
            meta["seller_types"],
        )

        engine = st.number_input(
            "Engine (CC)",
            min_value=600,
            max_value=5000,
            value=int(defaults["engine"]),
            step=1,
        )

        max_power = st.number_input(
            "Max power (bhp)",
            min_value=30.0,
            max_value=600.0,
            value=float(defaults["max_power"]),
            step=1.0,
        )

        mileage = st.number_input(
            "Mileage (kmpl)",
            min_value=5.0,
            max_value=40.0,
            value=float(defaults["mileage"]),
            step=0.5,
        )

        torque_nm = st.number_input(
            "Torque (Nm)",
            min_value=40.0,
            max_value=800.0,
            value=float(defaults["torque_nm"]),
            step=1.0,
        )

        seats = st.number_input(
            "Seats",
            min_value=2,
            max_value=14,
            value=int(defaults["seats"]),
            step=1,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div style="color:#c4d3ee;font-size:13px;'
        'margin-bottom:5px;">'
        '₹ &nbsp; Optional: asking price (₹) to check if it is fair'
        '</div>',
        unsafe_allow_html=True,
    )

    asking = st.number_input(
        "Asking price",
        min_value=0,
        max_value=100_000_000,
        value=0,
        step=10000,
        label_visibility="collapsed",
    )

    predict_clicked = st.button(
        "✨  Predict Price",
        type="primary",
        use_container_width=True,
    )



# ---------------- DECORATIVE CAR PANEL ----------------
with image_col:

  
    st.markdown("""
    <div class="car-image-container">
        <img
            class="car-image"
            src="https://images.unsplash.com/photo-1494976388531-d1058494cdd8?auto=format&fit=crop&w=900&q=85"
            alt="Featured car"
        />

       
    <div class="car-card-title">
        Find Your Car's Value
    </div>

    <div class="car-card-description">
        Enter your vehicle specifications to estimate
        its resale value using machine learning.
    </div>

    <div>
        <span class="feature-tag">AI Powered</span>
        <span class="feature-tag feature-tag-green">
            Instant Estimate
        </span>
    </div>
  """, unsafe_allow_html=True)




    # About the prediction card
    st.markdown("""
    <div class="info-card">
        <div class="info-heading">
            ⓘ &nbsp; About this prediction
        </div>
        <p>
            The estimate uses the selected car specifications
            and a previously trained machine learning model.
        </p>
        <p>
            Condition, city, service history and accident history
            are not included.
        </p>
        <p style="color:#8ea5ca;">
            For demonstration purposes only.
        </p>
    </div>
    """, unsafe_allow_html=True)



# ---------------- PREDICTION ----------------
if predict_clicked:

    if year > REFERENCE_YEAR:
        st.error(
            f"Please select a manufacture year of "
            f"{REFERENCE_YEAR} or earlier. This must match "
            "the reference year used during model training."
        )
        st.stop()

    row = pd.DataFrame([{
        "km_driven": km_driven,
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "torque_nm": torque_nm,
        "seats": seats,
        "car_age": REFERENCE_YEAR - year,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
        "brand": brand,
        "model": car_model,
    }])

    try:

        price = float(model.predict(row)[0])

        # Assumes the model predicts the original price scale.
        if price <= 0:
            st.error(
                "The model returned an invalid price. "
                "Check your model."
            )
            st.stop()

        err = float(meta["median_error_pct"]) / 100

        low = max(0, price * (1 - err))
        high = price * (1 + err)

        st.markdown("---")

        result_col, note_col = st.columns(
            [2.05, 1.0],
            gap="medium"
        )

        with result_col:
# ---------------- PREDICTION RESULT CARD ----------------
            st.markdown(f"""

            <div class="price-card">
                <div class="price-heading">
                    📈 &nbsp; Estimated Price
                </div>

              
            
            <div class="price-value">
             {lakh(price)}
            </div>

            <div class="price-range">
                Typical range: {lakh(low)} – {lakh(high)}
                <br>
                Model's median error is about
                {meta['median_error_pct']:.1f}%.
            </div>
            """, unsafe_allow_html=True)



# ---------------- PREDICTION DETAILS CARD ----------------
        
        with note_col:
            st.markdown("""
            <div class="info-card">
                <div class="info-heading">
                    ⓘ &nbsp; Prediction details
                </div>
            <div>
                Based on the selected specifications and historical
                listings. Actual resale prices may vary with condition,
                location and demand.
            </div>
        </div>
    """, unsafe_allow_html=True)




        # ---------------- ASKING PRICE COMPARISON ----------------
        if asking > 0:

            diff = (asking - price) / price * 100

            if asking > high:

                st.warning(
                    f"The asking price is about {diff:.1f}% above "
                    "the estimate and above the typical range."
                )

            elif asking < low:

                st.info(
                    f"The asking price is about {abs(diff):.1f}% "
                    "below the estimate. Inspect the car's condition "
                    "and history before buying."
                )

            else:

                st.success(
                    "The asking price is within the model's "
                    "typical range."
                )

        st.caption(
            "For demonstration only. This prediction is not a "
            "professional valuation or a guaranteed selling price."
        )

    except Exception as exc:

        st.error(
            "Prediction failed. Please check the model "
            "and input features."
        )
        st.exception(exc)

