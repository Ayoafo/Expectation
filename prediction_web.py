import streamlit as st
import pandas as pd
import joblib
#
# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Prediction System",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------------------------------------
# LOAD TRAINED MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    return model

model = load_model()


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #F4F7FB;
}

/* Main title */
.main-title {
    font-size: 42px;
    font-weight: 700;
    color: #12355B;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    font-size: 18px;
    color: #52616B;
    margin-bottom: 25px;
}

/* Cards */
.card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

/* Information box */
.info-box {
    background-color: #E8F1FA;
    border-left: 5px solid #2471A3;
    padding: 15px;
    border-radius: 8px;
}

/* Positive result */
.success-box {
    background-color: #E8F8F0;
    border-left: 6px solid #28A745;
    padding: 20px;
    border-radius: 10px;
    font-size: 18px;
}

/* Warning result */
.warning-box {
    background-color: #FFF3E0;
    border-left: 6px solid #F39C12;
    padding: 20px;
    border-radius: 10px;
    font-size: 18px;
}

/* Button */
.stButton > button {
    width: 100%;
    border-radius: 8px;
    height: 3em;
    font-size: 18px;
    font-weight: bold;
}

/* Footer */
.footer {
    text-align: center;
    color: #777777;
    font-size: 13px;
    padding-top: 30px;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">AI Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Enter the required information below to generate an AI-assisted prediction.'
    '</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# SYSTEM INFORMATION
# ---------------------------------------------------------

st.markdown("""
<div class="info-box">
<b>How it works:</b><br>
Enter the required information → Review your inputs → 
Click <b>Generate Prediction</b> → View the AI prediction.
</div>
""", unsafe_allow_html=True)

st.write("")


# ---------------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------------

st.subheader("1. Enter Information")

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=40,
        help="Enter the patient's age."
    )

    bmi = st.number_input(
        "Body Mass Index (BMI)",
        min_value=10.0,
        max_value=60.0,
        value=25.0,
        step=0.1,
        help="Enter the patient's BMI."
    )

    glucose = st.number_input(
        "Glucose Level",
        min_value=0.0,
        max_value=300.0,
        value=100.0,
        help="Enter the measured glucose level."
    )


with col2:

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0.0,
        max_value=250.0,
        value=120.0,
        help="Enter the patient's blood pressure."
    )

    cholesterol = st.number_input(
        "Cholesterol Level",
        min_value=0.0,
        max_value=500.0,
        value=180.0,
        help="Enter the cholesterol level."
    )

    smoker = st.selectbox(
        "Smoking Status",
        ["No", "Yes"],
        help="Select whether the patient currently smokes."
    )


# ---------------------------------------------------------
# REVIEW INPUT
# ---------------------------------------------------------

st.divider()

st.subheader("2. Review Information")

smoker_value = 1 if smoker == "Yes" else 0

input_data = pd.DataFrame({
    "Age": [age],
    "BMI": [bmi],
    "Glucose": [glucose],
    "BloodPressure": [blood_pressure],
    "Cholesterol": [cholesterol],
    "Smoker": [smoker_value]
})

display_data = pd.DataFrame({
    "Age": [age],
    "BMI": [bmi],
    "Glucose": [glucose],
    "Blood Pressure": [blood_pressure],
    "Cholesterol": [cholesterol],
    "Smoking Status": [smoker]
})

st.dataframe(
    display_data,
    use_container_width=True,
    hide_index=True
)


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

st.subheader("3. Generate Prediction")

predict_button = st.button(
    "Generate Prediction",
    type="primary"
)

if predict_button:

    try:

        prediction = model.predict(input_data)[0]

        st.subheader("Prediction Result")

        if prediction == 1:

            st.markdown("""
            <div class="warning-box">
            <b>Prediction: Higher Risk</b><br><br>
            The AI model identified patterns associated with a higher-risk
            classification.
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class="success-box">
            <b>Prediction: Lower Risk</b><br><br>
            The AI model identified patterns associated with a lower-risk
            classification.
            </div>
            """, unsafe_allow_html=True)


        # -----------------------------------------------
        # PREDICTION PROBABILITY
        # -----------------------------------------------

        if hasattr(model, "predict_proba"):

            probability = model.predict_proba(input_data)[0]

            confidence = max(probability)

            st.write("")
            st.write("### Prediction Confidence")

            st.progress(float(confidence))

            st.metric(
                "Model Confidence",
                f"{confidence * 100:.1f}%"
            )


    except Exception as e:

        st.error(
            "The prediction could not be generated. "
            "Please verify the information entered."
        )

        # During development you can uncomment:
        # st.exception(e)


# ---------------------------------------------------------
# MODEL INFORMATION
# ---------------------------------------------------------

with st.expander("About this AI system"):

    st.write("""
    This application uses a previously trained machine-learning model
    to generate predictions from information provided by the user.

    The interface is designed to provide clear input controls,
    immediate feedback, error handling, and understandable prediction
    results.
    """)


# ---------------------------------------------------------
# DISCLAIMER
# ---------------------------------------------------------

st.info(
    "AI predictions should support, rather than replace, "
    "appropriate human judgment."
)


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown("""
<div class="footer">
AI Architecture and Design • Prediction Interface
</div>
""", unsafe_allow_html=True)