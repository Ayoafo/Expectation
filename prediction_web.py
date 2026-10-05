import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():
    model = joblib.load("churn_model.pkl")
    return model


model = load_model()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #F4F7FB;
}

/* Main content */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
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

/* Information box */
.info-box {
    background-color: #E8F1FA;
    border-left: 5px solid #2471A3;
    padding: 18px;
    border-radius: 8px;
    color: #17202A;
    margin-bottom: 20px;
}

/* Positive result */
.success-box {
    background-color: #E8F8F0;
    border-left: 6px solid #28A745;
    padding: 20px;
    border-radius: 10px;
    font-size: 18px;
    color: #17202A;
}

/* Warning result */
.warning-box {
    background-color: #FFF3E0;
    border-left: 6px solid #F39C12;
    padding: 20px;
    border-radius: 10px;
    font-size: 18px;
    color: #17202A;
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


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">Customer Churn Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Enter customer information below to generate an
    AI-assisted customer churn prediction.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SYSTEM INFORMATION
# ============================================================

st.markdown("""
<div class="info-box">

<b>How it works:</b><br><br>

Enter customer information → Review the information →
Click <b>Generate Prediction</b> → View the AI prediction.

</div>
""", unsafe_allow_html=True)


st.write("")


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader("1. Enter Customer Information")

st.caption(
    "Provide the customer's information using the fields below."
)


# ============================================================
# CREATE THREE COLUMNS
# ============================================================

col1, col2, col3 = st.columns(3)


# ============================================================
# COLUMN 1
# ============================================================

with col1:

    call_failure = st.number_input(
        "Call Failure",
        min_value=0,
        value=0,
        step=1,
        help="Number of call failures experienced by the customer."
    )

    complains = st.selectbox(
        "Complains",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        help="Indicates whether the customer has complained."
    )

    subscription_length = st.number_input(
        "Subscription Length",
        min_value=0,
        value=20,
        step=1,
        help="Length of the customer's subscription."
    )

    charge_amount = st.number_input(
        "Charge Amount",
        min_value=0,
        value=1,
        step=1,
        help="Customer's charge amount."
    )


# ============================================================
# COLUMN 2
# ============================================================

with col2:

    seconds_of_use = st.number_input(
        "Seconds of Use",
        min_value=0,
        value=5000,
        step=100,
        help="Total seconds of service usage."
    )

    frequency_of_use = st.number_input(
        "Frequency of Use",
        min_value=0,
        value=50,
        step=1,
        help="Frequency with which the customer uses the service."
    )

    frequency_of_sms = st.number_input(
        "Frequency of SMS",
        min_value=0,
        value=20,
        step=1,
        help="Number of SMS messages sent by the customer."
    )

    distinct_called_numbers = st.number_input(
        "Distinct Called Numbers",
        min_value=0,
        value=20,
        step=1,
        help="Number of distinct telephone numbers called."
    )


# ============================================================
# COLUMN 3
# ============================================================

with col3:

    age_group = st.selectbox(
        "Age Group",
        options=[1, 2, 3, 4, 5],
        help="Select the customer's age group."
    )

    tariff_plan = st.selectbox(
        "Tariff Plan",
        options=[1, 2],
        help="Select the customer's tariff plan."
    )

    status = st.selectbox(
        "Status",
        options=[1, 2],
        help="Select the customer's status."
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30,
        step=1,
        help="Enter the customer's age."
    )

    customer_value = st.number_input(
        "Customer Value",
        min_value=0.0,
        value=500.0,
        step=10.0,
        help="Enter the customer's value."
    )


# ============================================================
# CREATE MODEL INPUT
# ============================================================
#
# IMPORTANT:
# These column names must match the columns used when the
# machine-learning model was trained.
#
# Some feature names contain TWO spaces.
# Do not remove those spaces.
# ============================================================

input_data = pd.DataFrame({

    "Call  Failure": [call_failure],

    "Complains": [complains],

    "Subscription  Length": [subscription_length],

    "Charge  Amount": [charge_amount],

    "Seconds of Use": [seconds_of_use],

    "Frequency of use": [frequency_of_use],

    "Frequency of SMS": [frequency_of_sms],

    "Distinct Called Numbers": [distinct_called_numbers],

    "Age Group": [age_group],

    "Tariff Plan": [tariff_plan],

    "Status": [status],

    "Age": [age],

    "Customer Value": [customer_value]

})


# ============================================================
# REVIEW INFORMATION
# ============================================================

st.divider()

st.subheader("2. Review Customer Information")

st.write(
    "Review the information below before generating the prediction."
)


# Create user-friendly version for display
display_data = pd.DataFrame({

    "Call Failure": [call_failure],

    "Complains": [
        "Yes" if complains == 1 else "No"
    ],

    "Subscription Length": [
        subscription_length
    ],

    "Charge Amount": [
        charge_amount
    ],

    "Seconds of Use": [
        seconds_of_use
    ],

    "Frequency of Use": [
        frequency_of_use
    ],

    "Frequency of SMS": [
        frequency_of_sms
    ],

    "Distinct Called Numbers": [
        distinct_called_numbers
    ],

    "Age Group": [
        age_group
    ],

    "Tariff Plan": [
        tariff_plan
    ],

    "Status": [
        status
    ],

    "Age": [
        age
    ],

    "Customer Value": [
        customer_value
    ]

})


st.dataframe(
    display_data,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# GENERATE PREDICTION
# ============================================================

st.divider()

st.subheader("3. Generate Prediction")

st.write(
    "Click the button below after reviewing the customer's "
    "information."
)


predict_button = st.button(
    "Generate Prediction",
    type="primary"
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        # ----------------------------------------------------
        # MAKE PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(input_data)[0]


        st.divider()

        st.subheader("4. Prediction Result")


        # ----------------------------------------------------
        # CUSTOMER LIKELY TO CHURN
        # ----------------------------------------------------

        if prediction == 1:

            st.markdown(
                """
                <div class="warning-box">

                <b>⚠ Prediction: Customer is Likely to Churn</b>

                <br><br>

                The machine-learning model identified patterns
                associated with customers who are likely to
                discontinue the service.

                </div>
                """,
                unsafe_allow_html=True
            )


        # ----------------------------------------------------
        # CUSTOMER UNLIKELY TO CHURN
        # ----------------------------------------------------

        else:

            st.markdown(
                """
                <div class="success-box">

                <b>✓ Prediction: Customer is Unlikely to Churn</b>

                <br><br>

                The machine-learning model identified patterns
                associated with customers who are likely to
                remain with the company.

                </div>
                """,
                unsafe_allow_html=True
            )


        # ====================================================
        # PREDICTION PROBABILITY
        # ====================================================

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_data)[0]

            # Probability of class 1 (churn)
            churn_probability = probabilities[1]

            st.write("")

            st.subheader("Churn Probability")


            # ------------------------------------------------
            # PROGRESS BAR
            # ------------------------------------------------

            st.progress(
                float(churn_probability)
            )


            # ------------------------------------------------
            # DISPLAY CHURN PROBABILITY
            # ------------------------------------------------

            st.metric(
                "Probability of Customer Churn",
                f"{churn_probability * 100:.1f}%"
            )


            # ------------------------------------------------
            # INTERPRET PROBABILITY
            # ------------------------------------------------

            if churn_probability >= 0.70:

                st.warning(
                    "The model estimates a high probability "
                    "of customer churn."
                )

            elif churn_probability >= 0.40:

                st.info(
                    "The model estimates a moderate probability "
                    "of customer churn."
                )

            else:

                st.success(
                    "The model estimates a low probability "
                    "of customer churn."
                )


        # ====================================================
        # SYSTEM FEEDBACK
        # ====================================================

        st.write("")

        st.success(
            "Prediction completed successfully."
        )


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        st.error(
            "The prediction could not be generated. "
            "Please verify the customer information."
        )

        # Useful during development
        st.exception(e)


# ============================================================
# ABOUT THE AI SYSTEM
# ============================================================

st.divider()


with st.expander("About this AI System"):

    st.write("""
    This application uses a previously trained machine-learning
    model to predict whether a customer is likely to churn.

    Customer information is entered through the Streamlit user
    interface and organized into the same feature structure that
    was used during model training.

    The information is then passed to the trained machine-learning
    model, which generates a churn prediction.
    """)


    st.write("### AI System Architecture")

    st.markdown("""
    **Customer Information**

    ↓

    **Streamlit User Interface**

    ↓

    **Input Processing**

    ↓

    **Trained Machine-Learning Model**

    ↓

    **Churn Prediction**

    ↓

    **Prediction Feedback**
    """)


# ============================================================
# INTERFACE DESIGN PRINCIPLES
# ============================================================

with st.expander("Interface Design Principles"):

    st.markdown("""
    **Visibility**

    Important information and system status are clearly visible
    to the user.

    **Consistency**

    Similar controls, labels, colors, and layouts are used
    throughout the interface.

    **Feedback**

    The interface informs the user after the prediction has
    successfully been generated.

    **Error Prevention**

    Input controls restrict users from entering certain invalid
    values.

    **Visual Hierarchy**

    Numbered sections, headings, spacing, and colors guide the
    user through the prediction process.

    **Accessibility**

    Prediction results are communicated using both text and
    color so that users do not have to rely only on color.

    **User Control**

    Customer information can be reviewed before the prediction
    is generated.
    """)


# ============================================================
# RESPONSIBLE AI NOTICE
# ============================================================

st.info(
    "AI predictions should support, rather than replace, "
    "appropriate human judgment."
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

AI Architecture and Design • Customer Churn Prediction System

</div>
""", unsafe_allow_html=True)