import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time

# -------------------------
# 1. APP CONFIGURATION
# -------------------------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="wq",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -------------------------
# 2. CUSTOM CSS (FOR DESIGN)
# -------------------------
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: linear-gradient(to bottom right, #f0f2f6, #e2eafc);
    }
    
    /* Header Styling */
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 20px;
    }
    
    /* Card-like containers for inputs */
    .css-1r6slb0 {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Button Styling */
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #155a8a;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }
    
    /* Result Box Styling */
    .result-box-neg {
        padding: 20px;
        background-color: #d4edda;
        color: #155724;
        border-radius: 10px;
        border: 1px solid #c3e6cb;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .result-box-pos {
        padding: 20px;
        background-color: #f8d7da;
        color: #721c24;
        border-radius: 10px;
        border: 1px solid #f5c6cb;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------
# 3. LOAD MODEL
# -------------------------
@st.cache_resource
def load_model():
    try:
        with open('diabetes_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file not found! Please run the training script first to generate 'diabetes_model.pkl'.")
        return None

model = load_model()

# -------------------------
# 4. SIDEBAR
# -------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=100)
    st.title("About")
    st.info(
        """
        This application uses a **Logistic Regression** machine learning model 
        to predict the likelihood of diabetes based on diagnostic measures.
        """
    )
    st.write("---")
    st.write("**Model Inputs:**")
    st.markdown("- Age\n- BMI (Mass)\n- Insulin Level\n- Plasma Glucose")

# -------------------------
# 5. MAIN INTERFACE
# -------------------------
st.markdown("<h1 class='main-header'>🩺 Diabetes Prediction AI</h1>", unsafe_allow_html=True)
st.write("Enter the patient's details below to generate a prediction.")

# Container for inputs to make it look like a card
with st.container():
    st.markdown("### Patient Information")
    
    # Create two columns for a better layout
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=30, step=1)
        mass = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=70.0, value=33.6, step=0.1)

    with col2:
        insu = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=0, step=1)
        plas = st.number_input("Plasma Glucose", min_value=0, max_value=300, value=148, step=1)

# -------------------------
# 6. PREDICTION LOGIC
# -------------------------
st.markdown("<br>", unsafe_allow_html=True)

if st.button("Analyze Risk"):
    if model:
        # Progress bar for effect
        with st.spinner('Analyzing clinical data...'):
            time.sleep(1) # Simulate processing time
            
            # Input order matches training: [age, mass, insu, plas]
            input_data = [[age, mass, insu, plas]]
            
            # Predict
            prediction = model.predict(input_data)
            result = prediction[0]

        # Display Result
        st.markdown("---")
        st.subheader("Prediction Result:")
        
        if result == "tested_negative":
            st.markdown(
                f"""
                <div class="result-box-neg">
                    ✅ Result: Negative <br>
                    <span style='font-size:16px; font-weight:normal'>The model predicts a low risk of diabetes.</span>
                </div>
                """, 
                unsafe_allow_html=True
            )
            st.balloons()
        else:
            st.markdown(
                f"""
                <div class="result-box-pos">
                    ⚠️ Result: Positive <br>
                    <span style='font-size:16px; font-weight:normal'>The model predicts a likelihood of diabetes.</span>
                </div>
                """, 
                unsafe_allow_html=True
            )
