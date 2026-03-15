import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# Custom Styling
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model pipeline
@st.cache_resource
def load_model():
    # We save the full pipeline which includes the preprocessor
    return joblib.load('model.pkl')

model = load_model()

# Header
st.title("🚢 Titanic Survival Prediction")
st.markdown("""
Welcome to the Titanic Survival Predictor! This app uses a **Logistic Regression** model trained on the classic Titanic dataset to predict whether a passenger would have survived the disaster.
""")

# Input Form
st.sidebar.header("Passenger Details")

def get_user_input():
    pclass = st.sidebar.selectbox("Passenger Class (1 = Upper, 2 = Middle, 3 = Lower)", [1, 2, 3])
    sex = st.sidebar.selectbox("Sex", ["male", "female"])
    age = st.sidebar.slider("Age", 0, 100, 25)
    sibsp = st.sidebar.number_input("Siblings/Spouses Aboard (SibSp)", 0, 10, 0)
    parch = st.sidebar.number_input("Parents/Children Aboard (Parch)", 0, 10, 0)
    fare = st.sidebar.number_input("Fare Paid", 0.0, 600.0, 32.2)
    embarked = st.sidebar.selectbox("Port of Embarkation", ["S", "C", "Q"])
    
    data = {
        'Pclass': pclass,
        'Sex': sex,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked': embarked
    }
    return pd.DataFrame([data])

input_df = get_user_input()

st.subheader("Selected Passenger Features")
st.write(input_df)

if st.button("Predict Survival"):
    # Generate prediction and probability
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    st.divider()
    
    # Display result
    if prediction == 1:
        st.success(f"### Prediction: Likely to Survive ✅")
        st.balloons()
    else:
        st.error(f"### Prediction: Not Likely to Survive ❌")
        
    st.metric(label="Survival Probability", value=f"{probability:.1%}")
    
    # Visual progress bar
    st.progress(probability)
    
    # Interpretation based on probability
    if probability > 0.8:
        st.info("The model is very confident about this prediction.")
    elif probability > 0.5:
        st.info("The model suggests survival, but with moderate confidence.")
    elif probability > 0.2:
        st.info("The model suggests survival is unlikely.")
    else:
        st.info("The model is very confident this passenger would not have survived.")

# Footer
st.divider()
st.markdown("Created by Senior ML Engineer | Powered by Scikit-Learn & Streamlit")
