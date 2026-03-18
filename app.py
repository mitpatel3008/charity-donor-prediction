import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Page configuration
st.set_page_config(
    page_title="Charity Donor Predictor",
    page_icon="💰",
    layout="wide"
)

# Title and description
st.title("💰 Charity Donor Prediction System")
st.markdown("**Predict if a person is likely to donate based on their profile**")
st.markdown("---")

# Load the trained model
@st.cache_resource
def load_model():
    with open('best_donor_model_xgboost.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)

    with open('model_threshold.pkl', 'rb') as f:
        threshold = pickle.load(f)

    return model, label_encoders, threshold

model, label_encoders, threshold = load_model()

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Personal Information")
    
    age = st.number_input("Age", min_value=17, max_value=90, value=35, step=1)
    
    sex = st.selectbox("Sex", ["Male", "Female"])
    
    race = st.selectbox("Race", [
        "White", "Black", "Asian-Pac-Islander", 
        "Amer-Indian-Eskimo", "Other"
    ])
    
    marital_status = st.selectbox("Marital Status", [
        "Married-civ-spouse", "Never-married", "Divorced",
        "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"
    ])
    
    relationship = st.selectbox("Relationship", [
        "Husband", "Not-in-family", "Own-child",
        "Unmarried", "Wife", "Other-relative"
    ])

with col2:
    st.subheader("💼 Professional Information")
    
    workclass = st.selectbox("Work Class", [
        "Private", "Self-emp-not-inc", "Self-emp-inc",
        "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"
    ])
    
    education = st.selectbox("Education", [
        "Bachelors", "HS-grad", "11th", "Masters", "9th",
        "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th",
        "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th",
        "Preschool", "12th"
    ])
    
    occupation = st.selectbox("Occupation", [
        "Tech-support", "Craft-repair", "Other-service", "Sales",
        "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
        "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
        "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"
    ])
    
    hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=100, value=40, step=1)

st.markdown("---")

col3, col4 = st.columns(2)

with col3:
    st.subheader("💵 Financial Information")
    
    capital_gain = st.number_input("Capital Gain", min_value=0, max_value=100000, value=0, step=100)
    capital_loss = st.number_input("Capital Loss", min_value=0, max_value=5000, value=0, step=100)

with col4:
    st.subheader("🌍 Other Information")
    
    native_country = st.selectbox("Native Country", [
        "United-States", "Cuba", "Jamaica", "India", "Mexico",
        "South", "Puerto-Rico", "Honduras", "England", "Canada",
        "Germany", "Iran", "Philippines", "Italy", "Poland",
        "Columbia", "Cambodia", "Thailand", "Ecuador", "Laos",
        "Taiwan", "Haiti", "Portugal", "Dominican-Republic",
        "El-Salvador", "France", "Guatemala", "China", "Japan",
        "Yugoslavia", "Peru", "Outlying-US(Guam-USVI-etc)",
        "Scotland", "Trinadad&Tobago", "Greece", "Nicaragua",
        "Vietnam", "Hong", "Ireland", "Hungary", "Holand-Netherlands"
    ])

st.markdown("---")

# Prediction button
if st.button("Predict Donor Likelihood", type="primary", use_container_width=True):

    education_num_map = {
        "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4,
        "9th": 5, "10th": 6, "11th": 7, "12th": 8,
        "HS-grad": 9, "Some-college": 10, "Assoc-voc": 11, "Assoc-acdm": 12,
        "Bachelors": 13, "Masters": 14, "Prof-school": 15, "Doctorate": 16
    }

    input_data = {
        'age':            age,
        'workclass':      int(label_encoders['workclass'].transform([workclass])[0]),
        'education':      int(label_encoders['education'].transform([education])[0]),
        'education-num':  education_num_map[education],
        'marital-status': int(label_encoders['marital-status'].transform([marital_status])[0]),
        'occupation':     int(label_encoders['occupation'].transform([occupation])[0]),
        'relationship':   int(label_encoders['relationship'].transform([relationship])[0]),
        'race':           int(label_encoders['race'].transform([race])[0]),
        'sex':            int(label_encoders['sex'].transform([sex])[0]),
        'capital-gain':   capital_gain,
        'capital-loss':   capital_loss,
        'hours-per-week': hours_per_week,
        'native-country': int(label_encoders['native-country'].transform([native_country])[0]),
    }

    input_df = pd.DataFrame([input_data])

    prediction_proba = model.predict_proba(input_df)
    prob_donor = float(prediction_proba[0][1])
    prediction = 1 if prob_donor >= threshold else 0

    st.markdown("---")
    st.subheader("📊 Prediction Results")

    if prediction == 1:
        st.success("**HIGH POTENTIAL DONOR** (Income >$50K)")
        confidence = prob_donor * 100
        st.metric("Confidence Level", f"{confidence:.1f}%")
        st.info("**Recommendation:** Add to priority donor contact list")
        st.progress(float(prob_donor))
    else:
        st.warning("❌ **LOW POTENTIAL DONOR** (Income ≤$50K)")
        confidence = prediction_proba[0][0] * 100
        st.metric("Confidence Level", f"{confidence:.1f}%")
        st.info("**Recommendation:** May not be ideal for high-value donor campaigns")
        st.progress(float(prediction_proba[0][0]))

# Footer
st.markdown("---")
st.markdown("*Built with XGBoost - Optimized for maximizing donor identification*")