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
    # Load your saved XGBoost model
    with open('best_donor_model_xgboost.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

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
    
    # Create encoding mappings
    workclass_map = {"Private": 2, "Self-emp-not-inc": 4, "Self-emp-inc": 3, 
                     "Federal-gov": 0, "Local-gov": 1, "State-gov": 5, 
                     "Without-pay": 6, "Never-worked": 7}
    
    education_map = {"Bachelors": 9, "HS-grad": 11, "11th": 1, "Masters": 12, 
                     "9th": 0, "Some-college": 13, "Assoc-acdm": 7, "Assoc-voc": 8,
                     "7th-8th": 2, "Doctorate": 10, "Prof-school": 14, "5th-6th": 3,
                     "10th": 4, "1st-4th": 5, "Preschool": 6, "12th": 15}
    
    marital_map = {"Married-civ-spouse": 2, "Never-married": 4, "Divorced": 0,
                   "Separated": 5, "Widowed": 6, "Married-spouse-absent": 3, 
                   "Married-AF-spouse": 1}
    
    occupation_map = {"Tech-support": 12, "Craft-repair": 2, "Other-service": 6,
                      "Sales": 10, "Exec-managerial": 3, "Prof-specialty": 9,
                      "Handlers-cleaners": 5, "Machine-op-inspct": 7, "Adm-clerical": 0,
                      "Farming-fishing": 4, "Transport-moving": 13, "Priv-house-serv": 8,
                      "Protective-serv": 11, "Armed-Forces": 1}
    
    relationship_map = {"Husband": 0, "Not-in-family": 1, "Own-child": 3,
                        "Unmarried": 4, "Wife": 5, "Other-relative": 2}
    
    race_map = {"White": 4, "Black": 2, "Asian-Pac-Islander": 1,
                "Amer-Indian-Eskimo": 0, "Other": 3}
    
    sex_map = {"Male": 1, "Female": 0}
    
    country_map = {"United-States": 39, "Cuba": 5, "Jamaica": 20, "India": 18,
                   "Mexico": 24, "South": 31, "Puerto-Rico": 28}
    
    education_num_map = {
        "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4,
        "9th": 5, "10th": 6, "11th": 7, "12th": 8,
        "HS-grad": 9, "Some-college": 10, "Assoc-voc": 11, "Assoc-acdm": 12,
        "Bachelors": 13, "Masters": 14, "Prof-school": 15, "Doctorate": 16
    }
    
    # Create input data
    input_data = {
        'age': age,
        'workclass': workclass_map.get(workclass, 2),
        'education': education_map.get(education, 9),
        'education-num': education_num_map.get(education, 13),
        'marital-status': marital_map.get(marital_status, 2),
        'occupation': occupation_map.get(occupation, 3),
        'relationship': relationship_map.get(relationship, 0),
        'race': race_map.get(race, 4),
        'sex': sex_map.get(sex, 1),
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'native-country': country_map.get(native_country, 39)
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Make prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    # Display results
    st.markdown("---")
    st.subheader("📊 Prediction Results")
    
    if prediction[0] == 1:
        st.success("**HIGH POTENTIAL DONOR** (Income >$50K)")
        confidence = prediction_proba[0][1] * 100
        st.metric("Confidence Level", f"{confidence:.1f}%")
        
        st.info("**Recommendation:** Add to priority donor contact list")
        
        # Convert to float before passing to progress
        st.progress(float(prediction_proba[0][1]))
        
    else:
        st.warning("❌ **LOW POTENTIAL DONOR** (Income ≤$50K)")
        confidence = prediction_proba[0][0] * 100
        st.metric("Confidence Level", f"{confidence:.1f}%")
        
        st.info("**Recommendation:** May not be ideal for high-value donor campaigns")
        
        # Convert to float before passing to progress
        st.progress(float(prediction_proba[0][0]))
    
    # Show probability distribution
    st.markdown("### Probability Distribution")
    prob_col1, prob_col2 = st.columns(2)
    
    with prob_col1:
        st.metric("≤$50K Probability", f"{prediction_proba[0][0]*100:.1f}%")
    
    with prob_col2:
        st.metric(">$50K Probability", f"{prediction_proba[0][1]*100:.1f}%")

# Footer
st.markdown("---")
st.markdown("*Built with XGBoost - Optimized for maximizing donor identification*")