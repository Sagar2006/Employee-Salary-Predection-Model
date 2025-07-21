import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Employee Income Classification", page_icon="💼", layout="centered")
st.title("💼 Employee Income Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Load the trained model
model = joblib.load("src/best_model.pkl")
scaler = joblib.load("src/scaler.pkl")
encoders = joblib.load("src/encoder.pkl")

# Helper to get encoder classes as sorted list
get_classes = lambda col: sorted(list(encoders[col].classes_))

# Sidebar inputs
st.sidebar.header("Input Employee Details")

age = st.sidebar.slider("Age", 17, 75, 30)
workclass = st.sidebar.selectbox("Workclass", get_classes('workclass'))
fnlwgt = st.sidebar.number_input("fnlwgt", min_value=0, value=100000)
education = st.sidebar.selectbox("Education", [
    "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"
])
educational_num = st.sidebar.slider("Educational-num", 1, 16, 10)
marital_status = st.sidebar.selectbox("Marital Status", get_classes('marital-status'))
occupation = st.sidebar.selectbox("Occupation", get_classes('occupation'))
relationship = st.sidebar.selectbox("Relationship", get_classes('relationship'))
race = st.sidebar.selectbox("Race", get_classes('race'))
gender = st.sidebar.selectbox("Gender", get_classes('gender'))
capital_gain = st.sidebar.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.sidebar.number_input("Capital Loss", min_value=0, value=0)
hours_per_week = st.sidebar.slider("Hours per week", 1, 99, 40)
native_country = st.sidebar.selectbox("Native Country", get_classes('native-country'))

input_df = pd.DataFrame({
    'age': [age],
    'workclass': [workclass],
    'fnlwgt': [fnlwgt],
    'education': [education],
    'educational-num': [educational_num],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'native-country': [native_country]
})

st.write("### 🔎 Input Data")
st.write(input_df)

if st.button("Predict Income Class"):
    # Preprocess input
    input_df_enc = input_df.copy()
    for col in ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']:
        input_df_enc[col] = encoders[col].transform(input_df_enc[col])
    input_df_enc = input_df_enc.drop(columns=['education'])
    input_df_scaled = scaler.transform(input_df_enc)
    prediction = model.predict(input_df_scaled)
    st.success(f"✅ Prediction: {'>50K' if prediction[0] == 1 else '<=50K'}")

st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")
if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    batch_data_enc = batch_data.copy()
    for col in ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']:
        batch_data_enc[col] = encoders[col].transform(batch_data_enc[col])
    batch_data_enc = batch_data_enc.drop(columns=['education'])
    batch_data_scaled = scaler.transform(batch_data_enc)
    batch_preds = model.predict(batch_data_scaled)
    batch_data['PredictedIncomeClass'] = ['>50K' if p == 1 else '<=50K' for p in batch_preds]
    st.write("✅ Predictions:")
    st.write(batch_data.head())
    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, file_name='predicted_income_classes.csv', mime='text/csv')
