# import streamlit as st
# import joblib
# import numpy as np

# # Load the saved model and scaler (Make sure you uploaded them to Colab first!)
# model = joblib.load('crop_svm_model.joblib')
# scaler = joblib.load('app_scaler.joblib')

# st.title("🌱 AgroSmart: Crop Recommendation")

# # Input fields
# n = st.number_input("Nitrogen", value=50)
# p = st.number_input("Phosphorus", value=50)
# k = st.number_input("Potassium", value=50)
# temp = st.number_input("Temperature", value=25.0)
# hum = st.number_input("Humidity", value=60.0)
# ph = st.number_input("pH", value=6.5)
# rain = st.number_input("Rainfall", value=100.0)

# if st.button("Predict"):
#     input_data = np.array([[n, p, k, temp, hum, ph, rain]])
#     scaled_data = scaler.transform(input_data)
#     prediction = model.predict(scaled_data)
#     st.success(f"Best Crop: {prediction[0]}")
import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="AgroSmart Pro", layout="wide")


with st.sidebar:
    st.title("🌱 AgroSmart")
    st.info("This tool uses SVM to predict the best crop based on soil nutrients and weather.")

    st.header("Pipeline Progress")
  
    st.checkbox("1. Data Input", value=True, disabled=True)
    st.checkbox("2. Feature Scaling", value=True, disabled=True)
    st.checkbox("3. Model Prediction", value=False)
    st.checkbox("4. Performance Metrics", value=False)


st.title("🌾 Crop Recommendation System")
st.markdown("ML-powered agricultural pipeline for soil assessment.")

tabs = st.tabs(["1. Input Data", "2. Scaler Info", "3. Prediction", "4. Metrics"])

with tabs[0]:
    st.header("5️⃣ Data Entry")
    st.write("Enter the soil records for analysis.")

    col1, col2 = st.columns(2)
    with col1:
        n = st.number_input("Nitrogen (N)", value=50)
        p = st.number_input("Phosphorus (P)", value=50)
        k = st.number_input("Potassium (K)", value=50)
    with col2:
        temp = st.number_input("Temperature", value=25.0)
        hum = st.number_input("Humidity", value=60.0)
        ph = st.number_input("pH Level", value=6.5)
        rain = st.number_input("Rainfall", value=100.0)


with tabs[1]:
    st.header("⚙️ Feature Scaling")
    st.info("Using StandardScaler to normalize features (Unit 1 Topic).")
    st.write("This ensures features like Rainfall (0-300) don't overpower pH (0-14).")

with tabs[2]:
    st.header("🔮 Model Prediction")
    if st.button("Run SVM Model"):
        # Load and Predict
        model = joblib.load('crop_svm_model.joblib')
        scaler = joblib.load('app_scaler.joblib')

        features = np.array([[n, p, k, temp, hum, ph, rain]])
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)

        st.success(f"### Predicted Crop: {prediction[0]}")


with tabs[3]:
    st.header("📊 Diagnostics")
    st.write("Model Performance Metrics (Unit 1).")

    st.image("confusion_matrix.png", caption="SVM Model Confusion Matrix")

    st.metric(label="Model Accuracy", value="98.2%")
