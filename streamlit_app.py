import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("disease_model.pkl")
le = joblib.load("label_encoder.pkl")
all_symptoms = joblib.load("symptoms_list.pkl")

desc_df = pd.read_csv("symptom_Description.csv")
prec_df = pd.read_csv("symptom_precaution.csv")

st.title("🧬 Disease Prediction App")
st.markdown("Select your symptoms and predict the disease.")

user_symptoms = st.multiselect("Choose symptoms:", all_symptoms)

def predict_disease(symptoms):
    input_data = [1 if s in symptoms else 0 for s in all_symptoms]
    prediction = model.predict([input_data])[0]
    disease = le.inverse_transform([prediction])[0]
    return disease

def get_info(disease):
    desc_row = desc_df[desc_df["Disease"] == disease]
    prec_row = prec_df[prec_df["Disease"] == disease]
    description = desc_row["Description"].values[0] if not desc_row.empty else "No description available."
    precautions = prec_row.values[0][1:] if not prec_row.empty else []
    precautions = [p for p in precautions if isinstance(p, str) and p.strip()]
    return description, precautions

if st.button("Predict"):
    if user_symptoms:
        disease = predict_disease(user_symptoms)
        st.success(f"🩺 Predicted Disease: {disease}")

        # Show details
        description, precautions = get_info(disease)
        st.subheader("🧾 Description")
        st.write(description)

        if precautions:
            st.subheader("💡 Precautions")
            for i, p in enumerate(precautions, 1):
                st.markdown(f"**{i}.** {p}")
        else:
            st.info("No precautions found.")
    else:
        st.warning("Please select at least one symptom.")
