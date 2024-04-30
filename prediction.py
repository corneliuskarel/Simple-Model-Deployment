import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import streamlit as st
import pickle

class ChurnPredictor:
    def __init__(self):
        self.encoder = None
        self.scaler = None
        self.classifier_model = None

    def load_models(self, encoder_path="encoder.pkl", scaler_path="scaler.pkl", classifier_path="XGBClassifier.pkl"):
        with open(encoder_path, "rb") as encoder_file:
            self.encoder = pickle.load(encoder_file)
        
        with open(scaler_path, "rb") as scaler_file:
            self.scaler = pickle.load(scaler_file)

        with open(classifier_path, "rb") as classifier_file:
            self.classifier_model = pickle.load(classifier_file)

    def preprocess_data(self, data):
        categorical_features = ['Geography', 'Gender']
        numeric_features = ['CreditScore', 'Balance', 'EstimatedSalary']
        
        data_categorical = data[categorical_features]
        data_encoded = pd.DataFrame(self.encoder.transform(data_categorical).toarray(), columns=self.encoder.get_feature_names_out(categorical_features))
        data = data.reset_index(drop=True)
        data = pd.concat([data, data_encoded], axis=1)
        data.drop(categorical_features, axis=1, inplace=True)
        data[numeric_features] = self.scaler.transform(data[numeric_features])

        return data

    def predict_churn(self, user_info):
        prediction = self.classifier_model.predict(user_info)[0]

        return prediction

def main():
    churn_predictor = ChurnPredictor()

    st.title("Churn Prediction Application")
    st.write("Predicting Churn Approval")

    country_choice = st.selectbox("Select your Country:", ["France", "Germany", "Spain"])
    credit_score = st.number_input("Enter your Credit Score:")
    gender = st.selectbox("Select your Gender:", ["Male", "Female"])
    age = st.number_input("Enter your Age:")
    tenure = st.number_input("Enter your Tenure (years with company):")
    balance = st.number_input("Enter your Account Balance:")
    num_of_products = st.number_input("Enter your Number of Products:")
    has_credit_card = st.selectbox("Do you have a Credit Card? (Y/N)", ["Y", "N"])
    is_active = st.selectbox("Are you an Active Member? (Y/N)", ["Y", "N"])
    estimated_salary = st.number_input("Enter your Estimated Salary:")
    
    user_info = pd.DataFrame({
        "CreditScore": [credit_score],
        "Geography": [country_choice],
        "Gender": [gender],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [1 if has_credit_card == "Y" else 0],
        "IsActiveMember": [1 if is_active == "Y" else 0],
        "EstimatedSalary": [estimated_salary]
    })

    churn_predictor.load_models()

    if st.button("Predict Churn Approval"):
        if not (balance and num_of_products and has_credit_card and is_active and estimated_salary and credit_score and country_choice and gender and age and tenure):
            st.error("Please fill in all fields before predicting!")
            return

        preprocessed_data = churn_predictor.preprocess_data(user_info)
       
        with st.spinner("Predicting..."):
            prediction = churn_predictor.predict_churn(preprocessed_data)
            
            if prediction == 0:
                st.write("Prediction: **Churn Not Approved**")
            else:
                st.write("Prediction: **Churn Approved**")

if __name__ == "__main__":
    main()
