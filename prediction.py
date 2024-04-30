#necessary libraries
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

#streamlit apps
import streamlit as st
import pickle

def main():
    #Description Title for Application
    st.title("Churn Prediction Application")
    st.write("Predicting Churn granted or not")

    geography_input = st.selectbox("Select your Geography :", ["France", "Germany", "Spain"])
    credit_score_input = st.number_input("Input your Credit Score :")
    gender_input = st.selectbox("Select your Gender :", ["Male", "Female"])
    age_input = st.number_input("Inputr your Age :")
    tenure_input = st.number_input("Input your Tenure (years with company) :")
    balance_input = st.number_input("Input your Account Balance :")
    num_of_products_input = st.number_input("Input your Number of Products :")
    has_cr_card_input = st.selectbox("Do you have a Credit Card ? (Y/N)", ["Y", "N"])
    is_active_member_input = st.selectbox("are you an Active Member ? (Y/N)", ["Y", "N"])
    estimated_salary_input = st.number_input("Input your Estimated Salary :")
    
    user_data = pd.DataFrame({
        "CreditScore": [credit_score_input],
        "Geography": [geography_input],
        "Gender": [gender_input],
        "Age": [age_input],
        "Tenure": [tenure_input],
        "Balance": [balance_input],
        "NumOfProducts": [num_of_products_input],
        "HasCrCard": [1 if has_cr_card_input == "Y" else 0],
        "IsActiveMember": [1 if is_active_member_input == "Y" else 0],
        "EstimatedSalary": [estimated_salary_input]
    })


    def load_scaler_and_encoder(encoder_path="encoder.pkl", scaler_path="scaler.pkl"):
        with open(encoder_path, "rb") as encoder_file:
            encoder = pickle.load(encoder_file)
        
        with open(scaler_path, "rb") as scaler_file:
            scaler = pickle.load(scaler_file)
    
        return encoder, scaler

    if st.button("Predict Churn granting"):
        #check all variables
        check_filled = balance_input and num_of_products_input and has_cr_card_input and is_active_member_input and estimated_salary_input and credit_score_input and geography_input and gender_input and age_input and tenure_input

        #decision
        if not check_filled:
            st.error("Please input all the field, before predicting!")
            return

        numeric = ['CreditScore', 'Balance', 'EstimatedSalary']
        categorical = ['Geography', 'Gender']        
        encoder,scaler = load_scaler_and_encoder()
        user_data_subset = user_data[categorical]
        user_data_encoded = pd.DataFrame(encoder.transform(user_data_subset).toarray(), columns=encoder.get_feature_names_out(categorical))
        user_data = user_data.reset_index(drop=True)
        user_data = pd.concat([user_data, user_data_encoded], axis=1)
        user_data.drop(categorical, axis=1, inplace=True)
        user_data[numeric] = scaler.transform(user_data[numeric])
       
        
        with st.spinner("Making prediction..."):
            with open("XGBClassifier.pkl", "rb") as XGB_Classifier:
                models = pickle.load(XGB_Classifier)
            prediction = models.predict(user_data)[0]  
            
            #Churn not granted
            if prediction == 0:
                st.write("Predicted: **You are not Granted a Churn**")
            #churn granted
            else:
                st.write("Predicted: **You are Granted a Churn**")

if __name__ == "__main__":
    main()
