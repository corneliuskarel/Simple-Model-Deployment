import pandas as pd
from sklearn.preprocessing import OneHotEncoder
#streamlit lib
import streamlit as st

#pickle lib
import pickle

def main():
    st.title("Customer Churn Prediction")
    st.write("Churn prediction application")

    score = st.number_input("Credit Score Value:")
    location = st.selectbox("Location :", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender Pick:", ["Male", "Female"])
    age = st.number_input("Age :")
    tenure = st.number_input("Tenure (years with company) Value :")
    balance = st.number_input("Account Balance :")
    products = st.number_input("Number of Products :")
    has_credit_card = st.selectbox("Has Credit Card? (Y/N) ", ["Y", "N"])
    active_member = st.selectbox("Is Active Member? (Y/N)", ["Y", "N"])
    salary = st.number_input("Estimated Salary Number :")

    user_data = pd.DataFrame({
        "CreditScore": [score],
        "Location": [location],
        "Gender": [gender],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [products],
        "HasCreditCard": [1 if has_credit_card == "Yes" else 0],
        "IsActiveMember": [1 if active_member == "Yes" else 0],
        "EstimatedSalary": [salary]
    })

    scalers, encoder = load_scalers_encoder()

    user_data = preprocess_user_data(user_data, encoder, scalers)

    if st.button("Predict Churn Risk"):
        with st.spinner("Making prediction..."):
            prediction = predict_churn(user_data)
            display_prediction(prediction)

def load_scalers_encoder(scaler_path="scaler.pkl", encoder_path="encoder.pkl"):
    """Load scaler and encoder"""
    with open(scaler_path, "rb") as scaler_file:
        scalers = pickle.load(scaler_file)
    with open(encoder_path, "rb") as encoder_file:
        encoder = pickle.load(encoder_file)
    return scalers, encoder

def preprocess_user_data(user_data, encoder, scalers):
    """Preprocess user data"""
    categorical = ['Geography', 'Gender']
    continuous = ['Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

    # Ensure the user_data DataFrame has the correct column names
    user_data.columns = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

    # Drop 'CreditScore' column
    user_data.drop('CreditScore', axis=1, inplace=True)

    # Encode categorical features
    user_data_categorical = user_data[categorical]
    user_data_encoded = pd.DataFrame(encoder.transform(user_data_categorical).toarray(), columns=encoder.get_feature_names_out(categorical))
    
    # Drop original categorical columns and concatenate encoded features
    user_data_encoded.index = user_data.index
    user_data = user_data.drop(columns=categorical)
    user_data = pd.concat([user_data, user_data_encoded], axis=1)

    # Scale continuous features
    user_data[continuous] = scalers.transform(user_data[continuous])

    return user_data

def predict_churn(user_data):
    """Predict churn"""
    with open("XGBClassifier.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    prediction = model.predict(user_data)[0]
    return prediction

def display_prediction(prediction):
    """Display prediction"""
    if prediction == 1:
        st.write("Predicted: **Churn granted*")
    else:
        st.write("Predicted: **Churn not granted**")

if __name__ == "__main__":
    main()
