#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import statistics

class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.input_df = None
        self.output_df = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        print(self.data.head())

    def remove_na(self):
        self.data.dropna(inplace=True)

    def create_input_output(self, target_column):
        self.output_df = self.data[target_column]
        self.input_df = self.data.drop([target_column], axis=1)

class ModelHandler:
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.x_train, self.x_test, self.y_train, self.y_test, self.y_predict = [None] * 5
        self.encoder = None
        self.scaler = None
    
    def drop_columns(self, columns):
        self.input_data = self.input_data.drop(columns, axis=1)

    def split_data(self, test_size=0.2, random_state=42):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.input_data, self.output_data, test_size=test_size, random_state=random_state)

    def one_hot_encode(self, columns):
        encoder = OneHotEncoder()
        x_train_subset = self.x_train[columns]
        train_encoded = pd.DataFrame(encoder.fit_transform(x_train_subset).toarray(), columns=encoder.get_feature_names_out())
        self.x_train = self.x_train.reset_index(drop=True)
        self.x_train = pd.concat([self.x_train, train_encoded], axis=1)

        x_test_subset = self.x_test[columns]
        test_data = pd.DataFrame(encoder.transform(x_test_subset).toarray(), columns=encoder.get_feature_names_out())
        self.x_test = self.x_test.reset_index(drop=True)
        self.x_test = pd.concat([self.x_test, test_data], axis=1)
        
        self.x_train.drop(columns, axis=1, inplace=True)
        self.x_test.drop(columns, axis=1, inplace=True)

        self.encoder = encoder

    def scale_features(self, columns):
        scaler = MinMaxScaler()
        self.x_train[columns] = scaler.fit_transform(self.x_train[columns])
        self.x_test[columns] = scaler.transform(self.x_test[columns])

        self.scaler = scaler
    
    def export_encoder(self, path="encoder.pkl"):
        with open(path, "wb") as encoder_file:
            pickle.dump(self.encoder, encoder_file)

    def export_scaler(self, path="scaler.pkl"):
        with open(path, "wb") as scaler_file:
            pickle.dump(self.scaler, scaler_file)
    
    def create_model(self):
        self.model = XGBClassifier(
            n_estimators=150,
            max_depth=7,
            learning_rate=0.1,
            gamma=0.4,
            colsample_bytree=0.8
        )

    def train_model(self):
        self.model.fit(self.x_train, self.y_train)

    def predict(self):
        self.y_predict =  self.model.predict(self.x_test)

    def pickle_dump(self, filename='finalized_model.pkl'):
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)
        
    def create_report(self):
        print('\nClassification Report\n')
        print(classification_report(self.y_test, self.y_predict))

    def evaluate_model(self):
        predictions = self.model.predict(self.x_test)
        return accuracy_score(self.y_test, predictions)


if __name__ == "__main__":
    file_path = 'data_A.csv'  
    data_handler = DataHandler(file_path)
    data_handler.load_data()
    data_handler.create_input_output('churn')
    data_handler.remove_na()

    input_df = data_handler.input_df
    output_df = data_handler.output_df

    model_handler = ModelHandler(input_df, output_df)
    model_handler.drop_columns(['Unnamed: 0', 'id', 'CustomerId', 'Surname'])
    model_handler.split_data()
    model_handler.one_hot_encode(['Geography', 'Gender'])
    model_handler.scale_features(['CreditScore', 'Balance', 'EstimatedSalary'])
    model_handler.create_model()
    model_handler.train_model()
    model_handler.predict()
    print("Model Accuracy:", model_handler.evaluate_model())
    model_handler.create_report()

    # Export the encoder and scaler
    model_handler.export_encoder()
    model_handler.export_scaler()


# In[ ]:




