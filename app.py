# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle

# st.write("""
# # Heart disease Prediction App

# This app predicts If a patient has a heart disease

# Data obtained from Kaggle: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset.
# """)

# st.sidebar.header('User Input Features')



# # Collects user input features into dataframe

# def user_input_features():
#     age = st.sidebar.number_input('Enter your age: ')

#     sex  = st.sidebar.selectbox('Sex', (0, 1))
#     cp = st.sidebar.selectbox('Chest pain type', (0, 1, 2, 3))
#     tres = st.sidebar.number_input('Resting blood pressure: ')
#     chol = st.sidebar.number_input('Serum cholestoral in mg/dl: ')
#     fbs = st.sidebar.selectbox('Fasting blood sugar', (0, 1))
#     res = st.sidebar.number_input('Resting electrocardiographic results: ')
#     tha = st.sidebar.number_input('Maximum heart rate achieved: ')
#     exa = st.sidebar.selectbox('Exercise induced angina', (0, 1))
#     old = st.sidebar.number_input('Oldpeak')
#     slope = st.sidebar.number_input('Slope of the peak exercise ST segment: ')
#     ca = st.sidebar.selectbox('Number of major vessels', (0, 1, 2, 3))
#     thal = st.sidebar.selectbox('Thal', (0, 1, 2))

#     # Create a dictionary containing user inputs
#     data = {
#         'age': age,
#         'sex': sex,
#         'cp': cp,
#         'trestbps': tres,
#         'chol': chol,
#         'fbs': fbs,
#         'restecg': res,
#         'thalach': tha,
#         'exang': exa,
#         'oldpeak': old,
#         'slope': slope,
#         'ca': ca,
#         'thal': thal
#     }

#     # Convert the dictionary to a DataFrame with a single row
#     features = pd.DataFrame(data, index=[0])

#     # Perform one-hot encoding on categorical columns
#     features = pd.get_dummies(features, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

#     return features

# # input_df = pd.DataFrame(data, index=[0])

# # # Perform one-hot encoding on categorical columns
# # input_df = pd.get_dummies(input_df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])



# # Combines user input features with entire dataset
# # This will be useful for the encoding phase
# # heart_dataset = pd.read_csv('heart.csv')
# # heart_dataset = heart_dataset.drop(columns=['target'])

# # df = pd.concat([input_df,heart_dataset],axis=0)

# # # Encoding of ordinal features
# # # https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
# # df = pd.get_dummies(df, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

# # df = df[:1] # Selects only the first row (the user input data)

# # st.write(input_df)
# # # Reads in saved classification model
# # load_clf = pickle.load(open('Random_forest_model.pkl', 'rb'))

# # # Apply model to make predictions
# # prediction = load_clf.predict(df)
# # prediction_proba = load_clf.predict_proba(df)


# # st.subheader('Prediction')
# # st.write(prediction)

# # st.subheader('Prediction Probability')
# # st.write(prediction_proba)
# # Drop the columns that aren't part of the user input

# # heart_dataset = pd.read_csv('heart.csv')
# # heart_dataset = heart_dataset.drop(columns=['target'])

# # # Encode user input data
# # input_df_encoded = pd.get_dummies(input_df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

# # # Ensure both DataFrames have the same columns
# # input_df_encoded = input_df_encoded.reindex(columns=heart_dataset.columns, fill_value=0)
# # # Reads in saved classification model
# # # with open('Random_forest_model.pkl', 'rb') as file:
# # load_clf = pickle.load(open('Random_forest_model.pkl', 'rb'))

# # # Apply model to make predictions
# # prediction = load_clf.predict(input_df_encoded)
# # prediction_proba = load_clf.predict_proba(input_df_encoded)

# # st.subheader('Prediction')
# # st.write(prediction)

# # st.subheader('Prediction Probability')
# # st.write(prediction_proba)


# # Drop the columns that aren't part of the user input
# # heart_dataset = pd.read_csv('heart.csv')
# # heart_dataset = heart_dataset.drop(columns=['target'])

# # # Encode user input data
# # input_df_encoded = pd.get_dummies(input_df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

# # # Ensure both DataFrames have the same columns
# # input_df_encoded = input_df_encoded.reindex(columns=heart_dataset.columns, fill_value=0)

# # # Reads in saved classification model
# # with open('Random_forest_model.pkl', 'rb') as file:
# #     load_clf = pickle.load(file)

# # # Apply model to make predictions
# # prediction = load_clf.predict(input_df_encoded)
# # prediction_proba = load_clf.predict_proba(input_df_encoded)

# # st.subheader('Prediction')
# # st.write(prediction)

# # st.subheader('Prediction Probability')
# # st.write(prediction_proba)

# # Drop the columns that aren't part of the user input

# # heart_dataset = pd.read_csv('heart.csv')
# # heart_dataset = heart_dataset.drop(columns=['target'])

# # # Encode user input data
# # input_df_encoded = pd.get_dummies(input_df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

# # # Ensure both DataFrames have the same columns
# # input_df_encoded = input_df_encoded.reindex(columns=heart_dataset.columns, fill_value=0)

# # # Load the trained model
# # with open('Random_forest_model.pkl', 'rb') as file:
# #     load_clf = pickle.load(file)

# # # Apply model to make predictions
# # try:
# #     prediction = load_clf.predict(input_df_encoded)
# #     prediction_proba = load_clf.predict_proba(input_df_encoded)

# #     st.subheader('Prediction')
# #     st.write(prediction)

# #     st.subheader('Prediction Probability')
# #     st.write(prediction_proba)
# # except ValueError as e:
# #     st.error(f"Prediction error: {e}")

# # Drop the columns that aren't part of the user input
# heart_dataset = pd.read_csv('heart.csv')
# heart_dataset = heart_dataset.drop(columns=['target'])

# # Encode user input data (ensure consistent one-hot encoding)
# input_df_encoded = pd.get_dummies(features, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

# # Ensure columns match the training data columns
# missing_cols = set(heart_dataset.columns) - set(input_df_encoded.columns)
# for col in missing_cols:
#     input_df_encoded[col] = 0

# # Reorder columns to match the model's expected order
# input_df_encoded = input_df_encoded[heart_dataset.columns]

# # Load the trained model
# with open('Random_forest_model.pkl', 'rb') as file:
#     load_clf = pickle.load(file)

# # Apply model to make predictions
# try:
#     prediction = load_clf.predict(input_df_encoded)
#     prediction_proba = load_clf.predict_proba(input_df_encoded)

#     st.subheader('Prediction')
#     st.write(prediction)

#     st.subheader('Prediction Probability')
#     st.write(prediction_proba)
# except ValueError as e:
#     st.error(f"Prediction error: {e}")

import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.write("""
# Heart Disease Prediction App

This app predicts if a patient has a heart disease

Data obtained from Kaggle: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset.
""")

st.sidebar.header('User Input Features')

# Collects user input features into DataFrame
def user_input_features():
    age = st.sidebar.number_input('Enter your age: ')
    sex = st.sidebar.selectbox('Sex', (0, 1))
    cp = st.sidebar.selectbox('Chest pain type', (0, 1, 2, 3))
    tres = st.sidebar.number_input('Resting blood pressure: ')
    chol = st.sidebar.number_input('Serum cholestoral in mg/dl: ')
    fbs = st.sidebar.selectbox('Fasting blood sugar', (0, 1))
    res = st.sidebar.number_input('Resting electrocardiographic results: ')
    tha = st.sidebar.number_input('Maximum heart rate achieved: ')
    exa = st.sidebar.selectbox('Exercise induced angina', (0, 1))
    old = st.sidebar.number_input('Oldpeak')
    slope = st.sidebar.number_input('Slope of the peak exercise ST segment: ')
    ca = st.sidebar.selectbox('Number of major vessels', (0, 1, 2, 3))
    thal = st.sidebar.selectbox('Thal', (0, 1, 2))

    # Create a dictionary containing user inputs
    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': tres,
        'chol': chol,
        'fbs': fbs,
        'restecg': res,
        'thalach': tha,
        'exang': exa,
        'oldpeak': old,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }

    # Convert the dictionary to a DataFrame with a single row
    features = pd.DataFrame(data, index=[0])

    # Perform one-hot encoding on categorical columns
    # features = pd.get_dummies(features, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

    return features

# Get user input features
input_df = user_input_features()

# Load heart dataset and drop target column
heart_dataset = pd.read_csv('heart.csv')
heart_dataset = heart_dataset.drop(columns=['target'])

# Ensure user input data has consistent encoding
# input_df_encoded = pd.get_dummies(input_df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

# Reorder columns to match the model's expected order and handle missing columns
# expected_cols = ['sex_0', 'sex_1', 'cp_0', 'cp_1', 'cp_2', 'cp_3', 'fbs_0', 'fbs_1', 'restecg_0', 'restecg_1', 'restecg_2', 'exang_0', 'exang_1', 'slope_0', 'slope_1', 'slope_2', 'ca_0', 'ca_1', 'ca_2', 'ca_3', 'thal_0', 'thal_1', 'thal_2']
# for col in expected_cols:
#     if col not in input_df_encoded.columns:
#         input_df_encoded[col] = 0

# Reorder columns to match the model's expected order
# input_df_encoded = input_df_encoded[expected_cols]

# Load the trained model
with open('Random_forest_model.pkl', 'rb') as file:
    load_clf = pickle.load(file)

# # Apply model to make predictions
try:
    prediction = load_clf.predict(input_df)
    prediction_proba = load_clf.predict_proba(input_df)
    target = prediction
    if (target==1):
        target = "you are likely to have heart disease"
    else:
        target = "You seem to be safe."
    
    st.subheader('Prediction')
    st.write(target)

    st.subheader('Prediction Probability')
    st.write(prediction_proba)
except ValueError as e:
    st.error(f"Prediction error: {e}")

