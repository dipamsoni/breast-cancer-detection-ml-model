import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import streamlit as st
import numpy as np
import pickle

def run_model():
    # Load breast cancer data
    data = load_breast_cancer()
    features = pd.DataFrame(data.data, columns=data.feature_names)
    target = pd.Series(data.target)

    # Data Preprocessing (Scaling)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

    # Hyperparameter Tuning (GridSearchCV)
    param_grid = {
        'n_estimators': [10, 50, 100],
        'criterion': ['gini', 'entropy'],
        'max_depth': [2, 4, 6, 8],
        'min_samples_split': [2, 5, 10]
    }

    # Create Random Forest model
    model = RandomForestClassifier()

    # Perform GridSearchCV
    grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=5)

    # Train model with hyperparameter tuning
    grid_search.fit(X_train, y_train)

    # Get best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Make predictions on the test set with best model
    predictions = best_model.predict(X_test)

    # Evaluate the model accuracy
    accuracy = accuracy_score(y_test, predictions)

    with open('RandomForestClassifier.pkl', 'wb') as f:
        pickle.dump((best_model, scaler), f)

    return {
        'best_model': best_model,
        'scaler': scaler,
        'accuracy': accuracy
    }


def run_prediction_from_json(json_data):
    # Convert JSON data to DataFrame
    df = pd.DataFrame.from_dict(json_data, orient='index').transpose()

    # Ensure all values are parsed as floats
    df = df.astype(float)

    with open('RandomForestClassifier.pkl', 'rb') as f:
        loaded_model, loaded_scaler = pickle.load(f)

    # Scale the input data
    input_data_scaled = loaded_scaler.transform(df)

    # Predict probabilities
    prediction = loaded_model.predict_proba(input_data_scaled)

    output = '{0:.{1}f}'.format(prediction[0][1], 2)
    output_print = float(output) * 100

    if float(output) > 0.5:
        return f'Good News, You are not likely to have cancer.\nProbability of not having cancer is {output_print}%.'
    else:
        return f'You have a chance of having cancer.\nProbability of cancer is {100.00 - output_print}%.\nPlease consult a good doctor for cancer treatment.'


def run_prediction(model_result):
    st.header("Breast Cancer Prediction")

    # Load breast cancer data
    data = load_breast_cancer()
    feature_names = data.feature_names

    # Create input fields for each feature
    input_data = []
    for feature in feature_names:
        value = st.number_input(f"Input value for {feature}", value=0.0)
        input_data.append(value)

    # Scale the input data
    input_data = np.array(input_data).reshape(1, -1)
    input_data_scaled = model_result['scaler'].transform(input_data)

    # Predict the probability of having breast cancer
    if st.button("Predict"):
        prediction = model_result["best_model"].predict_proba(input_data_scaled)[0][1]  # Get the probability of class 1 (having breast cancer)
        st.write(f"The probability of Not having breast cancer is {(1 - prediction) * 100:.2f}%")



###############################################################################################
################################## HAVING CANCER ##############################################
###############################################################################################


# {
#     "mean radius": 15.0,
#     "mean texture": 20.0,
#     "mean perimeter": 100.0,
#     "mean area": 750.0,
#     "mean smoothness": 0.1,
#     "mean compactness": 0.2,
#     "mean concavity": 0.2,
#     "mean concave points": 0.1,
#     "mean symmetry": 0.15,
#     "mean fractal dimension": 0.06,
#     "radius error": 0.8,
#     "texture error": 1.0,
#     "perimeter error": 5.0,
#     "area error": 50.0,
#     "smoothness error": 0.005,
#     "compactness error": 0.02,
#     "concavity error": 0.03,
#     "concave points error": 0.015,
#     "symmetry error": 0.015,
#     "fractal dimension error": 0.003,
#     "worst radius": 20.0,
#     "worst texture": 25.0,
#     "worst perimeter": 120.0,
#     "worst area": 1300.0,
#     "worst smoothness": 0.13,
#     "worst compactness": 0.35,
#     "worst concavity": 0.35,
#     "worst concave points": 0.18,
#     "worst symmetry": 0.25,
#     "worst fractal dimension": 0.08
# }


###############################################################################################
################################ NOT HAVING CANCER ############################################
###############################################################################################


# {
#     "mean radius": 10.0,
#     "mean texture": 12.0,
#     "mean perimeter": 60.0,
#     "mean area": 300.0,
#     "mean smoothness": 0.08,
#     "mean compactness": 0.1,
#     "mean concavity": 0.05,
#     "mean concave points": 0.03,
#     "mean symmetry": 0.2,
#     "mean fractal dimension": 0.06,
#     "radius error": 0.1,
#     "texture error": 0.2,
#     "perimeter error": 0.5,
#     "area error": 10.0,
#     "smoothness error": 0.002,
#     "compactness error": 0.005,
#     "concavity error": 0.005,
#     "concave points error": 0.002,
#     "symmetry error": 0.008,
#     "fractal dimension error": 0.001,
#     "worst radius": 12.0,
#     "worst texture": 18.0,
#     "worst perimeter": 70.0,
#     "worst area": 500.0,
#     "worst smoothness": 0.1,
#     "worst compactness": 0.15,
#     "worst concavity": 0.1,
#     "worst concave points": 0.05,
#     "worst symmetry": 0.25,
#     "worst fractal dimension": 0.07
# }
