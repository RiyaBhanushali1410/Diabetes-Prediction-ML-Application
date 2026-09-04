
import gradio as gr
import numpy as np
import joblib

# Load trained models and preprocessing objects
best_lr = joblib.load("../models/logistic_regression_model.pkl")
best_rf = joblib.load("../models/random_forest_model.pkl")
imputer = joblib.load("../models/imputer.pkl")
scaler = joblib.load("../models/scaler.pkl")


def predict_diabetes(pregnancies, glucose, skin_thickness, bmi, age):

    input_data = np.array([
        [pregnancies, glucose, skin_thickness, bmi, age]
    ])

    # Apply imputation
    input_imputed = imputer.transform(input_data)

    # Logistic Regression
    input_lr = scaler.transform(input_imputed)

    lr_prediction = best_lr.predict(input_lr)[0]
    lr_probability = best_lr.predict_proba(input_lr)[0][1]

    # Random Forest
    rf_prediction = best_rf.predict(input_imputed)[0]
    rf_probability = best_rf.predict_proba(input_imputed)[0][1]

    lr_result = (
        "Diabetes Detected"
        if lr_prediction == 1
        else "No Diabetes Detected"
    )

    rf_result = (
        "Diabetes Detected"
        if rf_prediction == 1
        else "No Diabetes Detected"
    )

    return (
        lr_result,
        f"{lr_probability * 100:.2f}%",
        rf_result,
        f"{rf_probability * 100:.2f}%"
    )


demo = gr.Interface(
    fn=predict_diabetes,

    inputs=[
        gr.Number(label="Pregnancies"),
        gr.Number(label="Glucose"),
        gr.Number(label="Skin Thickness"),
        gr.Number(label="BMI"),
        gr.Number(label="Age")
    ],

    outputs=[
        gr.Textbox(label="Logistic Regression Prediction"),
        gr.Textbox(label="Logistic Regression Probability"),
        gr.Textbox(label="Random Forest Prediction"),
        gr.Textbox(label="Random Forest Probability")
    ],

    title="Diabetes Prediction using Machine Learning",

    description=(
        "Enter patient information to get predictions "
        "from Logistic Regression and Random Forest models."
    )
)

demo.launch()
