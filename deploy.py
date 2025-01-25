import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Load models and preprocessing objects
random_forest = joblib.load("C:/Users/abhij/Desktop/ML Model/random_forest.pkl")
decision_tree = joblib.load("C:/Users/abhij/Desktop/ML Model/decision_tree.pkl")
logistic_regression = joblib.load("C:/Users/abhij/Desktop/ML Model/logistic_regression.pkl")
scaler = joblib.load("C:/Users/abhij/Desktop/ML Model/scaler.pkl")
label_encoder = joblib.load("C:/Users/abhij/Desktop/ML Model/label_encoder.pkl")
feature_names = joblib.load("C:/Users/abhij/Desktop/ML Model/features.pkl")

# Title of the app
st.title("Versatile Machine Learning Model Deployment")
st.write(
    "This app automatically aligns and preprocesses your data to predict or evaluate models based on your dataset."
)

# File uploader for CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Load uploaded dataset
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.dataframe(data.head())

    # Preprocessing
    st.write("Preprocessing the data...")
    data.columns = data.columns.str.strip()  # Remove whitespace from column names

    # Check if the target column ('churn') exists
    target_column = st.selectbox(
        "Please select the target column for accuracy evaluation:",
        options=data.columns,
        help="Select the column containing the target labels.",
    )

    if target_column:
        is_evaluation = True
        y_true = data[target_column]
        data = data.drop(columns=[target_column])  # Remove the selected target column from features
    else:
        is_evaluation = False

    # Encode gender column if present
    if 'gender' in data.columns:
        try:
            data['gender'] = label_encoder.transform(data['gender'])
        except Exception as e:
            st.error(f"Error encoding 'gender': {e}")
            st.stop()

    # Handle categorical columns dynamically
    if 'country' in data.columns:
        data = pd.get_dummies(data, columns=['country'], drop_first=True)

    # Align with model's feature set
    missing_cols = set(feature_names) - set(data.columns)
    for col in missing_cols:
        data[col] = 0  # Add missing columns with default value of 0

    # Remove any extra columns not in the model's features
    data = data[feature_names]

    # Scale the features
    st.write("Scaling the features...")
    try:
        X_scaled = scaler.transform(data)
    except Exception as e:
        st.error(f"Error scaling features: {e}")
        st.stop()

    # Define models
    models = {
        "Random Forest": random_forest,
        "Decision Tree": decision_tree,
        "Logistic Regression": logistic_regression
    }

    # Evaluate or Predict
    results = {}
    predictions = {}

    for model_name, model in models.items():
        try:
            model_predictions = model.predict(X_scaled)
            predictions[model_name] = model_predictions

            if is_evaluation:
                accuracy = accuracy_score(y_true, model_predictions) * 100
                results[model_name] = accuracy
        except Exception as e:
            st.error(f"Error during {model_name} operation: {e}")
            st.stop()

    if is_evaluation:
        # Determine the best model
        best_model_name = max(results, key=results.get)
        best_model_accuracy = results[best_model_name]

        st.warning(f"The best model is **{best_model_name}** with an accuracy of **{best_model_accuracy:.2f}%**.")
        st.write("Detailed Results for All Models:")
        for model_name, accuracy in results.items():
            st.write(f"- **{model_name}:** {accuracy:.2f}%")

        # Plot model comparison
        st.write("Model Accuracy Comparison:")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(results.keys(), results.values(), color=['blue', 'green', 'red'])
        ax.set_title('Model Accuracy Comparison')
        ax.set_xlabel('Models')
        ax.set_ylabel('Accuracy (%)')
        st.pyplot(fig)

    else:
        # Display predictions
        st.write("Predictions from All Models:")
        for model_name, model_predictions in predictions.items():
            data[f"Prediction ({model_name})"] = model_predictions

        st.dataframe(data)

    # Save the predictions for download
    csv = data.to_csv(index=False)
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv"
    )
