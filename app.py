import numpy as np
import pandas as pd
import pickle
import streamlit as st

# Load the model and encoders

model_path = 'C:/Users/srini/Desktop/New folder/MachineLearning/Cancer_prediction/trained_model.pkl'

with open(model_path, 'rb') as file:
    loaded_model = pickle.load(file)

loaded_model = pickle.load(open(model_path, 'rb'))

# Prediction function
def cancer_prediction(input_data):
    input_data_df = pd.DataFrame([input_data])

    # Apply encoders
    for column, encoder in loaded_encoders.items():
        try:
            input_data_df[column] = encoder.transform(input_data_df[[column]])
        except Exception as e:
            st.error(f"Encoding error in column '{column}': {e}")
            return

    # Predict
    prediction = loaded_model.predict(input_data_df)
    prediction_proba = loaded_model.predict_proba(input_data_df)

    return prediction[0], prediction_proba[0][1]  # Return class and probability

# Streamlit UI
def main():
    st.title('📊 Cancer Prediction')

    # Form inputs
    Gender = st.selectbox("Gender", ['Female', 'Male'])
    Age = st.number_input("Age", min_value=0)
    Smokes = st.selectbox("Smokes", ['Yes', 'No'])
    no_of_cigarettes_per_day = st.number_input("No of Cigarettes per day", min_value=0, value=0, step=1)
    no_of_hrs_sleep_per_day = st.number_input("No of Hrs Sleep per Day", min_value=0.0, value=0.0, step=0.5)
    no_of_hrs_exercise_per_day = st.number_input("No of Hrs Exercise per Day", min_value=0.0, max_value=24.0, value=0.0, step=0.5)
    Diet = st.selectbox("Diet", ['Vegetarian', 'NonVegetarian'])
    Alcoholic = st.selectbox("Alcoholic", ['Occasional', 'Regular', 'No'])
    Height = st.number_input("Height", min_value=0)
    Complexion = st.selectbox("Complexion", ['Fair','Dark','Wheatish','Brown'])
    

    # When button clicked
    if st.button("Predict"):
        input_data = {
            'Gender': Gender,
            'Age': Age,
            'Smokes': Smokes,
            'No of Ciggarets per day': no_of_cigarettes_per_day,
            'No of Hrs Sleep per Day': no_of_hrs_sleep_per_day,
            'No of Hrs Exercise per Day': no_of_hrs_exercise_per_day,
            'Diet': Diet,
            'Alcoholic': Alcoholic,
            'Height': Height,
            'Complexion': Complexion
        }

        result, proba = cancer_prediction(input_data)
        st.success(f"Prediction: {'Positive' if result == 1 else 'Negative'}")
        st.info(f"Cancer Probability: {proba:.2%}")

# Run the app
if __name__ == '__main__':
    main()
