# Deploy with Streamlit
import streamlit as st # type: ignore
import numpy as np
import pandas as pd
import joblib

best_model = joblib.load("best_model.pkl")

def main():
    st.title("Titanic Survival Prediction")

    # User input features
    Pclass = st.selectbox("Passenger Class", [1, 2, 3])
    Age = st.slider("Age", 0, 80, 30)
    SibSp = st.number_input("Number of Siblings/Spouses aboard", 0, 10, 0)
    Parch = st.number_input("Number of Parents/Children aboard", 0, 10, 0)
    Fare = st.slider("Fare", 0.0, 500.0, 32.2)
    Sex_male = st.selectbox("Sex", ["Female", "Male"])
    Embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

    # When the button is pressed, make the prediction
    if st.button("Predict"):
        # Encode input features
        sex = 1 if Sex_male == "Male" else 0
        embarked_Q = 1 if Embarked == "Q" else 0
        embarked_S = 1 if Embarked == "S" else 0

        # Prepare input data
        input_data = pd.DataFrame(
            [[Pclass, Age, SibSp, Parch, Fare, sex, embarked_Q, embarked_S]],
            columns=[
                "Pclass",
                "Age",
                "SibSp",
                "Parch",
                "Fare",
                "Sex_male",
                "Embarked_Q",
                "Embarked_S",
            ],
        )

        # Predict
        prediction = best_model.predict(input_data)
        # Display result
        if prediction[0] == 1:
            st.write("Survived")
        else:
            st.write("Did not survive")


if __name__ == "__main__":
    main()
