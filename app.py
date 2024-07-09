import streamlit as st
from regression.reg_testing import RegTestingPipeline
import os

# Streamlit App
st.sidebar.title("Navigation")
page = st.sidebar.radio("Pages", ["Home", "Regression"])

if page == "Home":
    st.title("Singapore Flat Resale Price Prediction")
    st.write("""
        ### Description of the Problem Statement:
        The objective of this project is to develop a machine learning model and deploy it as a user-friendly web application that predicts the resale prices of flats in Singapore. This predictive model will be based on historical data of resale flat transactions, and it aims to assist both potential buyers and sellers in estimating the resale value of a flat.
    """)

elif page == "Regression":
    st.title("Singapore Flat Resale Price Prediction")

    input_data = {
        'month': st.text_input("Month (e.g., 08-2017)"),
        'town': st.text_input("Town (e.g., 28)"),
        'flat_type': st.text_input("Flat Type (e.g., 3 ROOM)"),
        'block': st.text_input("Block (e.g., 2)"),
        'storey_range': st.text_input("Storey Range (e.g., 1500)"),
        'floor_area_sqm': st.text_input("Floor Area (e.g., 1670798778)"),
        'lease_commence_date': st.text_input("Lease Commence Date (e.g., 1977)")
    }

    if st.button("Predict"):
        try:
            input_data['month'] = input_data['month']
            input_data['lease_commence_date'] = input_data['lease_commence_date']
            input_data['town'] = input_data['town']
            input_data['flat_type'] = input_data['flat_type']
            input_data['block'] = input_data['block']
            input_data['storey_range'] = input_data['storey_range']
            input_data['floor_area_sqm'] = float(input_data['floor_area_sqm'])

            # pickle_folder_path = 'D:\\Saravanesh Personal\\Guvi\\Capstone Projects\\Singapore Flat Price\\regression\\reg_pickle_files'
            pickle_folder_path = os.path.join(os.path.dirname(__file__), 'regression', 'reg_pickle_files')
            test_pipeline = RegTestingPipeline(pickle_folder_path)
            preprocessed_df = test_pipeline.preprocess(input_data)
            predicted_selling_price = test_pipeline.predict(preprocessed_df)
            st.write(f"Predicted resale Price: {predicted_selling_price}")
        except ValueError as e:
            st.error(f"Error in input data: {e}")
