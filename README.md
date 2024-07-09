# Singapore Flat Resale Price Prediction - Regression Model

#### Problem Statement:
#### The objective of this project is to develop a machine learning model and deploy it as a user-friendly web application that predicts the resale prices of flats in Singapore. This predictive model will be based on historical data of resale flat transactions, and it aims to assist both potential buyers and sellers in estimating the resale value of a flat.

###### Link to the Dataset : https://beta.data.gov.sg/collections/189/view

### Download the dataset and store it in the project folder before proceeding

### Setting up the conda environment 
```conda create -p singaporenv python==3.10```

### Activate the conda environment
```conda activate singaporenv\```

### Install all the requirements 
```pip install -r requirements.txt```

### Regression Model 
#### Training - Regression 
#### Path : \Singapore Flat Price\regression
```python reg_training.py```
#### Model Training would be completed and the following pickle files would be generated 
#### pickle file path : \Industrial Copper Mining\regression\reg_pickle_files
#### boxcox_params.pkl, capping_bounds.pkl, label_encoders.pkl, rf_model.pkl, scaler.pkl 


### Model Testing 
### Run the Streamlit app, pass the required inputs and click on Predict
### In order to test the Regression Model click on Regressoon
#### Path : \Singapore Flat Price
```streamlit run app.py```

