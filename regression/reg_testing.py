import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
# from scipy.special import boxcox1p
from scipy.stats import boxcox
import pickle


class RegTestingPipeline:
    def __init__(self, pickle_folder_path):
        self.pickle_folder_path=pickle_folder_path
        self.feature_order = ['town','flat_type','block','storey_range','floor_area_sqm','remaining_lease']
        self.label_encoders = self.pickle_load(os.path.join(self.pickle_folder_path, 'label_encoders.pkl'))
        self.scaler = self.pickle_load(os.path.join(self.pickle_folder_path, 'scaler.pkl'))
        self.boxcox_params = self.pickle_load(os.path.join(self.pickle_folder_path, 'boxcox_params.pkl'))
        self.rf_model = self.pickle_load(os.path.join(self.pickle_folder_path, 'model.pkl'))
        self.mean_resale_price_train = self.pickle_load(os.path.join(self.pickle_folder_path, 'mean_resale_price_train.pkl'))
        self.overall_mean_train = self.pickle_load(os.path.join(self.pickle_folder_path, 'overall_mean_train.pkl'))
    
    def preprocess(self, input_data):
        df = pd.DataFrame([input_data])
        
        #Converting the dtype of month and lease_commence_date to datetime dtype
        df['month'] = pd.to_datetime(df['month'], format='%Y-%m')
        df['lease_commence_date'] = pd.to_datetime(df['lease_commence_date'], format='%Y')

        #Deriving remaining_lease from the month & lease_commence_date 
        df['remaining_lease']= 100 - (abs(df['month'].dt.year - df['lease_commence_date'].dt.year) + 1) 

        #converting the feature to integer
        df['remaining_lease']=df['remaining_lease'].astype(int)

        #Creating a new feature year derived from the month feature
        df['year']=df['month'].dt.year

        #Merge the mean resale prices back to the training data
        for features in ['town', 'block', 'storey_range']:
            df = df.merge(self.mean_resale_price_train[features], on=['year', features], how='left')
            df = df.merge(self.overall_mean_train[features], on=[features], how='left')
            # Fill NaNs with the overall mean
            df[f'{features}_mean_resale_price'].fillna(df[f'{features}_overall_mean_resale_price'], inplace=True)
            df.drop(columns=[f'{features}_overall_mean_resale_price'], inplace=True)
        
        ## Copying the features back to original Feature 
        for features in ['town', 'block', 'storey_range']:
            df[features]=df[f'{features}_mean_resale_price']
        
        ## Dropping unwanted features
        df.drop(['town_mean_resale_price','block_mean_resale_price','storey_range_mean_resale_price','month','year'],axis=1,inplace=True)
        
        
        # Apply the saved label encoders to the test data
        for feature in ['flat_type']:
            if feature in self.label_encoders:
                lb_encoder = self.label_encoders[feature]
                # Ensure the test data contains only seen labels
                df.loc[:, feature] = lb_encoder.transform(df[feature])
        
        # Apply the same Box-Cox transformation on the test data
        for feature in ['floor_area_sqm', 'remaining_lease','town','block','storey_range']:
            df[feature] = boxcox(df[feature] + 1, lmbda=self.boxcox_params[feature])

        # Apply the same Standard Scaler transformation on the test data
        df.loc[:, ['floor_area_sqm', 'remaining_lease','town','block','storey_range']] = self.scaler.transform(df[['floor_area_sqm', 'remaining_lease','town','block','storey_range']])
        
        #Convert the dtype of flat_type to int 
        df['flat_type'] = df['flat_type'].astype(int)

        #Ensure the features are in the same order as training
        df=df[self.feature_order]
        # print(df)

        print("The dtype of the dataframe :", df.dtypes)
        
        return df
    
    def predict(self, preprocessed_df):
        return self.rf_model.predict(preprocessed_df)
    
    def pickle_load(self, filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)


# if __name__ == "__main__":
#     # Usage
#     input_data = {'month'     : '1990-08',
#                   'town'      : 'ANG MO KIO',
#                   'flat_type' : '3 ROOM',
#                   'block'     : '541',
#                   'storey_range' : '04 TO 06',
#                   'floor_area_sqm' : 68.0,
#                   'lease_commence_date' : 1979
#                 }


# pickle_folder_path = 'D:\\Saravanesh Personal\\Guvi\\Capstone Projects\\Singapore Flat Price\\regression\\reg_pickle_files'
# test_pipeline = RegTestingPipeline(pickle_folder_path)
# print("Testing Pipline - Initiated Preprocess ")
# preprocessed_df = test_pipeline.preprocess(input_data)
# print("Testing Pipline - Preprocessing Completed")
# print("Testing Pipline - Initiated Prediction workflow")
# predicted_selling_price = test_pipeline.predict(preprocessed_df)
# print("Testing Pipline - Prediction workflow Completed")
# print(f"Predicted resale Price: {predicted_selling_price}")
