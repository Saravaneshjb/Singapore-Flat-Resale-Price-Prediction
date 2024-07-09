import os 
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import boxcox
import pickle
import xgboost as xgb

class TrainingPipeline:
    def __init__(self, filepath):
        self.filepath = filepath
        self.label_encoders = {}
        self.boxcox_params = {}
        self.scaler = StandardScaler()
        self.mean_resale_price_train = {}
        self.overall_mean_train={}
        self.params={'n_estimators': 903, 
        'learning_rate': 0.08527804159319546, 
        'max_depth': 10, 
        'subsample': 0.7197469503313069, 
        'colsample_bytree': 0.5702521935670366, 
        'gamma': 0.2926628016065498, 
        'reg_alpha': 0.21637910202752952, 
        'reg_lambda': 0.7074694994590962}
        # self.rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
        self.rf_model = xgb.XGBRegressor(objective='reg:squarederror', **self.params)
        # self.rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
    
    def preprocess(self):
        #Reading all the files downloaded and storing them in a dictionary 
        dataframes = dict()
        counter=1
        for files in os.listdir(self.filepath):
            # print("processing :",curr_dir+"\\"+files)
            file_to_process=self.filepath+"\\"+files
            dataframes[f"df_{counter}"]=pd.read_csv(file_to_process)
            counter=counter+1
        
        #Converting the dtype of month and lease_commence_date to datetime dtype
        for name, df in dataframes.items():
            df['month'] = pd.to_datetime(df['month'], format='%Y-%m')
            df['lease_commence_date'] = pd.to_datetime(df['lease_commence_date'], format='%Y')
        
        # Create the feature remaining_lease in df_1,df_2 and df_5 as present in df_3 & df_4
        for keys in dataframes.keys():
            if keys not in ['df_3','df_4']:
                dataframes[keys]['remaining_lease']= 100 - (abs(dataframes[keys]['month'].dt.year - dataframes[keys]['lease_commence_date'].dt.year) + 1) 
        
        #Rearranging the columns in df_1,df_2,df_5 like df_4 and df_4
        desired_column_order=['month', 'town', 'flat_type', 'block', 'street_name', 'storey_range',
                              'floor_area_sqm', 'flat_model', 'lease_commence_date',
                              'remaining_lease', 'resale_price']

        dataframes['df_1']=dataframes['df_1'][desired_column_order]
        dataframes['df_2']=dataframes['df_2'][desired_column_order]
        dataframes['df_5']=dataframes['df_5'][desired_column_order]

        # List specifying the desired order of dataframes
        order = ['df_1', 'df_2', 'df_5', 'df_3', 'df_4']

        # Stack the dataframes in the specified order
        final_df = pd.concat([dataframes[df_name] for df_name in order], ignore_index=True)

        ## Preprocessing the flat_type feature
        final_df['flat_type'] = final_df['flat_type'].replace('MULTI-GENERATION', 'MULTI GENERATION')

        # Preprocessing the flat_model feature - converting all the values to lower case 
        final_df['flat_model']=final_df['flat_model'].apply(lambda x : x.lower())

        #getting only the year part from remaining_lease feature 
        final_df['remaining_lease']=final_df['remaining_lease'].apply(lambda x: x.split()[0] if isinstance(x, str) else x)
        #converting the feature to integer
        final_df['remaining_lease']=final_df['remaining_lease'].astype(int)
        
        #Duplicate check 
        # print("No. of duplicates present in dataset :", final_df.duplicated().sum())
        #Dropping the duplicate records in the dataset 
        final_df.drop_duplicates(inplace=True)
        #Checking for duplicates after dropping 
        # print("No. of duplicates present in dataset after dropping them :", final_df.duplicated().sum())
        
        # Choosing only the features needed for model building
        final_df=final_df[['month','town','flat_type','block','storey_range','floor_area_sqm','remaining_lease','resale_price']]
        
        #Creating a new feature year from the month feature
        final_df['year']=final_df['month'].dt.year
       
        # Perform target guided mean encoding
        # Assuming X_train is your training dataframe and contains 'resale_price', 'year', 'town', 'block', and 'storey_range' columns

        # Step 1: Calculate the mean resale price for each (year, feature) combination
        for feature in ['town', 'block', 'storey_range']:
            self.mean_resale_price_train[feature] = final_df.groupby(['year', feature])['resale_price'].mean().reset_index()
            self.mean_resale_price_train[feature].rename(columns={'resale_price': f'{feature}_mean_resale_price'}, inplace=True)
            self.overall_mean_train[feature] = final_df.groupby([feature])['resale_price'].mean().reset_index()
            self.overall_mean_train[feature].rename(columns={'resale_price': f'{feature}_overall_mean_resale_price'}, inplace=True)
        
        # Step 2: Merge the mean resale prices back to the training data
        for features in ['town', 'block', 'storey_range']:
            final_df = final_df.merge(self.mean_resale_price_train[features], on=['year', features], how='left')
        
        ## Copying the features back to original Feature 
        for features in ['town', 'block', 'storey_range']:
            final_df[features]=final_df[f'{features}_mean_resale_price']
        
        #Label_encoding the flat_type feature alone 
        for features in ['flat_type']:
            # if features!='street_name':
            lb_encoder=LabelEncoder()
            final_df.loc[:,features] = lb_encoder.fit_transform(final_df[features])
            self.label_encoders[features]=lb_encoder
        
        #Dropping all the unecessary columns 
        final_df.drop(['town_mean_resale_price','block_mean_resale_price','storey_range_mean_resale_price','month','year'],axis=1,inplace=True)

        # Apply the Box-Cox Transformation on training data and save the lambda values
        for feature in ['floor_area_sqm', 'remaining_lease','town','block','storey_range']:
            final_df[feature], self.boxcox_params[feature] = boxcox(final_df[feature] + 1)

        #Apply Standard Scalar 
        final_df.loc[:, ['floor_area_sqm', 'remaining_lease','town','block','storey_range']] = self.scaler.fit_transform(final_df[['floor_area_sqm', 'remaining_lease','town','block','storey_range']])
        
        #Convert the dtype of flat_type to int 
        final_df['flat_type'] = final_df['flat_type'].astype(int)

        self.preprocessed_df = final_df

        print("The shape of preprocessed dataframe :",final_df.shape)
        print("The columns in preprocessed dataframe :",final_df.columns)
        print("The dtypes of all the columns are :",final_df.dtypes)
        print("The unique values in the flat_type are :",final_df['flat_type'].unique())
        print(final_df.head(1))
        
    def build_model(self):
        # 2. Split the dataframe into train and test
        X = self.preprocessed_df.drop('resale_price', axis=1)
        y = self.preprocessed_df['resale_price']
        y = y.squeeze()
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
        
        # 3. Build the Random Forest model
        self.rf_model.fit(X, y)
        # self.rf_model.fit(X_train, y_train)
    
    def pickle_dump(self, obj, filename):
        with open(filename, 'wb') as file:
            pickle.dump(obj, file)
    
    def save_objects(self,folder_path):
        # Ensure the folder exists, if not, create it
        os.makedirs(folder_path, exist_ok=True)

        self.pickle_dump(self.label_encoders, os.path.join(folder_path,'label_encoders.pkl'))
        self.pickle_dump(self.scaler, os.path.join(folder_path,'scaler.pkl'))
        self.pickle_dump(self.boxcox_params, os.path.join(folder_path,'boxcox_params.pkl'))
        self.pickle_dump(self.rf_model, os.path.join(folder_path,'model.pkl'))
        self.pickle_dump(self.mean_resale_price_train, os.path.join(folder_path,'mean_resale_price_train.pkl'))
        self.pickle_dump(self.overall_mean_train, os.path.join(folder_path,'overall_mean_train.pkl'))



if __name__ == "__main__":
    train_pipeline = TrainingPipeline('d:\\Saravanesh Personal\\Guvi\\Capstone Projects\\Singapore Flat Price\\Data')
    print("Starting the Preprocess workflow")
    train_pipeline.preprocess()
    print("Preprocess workflow - completed")
    print("Starting the model building workflow")
    train_pipeline.build_model()
    print("model building workflow-completed")

    # Specify the folder path where you want to save the pickle files
    pickle_path = 'D:\\Saravanesh Personal\\Guvi\\Capstone Projects\\Singapore Flat Price\\regression\\reg_pickle_files'
    print("Starting Pickling workflow")
    train_pipeline.save_objects(pickle_path)
    print("Pickling workflow completed")