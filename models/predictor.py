import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

class CropProductionPredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.data = None
        self.target = None
        self.categorical_features = ['State_Name', 'District_Name', 'Season', 'Crop']
        self.numerical_features = ['Area']
        self.model_type = None
        
        # State to districts mapping - this will be populated from the data
        self.state_to_districts = {}
        self.crops = []
        self.seasons = []

    def load_data(self, filepath):
        self.data = pd.read_csv(filepath)
        print(f"Data loaded successfully with shape: {self.data.shape}")
        
        # Populate state_to_districts mapping
        for state in self.data['State_Name'].unique():
            self.state_to_districts[state] = sorted(self.data[self.data['State_Name'] == state]['District_Name'].unique().tolist())
        
        # Get unique crops and seasons
        self.crops = sorted(self.data['Crop'].unique().tolist())
        self.seasons = sorted(self.data['Season'].unique().tolist())
        
        return self.data.head()

    def preprocess_data(self):
        if self.data.isnull().sum().sum() > 0:
            print("Handling missing values...")
            self.data = self.data.dropna()

        X = self.data[self.categorical_features + self.numerical_features]
        self.target = self.data['Production']

        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        numerical_transformer = StandardScaler()

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, self.categorical_features),
                ('num', numerical_transformer, self.numerical_features)
            ])

        self.preprocessor.fit(X)
        return X, self.target

    def train_model(self, model_type='knn', n_neighbors=5):
        X, y = self.preprocess_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model_type = model_type

        if model_type.lower() == 'knn':
            self.model = Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', KNeighborsRegressor(n_neighbors=n_neighbors))
            ])
        else:
            self.model = Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', LinearRegression())
            ])

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Model trained using {model_type.upper()}")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R² Score: {r2:.2f}")

        return self.model

    def predict_production(self, state_name, district_name, season, crop, area):
        if self.model is None:
            raise Exception("Model not trained. Please train the model first.")

        input_data = pd.DataFrame({
            'State_Name': [state_name],
            'District_Name': [district_name],
            'Season': [season],
            'Crop': [crop],
            'Area': [float(area)]
        })

        prediction = self.model.predict(input_data)[0]
        return prediction

    def save_model(self, filename='data/crop_production_model.pkl'):
        if self.model is None:
            raise Exception("No model to save. Please train the model first.")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        joblib.dump({
            'model': self.model,
            'state_to_districts': self.state_to_districts,
            'crops': self.crops,
            'seasons': self.seasons
        }, filename)
        
        print(f"Model saved as {filename}")

    def load_model(self, filename='data/crop_production_model.pkl'):
        loaded = joblib.load(filename)
        self.model = loaded['model']
        
        # Load the reference data if available
        if 'state_to_districts' in loaded:
            self.state_to_districts = loaded['state_to_districts']
        if 'crops' in loaded:
            self.crops = loaded['crops']
        if 'seasons' in loaded:
            self.seasons = loaded['seasons']
            
        print(f"Model loaded from {filename}")
    
    # Helper methods for the web interface
    def get_states(self):
        return sorted(list(self.state_to_districts.keys()))
    
    def get_districts_for_state(self, state):
        return self.state_to_districts.get(state, [])
    
    def get_crops(self):
        return self.crops
    
    def get_seasons(self):
        return self.seasons