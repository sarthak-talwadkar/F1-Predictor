import numpy as np
import pandas as pd
import fastf1
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import datetime as dt
import pickle
import os

class F1DataCollector:
    """

    """
    def __init__(self, cache_dir = "cache"):

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            print("Created Cache Directory")

        fastf1.Cache.enable_cache(cache_dir)
        self.years = [2021,2022,2023,2024]

    def get_race_calender(self, year):
        try :
            schedule = fastf1.get_event_schedule(year)
            return schedule
        except Exception as e :
            print(f"Race calender error for year {year}: {e}")
            return None

    def fetch_session_data(self, year, race_name, session_type):
        try:
            session = fastf1.get_session(year, race_name, session_type)
            session.load()
            return session
        except Exception as e :
            print(f"Session fetch error for {year} {race_name}: {e}")
            return None

    def get_driver_info(self, year):
        try:
            drivers = fastf1.get_driver_info(year)
            return drivers
        except Exception as e:
            print(f"Driver info error for year {year}: {e}")
            return None

    def collect_season_data(self, session_type = 'R'):
        try:
            all_race_data = []

            for year in self.years:
                schedule = self.get_race_calender(year)
                if schedule is  None:
                    continue

                for _,event in schedule.iterrows():
                    race_name = event['EventName']
                    print(f"Collecting {year}, {race_name}, {session_type} data .......")

                    session = self.fetch_session_data(year, race_name, session_type)
                    if session is None:
                        continue

                    laps = session.laps
                    laps = laps.copy()
                    laps['Year'] = year
                    laps['Race'] = race_name
                    laps['SessionType'] = session_type

                    try:
                        weather_data = session.weather_data
                        weather_data = weather_data.copy()
                        ##weather_data['Year'] = year
                        ##weather_data['Race'] = race_name
                        ##weather_data['SessionType'] = session_type

                        laps = pd.merge(laps, weather_data, on =['Time'], how = 'left')

                    except Exception as e:
                        print(f"No weather data found for year {year}: {race_name} - {e}")

                    all_race_data.append(laps)

            if all_race_data:
                return pd.concat(all_race_data, ignore_index=True)
            else:
                return None
        except Exception as e:
            print(f"Season info error : {e}")
            return None


class F1DataProcessor:
    """

    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def clean_data(self, df):
        if df is None or df.empty:
            return None

        df_clean = df.copy()

        # Check if required columns exist
        required_cols = ['LapTime', 'Driver', 'Position', 'Compound']
        missing_cols = [col for col in required_cols if col not in df_clean.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return None

        df_clean = df_clean.dropna(subset=required_cols)

        df_clean.loc[:, 'LapTimeSeconds'] = df_clean['LapTime'].dt.total_seconds()

        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_clean.loc[:,col] = df_clean[col].fillna(df_clean[col].median())

        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df_clean.loc[:, col] = df_clean[col].fillna('Unknown')

        return df_clean

    def encode_categorical_features(self, df, columns):
        if df is None or df.empty:
            return None

        df_encoded = df.copy()

        for col in columns:
            if col in df_encoded.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded.loc[:, col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
                else:
                    new_categories = set(df_encoded[col].astype(str)) - set(self.label_encoders[col].classes_)

                    if new_categories:
                        df_encoded.loc[:, col] = df_encoded[col].apply(
                            lambda x: 0 if str(x) in new_categories else self.label_encoders[col].transform([str(x)])[0])
                    else:
                        df_encoded.loc[:, col] = self.label_encoders[col].transform(df_encoded[col].astype(str))

        return df_encoded

    def create_features(self, df):
        if df is None or df.empty:
            return None

        df_featured = df.copy()

        # Create LapTimeSmooth feature (not overwrite LapTimeSeconds)
        df_featured.loc[:, 'LapTimeSmooth'] = df_featured.groupby('Driver')['LapTimeSeconds'].transform(
            lambda x : x.rolling(5, min_periods=1).mean()
        )

        # Tire age feature
        df_featured.loc[:, 'TireAge'] = df_featured.groupby(['Driver', 'Stint'])['LapNumber'].transform(
            lambda x: x - x.min() + 1
        )

        # Position change
        df_featured.loc[:, 'PositionChange'] = df_featured.groupby(['Driver', 'Race', 'Year'])['Position'].diff().fillna(0)

        # Track position features
        df_featured.loc[:, 'IsFrontRunner'] = (df_featured['Position'] <= 3).astype(int)
        df_featured.loc[:, 'IsMidField'] = ((df_featured['Position'] > 3) & (df_featured['Position'] <= 10)).astype(int)
        df_featured.loc[:, 'IsBackMarker'] = (df_featured['Position'] > 10).astype(int)

        # Pace differential to leader
        leader_pace = df_featured.groupby(['LapNumber', 'Race', 'Year'])['LapTimeSeconds'].transform('min')
        df_featured.loc[:, 'PaceToLeader'] = df_featured['LapTimeSeconds'] - leader_pace

        # Stint information
        df_featured.loc[:, 'StintLength'] = df_featured.groupby(['Driver', 'Race', 'Year', 'Stint'])['LapNumber'].transform('count')

        return df_featured

    def prepare_training_data(self, df, target_lap = 20):
        if df is None or df.empty:
            return None

        # Get the final positions for each driver in each race
        final_positions = df.groupby(['Year', 'Race', 'Driver'])['Position'].last().reset_index()
        final_positions.rename(columns={'Position': 'FinalPosition'}, inplace=True)  # Fixed syntax

        # Select data from the target lap
        mid_race_data = df[df['LapNumber'] == target_lap].copy()

        # Merge with final positions
        training_data = pd.merge(
            mid_race_data,
            final_positions,
            on=['Year', 'Race', 'Driver'],
            how= 'inner'
        )

        return training_data

class F1Model:
    """

    """
    def __init__(self, model_type = 'random_forest'):
        self.model_type = model_type
        self.model = None
        self.feature_columns = [
            'LapTimeSeconds', 'LapTimeSmooth', 'TireAge', 'Compound',
            'Position', 'PositionChange', 'IsFrontRunner', 'IsMidField',
            'IsBackMarker', 'PaceToLeader', 'Stint', 'StintLength'
        ]
        self.scaler = StandardScaler()

    def train(self, X, y, optimize_hyperparameters = False):
        X_scaled = self.scaler.fit_transform(X)

        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators = 100,
                max_depth = 10,
                random_state = 42,
                n_jobs = -1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators = 100,
                learning_rate = 0.1,
                max_depth = 5,
                random_state = 42
            )
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                objective = 'reg:squarederror',
                n_estimators = 100,
                max_depth = 6,
                learning_rate = 0.1,
                random_state = 42
            )
        else:
            raise ValueError(f"Unknown Model Type: {self.model_type}")

        if optimize_hyperparameters:
            self.optimize_hyperparameters(X_scaled, y)
        else:
            self.model.fit(X_scaled, y)

        return self.model

    def optimize_hyperparameters(self, X, y):
        if self.model_type == 'random_forest':
            param_grid = {
                'n_estimators' : [50,100,200],
                'max_depth' : [5,10,20],
                'min_samples_split' : [2,5,10]
            }
        elif self.model_type == 'gradient_boosting':
            param_grid = {
                'n_estimators' : [50,100,200],
                'max_depth' : [3,5,7],
                'learning_rate' : [0.01,0.1,0.2]
            }
        elif self.model_type == 'xgboost':
            param_grid = {
                'n_estimators' : [50,100,200],
                'max_depth' : [3,6,9],
                'learning_rate' : [0.01,0.1,0.2]
            }
        else:
            print("Hyperparameter optimization not implemented for this model type")
            self.model.fit(X, y)
            return

        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv = 5,
            scoring = 'neg_mean_absolute_error',
            n_jobs = -1
        )
        grid_search.fit(X, y)

        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Best Score: {-grid_search.best_score_}")

        self.model = grid_search.best_estimator_

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model is not trained yet")

        X_scaled = self.scaler.transform(X)

        return self.model.predict(X_scaled)

    def evaluate(self, X, y):
        predictions = self.predict(X)

        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))

        print(f"Model : {self.model_type}")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"Root Mean Squared Error: {rmse:.2f}")

        plt.figure(figsize=(10, 6))
        plt.scatter(y, predictions, alpha = 0.5)
        plt.plot([y.min(), y.max()],[y.min(), y.max()], 'r--')
        plt.xlabel('Actual Position')
        plt.ylabel('Predicted Position')
        plt.title(f"Model : {self.model_type}")
        plt.show()

        return mae, rmse

    def feature_importance(self, feature_names):
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]

            plt.figure(figsize=(12, 8))
            plt.title("Feature Importances")
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation = 45)
            plt.tight_layout()
            plt.show()

        else:
            print("No feature importances available")

def main():
    print("Starting F1 Prediction Pipeline")

    print("Collecting Data")
    collector = F1DataCollector(cache_dir = 'f1_cache')
    raw_data = collector.collect_season_data(session_type = 'R')

    if raw_data is None:
        print("Failed to collect data")
        return

    print("Processing Data")
    processor = F1DataProcessor()
    cleaned_data = processor.clean_data(raw_data)

    if cleaned_data is None:
        print("Data cleaning failed")
        return

    featured_data = processor.create_features(cleaned_data)

    if featured_data is None:
        print("Feature creation failed")
        return

    print("Preparing Training Data")
    training_data = processor.prepare_training_data(featured_data, target_lap = 20)

    if training_data is None or training_data.empty:
        print("Training data preparation failed")
        return

    feature_columns = [
            'LapTimeSeconds', 'LapTimeSmooth', 'TireAge', 'Compound',
            'Position', 'PositionChange', 'IsFrontRunner', 'IsMidField',
            'IsBackMarker', 'PaceToLeader', 'Stint', 'StintLength'
    ]

    # Check which features actually exist in our data
    available_features = [col for col in feature_columns if col in training_data.columns]
    print(f"Available Features : :{available_features}")

    training_data_encoded = processor.encode_categorical_features(training_data, ['Compound'])

    if training_data_encoded is None:
        print("Feature encoding failed")
        return

    X = training_data_encoded[available_features]
    y = training_data_encoded['FinalPosition']

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

    print("Training Model")
    models = {}
    model_types = ['random_forest', 'gradient_boosting', 'xgboost']

    for model_type in model_types:
        print(f"Training {model_type} Model")
        model = F1Model(model_type = model_type)
        model.train(X_train, y_train, optimize_hyperparameters = False)

        mae, rmse = model.evaluate(X_test, y_test)
        models[model_type] = {
            'model' : model,
            'mae' : mae,
            'rmse' : rmse
        }


    print("Selecting the Best Model....")
    best_model_type = min(models, key = lambda x : models[x]['mae'])
    best_model = models[best_model_type]['model']

    print(f"Best Model : {best_model_type} with MAE : {models[best_model_type]['mae']:.2f}")

    best_model.feature_importance(available_features)

    print("Saving the Model...")
    timestamp = dt.now().strftime("%Y_%m_%d_%H_%M_%S")
    model_filename = f"f1_prediction_model_{best_model_type}_{timestamp}.pkl"

    with open(model_filename, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"Model Saved as {model_filename}")

    print("Making Example Predictions")
    sample_idx = X_test.index[0]
    sample_data = X_test.loc[sample_idx: sample_idx]
    actual_position = y_test.loc[sample_idx]

    prediction = best_model.predict(sample_data)[0]

    print(f"Predicted Final Position: {prediction:.1f}")
    print(f"Actual Final Position: {actual_position}")
    print(f"Driver: {training_data_encoded.loc[sample_idx, 'Driver']}")
    print(f"Race: {training_data_encoded.loc[sample_idx, 'Race']} {training_data_encoded.loc[sample_idx, 'Year']}")  # Use 'Race' instead of 'EventName'

    print("Pipeline Completed Successfully.....")

if __name__ == "__main__":
    main()