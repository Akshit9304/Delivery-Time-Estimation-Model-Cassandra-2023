# Import necessary libraries
from google.colab import drive
from typing import Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn import metrics
import pandas as pd
import numpy as np

# Function to load data
def load_data(file_path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    data = pd.read_csv(file_path)
    return data

# Function to preprocess data
def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data for delivery time estimation."""
    # Handle missing values
    data['trip_creation_time'].fillna(data['od_start_time'], inplace=True)

    # Convert columns to datetime format
    data['od_start_time'] = pd.to_datetime(data['od_start_time'])
    data['od_end_time'] = pd.to_datetime(data['od_end_time'])
    data['trip_creation_time'] = pd.to_datetime(data['trip_creation_time'])

    # Calculate additional features based on datetime columns
    data['total_od_time'] = (data['od_end_time'] - data['od_start_time']).dt.total_seconds()
    data['creation_to_start'] = (data['od_start_time'] - data['trip_creation_time']).dt.total_seconds()
    data['creation_to_end'] = (data['od_end_time'] - data['trip_creation_time']).dt.total_seconds()

    # Data preprocessing
    data.dropna(subset=['cutoff_timestamp', 'destination_center'], inplace=True)

    # Encode categorical features
    categorical_cols = ['route_type', 'is_cutoff', 'source_center', 'destination_center',
                         'route_schedule_uuid', 'source_name', 'destination_name']
    label_encoder = LabelEncoder()
    for col in categorical_cols:
        data[col] = label_encoder.fit_transform(data[col])

    # Use forward fill to handle missing values
    data.fillna(method='pad', inplace=True)

    return data

# Function to split data
def split_data(data: pd.DataFrame, target_col: str = "actual_time", test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """Split data into features and target variable, and further split into training and testing sets."""
    X = data.drop(columns=target_col, axis=1)
    Y = data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# Function to train models
def train_models(X_train: pd.DataFrame, Y_train: pd.Series) -> Tuple:
    """Train machine learning models."""
    # Perform Scaling
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)

    # Initialize models
    rf_reg = RandomForestRegressor(random_state=42)
    cat_reg = CatBoostRegressor(random_state=42, verbose=0)

    # Create an ensemble of the models
    ensemble = [rf_reg, cat_reg]
    for model in ensemble:
        model.fit(X_train_scaled, Y_train)

    return ensemble

# Function to evaluate models
def evaluate_models(models: list, X_test: pd.DataFrame, Y_test: pd.Series) -> None:
    """Evaluate machine learning models."""
    # Make predictions using the ensemble
    ensemble_pred = [model.predict(X_test) for model in models]
    ensemble_pred = sum(ensemble_pred) / len(models)
    r2_test = metrics.r2_score(Y_test, ensemble_pred)
    print("R Squared Value on Test Data =", r2_test)

# Function to make predictions on new data
def make_predictions(models: list, new_data: pd.DataFrame, output_file: str = 'result.csv') -> None:
    """Make predictions on new data and save to a CSV file."""
    sc = StandardScaler()
    new_data_scaled = sc.fit_transform(new_data)
    predictions = [model.predict(new_data_scaled) for model in models]
    ensemble_pred = sum(predictions) / len(models)

    UID = new_data["UID"]
    main_prediction = pd.DataFrame({'UID': UID, 'actual_time': ensemble_pred})
    main_prediction.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

# Main function
def main():
    # Load and preprocess the training data
    train_data = load_data("/content/drive/MyDrive/cassandra2023ps2/train_data.csv")
    train_data = preprocess_data(train_data)

    # Split the training data
    X_train, X_test, y_train, y_test = split_data(train_data)

    # Train machine learning models
    trained_models = train_models(X_train, y_train)

    # Evaluate models on the test set
    evaluate_models(trained_models, X_test, y_test)

    # Load and preprocess the test data
    test_data = load_data("/content/drive/MyDrive/cassandra_ps2_test_data.csv")
    test_data = preprocess_data(test_data)

    # Make predictions on the test data
    make_predictions(trained_models, test_data)

if __name__ == "__main__":
    main()
