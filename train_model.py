# train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

def load_data(file_path):
    """Load and prepare the data."""
    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path, delimiter=';')
    
    # Parse datetime
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['Hour'] = df['DateTime'].dt.hour
    df['DayOfWeek'] = df['DateTime'].dt.dayofweek
    return df

def prepare_features(df):
    """Prepare features for training."""
    features = ['Hour', 'DayOfWeek', 'Global_intensity', 'Global_reactive_power', 'Voltage']
    target = 'Global_active_power'
    return df[features], df[target]

def train_model():
    """Train the Random Forest model."""
    # Load data
    data = load_data('household_power_consumption_clean_sample.csv')
    
    # Prepare features
    print("Preparing features...")
    X, y = prepare_features(data)
    
    # Split data
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    print("Training Random Forest model...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Save model
    print("Saving model...")
    joblib.dump(rf_model, 'household_rf_model.joblib')
    
    # Save test data for evaluation
    test_data = pd.concat([X_test, y_test], axis=1)
    test_data.to_csv('predictions_sample.csv', index=False)
    
    return rf_model, X_test, y_test

if __name__ == "__main__":
    model, X_test, y_test = train_model()
    print("Model training completed and saved as 'household_rf_model.joblib'")