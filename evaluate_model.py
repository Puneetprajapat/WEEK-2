# evaluate_model.py
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

def load_model(model_path):
    """Load the trained model."""
    return joblib.load(model_path)

def evaluate_predictions(y_true, y_pred):
    """Calculate and return evaluation metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

def evaluate_model():
    """Evaluate the model on test data."""
    # Load test data
    print("Loading test data...")
    test_data = pd.read_csv('predictions_sample.csv')
    
    # Separate features and target
    target = 'Global_active_power'
    features = [col for col in test_data.columns if col != target]
    X_test = test_data[features]
    y_test = test_data[target]
    
    # Load model
    print("Loading model...")
    model = load_model('household_rf_model.joblib')
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse, mae, r2 = evaluate_predictions(y_test, y_pred)
    
    # Print results
    print("\nModel Evaluation Results:")
    print(f"Root Mean Square Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R-squared Score: {r2:.4f}")
    
    return rmse, mae, r2

if __name__ == "__main__":
    evaluate_model()