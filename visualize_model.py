# visualize_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def plot_feature_importance(model, feature_names):
    """Plot feature importance from the Random Forest model."""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance')
    plt.bar(range(len(importance)), importance[indices])
    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.savefig('feature_importances.png')
    plt.close()

def plot_predictions_vs_actual(y_true, y_pred):
    """Create scatter plot of predicted vs actual values."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Global Active Power')
    plt.ylabel('Predicted Global Active Power')
    plt.title('Predicted vs Actual Values')
    plt.tight_layout()
    plt.savefig('predicted_vs_actual.png')
    plt.close()

def plot_residuals(y_true, y_pred):
    """Plot histogram of residuals."""
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50)
    plt.xlabel('Residual Value')
    plt.ylabel('Count')
    plt.title('Histogram of Residuals')
    plt.tight_layout()
    plt.savefig('residual_hist.png')
    plt.close()

def visualize_model():
    """Create visualizations for model analysis."""
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
    model = joblib.load('household_rf_model.joblib')
    
    # Generate predictions
    print("Generating predictions...")
    y_pred = model.predict(X_test)
    
    # Create visualizations
    print("Creating visualizations...")
    plot_feature_importance(model, features)
    plot_predictions_vs_actual(y_test, y_pred)
    plot_residuals(y_test, y_pred)
    
    print("Visualizations have been saved as PNG files.")

if __name__ == "__main__":
    visualize_model()