# Household Power Consumption Prediction

This project implements a machine learning model to predict household electric power consumption based on historical data. The model uses features like time of day, day of week, and other power-related measurements to predict the Global Active Power consumption.

## Project Structure

```
├── train_model.py              # Script for training the Random Forest model
├── evaluate_model.py           # Script for model evaluation
├── visualize_model.py          # Script for creating visualizations
├── requirements.txt            # Project dependencies
├── household_rf_model.joblib   # Trained model file (not in repository)
├── predictions_sample.csv      # Sample predictions for evaluation
└── household_power_consumption_clean_sample.csv  # Cleaned training data
```

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/Puneetprajapat/WEEK-2.git
cd WEEK-2
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Training the Model

Run the training script to create and save the model:

```bash
python train_model.py
```

This will:
- Load and preprocess the data
- Train a Random Forest model
- Save the model as 'household_rf_model.joblib'
- Create a test dataset in 'predictions_sample.csv'

### 2. Evaluating the Model

To evaluate the model's performance:

```bash
python evaluate_model.py
```

This will output:
- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared Score (R²)

### 3. Visualizing Results

Generate visualizations of the model's performance:

```bash
python visualize_model.py
```

This creates three visualization files:
- `feature_importances.png`: Bar plot of feature importance
- `predicted_vs_actual.png`: Scatter plot of predicted vs actual values
- `residual_hist.png`: Histogram of prediction residuals

## Features

The model uses the following features:
- Hour of day
- Day of week
- Global intensity
- Global reactive power
- Voltage

Target variable:
- Global active power

## Model Details

The project uses a Random Forest Regressor with the following characteristics:
- 100 trees (n_estimators=100)
- Standard train-test split (80% train, 20% test)
- Random state fixed at 42 for reproducibility

## Files Description

- `train_model.py`: Handles data loading, preprocessing, model training, and saving
- `evaluate_model.py`: Loads the trained model and test data to compute performance metrics
- `visualize_model.py`: Creates visualizations for model analysis and feature importance
- `requirements.txt`: Lists all Python dependencies
- `household_rf_model.joblib`: Serialized trained model (not included in repository)
- `predictions_sample.csv`: Sample of test data with predictions
- `household_power_consumption_clean_sample.csv`: Cleaned version of the dataset

## Dependencies

- pandas >= 1.5.0
- numpy >= 1.24.0
- scikit-learn >= 1.2.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0
- joblib >= 1.2.0

## Notes

- The model file (`household_rf_model.joblib`) is not included in the repository due to size constraints
- The full dataset is not included; only a cleaned sample is provided
- All scripts include progress messages to track execution

## Contributing

Feel free to submit issues and enhancement requests!
