import pandas as pd"""

import numpy as npTrain a simple regression model to predict Global_active_power using the cleaned sample CSV.

from sklearn.ensemble import RandomForestRegressorUsage:

from sklearn.model_selection import train_test_split    python train_model.py

import joblibThe script will:

 - load the trimmed sample CSV created earlier

# Load and prepare the data - parse Date+Time and engineer time features

def load_data(file_path): - split data into train/test

    df = pd.read_csv(file_path, delimiter=';') - train a RandomForestRegressor pipeline

    return df - print RMSE, MAE, R2

 - save the trained model to 'model.joblib'

def prepare_features(df):"""

    # Convert datetimeimport pandas as pd

    df['DateTime'] = pd.to_datetime(df['DateTime'])import numpy as np

    df['Hour'] = df['DateTime'].dt.hourfrom pathlib import Path

    df['DayOfWeek'] = df['DateTime'].dt.dayofweekfrom sklearn.model_selection import train_test_split

    from sklearn.ensemble import RandomForestRegressor

    # Select featuresfrom sklearn.pipeline import Pipeline

    features = ['Hour', 'DayOfWeek', 'Global_intensity', 'Global_reactive_power', 'Voltage']from sklearn.preprocessing import StandardScaler

    target = 'Global_active_power'from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    import joblib

    return df[features], df[target]import argparse



def train_model():BASE = Path(__file__).resolve().parent

    # Load data# Prefer trimmed sample if present, otherwise fall back to default sample filename

    data = load_data('household_power_consumption_clean_sample.csv')possible = [

        'household_power_consumption_clean_sample.csv',

    # Prepare features]

    X, y = prepare_features(data)for fname in possible:

        path = BASE / fname

    # Split data    if path.exists():

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)        CSV = path

            break

    # Train modelelse:

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)    CSV = BASE / 'household_power_consumption_clean_sample.csv'

    rf_model.fit(X_train, y_train)MODEL_OUT = BASE / 'household_rf_model.joblib'

    

    # Save modelparser = argparse.ArgumentParser()

    joblib.dump(rf_model, 'household_rf_model.joblib')parser.add_argument('--max-rows', type=int, default=100000, help='Maximum number of rows to use for training (default 100000). Use 0 or negative for all rows.')

    parser.add_argument('--n-estimators', type=int, default=50, help='Number of trees in RandomForest (default 50).')

    return rf_model, X_test, y_testargs = parser.parse_args()



if __name__ == "__main__":print('Loading', CSV)

    model, X_test, y_test = train_model()df = pd.read_csv(CSV)

    print("Model training completed and saved as 'household_rf_model.joblib'")print('Original rows:', len(df))

# Optionally subsample to keep training quick
if args.max_rows and args.max_rows > 0 and len(df) > args.max_rows:
    df = df.sample(n=args.max_rows, random_state=42).sort_index()
    print('Subsampled to rows:', len(df))
print('rows,cols', df.shape)

# Parse datetime
if 'Datetime' in df.columns:
    df['dt'] = pd.to_datetime(df['Datetime'], dayfirst=True, errors='coerce')
else:
    df['dt'] = pd.to_datetime(df['Date'].str.strip() + ' ' + df['Time'].str.strip(), dayfirst=True, errors='coerce')

# Feature engineering: hour, minute, second, weekday
df['hour'] = df['dt'].dt.hour
df['minute'] = df['dt'].dt.minute
df['second'] = df['dt'].dt.second
df['weekday'] = df['dt'].dt.weekday

# Select features and target
features = ['Global_reactive_power', 'Voltage', 'Global_intensity',
            'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
            'hour', 'minute', 'weekday']
# Ensure all features exist
features = [f for f in features if f in df.columns]
print('Using features:', features)

target = 'Global_active_power'

# Drop rows with missing dt or target
df = df.dropna(subset=[target, 'dt'])

X = df[features]
y = df[target]

# Quick train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Train shape:', X_train.shape, 'Test shape:', X_test.shape)

# Pipeline: scaler + RandomForest
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(n_estimators=args.n_estimators, n_jobs=-1, random_state=42))
])

print('Training RandomForest...')
pipe.fit(X_train, y_train)

print('Predicting...')
y_pred = pipe.predict(X_test)

rmse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(rmse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('\nEvaluation Metrics:')
print(f'RMSE: {rmse:.4f}')
print(f'MAE:  {mae:.4f}')
print(f'R2:   {r2:.4f}')

# Save model
joblib.dump(pipe, MODEL_OUT)
print(f'Model saved to: {MODEL_OUT}')

# Show sample predictions
sample = X_test.copy().head(10)
sample['y_true'] = y_test.head(10).values
sample['y_pred'] = pipe.predict(sample[features])
print('\nSample predictions:')
print(sample[['y_true', 'y_pred']])
