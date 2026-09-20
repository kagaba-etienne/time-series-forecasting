import os
import time

# Suppress TensorFlow logging spam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Import our modular components
from preprocessing import TrafficDataProcessor
from models import build_lstm_model, build_transformer_model, build_arimax_model

# Get the absolute paths to the root of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- HELPER FUNCTIONS ---

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculates evaluation metrics required by the rubric."""
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred)
    }

def plot_forecast(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, square_id: int, dates: pd.DatetimeIndex):
    """Generates the superposed time series plots (Deliverable II)."""
    plt.figure(figsize=(12, 5))
    plt.plot(dates, y_true, label='Actual Traffic', color='blue', alpha=0.7)
    plt.plot(dates, y_pred, label=f'{model_name} Prediction', color='red', alpha=0.7)
    plt.title(f'Actual vs Predicted Internet Traffic - {model_name} (Square {square_id})')
    plt.xlabel('Date (Dec 16 - Dec 22)')
    plt.ylabel('Internet Traffic')
    plt.legend()
    plt.tight_layout()
    
    # Save the plot automatically
    os.makedirs(os.path.join(BASE_DIR, 'reports', 'figures'), exist_ok=True)
    plt.savefig(os.path.join(BASE_DIR, 'reports', 'figures', f'{model_name}_square_{square_id}.png'))
    plt.close()

def inverse_transform_predictions(predictions: np.ndarray, scaler) -> np.ndarray:
    """
    Because the scaler was fitted on 3 columns (internet, hour, day), 
    we must reconstruct a dummy array to inverse_transform just the internet predictions.
    """
    dummy = np.zeros((len(predictions), 3))
    dummy[:, 0] = predictions.flatten()
    return scaler.inverse_transform(dummy)[:, 0]

# --- MAIN EXECUTION PIPELINE ---

def main():
    # 1. Load the processed dataset
    print("Loading optimized dataset...")
    df = pd.read_parquet(os.path.join(BASE_DIR, 'data', 'processed', 'milan_internet_traffic.parquet'))
    
    top_3_squares = [5161, 5059, 5259]
    sequence_length = 6 # 1-hour lookback
    
    # Dictionaries to store our final metric tables
    results_tables = {sq: [] for sq in top_3_squares}

    for square_id in top_3_squares:
        print(f"\n{'='*40}")
        print(f"BEGINNING EXPERIMENTS FOR SQUARE: {square_id}")
        print(f"{'='*40}")
        
        processor = TrafficDataProcessor(sequence_length=sequence_length)
        
        # --- MODEL 1: ARIMAX (Statistical Baseline) ---
        print("\n--- Training ARIMAX ---")
        y_train, exog_train, y_test, exog_test = processor.get_arimax_data(df, square_id)
        
        # Apply log transformation to the training target
        y_train_log = np.log1p(y_train)
        
        # Initialize and fit the model on the LOG TRANSFORMED data
        arimax_model = build_arimax_model(y_train_log, exog_train, order=(3,0,0))
        
        start_train = time.time()
        arimax_results = arimax_model.fit(disp=False)
        train_time_arimax = time.time() - start_train
        
        start_exec = time.time()
        arimax_preds_log = arimax_results.forecast(steps=len(y_test), exog=exog_test)
        exec_time_arimax = time.time() - start_exec
        
        # Exponentiate the predictions to return them to the original scale
        arimax_preds = np.expm1(arimax_preds_log)
        arimax_preds.index = y_test.index
        
        metrics_arimax = calculate_metrics(y_test.values, arimax_preds.values)
        plot_forecast(y_test.values, arimax_preds.values, "ARIMAX", square_id, y_test.index)
        
        results_tables[square_id].append({
            'Model': 'ARIMAX', 
            'Train Time (s)': round(train_time_arimax, 2), 
            'Exec Time (s)': round(exec_time_arimax, 4), 
            **metrics_arimax
        })
        
        # --- DEEP LEARNING DATA PREP ---
        tf_data = processor.get_tf_datasets(df, square_id)
        train_ds, test_ds, scaler = tf_data['train'], tf_data['test'], tf_data['scaler']
        
        # Extract the true Y values from the test dataset for metric calculation
        y_test_true_scaled = np.concatenate([y for x, y in test_ds], axis=0)
        y_test_true = inverse_transform_predictions(y_test_true_scaled, scaler)
        
        # We need the dates corresponding to the test predictions for plotting
        test_dates = y_test.index[sequence_length:]
        
        # --- MODEL 2: LSTM ---
        print("\n--- Training LSTM ---")
        lstm = build_lstm_model(input_shape=(sequence_length, 3))
        
        start_train = time.time()
        lstm.fit(train_ds, epochs=15, verbose=1)
        train_time_lstm = time.time() - start_train
        
        start_exec = time.time()
        lstm_preds_scaled = lstm.predict(test_ds, verbose=0)
        exec_time_lstm = time.time() - start_exec
        
        lstm_preds = inverse_transform_predictions(lstm_preds_scaled, scaler)
        metrics_lstm = calculate_metrics(y_test_true, lstm_preds)
        plot_forecast(y_test_true, lstm_preds, "LSTM", square_id, test_dates)
        
        results_tables[square_id].append({
            'Model': 'LSTM',
            'Train Time (s)': round(train_time_lstm, 2),
            'Exec Time (s)': round(exec_time_lstm, 4),
            **metrics_lstm
        })
        
        # --- MODEL 3: TRANSFORMER ---
        print("\n--- Training Transformer ---")
        transformer = build_transformer_model(input_shape=(sequence_length, 3))
        
        start_train = time.time()
        transformer.fit(train_ds, epochs=15, verbose=1)
        train_time_trans = time.time() - start_train
        
        start_exec = time.time()
        trans_preds_scaled = transformer.predict(test_ds, verbose=0)
        exec_time_trans = time.time() - start_exec
        
        trans_preds = inverse_transform_predictions(trans_preds_scaled, scaler)
        metrics_trans = calculate_metrics(y_test_true, trans_preds)
        plot_forecast(y_test_true, trans_preds, "Transformer", square_id, test_dates)
        
        results_tables[square_id].append({
            'Model': 'Transformer',
            'Train Time (s)': round(train_time_trans, 2),
            'Exec Time (s)': round(exec_time_trans, 4),
            **metrics_trans
        })

    # --- FINAL OUTPUT ---
    print("\n\n" + "="*50)
    print("FINAL PERFORMANCE TABLES")
    print("="*50)
    for square_id in top_3_squares:
        print(f"\nPerformance for Square ID: {square_id}")
        df_results = pd.DataFrame(results_tables[square_id])
        print(df_results.to_markdown(index=False))

if __name__ == "__main__":
    main()