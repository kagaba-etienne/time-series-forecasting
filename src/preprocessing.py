import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict

class TrafficDataProcessor:
    """
    Handles preprocessing, scaling, and tf.data.Dataset generation 
    for mobile network traffic forecasting.
    """
    def __init__(self, sequence_length: int = 3):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts deterministic seasonal features from the datetime index."""
        df = df.copy()
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        return df

    def _generate_fourier_terms(self, df: pd.DataFrame, period: int = 144, harmonics: int = 2) -> pd.DataFrame:
        """Generates Fourier terms to capture complex seasonal patterns as exogenous variables."""
        df = df.copy()
        t = np.arange(len(df))
        for k in range(1, harmonics + 1):
            df[f'sin_k{k}'] = np.sin(2 * np.pi * k * t / period)
            df[f'cos_k{k}'] = np.cos(2 * np.pi * k * t / period)
        return df

    def get_arimax_data(self, df: pd.DataFrame, square_id: int) -> Tuple[pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Returns the Train/Test splits required for the ARIMAX statistical baseline,
        including the endogenous target series and exogenous Fourier features.
        """
        df_sq = df[df['square_id'] == square_id].copy()
        df_sq = df_sq.set_index('datetime').sort_index()
        
        df_sq = self._generate_fourier_terms(df_sq, period=144, harmonics=3)
        
        target = df_sq['internet']
        exog = df_sq[[col for col in df_sq.columns if col.startswith('sin_') or col.startswith('cos_')]]
        
        y_train = target.loc[:'2013-12-15']
        y_test = target.loc['2013-12-16':'2013-12-22']
        exog_train = exog.loc[:'2013-12-15']
        exog_test = exog.loc['2013-12-16':'2013-12-22']
        
        return y_train, exog_train, y_test, exog_test

    def get_tf_datasets(self, df: pd.DataFrame, square_id: int, batch_size: int = 64) -> Dict[str, tf.data.Dataset]:
        """
        Generates scaled tf.data.Datasets using sliding windows for deep learning models.
        """
        df_sq = df[df['square_id'] == square_id].copy()
        df_sq = df_sq.set_index('datetime').sort_index()
        df_sq = self._engineer_features(df_sq)
        
        train_df = df_sq.loc[:'2013-12-15']
        test_df = df_sq.loc['2013-12-16':'2013-12-22']
        
        train_scaled = self.scaler.fit_transform(train_df[['internet', 'hour', 'day_of_week']])
        test_scaled = self.scaler.transform(test_df[['internet', 'hour', 'day_of_week']])
        
        y_train = train_scaled[:, 0]
        y_test = test_scaled[:, 0]
        
        train_dataset = tf.keras.utils.timeseries_dataset_from_array(
            data=train_scaled,
            targets=y_train[self.sequence_length:],
            sequence_length=self.sequence_length,
            batch_size=batch_size,
            shuffle=True
        )
        
        test_dataset = tf.keras.utils.timeseries_dataset_from_array(
            data=test_scaled,
            targets=y_test[self.sequence_length:],
            sequence_length=self.sequence_length,
            batch_size=batch_size,
            shuffle=False
        )
        
        return {
            'train': train_dataset,
            'test': test_dataset,
            'scaler': self.scaler
        }