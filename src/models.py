import tensorflow as tf
from keras import layers, Model
import statsmodels.api as sm
import pandas as pd
import warnings

def build_arimax_model(
    train_series: pd.Series, 
    exog_train: pd.DataFrame,
    order: tuple = (3, 0, 0)
):
    """
    Initializes an ARIMAX model with exogenous features (Fourier terms) without fitting it.
    
    Args:
        train_series: 1D Pandas Series of training data
        exog_train: Pandas DataFrame of exogenous regressor variables
        order: (p, d, q) parameters
        
    Returns:
        An unfitted statsmodels SARIMAX object operating as an ARIMAX model.
    """
    warnings.filterwarnings("ignore")
    
    model = sm.tsa.statespace.SARIMAX(
        endog=train_series,
        exog=exog_train,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    return model


def build_lstm_model(input_shape: tuple, units: int = 64, learning_rate: float = 0.001) -> Model:
    """
    Builds and compiles a standard LSTM network for time series forecasting.
    
    Args:
        input_shape: Tuple of (sequence_length, num_features)
        units: Number of LSTM units in the hidden layer
        learning_rate: Learning rate for the Adam optimizer
    """
    inputs = layers.Input(shape=input_shape)
    
    x = layers.LSTM(units, return_sequences=False, activation='tanh')(inputs)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1)(x)
    
    model = Model(inputs, outputs, name="LSTM_Forecaster")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    return model


def build_transformer_model(
    input_shape: tuple,
    head_size: int = 64,
    num_heads: int = 4,
    ff_dim: int = 64,
    num_transformer_blocks: int = 1,
    mlp_units: list = [32],
    dropout: float = 0.2,
    learning_rate: float = 0.001
) -> Model:
    """
    Builds and compiles a Time Series Transformer model.
    """
    inputs = layers.Input(shape=input_shape)
    x = inputs

    for _ in range(num_transformer_blocks):
        attention_output = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(x, x)
        
        x = layers.LayerNormalization(epsilon=1e-6)(x + attention_output)
        
        ffn_output = layers.Dense(ff_dim, activation="relu")(x)
        ffn_output = layers.Dropout(dropout)(ffn_output)
        ffn_output = layers.Dense(input_shape[-1])(ffn_output)
        
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)

    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)

    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        
    outputs = layers.Dense(1)(x)
    
    model = Model(inputs, outputs, name="Transformer_Forecaster")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse', 
        metrics=['mae']
    )
    return model