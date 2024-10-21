import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from typing import Tuple
from config import ModelConfig

def load_data(filepath: str, config: ModelConfig) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {filepath} was not found.")
    
    required_columns = config.feature_columns + ['labels_full']
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    return df

def preprocess_data(df: pd.DataFrame, config: ModelConfig) -> Tuple[np.ndarray, ...]:
    if df[config.feature_columns].isna().any().any():
        raise ValueError("NaN values found in feature columns. Please clean your data.")

    X = df[config.feature_columns].values

    invalid_labels = set(df['labels_full'].unique()) - set(config.label_mapping.keys())
    if invalid_labels:
        raise ValueError(f"Invalid labels found: {invalid_labels}. Please update `label_mapping` in config.")

    y = df['labels_full'].map(config.label_mapping).values

    if np.isnan(y).any():
        raise ValueError("NaN values found after mapping labels. Check your `label_mapping` or dataset.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    try:
        X_seq = np.array([
            X_scaled[i:i + config.window_size]
            for i in range(0, len(X_scaled) - config.window_size + 1, config.step_size)
        ])
        y_seq = y[config.window_size - 1::config.step_size]
    except ValueError as e:
        raise ValueError(f"Error during windowing: {e}")

    if np.isnan(X_seq).any() or np.isnan(y_seq).any():
        raise ValueError("NaN values introduced during windowing. Check your data or window parameters.")

    ros = RandomOverSampler(random_state=config.random_state)
    X_seq_reshaped = X_seq.reshape(X_seq.shape[0], -1)
    X_balanced, y_balanced = ros.fit_resample(X_seq_reshaped, y_seq)

    X_balanced = X_balanced.reshape(-1, config.window_size, config.n_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=config.test_size,
        random_state=config.random_state, stratify=y_balanced
    )

    if np.isnan(X_train).any() or np.isnan(X_test).any() or np.isnan(y_train).any() or np.isnan(y_test).any():
        raise ValueError("NaN values found in preprocessed data. Please inspect your preprocessing pipeline.")

    return X_train, X_test, y_train, y_test