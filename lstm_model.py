from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from metrics_utils import calculate_metrics
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

class LSTMModel:
    def __init__(self, config):
        self.config = config
        self.model = self.build_model()

    def build_model(self):
        model = Sequential([
            LSTM(self.config.lstm_units, return_sequences=True, 
                 input_shape=(self.config.window_size, self.config.n_features)),
            Dropout(self.config.lstm_dropout_rate),
            LSTM(self.config.lstm_units),
            Dropout(self.config.lstm_dropout_rate),
            Dense(self.config.n_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, validation_split=0.2):
        if X_train.ndim != 3:
            raise ValueError("X_train must be 3D: [samples, timesteps, features].")
        if y_train.ndim != 1:
            raise ValueError("y_train must be 1D: [samples].")

        y_train_cat = to_categorical(y_train, num_classes=self.config.n_classes)
        early_stopping = EarlyStopping(
        monitor='val_loss', patience=self.config.early_stopping_patience, restore_best_weights=True
    )

        self.model.fit(
            X_train, y_train_cat, 
            epochs=self.config.epochs, 
            batch_size=self.config.batch_size, 
            validation_split=validation_split,
            callbacks=[early_stopping],  # Add early stopping
            verbose=1
        )

    def predict(self, X):
        if X.ndim != 3:
            raise ValueError("X must be 3D: [samples, timesteps, features].")
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        return metrics, y_pred