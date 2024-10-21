from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, LSTM, Dropout
from tensorflow.keras.utils import to_categorical
from metrics_utils import calculate_metrics
from hmm_model import TrajectoryHMMModel
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

class CHybridModel:
    def __init__(self, config):
        self.config = config
        self.hmm_model = TrajectoryHMMModel(config)
        self.model = self.build_model()

    def build_model(self):
        lstm_input = Input(shape=(self.config.window_size, self.config.n_features), name="lstm_input")
        lstm_layer1 = LSTM(self.config.lstm_units, return_sequences=True)(lstm_input)
        lstm_dropout1 = Dropout(self.config.lstm_dropout_rate)(lstm_layer1)
        lstm_layer2 = LSTM(self.config.lstm_units)(lstm_dropout1)
        lstm_dropout2 = Dropout(self.config.lstm_dropout_rate)(lstm_layer2)

        # Change the input shape for HMM features to match LSTM input
        hmm_input = Input(shape=(self.config.window_size, self.config.n_classes), name="hmm_input")
        hmm_flatten = LSTM(self.config.lstm_units)(hmm_input)

        concatenated = Concatenate()([lstm_dropout2, hmm_flatten])

        dense1 = Dense(64, activation='relu')(concatenated)
        output = Dense(self.config.n_classes, activation='softmax')(dense1)

        model = Model(inputs=[lstm_input, hmm_input], outputs=output)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def train(self, X_train, y_train):
        self.hmm_model.train(X_train, y_train)
        
        # Generate HMM features for each time step in the sequence
        hmm_train = np.array([self.hmm_model.predict_proba(x) for x in X_train])
        
        y_train_cat = to_categorical(y_train, num_classes=self.config.n_classes)
        early_stopping = EarlyStopping(
        monitor='val_loss', patience=self.config.early_stopping_patience, restore_best_weights=True
    )

        self.model.fit(
            [X_train, hmm_train], y_train_cat, 
            epochs=self.config.epochs, 
            batch_size=self.config.batch_size, 
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )

    def predict(self, X):
        hmm_features = np.array([self.hmm_model.predict_proba(x) for x in X])
        return self.model.predict([X, hmm_features])

    def evaluate(self, X_test, y_test):
        hmm_features = np.array([self.hmm_model.predict_proba(x) for x in X_test])
        y_pred_proba = self.model.predict([X_test, hmm_features])
        y_pred = np.argmax(y_pred_proba, axis=1)
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        return metrics, y_pred