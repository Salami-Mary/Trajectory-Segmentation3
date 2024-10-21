from hmmlearn import hmm
import numpy as np
from metrics_utils import calculate_metrics

class TrajectoryHMMModel:
    def __init__(self, config):
        self.config = config
        self.models = {
            i: hmm.GaussianHMM(
                n_components=config.hmm_n_components,
                covariance_type=config.hmm_covariance_type,
                n_iter=config.hmm_n_iter,
                random_state=config.random_state
            )
            for i in range(config.n_classes)
        }

    def train(self, X_train, y_train):
        for i in range(self.config.n_classes):
            X_class = X_train[y_train == i]
            if X_class.size == 0:
                raise ValueError(f"No samples found for class {i}. Check your data.")

            X_class_flat = X_class.reshape(-1, X_class.shape[-1])
            self.models[i].fit(X_class_flat)

    def predict_proba(self, X):
        probas = []
        for x in X:
            scores = [model.score(x.reshape(-1, x.shape[-1])) for model in self.models.values()]
            log_probs = np.array(scores)
            log_probs -= np.max(log_probs)
            probs = np.exp(log_probs)
            probs /= np.sum(probs)
            probas.append(probs)
        return np.array(probas)

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        return metrics, y_pred