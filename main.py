import os
import json
from datetime import datetime
from config import get_config
from data_preprocessing import load_data, preprocess_data
from hmm_model import TrajectoryHMMModel
from lstm_model import LSTMModel
from hybrid_model import CHybridModel
from metrics_utils import calculate_metrics, plot_confusion_matrix,print_classification_report,plot_roc_curve

def setup_results_directory():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def display_metrics(metrics, model_name, window_size, results_dir):
    print(f"\n{model_name} Metrics (Window Size: {window_size}):")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    metrics_file = os.path.join(results_dir, f"{model_name}_metrics_window{window_size}.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)

def train_and_evaluate_model(model_class, model_name, X_train, y_train, X_test, y_test, config, window_size, results_dir):
    print(f"\nTraining {model_name} model...")
    try:
        model = model_class(config)
        model.train(X_train, y_train)
        metrics, y_pred = model.evaluate(X_test, y_test)
        display_metrics(metrics, model_name, window_size, results_dir)
        plot_confusion_matrix(y_test, y_pred, list(config.label_mapping.keys()), model_name, window_size, results_dir)
        # In the train_and_evaluate_model function in main.py
        print_classification_report(y_test, y_pred, list(config.label_mapping.keys()))
        if hasattr(model, 'predict_proba'):
          y_pred_proba = model.predict_proba(X_test)
          plot_roc_curve(y_test, y_pred_proba, list(config.label_mapping.keys()), model_name, window_size, results_dir)
        return True
    except Exception as e:
        print(f"Error training/evaluating {model_name} model: {e}")
        return False

def main():
    config = get_config()
    results_dir = setup_results_directory()
    print(f"\n--- Starting Model Training and Evaluation ---")
    print(f"Results will be saved in: {results_dir}")

    try:
        df = load_data('/content/drive/MyDrive/winter 2024/Tiger Data Research/normaliseddata/normaltiger.csv', config)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    for window_size in config.window_sizes:
        print(f"\n--- Processing Window Size: {window_size} ---")
        config = config.with_window_size(window_size)

        try:
            X_train, X_test, y_train, y_test = preprocess_data(df, config)
            print(f"Data preprocessing successful for window size {window_size}.")
        except ValueError as e:
            print(f"Error in data preprocessing for window size {window_size}: {e}")
            continue

        models_to_train = [
            (TrajectoryHMMModel, "HMM"),
            (LSTMModel, "LSTM"),
            (CHybridModel, "C-Hybrid")
        ]

        for model_class, model_name in models_to_train:
            success = train_and_evaluate_model(model_class, model_name, X_train, y_train, X_test, y_test, config, window_size, results_dir)
            if not success:
                print(f"Skipping further processing for {model_name} with window size {window_size}")

    print(f"\n--- Model Training and Evaluation Complete ---")
    print(f"All results have been saved in: {results_dir}")

if __name__ == "__main__":
    main()