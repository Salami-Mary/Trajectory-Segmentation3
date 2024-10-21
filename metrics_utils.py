from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score, confusion_matrix, log_loss
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calculate evaluation metrics for classification models.
    
    Args:
    y_true (array-like): True labels
    y_pred (array-like): Predicted labels
    y_pred_proba (array-like, optional): Predicted probabilities for each class
    
    Returns:
    dict: Dictionary containing calculated metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
        'kappa': cohen_kappa_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
    }

    if y_pred_proba is not None:
        try:
            metrics['log_loss'] = log_loss(y_true, y_pred_proba)
        except ValueError as e:
            print(f"Error calculating log loss: {e}")
            metrics['log_loss'] = None

    return metrics

def plot_confusion_matrix(y_true, y_pred, classes, model_name, window_size, results_dir):
    """
    Plot and save confusion matrix.
    
    Args:
    y_true (array-like): True labels
    y_pred (array-like): Predicted labels
    classes (list): List of class names
    model_name (str): Name of the model
    window_size (int): Size of the window used for prediction
    results_dir (str): Directory to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {model_name} (Window Size: {window_size})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Ensure the results directory exists
    os.makedirs('/content/drive/MyDrive/final Master thesis code-GITHUB/scaryfinals/results_dir', exist_ok=True)
    
    # Save the plot
    plot_path = os.path.join(results_dir, f'confusion_matrix_{model_name}_window{window_size}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Confusion matrix saved to {plot_path}")

def print_classification_report(y_true, y_pred, classes):
    """
    Print a formatted classification report.
    
    Args:
    y_true (array-like): True labels
    y_pred (array-like): Predicted labels
    classes (list): List of class names
    """
    from sklearn.metrics import classification_report
    
    report = classification_report(y_true, y_pred, target_names=classes)
    print("\nClassification Report:")
    print(report)

def plot_roc_curve(y_true, y_pred_proba, classes, model_name, window_size, results_dir):
    """
    Plot and save ROC curve for multi-class classification.
    
    Args:
    y_true (array-like): True labels
    y_pred_proba (array-like): Predicted probabilities for each class
    classes (list): List of class names
    model_name (str): Name of the model
    window_size (int): Size of the window used for prediction
    results_dir (str): Directory to save the plot
    """
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    from itertools import cycle
    
    n_classes = len(classes)
    y_test_bin = label_binarize(y_true, classes=range(n_classes))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {classes[i]} (area = {roc_auc[i]:0.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name} (Window Size: {window_size})')
    plt.legend(loc="lower right")
    
    # Ensure the results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Save the plot
    plot_path = os.path.join(results_dir, f'roc_curve_{model_name}_window{window_size}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"ROC curve saved to {plot_path}")