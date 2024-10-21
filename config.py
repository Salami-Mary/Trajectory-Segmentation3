from dataclasses import dataclass, field, replace
from typing import List, Dict

@dataclass
class ModelConfig:
    window_size: int = 50
    window_sizes: List[int] = field(default_factory=lambda: [10, 20, 30, 40, 50, 60])
    step_size: int = 5
    n_features: int = 3
    n_classes: int = 3
    feature_columns: List[str] = field(default_factory=lambda: ['sinuosity', 'speed', 'radius_gyr'])
    label_mapping: Dict[str, int] = field(default_factory=lambda: {
        'Exploring': 0, 
        'Hunting': 1, 
        'Resting': 2
    })
    test_size: float = 0.2
    random_state: int = 42

    hmm_n_components: int = 3
    hmm_covariance_type: str = "full"
    hmm_n_iter: int = 100

    lstm_units: int = 64
    lstm_dropout_rate: float = 0.3
    lstm_l2_lambda: float = 0.0001
    batch_size: int = 64
    epochs: int = 100
    early_stopping_patience: int = 20

    def with_window_size(self, window_size: int) -> "ModelConfig":
        return replace(self, window_size=window_size)

def get_config() -> ModelConfig:
    return ModelConfig()