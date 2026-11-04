import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # Paths
    base_dir: str = os.path.dirname(os.path.abspath(__file__))
    workspace_dir: str = os.path.dirname(base_dir)
    data_path: str = os.path.join(workspace_dir, "HMDAD.txt")
    output_dir: str = os.path.join(workspace_dir, "sgat_clmda_outputs")
    
    # Model Architecture
    hidden_dim: int = 256
    out_dim: int = 64
    n_layers: int = 2
    heads: int = 4
    dropout: float = 0.4
    
    # Sparsification / Gating Hyperparameters
    init_tau: float = 0.1
    sim_threshold: float = 0.3
    edge_gating_mode: str = "node_norm"  # Options: 'node_norm', 'differentiable_edge', 'both'
    
    # Contrastive Learning Hyperparameters
    use_contrastive: bool = True
    cl_weight: float = 0.1
    cl_temperature: float = 0.5
    cl_mode: str = "both"  # Options: 'both', 'disease_only', 'microbe_only', 'none'
    
    # Training Parameters
    epochs: int = 400
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 50
    k_folds: int = 5
    seed: int = 42
    multi_seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 2024])
    
    # Hardware
    device: str = "cuda"  # auto-detected at runtime

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
