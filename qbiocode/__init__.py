"""
QBioCode: Quantum Machine Learning for Biological Data Analysis

Provides tools for data generation, classical and quantum machine learning,
evaluation metrics, and visualization for biological data analysis.

Modules
-------
- learning: Classical and quantum ML algorithms
- embeddings: Feature embedding and dimensionality reduction
- evaluation: Model and dataset evaluation
- data_generation: Synthetic dataset generators
- visualization: Result visualization and correlation analysis
- utils: Helper functions and utilities
- apps: Command-line applications (QProfiler, QSage)
"""

from .data_generation import (
    generate_circles_datasets,
    generate_classification_datasets,
    generate_moons_datasets,
    generate_s_curve_datasets,
    generate_spheres_datasets,
    generate_spirals_datasets,
    generate_swiss_roll_datasets,
)
from .data_generation.generator import generate_data

from .embeddings.embed import get_embeddings, pqk

from .evaluation.dataset_evaluation import evaluate
from .evaluation.model_evaluation import modeleval
from .evaluation.model_run import model_run

from .learning.compute_dt import compute_dt, compute_dt_opt
from .learning.compute_lr import compute_lr, compute_lr_opt
from .learning.compute_mlp import compute_mlp, compute_mlp_opt
from .learning.compute_nb import compute_nb, compute_nb_opt
from .learning.compute_pqk import compute_pqk
from .learning.compute_qnn import compute_qnn
from .learning.compute_qsvc import compute_qsvc
from .learning.compute_rf import compute_rf, compute_rf_opt
from .learning.compute_svc import compute_svc, compute_svc_opt
from .learning.compute_vqc import compute_vqc

try:
    from .learning.compute_xgb import compute_xgb, compute_xgb_opt
except Exception:
    # XGBoost unavailable (e.g., OpenMP not installed on macOS)
    compute_xgb = None  # type: ignore
    compute_xgb_opt = None  # type: ignore

from .utils.dataset_checkpoint import checkpoint_restart
from .utils.helper_fn import feature_encoding, scaler_fn
from .utils.qc_winner_finder import qml_winner
from .version import __version__

from .visualization.visualize_correlation import (
    compute_results_correlation,
    plot_results_correlation,
)

from . import apps  # noqa: F401
from . import data_adapters  # noqa: F401

__all__ = [
    "__version__",
    # Classical ML
    "compute_svc", "compute_svc_opt",
    "compute_dt", "compute_dt_opt",
    "compute_nb", "compute_nb_opt",
    "compute_lr", "compute_lr_opt",
    "compute_rf", "compute_rf_opt",
    "compute_xgb", "compute_xgb_opt",
    "compute_mlp", "compute_mlp_opt",
    # Quantum ML
    "compute_qnn", "compute_qsvc", "compute_vqc", "compute_pqk",
    # Embeddings
    "get_embeddings", "pqk",
    # Utilities
    "scaler_fn", "feature_encoding", "qml_winner", "checkpoint_restart",
    # Evaluation
    "modeleval", "evaluate", "model_run",
    # Visualization
    "plot_results_correlation", "compute_results_correlation",
    # Data generation
    "generate_data",
    "generate_circles_datasets", "generate_moons_datasets",
    "generate_classification_datasets", "generate_s_curve_datasets",
    "generate_spheres_datasets", "generate_spirals_datasets",
    "generate_swiss_roll_datasets",
    # Apps
    "apps",
]
