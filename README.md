# QBioCode

**Quantum machine learning tools for healthcare and life sciences.**

[![PyPI version](https://badge.fury.io/py/qbiocode.svg)](https://badge.fury.io/py/qbiocode) [![Minimum Python Version](https://img.shields.io/badge/Python-%3E=%203.10-blue)](https://www.python.org/downloads/) [![Maximum Python Version Tested](https://img.shields.io/badge/Python-%3C=%203.12-blueviolet)](https://www.python.org/downloads/) [![Supported Python Versions](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/) [![GitHub Pages](https://img.shields.io/badge/docs-sphinx-blue)](https://ibm.github.io/QBioCode/)

<img src="docs/source/img/QBioCode_logo.png" width="300" />

QBioCode benchmarks quantum and classical machine learning models, analyzes data complexity, and supports model selection for healthcare and life science applications.

## Key Features

- **QProfiler**: Automated ML benchmarking with data complexity analysis
- **QSage**: Meta-learning tool for model selection
- **Data Generation**: Synthetic datasets with controlled complexity
- **Quantum ML**: QSVC, PQK, VQC, QNN, Quantum Ensemble
- **Classical ML**: RF, SVM, LR, DT, NB, MLP, XGBoost

## Requirements

Python **3.10–3.12**.

## Installation

### From PyPI

```bash
pip install qbiocode

# With apps (QProfiler, QSage)
pip install 'qbiocode[apps]'

# All optional dependencies
pip install 'qbiocode[all]'
```

### With Conda

```bash
conda create -n qbiocode python=3.10
conda activate qbiocode
pip install qbiocode
```

### From Source

```bash
git clone https://github.com/IBM/QBioCode.git
cd QBioCode
python -m venv .env
source .env/bin/activate  # Windows: .env\Scripts\activate
pip install -e '.[apps]'
```

**macOS:** XGBoost requires OpenMP:
```bash
brew install libomp
pip install --force-reinstall xgboost
```

Full installation guide: [https://ibm.github.io/QBioCode/installation.html](https://ibm.github.io/QBioCode/installation.html)

## Running Tests

```bash
pip install -e '.[dev]'
python -m pytest
```

## Basic Usage

```python
import qbiocode as qbc

# Generate synthetic data
qbc.generate_data(
    type_of_data='moons',
    save_path='data/moons',
    n_samples=[100, 200],
    noise=[0.1, 0.2],
    random_state=42
)
```

## Applications

### QProfiler

Benchmarks classical and quantum ML models on CSV datasets. Computes 15+ data complexity metrics and correlates them with model performance.

`folder_path` in the config resolves relative to the QBioCode repo root. Run from inside the repo tree.

```bash
cd /path/to/QBioCode

# Default config
qprofiler

# Tutorial config and data
qprofiler --config-dir=tutorial/QProfiler/configs --config-name=config

# Inline overrides (Hydra syntax)
qprofiler folder_path=tutorial/QProfiler/data/ld_data file_dataset=ALL backend=simulator
```

Results are written to `results/<config_name>/dataset=<file_dataset>/<backend>_<timestamp>/`.

```python
# Python API
from qbiocode.apps.qprofiler import qprofiler
qprofiler.main(config)
```

[QProfiler Documentation](https://ibm.github.io/QBioCode/apps/profiler.html) | [Tutorial](tutorial/QProfiler/example_qprofiler.ipynb)

### QSage

Trains surrogate regressors on QProfiler benchmark data to predict model performance on new datasets without running experiments.

```bash
# Train on QProfiler output
qsage --input compiled_results.csv --output sage_results/

# Custom options
qsage --input data.csv --output results/ --model-type xgboost_optuna --cv 10 --n-iter 100
```

```python
from qbiocode.apps.sage.sage import QuantumSage
sage = QuantumSage(data=benchmark_df)
sage.train_sub_sages(sage_type='random_forest')
predictions = sage.predict(new_dataset_features)
```

[QSage Documentation](https://ibm.github.io/QBioCode/apps/sage.html) | [Tutorial](tutorial/QSage/qsage.ipynb)

## Tutorials

1. [Artificial Data Generation](tutorial/Artificial_data_generation/example_data_generation.ipynb)
2. [QProfiler](tutorial/QProfiler/example_qprofiler.ipynb)
3. [QSage](tutorial/QSage/qsage.ipynb)
4. [Quantum Ensemble Learning](tutorial/QEnsemble/QEnsemble_example_blobs.ipynb)
5. [Quantum Projection Learning](tutorial/Quantum_Projection_Learning/QPL_example.ipynb)

## Core Modules

### Data Generation

```python
import qbiocode as qbc

qbc.generate_data(type_of_data='circles', save_path='data/circles')
qbc.generate_data(type_of_data='moons', save_path='data/moons')
qbc.generate_data(type_of_data='classes', save_path='data/classes')
```

Supported types: `circles`, `moons`, `s_curve`, `spheres`, `spirals`, `swiss_roll`, `classes`.

### Machine Learning Models

**Classical:** RF, SVM, LR, DT, NB, MLP, XGBoost

**Quantum:** QSVC, PQK, VQC, QNN, Quantum Ensemble (swap and random-unitary methods)

```python
from qbiocode.learning import compute_qensemble
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

results = compute_qensemble(
    X_train, X_test, y_train, y_test,
    ensemble_method='swap',
    n_ensemble=4,
    seed=42
)
```

### Embeddings

PCA, SVD, NMF, UMAP, Isomap, LLE, Spectral, t-SNE, MIFSR, VarThresh, KernelPCA

### Evaluation

Model performance metrics (accuracy, F1, AUC) and data complexity analysis.

## Utilities

### QML Config Generation

```python
from qbiocode.utils import generate_qml_experiment_configs

num_configs, used_files = generate_qml_experiment_configs(
    template_config_path='configs/config.yaml',
    output_dir='configs/qml_gridsearch',
    data_dirs=['data/my_datasets'],
    qmethods=['qnn', 'vqc', 'qsvc'],
    reps=[1, 2],
    n_components=[5, 10],
    embeddings=['none', 'pca', 'isomap']
)
```

## Documentation

[https://ibm.github.io/QBioCode/](https://ibm.github.io/QBioCode/)

- [Installation Guide](https://ibm.github.io/QBioCode/installation.html)
- [API Reference](https://ibm.github.io/QBioCode/api/qbiocode.html)
- [Tutorials](https://ibm.github.io/QBioCode/tutorials.html)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Citation

```bibtex
@software{qbiocode2024,
  title = {QBioCode: Quantum Machine Learning for Healthcare and Life Sciences},
  author = {Raubenolt, Bryan and Bose, Aritra and Rhrissorrakrai, Kahn and 
            Utro, Filippo and Mohan, Akhil and Blankenberg, Daniel and Parida, Laxmi},
  year = {2024},
  url = {https://github.com/IBM/QBioCode}
}
```

See [CITATION.cff](CITATION.cff) for details.

## Authors

- Bryan Raubenolt (raubenb@ccf.org) — Cleveland Clinic
- Aritra Bose (a.bose@ibm.com) — IBM Research
- Kahn Rhrissorrakrai (krhriss@us.ibm.com) — IBM Research
- Filippo Utro (futro@us.ibm.com) — IBM Research
- Akhil Mohan (mohana2@ccf.org) — Cleveland Clinic
- Daniel Blankenberg (blanked2@ccf.org) — Cleveland Clinic
- Laxmi Parida (parida@us.ibm.com) — IBM Research

## Support

- Open an issue on [GitHub](https://github.com/IBM/QBioCode/issues)
- See the [documentation](https://ibm.github.io/QBioCode/)
