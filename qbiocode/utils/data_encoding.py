"""
Quantum Data Encoding Utilities
================================

This module provides utility functions for encoding classical data into
quantum states, including normalization, label encoding, and training
set preparation for quantum machine learning algorithms.

These functions are generic and can be used across different quantum
algorithms, not just ensemble learning.
"""

import math

import numpy as np
from typing import List, Tuple


def normalize_data(x: np.ndarray, C: float = 1.0) -> List[complex]:
    """
    Normalize and pad a data vector for quantum state initialization.

    Pads ``x`` with zeros to the next power-of-two length, then normalises to
    unit L2 norm and converts to complex amplitudes.  This is required because
    Qiskit's ``QuantumCircuit.initialize()`` accepts only state vectors whose
    length is exactly ``2**n`` for some integer ``n``.

    Without padding, any input whose length is not already a power of 2 (e.g.
    10, 167, 200) causes ``QiskitError: Desired statevector length not a
    positive power of 2``.

    The zero-padding does not change the relative weights of the original
    features — it merely adds ``2**⌈log₂(N)⌉ - N`` zero-amplitude basis
    states that carry no information.

    Parameters
    ----------
    x : np.ndarray
        Classical data vector to normalize. May be any length > 0.
    C : float, optional
        Scaling constant applied to the squared norm (default: 1.0).

    Returns
    -------
    List[complex]
        Normalized, power-of-2-padded vector as a list of complex numbers.
        Length is always ``2**⌈log₂(len(x))⌉`` (or 2 if ``len(x) == 1``).

    Examples
    --------
    >>> x = np.array([3.0, 4.0])          # length 2 — already power of 2
    >>> x_norm = normalize_data(x)
    >>> round(sum(abs(xi)**2 for xi in x_norm), 10)
    1.0

    >>> x = np.array([1.0, 0.0, 1.0])     # length 3 — padded to 4
    >>> len(normalize_data(x))
    4
    """
    n = len(x)
    # Pad to next power of 2 so Qiskit's initialize() accepts the vector.
    n_padded = 2 ** math.ceil(math.log2(n)) if n > 1 else 2
    if n_padded != n:
        padded = np.zeros(n_padded, dtype=float)
        padded[:n] = x
    else:
        padded = np.asarray(x, dtype=float)
    M = np.sum(padded ** 2)
    if M == 0:
        # All-zero vector: return a uniform superposition over the first two
        # basis states to avoid division by zero.
        result = [complex(0, 0)] * n_padded
        result[0] = complex(1.0 / math.sqrt(2), 0)
        result[1] = complex(1.0 / math.sqrt(2), 0)
        return result
    scale = np.sqrt(M * C)
    return [complex(float(v) / scale, 0) for v in padded]


def label_to_array(y: np.ndarray) -> np.ndarray:
    """
    Convert binary labels to one-hot encoded arrays.
    
    Transforms binary classification labels (0 or 1) into one-hot encoded
    format required by quantum circuits. Label 0 becomes [1, 0] and label
    1 becomes [0, 1].
    
    Parameters
    ----------
    y : np.ndarray
        Binary labels (0 or 1)
    
    Returns
    -------
    np.ndarray
        One-hot encoded labels, shape (n_samples, 2)
    
    Examples
    --------
    >>> y = np.array([0, 1, 0])
    >>> label_to_array(y)
    array([[1, 0],
           [0, 1],
           [1, 0]])
    """
    Y = []
    for el in y:
        if el == 0:
            Y.append([1, 0])
        else:
            Y.append([0, 1])
    return np.asarray(Y)


def prepare_training_set(X: np.ndarray, y: np.ndarray,
                         n: int = 4, seed: int = 123) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select and prepare a balanced training subset for the quantum ensemble.

    Selects ``n/2`` samples from each class, normalises each sample via
    :func:`normalize_data` (which pads to the next power-of-2 length), and
    returns the results as a numpy array.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Training feature data.
    y : np.ndarray, shape (n_samples,)
        Training labels (binary: 0 or 1).
    n : int, optional
        Total number of training samples to select (must be even, default: 4).
    seed : int, optional
        Random seed for reproducibility (default: 123).

    Returns
    -------
    X_data : np.ndarray, shape (n, 2**⌈log₂(n_features)⌉)
        Normalised, power-of-2-padded training samples.
    Y_data : np.ndarray, shape (n, 2)
        One-hot encoded labels.

    Examples
    --------
    >>> X = np.random.rand(20, 8)   # 8 features — already power of 2
    >>> y = np.array([0]*10 + [1]*10)
    >>> X_data, Y_data = prepare_training_set(X, y, n=4, seed=42)
    >>> X_data.shape
    (4, 8)
    """
    np.random.seed(seed)

    # Select balanced samples from each class
    ix_y1 = np.random.choice(np.where(y == 1)[0], int(n / 2), replace=False)
    ix_y0 = np.random.choice(np.where(y == 0)[0], int(n / 2), replace=False)

    X_selected = np.concatenate([X[ix_y1], X[ix_y0]])
    Y_data = label_to_array(np.concatenate([y[ix_y1], y[ix_y0]]))

    # normalize_data pads each sample to the next power-of-2 length,
    # which is required by Qiskit's QuantumCircuit.initialize().
    X_data = np.array([normalize_data(x) for x in X_selected])

    return X_data, Y_data


