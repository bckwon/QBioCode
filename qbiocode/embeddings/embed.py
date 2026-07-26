import os
from functools import reduce

import numpy as np

# ====== Qiskit imports ======
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli

# ====== Embedding functions imports ======
from sklearn.decomposition import KernelPCA, NMF, PCA, TruncatedSVD
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_classif
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, SpectralEmbedding
from umap import UMAP

import qbiocode.utils.qutils as qutils


def pqk(
    X_train,
    X_test,
    args,
    store=False,
    data_key="",
    encoding="Z",
    data_map=True,
    primitive="estimator",
    entanglement="linear",
    reps=2,
):
    """
    This function generates quantum circuits, computes projections of the data onto these circuits.
    It uses a feature map to encode the data into quantum states and then measures the expectation values
    of Pauli operators to obtain the features.
    This function requires a quantum backend (simulator or real quantum hardware) for execution.
    It supports various configurations such as encoding methods, entanglement strategies, and repetitions
    of the feature map. Optionally the results are saved to files for training and test projections.

    Args:
        X_train (np.ndarray): Training data features.
        X_test (np.ndarray): Test data features.
        args (dict): Arguments containing backend and other configurations.
        store (bool): If true projections are stored, using data_key as indefitier
        data_key (str): Key for the dataset, default is ''.
        encoding (str): Encoding method for the quantum circuit, default is 'Z'.
        data_map (bool): If true ensures that all multiplicative factors of data features inside single qubit gates are 1.0. Not applicable for Hejsemberg feature maps
        primitive (str): Primitive type to use, default is 'estimator'.
        entanglement (str): Entanglement strategy, default is 'linear'.
        reps (int): Number of repetitions for the feature map, default is 2.

    Returns:
        modeleval (dict): A dictionary containing evaluation metrics and model parameters.
    """

    feat_dimension = X_train.shape[1]

    if data_map:
        #  This function ensures that all multiplicative factors of data features inside single qubit gates are 1.0
        def data_map_func(x: np.ndarray) -> float:
            """
            Define a function map from R^n to R.

            Args:
                x: data

            Returns:
                float: the mapped value
            """
            coeff = x[0] / 2 if len(x) == 1 else reduce(lambda m, n: (m * n) / 2, x)
            return float(coeff)

    else:
        data_map_func = None

    # choose a method for mapping your features onto the circuit
    feature_map, _ = qutils.get_feature_map(
        feature_map=encoding,
        feat_dimension=X_train.shape[1],
        reps=reps,
        entanglement=entanglement,
        data_map_func=data_map_func,
    )

    # Build quantum circuit
    circuit = QuantumCircuit(feature_map.num_qubits)
    circuit.compose(feature_map, inplace=True)
    num_qubits = circuit.num_qubits

    #  Generate the backend, session and primitive
    backend, session, prim = qutils.get_backend_session(args, "estimator", num_qubits=num_qubits)

    # Transpile
    if args["backend"] != "simulator":
        circuit = qutils.transpile_circuit(
            circuit, opt_level=3, backend=backend, PT=True, initial_layout=None
        )

    for f_tr in ["train", "test"]:

        if "train" in f_tr:
            dat = X_train.copy()
        else:
            dat = X_test.copy()

        # Identity operator on all qubits
        id = "I" * feat_dimension

        # We group all commuting observables
        # These groups are the Pauli X, Y and Z operators on individual qubits
        # Apply the circuit layout to the observable if mapped to device
        if args["backend"] != "simulator":
            observables_x = []
            observables_y = []
            observables_z = []
            for i in range(feat_dimension):
                observables_x.append(
                    Pauli(id[:i] + "X" + id[(i + 1) :]).apply_layout(
                        circuit.layout, num_qubits=backend.num_qubits
                    )
                )
                observables_y.append(
                    Pauli(id[:i] + "Y" + id[(i + 1) :]).apply_layout(
                        circuit.layout, num_qubits=backend.num_qubits
                    )
                )
                observables_z.append(
                    Pauli(id[:i] + "Z" + id[(i + 1) :]).apply_layout(
                        circuit.layout, num_qubits=backend.num_qubits
                    )
                )
        else:
            observables_x = [Pauli(id[:i] + "X" + id[(i + 1) :]) for i in range(feat_dimension)]
            observables_y = [Pauli(id[:i] + "Y" + id[(i + 1) :]) for i in range(feat_dimension)]
            observables_z = [Pauli(id[:i] + "Z" + id[(i + 1) :]) for i in range(feat_dimension)]

        # projections[i][j][k] will be the expectation value of the j-th Pauli operator (0: X, 1: Y, 2: Z)
        # of datapoint i on qubit k
        projections = []

        for i in range(len(dat)):

            # Get training sample
            parameters = dat[i]

            # We define the primitive unified blocs (PUBs) consisting of the embedding circuit,
            # set of observables and the circuit parameters
            pub_x = (circuit, observables_x, parameters)
            pub_y = (circuit, observables_y, parameters)
            pub_z = (circuit, observables_z, parameters)

            job = prim.run([pub_x, pub_y, pub_z])
            job_result_x = job.result()[0].data.evs
            job_result_y = job.result()[1].data.evs
            job_result_z = job.result()[2].data.evs

            # Record <X>, <Y> and <Z> on all qubits for the current datapoint
            projections.append([job_result_x, job_result_y, job_result_z])

        if store:
            if not os.path.exists("pqk_projections"):
                os.makedirs("pqk_projections")

            file_projection = os.path.join(
                "pqk_projections", "pqk_projection_" + data_key + "_" + f_tr + ".npy"
            )

            np.save(file_projection, projections)

        if "train" in f_tr:
            X_train_prj = np.array(projections.copy()).reshape(len(projections), -1)
        else:
            X_test_prj = np.array(projections.copy()).reshape(len(projections), -1)

    if not isinstance(session, type(None)):
        session.close()

    return X_train_prj, X_test_prj


def get_embeddings(
    embedding: str,
    X_train,
    X_test,
    n_neighbors: int = 30,
    n_components: int = None,
    method: str = None,
    y_train=None,
    random_state: int = 42,
):
    """Apply a dimensionality-reduction technique to training and test data.

    The function is designed to produce a fixed-width representation that fits
    within a quantum-simulator qubit budget (target: n_components ≤ 26, ideally
    a power of 2 for QEnsemble compatibility).

    Classical models receive ``embed='none'`` (full fingerprint). Quantum models
    receive one of the reduced representations below.  The distinction is made by
    the caller (``qprofiler.py``) which loops over ``embeddings`` in the config.

    Supported modes
    ---------------
    Linear / global structure
      ``pca``        — Principal Component Analysis (linear, fast, deterministic)
      ``svd``        — Truncated SVD / LSA (PCA without mean-centring; sparse-safe,
                       ideal for binary fingerprints such as ECFP4 and MACCS)
      ``nmf``        — Non-negative Matrix Factorisation (non-negative inputs only;
                       suitable for fingerprints, not for PCA-output)

    Non-linear / manifold
      ``umap``       — Uniform Manifold Approximation and Projection
                       (topology-preserving, fast, supports ``transform()``)
      ``isomap``     — Isometric mapping (geodesic distances)
      ``lle``        — Locally Linear Embedding
      ``spectral``   — Laplacian Eigenmaps

    Semi-supervised / discriminative
      ``tsne``       — t-SNE via sklearn.  *Note:* sklearn t-SNE has no
                       ``transform()`` — test data are embedded jointly with train.
                       Results are valid for benchmarking (same random state) but
                       the test split cannot be used for truly held-out inference.
                       For production use, replace with openTSNE which supports
                       ``transform()``.
      ``mifsr``      — Mutual-Information Feature Selection + Ranking: selects the
                       top-k original features ranked by MI(feature, label).
                       Supervised, deterministic once ``y_train`` is supplied.
                       Interpretable: each retained dimension is an original
                       molecular descriptor.
      ``varthresh``  — Variance-threshold feature selection: keeps the k highest-
                       variance features after removing near-zero-variance columns.
                       Unsupervised and deterministic.
      ``kpca``       — Kernel PCA with an RBF kernel (non-linear, unsupervised).

    Parameters
    ----------
    embedding : str
        Name of the reduction method (case-insensitive).
    X_train : array-like, shape (n_train, n_features)
        Training feature matrix.
    X_test : array-like, shape (n_test, n_features)
        Test feature matrix.
    n_neighbors : int, optional
        Neighbour count used by UMAP, IsoMap and LLE (default: 30).
    n_components : int, optional
        Target number of components / features.  Must be ≤ ``X_train.shape[1]``.
        Should be a power of 2 when QEnsemble is included (e.g. 8).
    method : str, optional
        Sub-method for LLE (``'modified'`` or ``None`` for standard).
    y_train : array-like, optional
        Class labels for supervised methods (``mifsr``).  Ignored by all others.
    random_state : int, optional
        Random seed for stochastic methods (UMAP, t-SNE, KernelPCA; default: 42).

    Returns
    -------
    X_train_emb : np.ndarray, shape (n_train, n_components)
    X_test_emb  : np.ndarray, shape (n_test,  n_components)

    Notes
    -----
    All methods are **fit on training data only** and applied to test data via
    ``transform()`` (or an equivalent held-out projection), preserving the
    train/test boundary required for unbiased evaluation.  The only exception
    is ``tsne`` — see the docstring note above.
    """
    embedding = embedding.lower()
    valid_modes = [
        "none",
        "pca", "svd", "nmf",
        "umap", "isomap", "lle", "spectral",
        "tsne", "mifsr", "varthresh", "kpca",
    ]
    if embedding not in valid_modes:
        raise ValueError(
            f"Invalid embedding mode '{embedding}'. "
            f"Must be one of: {valid_modes}"
        )

    if embedding == "none":
        return X_train, X_test

    if n_components is None:
        n_components = X_train.shape[1]

    if n_components > X_train.shape[1]:
        raise ValueError(
            f"n_components ({n_components}) exceeds the number of features "
            f"({X_train.shape[1]})."
        )

    X_train = np.asarray(X_train, dtype=float)
    X_test  = np.asarray(X_test,  dtype=float)

    # ── Linear / global ──────────────────────────────────────────────────────
    if embedding == "pca":
        model = PCA(n_components=n_components, random_state=random_state)
        X_train_emb = model.fit_transform(X_train)
        X_test_emb  = model.transform(X_test)

    elif embedding == "svd":
        # TruncatedSVD does NOT centre the data — preserves sparsity and is
        # better suited to binary fingerprints than PCA.
        model = TruncatedSVD(n_components=n_components, random_state=random_state)
        X_train_emb = model.fit_transform(X_train)
        X_test_emb  = model.transform(X_test)

    elif embedding == "nmf":
        # NMF requires non-negative inputs. MinMaxScaler applied upstream in
        # qprofiler.py ensures X ∈ [0, 1], satisfying this constraint.
        model = NMF(
            n_components=n_components,
            random_state=random_state,
            max_iter=500,
        )
        X_train_emb = model.fit_transform(X_train)
        X_test_emb  = model.transform(X_test)

    # ── Non-linear / manifold ────────────────────────────────────────────────
    elif embedding == "umap":
        model = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            random_state=random_state,
        )
        X_train_emb = model.fit_transform(X_train)
        X_test_emb  = model.transform(X_test)

    elif embedding == "isomap":
        model = Isomap(n_neighbors=n_neighbors, n_components=n_components)
        X_train_emb = model.fit_transform(X_train)
        X_test_emb  = model.transform(X_test)

    elif embedding == "lle":
        lle_method = "modified" if method == "modified" else "standard"
        model = LocallyLinearEmbedding(
            n_neighbors=n_neighbors,
            n_components=n_components,
            method=lle_method,
            random_state=random_state,
        )
        X_train_emb = model.fit_transform(X_train)
        X_test_emb  = model.transform(X_test)

    elif embedding == "spectral":
        model = SpectralEmbedding(
            n_components=n_components,
            n_neighbors=n_neighbors,
            random_state=random_state,
            eigen_solver="arpack",
        )
        # SpectralEmbedding has no transform(); embed train+test jointly and
        # split — valid for benchmarking, not for deployed inference.
        X_all = np.vstack([X_train, X_test])
        X_all_emb = model.fit_transform(X_all)
        X_train_emb = X_all_emb[:len(X_train)]
        X_test_emb  = X_all_emb[len(X_train):]

    # ── Semi-supervised / discriminative ────────────────────────────────────
    elif embedding == "tsne":
        # sklearn t-SNE: fit train+test jointly to enable test projection.
        # Fixed random_state ensures reproducibility across embedding passes.
        # Barnes-Hut approximation only supports n_components ≤ 3; for higher
        # dimensions we fall back to the exact (brute-force) method, which is
        # O(N²) but correct for any n_components ≥ 1.
        tsne_method = "exact" if n_components > 3 else "barnes_hut"
        model = TSNE(
            n_components=n_components,
            random_state=random_state,
            method=tsne_method,
            init="pca",
            learning_rate="auto",
            perplexity=min(30, len(X_train) - 1),
            n_jobs=1,
        )
        X_all = np.vstack([X_train, X_test])
        X_all_emb = model.fit_transform(X_all)
        X_train_emb = X_all_emb[:len(X_train)]
        X_test_emb  = X_all_emb[len(X_train):]

    elif embedding == "mifsr":
        # Mutual-Information Feature Selection + Ranking.
        # Supervised: fits on training labels only, then applies the same
        # column mask to the test set — no label leakage.
        if y_train is None:
            raise ValueError(
                "embedding='mifsr' requires y_train (training labels)."
            )
        selector = SelectKBest(
            score_func=mutual_info_classif,
            k=n_components,
        )
        X_train_emb = selector.fit_transform(X_train, y_train)
        X_test_emb  = selector.transform(X_test)

    elif embedding == "varthresh":
        # Step 1: remove near-zero-variance features (threshold=0.0 keeps all
        # non-constant columns for binary fingerprints).
        vt = VarianceThreshold(threshold=0.0)
        X_tr_vt = vt.fit_transform(X_train)
        # If VarianceThreshold removed ALL features (all-constant dataset),
        # fall back to the raw features so downstream code still runs.
        if X_tr_vt.shape[1] == 0:
            X_tr_vt = X_train
            X_te_vt = X_test
        else:
            X_te_vt = vt.transform(X_test)
        # Step 2: keep the top-k by descending variance.
        variances = np.var(X_tr_vt, axis=0)
        top_k_idx = np.argsort(variances)[::-1][:n_components]
        top_k_idx_sorted = np.sort(top_k_idx)  # preserve original feature order
        X_train_emb = X_tr_vt[:, top_k_idx_sorted]
        X_test_emb  = X_te_vt[:, top_k_idx_sorted]

    elif embedding == "kpca":
        model = KernelPCA(
            n_components=n_components,
            kernel="rbf",
            fit_inverse_transform=False,
            random_state=random_state,
        )
        X_train_emb = model.fit_transform(X_train)
        X_test_emb  = model.transform(X_test)

    return X_train_emb, X_test_emb
