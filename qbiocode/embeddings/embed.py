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
    Compute Projected Quantum Kernel (PQK) feature projections.

    Encodes data into quantum states via a feature map and measures Pauli
    expectation values (X, Y, Z) on each qubit to produce a classical feature
    representation.

    Args:
        X_train (np.ndarray): Training features.
        X_test (np.ndarray): Test features.
        args (dict): Config dict with ``backend`` and other runtime settings.
        store (bool): Save projections to ``pqk_projections/`` if True.
        data_key (str): Filename key when storing projections.
        encoding (str): Feature map type — ``'Z'``, ``'ZZ'``, or ``'P'``.
        data_map (bool): Normalize single-qubit gate coefficients to 1.0.
        primitive (str): Primitive type (``'estimator'``).
        entanglement (str): Entanglement strategy (``'linear'``).
        reps (int): Feature map repetitions.

    Returns:
        tuple: ``(X_train_prj, X_test_prj)`` — projected feature arrays.
    """

    feat_dimension = X_train.shape[1]

    if data_map:
        def data_map_func(x: np.ndarray) -> float:
            coeff = x[0] / 2 if len(x) == 1 else reduce(lambda m, n: (m * n) / 2, x)
            return float(coeff)
    else:
        data_map_func = None

    feature_map, _ = qutils.get_feature_map(
        feature_map=encoding,
        feat_dimension=X_train.shape[1],
        reps=reps,
        entanglement=entanglement,
        data_map_func=data_map_func,
    )

    circuit = QuantumCircuit(feature_map.num_qubits)
    circuit.compose(feature_map, inplace=True)
    num_qubits = circuit.num_qubits

    backend, session, prim = qutils.get_backend_session(args, "estimator", num_qubits=num_qubits)

    if args["backend"] != "simulator":
        circuit = qutils.transpile_circuit(
            circuit, opt_level=3, backend=backend, PT=True, initial_layout=None
        )

    for f_tr in ["train", "test"]:

        if "train" in f_tr:
            dat = X_train.copy()
        else:
            dat = X_test.copy()

        id = "I" * feat_dimension

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

        projections = []

        for i in range(len(dat)):
            parameters = dat[i]

            pub_x = (circuit, observables_x, parameters)
            pub_y = (circuit, observables_y, parameters)
            pub_z = (circuit, observables_z, parameters)

            job = prim.run([pub_x, pub_y, pub_z])
            job_result_x = job.result()[0].data.evs
            job_result_y = job.result()[1].data.evs
            job_result_z = job.result()[2].data.evs
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
    """Apply a dimensionality-reduction method to training and test data.

    All methods fit on training data only. ``tsne`` and ``spectral`` embed
    train and test jointly (no ``transform()``); use for benchmarking only.

    Supported methods
    -----------------
    - ``none``      — no reduction
    - ``pca``       — PCA (linear)
    - ``svd``       — Truncated SVD (sparse-safe, no mean-centring)
    - ``nmf``       — Non-negative Matrix Factorisation (requires non-negative input)
    - ``umap``      — UMAP (topology-preserving)
    - ``isomap``    — Isomap (geodesic distances)
    - ``lle``       — Locally Linear Embedding
    - ``spectral``  — Laplacian Eigenmaps
    - ``tsne``      — t-SNE (joint embedding)
    - ``mifsr``     — Mutual-Information Feature Selection + Ranking (supervised)
    - ``varthresh`` — Variance-threshold feature selection
    - ``kpca``      — Kernel PCA (RBF)

    Parameters
    ----------
    embedding : str
        Reduction method name (case-insensitive).
    X_train : array-like of shape (n_train, n_features)
        Training feature matrix.
    X_test : array-like of shape (n_test, n_features)
        Test feature matrix.
    n_neighbors : int, optional
        Neighbor count for UMAP, Isomap, LLE (default: 30).
    n_components : int, optional
        Output dimensionality. Must be ≤ ``X_train.shape[1]``.
    method : str, optional
        LLE variant — ``'modified'`` or ``None`` (standard).
    y_train : array-like, optional
        Training labels required by ``mifsr``.
    random_state : int, optional
        Random seed for stochastic methods (default: 42).

    Returns
    -------
    X_train_emb : np.ndarray of shape (n_train, n_components)
    X_test_emb  : np.ndarray of shape (n_test,  n_components)
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

    if embedding == "pca":
        model = PCA(n_components=n_components, random_state=random_state)
        X_train_emb = model.fit_transform(X_train)
        X_test_emb  = model.transform(X_test)

    elif embedding == "svd":
        model = TruncatedSVD(n_components=n_components, random_state=random_state)
        X_train_emb = model.fit_transform(X_train)
        X_test_emb  = model.transform(X_test)

    elif embedding == "nmf":
        model = NMF(
            n_components=n_components,
            random_state=random_state,
            max_iter=500,
        )
        X_train_emb = model.fit_transform(X_train)
        X_test_emb  = model.transform(X_test)

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
        X_all = np.vstack([X_train, X_test])
        X_all_emb = model.fit_transform(X_all)
        X_train_emb = X_all_emb[:len(X_train)]
        X_test_emb  = X_all_emb[len(X_train):]

    elif embedding == "tsne":
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
        vt = VarianceThreshold(threshold=0.0)
        X_tr_vt = vt.fit_transform(X_train)
        if X_tr_vt.shape[1] == 0:
            X_tr_vt = X_train
            X_te_vt = X_test
        else:
            X_te_vt = vt.transform(X_test)
        variances = np.var(X_tr_vt, axis=0)
        top_k_idx = np.argsort(variances)[::-1][:n_components]
        top_k_idx_sorted = np.sort(top_k_idx)
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
