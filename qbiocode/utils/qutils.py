from qiskit_ibm_runtime.qiskit_runtime_service import QiskitRuntimeService


import logging
import math
import os
import re

import numpy as np
import pandas as pd
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
from qiskit.circuit.library import (
    EfficientSU2,
    PauliFeatureMap,
    RealAmplitudes,
    TwoLocal,
    XGate,
    YGate,
    ZFeatureMap,
    ZZFeatureMap,
)
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B, NFT, SPSA, GradientDescent, spsa
from qiskit_ibm_runtime import EstimatorOptions
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime import SamplerOptions
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime import Session
from qiskit_ibm_transpiler.transpiler_service import TranspilerService
from qiskit_aer import AerSimulator

from qbiocode.utils.ibm_account import instantiate_runtime_service


def get_backend_session(args: dict, primitive: str, num_qubits: int):
    """Return a backend, session, and primitive for the configured backend.

    Args:
        args (dict): Config dict with ``backend``, ``seed``, ``shots``,
            ``resil_level``, and ``sim_method`` keys.
        primitive (str): ``'sampler'`` or ``'estimator'``.
        num_qubits (int): Minimum qubit count required.

    Returns:
        tuple: ``(backend, session, prim)`` — backend and session are ``None``
            when using the statevector simulator.
    """
    backend = None
    session = None
    prim = None

    if args["backend"] == "simulator":

        if primitive == "estimator":
            # Estimator primitive
            prim = StatevectorEstimator(seed=args["seed"])
        else:
            prim = StatevectorSampler(seed=args["seed"], default_shots=args["shots"])
    elif ("ibm" in args["backend"]) or (args["backend"] == "simulator_aer"):
        service: QiskitRuntimeService = instantiate_runtime_service(args)
        if args["backend"] == "simulator_aer":
            backend = AerSimulator.from_backend(method = args['sim_method'])
        elif "noisy" in args["backend"]:
            noisy_backend_name = re.sub("noisy_", "", args["backend"])
            backend = AerSimulator.from_backend(service.backend(name=noisy_backend_name), method = args['sim_method'])
        else:
            if args["backend"] == "ibm_least":
                backend = service.least_busy(
                    simulator=False, operational=True, min_num_qubits=num_qubits
                )
            else:
                backend = service.backend(name=args["backend"])

        session = Session(backend=backend)

        if primitive == "sampler":
            prim = get_sampler(mode=session, shots=args["shots"])
        else:
            prim = get_estimator(mode=session, shots=args["shots"], resil_level=args["resil_level"])
    return backend, session, prim


def transpile_circuit(circuit, opt_level, backend, initial_layout, PT=False, dd_sequence="XpXm"):
    """Transpile a quantum circuit for the target backend.

    Args:
        circuit (QuantumCircuit): Circuit to transpile.
        opt_level (int or str): Optimization level, or ``'AI'`` for
            AI-assisted transpilation via ``TranspilerService``.
        backend: Target backend.
        initial_layout: Initial qubit layout.
        PT (bool): Enable pulse twirling. Default False.
        dd_sequence (str): Dynamical decoupling sequence. Default ``'XpXm'``.

    Returns:
        QuantumCircuit: Transpiled circuit.
    """
    if str(opt_level) == "AI":
        pm = TranspilerService(
            backend_name=backend,
            ai="true",
            optimization_level=3,
        )
    else:
        pm = generate_preset_pass_manager(
            optimization_level=opt_level,
            backend=backend,
            seed_transpiler=42,
            initial_layout=initial_layout,
        )
    t_qc = pm.run(circuit)

    return t_qc


def get_observable(circuit, backend):
    observable = SparsePauliOp.from_list([("Z" * circuit.num_qubits, 1)])
    # observable = SparsePauliOp.from_list([("Z" + "I" * (int(circuit.num_qubits) - 1), 0.5)])
    if "ibm" in backend.name:
        observable = observable.apply_layout(circuit.layout)  # , num_qubits=backend.num_qubits)
    return observable


def get_sampler(
    mode=None,
    shots=1024,
    dd=True,
    dd_seq="XpXm",
    PT=True,
):
    """Create a Sampler primitive with error-suppression options.

    Args:
        mode (Session): Runtime session.
        shots (int): Measurement shots. Default 1024.
        dd (bool): Enable dynamical decoupling. Default True.
        dd_seq (str): DD sequence type. Default ``'XpXm'``.
        PT (bool): Enable pulse twirling. Default True.

    Returns:
        Sampler: Configured Sampler instance.
    """
    sampler_options = SamplerOptions()
    sampler_options.default_shots = shots
    if dd:
        sampler_options.dynamical_decoupling.enable = dd
        sampler_options.dynamical_decoupling.sequence_type = dd_seq
        sampler_options.dynamical_decoupling.extra_slack_distribution = "middle"
        sampler_options.dynamical_decoupling.scheduling_method = "alap"
    if PT:
        sampler_options.twirling.enable_gates = True
        sampler_options.twirling.enable_measure = False
        sampler_options.twirling.num_randomizations = "auto"
        sampler_options.twirling.shots_per_randomization = "auto"
        sampler_options.twirling.strategy = "active-accum"

    sampler = Sampler(mode=mode, options=sampler_options)

    return sampler


def get_estimator(
    mode=None,
    shots=1024,
    resil_level=2,
    dd=True,
    dd_seq="XpXm",
    PT=True,
):
    """Create an Estimator primitive with error-suppression options.

    Args:
        mode (Session): Runtime session.
        shots (int): Measurement shots. Default 1024.
        resil_level (int): Resilience level (0–2). Default 2.
        dd (bool): Enable dynamical decoupling. Default True.
        dd_seq (str): DD sequence type. Default ``'XpXm'``.
        PT (bool): Enable pulse twirling. Default True.

    Returns:
        Estimator: Configured Estimator instance.
    """
    estimator_options = EstimatorOptions(experimental={})
    estimator_options.default_shots = shots
    estimator_options.resilience_level = resil_level
    if dd:
        estimator_options.dynamical_decoupling.enable = dd
        estimator_options.dynamical_decoupling.sequence_type = dd_seq
        estimator_options.dynamical_decoupling.extra_slack_distribution = "middle"
        estimator_options.dynamical_decoupling.scheduling_method = "alap"
    if PT:
        estimator_options.twirling.enable_gates = True
        estimator_options.twirling.enable_measure = False
        estimator_options.twirling.num_randomizations = "auto"
        estimator_options.twirling.shots_per_randomization = "auto"
        estimator_options.twirling.strategy = "active-accum"

    estimator = Estimator(mode=mode, options=estimator_options)
    return estimator


def get_ansatz(ansatz_type, feat_dimension, reps=1, entanglement="linear"):
    """Return an ansatz circuit for the specified type.

    Args:
        ansatz_type (str): ``'esu2'``, ``'amp'``, or ``'twolocal'``.
        feat_dimension (int): Number of qubits.
        reps (int): Circuit repetitions. Default 1.
        entanglement (str): Entanglement strategy. Default ``'linear'``.

    Returns:
        Ansatz circuit instance.
    """
    if ansatz_type == "esu2":
        ansatz = EfficientSU2(feat_dimension, ["ry", "rz"], entanglement, reps=reps)
    elif ansatz_type == "amp":
        ansatz = RealAmplitudes(num_qubits=feat_dimension, reps=reps)
    elif ansatz_type == "twolocal":
        ansatz = TwoLocal(feat_dimension, ["ry", "rz"], "cz", entanglement, reps=reps)
    return ansatz


def get_feature_map(feature_map, feat_dimension, reps=1, entanglement="linear", data_map_func=None):
    """Return a Qiskit feature map circuit.

    Args:
        feature_map (str): ``'Z'``, ``'ZZ'``, or ``'P'``.
        feat_dimension (int): Number of qubits.
        reps (int): Circuit repetitions. Default 1.
        entanglement (str): Entanglement strategy. Default ``'linear'``.
        data_map_func (callable, optional): Custom data mapping function.

    Returns:
        tuple: ``(feature_map, feat_dimension)``.
    """
    if feature_map == "Z":
        feature_map = ZFeatureMap(
            feat_dimension, reps=reps, parameter_prefix="a", data_map_func=data_map_func
        )
    elif feature_map == "ZZ":
        feature_map = ZZFeatureMap(
            feature_dimension=feat_dimension,
            reps=reps,
            entanglement=entanglement,
            parameter_prefix="a",
            data_map_func=data_map_func,
        )
    elif feature_map == "P":
        feature_map = PauliFeatureMap(
            feature_dimension=feat_dimension,
            reps=reps,
            entanglement=entanglement,
            data_map_func=data_map_func,
        )

    return feature_map, feat_dimension


def get_optimizer(
    type="COBYLA", max_iter=100, learning_rate_a=None, perturbation_gamma=None, prior_iter=0
):
    """Return a Qiskit optimizer instance.

    Args:
        type (str): ``'SPSA'``, ``'COBYLA'``, ``'GradientDescent'``, or ``'L_BFGS_B'``.
        max_iter (int): Maximum iterations. Default 100.
        learning_rate_a (float, optional): SPSA initial learning rate.
        perturbation_gamma (float, optional): SPSA perturbation exponent.
        prior_iter (int): Prior iterations for SPSA power-series offset.

    Returns:
        Optimizer instance.
    """
    if type == "SPSA":
        if (learning_rate_a != None) & (perturbation_gamma != None):
            # set up the power series
            def learning_rate():
                return spsa.powerseries(learning_rate_a, 0.602, 0)

            gen = learning_rate()
            learning_rates = np.array([next(gen) for _ in range(max_iter + prior_iter)])
            learning_rates = learning_rates[prior_iter : (max_iter + prior_iter)]

            def perturbation():
                return spsa.powerseries(0.2, perturbation_gamma)

            gen = perturbation()
            perturbations = np.array([next(gen) for _ in range(max_iter + prior_iter)])
            perturbations = perturbations[prior_iter : (max_iter + prior_iter)]

            optimizer = SPSA(
                maxiter=max_iter, learning_rate=learning_rates, perturbation=perturbations
            )
        else:
            optimizer = SPSA(maxiter=max_iter)
    elif type == "COBYLA":
        optimizer = COBYLA(maxiter=max_iter)
    elif type == "GradientDescent":
        optimizer = GradientDescent(maxiter=max_iter)
    elif type == "L_BFGS_B":
        optimizer == L_BFGS_B(maxiter=max_iter)

    return optimizer



def retrieve_probabilities(counts: dict) -> list:
    """
    Extract probability predictions from measurement counts.
    
    Converts raw measurement counts from quantum circuit execution into
    probability predictions for binary classification. Handles edge cases
    where only one outcome is observed.
    
    Parameters
    ----------
    counts : dict
        Measurement counts with keys '0' and/or '1'
        Example: {'0': 4123, '1': 4069}
    
    Returns
    -------
    list of float
        [p0, p1] where p0 is probability of class 0 and p1 is probability
        of class 1. Always sums to 1.0.
    
    Notes
    -----
    - Handles missing keys gracefully (assigns probability 0 or 1)
    - If only '0' observed: returns [1.0, 0.0]
    - If only '1' observed: returns [0.0, 1.0]
    - If both observed: returns normalized probabilities
    
    Examples
    --------
    >>> counts = {'0': 6000, '1': 2000}
    >>> retrieve_probabilities(counts)
    [0.75, 0.25]
    
    >>> counts = {'0': 8192}  # Only one outcome
    >>> retrieve_probabilities(counts)
    [1.0, 0.0]
    """
    state_zero = '0'
    state_one = '1'
    
    try:
        p0 = counts[state_zero] / (counts[state_zero] + counts[state_one])
        p1 = 1 - p0
    except KeyError:
        if list(counts.keys())[0] == state_zero:
            p0, p1 = 1.0, 0.0
        else:
            p0, p1 = 0.0, 1.0
    
    return [p0, p1]


def execute_circuit(qc, n_shots: int = 8192, device: str = 'CPU'):
    """
    Execute quantum circuit on Aer simulator.
    
    General-purpose function for executing quantum circuits using the Qiskit
    Aer simulator with statevector method. Useful for custom quantum algorithms
    that need direct circuit execution without the full runtime service setup.
    
    Parameters
    ----------
    qc : QuantumCircuit
        Quantum circuit to execute
    n_shots : int, optional
        Number of measurement shots (default: 8192)
    device : str, optional
        Device type: 'CPU' or 'GPU' (default: 'CPU')
        Note: GPU requires qiskit-aer-gpu installation
    
    Returns
    -------
    dict
        Measurement counts dictionary with bitstring keys and count values
        Example: {'0': 4123, '1': 4069}
    
    Notes
    -----
    - Uses statevector simulation method for exact state evolution
    - Automatically transpiles circuit with optimization level 3
    - Parallel threshold set to 50 qubits for statevector parallelization
    - For hardware execution, use get_backend_session() instead
    
    Examples
    --------
    >>> from qiskit import QuantumCircuit
    >>> from qbiocode.utils import execute_circuit
    >>> qc = QuantumCircuit(2, 2)
    >>> qc.h(0)
    >>> qc.cx(0, 1)
    >>> qc.measure([0, 1], [0, 1])
    >>> counts = execute_circuit(qc, n_shots=1024)
    >>> print(counts)
    {'00': 512, '11': 512}
    """
    from qiskit.compiler import transpile
    from qiskit_aer import AerSimulator

    # Fall back to CPU gracefully if GPU is requested but unavailable
    _device = device
    if _device == "GPU":
        try:
            _backend_test = AerSimulator(method="statevector", device="GPU")
            _backend_test.configuration()  # triggers capability check
        except Exception:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "execute_circuit: GPU requested but not available — falling back to CPU"
            )
            _device = "CPU"

    backend = AerSimulator(method='statevector', device=_device,
                           statevector_parallel_threshold=50)
    tqc = transpile(qc, backend, optimization_level=3)
    result = backend.run([tqc], shots=n_shots).result()
    return result.get_counts(tqc)
