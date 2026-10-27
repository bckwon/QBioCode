import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timezone
import json
import pickle
import os
import re
import csv
import time
import sys

import hydra
from hydra import compose, initialize_config_dir

from sklearn.model_selection import train_test_split

from qiskit_algorithms.utils import algorithm_globals

import sys
dir_home = re.sub( 'QBioCode.*', 'QBioCode', os.getcwd() )
sys.path.append( dir_home )

from qbiocode import scaler_fn, feature_encoding
from qbiocode import get_embeddings
from qbiocode import evaluate
from qbiocode import model_run

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, 'configs')

current_dir = os.getcwd()

@hydra.main(config_path=None, config_name='config', version_base='1.1')
def main(args):
    """
    Run QProfiler: benchmark ML models on CSV datasets.

    Reads datasets from ``folder_path``, applies feature encoding, splits data,
    applies scaling and embeddings, and evaluates models. Writes per-iteration
    results to ``ModelResults.csv`` and a summary pickle to ``results.pkl``.

    Args:
        args (dict): Hydra configuration. Key fields:
            - folder_path (str): Path to dataset directory (relative to repo root).
            - file_dataset (str): Dataset filename or 'ALL'.
            - model (list): Models to evaluate.
            - backend (str): Quantum backend ('simulator' or IBM backend name).
            - embeddings (list): Embedding methods to apply.
            - n_components (int): Embedding dimensionality.
            - seed (int): NumPy random seed.
            - q_seed (int): Qiskit algorithm random seed.
            - test_size (float): Fraction of data held out for testing.
            - iter (int): Number of train/test split iterations.
            - scaling (list): Scaling method(s) to apply.
            - stratify (list): If non-empty, use stratified splitting.
    """
    beg_time = time.time()
    log = logging.getLogger(__name__)
    log.info(f"Main program initiated")
    log.info(f"The number of ML methods being parallelized is {min(args['n_jobs'], len(args['model']))}")
    log.info(f"Chosen backend for quantum algorithms is: {args['backend']}")

    folder_path = args['folder_path'].replace('/', os.sep).replace('\\', os.sep)
    path_to_input = os.path.join(dir_home, folder_path)
    if args['file_dataset'] == 'ALL':
        input_files = [file for file in os.listdir(path_to_input) if file.endswith('csv')]
    else:
        input_files = [file for file in os.listdir(path_to_input) if file in args['file_dataset'] and file.endswith('csv')]

    appended_raw_data_eval = []

    file_count = 0
    for file in sorted(input_files):
        print(f"Processing file: {file}")
        np.random.seed(args['seed'])
        algorithm_globals.random_seed = args['q_seed']

        dataset_start_time = time.time()
        TIMESTAMP = datetime.now(timezone.utc)
        dataset_timestamp_str = TIMESTAMP.strftime("%Y_%m_%d_%H_%M_%S_%f")

        summary = {}
        model_results = {}
        summary.update({'Dataset': file})
        model_results.update({'Dataset': file})

        if args.get('index_col', False):
            rawdata = pd.read_csv(os.path.join(path_to_input, file), sep=r'\t|,', index_col=0)
        else:
            rawdata = pd.read_csv(os.path.join(path_to_input, file), sep=r'\t|,')

        X = rawdata.iloc[:, :-1].to_numpy()
        y = rawdata.iloc[:, -1:].to_numpy()
        y_encoded = feature_encoding(y, feature_encoding='OrdinalEncoder')
        y_encoded = y_encoded.reshape(-1)
        y_encoded = y_encoded.astype(int)
        y_map = dict(zip(y_encoded.astype(str), y.tolist()))
        summary.update({'label_mapping': y_map})

        n_classes = len(np.unique(y_encoded))
        if n_classes != 2:
            log.warning(f"Dataset {file} has {n_classes} classes. QProfiler is optimized for binary classification.")
            print(f"\nWARNING: Dataset '{file}' has {n_classes} classes. Multi-class support is experimental.\n")

        df_dataset = pd.DataFrame(X)
        try:
            raw_data_eval = evaluate(df_dataset, y_encoded, file)
            appended_raw_data_eval.append(raw_data_eval)
        except Exception as _eval_exc:
            log.warning(f"Dataset evaluation skipped for '{file}' — {_eval_exc}")

        if appended_raw_data_eval:
            all_raw_data_evaluation = pd.concat(appended_raw_data_eval)
            all_raw_data_evaluation.to_csv('RawDataEvaluation.csv', index=False)

        log.info(f"Started processing data set {file}")
        log.info(f"Dataset has {n_classes} classes: {np.unique(y_encoded).tolist()}")

        use_stratify = args.get('stratify', [])
        test_size = args['test_size']
        iter = 0
        for iter in range(args['iter']):
            iter = iter + 1
            iter_start_time = time.time()

            if use_stratify and len(use_stratify) > 0:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, stratify=y_encoded, test_size=test_size
                )
                log.info(f"Begin processing iteration {iter} of {args['iter']} with stratified sampling")
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=test_size
                )
                log.info(f"Begin processing iteration {iter} of {args['iter']} without stratification")

            if 'True' in args['scaling']:
                X_train = scaler_fn(X_train, scaling='MinMaxScaler')
                X_test = scaler_fn(X_test, scaling='MinMaxScaler')

            for embed in args['embeddings']:
                if embed == 'none':
                    log.info(f"No embedding applied in this iteration")
                else:
                    log.info(f"Embedding applied: {embed}")
                X_train_emb, X_test_emb = get_embeddings(
                    embed,
                    X_train,
                    X_test,
                    n_components=args["n_components"],
                    method=None,
                    y_train=y_train,
                    random_state=args.get("seed", 42),
                )
                summary.update({'embeddings': embed})
                model_results.update({'embeddings': embed})

                df_dataset = pd.DataFrame(X_train_emb)
                evaluate_data = evaluate(df_dataset, y_train, file)
                evaluate_data_listofdict = evaluate_data.to_dict(orient='records')
                evaluate_data_dict = {k: v for d in evaluate_data_listofdict for k, v in d.items()}
                model_results.update(evaluate_data_dict)
                summary.update({'iteration': iter})
                model_results.update({'iteration': iter})
                data_key = '_'.join([re.sub(r'\..*', '', file), embed, str(args["n_components"]), str(iter)])
                summary.update(model_run(X_train_emb, X_test_emb, y_train, y_test, data_key, args))

                for outerkey, outervalue in summary.items():
                    if outerkey.startswith("results_"):
                        for inner_key, inner_value in outervalue[0].items():
                            update = {inner_key: inner_value}
                            model_results.update(**update)
                        with open('ModelResults.csv', 'a', newline='') as csvfile:
                            model_results_write = csv.writer(csvfile)
                            if csvfile.tell() == 0:
                                model_results_write.writerow(model_results.keys())
                            model_results_write.writerow(model_results.values())

                try:
                    with open("results.pkl", "rb") as pklfile:
                        results = pickle.load(pklfile)
                except FileNotFoundError:
                    results = []
                results.append(summary)
                with open('results.pkl', 'wb') as pklfile:
                    pickle.dump(results, pklfile)

            iter_run_time = time.time() - iter_start_time
            log.info(f"Run time for iteration {iter}: {iter_run_time}")

        file_count += 1
        dataset_run_time = time.time() - dataset_start_time
        log.info(f"Total run time for {file}: {dataset_run_time}")
        log.info(f"Processed {file_count} of {len(input_files)} datasets")
        log.info(f"{len(input_files) - file_count} datasets remaining")

    total_run_time = time.time() - beg_time
    log.info(f"Total program run time: {total_run_time}")

if __name__ == "__main__":
    main()
