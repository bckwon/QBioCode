import os, re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import optuna
import dill as pickle


class QuantumSage():
    """
    Meta-learning tool for predicting ML model performance from dataset complexity features.

    Trains a separate surrogate regressor (sub-sage) for each combination of
    ML model and performance metric. Use ``train_sub_sages`` to fit and
    ``predict`` to score a new dataset.
    """

    def __init__(self, data_input):
        """
        Initialize QSage with a benchmark DataFrame.

        Args:
            data_input (pd.DataFrame): DataFrame containing dataset complexity
                features, performance metrics, and metadata columns.
        """
        self._columns_data_features = [
            '# Features', '# Samples', 'Feature_Samples_ratio', 'Intrinsic_Dimension',
            'Condition number', 'Fisher Discriminant Ratio', 'Total Correlations',
            'Mutual information', '# Non-zero entries', '# Low variance features',
            'Variation', 'std_var', 'Coefficient of Variation %', 'std_co_of_v',
            'Skewness', 'std_skew', 'Kurtosis', 'std_kurt', 'Mean Log Kernel Density',
            'Isomap Reconstruction Error', 'Fractal dimension', 'Entropy', 'std_entropy',
        ]
        self._columns_metrics = ['accuracy', 'f1_score', 'auc']
        self._columns_metadata = [
            'Dataset', 'embeddings', 'datatype', 'model_embed_datatype',
            'iteration', 'model', 'BestParams_GridSearch', 'Model_Parameters',
        ]

        self._input_data_features_only = data_input[self._columns_data_features]
        self._input_data_metrics = data_input[self._columns_metrics]
        self._input_data_metadata = data_input[self._columns_metadata]

        self._available_models = list(set(self._input_data_metadata['model']))
        if 'none' in self._available_models:
            self._available_models.remove('none')
        self._available_models.sort()
        self._available_embeddings = list(set(self._input_data_metadata['embeddings']))
        self._available_embeddings.sort()
        self._available_metrics = self._columns_metrics
        self._available_metrics.sort()

        self._results_subsages = {}
        self.set_seed()

    def predict(self, input_data, metric='f1_score'):
        """
        Predict model performance for all available models on a new dataset.

        Args:
            input_data (pd.DataFrame): Dataset complexity features matching the
                training columns.
            metric (str): Performance metric to predict. One of
                ``'f1_score'``, ``'auc'``, ``'accuracy'``.

        Returns:
            pd.DataFrame: Predictions for each model sorted by
                ``metric * r2`` descending. Columns: model, metric, r2, metric*r2.
        """
        predictions = []
        for model in self._available_models:
            pred = self._results_subsages[metric][model]['fit_model'].predict(input_data)[0]
            r2 = self._results_subsages[metric][model]['r2']
            predictions.append([model, pred, r2])
        predictions_df = pd.DataFrame(predictions, columns=['model', metric, 'r2'])
        predictions_df[metric + '*r2'] = predictions_df[metric] * predictions_df['r2']
        predictions_df = predictions_df.sort_values(metric + '*r2', ascending=False)
        return predictions_df

    def train_sub_sages(self, test_size=0.2, sage_type='random_forest', n_iter=None, cv=5):
        """
        Train a surrogate regressor for each (model, metric) pair.

        Args:
            test_size (float): Fraction of data used for testing. Default 0.2.
            sage_type (str): Regressor type. One of ``'random_forest'``,
                ``'mlp'``, ``'xgboost_optuna'``. Default ``'random_forest'``.
            n_iter (int, optional): Hyperparameter search iterations (RF),
                max training epochs (MLP), or Optuna trials (XGBoost).
                Uses type-specific defaults when ``None``.
            cv (int): Cross-validation folds for hyperparameter search. Default 5.

        Raises:
            ValueError: If ``sage_type`` is not one of the valid types.

        Results are stored in ``_results_subsages[metric][model]`` with keys:
        ``fit_model``, ``preds``, ``y_test``, ``params``, ``mae``, ``mse``,
        ``rmse``, ``r2``.
        """
        valid_sage_types = ['random_forest', 'mlp', 'xgboost_optuna']
        if sage_type not in valid_sage_types:
            raise ValueError(
                f"Invalid sage_type '{sage_type}'. Must be one of {valid_sage_types}."
            )

        for metric in self._available_metrics:
            print(f"Working on {metric}")
            self._results_subsages[metric] = {}
            for model in self._available_models:
                model_indices = self._input_data_metadata[
                    self._input_data_metadata['model'] == model
                ].index
                X = self._input_data_features_only.loc[model_indices]
                X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
                y = self._input_data_metrics.loc[model_indices][metric].fillna(0).to_numpy()

                print(f"Working on {model}")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=self._seed
                )

                X_train = calculate_SLGH(X_train)
                X_test = calculate_SLGH(X_test)

                if sage_type == 'random_forest':
                    rf_n_iter = n_iter if n_iter is not None else 50
                    self._results_subsages[metric][model] = self._sage_random_forest(
                        X_train, X_test, y_train, y_test, n_iter=rf_n_iter, cv=cv
                    )
                elif sage_type == 'mlp':
                    mlp_n_iter = n_iter if n_iter is not None else 1000
                    self._results_subsages[metric][model] = self._sage_mlp(
                        X_train, X_test, y_train, y_test, n_iter=mlp_n_iter, cv=cv
                    )
                elif sage_type == 'xgboost_optuna':
                    xgb_n_iter = n_iter if n_iter is not None else 100
                    self._results_subsages[metric][model] = self._sage_xgboost_optuna(
                        X_train, X_test, y_train, y_test, n_iter=xgb_n_iter, cv=cv
                    )

    def _sage_mlp(self, X_train, X_test, y_train, y_test, n_iter=1000, cv=5):
        """
        Train an MLP regressor with grid search over hyperparameters.

        Args:
            X_train, X_test (pd.DataFrame): Train/test complexity features.
            y_train, y_test (np.ndarray): Train/test performance values.
            n_iter (int): Maximum training epochs. Default 1000.
            cv (int): Cross-validation folds. Default 5.

        Returns:
            dict: Keys — ``fit_model``, ``preds``, ``y_test``, ``params``,
                ``mae``, ``mse``, ``rmse``, ``r2``.
        """
        from sklearn.model_selection import GridSearchCV

        param_grid = {
            'hidden_layer_sizes': [(32, 10), (64, 32), (100,), (50, 25)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'lbfgs'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
        }

        mlp = MLPRegressor(
            batch_size='auto',
            learning_rate_init=0.001,
            max_iter=n_iter,
            random_state=self._seed,
            n_iter_no_change=10,
            early_stopping=True,
            validation_fraction=0.1,
        )

        mlp_grid = GridSearchCV(
            estimator=mlp,
            param_grid=param_grid,
            cv=cv,
            n_jobs=-1,
            scoring='r2',
        )

        X_train = X_train.astype(np.float64).replace([np.inf, -np.inf], np.nan).fillna(0)
        X_test = X_test.astype(np.float64).replace([np.inf, -np.inf], np.nan).fillna(0)

        mlp_grid.fit(X_train, y_train)
        preds = mlp_grid.predict(X_test)
        params = mlp_grid.best_params_

        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)

        return {
            'fit_model': mlp_grid,
            'preds': preds,
            'y_test': y_test,
            'params': params,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
        }

    def _sage_xgboost_optuna(self, X_train, X_test, y_train, y_test, n_iter=100, cv=5):
        """
        Train an XGBoost regressor with Optuna hyperparameter optimization.

        Args:
            X_train, X_test (pd.DataFrame): Train/test complexity features.
            y_train, y_test (np.ndarray): Train/test performance values.
            n_iter (int): Number of Optuna trials. Default 100.
            cv (int): Cross-validation folds per trial. Default 5.

        Returns:
            dict: Keys — ``fit_model``, ``preds``, ``y_test``, ``params``,
                ``mae``, ``mse``, ``rmse``, ``r2``, ``study``.
        """
        from sklearn.model_selection import cross_val_score

        X_train = X_train.astype(np.float64).replace([np.inf, -np.inf], np.nan).fillna(0)
        X_test = X_test.astype(np.float64).replace([np.inf, -np.inf], np.nan).fillna(0)

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': self._seed,
                'n_jobs': -1,
                'verbosity': 0,
            }
            model = xgb.XGBRegressor(**params)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2', n_jobs=-1)
            return scores.mean()

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self._seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_iter, show_progress_bar=False)

        best_params = study.best_params
        best_params.update({'random_state': self._seed, 'n_jobs': -1, 'verbosity': 0})

        best_model = xgb.XGBRegressor(**best_params)
        best_model.fit(X_train, y_train)
        preds = best_model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)

        return {
            'fit_model': best_model,
            'preds': preds,
            'y_test': y_test,
            'params': best_params,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'study': study,
        }

    def _sage_random_forest(self, X_train, X_test, y_train, y_test, n_iter=50, cv=5):
        """
        Train a Random Forest regressor with randomized hyperparameter search.

        Args:
            X_train, X_test (pd.DataFrame): Train/test complexity features.
            y_train, y_test (np.ndarray): Train/test performance values.
            n_iter (int): Randomized search iterations. Default 50.
            cv (int): Cross-validation folds. Default 5.

        Returns:
            dict: Keys — ``fit_model``, ``preds``, ``y_test``, ``params``,
                ``mae``, ``mse``, ``rmse``, ``r2``.
        """
        param_distributions = {
            'n_estimators': np.arange(100, 1000, 100),
            'max_depth': np.arange(5, 20),
            'min_samples_split': np.arange(2, 10),
            'min_samples_leaf': np.arange(1, 5),
            'bootstrap': [True, False],
        }

        rf = RandomForestRegressor(random_state=self._seed)
        rf_random = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            random_state=self._seed,
            n_jobs=-1,
        )

        X_train = X_train.astype(np.float64).replace([np.inf, -np.inf], np.nan).fillna(0)
        X_test = X_test.astype(np.float64).replace([np.inf, -np.inf], np.nan).fillna(0)
        rf_random.fit(X_train, y_train)
        preds = rf_random.predict(X_test)
        params = rf_random.best_params_

        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)

        return {
            'fit_model': rf_random,
            'preds': preds,
            'y_test': y_test,
            'params': params,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
        }

    def plot_results(self, figsize=(6, 4), saveFile=''):
        """
        Plot bar charts and scatter plots of sub-sage prediction performance.

        Generates one bar plot (MAE, MSE, RMSE, R²) and one scatter plot
        (predicted vs actual) per metric. Saves to PDF if ``saveFile`` is set.

        Args:
            figsize (tuple): Figure size. Default (6, 4).
            saveFile (str): Base path for output PDFs. If empty, displays plots.
        """
        results = []
        preds = pd.DataFrame()
        for metric in self._available_metrics:
            for model in self._available_models:
                scores = pd.Series(
                    self._results_subsages[metric][model].values(),
                    index=self._results_subsages[metric][model].keys(),
                )
                results.append([model, metric] + list(scores[['mae', 'mse', 'rmse', 'r2']]))
                p = self._results_subsages[metric][model]['preds']
                y = self._results_subsages[metric][model]['y_test']
                preds = pd.concat([
                    preds,
                    pd.DataFrame(
                        [[model] * len(p), [metric] * len(p), p, y],
                        index=['model', 'metric', 'pred', 'y_test'],
                    ).transpose(),
                ])

        if not results:
            print("No results to plot. Run train_sub_sages first.")
            return

        results_df = pd.DataFrame(results, columns=['model', 'metric', 'mae', 'mse', 'rmse', 'r2'])
        results_df = results_df.melt(id_vars=['model', 'metric'])

        for metric in self._available_metrics:
            plt.figure(figsize=figsize)
            metric_data = results_df[results_df['metric'] == metric]
            sns.barplot(data=metric_data, x='variable', y='value', hue='model', hue_order=self._available_models)  # type: ignore[arg-type]
            plt.title("Predictive performance for each model for " + metric)
            plt.xlabel("Metric")
            plt.ylabel("Value")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            if saveFile != '':
                plt.savefig(re.sub('.pdf', '', saveFile) + '_' + metric + '_barplot.pdf', bbox_inches='tight')
            plt.show()
            plt.close()

            toPlot = preds[preds['metric'] == metric]
            plt.figure(figsize=figsize)
            plt.title("Predictive performance for each model for " + metric)
            sns.scatterplot(data=toPlot, x='y_test', y='pred', hue='model')  # type: ignore[arg-type]
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            if saveFile != '':
                plt.savefig(re.sub('.pdf', '', saveFile) + '_' + metric + '_scatterplot.pdf', bbox_inches='tight')
            plt.show()
            plt.close()

    def set_seed(self, seed=42):
        """Set the random seed for reproducibility."""
        self._seed = seed


def calculate_SLGH(df, train_pct=0.7):
    """
    Compute the Scaled Latent Geometric Hardness (SLGH) feature and append it to df.

    Args:
        df (pd.DataFrame): Dataset complexity features. Must contain
            ``'Intrinsic_Dimension'``, ``'Fisher Discriminant Ratio'``,
            and ``'# Samples'`` columns.
        train_pct (float): Fraction of samples used for training. Default 0.7.

    Returns:
        pd.DataFrame: Input DataFrame with an added ``'SLGH'`` column.
    """
    id_col = 'Intrinsic_Dimension'
    fdr_col = 'Fisher Discriminant Ratio'
    num_samples = '# Samples'
    n_train = np.ceil(df[num_samples] * train_pct)
    eps = 1e-8

    df = pd.DataFrame(df)
    df['SLGH'] = (-np.log(df[id_col] + eps) - np.log(1.0 + df[fdr_col] * n_train))
    return df


def main():
    """Command-line entry point for QSage."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description='QSage: meta-learning model selection for QBioCode benchmarks.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  qsage --input compiled_results.csv --output sage_results/
  qsage --input data.csv --output results/ --model-type xgboost_optuna --n-iter 200
  qsage --input data.csv --output results/ --seed 123
        """,
    )

    parser.add_argument('--input', '-i', required=True,
                        help='Input CSV with dataset features and performance metrics')
    parser.add_argument('--output', '-o', required=True,
                        help='Output directory for results and plots')
    parser.add_argument('--seed', '-s', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--model-type', default='random_forest',
                        choices=['rf', 'mlp', 'random_forest', 'xgboost', 'xgboost_optuna'],
                        help='Sub-sage model type (default: random_forest)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test fraction (default: 0.2)')
    parser.add_argument('--n-iter', type=int, default=None,
                        help='Hyperparameter search iterations / training epochs / Optuna trials')
    parser.add_argument('--cv', type=int, default=5,
                        help='Cross-validation folds (default: 5)')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Model type: {args.model_type}  |  Test size: {args.test_size}  |  Seed: {args.seed}  |  CV: {args.cv}")

    try:
        data = pd.read_csv(args.input)
        data['embeddings'] = ['None' if str(x) == 'nan' else x for x in data['embeddings']]
        print(f"Loaded {len(data)} rows, {len(data.columns)} columns")
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        sage = QuantumSage(data)
        sage.set_seed(args.seed)
        print(f"Models: {sage._available_models}")
        print(f"Metrics: {sage._available_metrics}")
    except Exception as e:
        print(f"Error initializing QSage: {e}", file=sys.stderr)
        sys.exit(1)

    sage_type_map = {
        'rf': 'random_forest',
        'random_forest': 'random_forest',
        'mlp': 'mlp',
        'xgboost': 'xgboost_optuna',
        'xgboost_optuna': 'xgboost_optuna',
    }
    sage_type = sage_type_map[args.model_type]

    try:
        sage.train_sub_sages(
            test_size=args.test_size,
            sage_type=sage_type,
            n_iter=args.n_iter,
            cv=args.cv,
        )
    except Exception as e:
        print(f"Error training sub-sages: {e}", file=sys.stderr)
        sys.exit(1)

    pickle.dump(sage, open(os.path.join(args.output, 'trained_sage.pkl'), 'wb'))

    try:
        output_file = os.path.join(args.output, 'sage_results.pdf')
        sage.plot_results(saveFile=output_file)
        print(f"Plots saved to: {output_file}")
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}", file=sys.stderr)

    try:
        results_summary = []
        for metric in sage._available_metrics:
            for model in sage._available_models:
                if metric in sage._results_subsages and model in sage._results_subsages[metric]:
                    result = sage._results_subsages[metric][model]
                    results_summary.append({
                        'model': model, 'metric': metric,
                        'mae': result['mae'], 'mse': result['mse'],
                        'rmse': result['rmse'], 'r2': result['r2'],
                    })
        if results_summary:
            results_df = pd.DataFrame(results_summary)
            results_file = os.path.join(args.output, 'sage_summary.csv')
            results_df.to_csv(results_file, index=False)
            print(f"Summary saved to: {results_file}")
            print(results_df.to_string(index=False))
    except Exception as e:
        print(f"Warning: Could not save results summary: {e}", file=sys.stderr)

    print("QSage training complete.")


if __name__ == "__main__":
    main()
