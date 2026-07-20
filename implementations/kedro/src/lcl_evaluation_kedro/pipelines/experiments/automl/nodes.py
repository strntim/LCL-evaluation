import gc
import math
import os
import re
import shutil
import time
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _materialize(partitions: Mapping[str, Any]) -> dict[str, Any]:
    return {
        partition: value() if callable(value) else value
        for partition, value in partitions.items()
    }


def _prepare_umu(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Column7",
        "Column8",
        "Column14",
        "Column15",
        "Column42",
        "Column43",
        "Column45",
        "Column46",
        "Column47",
        "Column48",
        "Column87",
        "Column88",
        "Column78",
        "Column79",
    ]
    frame = frame.loc[:, columns].copy()
    frame.columns = frame.iloc[0]
    frame = frame.iloc[1:].copy()
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame.columns = [str(column).replace("nas_value_nr5g_", "") for column in frame.columns]
    frame = frame.dropna()
    frame = frame.loc[:, frame.nunique() > 1]
    return frame.reset_index(drop=True)


def _prepare_logatec(raw: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for position, measurements in raw.items():
        location = tuple(int(value) for value in re.findall(r"\d+", position))
        if len(location) == 1:
            location = (3, *location)
        if len(location) != 2:
            raise ValueError(f"Invalid LOG-a-TEC position: {position}")
        pos_x, pos_y = location
        for device_id, samples in measurements.items():
            for sample in samples:
                rows.append(
                    {
                        "pos_x": pos_x,
                        "pos_y": pos_y,
                        "node": int(device_id),
                        "timestamp": sample["timestamp"],
                        "value": sample["rss"],
                    }
                )

    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="s", origin="unix")
    frame = frame.astype({"pos_x": "uint8", "pos_y": "uint8", "value": "int8", "node": "uint8"})
    frame = (
        frame.groupby(["pos_x", "pos_y", "node", "timestamp"], as_index=False)["value"]
        .mean()
        .pivot(index=["timestamp", "pos_x", "pos_y"], columns="node", values="value")
        .reset_index()
    )
    frame.columns = [
        f"node{column}" if isinstance(column, (int, np.integer)) else str(column)
        for column in frame.columns
    ]
    return frame.fillna(-180).drop(columns="timestamp")


def prepare(raw_data: Any, dataset: dict[str, Any]) -> dict[str, pd.DataFrame]:
    if dataset["kind"] == "umu":
        return {"umu": _prepare_umu(raw_data)}
    raw_partitions = _materialize(raw_data)
    return {
        subset: _prepare_logatec(raw_partitions[subset])
        for subset in dataset["subsets"]
    }


def featurize(
    prepared_data: Mapping[str, Any], dataset: dict[str, Any]
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    prepared = _materialize(prepared_data)
    features = {}
    targets = {}
    for subset in dataset["subsets"]:
        frame = prepared[subset].copy()
        if dataset["kind"] == "umu":
            origin_lat = frame["gpsd_tpv_lat"].min()
            origin_lon = frame["gpsd_tpv_lon"].min()
            radius = 6_378_137
            frame["target_x"] = np.radians(frame["gpsd_tpv_lat"] - origin_lat) * radius
            frame["target_y"] = (
                np.radians(frame["gpsd_tpv_lon"] - origin_lon)
                * radius
                * math.cos(math.radians(origin_lat))
            )
            frame = frame.drop(columns=["gpsd_tpv_lat", "gpsd_tpv_lon"])
        else:
            frame["pos_x"] = (frame["pos_x"] - 1) * 1.2
            frame["pos_y"] = (frame["pos_y"] - 1) * 1.2
            frame = frame.rename(columns={"pos_x": "target_x", "pos_y": "target_y"})

        target_columns = ["target_x", "target_y"]
        features[subset] = frame.drop(columns=target_columns)
        targets[subset] = frame[target_columns]
    return features, targets


def create_splits(
    features: Mapping[str, Any],
    targets: Mapping[str, Any],
    dataset: dict[str, Any],
    splitters: dict[str, dict[str, Any]],
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    from sklearn import model_selection

    feature_partitions = _materialize(features)
    target_partitions = _materialize(targets)
    results = {}
    for subset in dataset["subsets"]:
        for name, settings in splitters.items():
            options = dict(settings)
            splitter_type = options.pop("type")
            splitter_class = getattr(model_selection, splitter_type)
            splitter = splitter_class(**options)
            results[f"{subset}/{name}"] = list(
                splitter.split(feature_partitions[subset], target_partitions[subset])
            )
    return results


def _load_class(path: str):
    module_name, class_name = path.rsplit(".", 1)
    return getattr(import_module(module_name), class_name)


def _scorers(metrics: list[str]) -> dict[str, Any]:
    from sklearn.metrics import make_scorer, r2_score, root_mean_squared_error

    available = {
        "rmse": make_scorer(root_mean_squared_error, greater_is_better=False),
        "r_squared": make_scorer(r2_score, greater_is_better=True),
    }
    return {metric: available[metric] for metric in metrics}


def grid_search(
    features: Mapping[str, Any],
    targets: Mapping[str, Any],
    split_indices: Mapping[str, Any],
    dataset: dict[str, Any],
    splitters: dict[str, dict[str, Any]],
    models: dict[str, dict[str, Any]],
    evaluation: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    from sklearn.model_selection import GridSearchCV

    feature_partitions = _materialize(features)
    target_partitions = _materialize(targets)
    splits = _materialize(split_indices)
    scorers = _scorers(evaluation["metrics"])
    best_models = {}
    results = {}

    for subset in dataset["subsets"]:
        for split_name in splitters:
            for model_name, settings in models.items():
                print(f"Grid search: {subset} / {split_name} / {model_name}", flush=True)
                estimator = _load_class(settings["class"])(**settings.get("parameters", {}))
                search = GridSearchCV(
                    estimator=estimator,
                    param_grid=settings["grid"],
                    scoring=scorers,
                    refit=evaluation["refit"],
                    cv=splits[f"{subset}/{split_name}"],
                    n_jobs=evaluation["workers"],
                    error_score="raise",
                )
                search.fit(feature_partitions[subset], target_partitions[subset])
                key = f"{subset}/{split_name}/{model_name}"
                best_models[key] = search.best_estimator_
                results[f"{key}-results"] = pd.DataFrame(search.cv_results_)
    return best_models, results


def _prediction_matrix(predictions: Any) -> np.ndarray:
    if isinstance(predictions, list):
        return np.column_stack([np.asarray(value).reshape(-1) for value in predictions])
    matrix = np.asarray(predictions)
    return matrix.reshape(len(matrix), -1)


def _automl_candidate(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    indices: list[tuple[np.ndarray, np.ndarray]],
    subset: str,
    split_name: str,
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    import autokeras as ak
    import keras
    import tensorflow as tf
    from sklearn.metrics import r2_score, root_mean_squared_error
    from sklearn.model_selection import train_test_split

    tf.config.experimental.enable_op_determinism()
    keras.utils.set_random_seed(settings["seed"])
    keras.backend.clear_session()

    x = features.to_numpy()
    y = targets.to_numpy()
    y_heads = [y[:, index : index + 1] for index in range(y.shape[1])]
    train, validation = train_test_split(
        np.arange(len(x)),
        test_size=settings["validation_size"],
        random_state=settings["seed"],
        shuffle=True,
    )

    directory = Path(settings["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    project_name = f"{subset}-{split_name}"
    auto_model = ak.AutoModel(
        inputs=[ak.Input(name="data_input")],
        outputs=[ak.RegressionHead(name="x_out"), ak.RegressionHead(name="y_out")],
        seed=settings["seed"],
        max_trials=settings["max_trials"],
        overwrite=True,
        directory=directory,
        project_name=project_name,
    )
    callback = keras.callbacks.EarlyStopping(
        patience=settings["patience"], min_delta=settings["min_delta"]
    )
    auto_model.fit(
        [x[train]],
        [head[train] for head in y_heads],
        validation_data=([x[validation]], [head[validation] for head in y_heads]),
        epochs=settings["epochs"],
        callbacks=[callback],
        verbose=settings["verbose"],
    )

    trial_count = max(1, int(len(auto_model.tuner.oracle.trials) * settings["top_fraction"]))
    trials = auto_model.tuner.oracle.get_best_trials(trial_count)
    reports = []
    for trial in trials:
        trained_model = auto_model.tuner.load_model(trial)
        optimizer = trained_model.optimizer
        del trained_model
        scores = {"rmse": [], "r_squared": []}
        fit_times = []
        score_times = []

        for train_indices, test_indices in indices:
            keras.utils.set_random_seed(settings["seed"])
            model = auto_model.tuner.hypermodel.build(trial.hyperparameters)
            model.compile(
                optimizer=type(optimizer).from_config(optimizer.get_config()),
                loss="mse",
            )
            started = time.perf_counter()
            model.fit(
                [x[train_indices]],
                [head[train_indices] for head in y_heads],
                validation_data=(
                    [x[test_indices]],
                    [head[test_indices] for head in y_heads],
                ),
                epochs=settings["epochs"],
                callbacks=[
                    keras.callbacks.EarlyStopping(
                        patience=settings["patience"],
                        min_delta=settings["min_delta"],
                    )
                ],
                verbose=settings["verbose"],
            )
            fit_times.append(time.perf_counter() - started)

            started = time.perf_counter()
            predictions = _prediction_matrix(
                model.predict([x[test_indices]], batch_size=32, verbose=settings["verbose"])
            )
            score_times.append(time.perf_counter() - started)
            expected = y[test_indices]
            scores["rmse"].append(root_mean_squared_error(expected, predictions))
            scores["r_squared"].append(r2_score(expected, predictions))
            del model
            keras.backend.clear_session()
            gc.collect()

        reports.append(
            {
                "scores": {
                    name: {"mean": float(np.mean(values)), "std": float(np.std(values))}
                    for name, values in scores.items()
                },
                "params": dict(trial.hyperparameters.values),
                "fit_time": {
                    "mean": float(np.mean(fit_times)),
                    "std": float(np.std(fit_times)),
                },
                "score_time": {
                    "mean": float(np.mean(score_times)),
                    "std": float(np.std(score_times)),
                },
            }
        )

    shutil.rmtree(directory / project_name, ignore_errors=True)
    keras.backend.clear_session()
    gc.collect()
    return reports


def run_automl(
    features: Mapping[str, Any],
    targets: Mapping[str, Any],
    split_indices: Mapping[str, Any],
    dataset: dict[str, Any],
    splitters: dict[str, dict[str, Any]],
    automl: dict[str, Any],
    _gridsearch_results: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    feature_partitions = _materialize(features)
    target_partitions = _materialize(targets)
    splits = _materialize(split_indices)
    results = {}
    for subset in dataset["subsets"]:
        for split_name in splitters:
            print(f"AutoML: {subset} / {split_name}", flush=True)
            key = f"{subset}/{split_name}/ExampleModel-results"
            results[key] = _automl_candidate(
                feature_partitions[subset],
                target_partitions[subset],
                splits[f"{subset}/{split_name}"],
                subset,
                split_name,
                automl,
            )
    return results
