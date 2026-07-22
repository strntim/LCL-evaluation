import math
from collections.abc import Mapping
from importlib import import_module
from typing import Any

import numpy as np
import pandas as pd


def _materialize(partitions: Mapping[str, Any]) -> dict[str, Any]:
    return {
        partition: value() if callable(value) else value
        for partition, value in partitions.items()
    }


def prepare(raw_data: pd.DataFrame, dataset: dict[str, Any]) -> dict[str, pd.DataFrame]:
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
    frame = raw_data.loc[:, columns].copy()
    frame.columns = frame.iloc[0]
    frame = frame.iloc[1:].copy()
    for column in frame.columns:
        if frame[column].dtype == "object":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame.columns = [str(column).replace("nas_value_nr5g_", "") for column in frame.columns]
    frame = frame.dropna()
    frame = frame.loc[:, frame.nunique() > 1]
    return {dataset["subsets"][0]: frame}


def featurize(
    prepared_data: Mapping[str, Any], dataset: dict[str, Any]
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    prepared = _materialize(prepared_data)
    features = {}
    targets = {}
    for subset in dataset["subsets"]:
        frame = prepared[subset].copy()
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
            splitter = getattr(model_selection, splitter_type)(**options)
            results[f"{subset}/{name}"] = list(
                splitter.split(feature_partitions[subset], target_partitions[subset])
            )
    return results


def _load_class(path: str):
    module_name, class_name = path.rsplit(".", 1)
    return getattr(import_module(module_name), class_name)


def _scorers(metrics: list[str]) -> dict[str, Any]:
    from sklearn.metrics import make_scorer, root_mean_squared_error

    available = {
        "rmse": make_scorer(root_mean_squared_error, greater_is_better=False),
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
