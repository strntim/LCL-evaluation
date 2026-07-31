import re
from collections.abc import Mapping
from importlib import import_module
from typing import Any

import numpy as np
import pandas as pd


def _materialize(partitions: Mapping[str, Any]) -> dict[str, Any]:
    return {partition: load() for partition, load in partitions.items()}


def prepare(
    raw_data: Mapping[str, Any], dataset: dict[str, Any]
) -> dict[str, pd.DataFrame]:
    raw_partitions = _materialize(raw_data)
    prepared = {}
    for subset in dataset["subsets"]:
        rows = []
        for position, measurements in raw_partitions[subset].items():
            location = tuple(int(value) for value in re.findall(r"\d+", position))
            if len(location) == 1:
                location = (3, *location)
            assert len(location) == 2, f"location identifier is not length 2: {location}"
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
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="s", origin="unix").astype("datetime64[s]")
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
        prepared[subset] = frame.fillna(-180).drop(columns="timestamp")
    return prepared


def featurize(
    prepared_data: Mapping[str, Any], dataset: dict[str, Any]
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    prepared = _materialize(prepared_data)
    features = {}
    targets = {}
    for subset in dataset["subsets"]:
        frame = prepared[subset].copy()
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
                estimator = _load_class(settings["class"])(**settings["parameters"])
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
