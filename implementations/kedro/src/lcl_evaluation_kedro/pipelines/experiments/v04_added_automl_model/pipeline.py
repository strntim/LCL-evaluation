from kedro.pipeline import Pipeline, node, pipeline

from .nodes import create_splits, featurize, grid_search, prepare, run_automl


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(prepare, ["raw_data", "params:dataset"], "prepared_data", name="prepare"),
            node(
                featurize,
                ["prepared_data", "params:dataset"],
                ["features", "targets"],
                name="featurize",
            ),
            node(
                create_splits,
                ["features", "targets", "params:dataset", "params:splitters"],
                "split_indices",
                name="split",
            ),
            node(
                grid_search,
                [
                    "features",
                    "targets",
                    "split_indices",
                    "params:dataset",
                    "params:splitters",
                    "params:models",
                    "params:evaluation",
                ],
                ["gridsearch_models", "gridsearch_results"],
                name="gridsearch",
            ),
            node(
                run_automl,
                [
                    "features",
                    "targets",
                    "split_indices",
                    "params:dataset",
                    "params:splitters",
                    "params:automl",
                    "gridsearch_results",
                ],
                "automl_results",
                name="automl",
            ),
        ]
    )
