from kedro.pipeline import Pipeline

from lcl_evaluation_kedro.pipelines.experiments.automl.pipeline import (
    create_pipeline as create_automl_pipeline,
)
from lcl_evaluation_kedro.pipelines.experiments.changed_dataset.pipeline import (
    create_pipeline as create_changed_dataset_pipeline,
)
from lcl_evaluation_kedro.pipelines.experiments.changed_model.pipeline import (
    create_pipeline as create_changed_model_pipeline,
)
from lcl_evaluation_kedro.pipelines.experiments.initial.pipeline import (
    create_pipeline as create_initial_pipeline,
)
from lcl_evaluation_kedro.pipelines.experiments.split_metric.pipeline import (
    create_pipeline as create_split_metric_pipeline,
)


def register_pipelines() -> dict[str, Pipeline]:
    automl = create_automl_pipeline()
    return {
        "00-Initial": create_initial_pipeline(),
        "01-Changed_and_added_model": create_changed_model_pipeline(),
        "02-Changed_dataset_to_logatec": create_changed_dataset_pipeline(),
        "03-Added_split_and_metric": create_split_metric_pipeline(),
        "04-Added_automl_model": automl,
        "Benchmarking": create_automl_pipeline(),
        "__default__": automl,
    }
