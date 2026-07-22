from kedro.pipeline import Pipeline

from lcl_evaluation_kedro.pipelines.experiments.v00_initial.pipeline import (
    create_pipeline as create_v00_initial_pipeline,
)
from lcl_evaluation_kedro.pipelines.experiments.v01_changed_and_added_model.pipeline import (
    create_pipeline as create_v01_changed_and_added_model_pipeline,
)
from lcl_evaluation_kedro.pipelines.experiments.v02_changed_dataset_to_logatec.pipeline import (
    create_pipeline as create_v02_changed_dataset_to_logatec_pipeline,
)
from lcl_evaluation_kedro.pipelines.experiments.v03_added_split_and_metric.pipeline import (
    create_pipeline as create_v03_added_split_and_metric_pipeline,
)
from lcl_evaluation_kedro.pipelines.experiments.v04_added_automl_model.pipeline import (
    create_pipeline as create_v04_added_automl_model_pipeline,
)


def register_pipelines() -> dict[str, Pipeline]:
    automl = create_v04_added_automl_model_pipeline()
    return {
        "00-Initial": create_v00_initial_pipeline(),
        "01-Changed_and_added_model": create_v01_changed_and_added_model_pipeline(),
        "02-Changed_dataset_to_logatec": create_v02_changed_dataset_to_logatec_pipeline(),
        "03-Added_split_and_metric": create_v03_added_split_and_metric_pipeline(),
        "04-Added_automl_model": automl,
        "Benchmarking": create_v04_added_automl_model_pipeline(),
        "__default__": automl,
    }
