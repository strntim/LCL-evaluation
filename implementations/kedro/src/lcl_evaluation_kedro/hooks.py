import os
from collections.abc import Iterable
from pathlib import Path

from kedro.framework.hooks import hook_impl


def _dataset_names(outputs: object) -> set[str]:
    if isinstance(outputs, str):
        return {outputs}
    if isinstance(outputs, dict):
        return {str(value) for value in outputs.values()}
    if isinstance(outputs, Iterable):
        return {str(value) for value in outputs}
    return set()


class BenchmarkHook:
    def __init__(self) -> None:
        self._stage: str | None = None
        self._remaining_outputs: set[str] = set()

    @property
    def output_directory(self) -> Path | None:
        value = os.environ.get("KEDRO_BENCHMARK_DIR")
        return Path(value) if value else None

    def _start(self, node) -> None:
        if self.output_directory is None or self._stage is not None:
            return
        from benchmarking.performance import start_resource_monitor

        start_resource_monitor()
        self._stage = node.name
        self._remaining_outputs = _dataset_names(node.outputs)

    def _stop(self, ignore_errors: bool = False) -> None:
        output_directory = self.output_directory
        if output_directory is None or self._stage is None:
            return
        from benchmarking.performance import stop_resource_monitor

        output_directory.mkdir(parents=True, exist_ok=True)
        stage = self._stage
        self._stage = None
        self._remaining_outputs.clear()
        try:
            stop_resource_monitor(output_directory / f"{stage}_usage.pkl")
        except RuntimeError:
            if not ignore_errors:
                raise

    @hook_impl
    def before_dataset_loaded(self, dataset_name: str, node) -> None:
        self._start(node)

    @hook_impl
    def after_dataset_saved(self, dataset_name: str, data, node) -> None:
        if node.name != self._stage:
            return
        self._remaining_outputs.discard(dataset_name)
        if not self._remaining_outputs:
            self._stop()

    @hook_impl
    def on_node_error(self, error: Exception, node, catalog, inputs, is_async: bool, run_id: str) -> None:
        if node.name == self._stage:
            self._stop(ignore_errors=True)

    @hook_impl
    def on_pipeline_error(self, error: Exception, run_params, pipeline, catalog) -> None:
        self._stop(ignore_errors=True)
