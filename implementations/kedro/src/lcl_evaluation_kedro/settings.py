from lcl_evaluation_kedro.hooks import BenchmarkHook


HOOKS = (BenchmarkHook(),)
CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "Benchmarking",
}

