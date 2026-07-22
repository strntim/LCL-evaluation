INITIAL_PROFILING_DURATION = 0.25
DEFAULT_PROFILING_INTERVAL = 0.1
NUM_CORES_TO_ALLOCATE      = 5
MONITOR_READY_TIMEOUT      = 30

DEBUG = False


import os
import sys

import psutil
from multiprocessing import Event, Process, Queue, current_process
from pathlib import Path

MEMORY_METRIC = "pss" if sys.platform == "linux" else "rss"

def get_cores_by_usage(interval: float = INITIAL_PROFILING_DURATION):
    cpu_usage = psutil.cpu_percent(percpu=True, interval=interval)
    return sorted(range(len(cpu_usage)), key=lambda i: cpu_usage[i])

if sys.platform == "linux":
    cores_by_usage = get_cores_by_usage()
    required_cores = NUM_CORES_TO_ALLOCATE + 1
    if len(cores_by_usage) < required_cores:
        raise RuntimeError(f"Benchmarking requires at least {required_cores} logical CPUs.")

    logging_core_id = cores_by_usage[0]
    task_execution_core_ids = cores_by_usage[1:required_cores]

    os.sched_setaffinity(0, {*task_execution_core_ids})

    print("Executing on core:", task_execution_core_ids)
    print("Logging on core:", logging_core_id)

    core = task_execution_core_ids
    proc = psutil.Process(os.getpid())

    for t in proc.threads():
        os.sched_setaffinity(t.id, {*core})

    for p in proc.children(recursive=True):
        os.sched_setaffinity(p.id, {*core})
else:
    logging_core_id = None
    if sys.platform == "darwin" and current_process().name == "MainProcess":
        print("Warning: CPU affinity is unavailable on macOS and memory is measured as RSS; run benchmarks on Linux for reliable results.")


import pandas as pd
import joblib, time

_monitor_proc = None
_stop_evt     = None
_ready_evt    = None
_path_q       = None

def _sample_once(root, exclude_pids: set[int]=frozenset()):
    # persistent cache so we always talk to the *same* Process objects
    if not hasattr(_sample_once, "cache"):
        _sample_once.cache = {}          # pid ➜ Process

    cache = _sample_once.cache
    procs = [root] + root.children(recursive=True)

    # make sure every pid we see has a cached Process object
    for p in procs:
        if p.pid in exclude_pids:
            continue
        if p.pid not in cache:
            try:
                p.cpu_percent(None)      # prime – first call always 0.0
                cache[p.pid] = p
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue

    total_cpu = 0.0
    total_mem = 0
    c = []
    for pid, p in list(cache.items()):
        try:
            if not p.is_running():
                cache.pop(pid, None)         # clean up dead workers
                continue
            cpu = p.cpu_percent(None)        # non-blocking, since last call
            memory = p.memory_full_info().pss if MEMORY_METRIC == "pss" else p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            cache.pop(pid, None)
            continue
        c.append(cpu)
        total_cpu += cpu                 # add this process’s %
        total_mem += memory

    return total_cpu / 100.0, total_mem / (1024 ** 2), c   # CPU cores, RAM MB

def _resource_worker(interval: float, logging_core_id: int | None, stop_evt, ready_evt, path_q: Queue):
    # Use separate core for logging to not affect performance
    if logging_core_id is not None:
        os.sched_setaffinity(0, {logging_core_id})

    parent_pid = os.getppid()
    logger_pid = os.getpid()
    proc       = psutil.Process(parent_pid)

    _sample_once(proc, exclude_pids={logger_pid})

    resource_log = []
    last_sample  = 0
    sleep_time   = interval
    print_log    = ""
    start_time   = time.perf_counter()

    cpu, ram, c = _sample_once(proc, exclude_pids={logger_pid})
    resource_log.append({"t": 0, "cpu_cores": cpu, "ram_mb": ram, "memory_metric": MEMORY_METRIC})
    ready_evt.set()
    stop_evt.wait(interval)

    while not stop_evt.is_set():
        now      = time.perf_counter() - start_time
        cpu, ram, c = _sample_once(proc, exclude_pids={logger_pid})
        resource_log.append({"t": now, "cpu_cores": cpu, "ram_mb": ram, "memory_metric": MEMORY_METRIC})

        elapsed     = now - last_sample
        sleep_time  = min(max(0.05, interval - (elapsed - sleep_time)), interval)
        if DEBUG:
            print_log += f"{now:>6.2f} s| {(interval-sleep_time)*1000.0:4.2f} ms| {cpu*100:>5.1f}% [" + "|".join([f"{process_cpu:>3.0f}%" for process_cpu in c]) +f"] - {len(c)}\n"
        last_sample = now
        stop_evt.wait(sleep_time)

    end_time = time.perf_counter() - start_time
    cpu, ram, c = _sample_once(proc, exclude_pids={logger_pid})
    resource_log.append({"t": end_time, "cpu_cores": cpu, "ram_mb": ram, "memory_metric": MEMORY_METRIC})
    if DEBUG:
        print(print_log)

    df      = pd.DataFrame(resource_log)
    outfile = Path(path_q.get())
    joblib.dump(df, outfile)

def start_resource_monitor(interval: float = DEFAULT_PROFILING_INTERVAL):
    global _monitor_proc, _stop_evt, _ready_evt, _path_q, logging_core_id
    if _monitor_proc is not None and _monitor_proc.is_alive():
        raise RuntimeError("Resource monitor already running.")

    _stop_evt = Event()
    _ready_evt = Event()
    _path_q = Queue(maxsize=1)
    _monitor_proc = Process(
        target = _resource_worker,
        args   = (
            interval,
            logging_core_id,
            _stop_evt,
            _ready_evt,
            _path_q
        ),
        daemon =True
    )
    _monitor_proc.start()
    if not _ready_evt.wait(MONITOR_READY_TIMEOUT):
        if _monitor_proc.is_alive():
            _monitor_proc.terminate()
        _monitor_proc.join()
        _monitor_proc = None
        _stop_evt = None
        _ready_evt = None
        _path_q = None
        raise RuntimeError("Resource monitor did not start.")

def stop_resource_monitor(outfile: Path):
    global _monitor_proc, _stop_evt, _ready_evt, _path_q
    if _monitor_proc is None or not _monitor_proc.is_alive():
        raise RuntimeError("Resource monitor is not running.")

    _path_q.put(outfile)
    _stop_evt.set()
    _monitor_proc.join()

    _monitor_proc = None
    _stop_evt     = None
    _ready_evt    = None
    _path_q       = None

def get_directory(path: Path) -> Path:
    path = path.resolve()  # Get absolute path
    return path if path.is_dir() else path.parent
