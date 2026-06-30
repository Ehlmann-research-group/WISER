"""Session-wide pytest configuration for the WISER test suite.

This module is imported by pytest before any test module is collected, which
makes it the right place to set process-wide environment that must be in place
before heavy libraries load.
"""

# Cap BLAS / OpenMP thread pools BEFORE numpy (or anything that loads OpenBLAS /
# OpenMP) is imported. This block must run before any `import numpy`, so keep it
# at the very top of the file, ahead of all other imports.
#
# Why: many WISER tests drive the scheduler's ProcessPoolExecutor. By default
# OpenBLAS and OpenMP each start one thread per CPU core *inside every worker
# process*, so you get (workers x cores) threads competing for `cores` cores
# (oversubscription), and each thread pre-allocates a working buffer. That
# exhausts memory and makes workers die with
# "OpenBLAS: Memory allocation failed" / BrokenProcessPool -- which shows up as
# flaky, order-dependent test failures.
#
# Fix: give each worker its fair share of cores, so (workers x threads) ~= cores.
# The scheduler runs `min(6, cores)` worker processes (SCHEDULER_PROCESS_BUDGET
# in wiser/utils/work_scheduler.py); we mirror the `6` here rather than import it
# (importing it would load numpy before we've capped it). Examples: 22 cores ->
# 6 workers -> 3 threads each; 8 cores -> 6 workers -> 1 thread each. A fixed
# value like 7 would be 6 x 7 = 42 threads on a 22-core box and OOM. Spawned
# worker processes inherit these vars from this environment; setdefault() lets an
# explicit shell / conda env var override it.
# NOTE: keep this in sync with the same block at the top of src/wiser/__main__.py.
import os

_cpu_count = os.cpu_count() or 1
_worker_budget = min(6, _cpu_count)  # mirrors SCHEDULER_PROCESS_BUDGET
_blas_threads = str(max(1, _cpu_count // _worker_budget))
for _blas_thread_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_blas_thread_var, _blas_threads)
