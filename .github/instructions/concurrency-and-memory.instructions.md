---
applyTo: "src/wiser/utils/**,src/wiser/Cache/**,src/wiser/profiling/**"
---

# Scheduler, storage, and memory lifecycle

This subsystem exists so that the GUI stays responsive while a multi-hour job runs over
a cube far larger than RAM. Its contracts are what keep WISER usable at 100 GB, and they
are easy to break from a distance.

## The resource contract is the whole point

- Every `WorkUnit` carries `ram_peak_est_bytes`, and the scheduler admits work against
  `SCHEDULER_RAM_BUDGET` (2 GB) and `SCHEDULER_PROCESS_BUDGET` (`min(6, cpu_count)`).
  This estimate is a load-bearing number, not a hint:
  - **Underestimated** and the machine swaps or the worker is killed.
  - **Overestimated** past the budget and the unit can never be admitted — the scheduler
    counts failed admissions and aborts the plan rather than hanging, so the user sees a
    task that dies for no visible reason.
- A new or modified `TaskStage` must declare a `ResourceModel`
  (`fixed_overhead_bytes`, `bytes_per_scalar_in`, `bytes_per_scalar_out`,
  `scratch_bytes_per_scalar_in`) that reflects what the stage function actually
  allocates. Read the stage function and check the model against it. A stage that makes
  an intermediate copy, promotes dtype, or holds two buffers at once but declares
  `scratch_bytes_per_scalar_in = 0` is a defect — flag it with the specific allocation
  you found.
- Chunking is what makes a 100 GB cube tractable. A stage whose work unit is the whole
  image, or whose chunk size does not shrink as the cube grows, defeats the design.
  Check that `chunking_scheme_type` and the chunking policy actually bound the working
  set, and that `output_region_for` matches what the function writes.

## Priority lanes and admission

- Three lanes: `interactive`, `render`, `background`. Putting long work in `interactive`
  starves the UI; putting user-visible work in `background` makes the app feel dead.
  Check that a new submission's priority matches what the user is waiting on.
- Queue transition logs are first-class debugging artifacts. A diff that removes or
  quiets them is removing the only way to diagnose a stuck plan in the field.
- Deadlock shapes to look for: a unit that waits on a dependency in the same lane, a
  reserved unit that can never satisfy RAM admission, and a dependency cycle introduced
  through `deps` or `stage_steps`.

## Storage refs, leases, and cleanup

- Refs may be RAM shared memory, memmap, zarr, json, or an external handle. The backing
  kind determines the cleanup obligation — shared memory and memmaps leak OS resources,
  not just Python objects. A leaked shared-memory segment survives the process.
- Lifetime is policy-driven through `DeletePolicy` and lease records. Any new output
  binding needs a deliberate retention policy; check that consumers release
  (`release_plan_consumer`) on **every** path, including the failure and cancellation
  paths. Cleanup in the success branch only is the classic version of this bug.
- Prefer `try`/`finally` or a context manager over cleanup at the end of a function
  body. An exception between allocation and release leaks for the life of the process.
- Worker processes reach storage over a local `multiprocessing.connection.Listener`
  bound with a per-run random `authkey`. Flag any change that widens the bind address,
  hardcodes or logs the key, removes the authentication, or accepts a payload from an
  unauthenticated peer — that listener deserializes data.

## Memory discipline

- Name the peak, not the average. A stage that holds input, output, and scratch
  simultaneously peaks at their sum.
- Flag in-place-looking operations that are not: `a = a + b` copies, `a += b` does not;
  `np.ma.masked_values` builds a new array plus a mask.
- Caching: `get_image_data()` stores the **entire cube** in the computation cache keyed
  by dataset. Any new `add_cache_item` on a full-cube-sized value needs an eviction
  story. Check whether the cache is bounded at all, and say so if it is not.
- Numba `njit` kernels compile per dtype signature. A new dtype at a call site either
  triggers a slow recompile on the user's first click or silently falls back to object
  mode. Check the explicit signatures in `src/wiser/gui/util.py` and the tool modules
  against the dtypes actually passed.

## Process pools and thread oversubscription

- BLAS and OpenMP start one thread per core **inside every worker**, so a process pool
  multiplies them. `src/tests/conftest.py` and `src/wiser/__main__.py` both cap
  `OPENBLAS_NUM_THREADS` and `OMP_NUM_THREADS` before NumPy is imported, and the two
  blocks must stay in sync. Flag any diff that changes one and not the other, or that
  imports NumPy above those lines.
- Worker payloads are pickled. Flag anything unpicklable crossing the boundary — a Qt
  object, an open GDAL dataset, a lambda, a local closure — and anything *large* being
  pickled where a storage ref should be passed instead. Sending a 4 GB array through a
  pipe is not the same cost as sending a handle to it.
- `fork` is unavailable on Windows and unsafe on macOS. Code that relies on inherited
  process state after `spawn` works only on Linux; see
  `.github/instructions/packaging-and-platform.instructions.md`.

## Cancellation and progress

- A task over a real cube runs for minutes to hours. Check that a new task reports
  `ProgressUpdate` at a granularity the user can read, and that cancellation actually
  reaches the workers rather than only flipping a flag the loop never checks.
- Check what happens to partial outputs on cancel. A half-written product left on disk
  with a valid-looking header is worse than no output.
