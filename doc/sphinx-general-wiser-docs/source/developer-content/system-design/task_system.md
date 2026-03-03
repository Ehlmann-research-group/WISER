# WISER Task, Storage, and Scheduler System Overview

This document describes the current architecture of WISER's task execution stack and how task planning, storage, and scheduling interact.

The system is designed to:
- execute chunked operations under RAM and disk constraints,
- keep interactive workflows responsive,
- support process-safe worker execution,
- support progress/cancellation,
- keep data movement explicit via storage-backed bindings.

## Core Concepts

### SemanticTask

A `SemanticTask` is a user-facing request (for example: PCA, continuum removal, band math, or derived products). It describes **intent**, not runtime execution.

It carries:
- priority class,
- entry input (`input_ref`),
- an `AlgorithmPipeline`.

`SemanticTask` does not allocate storage, choose chunking, or schedule compute.

### AlgorithmPipeline and TaskStage

An `AlgorithmPipeline` is an ordered list of `TaskStage` objects.

Each stage defines:
- input binding name (`input_binding`),
- output binding names (`output_bindings`),
- chunking scheme type,
- allocation requests for stage outputs,
- a top-level callable producer (`map_fn` / `reduce_fn`),
- optional broadcast refs (e.g. spectrum refs needed by every chunk).

Important: stage input refs are resolved by binding name during planning. Stages should not assume direct ownership of concrete `DataRef`s.

### TaskPlan, WorkUnit, WorkUnitMeta

`TaskPlanner` builds a concrete `TaskPlan`.

- `TaskPlan.bindings`: semantic binding name -> `DataRef`
- `TaskPlan.work_units`: schedulable units with executor kind, deps, and prebuilt callable
- `TaskPlan.work_units_meta`: planning I/O metadata (`input_ref`, `input_region`, output `WriteSpec`s, broadcast refs)

`WorkUnit` is intentionally lightweight at runtime; detailed I/O metadata lives in `WorkUnitMeta`.

### StorageService (Authoritative Storage Registry)

`StorageService` is the central storage authority used by planner and workers. It manages:
- `AllocationRequest -> DataRef` allocation,
- RAM vs disk materialization,
- metadata and region metadata,
- external data registration (datasets, spectra, spectra lists),
- access descriptors for reads/writes (`RamAccessDescriptor`, `MemmapAccessDescriptor`, `ZarrAccessDescriptor`, external descriptors, JSON descriptors).

Internally it exposes an RPC listener used by worker-side clients.

### StorageClient (Worker-Side Data Access)

`StorageClient` is the worker-facing API that talks to `StorageService` over RPC.

Workers use it to:
- resolve access descriptors (`get_access`),
- read full data or regions (`read_data`, `read_region`),
- write full data, regions, or `WriteSpec` outputs (`write_data`, `write_region`, `write_spec`),
- fetch metadata (`get_meta`, `get_region_meta`).

For dataset-shaped reads, client conventions are:
- dataset: `[y][x][b]`
- spectrum: `[b]`
- spectra list: `[i][b]`

### WorkScheduler

`WorkScheduler` executes a `TaskPlan` stage-by-stage.

It:
- validates stage/unit structure,
- enqueues units by priority and executor kind,
- dispatches to process/thread pools under token budgets,
- enforces a transient RAM budget across both process and thread units,
- tracks success/failure and fail-fast behavior,
- advances to next stage only when current stage is terminal.

Process workers are initialized with `initialize_process_storage_client(...)` so unit callables can use `get_process_storage_client()` safely.

#### Queue model (high level)

Each priority/executor lane has:
- a **main queue** (newly enqueued work),
- a **blocked queue** (work that failed RAM admission before),
- and the scheduler-wide **ReservedTracker** (aged/starved units promoted out of blocked/main).

Dispatch order is:
1. reserved work first,
2. then blocked work,
3. then main work.

`QueuedWorkUnit.defer_count` is incremented when RAM admission fails. If this count exceeds a threshold, the unit is promoted into `ReservedTracker`.

#### RAM and hold-byte model (high level)

The scheduler tracks:
- `in_flight` RAM bytes: admitted units currently running,
- `hold` bytes: bytes withheld for reserved fairness windows.

For non-reserved units, admission uses the held cap (`cap - hold`).
For reserved units, admission uses full cap (`cap`).

This prevents non-reserved work from continually consuming all slack while still allowing reserved units to run once feasible.

#### ReservedTracker (high level)

`ReservedTracker` keeps per-priority FIFO queues and computes hold bytes from configurable windows (currently interactive/render/background = 3/2/1).

It provides:
- deterministic reserved candidate selection,
- FIFO ownership for reserved units,
- hold-byte totals used by scheduler RAM admission.

#### Key scheduler methods

The main methods to understand runtime behavior are:
- `run_task_plan(...)` - validates and starts scheduling.
- `_enqueue_stage_locked(...)` - places stage units into main queues.
- `_drain_queues_locked(...)` - main dispatch loop (reserved, blocked, main).
- `_attempt_submit_reserved_locked(...)` - reserved admission path.
- `_submit_non_reserved_from_queues_locked(...)` - blocked/main admission path.
- `_submit_runnable_item_locked(...)` - final executor submission + in-flight accounting.
- `_on_unit_done(...)` - completion accounting, stage progression, and re-dispatch.

## Planning Flow

`TaskPlanner.plan_semantic_task(...)` performs:

1. **Initialize bindings**
   - `bindings["__task_input__"] = semantic_task.input_ref`
2. **For each stage in order**
   - resolve stage input ref from `bindings[stage.input_binding.name]`
   - choose chunking with `ChunkingPolicy`
   - call `stage.generate_allocation_requests(...)`
   - allocate outputs via `StorageService.allocate_data(...)`
   - store new output refs in `bindings`
   - expand chunks into units
   - build per-unit `WriteSpec` map and `WorkUnitMeta`
   - build runnable top-level callable via `stage.map_fn(...)`
3. **Record dependencies**
   - default behavior is stage barrier: stage N depends on all units in stage N-1

If allocation or binding resolution fails, planning fails before scheduling.

## Data Handoff Between Stages

Data does not flow directly in-memory between work units. Stage handoff is storage-backed:

1. Stage A writes to an allocated output ref (`WriteSpec.ref`) using regions.
2. That output ref is already stored in `TaskPlan.bindings` under the output binding name.
3. Stage B declares `input_binding` with that same name.
4. Planner resolves Stage B input to the same `DataRef`.
5. Stage B units read regions through `StorageClient`.

Benefits:
- process-safe cross-worker exchange,
- predictable memory behavior,
- disk spill compatibility,
- deterministic stage wiring via binding names.

## Storage and Execution Path

End-to-end:

`SemanticTask` -> `AlgorithmPipeline` -> `TaskPlanner` -> `TaskPlan` (bindings + work units + unit meta) -> `WorkScheduler` -> worker callable -> `StorageClient` RPC -> `StorageService` access/metadata -> data read/write.

This separation keeps:
- stage logic reusable,
- planning deterministic,
- scheduling policy-driven,
- storage concerns centralized and explicit.
