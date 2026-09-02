# System Design

This page documents internal architecture and design patterns in WISER.

## Concurrency Model: Why WISER Uses Subprocesses

The task, storage, and scheduler machinery documented below exists for one reason, and it is
worth stating before the mechanics: **in Python, keeping a GUI responsive while computing is not
a matter of spawning a thread.**

The question this answers comes up often, usually as "ENVI stays responsive and fast without any
of this — why can't WISER just spawn a thread and compute?" ENVI is written in IDL, which has
real parallel threads and is built for fast array computation: it can spawn a thread, share
memory, compute, and be done. Python cannot. The GIL makes threads concurrent but not parallel,
so genuinely CPU-bound work has to go to a **subprocess**, which means data is serialized out to
the worker, computed on, and serialized back. That transfer cost is real; it has been reduced,
but it cannot be eliminated. Python is also not built for array computation the way IDL is —
even with NumPy, there is a cost to context switches between Python and the underlying C/C++.

Two consequences follow, and both are load-bearing:

- Anything that runs on the GUI thread and takes real time freezes the whole application. Moving
  work off that thread is the standard, not an optimization.
- The architecture optimizes for **low memory overhead and a responsive GUI**, and pays for that
  in data-transfer time. This is a deliberate trade-off rather than a discovered constraint. It
  is a reasonable one to revisit with fresh measurements — but revisit it knowing it is there,
  not by accident.

### Choosing a mechanism

| The work | Use | Notes |
|----------|-----|-------|
| Long, expensive, needs progress and cancellation | Task system (`src/wiser/utils/task_system.py`) | Chunked execution under RAM and disk constraints; reports to the activity monitor. Documented in full below. Use it, but do not build on it: it is more machinery than the problem needs, and simplifying it is tracked in #759. |
| One-off CPU-bound work | `WorkScheduler` process pool (`src/wiser/utils/work_scheduler.py`) | No semantic task, no plan. Appropriate when the work is short or has no meaningful sub-stages. |
| I/O-bound work | `WorkScheduler` thread pool | Threads are fine here: the GIL is released while waiting on I/O. |
| Anything touching `QImage`, `QPixmap`, or a widget | GUI thread only | Qt objects cannot be created or mutated off the GUI thread. Cut the seam between the computation and the Qt handoff: return a finished array, build the Qt object on the GUI thread. |

A thread can also carry long CPU-bound work when most of the cost sits in NumPy or Numba code
that releases the GIL, at the price of taking resources from the main thread. That is sometimes
the right trade — notably where the result is large enough that serializing it back to the main
process would cost more than the parallelism gains.

The single documented standard for making this choice, including worked examples, is tracked in
issue #753; until it lands, prefer the table above over copying whichever nearby example is
closest to hand.

Work that still runs on the GUI thread and should not: the render path described in
[Rendering Pipeline](rendering-pipeline.md) (#758), and ROI average-spectra computation in
`src/wiser/gui/rasterpane.py` (#468).

### Reading concurrently

Because concurrent readers must not share one GDAL handle, `RasterDataImpl.reopen_dataset()`
opens a fresh handle for every read (`src/wiser/raster/dataset_impl.py`). This is why
multithreaded reads are safe, and it is also a real per-read cost on formats that are slow to
open.

```{warning}
GDAL 3.10 advertises multithreaded reads from a single handle, which would make the reopen
unnecessary. Testing it returned **corrupted data**. Do not remove the reopen without first
re-verifying multithreaded reads against a known-good reference output — see issue #752.
```

Recorded from handoff notes on the `joshua-handoff` branch, which is retained as the primary
source.

---

## WISER Task, Storage, and Scheduler System Overview

This document describes the current architecture of WISER's task execution stack and how task planning, storage, and
scheduling interact.

The system is designed to:

- execute chunked operations under RAM and disk constraints,
- keep interactive workflows responsive,
- support process-safe worker execution,
- support progress/cancellation,
- keep data movement explicit via storage-backed bindings.

## Core Concepts

### SemanticTask

A `SemanticTask` is a user-facing request (for example: PCA, continuum removal, band math, or derived products). It
describes **intent**, not runtime execution.

It carries:

- priority class,
- entry input (`input_ref`),
- an `AlgorithmPipeline`.

`SemanticTask` does not allocate storage, choose chunking, or schedule compute.

It also carries meta data that is displayed in the activity monitor. This metadata is
contains the task title and variables used in the task.

### AlgorithmPipeline and TaskStage

An `AlgorithmPipeline` is an ordered list of `TaskStage` objects. These `TaskStage` objects
run sequentially.

Each stage defines:

- input binding name (`input_binding`),
- output binding names (`output_bindings`),
- chunking scheme type,
- work-unit dependency mode (`work_unit_dependency`: `independent` or `sequential`),
- allocation requests for stage outputs,
- a top-level callable producer (`task_fn` / `pre_task_fn` (optional) / `post_task_fn` (optional)),
- optional broadcast refs (e.g. spectrum refs needed by every chunk).

Important: stage input refs are resolved by binding name during planning. Stages should not assume direct ownership of
concrete `DataRef`s.

#### TaskStage Allocations

Stages allocate outputs with `AllocationRequest`.

For retention and deletion details, see the WISER Storage and Memory Management section below.

### TaskPlan, WorkUnit, WorkUnitMeta

`TaskPlanner` builds a concrete `TaskPlan`.

- `TaskPlan.bindings`: semantic binding name -> `DataRef`
- `TaskPlan.work_units`: schedulable units with executor kind, deps, and prebuilt callable
- `TaskPlan.work_units_meta`: planning I/O metadata (`input_ref`, `input_region`, output `WriteSpec`s, broadcast refs)
- `TaskPlan.stage_work_units`: per-stage unit membership metadata
- `TaskPlan.stage_steps`: ordered barrier steps per stage; units in a step can run in parallel, steps run in order

By default:

- `MapStage` uses `work_unit_dependency="independent"` (single step containing all units)
- `SequentialStage` uses `work_unit_dependency="sequential"` (one unit per step)

`WorkUnit` is intentionally lightweight at runtime; detailed I/O metadata lives in `WorkUnitMeta`.

### StorageService (Authoritative Storage Registry)

`StorageService` is the central storage authority used by planner and workers. It manages:

- `AllocationRequest -> DataRef` allocation,
- RAM vs disk materialization,
- metadata and region metadata,
- external data registration (datasets, spectra, spectra lists),
- access descriptors for reads/writes (`RamAccessDescriptor`, `MemmapAccessDescriptor`, `ZarrAccessDescriptor`, external
  descriptors, JSON descriptors).

Internally it exposes an RPC listener used by worker-side clients.

#### Data Lifetime and Deletion

`StorageService` owns ref lifetime and reclamation.

For creation, read/write, and deletion behavior, see the WISER Storage and Memory Management section below.

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

`WorkScheduler` executes a `TaskPlan` stage-by-stage, and within each stage step-by-step.

It:

- validates stage/unit structure,
- enqueues units by priority and executor kind,
- dispatches to process/thread pools under token budgets,
- enforces a transient RAM budget across both process and thread units,
- tracks success/failure and fail-fast behavior,
- advances to next stage only when current stage is terminal.
- enforces intra-stage barriers by advancing to the next stage step only when the current step is terminal.

Process workers are initialized with `initialize_process_storage_client(...)` so unit callables can use
`get_process_storage_client()` safely.

#### Queue model (high level)

Each priority/executor lane has:

- a **main queue** (newly enqueued work),
- a **blocked queue** (work that failed RAM admission before),
- and the scheduler-wide **ReservedTracker** (aged/starved units promoted out of blocked/main).

Dispatch order is:

1. reserved work first,
2. then blocked work,
3. then main work.

`QueuedWorkUnit.defer_count` is incremented when RAM admission fails. If this count exceeds a threshold, the unit is
promoted into `ReservedTracker`.

Dispatch is normally **edge-triggered**: queues are drained when a plan is submitted and when a unit completes. A
periodic background **watchdog** thread (`_watchdog_loop`, interval configured by `SchedulerConfig._watchdog_interval_seconds`)
re-drains the queues as a safety net, so work that was left parked cannot wedge forever if a normal drain trigger is
missed. Draining when nothing is admissible is a cheap no-op.

#### RAM and hold-byte model (high level)

The scheduler tracks:

- `in_flight` RAM bytes: admitted units currently running,
- `hold` bytes: bytes withheld for reserved fairness windows.

For non-reserved units, admission uses the held cap (`cap - hold`).
For reserved units, admission uses full cap (`cap`).

This prevents non-reserved work from continually consuming all slack while still allowing reserved units to run once
feasible.

**Run-alone override:** when nothing is in flight (`in_flight == 0`), a unit that still does not fit — most importantly
one whose `ram_peak_est_bytes` exceeds the whole budget — is admitted anyway, since waiting can never free more RAM.
The over-budget admission is logged as a warning (running such a unit may exceed available memory). This guarantees a
single oversized unit with no siblings (e.g. an unchunked stage over a large dataset) runs alone instead of hanging the
plan forever.

#### ReservedTracker (high level)

`ReservedTracker` keeps per-priority FIFO queues and computes hold bytes from configurable windows (currently
interactive/render/background = 3/2/1).

It provides:

- weighted round-robin candidate selection across the priority windows, using a cursor that advances past each served
  slot (so admission rotates fairly across priorities rather than always draining the highest priority first),
- head-of-line avoidance: admission scans past an un-admittable head (e.g. an over-budget unit) to serve a fittable
  candidate in another priority queue, while preserving FIFO order *within* a priority,
- a run-alone override consistent with the scheduler's RAM model (admit the current head when nothing is in flight),
- FIFO ownership for reserved units,
- hold-byte totals used by scheduler RAM admission.

#### Key scheduler methods

The main methods to understand runtime behavior are:

- `run_task_plan(...)` - validates and starts scheduling.
- `_enqueue_stage_locked(...)` - places stage units into main queues.
- `_drain_queues_locked(...)` - main dispatch loop (reserved, blocked, main).
- `_attempt_submit_reserved_locked(...)` - reserved admission path.
- `ReservedTracker.next_admissible_reserved_unit(...)` / `advance_reservation_cursor(...)` - round-robin reserved
  selection (with head-of-line scan and run-alone override) and the cursor advance that keeps it fair.
- `_submit_non_reserved_from_queues_locked(...)` - blocked/main admission path.
- `_submit_runnable_item_locked(...)` - final executor submission + in-flight accounting (logs over-budget run-alone admissions).
- `_on_unit_done(...)` - completion accounting, stage progression, and re-dispatch.
- `_watchdog_loop(...)` - periodic safety-net re-drain so parked work cannot wedge if a drain trigger is missed.

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
    - build runnable top-level callable via `stage.task_fn(...)`
    - build `stage_steps` from `work_unit_dependency`
        - `independent`: `stage_steps[stage_id] = [all_stage_units]`
        - `sequential`: `stage_steps[stage_id] = [[u1], [u2], ...]`
3. **Record dependencies and stage step layout**
    - default behavior is stage barrier: stage N depends on all units in stage N-1
    - within-stage behavior is derived from each stage's `work_unit_dependency`

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

`SemanticTask` -> `AlgorithmPipeline` -> `TaskPlanner` -> `TaskPlan` (bindings + work units + unit meta) ->
`WorkScheduler` -> worker callable -> `StorageClient` RPC -> `StorageService` access/metadata -> data read/write.

This separation keeps:

- stage logic reusable,
- planning deterministic,
- scheduling policy-driven,
- storage concerns centralized and explicit.

---

## Activity Monitor Update Flow

This page describes how task execution state moves through the scheduler stack and ends up in the activity monitor UI.
The important design point is that the activity monitor does not derive progress on its own. It is a presentation layer
driven by scheduler and task-manager events.

## Responsibilities by Layer

### `WorkScheduler`

`WorkScheduler` owns runtime execution. It decides when a work unit has:

- succeeded,
- failed,
- caused a fail-fast termination,
- completed a stage,
- completed a plan,
- or been cancelled.

It does not update widgets directly. Instead, it emits lifecycle signals through `TaskManager`.

### `TaskManager`

`TaskManager` is the adapter between plan execution and the activity monitor. Its main jobs are:

- register a task row when a `TaskPlan` is submitted,
- maintain the mapping between `plan_id` and `activity_id`,
- convert scheduler lifecycle events into activity-monitor API calls.

If the scheduler speaks in plan IDs and the dialog speaks in activity-row IDs, `TaskManager` is the translation layer.

### `ActivityMonitorDialog`

`ActivityMonitorDialog` owns the visible state:

- active rows,
- finished rows,
- progress bars,
- cancel buttons,
- error lists,
- view-errors buttons,
- and terminal row transitions.

It should be treated as a small task-state UI API, not as a scheduler.

## Registration Flow

Task registration starts in `TaskManager.register_and_submit_task_plan(...)`.

That method:

1. builds task metadata from the `TaskPlan`,
2. calls `ActivityMonitorDialog.register_task(...)`,
3. stores:
    - `plan_id -> activity_id`
    - `activity_id -> plan_id`
4. submits the plan to `WorkScheduler.run_task_plan(...)`.

`ActivityMonitorDialog.register_task(...)` creates the active-task row and returns the integer `activity_id` that will
be used for all later updates.

The row starts in the `idle` state and includes:

- title and metadata display,
- progress bar,
- cancel button,
- disabled "View Errors" button.

The cancel button does not cancel the task by itself. It calls the callback that `TaskManager` provided, which in the
normal scheduler-backed path is `scheduler.cancel_plan(task_plan.plan_id)`.

## Progress Update Flow

Progress is work-unit-based, not time-based and not stage-percentage-based.

### When progress is emitted

`WorkScheduler` emits a progress update after a work unit reaches a terminal state inside `_on_unit_done(...)`.

That happens for both:

- successful work units,
- failed work units.

This is intentional. A failed work unit is still finished from the scheduler's point of view, so it contributes to
completed work.

### How progress is computed

After a unit finishes, the scheduler computes:

- `completed_units = succeeded units + failed units` across all stages,
- `total_units = len(task_plan.work_units)`.

It then emits:

- `task_progressed.emit((plan_id, completed_units, total_units))`

through the attached `TaskManager`.

### How `TaskManager` forwards progress

`TaskManager._on_task_progressed(...)` receives the tuple:

- `(task_plan_id, numerator, denominator)`

It looks up the corresponding `activity_id`, then calls `emit_progress_update(...)`, which emits:

- `(activity_id, ProgressUpdate(current_iteration, total_iterations))`

to `ActivityMonitorDialog.progress_update`.

This is the point where scheduler-facing progress is converted into UI-facing progress.

### How the dialog applies progress

`ActivityMonitorDialog._on_progress_update(...)` validates the payload and calls:

- `set_task_progress_update(activity_id, progress)`

`set_task_progress_update(...)` then:

- clamps the numerator and denominator,
- sets the progress bar range to `0..total_iterations`,
- sets the current value to `current_iteration`,
- computes the displayed percentage text,
- transitions the row from `idle` to `running` if needed.

The monitor also exposes `set_task_progress(activity_id, value)` for simpler percentage-style updates, but the
scheduler-backed flow uses `set_task_progress_update(...)`.

## Error Flow

Errors are reported separately from terminal failure state.

When a work unit raises an exception in `WorkScheduler._on_unit_done(...)`, the scheduler emits:

- `task_errored.emit((plan_id, f"{type(exc).__name__}: {exc}"))`

`TaskManager._on_task_errored(...)` receives that payload, resolves the `activity_id`, and calls:

- `ActivityMonitorDialog.append_task_error(activity_id, error_message)`

`append_task_error(...)`:

- appends the message to the row's stored error list,
- enables the row's "View Errors" button.

This means the UI can accumulate one or more error messages before the task reaches a terminal state.

## Completion Flow

When the scheduler reaches the end of the final stage and the plan has no recorded failures, the success path is:

1. run `completion_callback(...)` if one exists,
2. finalize plan outputs,
3. emit `task_finished.emit(plan_id)`,
4. complete the plan's future successfully.

`TaskManager._on_task_finished(...)` maps `plan_id` to `activity_id` and calls:

- `ActivityMonitorDialog.set_task_finished(activity_id)`

`set_task_finished(...)`:

- fills the progress bar to its maximum,
- marks the row as `finished`,
- schedules the row to move from the active table to the finished table.

The move is deferred through a short timer so the user briefly sees the terminal state before the row is relocated.

## Failure Flow

There are two related but different failure behaviors.

### Normal non-fail-fast failures

If a work unit fails and the plan is not being failed immediately, the scheduler:

- records the exception,
- emits `task_errored`,
- still emits `task_progressed`,
- continues evaluating whether the current step, stage, or plan can proceed.

If the plan eventually ends with recorded failures, the scheduler completes the plan future with an exception. In that
path, error text has already been surfaced via `append_task_error(...)`.

### Fail-fast failures

If `task_plan.fail_fast` is enabled, a work-unit failure causes the scheduler to:

- purge queued work for the plan,
- finalize outputs as failed,
- set the plan future to an exception,
- remove the plan from scheduler state.

The key point for the activity monitor is the same: error text is sent as soon as the failing work unit is observed.

## Cancellation Flow

Cancellation begins from the UI but is owned by the scheduler.

### UI-side initiation

Each registered task row stores a cancel callback. In the scheduler-backed path, that callback is:

- `scheduler.cancel_plan(task_plan.plan_id)`

When the user clicks the row's Cancel button, `ActivityMonitorDialog._cancel_task(...)` invokes that callback. If the
callback itself raises, the row is marked failed immediately through:

- `set_task_failed(activity_id, str(exc))`

If the callback returns normally, the dialog marks the row cancelled locally with:

- `set_task_cancelled(activity_id)`

Separately, the scheduler's own cancellation path emits `task_cancelled`, and `TaskManager._on_task_cancelled(...)` also
forwards that into:

- `ActivityMonitorDialog.set_task_cancelled(activity_id)`

In practice this means the UI can reflect cancellation immediately from the button path while still remaining compatible
with scheduler-driven cancellation events.

### Scheduler-side cancellation

`WorkScheduler.cancel_plan(...)` removes queued work for the plan and emits:

- `task_cancelled.emit(plan_id)`

This ensures cancellation can still be surfaced correctly even if it originated outside the dialog.

## `ActivityMonitorDialog` Update API

The dialog exposes a small API that other systems can drive directly.

### Registration

- `register_task(title, meta, cancel_callback) -> int`

Creates a new active row and returns its `activity_id`.

### Progress

- progress_update.emit

Use `progress_update.emit(activity_id, ProgressUpdate)` to update progress. Activity monitor handles the rest.

### Errors

- `append_task_error(activity_id, error_message)`
- `set_task_failed(activity_id, error_message=None)`

`append_task_error(...)` records error detail without forcing terminal state. `set_task_failed(...)` transitions the row
into the failed state and schedules it to move into the finished section.

### Terminal state transitions

- `complete_task(activity_id)`
- `set_task_finished(activity_id)`
- `set_task_cancelled(activity_id)`

These methods drive the visible terminal state of the row.

### Cleanup

- `remove_task(activity_id)`
- `get_active_task_count()`

These are UI-management helpers rather than execution-state methods.

## How `TaskManager` Uses the Dialog

`TaskManager` uses the activity monitor in two phases.

### Submission-time use

At submission time it calls:

- `register_task(...)`

and stores the returned `activity_id`.

### Runtime update use

During execution it forwards scheduler state into:

- `set_task_progress_update(...)`
- `append_task_error(...)`
- `set_task_finished(...)`
- `set_task_cancelled(...)`

That design keeps widget state changes in one place and avoids making the scheduler aware of Qt widgets.

## How Other Systems Can Integrate

Any subsystem can use the activity monitor as long as it respects the dialog API and keeps its state transitions honest.

There are two reasonable integration styles.

### Preferred: integrate through `TaskManager`

If your subsystem already models work as `TaskPlan` execution, integrate through `TaskManager`. This gives you:

- plan-to-activity ID mapping,
- consistent progress forwarding,
- shared cancellation wiring,
- a clean separation between execution and UI.

### Direct integration with `ActivityMonitorDialog`

If your subsystem does not use `TaskPlan` or `WorkScheduler`, you can still drive the dialog directly.

To do that correctly:

1. call `register_task(...)`,
2. store the returned `activity_id`,
3. call progress, error, and terminal-state methods with that ID,
4. provide a cancel callback that actually affects the underlying work.

If you take this path, your subsystem becomes responsible for ensuring that:

- progress updates are meaningful,
- errors are appended as soon as they are known,
- terminal states are only reported once the underlying work is truly terminal.

## Practical Guidance

- Treat progress as completed work units out of total work units unless your subsystem has a better discrete execution
  model.
- Send progress when work becomes terminal, not while it is merely in flight.
- Append errors immediately; do not wait until the entire plan has failed.
- Let the execution layer decide whether a task is truly finished, failed, or cancelled.
- Keep the dialog as a sink for state, not a source of execution truth.

## Summary

The scheduler owns execution truth, `TaskManager` owns translation and routing, and `ActivityMonitorDialog` owns
presentation. That split keeps execution policy out of the UI and gives other systems a small, usable API for reporting
progress, errors, cancellation, and completion in a consistent way.

---

## WISER Storage and Memory Management

This document describes how WISER-managed task data is allocated,
accessed, retained, and reclaimed.

The goal of this part of the system is to make stage-to-stage data
handoff explicit and safe across threads and processes while keeping
retention rules predictable.

## Overview

At a high level, a managed piece of task data moves through this flow:

1. A `TaskStage` asks for storage by returning an `AllocationRequest`.
2. `TaskPlanner` resolves that request and calls
   `StorageService.allocate_data(...)`.
3. `StorageService` creates a `DataRef`, backing storage, metadata, and a
   lease record.
4. Worker code uses `StorageClient` to read and write through that
   `DataRef`.
5. When the producing plan finishes, `StorageService` decides whether the
   object stays live or is reclaimed.

This means task stages do not pass large arrays directly to later stages.
They exchange data through named storage-backed refs.

## Storage-Side Allocations

Each `TaskStage` declares its outputs by returning
`AllocationRequest` objects from `generate_allocation_requests(...)`.
The planner allocates those outputs up front and binds them by name
before any stage work starts, so all pre-task, chunk-task, and
post-task callables write into known `DataRef`s.

If a stage wants an
output to remain available after the producing task plan finishes, it
should explicitly mark that output with `DeletePolicy.KEEP` either on
the `AllocationRequest` itself or by using the stage's output
delete-policy helpers. If no explicit policy is provided,
stage outputs default to `DeletePolicy.DELETE_WHEN_RELEASABLE`.

In practice:

- use `KEEP` for outputs that should remain available to the UI, later
  tasks, or user inspection after the current plan completes;
- use `DELETE_WHEN_RELEASABLE` for scratch or intermediate outputs that
  only exist to move data between stages inside the same plan.
- or delete all data and take advantange of the fact that the completion
  callback is called before data deletion, so you can copy data from
  storage before its deleted.

Behind the scenes, each allocation gets a lease record in
`StorageService`. Newly allocated outputs begin in a `WRITING`
producer state and are registered with the current task plan as a
planned consumer so they are not reclaimed while the plan is still
running. When the scheduler finalizes the plan, it marks produced refs
as `COMPLETED`, `FAILED`, or `ABORTED` and then releases the plan's own
consumer hold.

At that point:

- `KEEP` outputs remain live;
- `DELETE_WHEN_RELEASABLE` outputs are deleted immediately if nothing
  still depends on them (usually nothing will depend on them, so
  they are deleted when the plan finishes);
- otherwise they move into a pending-delete state until remaining
  dependencies clear.

## Creation

Managed task data is normally created from an `AllocationRequest`.
That request describes what kind of object is needed, whether it should
live in RAM or on disk, and enough shape and dtype information for the
storage layer to provision backing storage.

When `StorageService.allocate_data(...)` accepts the request, it:

- creates a `DataRef` that names the object;
- chooses RAM-backed or disk-backed storage;
- creates the underlying storage object;
- records metadata and region metadata support;
- creates a `StorageLeaseRecord` to track lifetime.

For RAM-backed numeric arrays, the service allocates named shared
memory and keeps one service-owned handle open while the ref is live.
For disk-backed data, the service creates files or zarr directories as
needed.

## Reading and Writing

Workers do not access storage by touching service internals directly.
They use `StorageClient`, which talks to `StorageService` over RPC to
resolve descriptors, fetch metadata, and read or write data.

Typical flow:

1. A worker asks the service for an access descriptor.
2. The descriptor tells the client how the data is backed.
3. The client reads or writes the whole object or a region.

For RAM-backed arrays, WISER uses shared memory so different workers can
attach to the same backing allocation safely. For disk-backed arrays,
the client reads from the file-backed store the service selected.

An important behavior to remember is that worker-facing reads generally
return copies. Whole-array and region reads from shared memory are
copied into NumPy arrays for the caller. Disk-backed reads are also
copied into arrays before they are returned. This avoids leaking live
worker attachments and keeps data ownership straightforward on the
consumer side.

## Storage Data Lifetime and Deletion

`StorageService` tracks lifetime explicitly for each internally managed
allocation by storing a `StorageLeaseRecord`. The important pieces of
that record are:

- `delete_policy`: whether the ref should be retained or reclaimed when
  possible;
- `producer_state`: whether the producing task is still writing or has
  reached a terminal state (`COMPLETED`, `FAILED`, or `ABORTED`);
- `planned_consumer_plan_ids`: task plans that still declare they need
  the ref;
- `borrowers` and `pins`: additional runtime holds that can temporarily
  block reclamation;
- `deletion_state`: whether the ref is currently `LIVE`,
  `PENDING_DELETE`, or `DELETED`.

An internally managed ref is deleted only when all of the following are
true:

- its `delete_policy` is `DELETE_WHEN_RELEASABLE`;
- its producer is in a terminal state;
- no planned consumer plans still depend on it;
- no borrowers or pins still hold it;
- it is not externally owned.

This means deletion is usually triggered by plan finalization rather
than by the last write itself. The scheduler first marks plan outputs as
terminal and then releases the plan's consumer hold.
`StorageService.try_reclaim(...)` re-checks the record after each
lifecycle transition and either:

- leaves the ref `LIVE`,
- marks it `PENDING_DELETE` because policy wants deletion but some
  dependency still exists, or
- deletes it immediately and marks it `DELETED`.

External refs are different. The service may register them and expose
descriptors for reading, but it does not own the source object and
therefore does not reclaim that underlying external data through the
managed deletion path.

For RAM-backed allocations, deletion has an important shared-memory
nuance. The service keeps one owner handle open for every live
shared-memory allocation. This is intentional because on Windows a named
shared-memory mapping can disappear as soon as the last handle closes.

During deletion the service:

- removes the ref from its registries and accounting;
- unlinks the named shared-memory segment while a service-owned handle
  is still open;
- then closes the final owner handle so the backing storage can
  actually be released.

This order matters. Unlinking removes the name so new attachments cannot
be created, but already attached workers may still hold live handles
until they close them. In other words, a ref can be deleted from the
service's point of view before the OS finishes reclaiming the last bytes
of shared memory.

For disk-backed managed refs, deletion removes the underlying file or
zarr directory. For JSON and memmap-backed refs this is a file delete;
for zarr-backed refs it is a directory-tree removal. `close()` also
clears the service's in-memory registries and closes owned handles, so
shutting down the service is effectively a final cleanup boundary for
any remaining service-owned allocations.

---

## Multiple Views in the Main Image Window

The main image window is capable of showing multiple views, either different
images of the same samples, or images of different samples. The views may
optionally be linked so that scrolling in one view will automatically update the
scroll position in the other views.

The primary constraint for multiple views in the main image window is that they
will all use the same zoom level. It is not possible to have different views
with different scroll levels.

When linked scrolling is turned on, the views will all automatically update to
match the scroll settings of the top left view.

## Main Window / Zoom Window Interactions

The interactions between the main image window and the zoom window become more
complex when multiple raster views are being displayed.

When the user clicks pixels in the main window, the zoom window is updated to
show both the dataset and the click location. Since different views may be
showing different datasets, the zoom window will switch to displaying the same
dataset that the clicked view was showing. However, if the views are linked, the
zoom window will not switch to display the different dataset.

When the user clicks pixels in the zoom window, the main window is updated to
display the click location. Since all views in the main window may not be
showing the same data set in the zoom window, the following behaviors are
followed:

* If linked scrolling is enabled, all views in the main window will be updated
  to show the click location.
* If linked scrolling is disabled, only the views in the main window that are
  showing the same dataset will be updated to show the click location; other
  views will not be updated (and any previously-showing click location will be
  discarded).

These behaviors are also followed when the user clicks in the zoom window.

## Confusing Scenarios (Open Questions)

```{note}
These are unresolved design questions captured for future discussion.
```

Scenario:  different "unrelated" spectral data sets

* Main image window showing two views with different datasets (e.g. oman1 and
  oman2 spectral data). Linked scrolling is OFF.
* Context window showing one of these data sets.

* What should the context window viewport highlight indicate? Should it
  always be drawn?
    * Issue:  The viewport reported from the unrelated dataset doesn't really
      mean anything.
    * Fix:    We only show the viewport highlight in the context pane if the
      matching dataset is open in the main window.

* Click in main window; zoom pane should switch to the clicked-on data set,
  and show the appropriate spectrum in the plot.

* Click in zoom pane. What should happen in the main image window?
    * We should only show highlighted pixel in views with the same data set.
      If the views are linked, we should show highlighted pixel in all views.

    * Same questions for viewport highlight.  (This is the same issue as with
      the main window and the context window.) ANSWER: The viewport highlight
      should only show up in the main window raster view's that have the same
      dataset as the zoom pane's rasterview. If they are linked, the viewport
      highlight should show up in all.

Scenario:  different "related" data sets over the same spatial area

* Main image window showing two views with related datasets (e.g. oman1
  spectral data and oman1 mineral map). Linked scrolling is ON.
* Context window showing one of these data sets.

* Click in main window.

    * Context window viewport highlight is easy; all main window views are
      showing the same area.

    * Should zoom pane switch to the clicked-on data set? Currently, zoom pane does not.

* Click in zoom pane. What should happen in the main image window?
    * Show highlighted pixel in all views.
    * Show viewport highlight in all views.
