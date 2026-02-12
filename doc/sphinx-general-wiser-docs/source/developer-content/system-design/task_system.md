# WISER Task and Scheduler System Overview

This document describes the high-level architecture of WISER's task execution system and how planning, storage, and scheduling work together.

The system is designed to:
- execute large operations under RAM and disk constraints,
- keep interactive workflows responsive,
- support progress/cancellation,
- and run chunked computation on large inputs.

## Core Concepts

### SemanticTask

A `SemanticTask` is a user-facing request (for example, PCA, continuum removal, or statistics generation). It describes **intent**, not execution.

It carries:
- priority,
- algorithm parameters,
- an `AlgorithmPipeline`,
- output allocation intent.

It does **not** allocate storage, create work units, or run compute directly.

### AlgorithmPipeline

An `AlgorithmPipeline` is an ordered list of `TaskStage` objects that define the logical computation flow.

Typical pattern:
- map-like stage(s) for chunk-local work,
- optional reduce stage(s) for aggregation,
- optional map stage(s) for final transforms.

The pipeline is still abstract at this point: no concrete unit graph exists yet.

### TaskStage

A `TaskStage` is a reusable template for one phase of computation. Stages provide:
- input/output binding definitions,
- resource model hints,
- chunking scheme requirements,
- stage compute function (`map_fn`/`reduce_fn`),
- allocation requests for stage outputs.

Stages do not own concrete storage handles (`DataRef`) or scheduling state. 
The `TaskPlan` owns a map between bindings and `DataRef`s.

### WorkUnit

A `WorkUnit` is the smallest schedulable executable item. Each unit contains:
- input reference + input region,
- one or more output write specs (target ref + region),
- executor kind (thread/process),
- RAM estimate,
- dependency IDs.

Work units are produced by the `Task Planner` from stage definitions and chunking decisions.

### StorageLayer

`StorageLayer` manages data materialization:
- `AllocationRequest -> DataRef`,
- RAM or disk residency,
- region read/write for refs.

It is intentionally independent of task graph logic and scheduler policy.

### WorkScheduler

`WorkScheduler` executes a `TaskPlan` by:
- enforcing dependency order,
- admitting ready units under resource budgets,
- dispatching to thread/process executors,
- tracking completion/cancellation/progress.

The scheduler executes units; it does not redefine bindings or storage layout.

## How Task Planning Works

The `TaskPlanner` converts one `SemanticTask` into a concrete `TaskPlan`.

### 1) Build Binding Table

Planner initializes a binding map:
- `bindings: Dict[str, DataRef]`

Bindings connect semantic names (for stage inputs/outputs) to concrete storage references.

### 2) Plan Each Stage

For each stage in pipeline order:

1. **Resolve stage input binding**
   - Find the input `DataRef` from `bindings`.
2. **Choose chunking**
   - Use `ChunkingPolicy` + stage resource model + scheduler constraints.
3. **Allocate outputs up front**
   - Stage emits `AllocationRequest`s.
   - Planner calls `StorageLayer.allocate_data(...)`.
   - Resulting output refs are inserted into `bindings`.
4. **Expand into work units**
   - Iterate stage input regions from the chosen chunking scheme.
   - Map input region to output region(s).
   - Create `WorkUnit`s with write specs and dependency links.

If required allocation fails during planning, task submission fails early.

## Data Transfer Between Work Units (Key Mechanism)

Data does not move directly from one work unit to another in memory. Instead, units communicate through storage-backed bindings:

1. Stage A writes to output refs (`DataRef`) using regioned writes.
2. Those refs are already registered in `bindings` by semantic output name.
3. Stage B resolves its input binding name to the same `DataRef`.
4. Stage B work units read required regions from storage.

This design gives:
- stable cross-stage handoff,
- disk spill support,
- process-safe exchange (no large in-memory payload passing),
- predictable memory behavior via chunked access.

## Up-Front Allocation Strategy

A core design choice is allocating stage outputs during planning (before execution starts):

- validates storage feasibility early,
- prevents mid-run "out of disk" surprises,
- ensures all work units target known refs from the start.

Computation memory is still budgeted at runtime by the scheduler, but output containers are planned and allocated ahead of execution.

## End-to-End Interaction

`SemanticTask` -> `AlgorithmPipeline` -> `TaskPlanner` -> (output allocation via `StorageLayer`) -> `TaskPlan` (work units + bindings) -> `WorkScheduler` -> executors -> `StorageLayer` reads/writes.

This separation keeps stage logic reusable, planning deterministic, scheduling policy-driven, and data movement explicit through bindings.
