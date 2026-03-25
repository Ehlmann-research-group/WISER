# WISER Storage and Memory Management

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

## TaskStage Allocations

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

## Data Lifetime and Deletion

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
