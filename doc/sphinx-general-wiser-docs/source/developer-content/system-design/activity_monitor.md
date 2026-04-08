# Activity Monitor Update Flow

This page describes how task execution state moves through the scheduler stack and ends up in the activity monitor UI. The important design point is that the activity monitor does not derive progress on its own. It is a presentation layer driven by scheduler and task-manager events.

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

`ActivityMonitorDialog.register_task(...)` creates the active-task row and returns the integer `activity_id` that will be used for all later updates.

The row starts in the `idle` state and includes:

- title and metadata display,
- progress bar,
- cancel button,
- disabled "View Errors" button.

The cancel button does not cancel the task by itself. It calls the callback that `TaskManager` provided, which in the normal scheduler-backed path is `scheduler.cancel_plan(task_plan.plan_id)`.

## Progress Update Flow

Progress is work-unit-based, not time-based and not stage-percentage-based.

### When progress is emitted

`WorkScheduler` emits a progress update after a work unit reaches a terminal state inside `_on_unit_done(...)`.

That happens for both:

- successful work units,
- failed work units.

This is intentional. A failed work unit is still finished from the scheduler's point of view, so it contributes to completed work.

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

The monitor also exposes `set_task_progress(activity_id, value)` for simpler percentage-style updates, but the scheduler-backed flow uses `set_task_progress_update(...)`.

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

If the plan eventually ends with recorded failures, the scheduler completes the plan future with an exception. In that path, error text has already been surfaced via `append_task_error(...)`.

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

When the user clicks the row's Cancel button, `ActivityMonitorDialog._cancel_task(...)` invokes that callback. If the callback itself raises, the row is marked failed immediately through:

- `set_task_failed(activity_id, str(exc))`

If the callback returns normally, the dialog marks the row cancelled locally with:

- `set_task_cancelled(activity_id)`

Separately, the scheduler's own cancellation path emits `task_cancelled`, and `TaskManager._on_task_cancelled(...)` also forwards that into:

- `ActivityMonitorDialog.set_task_cancelled(activity_id)`

In practice this means the UI can reflect cancellation immediately from the button path while still remaining compatible with scheduler-driven cancellation events.

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

`append_task_error(...)` records error detail without forcing terminal state. `set_task_failed(...)` transitions the row into the failed state and schedules it to move into the finished section.

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

- Treat progress as completed work units out of total work units unless your subsystem has a better discrete execution model.
- Send progress when work becomes terminal, not while it is merely in flight.
- Append errors immediately; do not wait until the entire plan has failed.
- Let the execution layer decide whether a task is truly finished, failed, or cancelled.
- Keep the dialog as a sink for state, not a source of execution truth.

## Summary

The scheduler owns execution truth, `TaskManager` owns translation and routing, and `ActivityMonitorDialog` owns presentation. That split keeps execution policy out of the UI and gives other systems a small, usable API for reporting progress, errors, cancellation, and completion in a consistent way.
