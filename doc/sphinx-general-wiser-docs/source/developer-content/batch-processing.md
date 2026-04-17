# Batch Processing

> **Status:** In development. Working branch: `feat/batch-proc`.

This document captures the requirements and design notes for batch processing
in WISER.

## Overview

We want a way to queue a batch of tasks and run processes on them. Tasks should
have a progress bar and the ability to be cancelled. This should also allow
many operations that previously blocked the Qt thread to run without hanging.

There are two primary use cases: batch processing for plugins and batch
processing for band math.

## Plugin Batch Processing

### UI

A "Batch Processing" button in the Tools menu opens a dialog that lets the user
select batch-processing plugins and, for each plugin, select the datasets to run
it on. Dataset selection could be done with a series of dropdowns similar to
the band-math variable binding UI.

### Backend

- Load all plugin functions and their parameters.
- Spawn a separate process for each plugin run up to a configurable limit; once
  the limit is reached, queue remaining runs per process.
- Shared memory will be required to pass dataset data to worker processes.

### Batch Processing Plugin Class

A batch-processing plugin class must:

- Accept inputs of types: image cube, image band, spectrum, or number. Any
  number of inputs is allowed, but types must be declared upfront so the GUI can
  present appropriate binding controls.
- Handle its own processing (i.e., be serialisable and runnable in a subprocess).
- Declare the type(s) of objects it produces as output.

The plugin must expose:

- A function listing input parameters (name + type) — used to build the GUI.
- A function listing output parameters (name + type) — used to determine where
  to save results.

### Order of Work

1. Confirm that multiprocessing in WISER can share dataset data across
   processes.
2. Write the `BatchProcessingPlugin` base class; confirm it is picklable and
   can be executed in a subprocess.
3. Write the GUI for the batch processing dialog.

## Band Math Batch Processing

### UI Changes

- A toggle button in a new row (Row A, index 0) enables/disables batch mode.
  When enabled, additional controls appear.
- **Row A**: "Input Folder Path:" label + disabled read-only line edit (display
  only) + "Select Folder" push button.
- **Row B**: "Output Folder Path:" label + similar folder picker. Two check
  boxes: "Load Into WISER" and "Save on File System". At least one must be
  selected before "Create Batch Job" is enabled.
- The "Result name (optional):" label changes to "Result suffix (required):" in
  batch mode.
- A "Create Batch Job" button (disabled when batch mode is off) appends the
  current expression and variable bindings to a batch job table. The expression
  is not reset after adding. Duplicate job detection with a confirmation prompt.
- When batch mode is enabled, two additional variable types appear: "Image
  (batch)" and "Image band (batch)". The "Image band (batch)" binding lets the
  user choose between band number or wavelength value.

### Backend

- Add `BandMathVariableType` entries for batch inputs.
- `BandMathExprInfo` must detect batch inputs and handle them without changing
  the standard (non-batch) code paths.
- When evaluating, extract each file from the input folder and feed it into the
  standard band-math function. Results go to the output folder and/or are loaded
  into WISER as specified.
- New state variables needed: `_input_folder`, `_output_folder`,
  `_load_into_wiser`, `_save_on_filesystem`.

### Known Technical Challenges

GDAL `Dataset` objects are not serialisable (cannot be pickled) and cannot be
memory-mapped. Windows does not support `fork`, so process spawning with GDAL
objects is non-trivial. Options being explored:

- Pass file paths to worker processes and re-open datasets there.
- Use shared memory for raw array data extracted before spawning.

Relevant background:

- [Picklability of GDAL files](https://chatgpt.com/c/689ce320-4f9c-8330-8bd0-2e97b3621e77)
- [Spawn process copying data](https://chatgpt.com/c/689ce040-289c-8323-89b6-8c6987d448a1)

## Notes on Concurrency

From Nia (internal discussion):

> I typically use `from concurrent.futures import ProcessPoolExecutor,
> ThreadPoolExecutor`. QThreads are useful if you need to emit Qt signals, but
> are otherwise unnecessary. For most compute work, Python's standard thread and
> process pool executors are preferred.

