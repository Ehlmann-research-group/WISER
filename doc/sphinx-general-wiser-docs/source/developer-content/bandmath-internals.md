# Band Math Internals

This page documents how the band-math evaluation engine works internally — from the moment a user submits an expression to the moment results are returned.

## Overview

Band math in WISER is a three-phase pipeline: **parse**, **analyze**, then **evaluate**.

```{mermaid}
flowchart LR
    A["User expression\n(string)"] --> B["Parser\nbandmath.lark"]
    B --> C["Parse tree\nlark.Tree"]
    C --> D["Analyzer\nBandMathAnalyzer"]
    D --> E["BandMathExprInfo\ntype · shape · metadata"]
    E --> F["Evaluator\n(subprocess)"]
    F --> G["Result\nnumpy array / on-disk dataset"]
```

Each phase has a distinct responsibility:

- **Parse** — turn the expression string into a tree without touching any data.
- **Analyze** — walk the tree to infer the result type, shape, and metadata.
- **Evaluate** — walk the tree a second time to actually compute values.

---

## Phase 1: Parsing

**Files:** `src/wiser/bandmath/parser.py`, `src/wiser/bandmath/bandmath.lark`

The Lark library parses the expression string against the grammar in `bandmath.lark`. The grammar covers arithmetic (`+`, `-`, `*`, `/`, `**`), comparison operators (`==`, `!=`, `<`, `>`, `<=`, `>=`), unary negation, variables, numeric/string/boolean literals, and function calls.

Key functions:

- `parse_bandmath(expr)` — returns a `lark.Tree`
- `get_bandmath_variables(expr)` — uses a `VariableCollector` visitor to return the set of variable names referenced in the expression
- `bandmath_parses(expr)` — returns `True` if the expression is syntactically valid

The parser does not load any raster data and has no side effects.

---

## Phase 2: Analysis

**Files:** `src/wiser/bandmath/analyzer.py`, `src/wiser/bandmath/types.py`

`BandMathAnalyzer` is a Lark `Transformer` that walks the parse tree and calls each operator's `analyze()` method instead of `apply()`. No pixel data is loaded.

The result is a `BandMathExprInfo` object that carries:

| Field | Purpose |
|---|---|
| `result_type` | `VariableType` enum — `IMAGE_CUBE`, `IMAGE_BAND`, `SPECTRUM`, `NUMBER`, `BOOLEAN` |
| `shape` | Array dimensions (e.g. `(bands, rows, cols)` for `IMAGE_CUBE`) |
| `elem_type` | NumPy dtype of the output (e.g. `float32`) |
| `spectral_metadata_source` | Where to inherit wavelength information from |
| `spatial_metadata_source` | Where to inherit projection / geotransform from |

`BandMathExprInfo.result_size()` multiplies `shape` by `elem_type.itemsize` to get the estimated byte size of the result — this feeds directly into the memory-management decision in Phase 3.

### Type-broadcasting rules (enforced at analysis time)

| Left operand | Right operand | Result |
|---|---|---|
| `IMAGE_CUBE` | `NUMBER` | `IMAGE_CUBE` (scalar broadcast) |
| `IMAGE_CUBE` | `SPECTRUM` | `IMAGE_CUBE` (spectrum broadcast along spatial dims) |
| `IMAGE_CUBE` | `IMAGE_BAND` | `IMAGE_CUBE` (band broadcast along spectral dim) |
| `IMAGE_CUBE` | `IMAGE_CUBE` | `IMAGE_CUBE` (element-wise; shapes must match) |
| `SPECTRUM` | `SPECTRUM` | `SPECTRUM` (element-wise; lengths must match) |
| `IMAGE_BAND` | `NUMBER` | `IMAGE_BAND` (scalar broadcast) |

Incompatible shapes or types raise a `BandMathEvalError` during analysis, before any subprocess is spawned.

---

## Phase 3: Evaluation Entry Point

**File:** `src/wiser/bandmath/evaluator.py`

### `start_bandmath_evaluation()`

This is the public entry point called by the GUI. It:

1. Lowercases all variable and function names for case-insensitive matching.
2. Parses the expression and assigns unique IDs to every tree node (via `UniqueIDAssigner`) — node IDs are used later to route per-operator read-ahead queues.
3. Counts intermediate values the expression will create (`NumberOfIntermediatesFinder`) to estimate peak memory.
4. Serializes all variables — `RasterDataSet`, `RasterDataBand`, and `Spectrum` objects wrap GDAL handles that cannot be pickled, so they are converted to a serializable form before being passed to the subprocess.
5. Creates a `BandMathJob` and passes it to a `ProcessManager`, which spawns a worker subprocess.
6. Returns immediately. Results are delivered later through callbacks:
   - `succeeded_callback(result)` — called with the computed value
   - `error_callback(task)` — called if evaluation raises
   - `status_callback(msg)` — called with progress updates

The subprocess entry point is `bandmath_subprocess_entrypoint()` → `eval_bandmath_expressions()`.

---

## Single vs. Batch Evaluation

### `BandMathJob` and `SingleBandMathJob`

`BandMathJob` is the serializable job container passed to the subprocess. On construction it inspects `serialized_variables` to determine whether the job is a batch job:

- If any variable has type `IMAGE_CUBE_BATCH` or `IMAGE_BAND_BATCH`, `is_batch = True` and `filepaths` is populated with the list of files from the batch folder.
- Otherwise, `is_batch = False` and `filepaths` is empty.

`BandMathJob` is **iterable**. Each iteration step yields a `SingleBandMathJob` — a fully resolved, self-contained job for one evaluation target (one file). For a single job it yields exactly one `SingleBandMathJob`; for a batch job it yields one per filepath.

For each filepath the iterator:
- Deserializes variables with the filepath as context (loading the specific raster file).
- Recomputes `BandMathExprInfo` for those variables.
- Prefixes the result name with the filename stem (e.g. a batch variable `_result` from file `ang20171108.img` becomes `ang20171108_result`).

### Dispatch flow

```{mermaid}
flowchart TD
    A["eval_bandmath_expressions()"] --> B{"BandMathJob.is_batch?"}
    B -- No --> C["get_single_bandmath_job()\nyields one SingleBandMathJob"]
    C --> D["eval_singular_bandmath_expr()"]
    B -- Yes --> E["eval_bandmath_batch()\niterates BandMathJob"]
    E -- "per filepath" --> F["yield SingleBandMathJob\nfor this filepath"]
    F --> D
    E -- "after all files complete" --> G["Collect results\nreport per-file progress\ncontinue on error"]
```

### Key differences

| Aspect | Single | Batch |
|---|---|---|
| Variable loading | Once from in-memory serialized refs | Once per file, re-deserialized from disk |
| `expr_info` computation | Once | Once per file |
| Result naming | Unchanged from user input | Prefixed with filename stem |
| Error handling | Propagates immediately; aborts | Continues; per-file errors collected alongside successes |
| Progress reporting | `1 / 1` | `N completed / total files` |
| Return value | Single result wrapped in a list | List of results (one per file) |

---

## Sync vs. Async Transformer

`eval_singular_bandmath_expr()` contains the central decision: which evaluator to use.

```{mermaid}
flowchart TD
    A["eval_singular_bandmath_expr()"] --> B["Compute estimated peak memory:\nexpr_info.result_size() × number_of_intermediates"]
    B --> C{"result_type == IMAGE_CUBE\nAND memory exceeds threshold?"}
    C -- Yes --> ASYNC["eval_singular_bandmath_expr_async()\nBandMathEvaluatorAsync"]
    C -- No --> D{"use_synchronous_method\nexplicitly False?"}
    D -- Yes --> ASYNC
    D -- No --> SYNC["eval_singular_bandmath_expr_sync()\nBandMathEvaluator"]
```

**Memory threshold:** `max_bytes_to_chunk(expr_info.result_size() * number_of_intermediates)`

Constants (from `src/wiser/bandmath/builtins/constants.py`):

| Constant | Value | Meaning |
|---|---|---|
| `MAX_RAM_BYTES` | 4 000 000 000 (4 GB) | Ceiling used if platform query fails |
| `RATIO_OF_MEM_TO_USE` | 0.25 | Use at most 25 % of available RAM |
| `NUM_READERS` | 4 | Async read-thread-pool size |
| `NUM_WRITERS` | 1 | Async write-thread-pool size |

**In practice:** spectra, scalars, and single-band results always take the sync path because they are not `IMAGE_CUBE`. Large hyperspectral cubes almost always exceed 25 % of available RAM and take the async path.

---

## Synchronous Transformer (`BandMathEvaluator`)

**Class:** `BandMathEvaluator(lark.visitors.Transformer)`

The synchronous evaluator is a standard Lark `Transformer`. It walks the parse tree depth-first, calling a handler method for each grammar rule. Each handler runs `asyncio.run_coroutine_threadsafe(operator.apply(...), event_loop).result()` — this executes the operator's async `apply()` coroutine on a background event loop and blocks the calling thread until it completes. Essentially making it synchronous, hence the name.

All data is loaded into memory before evaluation begins. There is no chunking, no read-ahead, and no intermediate disk writes. The result is a `BandMathValue` holding an in-memory NumPy array.

**Use cases:**

- Results of type `IMAGE_BAND`, `SPECTRUM`, `NUMBER`, `BOOLEAN`, or `STRING`
- Small `IMAGE_CUBE` results that fit within the memory threshold
- Scalar expressions (always negligible memory)

---

## Asynchronous Transformer (`BandMathEvaluatorAsync`)

### Class hierarchy

```
lark.visitors.Transformer
├── AsyncTransformer            ← custom base: async def visitor methods
│   └── BandMathEvaluatorAsync  ← band-math evaluation with chunking and I/O
└── BandMathEvaluator           ← synchronous, no chunking
```

### `AsyncTransformer`

`AsyncTransformer` mirrors Lark's `Transformer` but makes every step awaitable:

- `_transform_children()` — creates a `asyncio.Task` per child node, then awaits all of them concurrently with `asyncio.gather()`. This allows for I/O to happen in parallel which reduces time spent evaluating.
- `_transform_tree()` — awaits `_transform_children()`, then calls the handler for the current node.
- `_call_userfunc()` — detects whether the handler method is `async def` or a regular `def` and awaits it accordingly.

### Band-window chunking

For an `IMAGE_CUBE` result, the spectral dimension is split into **band windows**. `compute_bands_per_chunk()` calculates the maximum number of bands that fit in the available memory budget. `iter_band_windows(total_bands, num_bands)` yields `(current_bands, next_bands)` pairs — the second element is the window that will be needed next, enabling read-ahead.

```{mermaid}
flowchart LR
    A["All bands\ne.g. 432 bands"] -->|"compute_bands_per_chunk()"| B["Window size\ne.g. 50 bands"]
    B --> C["Window 0–49\ncurrent + next"]
    B --> D["Window 50–99\ncurrent + next"]
    B --> E["Window 100–149\n..."]
    B --> F["..."]
```

Each window is evaluated independently, and its result bands are written to an on-disk GDAL dataset incrementally.

### Read-ahead pipeline

The async evaluator runs two thread pools in addition to the `asyncio` event loop:

- `_read_thread_pool` — 4 workers; reads raster band data from disk
- `_write_thread_pool` — 1 worker; writes computed band windows back to disk

Each expression tree node (operator) gets a pair of `queue.Queue` objects keyed by `LHS_KEY` and `RHS_KEY`. The operator submits the *next* window's read to the thread pool before it even starts computing the *current* window. By the time the current window's computation finishes, the next window's data is often already in the queue.

```{mermaid}
sequenceDiagram
    participant EL as Event Loop
    participant RP as Read Thread Pool<br/>(4 workers)
    participant WP as Write Thread Pool<br/>(1 worker)
    participant Disk

    loop Each band window
        EL->>RP: submit read — current window (LHS + RHS)
        EL->>RP: submit read — NEXT window (prefetch)
        RP->>Disk: read current bands
        RP-->>EL: current data ready (via queue)
        EL->>EL: evaluate operators<br/>(asyncio.gather children)
        EL->>WP: submit write — current result bands
        Note over RP,Disk: next-window read runs concurrently<br/>with current-window write
    end
```

**Why this matters:** disk I/O is the bottleneck for large hyperspectral cubes. Without read-ahead, every band window waits for both a read and a compute before the next read begins. With read-ahead, the next read overlaps the current compute-and-write, hiding most of the I/O latency.

### Result storage

The async path writes results to a temporary on-disk GDAL dataset (`IMAGE_CUBE_DATASET` type) rather than accumulating them in RAM. The dataset is passed back to the GUI as a `RasterDataSet`. This is why the async path can handle datasets much larger than available RAM.

The file is written into `TEMP_FOLDER_PATH`, which is a `temp_output/` directory sitting inside the `wiser/bandmath/` package directory — not the OS temp folder. This means the files persist across runs if WISER crashes, but are explicitly swept on a clean exit.

Immediately after the GDAL dataset is created, `out_dataset.set_dirty()` is called on the `RasterDataSet` wrapper. This marks the dataset as dirty, which prevents the `RasterDataSet` from deleting its backing file when the **subprocess** exits — the file must survive long enough for the **main process** to read it back. Without this flag the file would be cleaned up as the subprocess tears down, and the result would be lost.

The main process cleans up all files in `TEMP_FOLDER_PATH` inside `app.py`'s `closeEvent()`, which runs when the user closes WISER. This means temp files from a previous session that survived a crash will also be removed on the next clean shutdown.

---

## Operator System

**Files:** `src/wiser/bandmath/builtins/`

Every operator is a class that implements two methods:

```python
class BandMathFunction:
    def analyze(self, infos: List[BandMathExprInfo]) -> BandMathExprInfo:
        """Called during Phase 2. Infers result type without touching data."""
        ...

    async def apply(self, args: List[BandMathValue], index_list, ...) -> BandMathValue:
        """Called during Phase 3. Computes the actual result for the current band window."""
        ...
```

Built-in operators: `OperatorAdd`, `OperatorSubtract`, `OperatorMultiply`, `OperatorDivide`, `OperatorPower`, `OperatorCompare`, `OperatorUnaryNegate`, `OperatorTrigFunction`, `OperatorDotProduct`.

`apply()` receives the current `index_list` (the set of band indices in the current window), which operators use to slice or broadcast their operands to the right shape before computing.

Plugin-defined functions follow the same `BandMathFunction` interface and are registered via `BandMathPlugin` — see the plugin documentation for details.

---

## End-to-End Data Flow

```{mermaid}
flowchart TD
    GUI["GUI\nstart_bandmath_evaluation()"]
    GUI -->|"serialize vars\ncreate BandMathJob"| PM["ProcessManager\nspawns subprocess"]
    PM -->|"IPC pipe + queue"| SUB["bandmath_subprocess_entrypoint()"]
    SUB --> EBE["eval_bandmath_expressions()"]

    EBE -->|"single job"| ESB["eval_singular_bandmath_expr()"]
    EBE -->|"batch: one per file"| ESB

    ESB -->|"IMAGE_BAND / SPECTRUM\nNUMBER / small IMAGE_CUBE"| SYNC["eval_singular_bandmath_expr_sync()\nBandMathEvaluator\nin-memory numpy array"]
    ESB -->|"large IMAGE_CUBE\nor use_synchronous_method=False"| ASYNC["eval_singular_bandmath_expr_async()\nBandMathEvaluatorAsync\nchunked + read-ahead"]

    SYNC -->|"in-memory numpy"| RES["Result tuple\n(type, value, name, expr_info)"]
    ASYNC -->|"on-disk GDAL dataset\n(RasterDataSet)"| RES

    RES -->|"serialized\nput on return_queue"| PM
    PM -->|"succeeded_callback"| GUI
```

---

## Summary: Choosing the Right Path

| Scenario | Path | Why |
|---|---|---|
| Expression produces a scalar or spectrum | Sync | Trivially small; no benefit to chunking |
| Expression produces a single-band image | Sync | 2-D data rarely exceeds the 25 % threshold |
| Expression produces a large hyperspectral cube | Async | Chunking prevents OOM; read-ahead hides I/O latency |
| `use_synchronous_method=False` passed explicitly | Async | Caller forces chunked execution regardless of size |
| Batch mode over a folder | Same decision per file | Each file is evaluated independently through `eval_singular_bandmath_expr()` |
