# WISER Handoff Notes

**Author:** Josh (departing maintainer)

**Primary reader:** Matthew.

**Purpose:** Capture the things that only live in my head. Anyone can read the code and see *what* it does; this doc records *why* it is that way, which parts are load-bearing vs. which parts I now think were mistakes, and the traps that are easy to fall back into.

## How this doc is ordered

Entries are ranked by **regression risk** — "how bad is it if I forget this and no one wrote it down." An item is high on this list if forgetting it causes one of:

- **Silent regression** — new code drifts back to a bad pattern, contribution by contribution.
- **Reintroduced bug** — someone re-enables something I already proved is broken.
- **Wrong rewrite** — someone preserves complexity I know is a mistake, or rips out something that is actually load-bearing.

Lower-ranked entries are rationale that mainly prevents *wasted effort* (someone re-litigating a decision or re-running an experiment), not active regression.

Each entry is written ADR-style: **Context → Current state / Decision → Why it regresses if forgotten → What I'd do → What I'm unsure about → Where to look.**

Forward-looking work (features to build, the roadmap) lives in [ROADMAP.md](ROADMAP.md). Concrete, filable work items live in [ISSUE_DRAFTS.md](ISSUE_DRAFTS.md). This doc is only the tacit knowledge.

---

## Tier 1 — Active landmines (regress silently or reintroduce bugs)

### 1. Moving work off the GUI thread is *the standard*, and there must be one documented way to do it

- **Context:** WISER's UI is PySide/Qt. Any non-trivial computation run on the GUI thread freezes the whole app. Because of Python's GIL, threads are concurrent but not parallel, so genuinely CPU-bound work goes to a **subprocess** (via the process pool in [work_scheduler.py](src/wiser/utils/work_scheduler.py)); I/O-bound work can go to a **thread**. See entry #7 for the full rationale on why subprocesses.
- **Current state:** Most heavy work is off-thread, but not all. Known offenders still on the GUI thread: **dataset loading**, **stretch builder** updates, and **ROI average-spectra computation** ([rasterpane.py](src/wiser/gui/rasterpane.py)). There is no single, documented "here is the one correct way to push work off-thread" — so people copy whichever nearby example they find.
- **Important correction on "dataset loading":** the freeze when opening a dataset is almost certainly **not the open — it's the first render**. [`load_from_file`](src/wiser/raster/loader.py#L87) reads metadata, not pixels; the pixel read, masking, normalization, stretch, and compositing all happen in [`RasterView.update_display_image()`](src/wiser/gui/rasterview.py#L892). That matters because `update_display_image()` is a **shared funnel**: opening a dataset ([`set_raster_data`](src/wiser/gui/rasterview.py#L703)), a stretch-builder change ([`set_stretches`](src/wiser/gui/rasterview.py#L643)), and a band-chooser change ([`set_display_bands`](src/wiser/gui/rasterview.py#L762)) all land in the same code. So "dataset loading" and "stretch builder" are not two independent offenders — they're one rendering problem with three entry points, and it renders once per pane (context, main, zoom). Off-threading it is therefore higher-value *and* higher-risk than it looks, and the `QImage`/`QPixmap` tail has to stay on the GUI thread regardless. Details and constraints in the issue draft.
- **Why it regresses if forgotten:** This is the #1 item because it regresses on *every future contribution*. Without a written standard, new features silently land on the GUI thread and the app gets choppier over time. The fix isn't a one-time cleanup — it's a documented pattern plus the remaining offenders being migrated.
- **What I'd do:** (1) Write the canonical pattern for both cases — long CPU-bound work via the task system, and one-off subprocess/thread work via `work_scheduler`. (2) Migrate the known offenders. (3) Make the pattern easy enough that doing the right thing is the path of least resistance. See ISSUE_DRAFTS.md items *Standardize off-GUI-thread pattern*, *Async dataset loading and rendering*, *ROI spectra off-thread*.
- **What I'm unsure about:** Whether the two mechanisms (task system vs. plain `work_scheduler` subprocess/thread) should stay separate or be unified under one entry point. I lean toward keeping both but documenting them as one decision tree. Separately, for the rendering path specifically I'm unsure whether thread or subprocess wins — the output is a full H×W×3 array, so serializing it back may cost more than the parallelism buys.
- **Where to look:** [work_scheduler.py](src/wiser/utils/work_scheduler.py), [task_system.py](src/wiser/utils/task_system.py), [rasterview.py:892](src/wiser/gui/rasterview.py#L892) (the shared render funnel), [rasterpane.py](src/wiser/gui/rasterpane.py), [loader.py](src/wiser/raster/loader.py), [stretch_builder.py](src/wiser/gui/stretch_builder.py).

### 2. Do **not** enable GDAL multithreaded reads without re-verifying — 3.10 corrupted data in my testing

- **Context:** Several file formats (notably JP2 and netCDF) read slowly. Part of that is the driver; part is that we **reopen the dataset on every read** so that multiple threads can read concurrently without sharing a handle. GDAL 3.10 advertised support for multithreaded reads from a single handle, which would let us stop reopening.
- **Current state / decision:** I tested GDAL 3.10 multithreaded reads and **the data came back corrupted.** Albeit, less so than GDAL 3.9 and before. We did **not** adopt it. We still reopen datasets per read. One place this matters is band-math chunking, which does multithreaded reads on large data.
- **Why it regresses if forgotten:** This is a landmine. Someone will read the GDAL changelog, see "multithreaded reads supported," enable it to kill the reopen overhead, and reintroduce **data corruption**.
- **What I'd do:** Treat "can we drop the per-read reopen?" as an explicit, gated experiment: re-test multithreaded reads on a current GDAL against a known-good reference output before trusting it. If (and only if) it's clean, wrote a regressiom test amd ensure that dropping the reopen speeds up *all* of WISER for those formats.
- **What I'm unsure about:** Whether later GDAL releases fixed it. Worth re-testing, but assume broken until proven otherwise. I for sure tested this in the past using the AsyncTransformer in bandmath and I am pretty sure I tested it outside the class in a .py file using threading.
- **Where to look:** the dataset read paths in [dataset.py](src/wiser/raster/dataset.py); band-math chunked reads in [evaluator.py](src/wiser/bandmath/evaluator.py).

---

## Tier 2 — Over-engineered code: which complexity is a mistake vs. load-bearing

These are ordered the way I'd rank the regret. In each case the code *works* — the risk is that someone either preserves the over-engineering thinking it was intentional, or (worse for #3) extends the temptingly-fast-but-fragile path.

### 3. `AsyncTransformer` in band math — works, faster, and I'd still delete it

- **Context:** Band math parses expressions with Lark. To speed up evaluation I wrote [`AsyncTransformer`](src/wiser/bandmath/evaluator.py#L113) — a custom subclass of Lark's `Transformer` that mirrors its behavior but makes the tree-transform methods `async` and evaluates children concurrently with `asyncio`.
- **Current state:** It is in use and it *does* speed up evaluation.
- **Why it regresses if forgotten:** It's faster, so the instinct may be to *lean on it more*. My judgment: the speedup is **not worth the maintainability cost** — it's a bespoke reimplementation of a third-party class, hard for anyone else to reason about, and it entangles async into the evaluator's core. If no one records "this was a mistake," someone may expand upon it instead of removing it.
- **The part I regret most — the LHS/RHS async read queueing:** To feed `AsyncTransformer`'s concurrent evaluation, every binary operator (`OperatorAdd`, `OperatorPower`, etc. in `src/wiser/bandmath/builtins/`) reads its operands through `get_lhs_rhs_values_async` ([utils.py:750](src/wiser/bandmath/utils.py#L750)), which calls `plan_lhs_read` ([utils.py:583](src/wiser/bandmath/utils.py#L583)) and `plan_rhs_read` ([utils.py:659](src/wiser/bandmath/utils.py#L659)). Each of those maintains a **per-expression-node** `LHS_KEY`/`RHS_KEY` queue pair (`_read_data_queue_dict`, keyed by `node_id`, built up in `BandMathEvaluatorAsync` — [evaluator.py:249](src/wiser/bandmath/evaluator.py#L249)), pushes a read future onto the queue via `read_lhs_future_onto_queue`/`read_rhs_future_onto_queue` ([utils.py:824](src/wiser/bandmath/utils.py#L824)), pops it back off with `pop_future` ([utils.py:548](src/wiser/bandmath/utils.py#L548)), and separately decides whether to *prefetch* the next band chunk via `should_continue_reading_bands` ([utils.py:853](src/wiser/bandmath/utils.py#L853)). It's a hand-rolled future-queue-per-node prefetch scheduler layered on top of a thread pool and a background event loop thread — genuinely hard to hold in your head, and it only exists to feed `AsyncTransformer`. All of this needs to be removed.
- **What I'd do:** Remove `AsyncTransformer` **and** this LHS/RHS read-queueing machinery together — they're one decision, not two. The speed up is not worth it. Just ensure that
bandmath is in a subprocess so it doesn't freeze the GUI thread.
- **What I'm unsure about:** How much raw speed we lose by dropping it. I believe it will be 
about 20%, but I don't remember the file that I documented it in.
- **Where to look:** [evaluator.py:113](src/wiser/bandmath/evaluator.py#L113) (`AsyncTransformer`), [evaluator.py:249](src/wiser/bandmath/evaluator.py#L249) (`BandMathEvaluatorAsync`'s per-node queue dict), and the read-planning/queueing functions in [utils.py](src/wiser/bandmath/utils.py) (`plan_lhs_read`, `plan_rhs_read`, `pop_future`, `read_lhs_future_onto_queue`, `read_rhs_future_onto_queue`, `should_continue_reading_bands`, `get_lhs_rhs_values_async`).

### 4. Band-math chunking is band-only — and a single band can exceed memory

- **Context:** Band math was originally written assuming the whole dataset is in memory. To handle large data I added chunking, but it chunks **by band** — see `compute_bands_per_chunk` and the band-index windows around [evaluator.py:1778](src/wiser/bandmath/evaluator.py#L1778).
- **Current state:** Works for datasets whose individual bands fit in memory. Breaks down when a *single band* is too large (e.g., the Gale HiRISE images, where one band can be 2 GB+).
- **Why it regresses if forgotten:** When someone hits an OOM on a huge-band image, the tempting fix is a local patch to the band loop. The real problem is the chunking *dimension*, and patching around it entrenches the band-only assumption further.
- **What I'd do:** Rewrite so chunking is **dimension-agnostic** — able to split along whichever axis is too big, so no single dimension's size can blow memory. Route the chunks straight to the `work_scheduler` process pool, *without* the task system (this is a "one-off subprocess" case, not a long-running semantic task) and without `AsyncTransformer`. This is harder than it sounds — spatial chunking interacts with expressions that mix bands — so scope it carefully.
- **What I'm unsure about:** How cleanly arbitrary-axis chunking composes with multi-band expressions. Also how band-math **plugins** survive this — see the note below.
- **Related trap — plugins:** Band-math plugins, the plugin API, and the plugin docs were all written under the original "full dataset in memory" assumption. I never fully worked through how plugins behave under chunked operations. Any chunking rewrite has to account for existing plugins or it will break them silently. (Tracked in ROADMAP under Plugins.)
- **Where to look:** [evaluator.py](src/wiser/bandmath/evaluator.py) (chunking logic), [analyzer.py](src/wiser/bandmath/analyzer.py).

### 5. The task system is over-engineered — simplify it, don't extend it

- **Context:** The async task backend is: `WorkScheduler` → `TaskManager` → `TaskPlanner` → `TaskPlan` → `SemanticTask` → `TaskStage` → `WorkUnit`. It exists to run long, computationally expensive work off the GUI thread with low memory overhead, reporting progress back to the UI.
- **Current state:** It works and gets the job done, but it's more machinery than the problem needs — I over-engineered it. Two concrete rough edges: (a) the `TaskPlanner` isn't "smart" — it essentially **divides the work into a fixed number of pieces** rather than reacting to system state; and (b) progress reporting is coarse — there's no clean way to report *granular* progress of the currently-finishing `SemanticTask` outside of `WorkUnit` completion.
- **Why it regresses if forgotten:** If it's undocumented that this is *intentionally to-be-simplified*, the next person will assume the complexity is load-bearing and build *on top of it*, making it heavier and harder to ever unwind.
- **What I'd do:** Rewrite it simpler. Keep the two real requirements — off-thread execution and progress-to-UI — and shed other layers or rewrite to be simpler. We would want to keep the mechanism for being memory aware of how much each task is.
- **What I'm unsure about:** (a) A genuinely clean way to report granular sub-task progress — I never found one; there may not be a tidy answer. (b) "Smart" scheduling is genuinely hard because work is enqueued under one system state (resources strained) but may run under another (resources freed) — reacting to instantaneous state can mislead. Don't over-invest here. I also do not know how to do this rewrite. Personally, I feel like this is a complicated problem to solve (memory aware scheduling for complex math) and I think I just don't have the experience to solve it in a simple way.
- **Where to look:** [task_system.py](src/wiser/utils/task_system.py) (all the classes above), [work_scheduler.py:750](src/wiser/utils/work_scheduler.py#L750).

### 6. Dataset read cache — re-look it, and know it's coupled to LoD

- **Context:** `Dataset` has a per-dataset read cache (`_data_cache` → computation cache) wired through the four read paths: `get_image_data` ([dataset.py:953](src/wiser/raster/dataset.py#L953)), `get_image_data_subset` ([dataset.py:981](src/wiser/raster/dataset.py#L981)), `get_band_data` ([dataset.py:1015](src/wiser/raster/dataset.py#L1015)), and `get_band_data_normalized` ([dataset.py:1046](src/wiser/raster/dataset.py#L1046)), keyed on `(dataset, band_index, normalized?)`.
- **Current state / why it exists:** The payoff isn't uniform. For image / subset / band reads it saves **re-masking** the data-ignore value. The real win is `get_band_data_normalized`, where it saves **re-normalizing** on every call — and that's the rendering hot path. I built it for one concrete scenario: keeping the image shown in a RasterView cached so that switching back and forth between datasets, and dragging in **stretch builder**, stays responsive.
- **Why it regresses if forgotten:** Less of a landmine, more of a "don't rewrite it in isolation" note. Its fate is **coupled to two other things**.
- **What I'd do:** Investigate whether it can fold cleanly into the newer **storage service / storage client** instead of living as a separate mechanism on `Dataset`. If it can't merge cleanly, leaving it standalone is a fine outcome — this is "investigate," not "must rewrite."
- **What I'm unsure about — the kicker:** If the **LoD-pyramid rendering** work (see ROADMAP) lands and makes data access + compute cheap enough, this cache may stop earning its keep **entirely**. So don't rewrite the cache before knowing where LoD lands.
- **Where to look:** [dataset.py:1046](src/wiser/raster/dataset.py#L1046) (the normalized path — the one that matters), the `_data_cache` field around [dataset.py:592](src/wiser/raster/dataset.py#L592), and the storage service/client.

---

## Tier 3 — Rationale that prevents wasted effort (re-litigation, not regression)

### 7. Why WISER uses subprocesses at all (the ENVI / IDL / GIL story)

- **The question people will ask:** "ENVI stays responsive and fast without all this machinery — why can't WISER just spawn a thread and compute?"
- **The answer:** ENVI is written in IDL, which (like most languages) has real parallel threads and is *built* for fast array computation — it can spawn a thread, share memory, compute, done. Python can't: the GIL makes threads concurrent but not parallel, so CPU-bound work has to go to a **subprocess**. That means we **serialize data to the subprocess, compute, and serialize back** — real overhead I worked to reduce but couldn't eliminate. On top of that, Python isn't built for array compute the way IDL is; even with NumPy you pay for context switches between Python and the underlying C/C++.
- **Why this matters to record:** Without it, someone "simplifies" the architecture by moving heavy work onto threads, gets no speedup (or makes things worse), and burns days rediscovering the GIL. This is the *why* behind entry #1 and the whole task/work-scheduler design.
- **Honest caveat:** I'm not certain the low-memory / off-thread abstraction I built is the *best* trade-off for speed. It optimizes for low memory overhead and a responsive GUI, and it pays for that in data-transfer time. That trade-off is worth revisiting with fresh eyes — but revisit it *knowing why it's there*, not by accident.

### 8. Telemetry — investigated, deliberately not pursued

- **Decision:** We looked at adding telemetry to see what features people actually use. We decided **not to**.
- **Why:** Desktop-app telemetry carries heavy legal obligations — EU GDPR, California CCPA, and more. The effort to do it compliantly outweighs the value.
- **What to do instead:** If we want usage signal, chase **download statistics** first — far less legal surface area for a similar signal.
- **Why record it:** So nobody re-runs the same investigation from scratch. If it's ever revisited, start from "compliance is the hard part," not "how do we collect events."

---

## Pointers

- Forward-looking priorities and the full feature taxonomy: [ROADMAP.md](ROADMAP.md)
- Filable, templated work items referenced above: [ISSUE_DRAFTS.md](ISSUE_DRAFTS.md)
