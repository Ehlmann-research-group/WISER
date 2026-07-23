# WISER Issue Drafts

Filable drafts for the work referenced in [ROADMAP.md](ROADMAP.md) and [HANDOFF.md](HANDOFF.md). Each is written so it can be pasted into a GitHub issue with minimal editing.

Every draft uses the same template. The field that matters most is **What I'm unsure about** — that's the part only I can fill in, and it's what saves the next person from a dead end.

> **Template:** Problem · Why it matters · What I'd do · What I'm unsure about · Where to look · Suggested labels

---

## Off the GUI thread

### Standardize the off-GUI-thread pattern

- **Problem:** There's no single documented way to push work off the GUI thread, so new code copies whatever nearby example it finds — and some of it lands on the GUI thread.
- **Why it matters:** This regresses on *every* future contribution (HANDOFF #1). A written standard is what stops the slow drift back to a choppy UI.
- **What I'd do:** Document the decision tree — long CPU-bound work → task system; one-off subprocess/thread work → `work_scheduler` directly. Provide a minimal helper/example for each so the right way is the easy way. Then migrate the known offenders (see the two drafts below).
- **What I'm unsure about:** Whether the task system and the plain `work_scheduler` path should be unified under one entry point or kept as two documented options. I lean toward keeping both.
- **Where to look:** [work_scheduler.py](src/wiser/utils/work_scheduler.py), [task_system.py](src/wiser/utils/task_system.py).
- **Labels:** backend, architecture, docs, priority:high

### Async dataset loading

- **Problem:** Dataset loading runs synchronously on the GUI thread and freezes the app for large files.
- **Why it matters:** Loading is one of the first things a user does; freezing there is a bad first impression and it's a known offender.
- **What I'd do:** Move loading off-thread using the standardized pattern above, with **activity-monitor status updates** so the user sees progress.
- **What I'm unsure about:** Whether the reopen-per-read model (HANDOFF #2) complicates an async load, and how it interacts with the read cache (HANDOFF #6).
- **Where to look:** dataset construction paths in [dataset.py](src/wiser/raster/dataset.py); the activity monitor.
- **Labels:** backend, ux, priority:high

### ROI average-spectra off the GUI thread

- **Problem:** Computing ROI average spectra runs synchronously on the GUI thread.
- **Why it matters:** Known offender; scales with ROI size and freezes interaction.
- **What I'd do:** Move the computation off-thread via the standardized pattern.
- **What I'm unsure about:** Whether the current call site makes it easy to return incrementally (for progress) or only as one result.
- **Where to look:** [rasterpane.py](src/wiser/gui/rasterpane.py).
- **Labels:** backend, ux

---

## Backend rewrites (see HANDOFF Tier 2)

### Rewrite band math chunking & remove AsyncTransformer

- **Problem:** Band-math chunking splits **by band**, so a single oversized band (e.g. ~2 GB Gale HiRISE bands) OOMs. Separately, `AsyncTransformer` is a bespoke async reimplementation of Lark's `Transformer` that's fast but a maintainability liability.
- **Why it matters:** The band-only assumption is baked in and will keep producing OOMs that get patched locally; `AsyncTransformer` is a trap that invites *more* use because it's faster (HANDOFF #3, #4).
- **What I'd do:** Make chunking **dimension-agnostic** so no single axis can blow memory. Route chunks to the `work_scheduler` process pool directly (no task system — this is one-off work), and **delete `AsyncTransformer`**, recovering parallelism from process-pool chunking instead.
- **What I'm unsure about:** How arbitrary-axis chunking composes with multi-band expressions; how much raw speed is lost dropping `AsyncTransformer` (measure it); and how existing band-math plugins survive chunking (see plugin draft).
- **Where to look:** [evaluator.py:113](src/wiser/bandmath/evaluator.py#L113) (`AsyncTransformer`), [evaluator.py:1778](src/wiser/bandmath/evaluator.py#L1778) (`compute_bands_per_chunk`), [analyzer.py](src/wiser/bandmath/analyzer.py).
- **Labels:** backend, band-math, refactor, priority:high

### Simplify the task system

- **Problem:** The task backend (`WorkScheduler → TaskManager → TaskPlanner → TaskPlan → SemanticTask → TaskStage → WorkUnit`) is more machinery than the problem needs.
- **Why it matters:** If it's not recorded as *intentionally to-be-simplified*, the next person builds on top of it and it gets heavier (HANDOFF #5).
- **What I'd do:** Rewrite simpler, preserving only the two real requirements — off-thread execution and progress-to-UI. Remove abstraction before adding any "smart" scheduling.
- **What I'm unsure about:** A clean way to report granular sub-task progress (may not exist); whether "smart" scheduling is worth it given work is enqueued and run under different system states.
- **Where to look:** [task_system.py](src/wiser/utils/task_system.py), [work_scheduler.py:750](src/wiser/utils/work_scheduler.py#L750).
- **Labels:** backend, refactor

### Re-look the dataset read cache

- **Problem:** `Dataset._data_cache` is a standalone read cache predating the storage service/client. Its main win is avoiding re-normalization in `get_band_data_normalized` (the render hot path).
- **Why it matters:** It shouldn't be rewritten in isolation — it's coupled to both the storage system and to LoD rendering (HANDOFF #6).
- **What I'd do:** Investigate folding it into the storage service/client. If it doesn't merge cleanly, leaving it standalone is acceptable.
- **What I'm unsure about — the kicker:** If LoD rendering lands, this cache may become unnecessary entirely. Don't rewrite before LoD lands.
- **Where to look:** [dataset.py:1046](src/wiser/raster/dataset.py#L1046), `_data_cache` around [dataset.py:592](src/wiser/raster/dataset.py#L592), storage service/client.
- **Labels:** backend, investigate

### Scheduler system-aware planning

- **Problem:** `TaskPlanner` divides work into a fixed number of pieces rather than reacting to system state.
- **Why it matters:** Fixed splitting leaves resources on the table (or oversubscribes them) depending on machine and moment.
- **What I'd do:** Explore reacting to available resources — cautiously.
- **What I'm unsure about:** Work is enqueued under one system state (strained) but may run under another (freed), so reacting to instantaneous state can mislead. Low priority; don't over-invest.
- **Where to look:** `TaskPlanner` in [task_system.py:386](src/wiser/utils/task_system.py#L386).
- **Labels:** backend, enhancement, low-priority

### Granular task progress reporting

- **Problem:** No clean way to report fine-grained progress of the finishing `SemanticTask`.
- **Why it matters:** Coarse progress bars feel stuck on long operations.
- **What I'd do:** Find a way to surface sub-task progress without threading progress plumbing through every layer.
- **What I'm unsure about:** Whether a *clean* solution exists at all — I never found one.
- **Where to look:** `SemanticTask` at [task_system.py:757](src/wiser/utils/task_system.py#L757).
- **Labels:** backend, ux

---

## Rendering & responsiveness

### LoD-pyramid rendering for RasterView

- **Problem:** `RasterView` holds the displayed image at ~O(H×W) RAM. `MosaicView` already renders via an O(1)-RAM level-of-detail pyramid.
- **Why it matters:** Highest-ceiling item in WISER — image memory from **O(H×W) → O(1)** would make it a genuinely respectable hyperspectral tool. It's also a dependency node: it may retire the read cache and largely fix the stretch-builder freeze.
- **What I'd do:** Make `RasterView` render through MosaicView's LoD pyramid. Design doc first.
- **What I'm unsure about:** What assumptions MosaicView's renderer bakes in (data source, tiling, CRS) that RasterView doesn't currently satisfy.
- **Where to look:** the MosaicView rendering path; [dataset.py](src/wiser/raster/dataset.py) read paths.
- **Labels:** backend, rendering, big-bet, priority:high

### Stretch builder cancellable & progress

- **Problem:** Stretch changes can freeze WISER for minutes on huge images; it updates context pane, zoom pane, and main view.
- **Why it matters:** This is the core of user interaction; a multi-minute freeze with no escape is unacceptable on large data.
- **What I'd do:** Add a loading indicator / progress (ideally per-stage) and make the stretch operation **cancellable** so users can reach background analysis tools. Ultimately subsumed by LoD rendering.
- **What I'm unsure about:** How to emit granular per-stage progress from the stretch pipeline cleanly (related to granular task progress).
- **Where to look:** stretch builder UI + the stretch computation path.
- **Labels:** ux, rendering

---

## File I/O

### Speed up JP2 / netCDF I/O

- **Problem:** TIFF and BSQ `.hdr` read fast; JP2 and netCDF are slow — partly the driver, partly that we reopen the dataset on every read.
- **Why it matters:** This I/O is on the critical path for *everything* WISER does with those formats.
- **What I'd do:** Profile driver cost vs. reopen cost. Gated experiment: re-test whether a current GDAL supports safe single-handle multithreaded reads so we can drop the reopen.
- **What I'm unsure about — landmine:** GDAL 3.10 multithreaded reads **corrupted data** in my testing (HANDOFF #2). Assume broken until re-verified against a known-good reference. Do not enable blindly.
- **Where to look:** read paths in [dataset.py](src/wiser/raster/dataset.py); band-math chunked reads in [evaluator.py](src/wiser/bandmath/evaluator.py).
- **Labels:** backend, io, performance

---

## GIS features

### Georeferencer errors as message boxes

- **Problem:** Georeferencer failures (e.g. mapping between CRSs with no valid transform) don't surface clearly to the user.
- **Why it matters:** Silent/opaque failures make georeferencing feel broken.
- **What I'd do:** Catch these and show a message box explaining what went wrong.
- **What I'm unsure about:** The full set of failure modes worth distinguishing vs. a generic dialog.
- **Where to look:** [georef_warp.py](src/wiser/raster/georef_warp.py).
- **Labels:** ux, gis, good-first-issue

---

## QoL

### Native dark / light mode

- **Problem:** WISER can be in OS dark mode while its icons stay dark and become invisible.
- **Why it matters:** Basic usability for dark-mode users.
- **What I'd do:** Detect the OS theme and swap icons to light/dark to match; support both natively.
- **What I'm unsure about:** Cross-platform theme detection reliability in PySide.
- **Where to look:** icon loading / theming in the GUI layer.
- **Labels:** ux, good-first-issue

### QoL grab-bag: save-to-project, help buttons, easier save/edit

- **Problem:** Several small QoL gaps: Save to Project File, help buttons on tools, easier ways to save artifacts, easier ways to edit datasets.
- **Why it matters:** Each is small but collectively shapes how polished WISER feels.
- **What I'd do:** File individually as they're picked up; listed together here so they aren't lost.
- **What I'm unsure about:** Priority order among them — driven by user feedback.
- **Where to look:** relevant tool dialogs; project persistence under [src/wiser/project/](src/wiser/project/).
- **Labels:** ux, enhancement

---

## Plugins

### Batch processing plugin

- **Problem:** Batch processing was meant as a programmatic way to drive WISER but drifted into band-math-only.
- **Why it matters:** A real batch system is the low-effort path to programmatic WISER before a full package/API exists.
- **What I'd do:** Build a batch plugin: wrap analysis tools in a data structure a batch back end + GUI can drive. Extend so one step's output feeds the next — needs rudimentary typing (cube / band / spectrum) to decide which next steps are valid; allow a band-math expression as a step.
- **What I'm unsure about:** How much typing is needed — likely lighter than band-math typing, but the output→input chaining forces *some*.
- **Where to look:** existing batch processing in band math; analysis tool entry points.
- **Labels:** feature, plugins, batch

### Packaged plugin files

- **Problem:** No easy way to share plugins; hard when a plugin has dependencies outside WISER's.
- **Why it matters:** Shareable plugins massively raise the odds people actually use plugins.
- **What I'd do:** A `.wiserproj`-style bundle for plugins. **Preferred: option 1** — package all dependencies into the bundle (simpler, larger files). Option 2 — a manifest WISER reloads at runtime — needs WISER to ship something like micromamba.
- **What I'm unsure about:** Bundle size limits in practice; how to resolve dep conflicts with WISER's own environment.
- **Where to look:** plugin loading; project-file packaging for the `.wiserproj` precedent.
- **Labels:** feature, plugins

### Plugins off the main thread

- **Problem:** No easy way for a plugin author to run their plugin off the main thread.
- **Why it matters:** Plugins on the GUI thread freeze WISER; ease-of-use drives adoption of the right behavior.
- **What I'd do:** Give authors a simple opt-in — ideally a single boolean flag to run in another thread/process. Consider making off-thread the default.
- **What I'm unsure about:** Whether process (not thread) can be the easy default given data-transfer overhead.
- **Where to look:** plugin API; [work_scheduler.py](src/wiser/utils/work_scheduler.py).
- **Labels:** plugins, backend, ux

### Plugin chunked-operation compatibility

- **Problem:** Band-math plugins, the plugin API, and the docs all assume the whole dataset is in memory.
- **Why it matters:** The band-math chunking rewrite (HANDOFF #4) will break these plugins silently if not accounted for.
- **What I'd do:** Define how plugins receive/emit chunks; update API + docs; provide a migration path for existing plugins.
- **What I'm unsure about:** Whether some plugin operations are fundamentally whole-dataset and need an explicit "no-chunk" opt-out.
- **Where to look:** plugin API in the band-math layer; the chunking rewrite.
- **Labels:** plugins, band-math, docs

### Plugin repository handoff

- **Problem:** The WISER plugin repository needs a clear owner after my departure.
- **Why it matters:** Orphaned infra rots.
- **What I'd do:** Identify the new owner; document access, release, and maintenance steps.
- **What I'm unsure about:** Current access/permissions state.
- **Where to look:** the plugin repo.
- **Labels:** ops, handoff

---

## DevOps

### Cloud code signing

- **Problem:** Code signing is painful and local.
- **Why it matters:** Deployment friction slows every release.
- **What I'd do:** Move signing to the cloud (Matt's idea): Windows via a secure signing service; macOS on GitHub. Recursively sign everything on macOS and verify nothing is missed.
- **What I'm unsure about:** Which Windows signing service is both secure and CI-friendly.
- **Where to look:** build/deploy config; the macOS spec [WISER-macOS.spec](WISER-macOS.spec).
- **Labels:** devops, deployment, security

### Test suite speed & parallelism

- **Problem:** `test_model.py` GUI tests are slow but necessary.
- **Why it matters:** Slow CI slows everyone.
- **What I'd do:** Speed up the GUI tests; run a safe subset in parallel (a pytest flag). **First** find tests across files that read/write the same file and isolate them (unique filenames) so parallel runs don't collide.
- **What I'm unsure about:** How many tests share on-disk state — needs an audit before enabling `-n`.
- **Where to look:** `test_model.py`; shared test fixtures/temp-file usage.
- **Labels:** devops, testing

---

## Institutional / legal

### Institutional rename sweep (Caltech → LASP/CU Boulder)

- **Problem:** Build, packaging, and doc strings still name Caltech / California Institute of Technology; WISER is moving to LASP / CU Boulder. Several of these are legal identifiers, not cosmetic.
- **Why it matters:** Wrong legal entity in installers, bundle IDs, and copyright notices is a compliance problem, not a typo.
- **What I'd do:** Single tracking issue with a checklist. **Requires legal sign-off** on the correct official entity name(s) before changing the legal-bearing strings — I don't know CU Boulder's official legal name for installers/copyright, so confirm rather than guess.
  - [ ] [Makefile:16](Makefile#L16) — `OSX_BUNDLE_ID=edu.caltech.gps.WISER`
  - [ ] [WISER-macOS.spec:138](WISER-macOS.spec#L138) — `bundle_identifier='edu.caltech.gps.WISER'`
  - [ ] [conf.py:18](doc/sphinx-general-wiser-docs/source/conf.py#L18) — `copyright = "2019-2026, California Institute of Technology"` (no CU Boulder / LASP entry)
  - [ ] [install-win/win-install.nsi](install-win/win-install.nsi) — Caltech-only; confirm the legal meaning of the CU Boulder name here
  - [ ] [about_dialog.ui](src/wiser/gui/ui_files/about_dialog.ui) — multiple Caltech / California Institute of Technology legal mentions
- **What I'm unsure about:** The official CU Boulder / LASP legal entity name and what each string legally asserts (bundle ID vs. copyright holder vs. installer publisher). Do not change these without confirmation.
- **Where to look:** the five files above.
- **Labels:** legal, packaging, docs, needs-legal-review

---

## Larger bets (design doc before coding)

### WISER as an installable package

- **Problem:** No pip/conda install or programmatic API; WISER is GUI-only.
- **Why it matters:** Lets people script WISER and pull artifacts programmatically; discussed with Matthew.
- **What I'd do:** Make WISER pip/conda-installable with a real programmatic interface. Requires beefing up internals for hyperspectral data handling.
- **What I'm unsure about:** How much internal refactor the API forces before it's usable.
- **Where to look:** package structure under [src/wiser/](src/wiser/).
- **Labels:** feature, api, big-bet

### WISER Basic (pruned legacy-OS build)

- **Problem:** WISER doesn't run on older OSes (e.g. Windows 7).
- **Why it matters:** Some users are stuck on old systems and mainly need spectra viewing.
- **What I'd do:** **Preferred: option 2** — a separate, rarely-updated build with just spectra viewing plus a few features (simple, low-maintenance). Option 1 — a build process that prunes and back-ports to an older Python — is much harder.
- **What I'm unsure about:** The minimum feature set worth shipping; how much shared code can be reused vs. forked.
- **Where to look:** build/packaging; the GUI spectra-viewing components.
- **Labels:** feature, packaging, big-bet

---

## Documentation

### Feature tutorials & videos

- **Problem:** WISER is severely lacking user-facing tutorials.
- **Why it matters:** High payoff-per-hour; cheap to produce. Only real risk is that beta UI still shifts.
- **What I'd do:** Mix of single-tool tutorials and task-oriented ones ("find algae in an algae bloom"), including short videos.
- **What I'm unsure about:** How to keep them from going stale as the UI changes — maybe favor task-level over pixel-level walkthroughs.
- **Where to look:** [doc/sphinx-general-wiser-docs/](doc/sphinx-general-wiser-docs/).
- **Labels:** docs, tutorials
