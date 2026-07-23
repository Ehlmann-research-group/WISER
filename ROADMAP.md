# WISER Roadmap

This is the forward-looking companion to [HANDOFF.md](HANDOFF.md). HANDOFF records *why the code is the way it is* and what I'd redo; this document is *what to build next* and how I'd prioritize it.

The taxonomy below is the spine — every area of WISER work has a home here. Leaves link to concrete, filable drafts in [ISSUE_DRAFTS.md](ISSUE_DRAFTS.md) where one exists.

---

## Where WISER needs the most help right now

My honest read on leaving: WISER is at a genuinely good point, and the three areas that would move it furthest are:

1. **WISER QoL + Backend — get remaining work off the GUI thread.** Not just a cleanup: it needs a *documented, standardized pattern* so new code always does the right thing. See HANDOFF entry #1. Draft: [Standardize the off-GUI-thread pattern](ISSUE_DRAFTS.md#standardize-the-off-gui-thread-pattern). Known offenders: [dataset loading](ISSUE_DRAFTS.md#async-dataset-loading), [ROI average-spectra](ISSUE_DRAFTS.md#roi-average-spectra-off-the-gui-thread), stretch builder (see the stretch builder draft below).
2. **Documentation & tutorials on WISER features.** Severely lacking. Cheap to make; the only real risk is that the UI still shifts in beta, but the payoff-per-hour is high.
3. **Backend correctness/simplicity** — the rewrites in HANDOFF Tier 2 (band math chunking, task system, `AsyncTransformer` removal).

### The one big bet: LoD-pyramid rendering for RasterView

Called out separately because it's the highest-*ceiling* item and many things would depend on it. Today `RasterView` holds the displayed image at roughly O(H×W) RAM. `MosaicView` already renders via a **level-of-detail (LoD) pyramid**, which is O(1) in RAM. Making `RasterView` use MosaicView's rendering could take image memory from **O(H×W) → O(1)** — this is what would make WISER a genuinely respectable hyperspectral tool.

It's a dependency node because two other items hang off it:
- The **dataset read cache** (HANDOFF #6) may become unnecessary if LoD makes data access cheap enough.
- The **stretch builder** freeze is worst on huge images precisely because of the O(H×W) render path.

Rank this by *ceiling*, not by regression risk — nothing here regresses if forgotten, but nothing else raises the ceiling as much. Draft: [LoD-pyramid rendering](ISSUE_DRAFTS.md#lod-pyramid-rendering-for-rasterview).

---

## Taxonomy

### DevOps

- **CI** — keep green; gate on the test tiers below.
- **Automated testing**
  - Unit tests.
  - Integration tests — drive the GUI with PySide in-process.
  - E2E tests — same, end to end.
  - **Speed & parallelism** — the `test_model.py` GUI tests are slow but necessary. Speeding them up, and running a safe subset in parallel (a pytest flag), is high-value. Caveat: some tests across different files read/write the **same file** — parallelizing requires isolating those writes (unique filenames) first. Draft: [Test suite speed & parallelism](ISSUE_DRAFTS.md#test-suite-speed--parallelism).
  - Formatting, linting.
- **Deployment**
  - Code signing macOS — recursively sign *everything* and verify nothing is missed.
  - Code signing Windows.
  - Installers for Windows / Linux (macOS bundle exists).
  - Reproducible per-platform build environments.
  - Re-bundle modules PyInstaller prunes; fix libraries incompatible with PyInstaller.
  - **Cloud code signing** — Matt's idea, and a good one: move signing to the cloud. Windows signing via a secure signing service; macOS signing on GitHub (reasonably secure). Would make deployment dramatically less painful. Draft: [Cloud code signing](ISSUE_DRAFTS.md#cloud-code-signing).

### Open source

- Automated DCO checks.
- Licensing.
- `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `GOVERNANCE.md`, and other contributor resources (some already exist at repo root).
- **Institutional rename (Caltech → LASP / CU Boulder)** — legal-tinged strings across build/packaging/docs still say Caltech. Needs legal sign-off, not just find-and-replace. Draft: [Institutional rename sweep](ISSUE_DRAFTS.md#institutional-rename-sweep-caltech--laspcu-boulder).

### Hyperspectral / GIS feature development

- Core analytics: PCA, MNF, MTMF, Georeferencer, Mosaic, etc.
- **File types** — we read TIFF and BSQ-interleaved `.hdr` fast; JP2 and netCDF slow. Speeding up that I/O speeds up *all* of WISER for those formats. See HANDOFF #2 (the GDAL multithread landmine) before touching the read path. Draft: [Speed up JP2 / netCDF I/O](ISSUE_DRAFTS.md#speed-up-jp2--netcdf-io).
- **Georeferencer errors as message boxes** — e.g. mapping between CRSs that have no transform should surface a dialog, not fail silently. Draft: [Georeferencer errors as message boxes](ISSUE_DRAFTS.md#georeferencer-errors-as-message-boxes).

### WISER QoL feature development

- **Get things off the GUI thread** — the top priority; see the priorities section and HANDOFF #1. Draft: [Standardize the off-GUI-thread pattern](ISSUE_DRAFTS.md#standardize-the-off-gui-thread-pattern).
- Save to Project File; help buttons; easier ways to save artifacts; easier ways to edit datasets. Draft: [QoL grab-bag](ISSUE_DRAFTS.md#qol-grab-bag-save-to-project-help-buttons-easier-saveedit).
- **Batch processing** — Matt's framing: batch processing was meant to be a *programmatic* way to drive WISER. It drifted into band-math-only. Revisit as a real plugin: wrap current analysis tools in a data structure that a batch back end + GUI can drive. Extension: let one batch step's output feed the next (needs rudimentary typing — is the output a cube, a band, or a spectrum?), and allow a band-math expression as a step. Draft: [Batch processing plugin](ISSUE_DRAFTS.md#batch-processing-plugin).
- **Native dark/light mode** — WISER can end up in OS "dark mode" while its icons stay dark and invisible. Detect OS theme and swap icons to light/dark accordingly. Draft: [Native dark/light mode](ISSUE_DRAFTS.md#native-dark--light-mode).
- **Stretch builder responsiveness** — stretch changes can freeze WISER for *minutes* on huge images (Gale HiRISE). It updates the context pane, zoom pane, and main view — the meat of user interaction — so a blocking modal with progress is defensible, but the operation must be **cancellable** so users can reach background analysis tools. Draft: [Stretch builder cancellable + progress](ISSUE_DRAFTS.md#stretch-builder-cancellable--progress). Ultimately subsumed by LoD rendering.

### WISER backend

- Get things off the GUI thread (shared with QoL above). Known offenders: [async dataset loading](ISSUE_DRAFTS.md#async-dataset-loading), [ROI average-spectra](ISSUE_DRAFTS.md#roi-average-spectra-off-the-gui-thread).
- Speed up calculations.
- Rewrites from HANDOFF Tier 2: [band math chunking](ISSUE_DRAFTS.md#rewrite-band-math-chunking--remove-asynctransformer), [task system](ISSUE_DRAFTS.md#simplify-the-task-system), [dataset cache re-look](ISSUE_DRAFTS.md#re-look-the-dataset-read-cache).
- Unify all async code under one system (`work_scheduler`) with documented ways to use both subprocessing and threading:
  - Subprocessing, two modes: (a) the **task system** for long, expensive work; (b) **one-off** subprocess wrapping for easy/short work.
  - Threading — currently for I/O-heavy work; could also carry long CPU-bound work at the cost of taking resources from the main thread (sometimes a worthwhile trade).
- **Scheduler "smart" logic** — the `TaskPlanner` splits work into a fixed number of pieces rather than reacting to system state. Reacting is genuinely hard (work enqueued under one system state may run under another); don't over-invest. Draft: [Scheduler system-aware planning](ISSUE_DRAFTS.md#scheduler-system-aware-planning).
- **Granular task progress** — no clean way today to report fine-grained progress of the finishing `SemanticTask`. May not have a tidy answer. Draft: [Granular task progress](ISSUE_DRAFTS.md#granular-task-progress-reporting).

### Plugins

- **Packaged plugin files** — a `.wiserproj`-style bundle but for plugins, so people can share and actually use them. Non-trivial when a plugin has dependencies outside WISER's. Two options: (1) package all deps into the bundle — simpler, larger files; (2) ship a manifest WISER reloads at runtime — smaller files, but WISER must bundle something like micromamba. **I prefer option 1.** Draft: [Packaged plugin files](ISSUE_DRAFTS.md#packaged-plugin-files).
- **Plugins off the main thread** — give users an easy opt-in (ideally a single boolean flag, maybe the default) to run a plugin in another thread/process. Draft: [Plugins off the main thread](ISSUE_DRAFTS.md#plugins-off-the-main-thread).
- **Chunked-operation compatibility** — band-math plugins, the API, and the docs all assume the full dataset is in memory. The band-math chunking rewrite must account for this or it breaks plugins silently. See HANDOFF #4. Draft: [Plugin chunked-op compatibility](ISSUE_DRAFTS.md#plugin-chunked-operation-compatibility).
- **Plugin repository handoff** — the WISER plugin repo needs an owner. Draft: [Plugin repository handoff](ISSUE_DRAFTS.md#plugin-repository-handoff).

### Documentation & tutorials

- Docs on WISER *internals* (this file, HANDOFF.md, and deeper architecture docs).
- Docs/tutorials on WISER *features* — the big gap. Mix of single-tool tutorials and task-oriented ones ("find algae in an algae bloom"). Videos included. Draft: [Feature tutorials & videos](ISSUE_DRAFTS.md#feature-tutorials--videos).

---

## Larger bets (own design docs before coding)

- **LoD-pyramid rendering for RasterView** — see the callout above. The highest-ceiling item.
- **WISER as an installable package with a programmatic API** — make WISER pip/conda-installable with a real programmatic interface, so people can script it or pull artifacts out programmatically. Requires beefing up the internals for handling hyperspectral data. Discussed with Matthew. Draft: [WISER as a package](ISSUE_DRAFTS.md#wiser-as-an-installable-package).
- **WISER Basic** — a pruned WISER that runs on older OSes (e.g. Windows 7). Two paths: (1) a build process that prunes and back-ports to an older Python — hard; (2) a **separate, rarely-updated** build that mostly just views spectra plus a few features — simpler and far less error-prone. **I prefer option 2.** Draft: [WISER Basic](ISSUE_DRAFTS.md#wiser-basic-pruned-legacy-os-build).
