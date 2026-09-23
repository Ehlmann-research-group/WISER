---
applyTo: "src/wiser/gui/**"
---

# GUI, Qt threading, and responsiveness

WISER is a PySide6 desktop application whose users open files far larger than RAM. The
recurring complaint this code has to defend against is "the app froze". Review with that
in mind.

## The main-thread rule

Qt's event loop, every `QWidget`, and every painter live on the main thread. Nothing
expensive may run there.

- Flag any pixel read, GDAL call, file I/O, network call, or NumPy operation over more
  than a thumbnail-sized array inside a slot, an event handler, a `paintEvent`, or a
  dialog's `accept()`. On a 40 GB cube these are not slow, they are an unresponsive
  application and a force-quit.
- Heavy work belongs in a `SemanticTask` / `TaskStage` pipeline submitted through
  `AppServices`, not in the widget. `src/wiser/utils/task_stage_utils.py` has the
  existing stage implementations to follow; a new dialog that does its own computation
  inline is working against the architecture and should be flagged as such.
- Widgets may only be created, mutated, or shown from the main thread. A worker that
  touches a widget — sets text, shows a message box, updates a progress bar directly —
  is a crash that reproduces on someone else's machine, not the author's. Communicate
  back through a Qt signal (queued across threads) or `QMetaObject.invokeMethod`.
- `QApplication.processEvents()` inside a long loop is not a fix for blocking; it invites
  re-entrancy, including the user triggering the same action again mid-run. Flag new
  calls. The two existing occurrences in `rasterview.py` are commented out — do not
  suggest restoring them.

## Signals and application state

- `ApplicationState` (`src/wiser/gui/app_state.py`) is the single source of truth for
  datasets, spectral libraries, ROIs, and spectra, and it broadcasts changes by signal.
  Widgets that cache state locally drift out of sync. Flag a new local copy where a
  signal subscription would do.
- Check signal/slot lifetime: a slot connected to a long-lived signal keeps the receiver
  alive, and a receiver deleted while still connected crashes on emit. `deleteLater` plus
  an unconnected signal is the usual shape of this bug.
- Check for emit storms. A signal emitted per pixel, per band, or per chunk floods the
  event loop; batching or throttling is usually needed on the data path.
- Re-entrancy: a signal handler that mutates the state that triggered the signal can
  recurse. Where a diff adds one, check for the guard.

## Rendering at scale

- The main raster display has **no pyramid**. Only the mosaic path builds internal
  overviews (`build_overviews` in `src/wiser/raster/mosaic_ingestion.py`). So a display
  change that reads at full resolution has no decimation to fall back on — it reads the
  whole band. Read only the visible viewport at the resolution actually shown.
- Zoom, pan, and band-change handlers fire continuously while the user drags. A read
  inside one must be cheap, cancellable, or debounced; check which.
- Stretch computation over all pixels is a full pass over the band. Sample instead —
  `compute_stretch_bounds` sampling from a coarse overview is the pattern the mosaic code
  already uses and the reasoning is documented there.
- Export and screenshot paths quietly materialize full-resolution arrays. Check them
  against the scale analysis in the repo-wide instructions.

## Failure that reaches the user

- When a file fails to open, a task dies, or a computation produces nothing, the user
  must be told what happened and what to do. Flag a new `except Exception` that logs and
  continues, leaving the UI showing stale or empty results as though they were valid.
  Note that ruff's config ignores `E722`, so a bare `except:` passes lint — judge it on
  its merits.
- Do not accept a `QMessageBox` raised from a worker thread as the error path; see the
  main-thread rule.
- A dialog that disables its own OK button on invalid input is better than one that
  accepts it and fails later in a worker, where the user has lost the context.

## Dialogs, plugins, and long-lived references

- Context-menu plugins receive an isolated mutable context copy including `wiser`
  (`ApplicationState`) and `app_services`. A plugin API change is a compatibility
  change — see `.github/instructions/compatibility.instructions.md`.
- A modal dialog holding a dataset reference blocks cleanup of everything that dataset
  owns. Check that dialogs release references on close.
- Files under `src/wiser/gui/generated/` are produced by `make generated` from
  `.ui`/`.qrc` sources. Never review them, and flag a diff that edits generated output
  by hand or that adds a `.ui` file without the matching `Makefile` entry — the build
  will not pick it up.
