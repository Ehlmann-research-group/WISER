## Is your feature request related to a problem? Please describe.

Saving is only half the feature. Opening a project must **rebuild the session in the correct order** so that every reference resolves: datasets must exist (with their original ids) before stretches, spectra, and run records that reference them are restored; ROIs must exist before ROI-average spectra are made live again. Done wrong, the load produces cross-wired references, missing-dependency crashes, or silently inert state.

## Describe the solution you'd like

A **load/restore orchestrator** that opens a bundle and reconstructs the session in dependency (topological) order, on a **cleared session** (no merge — per epic scope).

**Order of operations:**
1. Open the bundle ([01](01-project-bundle-format-and-pyrep.md)); read `manifest.json`; run the **migrate-up chain** to bring an older `format_version` to the current schema, or **refuse cleanly** if the file is newer than this WISER understands (both owned by [13](13-project-versioning-and-migration.md)). After this step the rest of the load only ever sees the current schema.
2. **Clear the current session** to a clean state.
3. Restore **datasets** first, **preserving original ids**, then set `_next_id = max(id)+1` ([03](03-dataset-persistence.md)).
4. Restore **user CRSs** [09](09-user-crs-persistence.md) and **band-math expressions** [10](10-band-math-expression-persistence.md) (independent — any time).
5. Restore **ROIs** [04](04-roi-persistence.md).
6. Restore **stretches** [07](07-contrast-stretch-persistence.md), keyed onto restored datasets; drop leaves whose dataset is absent.
7. Restore **spectra** [05](05-spectra-persistence.md) and **spectral libraries** [06](06-spectral-library-persistence.md): rebuild FAITHFUL objects against restored datasets/ROIs; otherwise rehydrate frozen snapshots as `NumPyArraySpectrum` instances.
8. Restore **run histories** [08](08-run-history-persistence.md): wire records to restored datasets, or place into "closed runs" when a dataset is absent.
9. Rebuild **`_all_spectra`** — the by-id index `{spectrum_id: Spectrum}` that other code uses to look up spectra by id. It is not saved (it duplicates the active/collected collections), but it must be repopulated after step 7 so that any id-based lookup resolves to the freshly-restored objects. The three other `[DERIVED]` items require no action: `DataCache` (render/computation/histogram caches), `RasterDataSet._cached_band_stats` (per-band min/max), and each spectrum's `_icon` thumbnail all refill themselves the first time something asks for them.
10. **Emit the change signals so the UI reflects the restored session.** WISER's UI is signal-driven: `ApplicationState` emits a Qt signal on each change and the panels (layer list, spectrum plot, main view) redraw in response. Because restore loads everything programmatically into a cleared (empty-UI) session, each restored item needs its corresponding signal fired or the data sits in `ApplicationState` with nothing on screen. Wrap this in a single helper (e.g. `emit_session_reload_signals()`) that fires the **granular** per-item signals ([app_state.py:73-104](../src/wiser/gui/app_state.py#L73-L104)):
   - Datasets → `dataset_added(id, bool)` per dataset, then `mainview_dataset_changed(id)` for the one being displayed.
   - Spectral libraries → `spectral_library_added(id)`.
   - ROIs → `roi_added(RegionOfInterest)`.
   - Stretches → `stretch_changed(dataset_id, bands_tuple)`.
   - Active spectrum → `active_spectrum_changed()`; collected spectra → `collected_spectra_changed(object, int, int)`.

   Two things this step must also handle, because they do **not** go through `ApplicationState` signals:
   - **Run histories** signal through their own managers, not `ApplicationState` — fire each `RunHistoryManagerBase.records_changed` (PCA / MNF / linear-unmixing / K-Means). (The legacy `kmeans_centroids_changed` goes away with [issue #613](https://github.com/Ehlmann-research-group/WISER/issues/613).)
   - **User CRSs and band-math expressions** have no `ApplicationState` signal at all — they live on their dialogs and are refreshed by repopulating those dialogs' stores directly (steps 4).

   **Granular vs. bulk note:** we deliberately choose the granular approach — reusing the same per-item signals that `add_dataset()` and friends already emit, just gathered behind `emit_session_reload_signals()`. There is no single "whole session reloaded" signal today, and we are not adding one right now. If a bulk restore ever produces a signal flurry that is too slow or causes visible flicker, introducing one coarse "session reloaded, refresh everything" signal would be a small, contained follow-up — but granular is the default and is sufficient for now.

**Robustness:** missing sidecar files, an unreadable file-backed dataset path, or an unknown `type` tag should degrade gracefully (warn + continue), never hard-crash the load. Produce a load report mirroring the save report (what restored faithfully / as snapshot / was dropped) and log.

## Describe alternatives you've considered

- **Restore in manifest/declaration order.** Rejected: references may not yet exist; topological order is required.
- **Merge into the existing session.** Out of scope (epic): open always clears first, which is what makes id-preservation safe.
- **Hard-fail on any missing dependency.** Rejected: graceful degradation (snapshots, closed runs, warnings) is the whole point of the design.

## Additional context

- Depends on [01](01-project-bundle-format-and-pyrep.md), [03](03-dataset-persistence.md), all item persisters [04](04-roi-persistence.md)–[10](10-band-math-expression-persistence.md), and the versioning/migration subsystem [13](13-project-versioning-and-migration.md); pairs with the save side [11](11-save-flow-and-dialog-ux.md).
- The preserve-ids-on-fresh-load approach (no remapping) is the key invariant that keeps every restored reference valid.
