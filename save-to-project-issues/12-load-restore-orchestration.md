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
7. Restore **spectra** [05](05-spectra-persistence.md) and **spectral libraries** [06](06-spectral-library-persistence.md): rebuild FAITHFUL objects against restored datasets/ROIs; otherwise rehydrate snapshots (with provenance so they remain upgradeable).
8. Restore **run histories** [08](08-run-history-persistence.md): wire records to restored datasets, or place into "closed runs" when a dataset is absent.
9. Rebuild `[DERIVED]` state (e.g. `_all_spectra` index); leave caches to lazily recompute.
10. Emit the appropriate `ApplicationState` signals so the UI reflects the restored session.

**Robustness:** missing sidecar files, an unreadable file-backed dataset path, or an unknown `type` tag should degrade gracefully (warn + continue), never hard-crash the load. Produce a load report mirroring the save report (what restored faithfully / as snapshot / was dropped).

## Describe how solution fits WISER's mission

Reliable restoration is what makes a saved session actually *usable* — a researcher or student can reopen exactly where they left off, and a shared project opens correctly on a collaborator's machine. This closes the reproducibility loop at the heart of WISER's mission.

## Describe alternatives you've considered

- **Restore in manifest/declaration order.** Rejected: references may not yet exist; topological order is required.
- **Merge into the existing session.** Out of scope (epic): open always clears first, which is what makes id-preservation safe.
- **Hard-fail on any missing dependency.** Rejected: graceful degradation (snapshots, closed runs, warnings) is the whole point of the design.

## Additional context

- Depends on [01](01-project-bundle-format-and-pyrep.md), [03](03-dataset-persistence.md), all item persisters [04](04-roi-persistence.md)–[10](10-band-math-expression-persistence.md), and the versioning/migration subsystem [13](13-project-versioning-and-migration.md); pairs with the save side [11](11-save-flow-and-dialog-ux.md).
- The preserve-ids-on-fresh-load approach (no remapping) is the key invariant that keeps every restored reference valid.
