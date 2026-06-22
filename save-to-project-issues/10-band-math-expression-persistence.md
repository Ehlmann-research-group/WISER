## Is your feature request related to a problem? Please describe.

The band-math **saved-expression list** is `[SOURCE]` state, but it lives in an unusual place: directly in the combo-box widget `_ui.cbox_saved_exprs` on the `BandMathDialog` ([bandmath_dialog.py](../src/wiser/gui/bandmath_dialog.py)), with `_saved_exprs_modified` tracking unsaved changes. It is **not** routed through `ApplicationState`, so a session-reconstruction routine has to reach into the dialog to capture it. Today these saved expressions are lost on close (unless the user separately exported them to a `.txt`).

## Describe the solution you'd like

Persist the saved-expression list as part of the project.

**Save:** capture the list of expression strings from the dialog's saved-expression store (the combo-box items) into the manifest — a simple list of strings. Independent of datasets and every other item.

**Load:** repopulate the saved-expression store on the `BandMathDialog` (creating/seeding it so the expressions are present whether or not the dialog has been opened yet).

**Out of scope** (`[EPHEMERAL]` / not session state): the current expression text in `_ui.ledit_expression`, variable bindings (`tbl_variables`), transient `_expr_info`, and batch-job definitions. **Old band-math runs are explicitly not tracked**; band-math *output rasters* are datasets, handled by [03](03-dataset-persistence.md).

## Describe alternatives you've considered

- **Rely on using just the existing `.txt` export/import.** Rejected: manual, separate from the session, easily forgotten.
- **Move the saved-expression list onto `ApplicationState` first.** A nice cleanup but not required; the persister can read/write the dialog's store directly. Could be a follow-up.

## Additional context

- Depends only on [01](01-project-bundle-format-and-pyrep.md).
- Note the architectural wrinkle (state on a dialog, not `ApplicationState`), shared with the georeferencer table — see [app-state.md](../doc/sphinx-general-wiser-docs/source/developer-content/app-state.md).
