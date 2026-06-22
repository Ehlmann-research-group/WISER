## Is your feature request related to a problem? Please describe.

Even with every item persister built, the user needs to **understand and control** what gets saved — especially which in-memory datasets (and in-memory libraries) to include, since those are the only roots that require a decision and everything else cascades from them. Without a clear UI, users would unknowingly drop dataset-backed spectra, ROI-average spectra, or runs when they skip a dataset, and would not understand why something "didn't save."

## Describe the solution you'd like

A **dependency-aware Save dialog** plus the top-level save flow, driven by the resolver ([02](02-dependency-resolver-and-policy.md)).

**Mental model presented to the user:** "You're really only deciding which RAM-backed datasets and in-memory libraries to include. Everything else follows automatically."

**Dialog behavior:**
1. List the **savable roots** (RAM-backed datasets, in-memory libraries) with checkboxes. File-backed datasets/libraries are free and need no decision (shown as auto-included).
2. When a root is unchecked, **live-cascade** the consequence: show, per dependent item, whether it will be **FAITHFUL**, **SNAPSHOT** (with a one-line "frozen, can be re-linked later" note), or **DROP** (with a warning).
3. For DROP-eligible items, offer the **snapshot escape hatch** ("freeze this spectrum so it survives without its dataset") where snapshotting is possible.
4. Offer **"promote to file"** for a RAM dataset (write to an external path and reference it) as an optional alternative to a bundle sidecar.
5. Summarize warnings (what will be dropped) before the user confirms.

**Save flow:**
- Orchestrate writing the bundle: manifest (pyrep) + dataset sidecars + array sidecars, in dependency order.
- Choose bundle directory vs. zipped `.wiserproj`.
- Wire up menu actions (Save Project / Save Project As) and remember the last project path.

## Describe how solution fits WISER's mission

Accessibility means users — including students and non-programmers — can confidently save their work and understand exactly what is preserved. A transparent, explainable save experience that never silently loses data embodies the mission's goals of usability and reproducibility.

## Describe alternatives you've considered

- **Save everything with no dialog.** Rejected: users can't control bundle size or understand drops; large RAM cubes would always be written.
- **Present the entire dependency tree as individually-toggleable nodes.** Rejected: overwhelming; the root-checkbox + automatic-cascade model matches the DAG (datasets are the only real decision points).
- **Block saving until all dependencies are saveable.** Rejected: too rigid; snapshot/drop with warnings is friendlier.

## Additional context

- Depends on the resolver/policy [02](02-dependency-resolver-and-policy.md), datasets [03](03-dataset-persistence.md), and all item persisters [04](04-roi-persistence.md)–[10](10-band-math-expression-persistence.md) (it orchestrates their save side).
- Pairs with load/restore [12](12-load-restore-orchestration.md).
