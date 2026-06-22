## Is your feature request related to a problem? Please describe.

A project file format that is saved by users *will* change as WISER evolves — new state items, renamed fields, restructured records. Without an explicit, enforced versioning strategy, one of two failures is inevitable:

- **Silent rot:** a schema change quietly breaks the ability to open older project files, and nobody notices until a user reports a corrupted/unopenable project.
- **Accreting cruft:** each change adds ad-hoc branching to the load path until it is unmaintainable.

The bundle format ([01](01-project-bundle-format-and-pyrep.md)) provides a `format_version` field and a migration *seam*, and the load orchestrator ([12](12-load-restore-orchestration.md)) checks the version — but neither commits to a **policy** or, crucially, **tests that old files still load**. This issue owns that policy and its enforcement.

## Describe the solution you'd like

A **versioning & migration subsystem** with a guarantee that is actually tested.

**Guarantee:** *Any project file written by a released WISER can be opened by every later WISER.* (Backward compatibility.) Forward compatibility is explicitly **not** promised.

**Components:**

1. **Migrate-up chain.** One loader understands only the **current** schema. On load, the `format_version` is read and a chain of small *pure* functions (`migrate_v1_to_v2`, `migrate_v2_to_v3`, …) transforms the manifest pyrep dict step-by-step up to the current version before the normal load runs. Sub-issues 03–12 never see old shapes. Each migration is independently unit-tested with a minimal before/after dict.
2. **Refuse-too-new behavior.** If a file's `format_version` is **greater** than the running WISER understands, abort the load with a clear, user-facing message ("This project was created with a newer version of WISER. Please upgrade to open it.") — never a crash, partial load, or silent data loss.
3. **Additive-change convention (documented).** Non-breaking additions (new *optional* fields) require **no** version bump and **no** migration: `from_pyrep` ignores unknown keys and defaults missing ones. The `format_version` is bumped (and a migration written) **only** for breaking changes — renamed, removed, restructured, or semantically-changed fields. Document this rule where contributors will see it.
4. **Golden-file regression suite (the enforcement).** Check in a real, representative project bundle for **every historical `format_version`** under test fixtures. A test loads each on the current code and asserts a successful, correct restore. This is what converts the guarantee from aspiration into something CI fails on. Add a checklist/CI reminder so that whenever `format_version` is bumped, a new golden fixture for the previous version is captured.
5. **A "too-new" fixture** verifying the refuse-cleanly path.

## Describe how solution fits WISER's mission

Reproducibility is meaningless if a project saved last year cannot be opened today. A tested backward-compatibility guarantee means researchers, educators, and students can trust that their saved analyses — and the projects their collaborators share — will keep opening across WISER releases. This durability directly serves the mission of accessible, reproducible imaging spectroscopy.

## Describe alternatives you've considered

- **Versioned readers (keep a parser per old version forever).** Rejected: the load path accumulates branching cruft and every old reader must be maintained; migrate-up keeps a single current-schema loader.
- **No migrations — "just don't make breaking changes."** Rejected: unrealistic over the life of the format; breaking changes will happen and must be handled deliberately.
- **Per-section version numbers from the start.** Deferred: a single global `format_version` is simpler; per-section versioning can be introduced later behind the same migration seam if one item churns heavily.
- **Promising forward compatibility.** Rejected as infeasible: old code cannot understand features added later; refusing cleanly is the correct behavior.

## Additional context

- Depends on the `format_version` field + migration seam from [01](01-project-bundle-format-and-pyrep.md) and is invoked by the load orchestrator [12](12-load-restore-orchestration.md).
- Cross-cutting and **ongoing**: every future schema change in any item persister (03–10) must respect the additive-vs-breaking convention and, on a version bump, add a golden fixture here.
- Pattern reference: database schema migrations and the QGIS project-file versioning approach.
