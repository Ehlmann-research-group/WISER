"""Dependency resolver and save-policy engine (issue #617).

WISER's saveable state forms a dependency DAG rooted at datasets: dataset-backed
spectra depend on a dataset, ROI-average spectra on a dataset and an ROI, run
records on input/output datasets, stretches on a dataset.  When the user chooses
not to save a dataset, everything that transitively depends on it becomes
unsaveable.  This module is the single place that decides, per item, whether it
is reconstructed live (FAITHFUL), frozen to a self-contained snapshot (SNAPSHOT),
or dropped (DROP) -- so no persister writes a dangling reference, and the save
dialog (#626) and the writers share one rule.

The resolver centralizes the *rule*; each persister declares its own items'
dependencies (a spectrum knows it holds a dataset and maybe an ROI) and asks the
resolver to classify them.  Coupling the resolver to every concrete item class
would make it the thing that must change for each new persister; the split keeps
it stable as the epic adds item types.
"""

import enum
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Set, Tuple


class SavePolicy(enum.Enum):
    """How a saveable item is written once the resolver classifies it."""

    FAITHFUL = "faithful"  # all dependencies saved -> reconstruct the live object
    SNAPSHOT = "snapshot"  # a dependency is cut -> freeze self-contained data
    DROP = "drop"  # cannot snapshot (or user declined) -> omit and warn


@dataclass(frozen=True)
class Dependency:
    """A reference from a saveable item to something it depends on."""

    kind: str  # "dataset" | "roi"
    id: object  # the dependency's numeric id (or None if unresolved)


@dataclass
class Decision:
    """The resolver's verdict for one item."""

    policy: SavePolicy
    reason: str
    cut: List[Dependency] = field(default_factory=list)


class DependencyResolver:
    """Classify saveable items given which roots the user is saving.

    ``saved_dataset_ids`` is the set of dataset ids being written to the project
    file -- the only roots that require a user decision.  ``saved_roi_ids`` is
    accepted for symmetry but defaults to "all ROIs saved", since ROIs are
    standalone ``[SOURCE]`` state with no dependencies of their own (#619); pass
    an explicit set only once ROI selection becomes a user choice.
    """

    def __init__(
        self,
        saved_dataset_ids: Iterable[int],
        saved_roi_ids: Optional[Iterable[int]] = None,
    ):
        self._saved_datasets: Set[int] = set(saved_dataset_ids)
        self._saved_rois: Optional[Set[int]] = None if saved_roi_ids is None else set(saved_roi_ids)

    def is_saved(self, dep: Dependency) -> bool:
        """Whether a single dependency will be present in the restored session."""
        if dep.id is None:
            # A dependency whose app_state id is unresolved (e.g. a spectrum's
            # dataset/ROI was never registered) is a cut edge for every kind,
            # forcing a snapshot rather than a reference that would dangle.
            return False
        if dep.kind == "dataset":
            return dep.id in self._saved_datasets
        if dep.kind == "roi":
            return self._saved_rois is None or dep.id in self._saved_rois
        raise ValueError(f"unknown dependency kind: {dep.kind!r}")

    def classify(self, dependencies: Iterable[Dependency], *, snapshotable: bool) -> Decision:
        """Apply the governing rule to one item's dependencies.

        No cut dependency -> FAITHFUL.  A cut dependency with a snapshot fallback
        -> SNAPSHOT.  A cut dependency with no fallback -> DROP.
        """
        cut = [dep for dep in dependencies if not self.is_saved(dep)]
        if not cut:
            return Decision(SavePolicy.FAITHFUL, "all dependencies saved")
        missing = ", ".join(f"{dep.kind} {dep.id}" for dep in cut)
        if snapshotable:
            return Decision(SavePolicy.SNAPSHOT, f"cut dependency ({missing}); frozen to a snapshot", cut)
        return Decision(SavePolicy.DROP, f"cut dependency ({missing}); no snapshot available", cut)


def resolver_for_all_datasets(app_state) -> DependencyResolver:
    """Build a resolver that treats every dataset in ``app_state`` as saved.

    The default until the save dialog (#626) lets the user deselect roots.
    """
    return DependencyResolver(dataset.get_id() for dataset in app_state.get_datasets())


def cascade_report(named_decisions: Sequence[Tuple[str, Decision]]) -> List[dict]:
    """Summarize classified items for the save dialog (#626) to render.

    Takes ``(label, Decision)`` pairs and returns JSON-friendly dicts so the
    dialog can show what will be faithful / snapshotted / dropped and why.
    """
    return [
        {"item": label, "policy": decision.policy.value, "reason": decision.reason}
        for label, decision in named_decisions
    ]
