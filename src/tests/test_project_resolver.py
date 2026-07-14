"""Unit tests for the dependency resolver (issue #617)."""

import pytest

import tests.context  # noqa: F401

from wiser.project.resolver import (
    Decision,
    Dependency,
    DependencyResolver,
    SavePolicy,
    cascade_report,
)


def test_no_dependencies_is_faithful():
    resolver = DependencyResolver(saved_dataset_ids={1, 2})
    decision = resolver.classify([], snapshotable=True)
    assert decision.policy is SavePolicy.FAITHFUL
    assert decision.cut == []


def test_saved_dataset_dep_is_faithful():
    resolver = DependencyResolver(saved_dataset_ids={7})
    decision = resolver.classify([Dependency("dataset", 7)], snapshotable=True)
    assert decision.policy is SavePolicy.FAITHFUL


def test_cut_dataset_dep_snapshots_when_snapshotable():
    resolver = DependencyResolver(saved_dataset_ids={7})
    decision = resolver.classify([Dependency("dataset", 99)], snapshotable=True)
    assert decision.policy is SavePolicy.SNAPSHOT
    assert decision.cut == [Dependency("dataset", 99)]


def test_cut_dataset_dep_drops_when_not_snapshotable():
    resolver = DependencyResolver(saved_dataset_ids={7})
    decision = resolver.classify([Dependency("dataset", 99)], snapshotable=False)
    assert decision.policy is SavePolicy.DROP


def test_rois_are_always_saved_by_default():
    resolver = DependencyResolver(saved_dataset_ids={7})
    decision = resolver.classify([Dependency("dataset", 7), Dependency("roi", 3)], snapshotable=True)
    assert decision.policy is SavePolicy.FAITHFUL


def test_roi_can_be_constrained_explicitly():
    resolver = DependencyResolver(saved_dataset_ids={7}, saved_roi_ids={1})
    decision = resolver.classify([Dependency("roi", 3)], snapshotable=True)
    assert decision.policy is SavePolicy.SNAPSHOT


def test_none_dataset_id_is_a_cut_edge():
    resolver = DependencyResolver(saved_dataset_ids={7})
    decision = resolver.classify([Dependency("dataset", None)], snapshotable=True)
    assert decision.policy is SavePolicy.SNAPSHOT


def test_none_roi_id_is_a_cut_edge():
    # ROIs default to "all saved", but an unresolved id still forces a snapshot
    # so the load path never dereferences a missing ROI and drops it instead.
    resolver = DependencyResolver(saved_dataset_ids={7})
    decision = resolver.classify([Dependency("dataset", 7), Dependency("roi", None)], snapshotable=True)
    assert decision.policy is SavePolicy.SNAPSHOT
    assert decision.cut == [Dependency("roi", None)]


def test_unknown_dependency_kind_raises():
    resolver = DependencyResolver(saved_dataset_ids=set())
    with pytest.raises(ValueError):
        resolver.is_saved(Dependency("galaxy", 1))


def test_cascade_report_shape():
    report = cascade_report(
        [
            ("spec-a", Decision(SavePolicy.FAITHFUL, "all deps saved")),
            ("spec-b", Decision(SavePolicy.SNAPSHOT, "cut dataset")),
        ]
    )
    assert report == [
        {"item": "spec-a", "policy": "faithful", "reason": "all deps saved"},
        {"item": "spec-b", "policy": "snapshot", "reason": "cut dataset"},
    ]


def test_standalone_kinds_are_saved_by_default():
    # Runs, libraries, CRSs, and band-math expressions are saved unless the user
    # explicitly excludes them -- they are nobody else's dependency.
    resolver = DependencyResolver(saved_dataset_ids=set())
    for kind in ("run", "library", "crs", "bandmath"):
        assert resolver.is_saved(Dependency(kind, 1))


def test_excluded_standalone_items_are_not_saved():
    resolver = DependencyResolver(saved_dataset_ids=set(), excluded_items={("run", 5), ("crs", "MyCRS")})
    assert not resolver.is_saved(Dependency("run", 5))
    assert not resolver.is_saved(Dependency("crs", "MyCRS"))
    # A same-kind different id, and an un-excluded kind, are still saved.
    assert resolver.is_saved(Dependency("run", 6))
    assert resolver.is_saved(Dependency("library", 5))
