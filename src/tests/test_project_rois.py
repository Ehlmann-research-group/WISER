"""Tests for the ROI persister's load path (issue #619).

The round-trip of ROI geometry is exercised through the pyrep convention in
``test_roi_storage``; here we lock the drop-not-fatal contract the load
orchestrator relies on -- a malformed ROI entry is reported, not fatal.
"""

import tests.context  # noqa: F401

from wiser.project.persisters.rois import load_rois


class _FakeAppState:
    def __init__(self):
        self.added = []

    def add_roi(self, roi):
        self.added.append(roi)


def test_load_rois_drops_malformed_entry():
    dst = _FakeAppState()
    dropped = load_rois({"rois": [{"type": "not-a-real-roi"}]}, dst)
    assert len(dropped) == 1
    assert dst.added == []


def test_load_rois_no_section_is_noop():
    dst = _FakeAppState()
    assert load_rois({}, dst) == []
    assert dst.added == []


def test_load_rois_non_list_section_is_noop():
    dst = _FakeAppState()
    assert load_rois({"rois": {"bad": "dict"}}, dst) == []
    assert dst.added == []


def test_load_rois_drops_non_dict_entry():
    dst = _FakeAppState()
    dropped = load_rois({"rois": ["not-a-dict"]}, dst)
    assert dropped == ["not-a-dict"]
    assert dst.added == []
