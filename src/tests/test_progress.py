"""
Unit tests for the Qt-free progress core (``wiser.utils.progress``).

These exercise sub-range mapping, weighted splitting, clamping, and the no-op sink —
all without Qt or GDAL, using a plain recording sink.
"""

import pytest

import tests.context  # noqa: F401  (adds src/ to sys.path)
from wiser.utils.progress import ProgressCancelled, ProgressReporter, gdal_progress_callback

pytestmark = [
    pytest.mark.unit,
    pytest.mark.smoke,
]


class _Recorder:
    """Records (fraction, message) pairs a sink receives."""

    def __init__(self):
        self.calls = []

    def __call__(self, fraction, message):
        self.calls.append((fraction, message))

    @property
    def fractions(self):
        return [fraction for fraction, _ in self.calls]


def test_report_maps_num_den_to_fraction():
    rec = _Recorder()
    reporter = ProgressReporter(sink=rec)
    reporter.report(1, 4)
    reporter.report(4, 4)
    assert rec.fractions == [pytest.approx(0.25), pytest.approx(1.0)]


def test_report_uses_reporter_range():
    rec = _Recorder()
    # A child covering [0.6, 1.0]: local 0.5 -> global 0.8.
    reporter = ProgressReporter(sink=rec, start=0.6, end=1.0)
    reporter.report(1, 2)
    assert rec.fractions == [pytest.approx(0.8)]


def test_report_clamps_and_handles_zero_denominator():
    rec = _Recorder()
    reporter = ProgressReporter(sink=rec)
    reporter.report(5, 0)  # zero denominator -> 0%
    reporter.report(10, 4)  # >1 ratio -> clamped to 1.0
    reporter.report(-3, 4)  # negative -> clamped to 0.0
    assert rec.fractions == [pytest.approx(0.0), pytest.approx(1.0), pytest.approx(0.0)]


def test_report_fraction_passes_message_and_label():
    rec = _Recorder()
    reporter = ProgressReporter(sink=rec, label="Working")
    reporter.report_fraction(0.5, "explicit")
    reporter.report_fraction(0.5)  # falls back to label
    assert rec.calls[0][1] == "explicit"
    assert rec.calls[1][1] == "Working"


def test_split_covers_range_proportionally():
    rec = _Recorder()
    parent = ProgressReporter(sink=rec)
    mat, ov, fp = parent.split((0.5, "mat"), (0.35, "ov"), (0.15, "fp"))

    # Each child, at its own 100%, must land at the top of its slice.
    mat.report(1, 1)
    ov.report(1, 1)
    fp.report(1, 1)
    assert rec.fractions == [pytest.approx(0.5), pytest.approx(0.85), pytest.approx(1.0)]

    # And each child's start is contiguous with the previous child's end.
    rec.calls.clear()
    mat.report(0, 1)
    ov.report(0, 1)
    fp.report(0, 1)
    assert rec.fractions == [pytest.approx(0.0), pytest.approx(0.5), pytest.approx(0.85)]


def test_split_last_child_pinned_to_end():
    rec = _Recorder()
    # Weights that don't divide the range cleanly still reach exactly 1.0 at the end.
    parent = ProgressReporter(sink=rec)
    children = parent.split((1.0, "a"), (1.0, "b"), (1.0, "c"))
    children[-1].report(1, 1)
    assert rec.fractions[-1] == pytest.approx(1.0)


def test_split_rejects_nonpositive_weights():
    parent = ProgressReporter()
    with pytest.raises(ValueError):
        parent.split((1.0, "a"), (0.0, "b"))


def test_none_sink_is_noop():
    reporter = ProgressReporter()  # sink=None
    # Must not raise and must be usable like any reporter.
    reporter.report(1, 2)
    child = reporter.split((1.0, "x"))[0]
    child.report_fraction(0.5)


def test_is_cancelled_defaults_false_and_never_raises():
    # Without an is_cancelled check, a reporter is never cancelled.
    reporter = ProgressReporter()
    assert reporter.is_cancelled() is False
    reporter.raise_if_cancelled()  # must not raise


def test_raise_if_cancelled_raises_once_requested():
    flag = {"stop": False}
    reporter = ProgressReporter(is_cancelled=lambda: flag["stop"])
    reporter.raise_if_cancelled()  # not yet requested -> no raise
    flag["stop"] = True
    assert reporter.is_cancelled() is True
    with pytest.raises(ProgressCancelled):
        reporter.raise_if_cancelled()


def test_split_children_share_cancel_check():
    flag = {"stop": False}
    parent = ProgressReporter(is_cancelled=lambda: flag["stop"])
    first, last = parent.split((0.5, "a"), (0.5, "b"))
    flag["stop"] = True
    # Both the proportional children and the end-pinned last child inherit the check.
    for child in (first, last):
        assert child.is_cancelled() is True
        with pytest.raises(ProgressCancelled):
            child.raise_if_cancelled()


def test_gdal_callback_forwards_fraction_and_continues():
    rec = _Recorder()
    reporter = ProgressReporter(sink=rec, start=0.0, end=1.0, label="ov")
    callback = gdal_progress_callback(reporter)

    result = callback(0.5, "", None)
    assert result == 1  # truthy -> GDAL keeps going
    assert rec.calls[-1] == (pytest.approx(0.5), "ov")

    callback(1.0, "custom", None)
    assert rec.calls[-1] == (pytest.approx(1.0), "custom")
