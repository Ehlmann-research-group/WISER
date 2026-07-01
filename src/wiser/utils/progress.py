"""
Qt-free progress reporting for long-running functions (mosaic ingestion #634 and
any other scheduler-submitted work).

A :class:`ProgressReporter` is handed to a worker function, which calls
``report(numerator, denominator, message)`` as it makes progress. The reporter maps
that local ``numerator / denominator`` onto its own ``[start, end]`` slice of the
overall ``0..1`` range and forwards a *global* fraction to a sink callable. Because a
multi-step job can :meth:`ProgressReporter.split` its range into weighted child
reporters, each sub-step reports in its own natural units while the overall progress
stays a single smooth 0..1 value.

This module intentionally imports **no Qt**: worker functions stay GUI-free and
unit-testable with a plain recording sink. The GUI side (``wiser/gui/progress_task``)
supplies a sink that emits a thread-safe Qt signal to drive the modal and the
Activity Monitor.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

# A sink receives the *global* progress fraction in [0.0, 1.0] and a message.
ProgressSink = Callable[[float, str], None]


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


class ProgressReporter:
    """
    Maps local ``(numerator, denominator)`` progress onto a ``[start, end]`` slice of
    a shared ``0..1`` range and forwards it to ``sink``.

    A reporter with ``sink=None`` is a safe no-op, which is the default for non-GUI
    and unit-test use so ``progress`` parameters can always be optional.
    """

    def __init__(
        self,
        sink: Optional[ProgressSink] = None,
        start: float = 0.0,
        end: float = 1.0,
        label: str = "",
    ) -> None:
        if end < start:
            raise ValueError(f"ProgressReporter end ({end}) must be >= start ({start})")
        self._sink = sink
        self._start = _clamp01(start)
        self._end = _clamp01(end)
        self._label = label

    @property
    def label(self) -> str:
        return self._label

    def report(self, numerator: float, denominator: float, message: str = "") -> None:
        """
        Report progress as ``numerator / denominator`` within this reporter's range.

        A non-positive ``denominator`` is treated as 0% (avoids division by zero for
        a not-yet-known total). The ratio is clamped to ``[0, 1]``.
        """
        fraction = 0.0 if denominator <= 0 else numerator / denominator
        self.report_fraction(fraction, message)

    def report_fraction(self, fraction: float, message: str = "") -> None:
        """
        Report progress as a local fraction in ``[0, 1]`` within this reporter's
        range. Convenient for GDAL-style ``callback(complete, ...)`` hooks.
        """
        local = _clamp01(fraction)
        global_fraction = self._start + local * (self._end - self._start)
        if self._sink is not None:
            self._sink(global_fraction, message or self._label)

    def split(self, *weights_and_labels: Tuple[float, str]) -> List["ProgressReporter"]:
        """
        Divide this reporter's range into child reporters, one per weight.

        Each child covers a slice of ``[start, end]`` proportional to its weight, in
        order, and shares this reporter's sink. Example::

            mat, ov, fp = progress.split((0.5, "Materializing"),
                                         (0.35, "Building overviews"),
                                         (0.15, "Computing footprint"))

        The child ``label`` is used as the default message when a child reports
        without an explicit one. Weights must be positive.
        """
        return self.split_weights(weights_and_labels)

    def split_weights(self, weights_and_labels: Sequence[Tuple[float, str]]) -> List["ProgressReporter"]:
        """List-taking form of :meth:`split` (kept separate so ``split`` stays varargs)."""
        weights = [float(weight) for weight, _ in weights_and_labels]
        if not weights:
            return []
        if any(weight <= 0.0 for weight in weights):
            raise ValueError("ProgressReporter.split weights must be positive")

        total = sum(weights)
        span = self._end - self._start
        children: List[ProgressReporter] = []
        cursor = self._start
        for weight, label in weights_and_labels:
            seg = span * (float(weight) / total)
            children.append(ProgressReporter(self._sink, cursor, cursor + seg, label))
            cursor += seg
        # Pin the last child's end exactly to self._end to avoid float drift leaving a
        # sliver of the range unreachable.
        if children:
            last = children[-1]
            children[-1] = ProgressReporter(self._sink, last._start, self._end, last._label)
        return children


def gdal_progress_callback(reporter: ProgressReporter) -> Callable[[float, str, object], int]:
    """
    Adapt a :class:`ProgressReporter` to GDAL's ``callback(complete, message, data)``.

    GDAL calls the returned function with ``complete`` in ``[0, 1]``; we forward it via
    :meth:`ProgressReporter.report_fraction`. Returning a truthy value tells GDAL to
    keep going (returning 0 would abort the operation).
    """

    def _callback(complete: float, message: str, _user_data: object) -> int:
        reporter.report_fraction(complete, message or reporter.label)
        return 1

    return _callback
