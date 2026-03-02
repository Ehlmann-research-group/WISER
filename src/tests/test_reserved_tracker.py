import unittest

import tests.context

from wiser.utils.primitives import PriorityClass
from wiser.utils.task_system import WorkUnit
from wiser.utils.work_scheduler import QueuedWorkUnit, ReservedTracker

import pytest

pytestmark = [
    pytest.mark.scheduler,
]


def _noop_work_fn() -> None:
    return None


def _make_queued_work_unit(
    *,
    unit_id: str,
    priority_class: PriorityClass,
    ram_peak_est_bytes: int = 10,
) -> QueuedWorkUnit:
    work_unit = WorkUnit(
        unit_id=unit_id,
        stage_id="s00",
        priority_class=priority_class,
        executor_kind="process",
        fn=_noop_work_fn,
        ram_peak_est_bytes=ram_peak_est_bytes,
    )
    return QueuedWorkUnit(plan_id="plan:test", stage_id="s00", work_unit=work_unit)


class TestReservedTracker(unittest.TestCase):
    def test_pop_next_admissible_returns_none_when_empty(self) -> None:
        tracker = ReservedTracker(
            interactive_reservation_window_size=3,
            render_reservation_window_size=2,
            background_reservation_window_size=1,
        )

        popped = tracker.pop_next_admissible_reserved_unit(
            in_flight_ram_bytes=0,
            scheduler_ram_cap_bytes=100,
        )

        self.assertIsNone(popped)

    def test_pop_order_with_one_unit_per_priority(self) -> None:
        tracker = ReservedTracker(
            interactive_reservation_window_size=3,
            render_reservation_window_size=2,
            background_reservation_window_size=1,
        )
        tracker.enqueue_reserved_unit(
            _make_queued_work_unit(unit_id="interactive-1", priority_class=PriorityClass.INTERACTIVE)
        )
        tracker.enqueue_reserved_unit(
            _make_queued_work_unit(unit_id="render-1", priority_class=PriorityClass.RENDER)
        )
        tracker.enqueue_reserved_unit(
            _make_queued_work_unit(unit_id="background-1", priority_class=PriorityClass.BACKGROUND)
        )

        popped_ids: list[str] = []
        for _ in range(3):
            popped = tracker.pop_next_admissible_reserved_unit(
                in_flight_ram_bytes=0,
                scheduler_ram_cap_bytes=100,
            )
            self.assertIsNotNone(popped)
            popped_ids.append(popped.work_unit.unit_id)  # type: ignore[union-attr]

        self.assertEqual(
            popped_ids,
            ["interactive-1", "render-1", "background-1"],
        )

    def test_pop_all_items_with_m2_n1_k1_keeps_expected_round_robin_order(self) -> None:
        tracker = ReservedTracker(
            interactive_reservation_window_size=2,
            render_reservation_window_size=1,
            background_reservation_window_size=1,
        )
        tracker.enqueue_reserved_unit(
            _make_queued_work_unit(unit_id="interactive-1", priority_class=PriorityClass.INTERACTIVE)
        )
        tracker.enqueue_reserved_unit(
            _make_queued_work_unit(unit_id="interactive-2", priority_class=PriorityClass.INTERACTIVE)
        )
        tracker.enqueue_reserved_unit(
            _make_queued_work_unit(unit_id="interactive-3", priority_class=PriorityClass.INTERACTIVE)
        )
        tracker.enqueue_reserved_unit(
            _make_queued_work_unit(unit_id="render-1", priority_class=PriorityClass.RENDER)
        )
        tracker.enqueue_reserved_unit(
            _make_queued_work_unit(unit_id="render-2", priority_class=PriorityClass.RENDER)
        )
        tracker.enqueue_reserved_unit(
            _make_queued_work_unit(unit_id="background-1", priority_class=PriorityClass.BACKGROUND)
        )

        popped_ids: list[str] = []
        while True:
            popped = tracker.pop_next_admissible_reserved_unit(
                in_flight_ram_bytes=0,
                scheduler_ram_cap_bytes=100,
            )
            if popped is None:
                break
            popped_ids.append(popped.work_unit.unit_id)

        self.assertEqual(
            popped_ids,
            [
                "interactive-1",
                "interactive-2",
                "render-1",
                "background-1",
                "interactive-3",
                "render-2",
            ],
        )


if __name__ == "__main__":
    unittest.main()
