"""Unit, functional, and integration tests for the task planner
in the task management and execution system.
"""
import unittest
from functools import partial

import numpy as np

import tests.context
from wiser.utils.primitives import (
    AllocationRequest,
    DataBinding,
    DataRef,
    DatasetRegionRef,
    SpatialTileScheme,
)
from wiser.utils.task_system import (
    AlgorithmPipeline,
    DatasetPlanMeta,
    MapStage,
    PlanningContext,
    ResourceModel,
    SequentialStage,
    SemanticTask,
    SimpleChunkingPolicy,
    TaskPlanner,
)

import pytest

pytestmark = [
    pytest.mark.task_manager,
]


class _NoopSchedulerConfig:
    pass


class _RecordingStorage:
    def __init__(self):
        self.requests = []

    def allocate_data(self, req: AllocationRequest) -> DataRef:
        self.requests.append(req)
        return DataRef(
            kind=req.kind,
            ref_id=f"out-{req.name}",
            uri=f"mem://{req.name}",
            disk_format=None,
            shape=req.shape,
            dtype=req.dtype,
            chunks=req.chunks,
            residency=req.residency,
            materialization_loc="ram",
            source="allocated",
            readonly=False,
        )


class _IdentityMapStage(MapStage):
    def output_region_for(self, input_region: DatasetRegionRef) -> DatasetRegionRef:
        return input_region

    def generate_allocation_requests(self, *, input_meta, chosen_scheme):
        _ = chosen_scheme
        return [
            AllocationRequest(
                name="stage_out",
                kind="dataset",
                residency="ram_cacheable",
                size_est=input_meta.height * input_meta.width * input_meta.bands * 4,
                shape=input_meta.shape,
                dtype=input_meta.dtype,
                chunks=None,
            )
        ]

    def task_fn(self, input_ref, input_region, output_writes, broadcast_inputs=None):
        _ = (input_ref, input_region, output_writes, broadcast_inputs)
        return None


class _IdentitySequentialStage(SequentialStage):
    def output_region_for(self, input_region: DatasetRegionRef) -> DatasetRegionRef:
        return input_region

    def generate_allocation_requests(self, *, input_meta, chosen_scheme):
        _ = chosen_scheme
        return [
            AllocationRequest(
                name="stage_out_seq",
                kind="dataset",
                residency="ram_cacheable",
                size_est=input_meta.height * input_meta.width * input_meta.bands * 4,
                shape=input_meta.shape,
                dtype=input_meta.dtype,
                chunks=None,
            )
        ]

    def task_fn(self, input_ref, input_region, output_writes, broadcast_inputs=None):
        _ = (input_ref, input_region, output_writes, broadcast_inputs)
        return None


_RECORDED_POST_TASK_CALLS = []
_RECORDED_PRE_TASK_CALLS = []


def _record_post_task_call(
    calls,
    input_ref,
    full_input_region,
    output_writes,
    broadcast_inputs,
):
    calls.append(
        {
            "input_ref": input_ref,
            "full_input_region": full_input_region,
            "output_writes": output_writes,
            "broadcast_inputs": broadcast_inputs,
        }
    )


def _record_pre_task_call(
    calls,
    input_ref,
    full_input_region,
    output_writes,
    broadcast_inputs,
):
    calls.append(
        {
            "input_ref": input_ref,
            "full_input_region": full_input_region,
            "output_writes": output_writes,
            "broadcast_inputs": broadcast_inputs,
        }
    )


class _PostTaskRecordingStage(_IdentityMapStage):
    def post_task_fn(self, input_ref, full_input_region, output_writes, broadcast_inputs=None):
        return partial(
            _record_post_task_call,
            _RECORDED_POST_TASK_CALLS,
            input_ref,
            full_input_region,
            output_writes,
            broadcast_inputs,
        )


class _PreTaskRecordingStage(_IdentityMapStage):
    def pre_task_fn(self, input_ref, full_input_region, output_writes, broadcast_inputs=None):
        return partial(
            _record_pre_task_call,
            _RECORDED_PRE_TASK_CALLS,
            input_ref,
            full_input_region,
            output_writes,
            broadcast_inputs,
        )


class TestTaskPlanner(unittest.TestCase):
    def setUp(self):
        _RECORDED_POST_TASK_CALLS.clear()
        _RECORDED_PRE_TASK_CALLS.clear()

    def test_plan_semantic_task(self):
        input_ref = DataRef(
            kind="dataset",
            ref_id="input-1",
            uri="mem://input-1",
            disk_format=None,
            shape=(6, 9, 3),
            dtype=np.dtype(np.float32),
            chunks=None,
            residency="ram_cacheable",
            materialization_loc="ram",
            source="allocated",
            readonly=False,
        )
        input_meta = DatasetPlanMeta(
            kind="dataset",
            dtype=np.dtype(np.float32),
            shape=input_ref.shape,
        )

        stage = _IdentityMapStage(
            default_executor="thread",
            input_plan_meta=input_meta,
            resource_model=ResourceModel(
                fixed_overhead_bytes=0,
                bytes_per_scalar_in=1,
                bytes_per_scalar_out=1,
                scratch_bytes_per_scalar_in=0,
            ),
            chunking_scheme_type=SpatialTileScheme,
            output_bindings=(DataBinding("stage_out"),),
        )

        semantic_task = SemanticTask(
            priority_class="interactive",
            input_ref=input_ref,
            algorithm_pipeline=AlgorithmPipeline(stages=[stage]),
        )
        semantic_task.id = 42

        storage = _RecordingStorage()
        ctx = PlanningContext(
            sched_cfg=_NoopSchedulerConfig(),
            storage=storage,
            chunking_policy=SimpleChunkingPolicy(),
        )
        task_planner = TaskPlanner(ctx)
        task_plan = task_planner.plan_semantic_task(semantic_task)

        # SpatialTileScheme should use height/3 and width/3 -> 2x3 tiles over 6x9 => 9 chunk units,
        # plus one pre-task unit and one post-task unit.
        self.assertEqual(len(task_plan.work_units), 11)
        self.assertIn("s00", task_plan.stage_work_units)
        self.assertEqual(len(task_plan.stage_work_units["s00"]), 11)
        self.assertIn("s00", task_plan.stage_steps)
        self.assertEqual(len(task_plan.stage_steps["s00"]), 3)
        self.assertEqual(task_plan.stage_steps["s00"][0], [task_plan.stage_work_units["s00"][0]])
        self.assertEqual(task_plan.stage_steps["s00"][1], task_plan.stage_work_units["s00"][1:-1])
        self.assertEqual(task_plan.stage_steps["s00"][2], [task_plan.stage_work_units["s00"][-1]])

        # Verify one output allocation was requested and shape matches full dataset.
        self.assertEqual(len(storage.requests), 1)
        alloc = storage.requests[0]
        self.assertEqual(alloc.name, "stage_out")
        self.assertEqual(alloc.shape, (6, 9, 3))

        # Verify chunk work units' metadata writes to the same region as their input region.
        for unit_id in task_plan.stage_work_units["s00"][1:-1]:
            unit_meta = task_plan.work_units_meta[unit_id]
            self.assertIsInstance(unit_meta.input_region, DatasetRegionRef)
            self.assertEqual(len(unit_meta.output_writes), 1)
            write = unit_meta.output_writes["stage_out"]
            self.assertEqual(write.region, unit_meta.input_region)

        pre_unit_id = task_plan.stage_work_units["s00"][0]
        pre_unit_meta = task_plan.work_units_meta[pre_unit_id]
        self.assertEqual(pre_unit_meta.input_region, DatasetRegionRef(0, 6, 0, 9, 0, 3))

        post_unit_id = task_plan.stage_work_units["s00"][-1]
        post_unit_meta = task_plan.work_units_meta[post_unit_id]
        self.assertEqual(post_unit_meta.input_region, DatasetRegionRef(0, 6, 0, 9, 0, 3))
        self.assertEqual(post_unit_meta.output_writes["stage_out"].region, DatasetRegionRef(0, 6, 0, 9, 0, 3))

    def test_plan_semantic_task_with_sequential_stage_creates_singleton_steps(self):
        input_ref = DataRef(
            kind="dataset",
            ref_id="input-seq-1",
            uri="mem://input-seq-1",
            disk_format=None,
            shape=(6, 9, 3),
            dtype=np.dtype(np.float32),
            chunks=None,
            residency="ram_cacheable",
            materialization_loc="ram",
            source="allocated",
            readonly=False,
        )
        input_meta = DatasetPlanMeta(
            kind="dataset",
            dtype=np.dtype(np.float32),
            shape=input_ref.shape,
        )

        stage = _IdentitySequentialStage(
            default_executor="thread",
            input_plan_meta=input_meta,
            resource_model=ResourceModel(
                fixed_overhead_bytes=0,
                bytes_per_scalar_in=1,
                bytes_per_scalar_out=1,
                scratch_bytes_per_scalar_in=0,
            ),
            chunking_scheme_type=SpatialTileScheme,
            output_bindings=(DataBinding("stage_out_seq"),),
        )

        semantic_task = SemanticTask(
            priority_class="interactive",
            input_ref=input_ref,
            algorithm_pipeline=AlgorithmPipeline(stages=[stage]),
        )
        semantic_task.id = 43

        storage = _RecordingStorage()
        ctx = PlanningContext(
            sched_cfg=_NoopSchedulerConfig(),
            storage=storage,
            chunking_policy=SimpleChunkingPolicy(),
        )
        task_planner = TaskPlanner(ctx)
        task_plan = task_planner.plan_semantic_task(semantic_task)

        self.assertEqual(stage.work_unit_dependency, "sequential")
        self.assertIn("s00", task_plan.stage_steps)
        stage_steps = task_plan.stage_steps["s00"]
        self.assertEqual(len(stage_steps), len(task_plan.stage_work_units["s00"]))
        for step in stage_steps:
            self.assertEqual(len(step), 1)

    def test_plan_semantic_task_adds_pre_task_work_unit_with_full_input_region(self):
        input_ref = DataRef(
            kind="dataset",
            ref_id="input-pre-1",
            uri="mem://input-pre-1",
            disk_format=None,
            shape=(6, 9, 3),
            dtype=np.dtype(np.float32),
            chunks=None,
            residency="ram_cacheable",
            materialization_loc="ram",
            source="allocated",
            readonly=False,
        )
        input_meta = DatasetPlanMeta(
            kind="dataset",
            dtype=np.dtype(np.float32),
            shape=input_ref.shape,
        )

        stage = _PreTaskRecordingStage(
            default_executor="thread",
            input_plan_meta=input_meta,
            resource_model=ResourceModel(
                fixed_overhead_bytes=0,
                bytes_per_scalar_in=1,
                bytes_per_scalar_out=1,
                scratch_bytes_per_scalar_in=0,
            ),
            chunking_scheme_type=SpatialTileScheme,
            output_bindings=(DataBinding("stage_out"),),
            broadcast_input={"constant": 7},
        )

        semantic_task = SemanticTask(
            priority_class="interactive",
            input_ref=input_ref,
            algorithm_pipeline=AlgorithmPipeline(stages=[stage]),
        )
        semantic_task.id = 441

        storage = _RecordingStorage()
        ctx = PlanningContext(
            sched_cfg=_NoopSchedulerConfig(),
            storage=storage,
            chunking_policy=SimpleChunkingPolicy(),
        )
        task_plan = TaskPlanner(ctx).plan_semantic_task(semantic_task)

        pre_unit_id = task_plan.stage_work_units["s00"][0]
        pre_unit = task_plan.work_units[pre_unit_id]
        pre_meta = task_plan.work_units_meta[pre_unit_id]

        self.assertEqual(pre_meta.input_ref, input_ref)
        self.assertEqual(pre_meta.input_region, DatasetRegionRef(0, 6, 0, 9, 0, 3))
        self.assertEqual(pre_meta.broadcast_inputs, {"constant": 7})
        self.assertEqual(pre_meta.output_writes["stage_out"].region, DatasetRegionRef(0, 6, 0, 9, 0, 3))
        self.assertEqual(pre_unit.deps, ())

        chunk_unit_id = task_plan.stage_work_units["s00"][1]
        self.assertEqual(task_plan.work_units[chunk_unit_id].deps, (pre_unit_id,))

        pre_unit.fn()

        self.assertEqual(len(_RECORDED_PRE_TASK_CALLS), 1)
        recorded = _RECORDED_PRE_TASK_CALLS[0]
        self.assertEqual(recorded["input_ref"], input_ref)
        self.assertEqual(recorded["full_input_region"], DatasetRegionRef(0, 6, 0, 9, 0, 3))
        self.assertEqual(recorded["broadcast_inputs"], {"constant": 7})
        self.assertEqual(
            recorded["output_writes"]["stage_out"].region,
            DatasetRegionRef(0, 6, 0, 9, 0, 3),
        )

    def test_plan_semantic_task_adds_post_task_work_unit_with_full_input_region(self):
        input_ref = DataRef(
            kind="dataset",
            ref_id="input-post-1",
            uri="mem://input-post-1",
            disk_format=None,
            shape=(6, 9, 3),
            dtype=np.dtype(np.float32),
            chunks=None,
            residency="ram_cacheable",
            materialization_loc="ram",
            source="allocated",
            readonly=False,
        )
        input_meta = DatasetPlanMeta(
            kind="dataset",
            dtype=np.dtype(np.float32),
            shape=input_ref.shape,
        )

        stage = _PostTaskRecordingStage(
            default_executor="thread",
            input_plan_meta=input_meta,
            resource_model=ResourceModel(
                fixed_overhead_bytes=0,
                bytes_per_scalar_in=1,
                bytes_per_scalar_out=1,
                scratch_bytes_per_scalar_in=0,
            ),
            chunking_scheme_type=SpatialTileScheme,
            output_bindings=(DataBinding("stage_out"),),
            broadcast_input={"constant": 7},
        )

        semantic_task = SemanticTask(
            priority_class="interactive",
            input_ref=input_ref,
            algorithm_pipeline=AlgorithmPipeline(stages=[stage]),
        )
        semantic_task.id = 44

        storage = _RecordingStorage()
        ctx = PlanningContext(
            sched_cfg=_NoopSchedulerConfig(),
            storage=storage,
            chunking_policy=SimpleChunkingPolicy(),
        )
        task_plan = TaskPlanner(ctx).plan_semantic_task(semantic_task)

        post_unit_id = task_plan.stage_work_units["s00"][-1]
        post_unit = task_plan.work_units[post_unit_id]
        post_meta = task_plan.work_units_meta[post_unit_id]

        self.assertEqual(post_meta.input_ref, input_ref)
        self.assertEqual(post_meta.input_region, DatasetRegionRef(0, 6, 0, 9, 0, 3))
        self.assertEqual(post_meta.broadcast_inputs, {"constant": 7})
        self.assertEqual(post_meta.output_writes["stage_out"].region, DatasetRegionRef(0, 6, 0, 9, 0, 3))
        self.assertEqual(post_unit.deps, tuple(task_plan.stage_work_units["s00"][1:-1]))

        post_unit.fn()

        self.assertEqual(len(_RECORDED_POST_TASK_CALLS), 1)
        recorded = _RECORDED_POST_TASK_CALLS[0]
        self.assertEqual(recorded["input_ref"], input_ref)
        self.assertEqual(recorded["full_input_region"], DatasetRegionRef(0, 6, 0, 9, 0, 3))
        self.assertEqual(recorded["broadcast_inputs"], {"constant": 7})
        self.assertEqual(
            recorded["output_writes"]["stage_out"].region,
            DatasetRegionRef(0, 6, 0, 9, 0, 3),
        )
