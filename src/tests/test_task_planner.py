"""Unit, functional, and integration tests for the task planner
in the task management and execution system.
"""
import unittest

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
    SemanticTask,
    SimpleChunkingPolicy,
    TaskPlanner,
)


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

    def make_allocation_requests(self, *, input_meta, chosen_scheme):
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

    def map_fn(self, input_ref, input_region, output_writes, broadcast_inputs=None):
        _ = (input_ref, input_region, output_writes, broadcast_inputs)
        return None


class TestTaskPlanner(unittest.TestCase):
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

        # SpatialTileScheme should use height/3 and width/3 -> 2x3 tiles over 6x9 => 9 units.
        self.assertEqual(len(task_plan.work_units), 9)
        self.assertIn("s00", task_plan.stage_work_units)
        self.assertEqual(len(task_plan.stage_work_units["s00"]), 9)

        # Verify one output allocation was requested and shape matches full dataset.
        self.assertEqual(len(storage.requests), 1)
        alloc = storage.requests[0]
        self.assertEqual(alloc.name, "stage_out")
        self.assertEqual(alloc.shape, (6, 9, 3))

        # Verify each work unit's metadata writes to the same region as its input region.
        for unit_id in task_plan.work_units:
            unit_meta = task_plan.work_units_meta[unit_id]
            self.assertIsInstance(unit_meta.input_region, DatasetRegionRef)
            self.assertEqual(len(unit_meta.output_writes), 1)
            write = unit_meta.output_writes["stage_out"]
            self.assertEqual(write.region, unit_meta.input_region)
