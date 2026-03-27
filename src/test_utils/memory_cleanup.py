from __future__ import annotations

from typing import Any

from wiser.utils.primitives import DeletePolicy
from wiser.utils.storage_service import StorageService


def release_kept_refs(target: Any) -> None:
    """
    Flip any remaining KEEP lease records to DELETE_WHEN_RELEASABLE.

    ``target`` may be either a ``StorageService`` or an object with a
    ``storage_service`` attribute such as ``AppServices``.
    """
    storage_service = target if isinstance(target, StorageService) else target.storage_service
    for ref_id, record in list(storage_service.lease_records.items()):
        if record.delete_policy == DeletePolicy.KEEP:
            storage_service.set_delete_policy(ref_id, DeletePolicy.DELETE_WHEN_RELEASABLE)
