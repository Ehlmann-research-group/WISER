from typing import Optional

from osgeo import osr


class CoordinateDisplayConfig:
    def __init__(self, spatial_ref: Optional[osr.SpatialReference]):
        self.spatial_ref = spatial_ref

        # Geographic CRS display configuration
        self.use_deg_min_sec = False
        self.use_negative_degrees = True

        # Projected CRS display configuration

    def __str__(self):
        return f"CoordinateDisplayConfig[{self.spatial_ref}]"


def srs_has_degrees(spatial_ref) -> bool:
    return (
        spatial_ref is not None
        and spatial_ref.IsGeographic()
        and spatial_ref.GetAngularUnitsName().lower().startswith("degree")
    )


def get_srs_units(spatial_ref) -> Optional[str]:
    unit_name = None
    if spatial_ref is not None:
        if spatial_ref.IsProjected():
            unit_name = spatial_ref.GetLinearUnitsName()
        elif spatial_ref.IsGeographic():
            unit_name = spatial_ref.GetAngularUnitsName()
    return unit_name
