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
        return f'CoordinateDisplayConfig[{self.spatial_ref}]'
