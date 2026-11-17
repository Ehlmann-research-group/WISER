"""
Built-in spectral-index presets for band math.

Each preset is a band-math expression whose band variables are bound by
wavelength rather than by band number, so the same preset selects the
physically equivalent bands on sensors with different band layouts (EMIT,
AVIRIS-NG, AVIRIS-classic, ...).  Every reference wavelength is in nm and is the
nominal band center used in the cited source.  A band is chosen as the nearest
band within a tolerance of the reference wavelength; when no band lies within
the tolerance, resolution raises instead of choosing a more distant band.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

from astropy import units as u

from wiser.raster.dataset import RasterDataDynamicBand, RasterDataSet

# Default band-selection tolerance in nm.  It is about one band spacing of the
# common imaging spectrometers (AVIRIS-NG ~5 nm, EMIT ~7.4 nm, AVIRIS-classic
# ~10 nm), whose nearest band always lies within half a spacing, so a regular
# hyperspectral grid always resolves; a sensor whose nearest band is further
# away than one hyperspectral band (e.g. a broadband multispectral sensor, or a
# grid that stops short of the wavelength) is refused.
DEFAULT_WAVELENGTH_TOLERANCE_NM: float = 10.0


@dataclass(frozen=True)
class SpectralIndexPreset:
    """
    A spectral index defined by a band-math expression and the reference
    wavelength of each band variable in it.

    Attributes:
        name: Short index name, e.g. ``"NDVI"``.
        title: Human-readable name of the index.
        expression: Band-math expression; its variables are exactly the keys of
            ``band_wavelengths``.
        band_wavelengths: Variable name to reference wavelength, in nm.
        reference: Literature source of the formula and wavelengths.
    """

    name: str
    title: str
    expression: str
    band_wavelengths: Dict[str, u.Quantity]
    reference: str


SPECTRAL_INDEX_PRESETS: Tuple[SpectralIndexPreset, ...] = (
    SpectralIndexPreset(
        name="NDVI",
        title="Normalized Difference Vegetation Index",
        expression="(nir - red) / (nir + red)",
        band_wavelengths={"nir": 860.0 * u.nm, "red": 660.0 * u.nm},
        reference=(
            "Rouse, J.W., Haas, R.H., Schell, J.A., Deering, D.W. (1974). Monitoring vegetation "
            "systems in the Great Plains with ERTS. Third ERTS Symposium, NASA SP-351, 309-317. "
            "Narrow-band red 660 nm and NIR 860 nm."
        ),
    ),
    SpectralIndexPreset(
        name="SAVI",
        title="Soil-Adjusted Vegetation Index (L = 0.5)",
        expression="1.5 * (nir - red) / (nir + red + 0.5)",
        band_wavelengths={"nir": 860.0 * u.nm, "red": 660.0 * u.nm},
        reference=(
            "Huete, A.R. (1988). A soil-adjusted vegetation index (SAVI). Remote Sensing of "
            "Environment 25(3), 295-309. SAVI = (1 + L)(NIR - Red)/(NIR + Red + L), L = 0.5; "
            "red 660 nm and NIR 860 nm as for NDVI."
        ),
    ),
    SpectralIndexPreset(
        name="NDWI",
        title="Normalized Difference Water Index (Gao)",
        expression="(r860 - r1240) / (r860 + r1240)",
        band_wavelengths={"r860": 860.0 * u.nm, "r1240": 1240.0 * u.nm},
        reference=(
            "Gao, B.-C. (1996). NDWI - A normalized difference water index for remote sensing of "
            "vegetation liquid water from space. Remote Sensing of Environment 58(3), 257-266. "
            "NDWI = (R860 - R1240)/(R860 + R1240)."
        ),
    ),
    SpectralIndexPreset(
        name="MSI",
        title="Moisture Stress Index",
        expression="r1600 / r820",
        band_wavelengths={"r1600": 1600.0 * u.nm, "r820": 820.0 * u.nm},
        reference=(
            "Hunt, E.R. Jr., Rock, B.N. (1989). Detection of changes in leaf water content using "
            "near- and middle-infrared reflectances. Remote Sensing of Environment 30(1), 43-54. "
            "MSI = R1600/R820."
        ),
    ),
)


def get_spectral_index_preset(name: str) -> SpectralIndexPreset:
    """
    Returns the built-in preset with the given name (e.g. ``"NDVI"``).

    Raises:
        KeyError: if no built-in preset has that name.
    """
    for preset in SPECTRAL_INDEX_PRESETS:
        if preset.name == name:
            return preset
    raise KeyError(f"No spectral-index preset named {name!r}")


def resolve_preset_bands(
    preset: SpectralIndexPreset, dataset: RasterDataSet, tolerance_nm: float
) -> Dict[str, int]:
    """
    Returns, for each band variable of ``preset``, the index of the band of
    ``dataset`` nearest the variable's reference wavelength among the bands
    within ``tolerance_nm`` nm of it.

    Raises:
        ValueError: if ``dataset`` has no wavelengths, or if no band lies within
            ``tolerance_nm`` of a reference wavelength.
    """
    return {
        var: RasterDataDynamicBand(
            dataset,
            wavelength_value=wavelength.to_value(u.nm),
            wavelength_units=u.nm,
            epsilon=tolerance_nm,
        ).resolve_band_index()
        for var, wavelength in preset.band_wavelengths.items()
    }
