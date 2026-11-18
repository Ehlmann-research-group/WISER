"""
Built-in spectral-index presets for band math.

Each preset is a band-math expression whose band variables are bound by
wavelength rather than by band number, so the same preset selects the
physically equivalent bands on sensors with different band layouts (EMIT,
AVIRIS-NG, AVIRIS-classic, ...).  Wavelengths are in nm.  NDWI and MSI use the
wavelengths given in their sources; NDVI and SAVI use 660 nm (red) and 860 nm
(NIR) as narrow-band equivalents of the broad red and NIR bands of the original
formulation.  A band is chosen as the nearest band within a tolerance of the
reference wavelength; when no band lies within the tolerance, resolution raises
instead of choosing a more distant band.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

from astropy import units as u

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
        band_wavelengths: Variable name to reference wavelength (astropy length
            Quantity, nm).
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
            "Original formulation uses broad MSS red (600-700 nm) and NIR (800-1100 nm) bands; "
            "660 nm and 860 nm are the narrow-band equivalents used here."
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
