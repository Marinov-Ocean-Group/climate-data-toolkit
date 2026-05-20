"""
constants.py
============
Project-wide constants and type aliases.
Import from here rather than defining magic numbers inline.
"""

from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

#: Spectral method selector used throughout the spectral analysis module.
SpectralMethod = Literal["fft", "welch"]


# ---------------------------------------------------------------------------
# Spectral analysis
# ---------------------------------------------------------------------------

#: Number of low-frequency bins to skip when searching for dominant peaks.
#: Removes the DC component (bin 0) and the bin immediately adjacent to it,
#: which are often artefacts rather than meaningful periodic signals.
SKIP_FREQS: int = 2

#: Degrees of freedom assigned to the red-noise background spectrum in the
#: F-test (Gilman et al. 1963).  A large value makes the background smooth.
SIGNIFICANCE_DOF_RED: int = 500

#: Welch segment length expressed as a fraction of the total series length.
#: E.g. 0.5 means each segment covers half the series (50 % overlap assumed).
WELCH_NPERSEG_FRACTION: float = 0.5

#: Significance levels and their star markers, ordered highest → lowest.
#: Used by :func:`spectral.sig_marker` and related formatters.
SIG_LEVELS: list[tuple[float, str]] = [
    (0.99, "***"),
    (0.95, "**"),
    (0.90, "*"),
]


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

#: Root directory for spectral-analysis figure output.
#: Subdirectories named after the method (``fft/``, ``welch/``) are created
#: automatically by :func:`spectral.figures_dir`.
FIGURES_DIR_ROOT: Path = Path("fft_figures")

#: Template for the CSV summary file produced by the spectral pipeline.
#: Use ``CSV_OUTPUT_TEMPLATE.format(method="fft")`` etc.
CSV_OUTPUT_TEMPLATE: str = "periodicity_{method}.csv"
