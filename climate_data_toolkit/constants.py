"""
constants.py
============
Project-wide constants and type aliases.
Import from here rather than defining magic numbers inline.
"""

from pathlib import Path
from typing import Literal

from .config import get_figures_dir_root

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

#: Spectral method selector used throughout the spectral analysis module.
SpectralMethod = Literal["fft", "welch"]


# ---------------------------------------------------------------------------
# Spectral analysis
# ---------------------------------------------------------------------------

#: Number of low-frequency bins to skip when searching for dominant peaks.
SKIP_FREQS: int = 2

#: Degrees of freedom assigned to the red-noise background spectrum in the
#: F-test (Gilman et al. 1963).
SIGNIFICANCE_DOF_RED: int = 500

#: Welch segment length expressed as a fraction of the total series length.
WELCH_NPERSEG_FRACTION: float = 0.5

#: Significance levels and their star markers, ordered highest → lowest.
SIG_LEVELS: list[tuple[float, str]] = [
    (0.99, "***"),
    (0.95, "**"),
    (0.90, "*"),
]

#: Root directory for spectral-analysis figure output (see ``CDT_FIGURES_DIR``).
FIGURES_DIR_ROOT: Path = get_figures_dir_root()

#: Template for the CSV summary file produced by the spectral pipeline.
CSV_OUTPUT_TEMPLATE: str = "periodicity_{method}.csv"

__all__ = [
    "SpectralMethod",
    "SKIP_FREQS",
    "SIGNIFICANCE_DOF_RED",
    "WELCH_NPERSEG_FRACTION",
    "SIG_LEVELS",
    "FIGURES_DIR_ROOT",
    "CSV_OUTPUT_TEMPLATE",
]
