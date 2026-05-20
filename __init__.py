"""
mylib
=====
Climate analysis library for sea-ice, spectral analysis, and plotting.

Package structure
-----------------
constants.py    — project-wide constants and type aliases
io_utils.py     — file I/O (pickle, NetCDF, cloud, NCAR catalog, Dask)
grid_utils.py   — coordinate/grid operations (area calc, regrid, alignment)
ice_analysis.py — sea ice & polynya detection, mixed-layer depth
spectral.py     — spectral analysis (FFT/Welch), filtering, lagged correlation
plot_utils.py   — all plotting utilities (maps, spectra, scatter, multi-panel)

Usage
-----
Import specific names::

    from mylib import load_pickle, compute_spectrum, detect_polynya

Or import everything (mirrors the old ``from myfunctions import *``)::

    from mylib import *
"""

from .constants    import *   # noqa: F401, F403
from .io_utils     import *   # noqa: F401, F403
from .grid_utils   import *   # noqa: F401, F403
from .ice_analysis import *   # noqa: F401, F403
from .spectral     import *   # noqa: F401, F403
from .plot_utils   import *   # noqa: F401, F403
