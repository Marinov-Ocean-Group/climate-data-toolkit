"""
config.py
=========
Centralized defaults for site-specific paths and cluster settings.

Override via environment variables or function keyword arguments.
"""

from __future__ import annotations

import os
from pathlib import Path

# NCAR GLADE CMIP6 intake-ESM catalog (Casper/Derecho)
_DEFAULT_NCAR_CATALOG = (
    "/glade/collections/cmip/catalog/intake-esm-datastore/catalogs/glade-cmip6.json"
)

# PBS / Dask defaults for NCAR Casper
_DEFAULT_PBS_QUEUE = "casper"
_DEFAULT_PBS_WALLTIME = "20:00:00"
_DEFAULT_PBS_LOG_DIR = "dask-logs"
_DEFAULT_PBS_MIN_WORKERS = 2
_DEFAULT_PBS_MAX_WORKERS = 80


def get_ncar_catalog_url() -> str:
    """Return the intake-ESM catalog path/URL."""
    return os.environ.get("CDT_NCAR_CATALOG", _DEFAULT_NCAR_CATALOG)


def get_figures_dir_root() -> Path:
    """Return the root directory for spectral-analysis figure output."""
    return Path(os.environ.get("CDT_FIGURES_DIR", "fft_figures"))


def get_pbs_cluster_defaults() -> dict:
    """
    Return default keyword arguments for :func:`io_utils.start_pbs_dask_cluster`.

    Keys: queue, walltime, log_directory, minimum, maximum, local_directory.
    """
    return {
        "queue": os.environ.get("CDT_PBS_QUEUE", _DEFAULT_PBS_QUEUE),
        "walltime": os.environ.get("CDT_PBS_WALLTIME", _DEFAULT_PBS_WALLTIME),
        "log_directory": os.environ.get("CDT_PBS_LOG_DIR", _DEFAULT_PBS_LOG_DIR),
        "minimum": int(os.environ.get("CDT_PBS_MIN_WORKERS", _DEFAULT_PBS_MIN_WORKERS)),
        "maximum": int(os.environ.get("CDT_PBS_MAX_WORKERS", _DEFAULT_PBS_MAX_WORKERS)),
        "local_directory": os.environ.get(
            "CDT_PBS_LOCAL_DIR",
            "${SCRATCH}/dask_scratch/pbs.$PBS_JOBID/dask/spill",
        ),
    }
