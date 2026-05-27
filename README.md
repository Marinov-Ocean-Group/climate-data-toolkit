# Climate Data Toolkit

A Python library for processing, cleaning, and analyzing climate model data: sea ice, polynyas, spectral analysis, grid utilities, and publication-style plots.

## Installation

This project targets **Python 3.12**. The recommended setup installs **all dependencies from conda-forge** and registers the environment as a Jupyter kernel for use on HPC systems and JupyterHub deployments.

### Recommended: conda-forge + `environment.yml`

Run these commands from the **repository root** (required so `pip install -e .` resolves correctly):

```bash
cd /path/to/climate-data-toolkit

# Create environment
mamba env create -f environment.yml

# Activate environment
conda activate cdt

# Register the environment as a Jupyter kernel
python -m ipykernel install --user \
    --name cdt \
    --display-name "Python (cdt)"

# Verify installation
python -c "import climate_data_toolkit as cdt; print(cdt.__version__)"
```

This creates the `cdt` environment, installs runtime libraries from **conda-forge only** (`channel_priority: strict`), editable-installs `climate-data-toolkit`, and registers the environment so it appears in JupyterLab kernel selection menus.

After launching JupyterLab, select:

```text
Kernel -> Change Kernel -> Python (cdt)
```

You can verify that the kernel was registered successfully with:

```bash
jupyter kernelspec list
```

You should see an entry similar to:

```text
cdt    ~/.local/share/jupyter/kernels/cdt
```

> **Important (HPC/JupyterHub users):**
>
> On many HPC systems, creating a Conda environment alone does **not** make it visible in JupyterLab. The `ipykernel install` step above is required so the environment appears in the kernel selector.

### Updating an existing environment

After pulling repository changes:

```bash
conda activate cdt
mamba env update -f environment.yml --prune
```

If the kernel disappears or Python version changes, re-register it:

```bash
python -m ipykernel install --user \
    --name cdt \
    --display-name "Python (cdt)"
```

### Alternative: pip-only install

If you cannot use Conda, install from the repo with pip extras (dependencies come from PyPI, not conda-forge):

```bash
cd /path/to/climate-data-toolkit

pip install -e ".[all]"    # full feature set
# pip install -e ".[plot]" # subset, etc.
```

To use the pip environment in Jupyter, also install and register an IPython kernel:

```bash
pip install ipykernel

python -m ipykernel install --user \
    --name climate-data-toolkit \
    --display-name "Python (climate-data-toolkit)"
```

Do **not** run:

```bash
pip install -e ".[all]"
```

inside the `cdt` Conda environment if you already created it from `environment.yml` — dependencies are already satisfied by Conda.

| Extra     | Packages                                 | Used for                   |
| --------- | ---------------------------------------- | -------------------------- |
| *(core)*  | numpy, pandas, xarray, scipy, pyproj     | grids, spectra, I/O basics |
| `plot`    | matplotlib, cartopy, cmocean, adjusttext | `plot_utils`               |
| `regrid`  | xesmf                                    | `regrid_data`              |
| `ice`     | scikit-image                             | polynya / land masking     |
| `cloud`   | gcsfs, zarr                              | `load_zarr_from_gcs`       |
| `catalog` | intake, intake-esm                       | `load_from_ncar_catalog`   |
| `hpc`     | dask, distributed, dask-jobqueue         | `start_pbs_dask_cluster`   |
| `all`     | union of the above                       | full feature set           |

## Quick start

```python
from climate_data_toolkit import (
    load_pickle,
    detect_polynya,
    compute_spectrum,
    load_from_ncar_catalog,
)

data = load_pickle("GFDL-CM4", "/data/processed/")
spec = compute_spectrum(values, method="welch")
```

## Configuration (environment variables)

| Variable | Default | Purpose |
|----------|---------|---------|
| `CDT_NCAR_CATALOG` | GLADE CMIP6 JSON path | `load_from_ncar_catalog` |
| `CDT_FIGURES_DIR` | `fft_figures` | Spectral figure output root |
| `CDT_PBS_QUEUE` | `casper` | Dask PBS queue |
| `CDT_PBS_WALLTIME` | `20:00:00` | Dask job walltime |
| `CDT_PBS_LOG_DIR` | `dask-logs` | Dask log directory |
| `CDT_PBS_MIN_WORKERS` | `2` | Dask adapt minimum |
| `CDT_PBS_MAX_WORKERS` | `80` | Dask adapt maximum |
| `CDT_PBS_LOCAL_DIR` | `$SCRATCH/.../spill` | Dask spill directory |

## Project structure

```
climate_data_toolkit/
  config.py         — NCAR/PBS/figure path defaults
  constants.py      — spectral constants, SpectralMethod
  io_utils.py       — pickle, NetCDF, GCS zarr, intake-ESM, Dask
  grid_utils.py     — lat/lon, area, regrid, alignment
  convection_analysis.py — convection-area helpers and sigma0 thermodynamics
  ice_analysis.py   — polynyas, land mask
  spectral.py       — FFT/Welch, significance, filtering, lag corr
  plot_utils.py     — maps, multi-panel figures, scatter plots
```

### Public API (by module)

**io_utils** — `pickle_exists`, `load_pickle`, `save_pickle`, `load_netcdf_files`, `load_netcdf_by_month`, `save_monthly_temp_files`, `load_model_netcdf`, `load_zarr_from_gcs`, `filter_cmip_catalog`, `load_from_ncar_catalog`, `start_pbs_dask_cluster`

**grid_utils** — `get_latlon`, `fill_missing_coordinates`, `compute_grid_area_from_bounds`, `compute_grid_area_from_vertices`, `load_cell_area`, `build_dataset`, `drop_non_dim_coords`, `copy_latlon_coords`, `rename_spatial_dims`, `copy_spatial_coords`, `regrid_data`, `regrid_to_target_grid`, `flip_latitude`, `shift_longitude_origin`, `roll_longitude_axis`

**ice_analysis** — `find_ice_edge`, `mask_land_pixels`, `fill_ocean_with_zero`, `detect_polynya`, `compute_polynya_area_stats`

**convection_analysis** — `find_convection_area`, `compute_sigma0`, `compute_mixed_layer_depth`, `convert_depth_to_meters`, `identify_convection_area_by_depth`, `identify_convection_area_by_variability`, `compute_layer_thickness`, `compute_effective_layer_thickness`

**spectral** — `to_numpy`, `normalize_data`, `create_index`, `calculate_alpha`, `red_noise_spectrum_at_freqs`, `compute_spectrum`, `significance_threshold`, `sig_marker`, `format_period`, `calculate_dominant_periods`, `apply_butterworth_filter`, `compute_lagged_correlation`, `figures_dir`, `output_csv`

**plot_utils** — `plot_global_map`, `plot_spectrum`, `plot_polynya_maps`, `plot_polynya_maps_from_precomputed`, `plot_convection_maps`, `plot_ice_hist`, `plot_by_type`, `plot_corr`, and styling helpers (`style_south_polar_map`, `build_color_map`, …)

### NCAR-specific functions

These assume NCAR Casper/Derecho paths or PBS:

- `load_from_ncar_catalog` (GLADE intake-ESM catalog)
- `start_pbs_dask_cluster` (Casper PBS + Dask)

## License

MIT — see [LICENSE](LICENSE).
