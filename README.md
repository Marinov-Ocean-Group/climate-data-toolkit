# Climate Model Data Processing and Analysis

A Python library for processing, cleaning, and analyzing climate model data with ease and efficiency.

## Overview

This repository contains a collection of Python functions designed to streamline climate model data processing and analysis workflows. The code has been refined with the assistance of Claude AI to ensure clarity, efficiency, and best practices.


## Requirements

- Python 3.8+
- NumPy
- Pandas
- xarray


## Quick Start

```python
from climate_processing import load_data, clean_data, analyze_trends

# Load climate model data
data = load_data('path/to/data.nc')

# Clean and process
cleaned_data = clean_data(data)

# Analyze trends
results = analyze_trends(cleaned_data)
```

## Project Structure

```
constants.py
  └── SKIP_FREQS, SIGNIFICANCE_DOF_RED, WELCH_NPERSEG_FRACTION,
      FIGURES_DIR_ROOT, CSV_OUTPUT_TEMPLATE, SIG_LEVELS, SpectralMethod

io_utils.py
  └── load_pickle, save_pickle, pickle_exists
  └── load_zarr_from_gcs, load_netcdf_files, load_netcdf_by_month,
      save_monthly_temp_files, load_model_netcdf
  └── load_from_ncar_catalog, filter_cmip_catalog
  └── start_pbs_dask_cluster

grid_utils.py
  └── get_latlon, load_cell_area, build_dataset
  └── compute_grid_area_from_bounds, compute_grid_area_from_vertices
  └── fill_missing_coordinates
  └── drop_coords, copy_coords, rename_xy, copy_xy
  └── regrid_data, regrid_based_on_dsgxy
  └── flip_y, shift_x, change_start_x

ice_analysis.py
  └── find_ice_edge (was find_first_non_nan_row)
  └── detect_polynya, count_polynya_area
  └── set_land_to_nan, set_ocean_to_zero
  └── compute_mixed_layer_depth, convert_depth_to_meters

spectral.py
  └── to_numpy, normalize_data, create_index
  └── calculate_alpha, red_noise_spectrum_at_freqs
  └── _periods_from_freqs, SpectrumResult
  └── compute_fft_spectrum, compute_welch_spectrum, compute_spectrum
  └── significance_threshold, sig_marker, format_period
  └── calculate_dominant_periods
  └── apply_butterworth_filter, _filter_along_axis
  └── compute_lagged_correlation
  └── figures_dir, output_csv

plot_utils.py 
  └── lighten_color (single copy)
  └── create_circle, style_south_polar_map, plot_global_map
  └── plot_spectrum (from plotfunctions.py)
  └── calculate_mean_ice, calculate_pmean_ice
  └── build_color_map, build_marker_map
  └── add_color_band, add_subplot_mld, add_subplot_icepolynya
  └── add_cbar, add_cbars, add_type_color_legend, annotate_subplot
  └── get_icemax_polynya
  └── plot_ice_hist, plot_polynya_maps, plot_polynya_maps_from_polynya_data
  └── plot_convection_maps
  └── plot_by_type, add_text_annotation, style_scatter_axes, plot_corr
  └── to_percentage
```


## License

This project is licensed under the MIT License - see the LICENSE file for details.
