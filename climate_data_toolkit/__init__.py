"""
climate_data_toolkit
====================
Climate analysis library for sea-ice, spectral analysis, and plotting.

Recommended import style::

    from climate_data_toolkit import load_pickle, compute_spectrum, detect_polynya

Submodules
----------
constants     — project-wide constants and type aliases
config        — environment-driven defaults (NCAR paths, PBS cluster)
io_utils      — file I/O (pickle, NetCDF, cloud, NCAR catalog, Dask)
grid_utils    — coordinate/grid operations
ice_analysis  — sea ice, polynyas, mixed-layer depth
convection_analysis — convection-area helpers and sigma0 thermodynamics
spectral      — spectral analysis, filtering, lagged correlation
plot_utils    — plotting utilities
"""

__version__ = "0.1.0"

from .constants import (
    CSV_OUTPUT_TEMPLATE,
    FIGURES_DIR_ROOT,
    SIG_LEVELS,
    SIGNIFICANCE_DOF_RED,
    SKIP_FREQS,
    SpectralMethod,
    WELCH_NPERSEG_FRACTION,
)
from .config import get_figures_dir_root, get_ncar_catalog_url, get_pbs_cluster_defaults
from .grid_utils import (
    build_dataset,
    compute_grid_area_from_bounds,
    compute_grid_area_from_vertices,
    copy_latlon_coords,
    copy_spatial_coords,
    drop_non_dim_coords,
    fill_missing_coordinates,
    flip_latitude,
    get_latlon,
    load_cell_area,
    regrid_data,
    regrid_to_target_grid,
    rename_spatial_dims,
    roll_longitude_axis,
    shift_longitude_origin,
)
from .ice_analysis import (
    compute_polynya_area_stats,
    detect_polynya,
    fill_ocean_with_zero,
    find_ice_edge,
    mask_land_pixels,
)
from .convection_analysis import (
    find_convection_area,
    compute_sigma0,
    compute_mixed_layer_depth,
    convert_depth_to_meters,
    identify_convection_area_by_depth,
    identify_convection_area_by_variability,
    compute_layer_thickness,
    compute_effective_layer_thickness,
)
from .io_utils import (
    filter_cmip_catalog,
    load_from_ncar_catalog,
    load_model_netcdf,
    load_netcdf_by_month,
    load_netcdf_files,
    load_pickle,
    load_zarr_from_gcs,
    pickle_exists,
    save_monthly_temp_files,
    save_pickle,
    select_month,
    start_pbs_dask_cluster,
)
from .plot_utils import (
    add_cbar,
    add_cbars,
    add_color_band,
    add_subplot_icepolynya,
    add_text_annotation,
    add_type_color_legend,
    annotate_subplot,
    build_color_map,
    build_marker_map,
    calculate_mean_ice,
    calculate_persistent_mean_ice,
    create_circle,
    get_icemax_polynya,
    lighten_color,
    plot_by_type,
    plot_convection_maps,
    plot_corr,
    plot_global_map,
    plot_ice_hist,
    plot_polynya_maps,
    plot_polynya_maps_from_precomputed,
    plot_spectrum,
    style_scatter_axes,
    style_south_polar_map,
)
from .spectral import (
    SpectrumResult,
    apply_butterworth_filter,
    calculate_alpha,
    calculate_dominant_periods,
    compute_lagged_correlation,
    compute_spectrum,
    create_index,
    figures_dir,
    format_period,
    normalize_data,
    output_csv,
    red_noise_spectrum_at_freqs,
    significance_threshold,
    sig_marker,
    to_numpy,
    calculate_anomalies,
    boxcar_smooth,
)

__all__ = [
    "__version__",
    # config
    "get_figures_dir_root",
    "get_ncar_catalog_url",
    "get_pbs_cluster_defaults",
    # constants
    "SpectralMethod",
    "SKIP_FREQS",
    "SIGNIFICANCE_DOF_RED",
    "WELCH_NPERSEG_FRACTION",
    "SIG_LEVELS",
    "FIGURES_DIR_ROOT",
    "CSV_OUTPUT_TEMPLATE",
    # io
    "pickle_exists",
    "load_pickle",
    "save_pickle",
    "load_netcdf_files",
    "load_netcdf_by_month",
    "select_month",
    "save_monthly_temp_files",
    "load_model_netcdf",
    "load_zarr_from_gcs",
    "filter_cmip_catalog",
    "load_from_ncar_catalog",
    "start_pbs_dask_cluster",
    # grid
    "get_latlon",
    "fill_missing_coordinates",
    "compute_grid_area_from_bounds",
    "compute_grid_area_from_vertices",
    "load_cell_area",
    "build_dataset",
    "drop_non_dim_coords",
    "copy_latlon_coords",
    "rename_spatial_dims",
    "copy_spatial_coords",
    "regrid_data",
    "regrid_to_target_grid",
    "flip_latitude",
    "shift_longitude_origin",
    "roll_longitude_axis",
    # ice
    "find_ice_edge",
    "mask_land_pixels",
    "fill_ocean_with_zero",
    "detect_polynya",
    "compute_polynya_area_stats",
    # convection
    "find_convection_area",
    "compute_sigma0",
    "compute_mixed_layer_depth",
    "convert_depth_to_meters",
    "identify_convection_area_by_depth",
    "identify_convection_area_by_variability",
    "compute_layer_thickness",
    "compute_effective_layer_thickness",
    # spectral
    "to_numpy",
    "normalize_data",
    "create_index",
    "calculate_alpha",
    "red_noise_spectrum_at_freqs",
    "SpectrumResult",
    "compute_spectrum",
    "significance_threshold",
    "sig_marker",
    "format_period",
    "calculate_dominant_periods",
    "apply_butterworth_filter",
    "compute_lagged_correlation",
    "figures_dir",
    "output_csv",
    "calculate_anomalies",
    "boxcar_smooth",
    # plot
    "lighten_color",
    "create_circle",
    "style_south_polar_map",
    "plot_global_map",
    "plot_spectrum",
    "calculate_mean_ice",
    "calculate_persistent_mean_ice",
    "build_color_map",
    "build_marker_map",
    "add_color_band",
    "add_subplot_icepolynya",
    "add_cbar",
    "add_cbars",
    "add_type_color_legend",
    "annotate_subplot",
    "get_icemax_polynya",
    "plot_ice_hist",
    "plot_polynya_maps",
    "plot_polynya_maps_from_precomputed",
    "plot_convection_maps",
    "plot_by_type",
    "add_text_annotation",
    "style_scatter_axes",
    "plot_corr",
]
