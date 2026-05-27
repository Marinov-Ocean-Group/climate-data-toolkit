"""
convection_analysis.py
======================

Utilities for diagnosing ocean convection and mixed-layer properties
from salinity and temperature fields.

The module includes tools for:

- vertical grid calculations
- mixed-layer depth (MLD) diagnostics
- convection-area identification
- potential density anomaly calculations

All functions operate on ``xarray.DataArray`` objects.
"""

from __future__ import annotations

import xarray as xr

__all__ = [
    "convert_depth_to_meters",
    "compute_layer_thickness",
    "compute_effective_layer_thickness",
    "compute_mixed_layer_depth",
    "identify_convection_area_by_depth",
    "identify_convection_area_by_variability",
    "compute_sigma0",
    "find_convection_area",
]

# =============================================================================
# Constants
# =============================================================================

DEFAULT_CONVECTION_DEPTH = 2000.0
DEFAULT_STD_THRESHOLD = 200.0

MLD_DENSITY_THRESHOLD = 0.03  # kg m^-3
REFERENCE_DEPTH = 10.0        # meters


# =============================================================================
# Internal utilities
# =============================================================================

def _binary_mask(condition: xr.DataArray) -> xr.DataArray:
    """
    Convert a boolean condition into a binary integer mask.

    Parameters
    ----------
    condition : xr.DataArray
        Boolean condition.

    Returns
    -------
    xr.DataArray
        Integer mask containing 1 where condition is True and 0 otherwise.
    """
    return xr.where(condition, 1, 0)


def _require_dimension(da: xr.DataArray, dim: str) -> None:
    """
    Ensure that a required dimension exists.

    Parameters
    ----------
    da : xr.DataArray
        Input data array.
    dim : str
        Required dimension name.

    Raises
    ------
    ValueError
        If the dimension is missing.
    """
    if dim not in da.dims:
        raise ValueError(
            f"Required dimension {dim!r} not found in {da.dims}."
        )


# =============================================================================
# Coordinate utilities
# =============================================================================

def convert_depth_to_meters(
    da: xr.DataArray,
    depth_dim: str = "lev",
) -> xr.DataArray:
    """
    Convert a depth coordinate from centimeters to meters.

    Conversion is only applied when the coordinate ``units`` attribute
    equals ``"centimeters"``.

    Parameters
    ----------
    da : xr.DataArray
        Input data containing a depth coordinate.
    depth_dim : str, default="lev"
        Name of the depth coordinate.

    Returns
    -------
    xr.DataArray
        DataArray with depth coordinates converted to meters when needed.
    """
    _require_dimension(da, depth_dim)

    depth = da[depth_dim]

    if depth.attrs.get("units") == "centimeters":
        da = da.assign_coords({depth_dim: depth / 100.0})

    return da


# =============================================================================
# Vertical grid utilities
# =============================================================================

def compute_layer_thickness(
    level_bounds: xr.DataArray,
    bounds_dim: str = "bnds",
) -> xr.DataArray:
    """
    Compute vertical layer thickness from level bounds.

    Parameters
    ----------
    level_bounds : xr.DataArray
        Vertical level bounds with two entries per layer.
    bounds_dim : str, default="bnds"
        Name of the bounds dimension.

    Returns
    -------
    xr.DataArray
        Layer thickness for each vertical level.
    """
    _require_dimension(level_bounds, bounds_dim)

    upper_bound = level_bounds.isel({bounds_dim: 0})
    lower_bound = level_bounds.isel({bounds_dim: 1})

    return lower_bound - upper_bound


def compute_effective_layer_thickness(
    level_bounds: xr.DataArray,
    bottom_depth: xr.DataArray,
    *,
    bounds_dim: str = "bnds",
    drop_bounds_coordinate: bool = True,
) -> xr.DataArray:
    """
    Compute effective vertical layer thickness above the ocean bottom.

    Layers fully above the bathymetry retain their full thickness.
    The bottom-intersecting layer is clipped to the ocean depth.
    Layers fully below the bathymetry are masked.

    Parameters
    ----------
    level_bounds : xr.DataArray
        Vertical level bounds with two entries per layer.
    bottom_depth : xr.DataArray
        Ocean bottom depth at each horizontal location.
    bounds_dim : str, default="bnds"
        Name of the bounds dimension.
    drop_bounds_coordinate : bool, default=True
        Whether to remove the bounds coordinate from the output.

    Returns
    -------
    xr.DataArray
        Effective layer thickness masked below the ocean bottom.
    """
    _require_dimension(level_bounds, bounds_dim)

    layer_top = level_bounds.isel({bounds_dim: 0}, drop=True)
    layer_bottom = level_bounds.isel({bounds_dim: 1}, drop=True)

    full_thickness = layer_bottom - layer_top
    clipped_thickness = bottom_depth - layer_top

    effective_thickness = xr.where(
        layer_bottom <= bottom_depth,
        full_thickness,
        clipped_thickness,
    )

    effective_thickness = effective_thickness.where(
        layer_top < bottom_depth
    )

    if (
        drop_bounds_coordinate
        and bounds_dim in effective_thickness.coords
    ):
        effective_thickness = effective_thickness.drop_vars(bounds_dim)

    return effective_thickness


# =============================================================================
# Mixed-layer diagnostics
# =============================================================================

def compute_mixed_layer_depth(
    sigma0: xr.DataArray,
    depth_dim: str = "lev",
    density_threshold: float = MLD_DENSITY_THRESHOLD,
    reference_depth: float = REFERENCE_DEPTH,
) -> xr.DataArray:
    """
    Compute mixed-layer depth (MLD) from potential density anomaly.

    The MLD is defined as the shallowest depth satisfying:

    .. code-block:: text

        sigma0(z) - sigma0(reference_depth) >= density_threshold

    A linear interpolation is used to estimate the threshold-crossing
    depth between model levels.

    Parameters
    ----------
    sigma0 : xr.DataArray
        Potential density anomaly profile.
    depth_dim : str, default="lev"
        Name of the vertical coordinate dimension.
    density_threshold : float, default=0.03
        Density anomaly threshold in kg m^-3.
    reference_depth : float, default=10.0
        Reference depth used in the density criterion.

    Returns
    -------
    xr.DataArray
        Mixed-layer depth in the same units as the depth coordinate.
    """
    _require_dimension(sigma0, depth_dim)

    depth = sigma0[depth_dim]

    # Deepest valid ocean level
    ocean_bottom = depth.where(
        ~sigma0.isnull()
    ).max(dim=depth_dim)

    # Density referenced to shallow depth
    sigma_reference = sigma0.interp(
        {depth_dim: reference_depth}
    )

    # Density anomaly relative to reference depth
    delta_sigma = sigma0 - sigma_reference

    # Deepest level below threshold
    shallow_depth = depth.where(
        delta_sigma < density_threshold
    ).max(dim=depth_dim)

    # First level exceeding threshold
    deep_depth = depth.where(
        depth > shallow_depth
    ).min(dim=depth_dim)

    # Density values surrounding threshold crossing
    sigma_shallow = sigma0.where(
        depth >= shallow_depth
    ).min(dim=depth_dim)

    sigma_deep = sigma0.where(
        depth >= deep_depth
    ).min(dim=depth_dim)

    # Linear interpolation of threshold crossing depth
    interpolated_mld = (
        shallow_depth
        + (
            (deep_depth - shallow_depth)
            * (
                sigma_reference
                + density_threshold
                - sigma_shallow
            )
            / (sigma_deep - sigma_shallow)
        )
    )

    # Use ocean bottom when threshold is never exceeded
    return xr.where(
        shallow_depth >= ocean_bottom,
        ocean_bottom,
        interpolated_mld,
    )


# =============================================================================
# Convection diagnostics
# =============================================================================

def identify_convection_area_by_depth(
    mld: xr.DataArray,
    depth_threshold: float = DEFAULT_CONVECTION_DEPTH,
    *,
    time_dim: str = "time",
    drop_coordinate: str | None = "lev",
) -> xr.DataArray:
    """
    Identify regions where mixed-layer depth exceeds a threshold.

    Parameters
    ----------
    mld : xr.DataArray
        Mixed-layer depth with a time dimension.
    depth_threshold : float, default=2000
        Convection threshold depth.
    time_dim : str, default="time"
        Name of the time dimension.
    drop_coordinate : str or None, default="lev"
        Coordinate to remove from the output if present.

    Returns
    -------
    xr.DataArray
        Binary mask indicating where convection occurs at least once
        over the time dimension.
    """
    _require_dimension(mld, time_dim)

    convection_events = _binary_mask(
        mld >= depth_threshold
    )

    convection_mask = _binary_mask(
        convection_events.sum(dim=time_dim) > 0
    )

    if (
        drop_coordinate is not None
        and drop_coordinate in convection_mask.coords
    ):
        convection_mask = convection_mask.drop_vars(
            drop_coordinate
        )

    return convection_mask


def identify_convection_area_by_variability(
    mld: xr.DataArray,
    std_threshold: float = DEFAULT_STD_THRESHOLD,
    *,
    time_dim: str = "time",
) -> xr.DataArray:
    """
    Identify convection regions using temporal MLD variability.

    Parameters
    ----------
    mld : xr.DataArray
        Mixed-layer depth with a time dimension.
    std_threshold : float, default=200
        Standard deviation threshold.
    time_dim : str, default="time"
        Name of the time dimension.

    Returns
    -------
    xr.DataArray
        Binary mask where temporal MLD standard deviation exceeds the
        specified threshold.
    """
    _require_dimension(mld, time_dim)

    mld_std = mld.std(
        dim=time_dim,
        skipna=True,
    )

    return _binary_mask(
        mld_std > std_threshold
    )


# =============================================================================
# Density calculations
# =============================================================================

def compute_sigma0(
    salinity: xr.DataArray,
    temperature: xr.DataArray,
) -> xr.DataArray:
    """
    Compute potential density anomaly referenced to the surface.

    Parameters
    ----------
    salinity : xr.DataArray
        Practical salinity.
    temperature : xr.DataArray
        Potential temperature.

    Returns
    -------
    xr.DataArray
        Potential density anomaly (sigma0).

    Notes
    -----
    Requires the ``gsw`` package.
    """
    try:
        import gsw
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "compute_sigma0 requires the `gsw` package.\n"
            "Install with:\n"
            "    conda install -c conda-forge gsw"
        ) from exc

    valid_temperature = temperature.where(
        temperature > -10
    )

    valid_salinity = salinity.where(
        salinity > 0
    )

    return gsw.sigma0(
        valid_salinity,
        valid_temperature,
    )


def find_convection_area(
    salinity: xr.DataArray,
    temperature: xr.DataArray,
    criterion: float | None = None,
    *,
    method: str = "depth",
    depth_dim: str = "lev",
    time_dim: str = "time",
) -> xr.DataArray:
    """
    Compute a convection-area mask from salinity and temperature fields.

    Parameters
    ----------
    salinity : xr.DataArray
        Salinity field.
    temperature : xr.DataArray
        Potential temperature field.
    criterion : float or None, optional
        Threshold used by the selected method.
    method : {"depth", "std"}, default="depth"
        Convection detection method.

        - ``"depth"``
            Convection occurs where MLD exceeds a depth threshold.

        - ``"std"``
            Convection occurs where temporal MLD variability exceeds
            a standard deviation threshold.

    depth_dim : str, default="lev"
        Name of the vertical coordinate dimension.
    time_dim : str, default="time"
        Name of the time dimension.

    Returns
    -------
    xr.DataArray
        Binary convection-area mask.
    """
    sigma0 = compute_sigma0(
        salinity,
        temperature,
    )

    mld = compute_mixed_layer_depth(
        sigma0,
        depth_dim=depth_dim,
    )

    if method == "depth":
        threshold = (
            DEFAULT_CONVECTION_DEPTH
            if criterion is None
            else criterion
        )

        return identify_convection_area_by_depth(
            mld,
            depth_threshold=threshold,
            time_dim=time_dim,
        )

    if method == "std":
        threshold = (
            DEFAULT_STD_THRESHOLD
            if criterion is None
            else criterion
        )

        return identify_convection_area_by_variability(
            mld,
            std_threshold=threshold,
            time_dim=time_dim,
        )

    valid_methods = {"depth", "std"}

    raise ValueError(
        f"Invalid method: {method!r}. "
        f"Expected one of {valid_methods}."
    )