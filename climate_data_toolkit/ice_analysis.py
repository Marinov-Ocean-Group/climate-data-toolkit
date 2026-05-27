"""
ice_analysis.py
===============
Sea-ice and ocean mixed-layer analysis routines.

Includes polynya detection, polynya area statistics, sea-ice masking,
mixed-layer depth calculation, and depth-unit conversion.
"""

import numpy as np
import xarray as xr

__all__ = [
    "find_ice_edge",
    "mask_land_pixels",
    "fill_ocean_with_zero",
    "detect_polynya",
    "compute_polynya_area_stats",
]


# ---------------------------------------------------------------------------
# Sea-ice edge / boundary helpers
# ---------------------------------------------------------------------------

def find_ice_edge(da: xr.DataArray) -> xr.DataArray:
    """
    Find the latitude of the ice edge (first non-zero row) in each column.

    Any value ≤ 0 is treated as ice-free.  The function returns the
    *coordinate value* (e.g. latitude in degrees) of the first row where
    ``da > 0`` in each column.

    Parameters
    ----------
    da : xr.DataArray
        2-D or 3-D array; the second-to-last dimension is treated as the
        row (latitude) axis.

    Returns
    -------
    xr.DataArray
        1-D array of ice-edge coordinate values, one per column.
    """
    da = da.where(da > 0)
    not_nan = ~da.isnull()
    first_idx = not_nan.argmax(da.dims[-2])
    return da[da.dims[-2]][first_idx]


# ---------------------------------------------------------------------------
# Sea-ice masking
# ---------------------------------------------------------------------------

def mask_land_pixels(ds: xr.Dataset) -> xr.Dataset:
    """
    Set Antarctic land pixels in ``ds.siconc`` to NaN using flood fill.

    Intended for models that do not mask land in their sea-ice output
    (e.g. GISS, INM-CM4-8).  The flood fill propagates from the top-left
    corner to identify the continental mask.

    The land mask is derived from a single representative time slice (the
    first time step).  This assumes the land mask does not vary in time.

    Parameters
    ----------
    ds : xr.Dataset
        Must contain a ``siconc`` variable with a ``time`` dimension.

    Returns
    -------
    xr.Dataset
        Copy of *ds* with land pixels set to NaN in ``siconc``.
    """
    from skimage.segmentation import flood_fill

    ice = ds.siconc
    if "time" in ice.dims:
        ice_slice = ice.isel(time=0).values
    else:
        ice_slice = ice.values

    filled = flood_fill(ice_slice, (0, 0), np.nan, tolerance=0)
    if not np.isnan(filled[-1, -1]):
        return ds  # could not determine mask; return unchanged

    land_mask = np.isnan(filled)

    mask_da = xr.DataArray(
        data=land_mask,
        dims=ice.dims[-2:],
        coords={d: ice[d] for d in ice.dims[-2:]},
    )
    ds["siconc"] = ice.where(~mask_da)
    return ds


def fill_ocean_with_zero(ds: xr.Dataset) -> xr.Dataset:
    """
    Fill NaN ocean pixels in ``ds.siconc`` with zero, then re-mask land.

    This two-step approach (fill NaN → mask land) handles models (e.g.
    E3SM-2-0) that use NaN for both ocean and land rather than just land.

    Parameters
    ----------
    ds : xr.Dataset

    Returns
    -------
    xr.Dataset
    """
    ds["siconc"] = ds.siconc.fillna(0)
    return mask_land_pixels(ds)


# ---------------------------------------------------------------------------
# Polynya detection
# ---------------------------------------------------------------------------

def detect_polynya(
    da_ice: xr.DataArray,
    da_area: xr.DataArray,
    ice_threshold: float,
    area_threshold: tuple[float, float] = (100.0, 1000.0),
    flood_points: list[tuple[int, int]] = [(0, 0)],
    buffering: float = 15.0,
) -> xr.DataArray:
    """
    Detect polynyas (open-water regions surrounded by sea ice) for each
    time step.

    The algorithm:

    1. Flood-fill from coastal entry points to remove open-ocean pixels.
    2. Threshold the remaining sea-ice concentration.
    3. Label connected components and filter by area.

    Parameters
    ----------
    da_ice : xr.DataArray
        Sea-ice concentration (``time`` × y × x), in % or fraction.
    da_area : xr.DataArray
        Grid-cell area array (y × x), in m².
    ice_threshold : float
        Maximum SIC (same units as *da_ice*) to count as open water.
    area_threshold : (min, max)
        Polynya size bounds in 10³ km².
    flood_points : list of (row, col) tuples
        Starting points for the coastal flood fill.  Extend this list for
        models with unusual domain geometries.
    buffering : float
        Tolerance for the flood-fill step.

    Returns
    -------
    xr.DataArray
        Same shape as *da_ice*.  Polynya pixels retain their SIC values;
        all other pixels are NaN.
    """
    from scipy import ndimage
    from skimage.segmentation import flood_fill

    da_ice = da_ice.copy()
    da_area = da_area.copy().fillna(float(da_area.mean().values))

    struct = ndimage.generate_binary_structure(2, 2)
    result = xr.DataArray(
        np.nan * np.empty_like(da_ice),
        dims=da_ice.dims,
        coords=da_ice.coords,
    )

    for year in da_ice.time:
        ice_slice = da_ice.sel(time=year).fillna(0).values.copy()

        # Remove coastal open water via flood fill
        for pt in flood_points:
            ice_slice = flood_fill(ice_slice, pt, 0, tolerance=buffering)
        # Remove open ocean from the bottom-right corner
        ice_slice = flood_fill(
            ice_slice, (ice_slice.shape[0] - 1, ice_slice.shape[1] - 1),
            0, tolerance=buffering,
        )

        open_water = ice_slice <= ice_threshold
        labeled, n_features = ndimage.label(open_water, structure=struct)

        if n_features < 2:
            continue

        mask = np.zeros_like(labeled)
        for i in range(1, n_features + 1):
            area_km3 = float(da_area.where(labeled == i).sum()) / 1e9  # m² → 10³ km²
            if area_threshold[0] < area_km3 < area_threshold[1]:
                mask[labeled == i] = 1

        ice_values = da_ice.sel(time=year).values.copy()
        ice_values[mask == 0] = np.nan
        result.loc[year] = ice_values

    return result


def compute_polynya_area_stats(
    ds: xr.Dataset,
    ice_threshold: float,
    area_threshold: tuple[float, float],
    flood_points: list[tuple[int, int]],
    buffering: float,
    min_occurrences: int | bool = False,
) -> list[float]:
    """
    Compute summary area statistics for polynyas detected in *ds*.

    Parameters
    ----------
    ds : xr.Dataset
        Must contain ``siconc`` and ``areacello``.
    ice_threshold : float
        SIC threshold passed to :func:`detect_polynya`.
    area_threshold : (min, max)
        Area bounds in 10³ km².
    flood_points : list of tuples
        Coastal flood-fill entry points.
    buffering : float
        Flood-fill tolerance.
    min_occurrences : int or False
        If an integer, only count grid cells where a polynya appeared at
        least this many times across the time series.

    Returns
    -------
    [area_total, area_max, area_mean] : list of float
        Total, maximum single-year, and mean annual polynya area (m²).
    """
    masked = detect_polynya(
        ds.siconc, ds.areacello,
        ice_threshold, area_threshold,
        flood_points=flood_points,
        buffering=buffering,
    )
    count = masked.count("time")
    if min_occurrences:
        count = count.where(count >= min_occurrences)

    spatial_dims = tuple(ds.areacello.dims)
    area_total = float(ds.areacello.where(count > 0).sum().values)
    area_max   = float(ds.areacello.where(count > 0).sum(spatial_dims).max().values)
    area_mean  = float(ds.areacello.where(count > 0).sum(spatial_dims).mean().values)
    return [area_total, area_max, area_mean]

