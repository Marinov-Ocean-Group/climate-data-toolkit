"""
ice_analysis.py
===============
Sea-ice and ocean mixed-layer analysis routines.

Includes polynya detection, polynya area statistics, sea-ice masking,
mixed-layer depth calculation, and depth-unit conversion.
"""

import numpy as np
import xarray as xr


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
    land_mask = None
    for t in ice.time:
        ice_slice = ice.sel(time=t).values
        filled = flood_fill(ice_slice, (0, 0), np.nan, tolerance=0)
        if np.isnan(filled[-1, -1]):
            land_mask = np.isnan(filled)
            break

    if land_mask is None:
        return ds  # could not determine mask; return unchanged

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


# ---------------------------------------------------------------------------
# Mixed-layer depth
# ---------------------------------------------------------------------------

def convert_depth_to_meters(da: xr.DataArray, depth_dim: str = "lev") -> xr.DataArray:
    """
    Convert the depth coordinate of *da* from centimetres to metres if
    its ``units`` attribute indicates centimetres.

    Parameters
    ----------
    da : xr.DataArray
    depth_dim : str
        Name of the depth dimension.

    Returns
    -------
    xr.DataArray
        *da* with the depth coordinate rescaled if necessary.
    """
    if "units" in da[depth_dim].attrs:
        if da[depth_dim].attrs["units"] == "centimeters":
            da[depth_dim] = da[depth_dim] / 100.0
    return da


def compute_mixed_layer_depth(
    sigma0: xr.DataArray,
    depth_dim: str = "lev",
) -> xr.DataArray:
    """
    Calculate mixed-layer depth (MLD) from a potential density profile.

    The MLD is defined as the shallowest depth where
    ``σ₀(z) − σ₀(10 m) ≥ 0.03 kg m⁻³``.  A linear interpolation refines
    the estimate between the last satisfying level and the next one.  When
    the criterion is never exceeded the ocean-bottom depth is returned.

    Parameters
    ----------
    sigma0 : xr.DataArray
        Potential density anomaly (σ₀) with a depth dimension.
    depth_dim : str
        Name of the vertical dimension (default ``"lev"``).

    Returns
    -------
    xr.DataArray
        MLD in the same units as *depth_dim* (metres if
        :func:`convert_depth_to_meters` has been applied).
    """
    # Deepest wet level (bottom topography)
    bottom = sigma0[depth_dim].where(~sigma0.isnull()).max(dim=depth_dim)

    sigma_10 = sigma0.interp({depth_dim: 10})

    # Deepest level where the density criterion is not yet met
    mld_shallow = sigma0[depth_dim].where(sigma0 - sigma_10 < 0.03).max(dim=depth_dim)
    # First level where criterion IS met
    mld_deep    = sigma0[depth_dim].where(sigma0[depth_dim] > mld_shallow).min(depth_dim)

    rho_shallow = sigma0.where(sigma0[depth_dim] >= mld_shallow).min(depth_dim)
    rho_deep    = sigma0.where(sigma0[depth_dim] >= mld_deep).min(depth_dim)

    # Linear interpolation to the 0.03 threshold
    mld_interp = (
        (mld_deep - mld_shallow) / (rho_deep - rho_shallow)
        * (sigma_10 + 0.03 - rho_shallow)
        + mld_shallow
    )

    # Cap at ocean bottom where criterion is never exceeded
    return xr.where(mld_shallow >= bottom, bottom, mld_interp)
