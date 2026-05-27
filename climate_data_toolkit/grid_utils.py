"""
grid_utils.py
=============
Coordinate and grid utilities: lat/lon extraction, area calculation,
regridding, coordinate manipulation, and grid alignment helpers.
"""

import numpy as np
import xarray as xr
import pandas as pd
from pyproj import Geod

__all__ = [
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
]

_REGRIDDER_CACHE: dict[tuple, object] = {}


def _regridder_cache_key(ds_in, ds_out) -> tuple:
    """Build a hashable key from source and target grid coordinate arrays."""
    def _coord_tuple(ds, dim):
        if dim not in ds:
            return (dim, None, None)
        val = ds[dim]
        arr = np.asarray(val.values if hasattr(val, "values") else val)
        return (dim, arr.shape, arr.tobytes())

    keys = []
    for ds in (ds_in, ds_out):
        dims = sorted(ds.keys()) if isinstance(ds, dict) else sorted(ds)
        for dim in dims:
            keys.append(_coord_tuple(ds, dim))
    return tuple(keys)


# ---------------------------------------------------------------------------
# Latitude / longitude extraction
# ---------------------------------------------------------------------------

def get_latlon(
    data_info: dict,
    ds: xr.Dataset,
    new_latlon_names=False,
    no_latlon: bool = False,
):
    """
    Extract or construct latitude and longitude DataArrays from a dataset.

    Parameters
    ----------
    data_info : dict
        Metadata dict with keys ``latname``, ``lonname``, ``xname``, ``yname``.
    ds : xr.Dataset
        Source dataset.
    new_latlon_names : tuple or False
        If provided, a ``(lat_name, lon_name)`` pair that overrides
        whatever is in *data_info*.
    no_latlon : bool
        Force the function to construct a meshgrid from ``xname`` / ``yname``
        coordinates rather than reading ``latname`` / ``lonname`` directly.

    Returns
    -------
    dlat, dlon : xr.DataArray
        Latitude and longitude arrays (2-D if curvilinear).
    """
    if not no_latlon:
        no_latlon = pd.isna(data_info["latname"])

    if no_latlon:
        # Build lat/lon as a meshgrid from the x/y coordinate axes
        latname = new_latlon_names[0] if new_latlon_names else data_info["yname"]
        lonname = new_latlon_names[1] if new_latlon_names else data_info["xname"]
        x = ds[lonname]
        y = ds[latname]
        newlon, newlat = np.meshgrid(x, y)
        dlat = xr.DataArray(newlat, dims={latname: y.values, lonname: x.values})
        dlon = xr.DataArray(newlon, dims={latname: y.values, lonname: x.values})
    else:
        latname = new_latlon_names[0] if new_latlon_names else data_info["latname"]
        lonname = new_latlon_names[1] if new_latlon_names else data_info["lonname"]
        dlat = ds[latname].load()
        dlon = ds[lonname].load()
        # Mask out-of-range fill values
        dlat = dlat.where(dlat < 91).where(dlat > -91)
        dlon = dlon.where(dlon < 361).where(dlon > -361)

    # Strip time coordinates that some models attach to static lat/lon vars
    for extra in ("time", "time_bounds"):
        if extra in dlat.coords:
            dlat = dlat.reset_coords(extra, drop=True)
            dlon = dlon.reset_coords(extra, drop=True)

    return dlat, dlon


def fill_missing_coordinates(
    dlon: xr.DataArray,
    dlat: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Fill NaN values in 2-D curvilinear coordinate arrays using a meshgrid
    constructed from the first fully-valid row/column.

    Parameters
    ----------
    dlon, dlat : xr.DataArray
        2-D longitude and latitude arrays that may contain NaN fill values.

    Returns
    -------
    dlon_filled, dlat_filled : xr.DataArray
    """
    lonv = dlon.where((dlon > -361) & (dlon < 361)).values
    latv = dlat.where((dlat > -91) & (dlat < 91)).values

    newx0 = lonv[~np.isnan(lonv).any(axis=1)][0]
    newy0 = latv[:, ~np.isnan(latv).any(axis=0)][:, 0]

    newx, newy = np.meshgrid(newx0, newy0)

    lon_filled = np.where(np.isnan(lonv), newx, lonv)
    lat_filled = np.where(np.isnan(latv), newy, latv)

    return (
        xr.DataArray(lon_filled, dims=dlon.dims, coords=dlon.coords),
        xr.DataArray(lat_filled, dims=dlat.dims, coords=dlat.coords),
    )


# ---------------------------------------------------------------------------
# Grid cell area calculation
# ---------------------------------------------------------------------------

def compute_grid_area_from_bounds(ds: xr.Dataset, variable: str) -> xr.DataArray:
    """
    Compute grid-cell area (m²) from x/y boundary coordinate arrays.

    Expects boundary variables named ``<xdim>_bnds`` and ``<ydim>_bnds``
    in *ds*.

    Parameters
    ----------
    ds : xr.Dataset
    variable : str
        Name of the data variable whose last two dimensions define the grid.

    Returns
    -------
    xr.DataArray
        Area in m², NaN where non-positive.

    Raises
    ------
    TypeError
        If the boundary coordinates are missing from *ds*.
    """
    g = Geod(ellps="sphere")
    da = ds[variable]
    ydim, xdim = da.dims[-2], da.dims[-1]
    ybnds_name, xbnds_name = ydim + "_bnds", xdim + "_bnds"

    if not (
        (ybnds_name in ds.data_vars or ybnds_name in ds.coords)
        and (xbnds_name in ds.data_vars or xbnds_name in ds.coords)
    ):
        raise TypeError(f"Boundary coordinates '{ybnds_name}' / '{xbnds_name}' not found.")

    ybnds = ds[ybnds_name].values
    xbnds = ds[xbnds_name].values
    y = ds[ydim].values
    x = ds[xdim].values

    dx = np.empty((len(y), len(x))) * np.nan
    dy = np.empty((len(y), len(x))) * np.nan

    for i in range(len(x)):
        for j in range(len(y)):
            _, _, dx[j, i] = g.inv(xbnds[i, 0], y[j], xbnds[i, 1], y[j])

    for j in range(len(y)):
        _, _, dy[j, :] = g.inv(x[0], ybnds[j, 0], x[0], ybnds[j, 1])

    area = xr.DataArray(
        data=dx * dy,
        dims=da.dims[-2:],
        coords={xdim: da[xdim], ydim: da[ydim]},
    )
    return area.where(area > 0)


def compute_grid_area_from_vertices(ds: xr.Dataset, ds_info: dict) -> xr.DataArray:
    """
    Compute grid-cell area (m²) from cell-corner vertex arrays on a
    curvilinear grid.

    Expects vertex variables named ``vertices_<latname>`` and
    ``vertices_<lonname>`` in *ds*, with shape ``(..., 4)`` following the
    corner ordering::

        (1)-----(2)
         |       |
        (4)-----(3)

    Parameters
    ----------
    ds : xr.Dataset
    ds_info : dict
        Dict with keys ``latname`` and ``lonname``.

    Returns
    -------
    xr.DataArray
        Area in m², NaN where non-positive.

    Raises
    ------
    TypeError
        If the vertex arrays are missing from *ds*.
    """
    g = Geod(ellps="sphere")
    latname = ds_info["latname"]
    lonname = ds_info["lonname"]
    latv_name = "vertices_" + latname
    lonv_name = "vertices_" + lonname

    if not (
        (latv_name in ds.data_vars or latv_name in ds.coords)
        and (lonv_name in ds.data_vars or lonv_name in ds.coords)
    ):
        raise TypeError(f"Vertex coordinates '{latv_name}' / '{lonv_name}' not found.")

    latb = ds[latv_name].values
    lonb = ds[lonv_name].values
    lat = ds[latname].values
    lon = ds[lonname].values

    dx = np.empty_like(lon) * np.nan
    dy = np.empty_like(lat) * np.nan

    for i in range(dy.shape[0]):
        for j in range(dy.shape[1]):
            _, _, dy[i, j] = g.inv(lonb[i, j, 1], latb[i, j, 1],
                                   lonb[i, j, 2], latb[i, j, 2])

    for i in range(dx.shape[0]):
        for j in range(dx.shape[1]):
            _, _, dx[i, j] = g.inv(lonb[i, j, 0], latb[i, j, 0],
                                   lonb[i, j, 1], latb[i, j, 1])

    lat_dims = ds[latname].dims
    area = xr.DataArray(
        data=dx * dy,
        dims=lat_dims,
        coords={
            lat_dims[0]: ds[latname][lat_dims[0]],
            lat_dims[1]: ds[latname][lat_dims[1]],
        },
    )
    return area.where(area > 0)


def load_cell_area(
    base_path: str,
    model_name: str,
    variable: str,
    data_info: dict,
) -> xr.Dataset:
    """
    Load the ocean cell-area (``areacello``-style) NetCDF file for a model.

    Parameters
    ----------
    base_path : str
        Directory containing the grid files (with trailing slash).
    model_name : str
        CMIP model identifier.
    variable : str
        Variable name, typically ``"areacello"``.
    data_info : dict
        Metadata dict; used for ``grid_label`` when multiple matches exist.

    Returns
    -------
    xr.Dataset

    Raises
    ------
    ValueError
        If zero or more than one matching file is found after disambiguation.
    """
    import glob
    pattern = base_path + variable + "_Ofx_" + model_name + "_*.nc"
    matches = glob.glob(pattern)

    if len(matches) == 1:
        return xr.open_mfdataset(matches)
    elif len(matches) > 1:
        pattern2 = (
            base_path + variable + "_Ofx_" + model_name
            + "_*" + data_info["grid_label"] + "*.nc"
        )
        matches = glob.glob(pattern2)
        if len(matches) == 1:
            return xr.open_mfdataset(matches)
    raise ValueError(f"Could not unambiguously locate cell-area file for {model_name!r}.")


# ---------------------------------------------------------------------------
# Dataset / coordinate construction helpers
# ---------------------------------------------------------------------------

def build_dataset(
    da: xr.DataArray,
    area: xr.DataArray,
    lat: xr.DataArray,
    lon: xr.DataArray,
    variable_name: str,
) -> xr.Dataset:
    """
    Assemble a new :class:`xr.Dataset` containing a data variable plus
    the associated area and coordinate arrays.

    Parameters
    ----------
    da : xr.DataArray
        Primary data variable.
    area : xr.DataArray
        Grid-cell area (e.g. ``areacello``).
    lat, lon : xr.DataArray
        Latitude and longitude arrays (``newlat``, ``newlon``).
    variable_name : str
        Name to give *da* in the output dataset.

    Returns
    -------
    xr.Dataset
    """
    try:
        return xr.Dataset(
            data_vars={
                variable_name: da,
                "areacello": (lat.dims, area.values),
                "newlat": lat,
                "newlon": lon,
            }
        )
    except Exception as err:
        print(f"build_dataset error: {err}")
        raise


def drop_non_dim_coords(ds: xr.Dataset) -> xr.Dataset:
    """
    Drop all coordinate variables that are not also dimension coordinates.

    Useful for cleaning up datasets before concatenation or regridding.

    Parameters
    ----------
    ds : xr.Dataset

    Returns
    -------
    xr.Dataset
    """
    for v in list(ds.coords):
        if v not in ds.dims:
            ds = ds.drop_vars(v)
    return ds


def copy_latlon_coords(
    source: xr.Dataset,
    target: xr.Dataset,
    latname: str,
    lonname: str,
) -> xr.Dataset:
    """
    Assign latitude and longitude coordinate arrays from *source* onto *target*.

    Parameters
    ----------
    source : xr.Dataset
        Dataset whose lat/lon arrays are to be copied.
    target : xr.Dataset
        Dataset to receive the new coordinates.
    latname, lonname : str
        Names of the lat/lon variables in both datasets.

    Returns
    -------
    xr.Dataset
    """
    return target.assign_coords({
        latname: source[latname],
        lonname: source[lonname],
    })


def rename_spatial_dims(
    source,
    target,
):
    """
    Rename the last two dimensions of *target* to match those of *source*.

    Useful for aligning datasets on different grids before operations.

    Parameters
    ----------
    source, target : xr.DataArray or xr.Dataset

    Returns
    -------
    Same type as *target*.
    """
    return target.rename({
        target.dims[-1]: source.dims[-1],
        target.dims[-2]: source.dims[-2],
    })


def copy_spatial_coords(source: xr.DataArray, target: xr.DataArray) -> xr.DataArray:
    """
    Copy the coordinate values of the last two dimensions from *source* to
    *target*, renaming *target*'s dimensions to match first.

    Parameters
    ----------
    source, target : xr.DataArray

    Returns
    -------
    xr.DataArray
    """
    target = rename_spatial_dims(source, target)
    target[target.dims[-1]] = source[source.dims[-1]].values
    target[target.dims[-2]] = source[source.dims[-2]].values
    return target


# ---------------------------------------------------------------------------
# Grid alignment / transformation helpers
# ---------------------------------------------------------------------------

def regrid_data(
    da: xr.DataArray,
    ds_in: xr.Dataset,
    ds_out: xr.Dataset,
    reuse: bool = True,
) -> xr.DataArray:
    """
    Regrid *da* from the source grid *ds_in* to *ds_out* using bilinear
    interpolation (via xESMF).

    Parameters
    ----------
    da : xr.DataArray
        Data to regrid.
    ds_in : xr.Dataset
        Source grid descriptor.
    ds_out : xr.Dataset
        Target grid descriptor.
    reuse : bool
        If ``True`` (default), cache and reuse the xESMF Regridder for
        identical grid pairs.  Set ``False`` for one-off regrids.

    Returns
    -------
    xr.DataArray
    """
    import xesmf as xe

    if reuse:
        key = _regridder_cache_key(ds_in, ds_out)
        regridder = _REGRIDDER_CACHE.get(key)
        if regridder is None:
            regridder = xe.Regridder(ds_in, ds_out, "bilinear", periodic=True)
            _REGRIDDER_CACHE[key] = regridder
    else:
        regridder = xe.Regridder(ds_in, ds_out, "bilinear", periodic=True)

    return regridder(da)


def regrid_to_target_grid(
    da: xr.DataArray,
    target_grid: xr.Dataset,
    ds_info: dict,
) -> xr.DataArray:
    """
    Regrid *da* to the grid described by *target_grid*, using coordinate
    names from *ds_info*.

    Parameters
    ----------
    da : xr.DataArray
        Data to regrid.
    target_grid : xr.Dataset
        Dataset on the target grid.
    ds_info : dict
        Dict with keys ``xname`` and ``yname``.

    Returns
    -------
    xr.DataArray
    """
    xname = ds_info["xname"]
    yname = ds_info["yname"]
    ds_in = {xname: da[xname].values, yname: da[yname].values}
    ds_out = {xname: target_grid[xname].values, yname: target_grid[yname].values}
    return regrid_data(da, ds_in, ds_out)


def flip_latitude(ds: xr.Dataset) -> xr.Dataset:
    """
    Reverse the second-to-last dimension (latitude) of *ds* and reset its
    coordinate values to match the original ordering.

    Useful for models that store data south-to-north when north-to-south is
    expected (or vice versa).
    """
    lat_dim = ds.dims[-2]
    flipped = ds.reindex({lat_dim: ds[lat_dim][::-1]})
    return flipped.assign_coords({lat_dim: ds[lat_dim]})


def shift_longitude_origin(ds: xr.Dataset, new_start_lon: float) -> xr.Dataset:
    """
    Roll the longitude axis of *ds* so that it starts near *new_start_lon*.

    Uses ``ds.newlon`` (the 2-D longitude array) to find the correct split
    point.

    Parameters
    ----------
    ds : xr.Dataset
    new_start_lon : float
        Target starting longitude (degrees).

    Returns
    -------
    xr.Dataset
    """
    lon_dim = ds.newlon.dims[-1]
    lon360 = (ds.newlon + 360) % 360 - new_start_lon
    if np.abs(lon360[0, 0]) > 5:
        split = int(np.argmin(np.abs(lon360[0, :]).values))
        part_a = ds.isel({lon_dim: slice(split, None)})
        part_b = ds.isel({lon_dim: slice(0, split)})
        return xr.concat([part_a, part_b], dim=lon_dim)
    return ds


def roll_longitude_axis(da: xr.DataArray) -> xr.DataArray:
    """
    Shift the longitude axis of *da* by one step, wrapping the first column
    to the end and assigning it longitude = first_value + 360.

    This is a model-specific workaround for grids where the area and sea-ice
    longitude axes are offset by one step (e.g. CAS-ESM2-0).

    Parameters
    ----------
    da : xr.DataArray

    Returns
    -------
    xr.DataArray
    """
    lon_dim = da.dims[1]
    first_col = da.isel({lon_dim: 0})
    first_col[lon_dim] = da[lon_dim][0].values + 360
    rest = da.sel({lon_dim: slice(1, None)})
    return xr.concat([rest, first_col], dim=lon_dim)
