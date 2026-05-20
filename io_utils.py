"""
io_utils.py
===========
File I/O utilities: pickle helpers, NetCDF loaders, cloud/NCAR catalog
access, and Dask cluster setup.

All functions that touch the filesystem or a remote data store live here.
"""

import os
import glob
import pickle
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd
import xarray as xr


# ---------------------------------------------------------------------------
# Pickle helpers
# ---------------------------------------------------------------------------

def pickle_exists(name: str, base_path: str) -> bool:
    """
    Check whether a pickle file exists on disk.

    Parameters
    ----------
    name : str
        Filename stem (no extension).
    base_path : str
        Directory path (with trailing slash) where the file would live.

    Returns
    -------
    bool
        ``True`` if ``<base_path><name>.pickle`` exists.
    """
    path = base_path + name + ".pickle"
    return os.path.exists(path)


def load_pickle(name: str, base_path: str):
    """
    Load and return the contents of a pickle file.

    Parameters
    ----------
    name : str
        Filename stem (no extension).
    base_path : str
        Directory containing the pickle file (with trailing slash).

    Returns
    -------
    object
        Deserialised Python object stored in the pickle.

    Example
    -------
    >>> data = load_pickle("model_output", "/data/processed/")
    """
    path = base_path + name + ".pickle"
    return pd.read_pickle(path)


def save_pickle(name: str, base_path: str, obj) -> None:
    """
    Serialise *obj* and save it as a pickle file.

    Parameters
    ----------
    name : str
        Filename stem (no extension).
    base_path : str
        Directory in which to write the file (with trailing slash).
    obj : object
        Python object to serialise.
    """
    path = base_path + name + ".pickle"
    with open(path, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# ---------------------------------------------------------------------------
# NetCDF loaders
# ---------------------------------------------------------------------------

def _drop_type_coord(ds: xr.Dataset) -> xr.Dataset:
    """Drop the spurious 'type' coordinate that some CMIP models write."""
    if "type" in ds.coords:
        ds = ds.reset_coords("type", drop=True)
    return ds


def load_netcdf_files(file_list: list[str]) -> xr.Dataset:
    """
    Open and concatenate one or more NetCDF files along the time dimension.

    For very large file lists (> 50 files) the files are opened and
    concatenated one-by-one to avoid memory spikes.

    Parameters
    ----------
    file_list : list of str
        Paths to NetCDF files.

    Returns
    -------
    xarray.Dataset
    """
    if len(file_list) > 50:
        ds = xr.open_mfdataset(file_list[0], use_cftime=True)
        for path in file_list[1:]:
            ds0 = xr.open_mfdataset(path, use_cftime=True)
            ds = xr.concat([ds, ds0], dim="time")
    else:
        ds = xr.open_mfdataset(file_list, use_cftime=True)
    return _drop_type_coord(ds)


def load_netcdf_by_month(file_list: list[str], month: int) -> xr.Dataset:
    """
    Open NetCDF files and retain only the specified calendar month.

    For large file lists (> 5 files) each file is filtered individually
    before concatenation to limit peak memory usage.

    Parameters
    ----------
    file_list : list of str
        Paths to NetCDF files.
    month : int
        Calendar month to keep (1 = January … 12 = December).

    Returns
    -------
    xarray.Dataset
    """
    def _select_month(ds: xr.Dataset, n: int) -> xr.Dataset:
        return ds.isel(time=(ds.time.dt.month == n))

    if len(file_list) > 5:
        ds = xr.open_mfdataset(file_list[0], use_cftime=True)
        ds = _select_month(ds, month)
        for path in file_list[1:]:
            ds0 = xr.open_mfdataset(path, use_cftime=True)
            ds0 = _select_month(ds0, month)
            ds = xr.concat([ds, ds0], dim="time")
    else:
        ds = xr.open_mfdataset(file_list, use_cftime=True, chunks={"time": 12})
        ds = _select_month(ds, month)

    return _drop_type_coord(ds)


def save_monthly_temp_files(file_list: list[str], month: int) -> None:
    """
    For each file in *file_list*, filter to *month* and write a ``_temp.nc``
    copy alongside the original.

    This is a workaround for models (e.g. CanESM5-1) whose files cannot be
    opened in multi-file mode with month selection applied simultaneously.

    Parameters
    ----------
    file_list : list of str
        Original NetCDF file paths.
    month : int
        Calendar month to keep (1–12).
    """
    for path in file_list:
        ds = xr.open_mfdataset(path, use_cftime=True)
        ds = ds.isel(time=(ds.time.dt.month == month))
        ds.to_netcdf(path.replace(".nc", "_temp.nc"))
        ds.close()
        del ds


def load_model_netcdf(
    base_path: str,
    model_name: str,
    variable: str,
    selected_month: Optional[int] = None,
) -> xr.Dataset:
    """
    Locate and open CMIP-style NetCDF files for a given model and variable.

    File names are expected to follow the pattern::

        <base_path><variable>_*<model_name>_piControl_*.nc

    For CanESM5-1 with month selection, intermediate ``_temp.nc`` files are
    written to disk first (see :func:`save_monthly_temp_files`).

    Parameters
    ----------
    base_path : str
        Directory containing the NetCDF files (with trailing slash).
    model_name : str
        CMIP model identifier, e.g. ``"GFDL-CM4"``.
    variable : str
        CMIP variable name, e.g. ``"siconc"``.
    selected_month : int or None
        If given, only this calendar month is loaded.

    Returns
    -------
    xarray.Dataset

    Raises
    ------
    ValueError
        If no matching files are found.
    """
    pattern = base_path + variable + "_*" + model_name + "_piControl_" + "*.nc"
    matching = glob.glob(pattern)

    if not matching:
        raise ValueError(f"No {variable!r} data found for model {model_name!r}.")

    if selected_month:
        if model_name == "CanESM5-1":
            save_monthly_temp_files(matching, selected_month)
            temp_pattern = base_path + variable + "_*" + model_name + "_piControl_*_temp.nc"
            matching = glob.glob(temp_pattern)
            return load_netcdf_by_month(matching, selected_month)
        return load_netcdf_by_month(matching, selected_month)

    return load_netcdf_files(matching)


# ---------------------------------------------------------------------------
# Cloud / remote data access
# ---------------------------------------------------------------------------

def load_zarr_from_gcs(link: str) -> xr.Dataset:
    """
    Open a Zarr dataset stored on Google Cloud Storage (anonymous access).

    Parameters
    ----------
    link : str
        GCS URI, e.g. ``"gs://bucket/path/to/store.zarr"``.

    Returns
    -------
    xarray.Dataset
    """
    import gcsfs
    gcs = gcsfs.GCSFileSystem(token="anon")
    mapper = gcs.get_mapper(link)
    ds = xr.open_zarr(mapper, consolidated=True)
    return _drop_type_coord(ds)


# ---------------------------------------------------------------------------
# NCAR / intake-ESM catalog access
# ---------------------------------------------------------------------------

def filter_cmip_catalog(cat, **kwargs):
    """
    Search an intake-ESM catalog with the supplied criteria.

    Defaults to ``experiment_id="piControl"`` and ``source_id="GFDL-CM4"``
    when those keys are not provided.

    Parameters
    ----------
    cat : intake_esm.core.esm_datastore
        An open intake-ESM catalog object.
    **kwargs
        Any keyword accepted by ``cat.search()``.  Pass ``None`` to suppress
        a default.

    Returns
    -------
    intake_esm.core.esm_datastore
        Filtered catalog subset.
    """
    kwargs.setdefault("experiment_id", "piControl")
    kwargs.setdefault("source_id", "GFDL-CM4")
    query = {k: v for k, v in kwargs.items() if v is not None}
    return cat.search(**query)


def load_from_ncar_catalog(
    url: str = "/glade/collections/cmip/catalog/intake-esm-datastore/catalogs/glade-cmip6.json",
    show_only: bool = False,
    target_chunks: Optional[Dict[str, int]] = None,
    **kwargs,
) -> Optional[xr.Dataset]:
    """
    Open the NCAR GLADE CMIP6 intake-ESM catalog and return a dataset.

    Parameters
    ----------
    url : str
        Path to the intake-ESM JSON catalog file.
    show_only : bool
        If ``True``, return the filtered catalog DataFrame without loading data.
    target_chunks : dict or None
        Dask chunking specification passed to ``xarray_open_kwargs``.
        Defaults to ``{"time": 120}``.
    **kwargs
        Forwarded to :func:`filter_cmip_catalog`.

    Returns
    -------
    xarray.Dataset or pandas.DataFrame or None
        The dataset, or the catalog DataFrame if *show_only* is ``True``,
        or ``None`` if no data matched the query.
    """
    import intake
    cat = intake.open_esm_datastore(url)
    cat_subset = filter_cmip_catalog(cat, **kwargs)

    if show_only:
        return cat_subset.df

    if cat_subset.df.empty:
        print("No data found for the specified query.")
        return None

    if target_chunks is None:
        target_chunks = {"time": 120}

    dataset_dict = cat_subset.to_dataset_dict(
        xarray_open_kwargs={"chunks": target_chunks}
    )
    key = list(dataset_dict)[0]
    ds = dataset_dict[key]

    # Drop ensemble/initialisation dimensions if present (take first member)
    selection = {
        dim: 0
        for dim in ["member_id", "dcpp_init_year"]
        if dim in ds.dims
    }
    if selection:
        ds = ds.isel(selection, drop=True)

    return ds


# ---------------------------------------------------------------------------
# Dask cluster
# ---------------------------------------------------------------------------

def start_pbs_dask_cluster():
    """
    Start a PBS-backed Dask cluster on NCAR's Casper system.

    The cluster adapts dynamically between 2 and 80 workers.

    Returns
    -------
    client : dask.distributed.Client
    cluster : dask_jobqueue.PBSCluster
    """
    from dask_jobqueue import PBSCluster
    from dask.distributed import Client

    cluster = PBSCluster(
        job_name="dask",
        queue="casper",
        walltime="20:00:00",
        log_directory="dask-logs",
        cores=1,
        memory="8GiB",
        resource_spec="select=1:ncpus=1:mem=8GB",
        processes=1,
        local_directory="${SCRATCH}/dask_scratch/pbs.$PBS_JOBID/dask/spill",
        interface="ext",
        silence_logs="error",
    )
    cluster.adapt(minimum=2, maximum=80)
    client = Client(cluster)
    return client, cluster
