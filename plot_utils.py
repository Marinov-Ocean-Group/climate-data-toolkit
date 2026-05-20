"""
plot_utils.py
=============
Plotting utilities: map helpers, spectrum plots, polynya / convection
multi-panel figures, scatter-with-annotation plots, and colour/marker
palette builders.

Merges the content of the original ``plotfunctions.py`` and
``PlotFunctions2`` files into a single, deduplicated module.
"""

import gc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import colorsys
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

from .io_utils import load_pickle, pickle_exists
from .ice_analysis import detect_polynya
from .spectral import (
    SpectralMethod, to_numpy, calculate_alpha,
    compute_spectrum, red_noise_spectrum_at_freqs,
    significance_threshold, figures_dir,
)

__all__ = [
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
    "add_cbar",
    "add_cbars",
    "add_type_color_legend",
    "annotate_subplot",
    "get_icemax_polynya",
    "add_subplot_icepolynya",
    "plot_ice_hist",
    "plot_polynya_maps",
    "plot_polynya_maps_from_precomputed",
    "plot_convection_maps",
    "plot_by_type",
    "add_text_annotation",
    "style_scatter_axes",
    "plot_corr",
]


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def lighten_color(color, amount: float = 0.5):
    """
    Lighten (or darken) *color* by scaling its luminosity.

    Adapted from https://stackoverflow.com/a/49601444.

    Parameters
    ----------
    color : str or RGB tuple
        Any Matplotlib-compatible colour specification.
    amount : float
        Scaling factor.  < 1 darkens; > 1 lightens.

    Returns
    -------
    tuple
        RGB tuple.

    Examples
    --------
    >>> lighten_color("g", 0.3)
    >>> lighten_color("#F034A3", 0.6)
    >>> lighten_color((.3, .55, .1), 0.5)
    """
    c = np.array(colorsys.rgb_to_hls(*mcolors.to_rgb(color)))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


# ---------------------------------------------------------------------------
# Map helpers
# ---------------------------------------------------------------------------

def create_circle() -> mpath.Path:
    """Return a unit-circle Path for clipping polar stereographic axes."""
    theta = np.linspace(0, 2 * np.pi, 100)
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    return mpath.Path(verts * 0.5 + 0.5)


def style_south_polar_map(
    ax,
    grid: bool = True,
    grid_labels: bool = True,
) -> None:
    """
    Apply standard Southern Ocean styling to a polar stereographic axes.

    Sets the circular boundary, adds land/ocean features, and optionally
    draws gridlines.

    Parameters
    ----------
    ax : cartopy GeoAxes
    grid : bool
        Draw gridlines.
    grid_labels : bool
        Annotate gridlines with latitude labels.
    """
    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, zorder=1, color="grey")
    ax.add_feature(cfeature.OCEAN, alpha=0.15)
    ax.set_boundary(create_circle(), transform=ax.transAxes)
    ax.spines["geo"].set_edgecolor(None)
    if grid:
        ax.gridlines(
            draw_labels=grid_labels,
            ylocs=np.linspace(-90, 90, 7),
            color="grey",
            linestyle="-.",
            linewidth=0.5,
            alpha=0.8,
        )


def plot_global_map(da, **kwargs):
    """
    Quick global Robinson-projection map of an xarray DataArray.

    Parameters
    ----------
    da : xr.DataArray
    **kwargs
        Forwarded to ``da.plot()``.

    Returns
    -------
    fig, ax
    """
    fig = plt.figure(figsize=(12, 4))
    ax = plt.axes(projection=ccrs.Robinson())
    da.plot(ax=ax, transform=ccrs.PlateCarree(), **kwargs)
    ax.coastlines()
    return fig, ax


# ---------------------------------------------------------------------------
# Spectrum plot
# ---------------------------------------------------------------------------

def plot_spectrum(
    da,
    plot_title: str,
    method: SpectralMethod = "fft",
) -> None:
    """
    Compute and save a power-spectrum plot with red-noise confidence bounds.

    The figure is written to ``<figures_dir(method)>/<plot_title>.png``.

    Parameters
    ----------
    da : xr.DataArray or array-like
        1-D time series.
    plot_title : str
        Figure title and output filename stem.
    method : "fft" or "welch"
    """
    out_dir = figures_dir(method)
    out_dir.mkdir(parents=True, exist_ok=True)

    values = to_numpy(da)
    alpha = calculate_alpha(values)
    spec = compute_spectrum(values, method=method)

    rspec = red_noise_spectrum_at_freqs(spec.freqs, alpha)
    rspec_norm = rspec / rspec.sum()
    spec99 = significance_threshold(rspec_norm, 0.99, dof_signal=spec.dof_signal)
    spec90 = significance_threshold(rspec_norm, 0.90, dof_signal=spec.dof_signal)

    mask = np.isfinite(spec.periods)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlabel("Period (years)", fontsize=11)
    ax.set_ylabel("Normalised Power", fontsize=11)
    method_label = "Welch" if method == "welch" else "FFT"
    ax.set_title(f"{plot_title}  [{method_label}]", fontsize=12)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))

    finite_periods = spec.periods[mask & (spec.periods > 1)]
    if finite_periods.size:
        ax.set_xlim(finite_periods.min(), finite_periods.max())

    ax.plot(spec.periods[mask], spec.power[mask],    "-k",  lw=1.5, label="data")
    ax.plot(spec.periods[mask], rspec_norm[mask],    "-",   lw=1.2, label="red-noise fit",   color="red")
    ax.plot(spec.periods[mask], spec99[mask],        "--",  lw=1.0, label="99 % confidence", color="steelblue")
    ax.plot(spec.periods[mask], spec90[mask],        "-.",  lw=1.0, label="90 % confidence", color="orange")

    ax.legend(
        bbox_to_anchor=(0.0, -0.02, 1.0, 0.102),
        loc="lower left",
        frameon=False,
        ncols=4,
        mode="expand",
        borderaxespad=0.0,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"{plot_title}.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sea-ice area statistics (used in ice histogram plots)
# ---------------------------------------------------------------------------

def calculate_mean_ice(
    ice: "xr.DataArray",
    area: "xr.DataArray",
) -> "xr.DataArray":
    """Area-weighted mean sea-ice concentration over all non-negative pixels."""
    return (ice * area).sum() / area.where(ice >= 0).sum()


def calculate_persistent_mean_ice(
    ice: "xr.DataArray",
    area: "xr.DataArray",
) -> "xr.DataArray":
    """
    Area-weighted mean SIC considering only pixels where ice ever exists,
    normalised by the time length.
    """
    ice_max = ice.max("time")
    return (ice.where(ice_max > 0) * area).sum() / area.where(ice_max > 0).sum() / len(ice.time)


# ---------------------------------------------------------------------------
# Palette / style builders
# ---------------------------------------------------------------------------

def build_color_map(
    color_palette: list,
    categories: list,
    lighten: float = 1.0,
) -> dict:
    """
    Map *categories* to colours from *color_palette*, optionally lightened.

    Parameters
    ----------
    color_palette : list
        List of Matplotlib colours (must be at least as long as *categories*).
    categories : list
        Unique category labels.
    lighten : float
        Passed to :func:`lighten_color`.  1.0 = no change.

    Returns
    -------
    dict
        ``{category: colour}``

    Raises
    ------
    ValueError
        If *color_palette* is shorter than *categories*.
    """
    if len(color_palette) < len(categories):
        raise ValueError("color_palette has fewer entries than categories.")
    return {
        cat: lighten_color(color_palette[i], lighten)
        for i, cat in enumerate(categories)
    }


def build_marker_map(marker_list: list, categories: list) -> dict:
    """
    Map *categories* to markers from *marker_list*.

    Parameters
    ----------
    marker_list : list
        Matplotlib marker strings.
    categories : list
        Unique category labels.

    Returns
    -------
    dict
        ``{category: marker}``

    Raises
    ------
    ValueError
        If *marker_list* is shorter than *categories*.
    """
    if len(marker_list) < len(categories):
        raise ValueError("marker_list is shorter than categories.")
    return {cat: marker_list[i] for i, cat in enumerate(categories)}


# ---------------------------------------------------------------------------
# Colorbar / legend helpers
# ---------------------------------------------------------------------------

def _percent_formatter(x, pos):
    return f"{x * 100:.0f}%"


def add_cbar(
    fig,
    axes_loc: list,
    im,
    label_text: str,
    format_percent: bool = False,
) -> None:
    """
    Add a horizontal colorbar to *fig* at the specified axes location.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    axes_loc : [left, bottom, width, height]
    im : mappable
    label_text : str
    format_percent : bool
        If True, tick labels are shown as percentages.
    """
    cbar_ax = fig.add_axes(axes_loc)
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(label_text, size=8, labelpad=-0.1)
    cbar.ax.tick_params(labelsize=6, direction="in")
    cbar.outline.set_visible(False)
    if format_percent:
        cbar.formatter = FuncFormatter(_percent_formatter)
        cbar.update_ticks()


def add_cbars(fig, im_ice, im_polynya) -> None:
    """
    Add the standard two-colorbar layout for ice-concentration + polynya
    frequency panels.
    """
    add_cbar(fig, [0.62, 0.05, 0.35, 0.01], im_ice, "Sea ice concentration (%)")
    add_cbar(fig, [0.62, 0.11, 0.35, 0.01], im_polynya, "Frequency of occurrence",
             format_percent=True)


def add_type_color_legend(
    fig,
    color_dict: dict,
    legend_title: str,
    bbox_to_anchor: tuple = (0.55, 0.135),
    n_cols: int = 3,
    fontsize: int = 6,
) -> None:
    """
    Add a legend of colour-coded category rectangles to *fig*.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    color_dict : dict
        ``{category: colour}`` mapping.
    legend_title : str
    bbox_to_anchor : tuple
    n_cols : int
    fontsize : int
    """
    categories = list(color_dict.keys())
    proxies = [Rectangle((0, 0), 1, 1, color=color_dict[c]) for c in categories]
    fig.legend(
        proxies, categories,
        title=legend_title,
        frameon=False,
        bbox_to_anchor=bbox_to_anchor,
        ncols=n_cols,
        fontsize=fontsize,
    )


# ---------------------------------------------------------------------------
# Subplot annotation
# ---------------------------------------------------------------------------

def annotate_subplot(
    ax,
    model_name: str,
    panel_number: int,
    time_length: int,
    resolution_text,
) -> None:
    """
    Add title, time-length label, resolution text, and panel number to a
    polar-map subplot.

    Parameters
    ----------
    ax : cartopy GeoAxes
    model_name : str
    panel_number : int
    time_length : int
        Number of time steps (displayed at the South Pole).
    resolution_text : str or None
        Resolution label displayed near the boundary.
    """
    ax.set_title(model_name, fontsize=6, pad=-0.5)
    ax.text(0, -90, time_length, fontsize=6, color="w", ha="center")
    ax.text(180, -55, resolution_text, transform=ccrs.PlateCarree(),
            fontsize=6, color="k", ha="center")
    ax.text(45, -47, panel_number, transform=ccrs.PlateCarree(),
            fontsize=6, color="k", ha="center")


def add_color_band(fig, ax, color, title_only: bool = True) -> None:
    """
    Draw a coloured rectangle behind or above a subplot to indicate its
    category.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    color : Matplotlib colour
    title_only : bool
        If True, draw only a narrow band above the title; otherwise shade
        the full subplot border.
    """
    bbox = ax.get_position()
    if title_only:
        rect = Rectangle(
            (bbox.x0 - 0.002, bbox.y1 + 0.001),
            bbox.width + 0.004, 0.012,
            fill=True, color=color, alpha=1, zorder=-1,
            transform=fig.transFigure, clip_on=False,
        )
    else:
        rect = Rectangle(
            (bbox.x0 - 0.002367, bbox.y0),
            bbox.width + 0.004734, bbox.height + 0.02,
            fill=True, color=color, alpha=1, zorder=-1,
            transform=fig.transFigure, clip_on=False,
        )
    fig.add_artist(rect)


# ---------------------------------------------------------------------------
# Polynya panel helpers
# ---------------------------------------------------------------------------

def get_icemax_polynya(
    model_name: str,
    pickle_path: str,
    ice_threshold: float,
):
    """
    Load precomputed ice data from a pickle and compute polynya frequency.

    Parameters
    ----------
    model_name : str
    pickle_path : str
        Base path for pickle files.
    ice_threshold : float
        SIC threshold passed to :func:`~ice_analysis.detect_polynya`.

    Returns
    -------
    pltx, plty, plt_icemax, plt_polynya, time_length
    """
    ds = load_pickle(model_name, pickle_path)
    plt_icemax = ds.siconc.max("time")
    pltx = ds.newlon
    plty = ds.newlat

    flood_points = [(0, 0)]
    if model_name == "MRI-ESM2-0":
        flood_points = [(0, 0), (0, 40), (0, 100), (0, 200)]

    mask = detect_polynya(ds.siconc, ds.areacello, ice_threshold,
                          flood_points=flood_points)
    count = mask.count("time")
    plt_polynya = count.where(count > 0) / len(mask.time)
    return pltx, plty, plt_icemax, plt_polynya, len(ds.time)


def add_subplot_icepolynya(
    fig,
    panel_number: int,
    model_name: str,
    color,
    title_only: bool,
    resolution_text,
    pltx,
    plty,
    plt_icemax,
    plt_polynya,
    time_length: int,
):
    """
    Add one ice-concentration + polynya-frequency panel to *fig*.

    Returns the two pcolormesh mappables for colorbar construction.
    """
    import cmocean
    ax = fig.add_subplot(7, 8, panel_number, projection=ccrs.SouthPolarStereo())
    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99,
                        wspace=0.04, hspace=0.04)
    add_color_band(fig, ax, color, title_only)
    style_south_polar_map(ax, grid=False, grid_labels=False)
    annotate_subplot(ax, model_name, panel_number, time_length, resolution_text)

    im = ax.pcolormesh(pltx, plty, plt_icemax,
                       vmin=0, vmax=100,
                       transform=ccrs.PlateCarree(),
                       cmap=cmocean.cm.ice)
    im2 = ax.pcolormesh(pltx, plty, plt_polynya,
                        vmin=0, vmax=1,
                        transform=ccrs.PlateCarree(),
                        cmap=plt.cm.plasma,
                        alpha=0.6)
    return im, im2


# ---------------------------------------------------------------------------
# Multi-panel figure functions
# ---------------------------------------------------------------------------

def plot_ice_hist(
    model_df,
    pickle_path_count: str,
    pickle_path_ice: str,
    save_path: str,
    figsize: tuple = (6.5, 7),
) -> None:
    """
    Multi-panel sea-ice concentration histogram figure.

    One panel per model showing polynya area vs. threshold alongside the SIC
    distribution.

    Parameters
    ----------
    model_df : pandas.DataFrame
        Must contain a ``source_id`` column.
    pickle_path_count : str
        Base path for polynya-count pickle files.
    pickle_path_ice : str
        Base path for sea-ice pickle files.
    save_path : str
        Output file path (PDF recommended).
    figsize : tuple
    """
    fig = plt.figure(figsize=figsize)
    n = 0
    ice_thresholds = np.arange(0, 100, step=1)

    for i in range(len(model_df)):
        name = model_df.at[i, "source_id"]
        if not (pickle_exists(name, pickle_path_count) and
                pickle_exists(name, pickle_path_ice)):
            continue

        n += 1
        count_data = np.array(load_pickle(name, pickle_path_count))
        ds_ice = load_pickle(name, pickle_path_ice)
        ice_mean = calculate_mean_ice(ds_ice.siconc, ds_ice.areacello)
        ice_mean_not0 = calculate_persistent_mean_ice(ds_ice.siconc, ds_ice.areacello)

        ax = fig.add_subplot(10, 5, n)
        plt.subplots_adjust(left=0.015, bottom=0.055, right=0.985, top=0.99,
                            wspace=0.08, hspace=0.08)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-1, 101)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.tick_params(labelsize=6)
        ax.set_title(f"{name}, {np.max(count_data[1:]) / 1e12:.1f}",
                     fontsize=6, y=1.0, pad=-4)

        ax2 = ax.twinx()
        ice_vals = ds_ice.siconc.values.flatten()
        ice_vals = ice_vals[ice_vals > 0]

        ax.plot(ice_thresholds, count_data[:, 0] / np.max(count_data),
                lw=1, color="b", label="polynya area")
        ax2.axvline(x=float(ice_mean.values), color="g", ls="--",
                    lw=1, label="mean SIC in SO")
        ax.axvline(x=float(ice_mean_not0.values), color="orange", ls="-.",
                   lw=1, label="mean SIC within SIE")
        ax2.hist(ice_vals, bins=100, color="red", edgecolor=None,
                 alpha=0.6, label="sea ice concentration")

        if n <= 44:
            ax.set_xticklabels([])
        if n == 48:
            ax.set_xlabel("ice concentration (%)", fontsize=10)

        for _ax in (ax, ax2):
            _ax.set_yticklabels([])
            _ax.set_yticks([])
            _ax.set_frame_on(False)
            _ax.tick_params(tick1On=False)

        ax.xaxis.grid(True, "major", ls="-", lw=0.2, alpha=0.6)

        if n == 49:
            ax.legend(fontsize=6, frameon=False, loc="center left",
                      bbox_to_anchor=(1.05, 0.4))
            ax2.legend(fontsize=6, frameon=False, loc="center left",
                       bbox_to_anchor=(1.05, 0))

    fig.savefig(save_path, format="pdf")


def plot_polynya_maps(
    model_df,
    pickle_path_ice: str,
    color_palette: list,
    ice_threshold: float,
    type_column: str = "type_ice",
    title_only: bool = True,
    show_resolution: bool = False,
    lighten: float = 1.0,
    figsize: tuple = (6.5, 7.5),
    save_prefix: str = "Figures/polynya_maps_",
) -> None:
    """
    Multi-panel Southern Ocean map figure showing sea-ice max and polynya
    frequency for each model.

    Polynya detection is run on-the-fly from the raw ice data.

    Parameters
    ----------
    model_df : pandas.DataFrame
    pickle_path_ice : str
    color_palette : list
    ice_threshold : float
    type_column : str
    title_only : bool
    show_resolution : bool
    lighten : float
    figsize : tuple
    save_prefix : str
    """
    fig = plt.figure(figsize=figsize)
    n = 0
    color_dict = build_color_map(color_palette, list(model_df[type_column].unique()), lighten)

    for i in range(len(model_df)):
        name = model_df.at[i, "source_id"]
        category = model_df.at[i, type_column]
        pltx, plty, plt_icemax, plt_polynya, time_length = get_icemax_polynya(
            name, pickle_path_ice, ice_threshold)
        n += 1
        reso_text = str(model_df.at[i, "resolution"]) if show_resolution else None
        im, im2 = add_subplot_icepolynya(
            fig, n, name, color_dict[category], title_only,
            reso_text, pltx, plty, plt_icemax, plt_polynya, time_length)
        gc.collect()

    add_cbars(fig, im, im2)
    add_type_color_legend(fig, color_dict, "sea ice module types")
    fig.text(0.15, 0.06, f"{ice_threshold}%")
    fig.savefig(f"{save_prefix}{ice_threshold}.png", dpi=300)


def plot_polynya_maps_from_precomputed(
    model_df,
    pickle_path_ice: str,
    pickle_path_polynya: str,
    color_palette: list,
    type_column: str = "type_ice",
    title_only: bool = True,
    show_resolution: bool = False,
    lighten: float = 1.0,
    figsize: tuple = (6.5, 7),
    save_path: str = "polynya_maps.png",
) -> None:
    """
    Multi-panel polynya map figure using pre-detected polynya data loaded
    from pickle files (faster than re-running detection).

    Parameters
    ----------
    model_df : pandas.DataFrame
    pickle_path_ice : str
    pickle_path_polynya : str
    color_palette : list
    type_column : str
    title_only : bool
    show_resolution : bool
    lighten : float
    figsize : tuple
    save_path : str
    """
    import xarray as xr
    fig = plt.figure(figsize=figsize)
    n = 0
    color_dict = build_color_map(color_palette, list(model_df[type_column].unique()), lighten)

    for i in range(len(model_df)):
        name = model_df.at[i, "source_id"]
        category = model_df.at[i, type_column]

        ds_ice = load_pickle(name, pickle_path_ice)
        ds_polynya = load_pickle(name, pickle_path_polynya)

        pltx = ds_ice.newlon
        plty = ds_ice.newlat
        plt_icemax = ds_ice.siconc.max("time")
        count = ds_polynya.where(ds_polynya > 0).count("time")
        time_length = len(ds_polynya.time)
        plt_polynya = count.where(count > 0) / time_length

        n += 1
        reso_text = str(model_df.at[i, "resolution"]) if show_resolution else None
        im, im2 = add_subplot_icepolynya(
            fig, n, name, color_dict[category], title_only,
            reso_text, pltx, plty, plt_icemax, plt_polynya, time_length)
        gc.collect()

    add_cbars(fig, im, im2)
    add_type_color_legend(fig, color_dict, "sea ice module types")
    fig.savefig(save_path, dpi=300)


def plot_convection_maps(
    model_df,
    pickle_path_mlotst: str,
    pickle_path_mld: str,
    color_palette: list,
    open_mld_fn,
    type_column: str = "type",
    title_only: bool = True,
    show_resolution: bool = True,
    lighten: float = 1.0,
    figsize: tuple = (6.5, 7),
    save_path: str = "convection_maps.png",
) -> None:
    """
    Multi-panel deep-convection frequency map figure.

    Parameters
    ----------
    model_df : pandas.DataFrame
    pickle_path_mlotst, pickle_path_mld : str
    color_palette : list
    open_mld_fn : callable
        Function ``open_mld(path_mlotst, path_mld, model_name)`` that
        returns ``(da_mld, ds_mld)``.
    type_column : str
    title_only : bool
    show_resolution : bool
    lighten : float
    figsize : tuple
    save_path : str
    """
    import cmocean
    fig = plt.figure(figsize=figsize)
    n = 0
    color_dict = build_color_map(color_palette, list(model_df[type_column].unique()), lighten)

    for i in range(len(model_df)):
        name = model_df.at[i, "source_id"]
        category = model_df.at[i, type_column]

        da_mld, ds_mld = open_mld_fn(pickle_path_mlotst, pickle_path_mld, name)
        if len(da_mld.time) > 500:
            da_mld = da_mld.isel(time=slice(-500, None))

        pltx = ds_mld.newlon
        plty = ds_mld.newlat
        count = da_mld.where(da_mld >= 2000).count("time")
        time_length = len(da_mld.time)
        plt_mld = count.where(count > 0) / time_length

        n += 1
        reso_text = str(model_df.at[i, "resolution"]) if show_resolution else None

        ax = fig.add_subplot(7, 8, n, projection=ccrs.SouthPolarStereo())
        plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99,
                            wspace=0.04, hspace=0.04)
        style_south_polar_map(ax, grid=False, grid_labels=False)
        add_color_band(fig, ax, color_dict[category], title_only)
        annotate_subplot(ax, name, n, time_length, reso_text)
        im = ax.pcolormesh(pltx, plty, plt_mld,
                           vmin=0, vmax=0.2,
                           transform=ccrs.PlateCarree(),
                           cmap=plt.cm.plasma)
        gc.collect()

    add_cbar(fig, [0.62, 0.08, 0.35, 0.01], im, "Frequency of occurrence",
             format_percent=True)
    add_type_color_legend(fig, color_dict, "ocean module types")
    fig.savefig(save_path, dpi=300)


# ---------------------------------------------------------------------------
# Scatter / correlation plots
# ---------------------------------------------------------------------------

def plot_by_type(
    ax,
    df_plot,
    marker_dict: dict,
    color_dict: dict,
    marker_size: int = 6,
    x_col: int = -2,
    y_col: int = -1,
) -> None:
    """
    Scatter plot of *df_plot* coloured and marked by model type.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    df_plot : pandas.DataFrame
        Must contain a ``type`` column.
    marker_dict, color_dict : dict
    marker_size : int
    x_col, y_col : int
        Column indices for the x and y variables (default: last two).
    """
    var1 = df_plot.columns[x_col]
    var2 = df_plot.columns[y_col]
    for mtype in df_plot["type"].unique():
        sub = df_plot.loc[df_plot["type"] == mtype]
        ax.plot(
            sub[var1], sub[var2],
            marker=marker_dict[mtype],
            markerfacecolor=lighten_color(color_dict[mtype], 1.5),
            markeredgecolor=lighten_color(color_dict[mtype], 2),
            linestyle="",
            ms=marker_size,
            label=mtype,
        )


def add_text_annotation(
    ax,
    df_plot,
    label_column: str = "name",
    fontsize: int = 6,
    expand_rate: float = 2.0,
    x_col: int = -2,
    y_col: int = -1,
) -> None:
    """
    Add non-overlapping text annotations to a scatter plot using adjustText.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    df_plot : pandas.DataFrame
    label_column : str
        Column to use for annotation text.
    fontsize : int
    expand_rate : float
        adjustText expansion factor.
    x_col, y_col : int
        Column indices for x and y positions.
    """
    from adjustText import adjust_text

    var1 = df_plot.columns[x_col]
    var2 = df_plot.columns[y_col]

    texts, xs, ys = [], [], []
    for ind in df_plot.index:
        texts.append(ax.text(df_plot[var1][ind], df_plot[var2][ind],
                             df_plot[label_column][ind],
                             fontsize=fontsize, alpha=0.6))
        xs.append(df_plot[var1][ind])
        ys.append(df_plot[var2][ind])

    adjust_text(
        texts, xs, ys, ax=ax, avoid_self=True,
        expand=(expand_rate, expand_rate),
        time_lim=1,
        arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5),
    )


def style_scatter_axes(
    ax,
    xlabel: str,
    ylabel: str,
    equal_aspect: bool = False,
    fontsize: int = 8,
) -> None:
    """
    Apply standard styling to a scatter / correlation axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    xlabel, ylabel : str
    equal_aspect : bool
        If True, force equal x/y scaling.
    fontsize : int
    """
    ax.grid(which="both", axis="both", alpha=0.5)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(axis="both", which="both", labelsize=fontsize)
    if equal_aspect:
        ax.set_aspect("equal")
        ax.set_box_aspect(1)


def plot_corr(
    df_plot,
    marker_dict: dict,
    color_dict: dict,
    vline: bool = False,
    hline: bool = False,
    label_column: str = "name",
    equal_aspect: bool = False,
    save_dir: str = "Fig_corr/",
) -> None:
    """
    Create, annotate, and save a correlation scatter plot.

    Parameters
    ----------
    df_plot : pandas.DataFrame
        The last two columns are used as x and y.
    marker_dict, color_dict : dict
    vline, hline : bool
        Draw vertical / horizontal reference lines at zero.
    label_column : str
    equal_aspect : bool
    save_dir : str
    """
    fig, ax = plt.subplots()
    if vline:
        ax.axvline(c="grey", lw=1)
    if hline:
        ax.axhline(c="grey", lw=1)

    plot_by_type(ax, df_plot, marker_dict, color_dict)
    add_text_annotation(ax, df_plot, label_column=label_column)
    style_scatter_axes(ax,
                       xlabel=df_plot.columns[-2],
                       ylabel=df_plot.columns[-1],
                       equal_aspect=equal_aspect)
    ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")

    save_name = save_dir + df_plot.columns[-2] + "_vs_" + df_plot.columns[-1] + ".png"
    plt.tight_layout()
    fig.savefig(save_name, dpi=150)
