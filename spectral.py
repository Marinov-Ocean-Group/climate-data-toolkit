"""
spectral.py
===========
Spectral analysis, significance testing, signal filtering, and
cross-lagged correlation.

Public API
----------
- to_numpy               — safe DataArray → 1-D numpy conversion
- normalize_data         — zero-mean, unit-variance normalisation
- create_index           — build a climate index from a heat-content array
- calculate_alpha        — lag-1 autocorrelation (red-noise parameter)
- red_noise_spectrum_at_freqs — theoretical Gilman et al. red-noise spectrum
- compute_spectrum        — dispatcher: FFT or Welch spectrum
- significance_threshold  — F-test power threshold
- sig_marker             — significance star string
- format_period          — formatted period label with significance stars
- calculate_dominant_periods — top-N spectral peaks with significance
- apply_butterworth_filter   — zero-phase Butterworth filter for DataArrays
- compute_lagged_correlation — cross-lagged correlation map
- figures_dir            — output directory helper
- output_csv             — output CSV path helper
"""

import numpy as np
import scipy.fft as fft
import scipy.signal as signal
import scipy.stats as stats
import xarray as xr
from typing import NamedTuple

from .constants import (
    SpectralMethod,
    SKIP_FREQS,
    SIGNIFICANCE_DOF_RED,
    WELCH_NPERSEG_FRACTION,
    SIG_LEVELS,
    FIGURES_DIR_ROOT,
    CSV_OUTPUT_TEMPLATE,
)

__all__ = [
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
]


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def to_numpy(da) -> np.ndarray:
    """
    Convert an xarray DataArray **or** plain array to a clean 1-D numpy array.

    Squeezes size-1 dimensions and drops NaN / infinite values.

    Parameters
    ----------
    da : xr.DataArray or array-like

    Returns
    -------
    np.ndarray
        1-D float array with no NaNs.

    Raises
    ------
    ValueError
        If the result is not 1-D after squeezing.
    """
    values = da.values if isinstance(da, xr.DataArray) else np.asarray(da)
    values = values.squeeze()
    if values.ndim != 1:
        raise ValueError(
            f"Expected a 1-D time series after squeezing; got shape {values.shape}."
        )
    return values[np.isfinite(values)]


def normalize_data(data: np.ndarray) -> np.ndarray:
    """
    Standardise *data* to zero mean and unit variance (ignoring NaNs).

    Parameters
    ----------
    data : array-like

    Returns
    -------
    np.ndarray
    """
    return (data - np.nanmean(data)) / np.nanstd(data)


def create_index(
    heat_content: xr.DataArray,
    rename_time_dim: str | None = "time",
) -> xr.DataArray:
    """
    Build a climate index from a heat-content DataArray.

    The index is the *negative* normalised heat content (sign convention:
    positive index ↔ heat content anomaly below average).

    Parameters
    ----------
    heat_content : xr.DataArray
        1-D heat-content time series.
    rename_time_dim : str or None
        If given, rename the first dimension to this string.

    Returns
    -------
    xr.DataArray
    """
    index = -normalize_data(heat_content)
    if rename_time_dim:
        index = index.rename({index.dims[0]: rename_time_dim})
    return index


# ---------------------------------------------------------------------------
# Red-noise background
# ---------------------------------------------------------------------------

def calculate_alpha(values: np.ndarray) -> float:
    """
    Estimate the lag-1 autocorrelation coefficient (red-noise parameter α).

    Parameters
    ----------
    values : np.ndarray
        1-D time series (no NaNs).

    Returns
    -------
    float
        α ∈ [-1, 1].  Returns 0.0 for a constant series.
    """
    std = np.std(values)
    if std == 0:
        return 0.0
    normalised = (values - np.mean(values)) / std
    return float(np.corrcoef(normalised[:-1], normalised[1:])[0, 1])


def red_noise_spectrum_at_freqs(freqs: np.ndarray, alpha: float) -> np.ndarray:
    """
    Theoretical red-noise power spectrum (Gilman et al. 1963).

    Evaluated directly at the supplied normalised frequencies (0–0.5) so
    that FFT and Welch curves are on identical scales.

    Parameters
    ----------
    freqs : np.ndarray
        Normalised frequencies in [0, 0.5].
    alpha : float
        Lag-1 autocorrelation coefficient.

    Returns
    -------
    np.ndarray
        Unnormalised red-noise power at each frequency.
    """
    return (1.0 - alpha ** 2) / (
        1.0 - 2.0 * alpha * np.cos(2.0 * np.pi * freqs) + alpha ** 2
    )


# ---------------------------------------------------------------------------
# Spectrum computation
# ---------------------------------------------------------------------------

class SpectrumResult(NamedTuple):
    """Container returned by all spectrum-computation functions."""
    freqs:      np.ndarray  #: Normalised frequencies (0 … 0.5)
    periods:    np.ndarray  #: 1/freq; DC mapped to inf
    power:      np.ndarray  #: Normalised power (sums to 1)
    dof_signal: int         #: DOF per bin (2 for FFT, 2K for Welch)


def _freqs_to_periods(freqs: np.ndarray) -> np.ndarray:
    """Convert normalised frequencies to periods; DC → inf."""
    with np.errstate(divide="ignore"):
        return np.where(freqs == 0, np.inf, 1.0 / freqs)


def compute_fft_spectrum(values: np.ndarray) -> SpectrumResult:
    """
    Normalised one-sided power spectrum via plain FFT.

    Each frequency bin has 2 degrees of freedom (one complex coefficient →
    chi-squared with 2 DOF).

    Parameters
    ----------
    values : np.ndarray
        Mean-removed time series.

    Returns
    -------
    SpectrumResult
    """
    T = len(values)
    freqs = fft.rfftfreq(T, d=1)
    xf = fft.rfft(values - values.mean())
    power = (2.0 / T * xf * xf.conj()).real
    power = power / power.sum()
    return SpectrumResult(freqs, _freqs_to_periods(freqs), power, dof_signal=2)


def compute_welch_spectrum(values: np.ndarray) -> SpectrumResult:
    """
    Normalised one-sided power spectrum via Welch's method.

    Uses a Hann window, segment length = ``WELCH_NPERSEG_FRACTION × T``,
    and 50 % overlap.  The DOF per bin scales as ``2 × n_segments``.

    Parameters
    ----------
    values : np.ndarray
        Mean-removed time series.

    Returns
    -------
    SpectrumResult
    """
    T = len(values)
    nperseg = max(4, int(T * WELCH_NPERSEG_FRACTION))
    noverlap = nperseg // 2
    step = nperseg - noverlap
    n_seg = max(1, 1 + (T - nperseg) // step)

    freqs, power = signal.welch(
        values - values.mean(),
        fs=1.0,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="spectrum",
    )
    power = power / power.sum()
    return SpectrumResult(freqs, _freqs_to_periods(freqs), power, dof_signal=2 * n_seg)


def compute_spectrum(
    values: np.ndarray,
    method: SpectralMethod = "fft",
) -> SpectrumResult:
    """
    Compute the normalised power spectrum using the specified method.

    Parameters
    ----------
    values : np.ndarray
    method : "fft" or "welch"

    Returns
    -------
    SpectrumResult
    """
    if method == "welch":
        return compute_welch_spectrum(values)
    return compute_fft_spectrum(values)


# ---------------------------------------------------------------------------
# Significance testing
# ---------------------------------------------------------------------------

def significance_threshold(
    rspec_normalised: np.ndarray,
    confidence_level: float,
    dof_signal: int = 2,
) -> np.ndarray:
    """
    Power threshold for a given confidence level (F-test, Gilman et al.).

    Parameters
    ----------
    rspec_normalised : np.ndarray
        Normalised red-noise background spectrum (sums to 1).
    confidence_level : float
        Confidence level, e.g. ``0.99``.
    dof_signal : int
        Degrees of freedom for the signal (2 for FFT, 2K for Welch).

    Returns
    -------
    np.ndarray
        Power threshold at each frequency.
    """
    f_crit = stats.f.ppf(confidence_level, dof_signal, SIGNIFICANCE_DOF_RED)
    return f_crit * rspec_normalised


def sig_marker(significance: float) -> str:
    """
    Return a star string for a given significance value.

    ★★★ ≥ 0.99  |  ★★ ≥ 0.95  |  ★ ≥ 0.90  |  (empty) otherwise

    Parameters
    ----------
    significance : float

    Returns
    -------
    str
        One of ``"***"``, ``"**"``, ``"*"``, or ``""``.
    """
    if not np.isfinite(significance):
        return ""
    for threshold, marker in SIG_LEVELS:
        if significance >= threshold:
            return marker
    return ""


def format_period(period: float, significance: float, decimals: int = 1) -> str:
    """
    Format a period value with its significance star marker.

    Returns an empty string when *period* is NaN.

    Parameters
    ----------
    period : float
    significance : float
    decimals : int
        Decimal places in the formatted period string.

    Returns
    -------
    str
        E.g. ``"50.0***"`` or ``""``.
    """
    if not np.isfinite(period):
        return ""
    return f"{period:.{decimals}f}{sig_marker(significance)}"


# ---------------------------------------------------------------------------
# Dominant period detection
# ---------------------------------------------------------------------------

def calculate_dominant_periods(
    da,
    method: SpectralMethod = "fft",
    n_peaks: int = 3,
) -> list[tuple[float, float]]:
    """
    Return the *n_peaks* most dominant spectral periods and their significance.

    Local maxima are required so that adjacent bins of the same broad peak
    are not all returned as separate entries.  The first ``SKIP_FREQS`` bins
    are excluded to avoid DC artefacts.

    Parameters
    ----------
    da : xr.DataArray or array-like
    method : "fft" or "welch"
    n_peaks : int
        Number of peaks to return.

    Returns
    -------
    list of (period, significance) tuples
        Length *n_peaks*.  Entries are ``(NaN, NaN)`` when no valid peak
        is available.
    """
    nan_result = [(np.nan, np.nan)] * n_peaks

    values = to_numpy(da)
    if np.all(values == values[0]):
        return nan_result

    alpha = calculate_alpha(values)
    spec = compute_spectrum(values, method=method)

    rspec = red_noise_spectrum_at_freqs(spec.freqs, alpha)
    rspec_norm = rspec / rspec.sum()

    power_t   = spec.power[SKIP_FREQS:]
    periods_t = spec.periods[SKIP_FREQS:]
    rspec_t   = rspec_norm[SKIP_FREQS:]

    from scipy.signal import argrelmax
    (local_max_idx,) = argrelmax(power_t, order=1)
    if len(local_max_idx) < n_peaks:
        local_max_idx = np.argsort(power_t)[::-1]

    top_idx = local_max_idx[np.argsort(power_t[local_max_idx])[::-1][:n_peaks]]

    results = []
    for idx in top_idx:
        period = float(periods_t[idx])
        sig = float(stats.f.cdf(
            power_t[idx] / rspec_t[idx],
            dfn=spec.dof_signal,
            dfd=SIGNIFICANCE_DOF_RED,
        ))
        results.append((period, sig))

    while len(results) < n_peaks:
        results.append((np.nan, np.nan))

    return results


# ---------------------------------------------------------------------------
# Butterworth filtering
# ---------------------------------------------------------------------------

def _filter_along_axis(
    arr: np.ndarray,
    filt_obj,
    method: str,
    axis: int,
) -> np.ndarray:
    """
    Apply the chosen Butterworth filter along a single axis, interpolating
    over NaNs before filtering and restoring them afterwards.
    """
    out = np.empty_like(arr, dtype=float)

    for idx in np.ndindex(*[s for i, s in enumerate(arr.shape) if i != axis]):
        sl = tuple(
            idx[i if i < axis else i - 1] if i != axis else slice(None)
            for i in range(arr.ndim)
        )
        y = arr[sl].astype(float)
        nan_mask = np.isnan(y)

        if nan_mask.all():
            out[sl] = np.nan
            continue
        if nan_mask.any():
            xp = np.arange(len(y))
            y = np.interp(xp, xp[~nan_mask], y[~nan_mask])

        if method == "sos":
            filtered = signal.sosfiltfilt(filt_obj, y)
        else:  # "pad" or "gust"
            b, a = filt_obj
            filtered = signal.filtfilt(b, a, y, method=method)

        if nan_mask.any():
            filtered[nan_mask] = np.nan

        out[sl] = filtered

    return out


def apply_butterworth_filter(
    da: xr.DataArray,
    cutoff: float,
    dim: str | list[str] = "time",
    sample_rate: float = 1.0,
    poles: int = 4,
    btype: str = "lowpass",
    method: str = "sos",
) -> xr.DataArray:
    """
    Apply a zero-phase Butterworth filter to an xarray DataArray.

    Parameters
    ----------
    da : xr.DataArray
    cutoff : float or [float, float]
        Cutoff frequency (or [low, high] for bandpass/bandstop).
    dim : str or list of str
        Dimension(s) along which to filter (applied sequentially).
    sample_rate : float
        Sampling frequency (default 1.0).
    poles : int
        Filter order (default 4).
    btype : str
        ``'lowpass'``, ``'highpass'``, ``'bandpass'``, or ``'bandstop'``.
    method : str
        | ``'sos'``  — sosfiltfilt (most numerically stable, **default**)
        | ``'pad'``  — filtfilt with padding
        | ``'gust'`` — filtfilt with Gustafsson boundary handling

    Returns
    -------
    xr.DataArray
    """
    if isinstance(dim, str):
        dim = [dim]
    if cutoff == 0:
        return da

    if method == "sos":
        filt_obj = signal.butter(poles, cutoff, btype=btype, output="sos", fs=sample_rate)
    elif method in ("pad", "gust"):
        if poles >= 6:
            import warnings
            warnings.warn(
                f"poles={poles} may cause instability with method={method!r}. "
                "Consider method='sos'.",
                RuntimeWarning, stacklevel=2,
            )
        filt_obj = signal.butter(poles, cutoff, btype=btype, output="ba", fs=sample_rate)
    else:
        raise ValueError(f"method must be 'sos', 'pad', or 'gust'; got {method!r}.")

    result = da.copy(deep=True)
    for d in dim:
        axis = result.dims.index(d)
        result = xr.apply_ufunc(
            _filter_along_axis,
            result,
            kwargs={"filt_obj": filt_obj, "method": method, "axis": axis},
            dask="parallelized",
            output_dtypes=[float],
        )

    result.attrs = da.attrs
    return result


# ---------------------------------------------------------------------------
# Cross-lagged correlation
# ---------------------------------------------------------------------------

def compute_lagged_correlation(
    x: xr.DataArray,
    y: xr.DataArray | xr.Dataset,
    max_lag: int,
    lowpass_cutoff: float | None = None,
    time_dim: str = "time",
) -> xr.DataArray | xr.Dataset:
    """
    Compute cross-lagged correlations between a 1-D reference series and
    a (possibly higher-dimensional) DataArray or Dataset.

    Parameters
    ----------
    x : xr.DataArray
        1-D reference time series.
    y : xr.DataArray or xr.Dataset
        Field with at least a time dimension.  May have additional spatial
        dimensions.
    max_lag : int
        Maximum lag in time steps.  Correlations are computed for all lags
        in ``[-max_lag, max_lag]``.
    lowpass_cutoff : float or None
        If provided, both *x* and *y* are low-pass filtered at this period
        (in the same units as the time axis) before computing correlations.
    time_dim : str
        Name of the time dimension (default ``"time"``).

    Returns
    -------
    xr.DataArray or xr.Dataset
        Correlation values with a new ``lag`` dimension.  Positive lag means
        *y* lags behind *x*.  Return type matches *y*.

    Raises
    ------
    ValueError
        For invalid inputs (wrong dimensionality, lag too large, etc.).
    """
    if x.ndim != 1:
        raise ValueError("'x' must be a 1-D DataArray.")
    if time_dim not in x.dims:
        raise ValueError(f"Dimension '{time_dim}' not found in 'x'.")
    if time_dim not in y.dims:
        raise ValueError(f"Dimension '{time_dim}' not found in 'y'.")
    if max_lag <= 0:
        raise ValueError(f"'max_lag' must be a positive integer; got {max_lag!r}.")
    if 2 * max_lag >= x.sizes[time_dim]:
        raise ValueError(
            f"max_lag={max_lag} is too large for time length {x.sizes[time_dim]}. "
            f"Require 2 * max_lag < len(time)."
        )
    # Align time coordinates if lengths match but values differ
    if not x[time_dim].equals(y[time_dim]):
        if len(x[time_dim]) != len(y[time_dim]):
            raise ValueError(f"'{time_dim}' of 'x' and 'y' must have the same length.")
        x = x.assign_coords({time_dim: y[time_dim]})

    if lowpass_cutoff:
        cutoff_freq = 1.0 / lowpass_cutoff
        x = apply_butterworth_filter(x, cutoff_freq, dim=time_dim, method="gust")
        if isinstance(y, xr.Dataset):
            y = y.map(lambda da: apply_butterworth_filter(
                da, cutoff_freq, dim=time_dim, method="gust"))
        else:
            y = apply_butterworth_filter(y, cutoff_freq, dim=time_dim, method="gust")

    x_window = x.isel({time_dim: slice(max_lag, -max_lag)})

    y_rolling = (
        y.rolling({time_dim: 2 * max_lag + 1}, center=True)
        .construct("lag")
        .isel({time_dim: slice(max_lag, -max_lag)})
    )
    y_rolling = y_rolling.assign_coords(lag=np.arange(-max_lag, max_lag + 1))

    if isinstance(y_rolling, xr.Dataset):
        return xr.Dataset(
            {var: xr.corr(y_rolling[var], x_window, dim=time_dim)
             for var in y_rolling.data_vars}
        )
    return xr.corr(y_rolling, x_window, dim=time_dim)


# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------

def figures_dir(method: SpectralMethod):
    """Return the output directory Path for a given spectral method."""
    return FIGURES_DIR_ROOT / method


def output_csv(method: SpectralMethod):
    """Return the CSV output Path for a given spectral method."""
    from pathlib import Path
    return Path(CSV_OUTPUT_TEMPLATE.format(method=method))
