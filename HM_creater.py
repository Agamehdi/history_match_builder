import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Generic Field History Match & RelPerm Generator",
    layout="wide"
)

# =========================================================
# DEFAULTS
# =========================================================
DEFAULTS = {
    # General field setup
    "field_type": "Oil Field",
    "field_name": "Synthetic Oil Field",
    "gas_field_name": "Synthetic Gas-Condensate Field",
    "seed": 2026,
    "start_year": 2000,
    "history_years": 24,
    "time_frequency": "Monthly",

    # Plot view
    "unit_system": "Metric",      # Metric / Field
    "x_axis_mode": "Date",        # Date / Years
    "plot_start_year": 2000,
    "plot_years": 24,

    # Measured field behavior controls
    "initial_liquid_rate": 900.0,
    "peak_liquid_rate": 1450.0,
    "plateau_years": 4.0,
    "annual_decline_pct": 8.0,
    "late_redevelopment": True,
    "redevelopment_year": 13.0,
    "redevelopment_strength": 0.35,
    "liquid_noise_pct": 5.0,

    "initial_water_cut_pct": 18.0,
    "maximum_water_cut_pct": 82.0,
    "water_cut_rise_year": 8.0,
    "water_cut_rise_speed": 2.4,
    "water_cut_noise_pct": 2.0,

    "initial_gor": 90.0,
    "maximum_gor": 240.0,
    "gor_pressure_sensitivity": 1.1,
    "gor_noise_pct": 5.0,

    "initial_pressure_bar": 255.0,
    "abandonment_pressure_bar": 135.0,
    "aquifer_support": 0.35,
    "pressure_noise_bar": 3.0,

    # Model matching controls
    "liq_quality": 0.82,
    "liq_global_bias": 0,
    "liq_early_bias": 3,
    "liq_late_bias": -3,
    "liq_lag": 0,
    "liq_smoothness": 9,
    "liq_noise": 0.012,
    "liq_offzones": 3,
    "liq_offstrength": 0.05,

    "wcut_quality": 0.84,
    "wcut_global_bias": 0,
    "wcut_early_bias": -2,
    "wcut_late_bias": 2,
    "wcut_lag": 0,
    "wcut_smoothness": 9,
    "wcut_noise": 0.006,
    "wcut_offzones": 3,
    "wcut_offstrength": 0.025,

    "gor_quality": 0.78,
    "gor_global_bias": 0,
    "gor_early_bias": 4,
    "gor_late_bias": -3,
    "gor_lag": 0,
    "gor_smoothness": 11,
    "gor_noise": 0.012,
    "gor_offzones": 4,
    "gor_offstrength": 0.05,

    "prs_quality": 0.86,
    "prs_global_bias": 0,
    "prs_early_bias": 2,
    "prs_late_bias": -2,
    "prs_lag": 0,
    "prs_smoothness": 13,
    "prs_noise": 0.005,
    "prs_offzones": 3,
    "prs_offstrength": 0.02,

    # Display controls
    "oil_dot_size": 7,
    "oil_dot_stride": 2,
    "oil_meas_color": "#d97a6c",
    "oil_model_color": "#2fa84f",

    "gas_dot_size": 7,
    "gas_dot_stride": 2,
    "gas_meas_color": "#d97a6c",
    "gas_model_color": "#e96be0",

    "water_dot_size": 7,
    "water_dot_stride": 2,
    "water_meas_color": "#d97a6c",
    "water_model_color": "#0b3c8c",

    "wcut_dot_size": 7,
    "wcut_dot_stride": 2,
    "wcut_meas_color": "#d97a6c",
    "wcut_model_color": "#e96be0",

    "gor_dot_size": 7,
    "gor_dot_stride": 2,
    "gor_meas_color": "#d97a6c",
    "gor_model_color": "#e96be0",

    "prs_dot_size": 7,
    "prs_dot_stride": 2,
    "prs_meas_color": "#d97a6c",
    "prs_model_color": "#6f35a5",

    "cum_dot_size": 7,
    "cum_dot_stride": 3,
    "cum_meas_color": "#d97a6c",
    "cum_model_color": "#2fa84f",

    # RelPerm default values
    "swc_b": 0.22,
    "sorw_b": 0.25,
    "krw_end_b": 0.85,
    "krow_end_b": 1.00,
    "nw_b": 2.40,
    "no_b": 2.20,

    "swc_a": 0.27,
    "sorw_a": 0.22,
    "krw_end_a": 0.62,
    "krow_end_a": 0.96,
    "nw_a": 3.20,
    "no_a": 3.00,

    "sgc_b": 0.04,
    "sorg_b": 0.25,
    "krg_end_b": 1.00,
    "krog_end_b": 1.00,
    "ng_b": 2.80,
    "nog_b": 2.00,

    "sgc_a": 0.07,
    "sorg_a": 0.28,
    "krg_end_a": 0.72,
    "krog_end_a": 0.95,
    "ng_a": 3.80,
    "nog_a": 2.70,

    # Gas-condensate field production controls
    "gas_initial_rate": 2200.0,          # 10^3 sm3/d
    "gas_peak_rate": 4200.0,             # 10^3 sm3/d
    "gas_plateau_years": 5.0,
    "gas_annual_decline_pct": 6.5,
    "gas_late_redevelopment": True,
    "gas_redevelopment_year": 14.0,
    "gas_redevelopment_strength": 0.25,
    "gas_rate_noise_pct": 4.0,
    "gas_giip_bsm3": 50.0,

    "gas_initial_cgr": 0.22,             # m3 condensate / 10^3 sm3 gas
    "gas_final_cgr": 0.08,
    "gas_cgr_pressure_sensitivity": 1.4,
    "gas_cgr_noise_pct": 4.0,

    "gas_initial_water_rate": 15.0,
    "gas_final_water_rate": 180.0,
    "gas_water_onset_year": 12.0,
    "gas_water_rise_speed": 2.5,
    "gas_water_noise_pct": 8.0,

    "gas_initial_pressure_bar": 420.0,
    "gas_late_pressure_bar": 160.0,
    "gas_dewpoint_bar": 340.0,
    "gas_pressure_support": 0.20,
    "gas_pressure_noise_bar": 3.0,

    # Gas PVT controls
    "gas_gravity": 0.68,
    "gas_temperature_c": 110.0,
    "gas_max_dropout_pct": 16.0,
    "gas_dropout_peak_fraction": 0.45,

    # Gas-field history-match controls
    "gpr_quality": 0.84,
    "gpr_global_bias": 0,
    "gpr_early_bias": 3,
    "gpr_late_bias": -3,
    "gpr_lag": 0,
    "gpr_smoothness": 9,
    "gpr_noise": 0.012,
    "gpr_offzones": 3,
    "gpr_offstrength": 0.05,

    "cgr_quality": 0.82,
    "cgr_global_bias": 0,
    "cgr_early_bias": 3,
    "cgr_late_bias": -3,
    "cgr_lag": 0,
    "cgr_smoothness": 11,
    "cgr_noise": 0.010,
    "cgr_offzones": 3,
    "cgr_offstrength": 0.04,

    "gwat_quality": 0.80,
    "gwat_global_bias": 0,
    "gwat_early_bias": -3,
    "gwat_late_bias": 4,
    "gwat_lag": 0,
    "gwat_smoothness": 11,
    "gwat_noise": 0.012,
    "gwat_offzones": 3,
    "gwat_offstrength": 0.05,

    "gprs_quality": 0.88,
    "gprs_global_bias": 0,
    "gprs_early_bias": 2,
    "gprs_late_bias": -2,
    "gprs_lag": 0,
    "gprs_smoothness": 13,
    "gprs_noise": 0.004,
    "gprs_offzones": 3,
    "gprs_offstrength": 0.018,

    # Gas-field display controls
    "gfr_dot_size": 7,
    "gfr_dot_stride": 2,
    "gfr_meas_color": "#d97a6c",
    "gfr_model_color": "#e96be0",

    "cond_dot_size": 7,
    "cond_dot_stride": 2,
    "cond_meas_color": "#d97a6c",
    "cond_model_color": "#73b82f",

    "cgr_dot_size": 7,
    "cgr_dot_stride": 2,
    "cgr_meas_color": "#d97a6c",
    "cgr_model_color": "#f39c12",

    "gfwat_dot_size": 7,
    "gfwat_dot_stride": 2,
    "gfwat_meas_color": "#d97a6c",
    "gfwat_model_color": "#0b3c8c",

    "gfprs_dot_size": 7,
    "gfprs_dot_stride": 2,
    "gfprs_meas_color": "#d97a6c",
    "gfprs_model_color": "#6f35a5",

    "cumgas_dot_size": 7,
    "cumgas_dot_stride": 3,
    "cumgas_meas_color": "#d97a6c",
    "cumgas_model_color": "#e96be0",

    "cumcond_dot_size": 7,
    "cumcond_dot_stride": 3,
    "cumcond_meas_color": "#d97a6c",
    "cumcond_model_color": "#73b82f",

    "rf_dot_size": 7,
    "rf_dot_stride": 3,
    "rf_meas_color": "#d97a6c",
    "rf_model_color": "#2fa84f",

    # Gas-condensate relative permeability: before / after HM
    "gc_swc_b": 0.20,
    "gc_scc_b": 0.02,
    "gc_sgr_b": 0.05,
    "gc_krg_end_b": 0.95,
    "gc_krc_end_b": 0.70,
    "gc_ng_b": 3.00,
    "gc_nc_b": 2.00,

    "gc_swc_a": 0.20,
    "gc_scc_a": 0.05,
    "gc_sgr_a": 0.08,
    "gc_krg_end_a": 0.76,
    "gc_krc_end_a": 0.52,
    "gc_ng_a": 4.00,
    "gc_nc_a": 2.80,

    # Gas-water relative permeability: before / after HM
    "gw_swc_b": 0.18,
    "gw_sgr_b": 0.05,
    "gw_krw_end_b": 0.75,
    "gw_krg_end_b": 0.95,
    "gw_nw_b": 2.50,
    "gw_ng_b": 2.50,

    "gw_swc_a": 0.22,
    "gw_sgr_a": 0.08,
    "gw_krw_end_a": 0.60,
    "gw_krg_end_a": 0.80,
    "gw_nw_a": 3.20,
    "gw_ng_a": 3.50,
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =========================================================
# Helpers
# =========================================================
def rolling_np(values, window):
    window = max(1, int(window))
    return pd.Series(values).rolling(window, min_periods=1, center=True).mean().to_numpy()


def ema_np(values, span):
    span = max(2, int(span))
    return pd.Series(values).ewm(span=span, adjust=False).mean().to_numpy()


def shift_curve(values, lag):
    values = np.asarray(values, dtype=float)
    idx = np.arange(len(values))
    return np.interp(idx - lag, idx, values, left=values[0], right=values[-1])


def calc_cumulative(rate_m3d, dates):
    days = pd.Series(dates).diff().dt.days.fillna(0).to_numpy()
    return np.cumsum(rate_m3d * days) / 1e6


def calc_cumulative_gas(rate_ksm3d, dates):
    """Return cumulative gas in billion standard cubic metres (BSm3)."""
    days = pd.Series(dates).diff().dt.days.fillna(0).to_numpy()
    return np.cumsum(np.asarray(rate_ksm3d, dtype=float) * 1e3 * days) / 1e9


def normalized_rmse(measured, model):
    measured = np.asarray(measured, dtype=float)
    model = np.asarray(model, dtype=float)
    span = max(float(np.nanmax(measured) - np.nanmin(measured)), 1e-12)
    return float(np.sqrt(np.nanmean((model - measured) ** 2)) / span)


def mean_bias_pct(measured, model):
    measured = np.asarray(measured, dtype=float)
    model = np.asarray(model, dtype=float)
    scale = max(float(np.nanmean(np.abs(measured))), 1e-12)
    return float(100.0 * np.nanmean(model - measured) / scale)


def gas_z_dak(pressure_bar, temperature_c, gas_gravity):
    """Dranchuk-Abou-Kassem dry-gas Z-factor correlation.

    The implementation is intentionally self-contained for this synthetic tool.
    It is suitable for screening and demonstration, not a substitute for a
    tuned compositional EOS or laboratory PVT report.
    """
    pressure_bar = np.asarray(pressure_bar, dtype=float)
    gamma_g = float(np.clip(gas_gravity, 0.55, 1.30))

    p_psia = np.clip(pressure_bar, 1e-6, None) * 14.5037738
    t_rankine = (float(temperature_c) + 273.15) * 9.0 / 5.0

    ppc_psia = 756.8 - 131.0 * gamma_g - 3.6 * gamma_g ** 2
    tpc_rankine = 169.2 + 349.5 * gamma_g - 74.0 * gamma_g ** 2
    pr = p_psia / max(ppc_psia, 1e-9)
    tr = t_rankine / max(tpc_rankine, 1e-9)

    a1, a2, a3, a4, a5 = 0.3265, -1.0700, -0.5339, 0.01569, -0.05165
    a6, a7, a8, a9 = 0.5475, -0.7361, 0.1844, 0.1056
    a10, a11 = 0.6134, 0.7210

    z = np.ones_like(pr, dtype=float)
    for _ in range(120):
        rho_r = np.clip(0.27 * pr / np.clip(z * tr, 1e-9, None), 0.0, 3.5)
        c1 = a1 + a2 / tr + a3 / tr ** 3 + a4 / tr ** 4 + a5 / tr ** 5
        c2 = a6 + a7 / tr + a8 / tr ** 2
        z_new = (
            1.0
            + c1 * rho_r
            + c2 * rho_r ** 2
            - a9 * (a7 / tr + a8 / tr ** 2) * rho_r ** 5
            + a10
            * (1.0 + a11 * rho_r ** 2)
            * (rho_r ** 2 / tr ** 3)
            * np.exp(-a11 * rho_r ** 2)
        )
        z_new = np.clip(z_new, 0.20, 2.00)
        updated = 0.55 * z + 0.45 * z_new
        if np.nanmax(np.abs(updated - z)) < 1e-8:
            z = updated
            break
        z = updated
    return z


def gas_pvt_properties(pressure_bar, temperature_c, gas_gravity):
    pressure_bar = np.asarray(pressure_bar, dtype=float)
    pressure_bar = np.clip(pressure_bar, 1.0, None)
    temperature_k = float(temperature_c) + 273.15
    z_factor = gas_z_dak(pressure_bar, temperature_c, gas_gravity)

    # Reservoir m3 per standard m3 at 15 C and 1.01325 bar.
    bg_m3_sm3 = z_factor * (temperature_k / 288.15) * (1.01325 / pressure_bar)

    molecular_weight = 28.97 * float(gas_gravity) / 1000.0  # kg/mol
    density_kg_m3 = pressure_bar * 1e5 * molecular_weight / (
        np.clip(z_factor, 1e-9, None) * 8.314462618 * temperature_k
    )

    # Lee-Gonzalez-Eakin viscosity correlation; density is required in g/cm3.
    t_rankine = temperature_k * 9.0 / 5.0
    molecular_weight_g = 28.97 * float(gas_gravity)
    density_g_cm3 = density_kg_m3 / 1000.0
    k_term = (
        (9.379 + 0.01607 * molecular_weight_g)
        * t_rankine ** 1.5
        / (209.2 + 19.26 * molecular_weight_g + t_rankine)
    )
    x_term = 3.448 + 986.4 / t_rankine + 0.01009 * molecular_weight_g
    y_term = 2.447 - 0.2224 * x_term
    viscosity_cp = 1e-4 * k_term * np.exp(
        np.clip(x_term * density_g_cm3 ** y_term, -20, 20)
    )

    return z_factor, bg_m3_sm3, viscosity_cp, density_kg_m3


def retrograde_dropout(
    pressure_bar,
    dewpoint_bar,
    minimum_pressure_bar,
    maximum_dropout_pct,
    peak_fraction,
):
    """Synthetic CVD-like retrograde liquid-dropout envelope."""
    pressure_bar = np.asarray(pressure_bar, dtype=float)
    denominator = max(float(dewpoint_bar) - float(minimum_pressure_bar), 1e-9)
    depletion = np.clip((float(dewpoint_bar) - pressure_bar) / denominator, 0.0, 1.0)
    peak = float(np.clip(peak_fraction, 0.10, 0.90))
    shape = 2.2
    curve = np.zeros_like(depletion)
    active = depletion > 0
    ratio = depletion[active] / peak
    curve[active] = ratio ** shape * np.exp(shape * (1.0 - ratio))
    return float(maximum_dropout_pct) * np.clip(curve, 0.0, 1.0)


def add_smooth_offzones(model, measured, count, strength, seed):
    rng = np.random.default_rng(seed)
    n = len(model)
    idx = np.arange(n)
    scale = max(np.nanmax(measured) - np.nanmin(measured), 1.0)
    out = model.copy()

    for _ in range(int(count)):
        center = rng.integers(5, max(6, n - 5))
        width = rng.integers(4, max(5, min(20, n // 4)))
        sign = rng.choice([-1, 1])
        amp = scale * strength * rng.uniform(0.5, 1.2)
        bump = np.exp(-0.5 * ((idx - center) / width) ** 2)
        out += sign * amp * bump

    return out


def build_model_curve(
    measured,
    quality,
    global_bias,
    early_bias,
    late_bias,
    lag,
    smoothness,
    noise,
    offzones,
    offstrength,
    seed,
    min_value=0,
    max_value=None,
):
    rng = np.random.default_rng(seed)
    measured = np.asarray(measured, dtype=float)
    n = len(measured)
    progress = np.linspace(0, 1, n)

    shifted = shift_curve(measured, lag)

    short = rolling_np(shifted, max(3, smoothness // 2))
    long = rolling_np(shifted, smoothness)
    ema = ema_np(shifted, smoothness)

    backbone = 0.35 * short + 0.45 * long + 0.20 * ema
    bias_profile = global_bias + early_bias * (1 - progress) + late_bias * progress
    biased = backbone * (1 + bias_profile / 100.0)

    scale = max(np.nanmax(measured) - np.nanmin(measured), 1.0)
    smooth_noise = rolling_np(rng.normal(0, noise * scale, n), max(5, smoothness))
    distorted = biased + smooth_noise

    distorted = add_smooth_offzones(
        distorted,
        measured,
        count=offzones,
        strength=offstrength,
        seed=seed + 101,
    )

    smooth_measured = rolling_np(measured, max(3, smoothness // 2))
    model = quality * smooth_measured + (1 - quality) * distorted
    model = rolling_np(model, max(3, smoothness // 3))

    if max_value is not None:
        return np.clip(model, min_value, max_value)

    return np.clip(model, min_value, None)


def convert_display_units(
    unit_system,
    oil_measured, oil_model,
    gas_measured, gas_model,
    water_measured, water_model,
    wcut_measured, wcut_model,
    gor_measured, gor_model,
    pressure_measured, pressure_model,
    cum_oil_measured, cum_oil_model,
):
    if unit_system == "Metric":
        return {
            "oil_measured": oil_measured,
            "oil_model": oil_model,
            "gas_measured": gas_measured,
            "gas_model": gas_model,
            "water_measured": water_measured,
            "water_model": water_model,
            "wcut_measured": wcut_measured,
            "wcut_model": wcut_model,
            "gor_measured": gor_measured,
            "gor_model": gor_model,
            "pressure_measured": pressure_measured,
            "pressure_model": pressure_model,
            "cum_oil_measured": cum_oil_measured,
            "cum_oil_model": cum_oil_model,
            "oil_label": "OIL_RATE, m³/d",
            "gas_label": "GAS_RATE, 10³ m³/d",
            "water_label": "WATER_RATE, m³/d",
            "wcut_label": "WCUT, %",
            "gor_label": "GOR, m³/m³",
            "pressure_label": "P, bar",
            "cum_oil_label": "CUMULATIVE OIL, MMm³",
        }

    M3_TO_STB = 6.28981
    M3M3_TO_SCFSTB = 5.615
    BAR_TO_PSI = 14.5038
    KSM3D_TO_MMSCFD = 0.0353147

    return {
        "oil_measured": oil_measured * M3_TO_STB,
        "oil_model": oil_model * M3_TO_STB,
        "gas_measured": gas_measured * KSM3D_TO_MMSCFD,
        "gas_model": gas_model * KSM3D_TO_MMSCFD,
        "water_measured": water_measured * M3_TO_STB,
        "water_model": water_model * M3_TO_STB,
        "wcut_measured": wcut_measured,
        "wcut_model": wcut_model,
        "gor_measured": gor_measured * M3M3_TO_SCFSTB,
        "gor_model": gor_model * M3M3_TO_SCFSTB,
        "pressure_measured": pressure_measured * BAR_TO_PSI,
        "pressure_model": pressure_model * BAR_TO_PSI,
        "cum_oil_measured": cum_oil_measured * M3_TO_STB,
        "cum_oil_model": cum_oil_model * M3_TO_STB,
        "oil_label": "OIL_RATE, STB/d",
        "gas_label": "GAS_RATE, MMscf/d",
        "water_label": "WATER_RATE, STB/d",
        "wcut_label": "WCUT, %",
        "gor_label": "GOR, scf/STB",
        "pressure_label": "P, psi",
        "cum_oil_label": "CUMULATIVE OIL, MMSTB",
    }


def make_plot_window(dates, x_axis_mode, plot_start_year, plot_years):
    start_date = pd.Timestamp(f"{int(plot_start_year)}-01-01")
    end_date = start_date + pd.DateOffset(years=int(plot_years))
    mask = (dates >= start_date) & (dates < end_date)

    if mask.sum() == 0:
        mask = np.ones(len(dates), dtype=bool)

    filtered_dates = dates[mask]

    if x_axis_mode == "Years":
        x_values = (filtered_dates - filtered_dates[0]).days / 365.25
        return mask, x_values, "YEARS"

    return mask, filtered_dates, "DATE"


def add_history_trace(
    fig,
    row,
    col,
    x,
    measured,
    model,
    y_title,
    measured_color,
    model_color,
    dot_size,
    dot_stride,
    x_title="DATE",
    show_legend=False,
):
    fig.add_trace(
        go.Scatter(
            x=x[::dot_stride],
            y=measured[::dot_stride],
            mode="markers",
            name="Measured Data",
            marker=dict(
                color=measured_color,
                size=dot_size,
                opacity=0.80,
                line=dict(color="rgba(60,60,60,0.55)", width=0.8),
            ),
            showlegend=show_legend,
        ),
        row=row,
        col=col,
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=model,
            mode="lines",
            name="Model",
            line=dict(color=model_color, width=2.8),
            showlegend=show_legend,
        ),
        row=row,
        col=col,
    )

    fig.update_yaxes(title_text=y_title, row=row, col=col)
    fig.update_xaxes(title_text=x_title, row=row, col=col)


def corey_water_oil(sw, swc, sorw, krw_end, krow_end, nw, no):
    denom = max(1e-6, 1 - swc - sorw)
    se = np.clip((sw - swc) / denom, 0, 1)
    krw = krw_end * se ** nw
    krow = krow_end * (1 - se) ** no
    return krw, krow


def corey_gas_oil(sg, sgc, sorg, krg_end, krog_end, ng, nog):
    denom = max(1e-6, 1 - sgc - sorg)
    se = np.clip((sg - sgc) / denom, 0, 1)
    krg = krg_end * se ** ng
    krog = krog_end * (1 - se) ** nog
    return krg, krog


def corey_gas_condensate(
    sc,
    swc,
    scc,
    sgr,
    krg_end,
    krc_end,
    ng,
    nc,
):
    """Corey gas-condensate curves at fixed connate-water saturation."""
    mobile_interval = max(1e-6, 1.0 - swc - scc - sgr)
    se_c = np.clip((np.asarray(sc, dtype=float) - scc) / mobile_interval, 0, 1)
    krc = krc_end * se_c ** nc
    krg = krg_end * (1.0 - se_c) ** ng
    return krg, krc


def corey_gas_water(sw, swc, sgr, krw_end, krg_end, nw, ng):
    """Corey gas-water curves expressed on the water-saturation axis."""
    mobile_interval = max(1e-6, 1.0 - swc - sgr)
    se_w = np.clip((np.asarray(sw, dtype=float) - swc) / mobile_interval, 0, 1)
    krw = krw_end * se_w ** nw
    krg = krg_end * (1.0 - se_w) ** ng
    return krg, krw


def model_control_group(title, prefix):
    with st.sidebar.expander(title, expanded=False):
        quality = st.slider("Match quality", 0.0, 1.0, step=0.01, key=f"{prefix}_quality")
        global_bias = st.slider("Global bias, %", -30, 30, step=1, key=f"{prefix}_global_bias")
        early_bias = st.slider("Early-period bias, %", -30, 30, step=1, key=f"{prefix}_early_bias")
        late_bias = st.slider("Late-period bias, %", -30, 30, step=1, key=f"{prefix}_late_bias")
        lag = st.slider("Lag / lead, timesteps", -12, 12, step=1, key=f"{prefix}_lag")
        smoothness = st.slider("Model smoothness", 3, 35, step=2, key=f"{prefix}_smoothness")
        noise = st.slider("Low-frequency noise", 0.0, 0.08, step=0.002, key=f"{prefix}_noise")
        offzones = st.slider("Off-zone count", 0, 8, step=1, key=f"{prefix}_offzones")
        offstrength = st.slider("Off-zone strength", 0.0, 0.25, step=0.005, key=f"{prefix}_offstrength")

    return {
        "quality": quality,
        "global_bias": global_bias,
        "early_bias": early_bias,
        "late_bias": late_bias,
        "lag": lag,
        "smoothness": smoothness,
        "noise": noise,
        "offzones": offzones,
        "offstrength": offstrength,
    }


def display_control_group(title, prefix):
    with st.sidebar.expander(title, expanded=False):
        dot_size = st.slider("Measured dot size", 2, 18, step=1, key=f"{prefix}_dot_size")
        dot_stride = st.slider("Measured dot density: show every Nth point", 1, 12, step=1, key=f"{prefix}_dot_stride")
        meas_color = st.color_picker("Measured dot color", key=f"{prefix}_meas_color")
        model_color = st.color_picker("Model line color", key=f"{prefix}_model_color")

    return {
        "dot_size": dot_size,
        "dot_stride": dot_stride,
        "meas_color": meas_color,
        "model_color": model_color,
    }


def convert_gas_field_display_units(
    unit_system,
    gas_measured,
    gas_model,
    condensate_measured,
    condensate_model,
    water_measured,
    water_model,
    cgr_measured,
    cgr_model,
    pressure_measured,
    pressure_model,
    cumulative_gas_measured,
    cumulative_gas_model,
    cumulative_condensate_measured,
    cumulative_condensate_model,
    recovery_measured,
    recovery_model,
):
    if unit_system == "Metric":
        return {
            "gas_measured": gas_measured,
            "gas_model": gas_model,
            "condensate_measured": condensate_measured,
            "condensate_model": condensate_model,
            "water_measured": water_measured,
            "water_model": water_model,
            "cgr_measured": cgr_measured,
            "cgr_model": cgr_model,
            "pressure_measured": pressure_measured,
            "pressure_model": pressure_model,
            "cumulative_gas_measured": cumulative_gas_measured,
            "cumulative_gas_model": cumulative_gas_model,
            "cumulative_condensate_measured": cumulative_condensate_measured,
            "cumulative_condensate_model": cumulative_condensate_model,
            "recovery_measured": recovery_measured,
            "recovery_model": recovery_model,
            "gas_label": "GAS RATE, 10³ sm³/d",
            "condensate_label": "CONDENSATE RATE, m³/d",
            "water_label": "WATER RATE, m³/d",
            "cgr_label": "CGR, m³/10³ sm³",
            "pressure_label": "P, bar",
            "cumulative_gas_label": "CUMULATIVE GAS, BSm³",
            "cumulative_condensate_label": "CUMULATIVE CONDENSATE, MMm³",
            "recovery_label": "GAS RECOVERY, % GIIP",
        }

    ksm3d_to_mmscfd = 0.0353147
    m3_to_stb = 6.28981
    bar_to_psi = 14.5037738
    bsm3_to_bscf = 35.3147
    cgr_metric_to_field = m3_to_stb / ksm3d_to_mmscfd

    return {
        "gas_measured": gas_measured * ksm3d_to_mmscfd,
        "gas_model": gas_model * ksm3d_to_mmscfd,
        "condensate_measured": condensate_measured * m3_to_stb,
        "condensate_model": condensate_model * m3_to_stb,
        "water_measured": water_measured * m3_to_stb,
        "water_model": water_model * m3_to_stb,
        "cgr_measured": cgr_measured * cgr_metric_to_field,
        "cgr_model": cgr_model * cgr_metric_to_field,
        "pressure_measured": pressure_measured * bar_to_psi,
        "pressure_model": pressure_model * bar_to_psi,
        "cumulative_gas_measured": cumulative_gas_measured * bsm3_to_bscf,
        "cumulative_gas_model": cumulative_gas_model * bsm3_to_bscf,
        "cumulative_condensate_measured": cumulative_condensate_measured * m3_to_stb,
        "cumulative_condensate_model": cumulative_condensate_model * m3_to_stb,
        "recovery_measured": recovery_measured,
        "recovery_model": recovery_model,
        "gas_label": "GAS RATE, MMscf/d",
        "condensate_label": "CONDENSATE RATE, STB/d",
        "water_label": "WATER RATE, STB/d",
        "cgr_label": "CGR, STB/MMscf",
        "pressure_label": "P, psi",
        "cumulative_gas_label": "CUMULATIVE GAS, Bscf",
        "cumulative_condensate_label": "CUMULATIVE CONDENSATE, MMSTB",
        "recovery_label": "GAS RECOVERY, % GIIP",
    }


def make_excel(history_df, relperm_wo_df, relperm_go_df, controls_df):
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        history_df.to_excel(writer, sheet_name="History_Match_Metric", index=False)
        relperm_wo_df.to_excel(writer, sheet_name="RelPerm_WaterOil", index=False)
        relperm_go_df.to_excel(writer, sheet_name="RelPerm_GasOil", index=False)
        controls_df.to_excel(writer, sheet_name="Controls", index=False)

        workbook = writer.book
        header_fmt = workbook.add_format({
            "bold": True,
            "bg_color": "#D9EAF7",
            "border": 1,
        })

        sheets = {
            "History_Match_Metric": history_df,
            "RelPerm_WaterOil": relperm_wo_df,
            "RelPerm_GasOil": relperm_go_df,
            "Controls": controls_df,
        }

        for sheet_name, df in sheets.items():
            ws = writer.sheets[sheet_name]
            ws.freeze_panes(1, 0)
            ws.set_column(0, len(df.columns) - 1, 22)

            for c, col in enumerate(df.columns):
                ws.write(0, c, col, header_fmt)

    output.seek(0)
    return output


def make_excel_sheets(sheets):
    """Build a formatted workbook from an ordered mapping of sheet names."""
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        workbook = writer.book
        header_fmt = workbook.add_format({
            "bold": True,
            "bg_color": "#D9EAF7",
            "border": 1,
        })

        for sheet_name, dataframe in sheets.items():
            safe_name = str(sheet_name)[:31]
            dataframe.to_excel(writer, sheet_name=safe_name, index=False)
            worksheet = writer.sheets[safe_name]
            worksheet.freeze_panes(1, 0)
            if len(dataframe.columns):
                worksheet.set_column(0, len(dataframe.columns) - 1, 22)
            for column_index, column_name in enumerate(dataframe.columns):
                worksheet.write(0, column_index, column_name, header_fmt)

    output.seek(0)
    return output


def render_gas_field(
    field_name,
    seed,
    start_year,
    history_years,
    time_frequency,
    unit_system,
    x_axis_mode,
    plot_start_year,
    plot_years,
):
    """Render the complete synthetic gas-condensate history-match workflow."""
    st.sidebar.header("Gas-Condensate Measured History Controls")

    with st.sidebar.expander("Gas Production Profile", expanded=True):
        gas_initial_rate = st.slider(
            "Initial gas rate, 10³ sm³/d", 100.0, 20000.0, step=100.0,
            key="gas_initial_rate",
        )
        gas_peak_rate = st.slider(
            "Peak gas rate, 10³ sm³/d", 100.0, 30000.0, step=100.0,
            key="gas_peak_rate",
        )
        gas_plateau_years = st.slider(
            "Gas plateau years", 0.0, 20.0, step=0.5,
            key="gas_plateau_years",
        )
        gas_annual_decline_pct = st.slider(
            "Annual gas decline, %", 0.5, 30.0, step=0.5,
            key="gas_annual_decline_pct",
        )
        gas_late_redevelopment = st.checkbox(
            "Add late compression / infill hump",
            key="gas_late_redevelopment",
        )
        st.session_state["gas_redevelopment_year"] = min(
            float(st.session_state["gas_redevelopment_year"]),
            float(history_years),
        )
        gas_redevelopment_year = st.slider(
            "Compression / infill timing, years from start",
            1.0,
            float(history_years),
            step=0.5,
            key="gas_redevelopment_year",
        )
        gas_redevelopment_strength = st.slider(
            "Compression / infill strength", 0.0, 1.5, step=0.05,
            key="gas_redevelopment_strength",
        )
        gas_rate_noise_pct = st.slider(
            "Measured gas-rate noise, %", 0.0, 20.0, step=0.5,
            key="gas_rate_noise_pct",
        )
        gas_giip_bsm3 = st.slider(
            "GIIP, BSm³", 1.0, 500.0, step=1.0,
            key="gas_giip_bsm3",
        )

    with st.sidebar.expander("Condensate / CGR Profile", expanded=True):
        gas_initial_cgr = st.slider(
            "Initial CGR, m³/10³ sm³", 0.01, 1.50, step=0.01,
            key="gas_initial_cgr",
        )
        gas_final_cgr = st.slider(
            "Late CGR, m³/10³ sm³", 0.005, 1.000, step=0.005,
            key="gas_final_cgr",
        )
        gas_cgr_pressure_sensitivity = st.slider(
            "CGR pressure sensitivity", 0.2, 4.0, step=0.1,
            key="gas_cgr_pressure_sensitivity",
        )
        gas_cgr_noise_pct = st.slider(
            "Measured CGR noise, %", 0.0, 20.0, step=0.5,
            key="gas_cgr_noise_pct",
        )

    with st.sidebar.expander("Produced Water Profile", expanded=False):
        gas_initial_water_rate = st.slider(
            "Initial water rate, m³/d", 0.0, 1000.0, step=5.0,
            key="gas_initial_water_rate",
        )
        gas_final_water_rate = st.slider(
            "Late water rate, m³/d", 0.0, 5000.0, step=10.0,
            key="gas_final_water_rate",
        )
        st.session_state["gas_water_onset_year"] = min(
            float(st.session_state["gas_water_onset_year"]),
            float(history_years),
        )
        gas_water_onset_year = st.slider(
            "Water-rise timing, years from start",
            0.5,
            float(history_years),
            step=0.5,
            key="gas_water_onset_year",
        )
        gas_water_rise_speed = st.slider(
            "Water-rise speed", 0.5, 8.0, step=0.1,
            key="gas_water_rise_speed",
        )
        gas_water_noise_pct = st.slider(
            "Measured water-rate noise, %", 0.0, 25.0, step=0.5,
            key="gas_water_noise_pct",
        )

    with st.sidebar.expander("Pressure and Gas PVT", expanded=True):
        gas_initial_pressure_bar_input = st.slider(
            "Initial reservoir pressure, bar", 80.0, 800.0, step=5.0,
            key="gas_initial_pressure_bar",
        )
        gas_late_pressure_bar_input = st.slider(
            "Late reservoir pressure, bar", 20.0, 600.0, step=5.0,
            key="gas_late_pressure_bar",
        )
        gas_dewpoint_bar_input = st.slider(
            "Dew-point pressure, bar", 40.0, 700.0, step=5.0,
            key="gas_dewpoint_bar",
        )
        gas_pressure_support = st.slider(
            "Aquifer / pressure support", 0.0, 0.9, step=0.05,
            key="gas_pressure_support",
        )
        gas_pressure_noise_bar = st.slider(
            "Measured pressure noise, bar", 0.0, 20.0, step=0.5,
            key="gas_pressure_noise_bar",
        )
        gas_gravity = st.slider(
            "Gas specific gravity", 0.55, 1.20, step=0.01,
            key="gas_gravity",
        )
        gas_temperature_c = st.slider(
            "Reservoir temperature, °C", 20.0, 220.0, step=2.0,
            key="gas_temperature_c",
        )
        gas_max_dropout_pct = st.slider(
            "Maximum retrograde liquid dropout, % PV",
            1.0,
            35.0,
            step=0.5,
            key="gas_max_dropout_pct",
        )
        gas_dropout_peak_fraction = st.slider(
            "Dropout-peak position below dew point",
            0.10,
            0.90,
            step=0.05,
            key="gas_dropout_peak_fraction",
        )

    gas_initial_pressure_bar = max(
        float(gas_initial_pressure_bar_input),
        float(gas_late_pressure_bar_input) + 1.0,
    )
    gas_late_pressure_bar = min(
        float(gas_late_pressure_bar_input),
        gas_initial_pressure_bar - 1.0,
    )
    gas_dewpoint_bar = float(np.clip(
        gas_dewpoint_bar_input,
        gas_late_pressure_bar + 0.5,
        gas_initial_pressure_bar - 0.5,
    ))
    if (
        gas_initial_pressure_bar != gas_initial_pressure_bar_input
        or gas_late_pressure_bar != gas_late_pressure_bar_input
        or gas_dewpoint_bar != gas_dewpoint_bar_input
    ):
        st.sidebar.warning(
            "For calculation stability, the pressure order is enforced as "
            "initial pressure > dew point > late pressure."
        )

    st.sidebar.header("Gas-Field Model Match Controls")
    gas_cfg = model_control_group("Primary Gas Production", "gpr")
    cgr_cfg = model_control_group("CGR / Condensate System", "cgr")
    gas_water_cfg = model_control_group("Produced Water System", "gwat")
    gas_pressure_cfg = model_control_group("Reservoir Pressure System", "gprs")

    st.sidebar.header("Gas-Field Display Controls")
    gas_disp = display_control_group("Gas Rate Display", "gfr")
    condensate_disp = display_control_group("Condensate Rate Display", "cond")
    gas_water_disp = display_control_group("Water Rate Display", "gfwat")
    cgr_disp = display_control_group("CGR Display", "cgr")
    pressure_disp = display_control_group("Pressure Display", "gfprs")
    cumulative_gas_disp = display_control_group("Cumulative Gas Display", "cumgas")
    cumulative_condensate_disp = display_control_group(
        "Cumulative Condensate Display", "cumcond"
    )
    recovery_disp = display_control_group("Gas Recovery Display", "rf")

    # -----------------------------------------------------
    # Synthetic measured history
    # -----------------------------------------------------
    freq = "MS" if time_frequency == "Monthly" else "QS"
    steps_per_year = 12 if time_frequency == "Monthly" else 4
    periods = int(history_years * steps_per_year)
    dates = pd.date_range(f"{int(start_year)}-01-01", periods=periods, freq=freq)
    t_years = np.arange(periods) / steps_per_year
    rng = np.random.default_rng(seed)

    ramp = gas_initial_rate + (gas_peak_rate - gas_initial_rate) * (
        1.0 - np.exp(-t_years / 0.9)
    )
    gas_decline_factor = np.exp(
        -gas_annual_decline_pct / 100.0
        * np.maximum(t_years - gas_plateau_years, 0.0)
    )
    gas_rate_clean = ramp * gas_decline_factor

    if gas_late_redevelopment:
        hump_width = max(1.0, history_years / 9.0)
        hump = (
            gas_redevelopment_strength
            * gas_peak_rate
            * np.exp(-0.5 * ((t_years - gas_redevelopment_year) / hump_width) ** 2)
        )
        gas_rate_clean += hump

    gas_seasonality = 1.0 + 0.018 * np.sin(2.0 * np.pi * t_years)
    gas_rate_measured = gas_rate_clean * gas_seasonality
    gas_rate_measured *= 1.0 + rng.normal(0, gas_rate_noise_pct / 100.0, periods)
    gas_rate_measured = np.clip(gas_rate_measured, 1.0, None)

    cumulative_gas_measured = calc_cumulative_gas(gas_rate_measured, dates)
    cumulative_gas_norm = cumulative_gas_measured / max(
        float(cumulative_gas_measured.max()), 1e-12
    )
    gas_pressure_drop = (
        gas_initial_pressure_bar - gas_late_pressure_bar
    ) * (1.0 - gas_pressure_support)
    pressure_clean = (
        gas_initial_pressure_bar
        - gas_pressure_drop * cumulative_gas_norm ** 0.78
        + 1.5 * np.sin(2.0 * np.pi * t_years / 8.0)
    )
    pressure_measured = pressure_clean + rng.normal(
        0, gas_pressure_noise_bar, periods
    )
    pressure_measured = np.clip(
        pressure_measured,
        gas_late_pressure_bar,
        gas_initial_pressure_bar * 1.03,
    )

    below_dew = np.clip(
        (gas_dewpoint_bar - pressure_measured)
        / max(gas_dewpoint_bar - gas_late_pressure_bar, 1e-9),
        0.0,
        1.0,
    )
    dewpoint_enrichment = 0.08 * gas_initial_cgr * np.exp(
        -0.5
        * ((pressure_measured - gas_dewpoint_bar) / max(0.08 * gas_dewpoint_bar, 1.0)) ** 2
    )
    cgr_clean = (
        gas_initial_cgr
        + (gas_final_cgr - gas_initial_cgr)
        * below_dew ** gas_cgr_pressure_sensitivity
        + dewpoint_enrichment
    )
    cgr_measured = cgr_clean * (
        1.0 + rng.normal(0, gas_cgr_noise_pct / 100.0, periods)
    )
    cgr_upper = max(gas_initial_cgr, gas_final_cgr) * 1.35
    cgr_measured = np.clip(cgr_measured, 0.001, max(cgr_upper, 0.01))

    # In metric base units: (10^3 sm3/d) * (m3 / 10^3 sm3) = m3/d.
    condensate_rate_measured = gas_rate_measured * cgr_measured
    cumulative_condensate_measured = calc_cumulative(
        condensate_rate_measured, dates
    )

    water_logistic = 1.0 / (
        1.0 + np.exp(-(t_years - gas_water_onset_year) / gas_water_rise_speed)
    )
    water_rate_clean = gas_initial_water_rate + (
        gas_final_water_rate - gas_initial_water_rate
    ) * water_logistic
    water_rate_measured = water_rate_clean * (
        1.0 + rng.normal(0, gas_water_noise_pct / 100.0, periods)
    )
    water_rate_measured = np.clip(water_rate_measured, 0.0, None)

    gas_recovery_measured = np.clip(
        100.0 * cumulative_gas_measured / max(gas_giip_bsm3, 1e-12),
        0.0,
        100.0,
    )

    # -----------------------------------------------------
    # Model curves. Condensate remains exactly gas x CGR.
    # -----------------------------------------------------
    gas_rate_model = build_model_curve(
        gas_rate_measured,
        seed=seed + 21,
        min_value=0,
        max_value=None,
        **gas_cfg,
    )
    cgr_model = build_model_curve(
        cgr_measured,
        seed=seed + 22,
        min_value=0.001,
        max_value=max(cgr_upper, 0.01),
        **cgr_cfg,
    )
    water_rate_model = build_model_curve(
        water_rate_measured,
        seed=seed + 23,
        min_value=0,
        max_value=None,
        **gas_water_cfg,
    )
    condensate_rate_model = gas_rate_model * cgr_model
    cumulative_gas_model = calc_cumulative_gas(gas_rate_model, dates)
    cumulative_condensate_model = calc_cumulative(condensate_rate_model, dates)
    gas_recovery_model = np.clip(
        100.0 * cumulative_gas_model / max(gas_giip_bsm3, 1e-12),
        0.0,
        100.0,
    )

    cumulative_gas_model_norm = cumulative_gas_model / max(
        float(cumulative_gas_model.max()), 1e-12
    )
    pressure_consistent_model = (
        gas_initial_pressure_bar
        - gas_pressure_drop * cumulative_gas_model_norm ** 0.78
    )
    pressure_model = build_model_curve(
        pressure_consistent_model,
        seed=seed + 24,
        min_value=gas_late_pressure_bar,
        max_value=gas_initial_pressure_bar * 1.03,
        **gas_pressure_cfg,
    )

    z_history_measured, bg_history_measured, mug_history_measured, rho_history_measured = (
        gas_pvt_properties(pressure_measured, gas_temperature_c, gas_gravity)
    )
    z_history_model, bg_history_model, mug_history_model, rho_history_model = (
        gas_pvt_properties(pressure_model, gas_temperature_c, gas_gravity)
    )
    dropout_history_measured = retrograde_dropout(
        pressure_measured,
        gas_dewpoint_bar,
        gas_late_pressure_bar,
        gas_max_dropout_pct,
        gas_dropout_peak_fraction,
    )
    dropout_history_model = retrograde_dropout(
        pressure_model,
        gas_dewpoint_bar,
        gas_late_pressure_bar,
        gas_max_dropout_pct,
        gas_dropout_peak_fraction,
    )

    display = convert_gas_field_display_units(
        unit_system,
        gas_rate_measured,
        gas_rate_model,
        condensate_rate_measured,
        condensate_rate_model,
        water_rate_measured,
        water_rate_model,
        cgr_measured,
        cgr_model,
        pressure_measured,
        pressure_model,
        cumulative_gas_measured,
        cumulative_gas_model,
        cumulative_condensate_measured,
        cumulative_condensate_model,
        gas_recovery_measured,
        gas_recovery_model,
    )
    plot_mask, x_plot, x_title = make_plot_window(
        dates,
        x_axis_mode,
        plot_start_year,
        plot_years,
    )

    # -----------------------------------------------------
    # Production and pressure history-match dashboard
    # -----------------------------------------------------
    history_fig = make_subplots(
        rows=4,
        cols=2,
        subplot_titles=(
            "GAS RATE – HISTORY MATCH",
            "CONDENSATE RATE – HISTORY MATCH",
            "WATER RATE – HISTORY MATCH",
            "CGR – HISTORY MATCH",
            "RESERVOIR PRESSURE – HISTORY MATCH",
            "CUMULATIVE GAS – HISTORY MATCH",
            "CUMULATIVE CONDENSATE – HISTORY MATCH",
            "GAS RECOVERY FACTOR",
        ),
        vertical_spacing=0.105,
        horizontal_spacing=0.08,
    )

    history_items = [
        (1, 1, "gas", gas_disp, True),
        (1, 2, "condensate", condensate_disp, False),
        (2, 1, "water", gas_water_disp, False),
        (2, 2, "cgr", cgr_disp, False),
        (3, 1, "pressure", pressure_disp, False),
        (3, 2, "cumulative_gas", cumulative_gas_disp, False),
        (4, 1, "cumulative_condensate", cumulative_condensate_disp, False),
        (4, 2, "recovery", recovery_disp, False),
    ]
    for row, col, key, style, show_legend in history_items:
        add_history_trace(
            history_fig,
            row,
            col,
            x_plot,
            display[f"{key}_measured"][plot_mask],
            display[f"{key}_model"][plot_mask],
            display[f"{key}_label"],
            style["meas_color"],
            style["model_color"],
            style["dot_size"],
            style["dot_stride"],
            x_title=x_title,
            show_legend=show_legend,
        )

    history_fig.update_layout(
        title_text=(
            f"{field_name} – Material-Balanced Gas-Condensate History Match "
            f"({unit_system}, X-axis: {x_axis_mode}, {plot_years} years from {plot_start_year})"
        ),
        title_x=0.5,
        height=1450,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.055,
            xanchor="center",
            x=0.5,
        ),
    )
    history_fig.update_xaxes(showgrid=True, gridcolor="#D5E5F7")
    history_fig.update_yaxes(showgrid=True, gridcolor="#D5E5F7")
    if x_axis_mode == "Years":
        history_fig.update_xaxes(dtick=1)
    st.plotly_chart(history_fig, use_container_width=True)

    # -----------------------------------------------------
    # Consistency and match diagnostics
    # -----------------------------------------------------
    st.subheader("Gas-Condensate Material Balance / Consistency Checks")
    balance_condensate = gas_rate_measured * cgr_measured
    diagnostic_columns = st.columns(4)
    diagnostic_columns[0].metric(
        "Max gas×CGR error",
        f"{np.max(np.abs(condensate_rate_measured - balance_condensate)):.6f} m³/d",
    )
    diagnostic_columns[1].metric(
        "Final cumulative gas",
        f"{cumulative_gas_measured[-1]:.3f} BSm³",
    )
    diagnostic_columns[2].metric(
        "Final cumulative condensate",
        f"{cumulative_condensate_measured[-1]:.3f} MMm³",
    )
    diagnostic_columns[3].metric(
        "Final gas recovery",
        f"{gas_recovery_measured[-1]:.1f}% GIIP",
    )

    match_metrics_df = pd.DataFrame({
        "Response": [
            "Gas Rate",
            "Condensate Rate",
            "Water Rate",
            "CGR",
            "Reservoir Pressure",
            "Cumulative Gas",
            "Cumulative Condensate",
        ],
        "NRMSE": [
            normalized_rmse(gas_rate_measured, gas_rate_model),
            normalized_rmse(condensate_rate_measured, condensate_rate_model),
            normalized_rmse(water_rate_measured, water_rate_model),
            normalized_rmse(cgr_measured, cgr_model),
            normalized_rmse(pressure_measured, pressure_model),
            normalized_rmse(cumulative_gas_measured, cumulative_gas_model),
            normalized_rmse(
                cumulative_condensate_measured,
                cumulative_condensate_model,
            ),
        ],
        "Mean_Bias_pct": [
            mean_bias_pct(gas_rate_measured, gas_rate_model),
            mean_bias_pct(condensate_rate_measured, condensate_rate_model),
            mean_bias_pct(water_rate_measured, water_rate_model),
            mean_bias_pct(cgr_measured, cgr_model),
            mean_bias_pct(pressure_measured, pressure_model),
            mean_bias_pct(cumulative_gas_measured, cumulative_gas_model),
            mean_bias_pct(
                cumulative_condensate_measured,
                cumulative_condensate_model,
            ),
        ],
    })
    match_columns = st.columns(4)
    match_columns[0].metric(
        "Gas NRMSE", f"{match_metrics_df.loc[0, 'NRMSE']:.3f}"
    )
    match_columns[1].metric(
        "Condensate NRMSE", f"{match_metrics_df.loc[1, 'NRMSE']:.3f}"
    )
    match_columns[2].metric(
        "CGR NRMSE", f"{match_metrics_df.loc[3, 'NRMSE']:.3f}"
    )
    match_columns[3].metric(
        "Pressure NRMSE", f"{match_metrics_df.loc[4, 'NRMSE']:.3f}"
    )
    st.success(
        "Gas, condensate and CGR are internally consistent: condensate rate is "
        "calculated at every timestep as Gas Rate × CGR."
    )

    # -----------------------------------------------------
    # Gas PVT and material-balance diagnostics
    # -----------------------------------------------------
    st.header("Gas-Condensate PVT and Reservoir Diagnostics")
    st.caption(
        "Synthetic screening correlations are used: Sutton pseudo-critical properties, "
        "Dranchuk-Abou-Kassem Z-factor and Lee-Gonzalez-Eakin gas viscosity. "
        "Replace these curves with tuned laboratory/EOS PVT for field decisions."
    )

    pvt_pressure_bar = np.linspace(
        max(10.0, gas_late_pressure_bar * 0.65),
        gas_initial_pressure_bar * 1.05,
        180,
    )
    z_pvt, bg_pvt, mug_pvt, rho_pvt = gas_pvt_properties(
        pvt_pressure_bar,
        gas_temperature_c,
        gas_gravity,
    )
    dropout_pvt = retrograde_dropout(
        pvt_pressure_bar,
        gas_dewpoint_bar,
        gas_late_pressure_bar,
        gas_max_dropout_pct,
        gas_dropout_peak_fraction,
    )

    pvt_tab, pz_tab, dropout_tab = st.tabs([
        "Gas PVT Curves",
        "P/Z Material Balance",
        "Retrograde Dropout",
    ])

    if unit_system == "Metric":
        pvt_pressure_display = pvt_pressure_bar
        pvt_pressure_label = "Pressure, bar"
        bg_display = bg_pvt
        bg_label = "Bg, rm³/sm³"
        rho_display = rho_pvt
        rho_label = "Gas density, kg/m³"
        pz_pressure_measured = pressure_measured
        pz_pressure_model = pressure_model
        pz_label = "P/Z, bar"
    else:
        pvt_pressure_display = pvt_pressure_bar * 14.5037738
        pvt_pressure_label = "Pressure, psi"
        bg_display = bg_pvt * (6.28981 / 0.0353147)
        bg_label = "Bg, rb/Mscf"
        rho_display = rho_pvt * 0.06242796
        rho_label = "Gas density, lb/ft³"
        pz_pressure_measured = pressure_measured * 14.5037738
        pz_pressure_model = pressure_model * 14.5037738
        pz_label = "P/Z, psi"

    with pvt_tab:
        pvt_fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Gas Z-Factor",
                "Gas Formation Volume Factor",
                "Gas Viscosity",
                "Gas Density",
            ),
            vertical_spacing=0.14,
            horizontal_spacing=0.10,
        )
        pvt_series = [
            (1, 1, z_pvt, "Z, dimensionless", "#264653"),
            (1, 2, bg_display, bg_label, "#2a9d8f"),
            (2, 1, mug_pvt, "Gas viscosity, cP", "#e76f51"),
            (2, 2, rho_display, rho_label, "#6f35a5"),
        ]
        for row, col, values, y_label, color in pvt_series:
            pvt_fig.add_trace(
                go.Scatter(
                    x=pvt_pressure_display,
                    y=values,
                    mode="lines",
                    line=dict(color=color, width=3),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            pvt_fig.update_xaxes(
                title_text=pvt_pressure_label,
                showgrid=True,
                gridcolor="#D5E5F7",
                row=row,
                col=col,
            )
            pvt_fig.update_yaxes(
                title_text=y_label,
                showgrid=True,
                gridcolor="#D5E5F7",
                row=row,
                col=col,
            )
        pvt_fig.update_layout(
            title_text=(
                f"Gas PVT at {gas_temperature_c:.0f} °C and gas gravity "
                f"{gas_gravity:.2f}"
            ),
            title_x=0.5,
            height=820,
            template="plotly_white",
        )
        st.plotly_chart(pvt_fig, use_container_width=True)

    with pz_tab:
        pz_fig = go.Figure()
        pz_fig.add_trace(go.Scatter(
            x=display["cumulative_gas_measured"],
            y=pz_pressure_measured / np.clip(z_history_measured, 1e-9, None),
            mode="markers",
            name="Measured P/Z",
            marker=dict(color="#d97a6c", size=7),
        ))
        pz_fig.add_trace(go.Scatter(
            x=display["cumulative_gas_model"],
            y=pz_pressure_model / np.clip(z_history_model, 1e-9, None),
            mode="lines",
            name="Model P/Z",
            line=dict(color="#264653", width=3),
        ))
        pz_fig.update_layout(
            title="P/Z versus Cumulative Gas",
            xaxis_title=display["cumulative_gas_label"],
            yaxis_title=pz_label,
            template="plotly_white",
            height=560,
        )
        pz_fig.update_xaxes(showgrid=True, gridcolor="#D5E5F7")
        pz_fig.update_yaxes(showgrid=True, gridcolor="#D5E5F7")
        st.plotly_chart(pz_fig, use_container_width=True)

    with dropout_tab:
        dropout_fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                "Synthetic CVD-Like Liquid Dropout",
                "Dropout Proxy through History",
            ),
            horizontal_spacing=0.10,
        )
        dropout_fig.add_trace(go.Scatter(
            x=pvt_pressure_display,
            y=dropout_pvt,
            mode="lines",
            name="PVT dropout",
            line=dict(color="#73b82f", width=3),
        ), row=1, col=1)
        dropout_fig.add_vline(
            x=gas_dewpoint_bar if unit_system == "Metric" else gas_dewpoint_bar * 14.5037738,
            line_dash="dash",
            line_color="#e76f51",
            annotation_text="Dew point",
            row=1,
            col=1,
        )
        dropout_fig.add_trace(go.Scatter(
            x=x_plot,
            y=dropout_history_measured[plot_mask],
            mode="markers",
            name="Measured-pressure proxy",
            marker=dict(color="#d97a6c", size=6),
        ), row=1, col=2)
        dropout_fig.add_trace(go.Scatter(
            x=x_plot,
            y=dropout_history_model[plot_mask],
            mode="lines",
            name="Model-pressure proxy",
            line=dict(color="#73b82f", width=3),
        ), row=1, col=2)
        dropout_fig.update_xaxes(
            title_text=pvt_pressure_label,
            showgrid=True,
            gridcolor="#D5E5F7",
            row=1,
            col=1,
        )
        dropout_fig.update_xaxes(
            title_text=x_title,
            showgrid=True,
            gridcolor="#D5E5F7",
            row=1,
            col=2,
        )
        dropout_fig.update_yaxes(
            title_text="Liquid dropout, % PV",
            showgrid=True,
            gridcolor="#D5E5F7",
            row=1,
            col=1,
        )
        dropout_fig.update_yaxes(
            title_text="Liquid dropout, % PV",
            showgrid=True,
            gridcolor="#D5E5F7",
            row=1,
            col=2,
        )
        dropout_fig.update_layout(
            template="plotly_white",
            height=560,
            title_text="Retrograde Condensate Dropout Diagnostics",
            title_x=0.5,
        )
        st.plotly_chart(dropout_fig, use_container_width=True)

    # -----------------------------------------------------
    # Gas-condensate and gas-water relative permeability
    # -----------------------------------------------------
    st.header("Gas-Field Relative Permeability – Corey Function Playground")

    with st.sidebar.expander("Gas-Condensate Corey Controls", expanded=False):
        st.markdown("**Before History Match**")
        gc_swc_b = st.slider("GC Before Swc", 0.05, 0.40, step=0.01, key="gc_swc_b")
        gc_scc_b = st.slider("GC Before Scc", 0.00, 0.25, step=0.01, key="gc_scc_b")
        gc_sgr_b = st.slider("GC Before Sgr", 0.00, 0.30, step=0.01, key="gc_sgr_b")
        gc_krg_end_b = st.slider("GC Before Krg end", 0.10, 1.00, step=0.01, key="gc_krg_end_b")
        gc_krc_end_b = st.slider("GC Before Krc end", 0.10, 1.00, step=0.01, key="gc_krc_end_b")
        gc_ng_b = st.slider("GC Before ng", 1.0, 8.0, step=0.05, key="gc_ng_b")
        gc_nc_b = st.slider("GC Before nc", 1.0, 8.0, step=0.05, key="gc_nc_b")

        st.markdown("**After History Match**")
        gc_swc_a = st.slider("GC After Swc", 0.05, 0.40, step=0.01, key="gc_swc_a")
        gc_scc_a = st.slider("GC After Scc", 0.00, 0.25, step=0.01, key="gc_scc_a")
        gc_sgr_a = st.slider("GC After Sgr", 0.00, 0.30, step=0.01, key="gc_sgr_a")
        gc_krg_end_a = st.slider("GC After Krg end", 0.10, 1.00, step=0.01, key="gc_krg_end_a")
        gc_krc_end_a = st.slider("GC After Krc end", 0.10, 1.00, step=0.01, key="gc_krc_end_a")
        gc_ng_a = st.slider("GC After ng", 1.0, 8.0, step=0.05, key="gc_ng_a")
        gc_nc_a = st.slider("GC After nc", 1.0, 8.0, step=0.05, key="gc_nc_a")

    with st.sidebar.expander("Gas-Water Corey Controls", expanded=False):
        st.markdown("**Before History Match**")
        gw_swc_b = st.slider("GW Before Swc", 0.05, 0.45, step=0.01, key="gw_swc_b")
        gw_sgr_b = st.slider("GW Before Sgr", 0.00, 0.35, step=0.01, key="gw_sgr_b")
        gw_krw_end_b = st.slider("GW Before Krw end", 0.10, 1.00, step=0.01, key="gw_krw_end_b")
        gw_krg_end_b = st.slider("GW Before Krg end", 0.10, 1.00, step=0.01, key="gw_krg_end_b")
        gw_nw_b = st.slider("GW Before nw", 1.0, 8.0, step=0.05, key="gw_nw_b")
        gw_ng_b = st.slider("GW Before ng", 1.0, 8.0, step=0.05, key="gw_ng_b")

        st.markdown("**After History Match**")
        gw_swc_a = st.slider("GW After Swc", 0.05, 0.45, step=0.01, key="gw_swc_a")
        gw_sgr_a = st.slider("GW After Sgr", 0.00, 0.35, step=0.01, key="gw_sgr_a")
        gw_krw_end_a = st.slider("GW After Krw end", 0.10, 1.00, step=0.01, key="gw_krw_end_a")
        gw_krg_end_a = st.slider("GW After Krg end", 0.10, 1.00, step=0.01, key="gw_krg_end_a")
        gw_nw_a = st.slider("GW After nw", 1.0, 8.0, step=0.05, key="gw_nw_a")
        gw_ng_a = st.slider("GW After ng", 1.0, 8.0, step=0.05, key="gw_ng_a")

    sc = np.linspace(0.0, 0.80, 161)
    sw = np.linspace(0.0, 1.00, 161)
    krg_gc_b, krc_b = corey_gas_condensate(
        sc, gc_swc_b, gc_scc_b, gc_sgr_b,
        gc_krg_end_b, gc_krc_end_b, gc_ng_b, gc_nc_b,
    )
    krg_gc_a, krc_a = corey_gas_condensate(
        sc, gc_swc_a, gc_scc_a, gc_sgr_a,
        gc_krg_end_a, gc_krc_end_a, gc_ng_a, gc_nc_a,
    )
    krg_gw_b, krw_gw_b = corey_gas_water(
        sw, gw_swc_b, gw_sgr_b,
        gw_krw_end_b, gw_krg_end_b, gw_nw_b, gw_ng_b,
    )
    krg_gw_a, krw_gw_a = corey_gas_water(
        sw, gw_swc_a, gw_sgr_a,
        gw_krw_end_a, gw_krg_end_a, gw_nw_a, gw_ng_a,
    )

    gc_tab, gw_tab, overlay_tab = st.tabs([
        "Gas-Condensate RelPerm",
        "Gas-Water RelPerm",
        "Before vs After Overlay",
    ])

    with gc_tab:
        gc_fig = go.Figure()
        gc_fig.add_trace(go.Scatter(x=sc, y=krg_gc_b, mode="lines", name="KRG before HM"))
        gc_fig.add_trace(go.Scatter(x=sc, y=krc_b, mode="lines", name="KRC before HM"))
        gc_fig.add_trace(go.Scatter(x=sc, y=krg_gc_a, mode="lines", name="KRG after HM", line=dict(dash="dash")))
        gc_fig.add_trace(go.Scatter(x=sc, y=krc_a, mode="lines", name="KRC after HM", line=dict(dash="dash")))
        gc_fig.update_layout(
            title="Gas-Condensate Relative Permeabilities – Corey Function",
            xaxis_title="Condensate saturation, Sc",
            yaxis_title="Relative Permeability",
            template="plotly_white",
            height=540,
        )
        gc_fig.update_xaxes(range=[0, 0.8], showgrid=True, gridcolor="#D5E5F7")
        gc_fig.update_yaxes(range=[0, 1.05], showgrid=True, gridcolor="#D5E5F7")
        st.plotly_chart(gc_fig, use_container_width=True)

    with gw_tab:
        gw_fig = go.Figure()
        gw_fig.add_trace(go.Scatter(x=sw, y=krg_gw_b, mode="lines", name="KRG before HM"))
        gw_fig.add_trace(go.Scatter(x=sw, y=krw_gw_b, mode="lines", name="KRW before HM"))
        gw_fig.add_trace(go.Scatter(x=sw, y=krg_gw_a, mode="lines", name="KRG after HM", line=dict(dash="dash")))
        gw_fig.add_trace(go.Scatter(x=sw, y=krw_gw_a, mode="lines", name="KRW after HM", line=dict(dash="dash")))
        gw_fig.update_layout(
            title="Gas-Water Relative Permeabilities – Corey Function",
            xaxis_title="Water saturation, Sw",
            yaxis_title="Relative Permeability",
            template="plotly_white",
            height=540,
        )
        gw_fig.update_xaxes(range=[0, 1], showgrid=True, gridcolor="#D5E5F7")
        gw_fig.update_yaxes(range=[0, 1.05], showgrid=True, gridcolor="#D5E5F7")
        st.plotly_chart(gw_fig, use_container_width=True)

    with overlay_tab:
        overlay_fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                "Gas-Condensate Before vs After",
                "Gas-Water Before vs After",
            ),
        )
        for values, name, dash in [
            (krg_gc_b, "GC KRG before", None),
            (krc_b, "GC KRC before", None),
            (krg_gc_a, "GC KRG after", "dash"),
            (krc_a, "GC KRC after", "dash"),
        ]:
            overlay_fig.add_trace(
                go.Scatter(x=sc, y=values, mode="lines", name=name, line=dict(dash=dash)),
                row=1,
                col=1,
            )
        for values, name, dash in [
            (krg_gw_b, "GW KRG before", None),
            (krw_gw_b, "GW KRW before", None),
            (krg_gw_a, "GW KRG after", "dash"),
            (krw_gw_a, "GW KRW after", "dash"),
        ]:
            overlay_fig.add_trace(
                go.Scatter(x=sw, y=values, mode="lines", name=name, line=dict(dash=dash)),
                row=1,
                col=2,
            )
        overlay_fig.update_xaxes(title_text="Sc", range=[0, 0.8], row=1, col=1)
        overlay_fig.update_xaxes(title_text="Sw", range=[0, 1], row=1, col=2)
        overlay_fig.update_yaxes(title_text="RelPerm", range=[0, 1.05], row=1, col=1)
        overlay_fig.update_yaxes(title_text="RelPerm", range=[0, 1.05], row=1, col=2)
        overlay_fig.update_layout(
            title_text="Gas-Field Relative Permeability Overlay",
            title_x=0.5,
            template="plotly_white",
            height=560,
        )
        overlay_fig.update_xaxes(showgrid=True, gridcolor="#D5E5F7")
        overlay_fig.update_yaxes(showgrid=True, gridcolor="#D5E5F7")
        st.plotly_chart(overlay_fig, use_container_width=True)

    # -----------------------------------------------------
    # Export tables and downloads
    # -----------------------------------------------------
    history_df = pd.DataFrame({
        "Date": dates,
        "Time_Years": t_years,
        "Gas_Rate_Measured_10e3sm3d": gas_rate_measured,
        "Gas_Rate_Model_10e3sm3d": gas_rate_model,
        "Condensate_Rate_Measured_m3d": condensate_rate_measured,
        "Condensate_Rate_Model_m3d": condensate_rate_model,
        "Water_Rate_Measured_m3d": water_rate_measured,
        "Water_Rate_Model_m3d": water_rate_model,
        "CGR_Measured_m3_per_10e3sm3": cgr_measured,
        "CGR_Model_m3_per_10e3sm3": cgr_model,
        "Cumulative_Gas_Measured_BSm3": cumulative_gas_measured,
        "Cumulative_Gas_Model_BSm3": cumulative_gas_model,
        "Cumulative_Condensate_Measured_MMm3": cumulative_condensate_measured,
        "Cumulative_Condensate_Model_MMm3": cumulative_condensate_model,
        "Reservoir_Pressure_Measured_bar": pressure_measured,
        "Reservoir_Pressure_Model_bar": pressure_model,
        "Gas_Recovery_Measured_pct_GIIP": gas_recovery_measured,
        "Gas_Recovery_Model_pct_GIIP": gas_recovery_model,
        "Z_Factor_Measured_Pressure": z_history_measured,
        "Z_Factor_Model_Pressure": z_history_model,
        "Bg_Measured_rm3_per_sm3": bg_history_measured,
        "Bg_Model_rm3_per_sm3": bg_history_model,
        "Gas_Viscosity_MeasuredPressure_cP": mug_history_measured,
        "Gas_Viscosity_ModelPressure_cP": mug_history_model,
        "Gas_Density_MeasuredPressure_kgm3": rho_history_measured,
        "Gas_Density_ModelPressure_kgm3": rho_history_model,
        "Liquid_Dropout_MeasuredPressure_pctPV": dropout_history_measured,
        "Liquid_Dropout_ModelPressure_pctPV": dropout_history_model,
    })

    pvt_df = pd.DataFrame({
        "Pressure_bar": pvt_pressure_bar,
        "Pressure_psi": pvt_pressure_bar * 14.5037738,
        "Z_Factor": z_pvt,
        "Bg_rm3_per_sm3": bg_pvt,
        "Bg_rb_per_Mscf": bg_pvt * (6.28981 / 0.0353147),
        "Gas_Viscosity_cP": mug_pvt,
        "Gas_Density_kgm3": rho_pvt,
        "Gas_Density_lbft3": rho_pvt * 0.06242796,
        "Retrograde_Liquid_Dropout_pctPV": dropout_pvt,
    })
    relperm_gc_df = pd.DataFrame({
        "Sc": sc,
        "KRG_before_HM": krg_gc_b,
        "KRC_before_HM": krc_b,
        "KRG_after_HM": krg_gc_a,
        "KRC_after_HM": krc_a,
    })
    relperm_gw_df = pd.DataFrame({
        "Sw": sw,
        "KRG_before_HM": krg_gw_b,
        "KRW_before_HM": krw_gw_b,
        "KRG_after_HM": krg_gw_a,
        "KRW_after_HM": krw_gw_a,
    })
    current_values = {
        key: st.session_state[key]
        for key in DEFAULTS.keys()
        if key in st.session_state
    }
    controls_df = pd.DataFrame({
        "Parameter": list(current_values.keys()),
        "Value": list(current_values.values()),
    })

    with st.expander("Show generated gas-field data tables"):
        st.subheader("Gas-Condensate History Match Data – Metric Base")
        st.dataframe(history_df, use_container_width=True)
        st.subheader("Gas PVT")
        st.dataframe(pvt_df, use_container_width=True)
        st.subheader("Gas-Condensate Relative Permeability")
        st.dataframe(relperm_gc_df, use_container_width=True)
        st.subheader("Gas-Water Relative Permeability")
        st.dataframe(relperm_gw_df, use_container_width=True)
        st.subheader("History-Match Metrics")
        st.dataframe(match_metrics_df, use_container_width=True)
        st.subheader("Controls")
        st.dataframe(controls_df, use_container_width=True)

    excel_file = make_excel_sheets({
        "History_Match_Metric": history_df,
        "Gas_PVT": pvt_df,
        "RelPerm_GasCond": relperm_gc_df,
        "RelPerm_GasWater": relperm_gw_df,
        "Match_Metrics": match_metrics_df,
        "Controls": controls_df,
    })
    st.download_button(
        label="Download Excel – Gas-Condensate History Match, PVT & RelPerm",
        data=excel_file,
        file_name="Generic_Gas_Field_HistoryMatch_PVT_RelPerm.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    json_payload = {
        "app": "Generic Oil & Gas Field Artificial History Match, PVT & RelPerm Generator",
        "version": "5.0",
        "active_field_type": "Gas Field",
        "values": current_values,
    }
    st.download_button(
        label="Save Current Settings as JSON",
        data=json.dumps(json_payload, indent=2).encode("utf-8"),
        file_name="Generic_Gas_Field_HistoryMatch_Settings.json",
        mime="application/json",
    )


# =========================================================
# App title
# =========================================================
st.title("Generic Oil & Gas Field – Artificial History Match, PVT & RelPerm Generator")
st.caption(
    "Dual-mode synthetic field generator. Oil Field preserves the original "
    "oil-water-gas workflow; Gas Field adds gas-condensate production, PVT, "
    "P/Z and gas-field relative-permeability diagnostics."
)

# =========================================================
# JSON load / reset
# =========================================================
st.sidebar.header("Load / Save Settings")

uploaded_json = st.sidebar.file_uploader("Load saved JSON settings", type=["json"])

if uploaded_json is not None:
    if st.sidebar.button("Apply loaded JSON"):
        loaded = json.load(uploaded_json)
        values = loaded.get("values", loaded)

        for k, v in values.items():
            if k in DEFAULTS:
                st.session_state[k] = v

        st.rerun()

if st.sidebar.button("Reset to default settings"):
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.rerun()

# =========================================================
# Sidebar controls
# =========================================================
st.sidebar.header("Field Setup")

field_type = st.sidebar.radio(
    "Field type",
    ["Oil Field", "Gas Field"],
    key="field_type",
    horizontal=True,
)

if field_type == "Oil Field":
    field_name = st.sidebar.text_input("Field name", key="field_name")
else:
    field_name = st.sidebar.text_input("Field name", key="gas_field_name")

seed = st.sidebar.number_input("Random seed", min_value=1, max_value=999999, step=1, key="seed")

start_year = st.sidebar.slider("History start year", 1950, 2030, step=1, key="start_year")
history_years = st.sidebar.slider("Total history length, years", 3, 80, step=1, key="history_years")

time_frequency = st.sidebar.radio(
    "History data frequency",
    ["Monthly", "Quarterly"],
    key="time_frequency"
)

st.sidebar.header("Plot View Controls")

unit_system = st.sidebar.radio(
    "Unit system",
    ["Metric", "Field"],
    key="unit_system",
)

x_axis_mode = st.sidebar.radio(
    "X-axis mode",
    ["Date", "Years"],
    key="x_axis_mode",
)

plot_start_min = int(start_year)
plot_start_max = int(start_year + history_years - 1)
if not plot_start_min <= int(st.session_state["plot_start_year"]) <= plot_start_max:
    st.session_state["plot_start_year"] = plot_start_min

plot_start_year = st.sidebar.slider(
    "Plot start year",
    plot_start_min,
    plot_start_max,
    key="plot_start_year",
    step=1,
)

max_plot_years = max(1, int(start_year + history_years - plot_start_year))
if st.session_state["plot_years"] > max_plot_years:
    st.session_state["plot_years"] = max_plot_years

plot_years = st.sidebar.slider(
    "Number of years to display",
    1,
    max_plot_years,
    key="plot_years",
    step=1,
)

if field_type == "Gas Field":
    render_gas_field(
        field_name=field_name,
        seed=seed,
        start_year=start_year,
        history_years=history_years,
        time_frequency=time_frequency,
        unit_system=unit_system,
        x_axis_mode=x_axis_mode,
        plot_start_year=plot_start_year,
        plot_years=plot_years,
    )
    st.stop()

st.sidebar.header("Measured Field History Controls")

with st.sidebar.expander("Production Profile", expanded=True):
    initial_liquid_rate = st.slider("Initial liquid rate, m³/d", 50.0, 3000.0, step=10.0, key="initial_liquid_rate")
    peak_liquid_rate = st.slider("Peak liquid rate, m³/d", 100.0, 6000.0, step=10.0, key="peak_liquid_rate")
    plateau_years = st.slider("Plateau years", 0.0, 20.0, step=0.5, key="plateau_years")
    annual_decline_pct = st.slider("Annual decline, %", 1.0, 25.0, step=0.5, key="annual_decline_pct")
    late_redevelopment = st.checkbox("Add late redevelopment / infill hump", key="late_redevelopment")
    st.session_state["redevelopment_year"] = min(
        float(st.session_state["redevelopment_year"]),
        float(history_years),
    )
    redevelopment_year = st.slider("Redevelopment timing, years from start", 1.0, float(history_years), step=0.5, key="redevelopment_year")
    redevelopment_strength = st.slider("Redevelopment strength", 0.0, 1.5, step=0.05, key="redevelopment_strength")
    liquid_noise_pct = st.slider("Measured liquid noise, %", 0.0, 20.0, step=0.5, key="liquid_noise_pct")

with st.sidebar.expander("Water Cut Profile", expanded=True):
    initial_water_cut_pct = st.slider("Initial water cut, %", 0.0, 60.0, step=1.0, key="initial_water_cut_pct")
    maximum_water_cut_pct = st.slider("Maximum water cut, %", 30.0, 95.0, step=1.0, key="maximum_water_cut_pct")
    st.session_state["water_cut_rise_year"] = min(
        float(st.session_state["water_cut_rise_year"]),
        float(history_years),
    )
    water_cut_rise_year = st.slider("Water cut rise timing, years from start", 0.5, float(history_years), step=0.5, key="water_cut_rise_year")
    water_cut_rise_speed = st.slider("Water cut rise speed", 0.5, 8.0, step=0.1, key="water_cut_rise_speed")
    water_cut_noise_pct = st.slider("Measured WCUT noise, %", 0.0, 10.0, step=0.2, key="water_cut_noise_pct")

with st.sidebar.expander("GOR / Gas Profile", expanded=False):
    initial_gor = st.slider("Initial GOR, m³/m³", 10.0, 800.0, step=5.0, key="initial_gor")
    maximum_gor = st.slider("Maximum GOR, m³/m³", 50.0, 2000.0, step=10.0, key="maximum_gor")
    gor_pressure_sensitivity = st.slider("GOR pressure sensitivity", 0.0, 3.0, step=0.1, key="gor_pressure_sensitivity")
    gor_noise_pct = st.slider("Measured GOR noise, %", 0.0, 20.0, step=0.5, key="gor_noise_pct")

with st.sidebar.expander("Pressure Profile", expanded=False):
    initial_pressure_bar = st.slider("Initial pressure, bar", 80.0, 600.0, step=5.0, key="initial_pressure_bar")
    abandonment_pressure_bar = st.slider("Late pressure, bar", 30.0, 400.0, step=5.0, key="abandonment_pressure_bar")
    aquifer_support = st.slider("Aquifer / pressure support", 0.0, 0.9, step=0.05, key="aquifer_support")
    pressure_noise_bar = st.slider("Measured pressure noise, bar", 0.0, 20.0, step=0.5, key="pressure_noise_bar")

st.sidebar.header("Model Match Controls")
liq_cfg = model_control_group("Total Liquid / Oil-Water System", "liq")
wcut_cfg = model_control_group("Water Cut System", "wcut")
gor_cfg = model_control_group("GOR / Gas System", "gor")
prs_cfg = model_control_group("Reservoir Pressure System", "prs")

st.sidebar.header("Display Controls")
oil_disp = display_control_group("Oil Rate Display", "oil")
gas_disp = display_control_group("Gas Rate Display", "gas")
water_disp = display_control_group("Water Rate Display", "water")
wcut_disp = display_control_group("Water Cut Display", "wcut")
gor_disp = display_control_group("GOR Display", "gor")
prs_disp = display_control_group("Pressure Display", "prs")
cum_disp = display_control_group("Cumulative Oil Display", "cum")

# =========================================================
# Generate measured synthetic field history
# =========================================================
freq = "MS" if time_frequency == "Monthly" else "QS"
periods = int(history_years * (12 if time_frequency == "Monthly" else 4))
dates = pd.date_range(f"{int(start_year)}-01-01", periods=periods, freq=freq)
t_years = np.arange(periods) / (12 if time_frequency == "Monthly" else 4)
rng = np.random.default_rng(seed)

# Liquid profile
ramp = initial_liquid_rate + (peak_liquid_rate - initial_liquid_rate) * (1 - np.exp(-t_years / 0.9))
decline_start = plateau_years
decline_factor = np.exp(-annual_decline_pct / 100.0 * np.maximum(t_years - decline_start, 0))
total_liq_clean = ramp * decline_factor

if late_redevelopment:
    hump_width = max(1.0, history_years / 9.0)
    hump = redevelopment_strength * peak_liquid_rate * np.exp(-0.5 * ((t_years - redevelopment_year) / hump_width) ** 2)
    total_liq_clean += hump

seasonality = 1 + 0.025 * np.sin(2 * np.pi * t_years)
total_liq_measured = total_liq_clean * seasonality
total_liq_measured *= 1 + rng.normal(0, liquid_noise_pct / 100.0, periods)
total_liq_measured = np.clip(total_liq_measured, 1, None)

# Water cut profile capped below 100
wc_mid = water_cut_rise_year
wc_speed = water_cut_rise_speed
wcut_clean = initial_water_cut_pct + (maximum_water_cut_pct - initial_water_cut_pct) / (
    1 + np.exp(-(t_years - wc_mid) / wc_speed)
)
wcut_clean += 1.5 * np.sin(2 * np.pi * t_years / 5.5)
wcut_measured = wcut_clean + rng.normal(0, water_cut_noise_pct, periods)
wcut_measured = np.clip(wcut_measured, 0, maximum_water_cut_pct)

# Oil and water are derived from liquid and WCUT
oil_measured = total_liq_measured * (1 - wcut_measured / 100.0)
water_measured = total_liq_measured * (wcut_measured / 100.0)

# Cumulative oil
cum_oil_measured = calc_cumulative(oil_measured, dates)

# Pressure from cumulative offtake and support
cum_norm = cum_oil_measured / max(cum_oil_measured.max(), 1e-9)
pressure_drop = (initial_pressure_bar - abandonment_pressure_bar) * (1 - aquifer_support)
pressure_clean = initial_pressure_bar - pressure_drop * (cum_norm ** 0.75)
pressure_clean += 2.0 * np.sin(2 * np.pi * t_years / 8.0)
pressure_measured = pressure_clean + rng.normal(0, pressure_noise_bar, periods)
pressure_measured = np.clip(pressure_measured, abandonment_pressure_bar, initial_pressure_bar * 1.05)

# GOR linked to pressure depletion; bounded
pressure_depletion = np.clip((initial_pressure_bar - pressure_measured) / max(initial_pressure_bar - abandonment_pressure_bar, 1e-9), 0, 1)
gor_clean = initial_gor + (maximum_gor - initial_gor) * (pressure_depletion ** gor_pressure_sensitivity)
gor_clean += 0.05 * initial_gor * np.sin(2 * np.pi * t_years / 4.0)
gor_measured = gor_clean * (1 + rng.normal(0, gor_noise_pct / 100.0, periods))
gor_measured = np.clip(gor_measured, 1, maximum_gor)

# Gas rate derived from oil × GOR
gas_measured = oil_measured * gor_measured / 1000.0  # 10³ m³/d

# =========================================================
# Model curves and material-balanced derived model
# =========================================================
total_liq_model = build_model_curve(
    total_liq_measured,
    seed=seed + 1,
    min_value=0,
    max_value=None,
    **liq_cfg
)

wcut_model = build_model_curve(
    wcut_measured,
    seed=seed + 2,
    min_value=0,
    max_value=maximum_water_cut_pct,
    **wcut_cfg
)

gor_model = build_model_curve(
    gor_measured,
    seed=seed + 3,
    min_value=1,
    max_value=maximum_gor,
    **gor_cfg
)

oil_model = total_liq_model * (1 - wcut_model / 100.0)
water_model = total_liq_model * (wcut_model / 100.0)
gas_model = oil_model * gor_model / 1000.0
cum_oil_model = calc_cumulative(oil_model, dates)

cum_model_norm = cum_oil_model / max(cum_oil_model.max(), 1e-9)
pressure_consistent_model = initial_pressure_bar - pressure_drop * (cum_model_norm ** 0.75)
pressure_model = build_model_curve(
    pressure_consistent_model,
    seed=seed + 4,
    min_value=abandonment_pressure_bar,
    max_value=initial_pressure_bar * 1.05,
    **prs_cfg
)

# =========================================================
# Display unit conversion and plot window
# =========================================================
display = convert_display_units(
    unit_system,
    oil_measured, oil_model,
    gas_measured, gas_model,
    water_measured, water_model,
    wcut_measured, wcut_model,
    gor_measured, gor_model,
    pressure_measured, pressure_model,
    cum_oil_measured, cum_oil_model,
)

plot_mask, x_plot, x_title = make_plot_window(
    dates,
    x_axis_mode,
    plot_start_year,
    plot_years,
)

# =========================================================
# Plot history match
# =========================================================
history_fig = make_subplots(
    rows=4,
    cols=2,
    specs=[
        [{}, {}],
        [{}, {}],
        [{}, {}],
        [{"colspan": 2}, None],
    ],
    subplot_titles=(
        "OIL_RATE – HISTORY MATCH",
        "GAS_RATE – HISTORY MATCH",
        "WATER_RATE – HISTORY MATCH",
        "WATER CUT – HISTORY MATCH",
        "GOR – HISTORY MATCH",
        "RESERVOIR PRESSURE – HISTORY MATCH",
        "CUMULATIVE OIL – HISTORY MATCH",
    ),
    vertical_spacing=0.105,
    horizontal_spacing=0.08,
)

add_history_trace(
    history_fig, 1, 1,
    x_plot,
    display["oil_measured"][plot_mask],
    display["oil_model"][plot_mask],
    display["oil_label"],
    oil_disp["meas_color"],
    oil_disp["model_color"],
    oil_disp["dot_size"],
    oil_disp["dot_stride"],
    x_title=x_title,
    show_legend=True,
)

add_history_trace(
    history_fig, 1, 2,
    x_plot,
    display["gas_measured"][plot_mask],
    display["gas_model"][plot_mask],
    display["gas_label"],
    gas_disp["meas_color"],
    gas_disp["model_color"],
    gas_disp["dot_size"],
    gas_disp["dot_stride"],
    x_title=x_title,
)

add_history_trace(
    history_fig, 2, 1,
    x_plot,
    display["water_measured"][plot_mask],
    display["water_model"][plot_mask],
    display["water_label"],
    water_disp["meas_color"],
    water_disp["model_color"],
    water_disp["dot_size"],
    water_disp["dot_stride"],
    x_title=x_title,
)

add_history_trace(
    history_fig, 2, 2,
    x_plot,
    display["wcut_measured"][plot_mask],
    display["wcut_model"][plot_mask],
    display["wcut_label"],
    wcut_disp["meas_color"],
    wcut_disp["model_color"],
    wcut_disp["dot_size"],
    wcut_disp["dot_stride"],
    x_title=x_title,
)

add_history_trace(
    history_fig, 3, 1,
    x_plot,
    display["gor_measured"][plot_mask],
    display["gor_model"][plot_mask],
    display["gor_label"],
    gor_disp["meas_color"],
    gor_disp["model_color"],
    gor_disp["dot_size"],
    gor_disp["dot_stride"],
    x_title=x_title,
)

add_history_trace(
    history_fig, 3, 2,
    x_plot,
    display["pressure_measured"][plot_mask],
    display["pressure_model"][plot_mask],
    display["pressure_label"],
    prs_disp["meas_color"],
    prs_disp["model_color"],
    prs_disp["dot_size"],
    prs_disp["dot_stride"],
    x_title=x_title,
)

add_history_trace(
    history_fig, 4, 1,
    x_plot,
    display["cum_oil_measured"][plot_mask],
    display["cum_oil_model"][plot_mask],
    display["cum_oil_label"],
    cum_disp["meas_color"],
    cum_disp["model_color"],
    cum_disp["dot_size"],
    cum_disp["dot_stride"],
    x_title=x_title,
)

history_fig.update_layout(
    title_text=(
        f"{field_name} – Material-Balanced Artificial History Match "
        f"({unit_system}, X-axis: {x_axis_mode}, {plot_years} years from {plot_start_year})"
    ),
    title_x=0.5,
    height=1350,
    template="plotly_white",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.055,
        xanchor="center",
        x=0.5,
    ),
)

history_fig.update_xaxes(showgrid=True, gridcolor="#D5E5F7")
history_fig.update_yaxes(showgrid=True, gridcolor="#D5E5F7")

if x_axis_mode == "Years":
    history_fig.update_xaxes(dtick=1)

st.plotly_chart(history_fig, use_container_width=True)

# =========================================================
# Checks
# =========================================================
st.subheader("Material Balance / Consistency Checks")

c1, c2, c3, c4, c5 = st.columns(5)

oil_balance = total_liq_measured * (1 - wcut_measured / 100.0)
water_balance = total_liq_measured * (wcut_measured / 100.0)
gas_balance = oil_measured * gor_measured / 1000.0

c1.metric("Max oil balance error", f"{np.max(np.abs(oil_measured - oil_balance)):.4f} m³/d")
c2.metric("Max water balance error", f"{np.max(np.abs(water_measured - water_balance)):.4f} m³/d")
c3.metric("Max gas/GOR error", f"{np.max(np.abs(gas_measured - gas_balance)):.4f} 10³ m³/d")
c4.metric("Final water cut", f"{wcut_measured[-1]:.1f}%")
c5.metric("Final Cum Oil", f"{cum_oil_measured[-1]:.3f} MMm³")

if wcut_measured.max() >= 99:
    st.warning("Water cut is very close to 100%. Reduce Maximum water cut if you want an earlier-life or mid-life field.")
else:
    st.success(f"Water cut is capped below 100%. Maximum measured WCUT = {wcut_measured.max():.1f}%.")

# =========================================================
# Relative permeability
# =========================================================
st.header("Relative Permeability – Corey Function Playground")

with st.sidebar.expander("Water-Oil Corey Controls", expanded=False):
    st.markdown("**Before History Match**")
    swc_b = st.slider("WO Before Swc", 0.05, 0.45, step=0.01, key="swc_b")
    sorw_b = st.slider("WO Before Sorw", 0.05, 0.45, step=0.01, key="sorw_b")
    krw_end_b = st.slider("WO Before Krw end", 0.10, 1.00, step=0.01, key="krw_end_b")
    krow_end_b = st.slider("WO Before Kro end", 0.10, 1.00, step=0.01, key="krow_end_b")
    nw_b = st.slider("WO Before nw", 1.0, 6.0, step=0.05, key="nw_b")
    no_b = st.slider("WO Before no", 1.0, 6.0, step=0.05, key="no_b")

    st.markdown("**After History Match**")
    swc_a = st.slider("WO After Swc", 0.05, 0.45, step=0.01, key="swc_a")
    sorw_a = st.slider("WO After Sorw", 0.05, 0.45, step=0.01, key="sorw_a")
    krw_end_a = st.slider("WO After Krw end", 0.10, 1.00, step=0.01, key="krw_end_a")
    krow_end_a = st.slider("WO After Kro end", 0.10, 1.00, step=0.01, key="krow_end_a")
    nw_a = st.slider("WO After nw", 1.0, 6.0, step=0.05, key="nw_a")
    no_a = st.slider("WO After no", 1.0, 6.0, step=0.05, key="no_a")

with st.sidebar.expander("Gas-Oil Corey Controls", expanded=False):
    st.markdown("**Before History Match**")
    sgc_b = st.slider("GO Before Sgc", 0.00, 0.30, step=0.01, key="sgc_b")
    sorg_b = st.slider("GO Before Sorg", 0.05, 0.45, step=0.01, key="sorg_b")
    krg_end_b = st.slider("GO Before Krg end", 0.10, 1.00, step=0.01, key="krg_end_b")
    krog_end_b = st.slider("GO Before Krog end", 0.10, 1.00, step=0.01, key="krog_end_b")
    ng_b = st.slider("GO Before ng", 1.0, 6.0, step=0.05, key="ng_b")
    nog_b = st.slider("GO Before nog", 1.0, 6.0, step=0.05, key="nog_b")

    st.markdown("**After History Match**")
    sgc_a = st.slider("GO After Sgc", 0.00, 0.30, step=0.01, key="sgc_a")
    sorg_a = st.slider("GO After Sorg", 0.05, 0.45, step=0.01, key="sorg_a")
    krg_end_a = st.slider("GO After Krg end", 0.10, 1.00, step=0.01, key="krg_end_a")
    krog_end_a = st.slider("GO After Krog end", 0.10, 1.00, step=0.01, key="krog_end_a")
    ng_a = st.slider("GO After ng", 1.0, 6.0, step=0.05, key="ng_a")
    nog_a = st.slider("GO After nog", 1.0, 6.0, step=0.05, key="nog_a")

sw = np.linspace(0, 1, 101)
sg = np.linspace(0, 1, 101)

krw_b, krow_b = corey_water_oil(sw, swc_b, sorw_b, krw_end_b, krow_end_b, nw_b, no_b)
krw_a, krow_a = corey_water_oil(sw, swc_a, sorw_a, krw_end_a, krow_end_a, nw_a, no_a)

krg_b, krog_b = corey_gas_oil(sg, sgc_b, sorg_b, krg_end_b, krog_end_b, ng_b, nog_b)
krg_a, krog_a = corey_gas_oil(sg, sgc_a, sorg_a, krg_end_a, krog_end_a, ng_a, nog_a)

tab1, tab2, tab3 = st.tabs([
    "Water-Oil RelPerm",
    "Gas-Oil RelPerm",
    "Overlay Before vs After",
])

with tab1:
    fig_wo = go.Figure()
    fig_wo.add_trace(go.Scatter(x=sw, y=krow_b, mode="lines", name="KROW before HM"))
    fig_wo.add_trace(go.Scatter(x=sw, y=krw_b, mode="lines", name="KRW before HM"))
    fig_wo.add_trace(go.Scatter(x=sw, y=krow_a, mode="lines", name="KROW after HM", line=dict(dash="dash")))
    fig_wo.add_trace(go.Scatter(x=sw, y=krw_a, mode="lines", name="KRW after HM", line=dict(dash="dash")))
    fig_wo.update_layout(
        title="Water-Oil Relative Permeabilities – Corey Function",
        xaxis_title="Sw",
        yaxis_title="Relative Permeability",
        template="plotly_white",
        height=520,
    )
    fig_wo.update_xaxes(range=[0, 1], showgrid=True, gridcolor="#D5E5F7")
    fig_wo.update_yaxes(range=[0, 1.05], showgrid=True, gridcolor="#D5E5F7")
    st.plotly_chart(fig_wo, use_container_width=True)

with tab2:
    fig_go = go.Figure()
    fig_go.add_trace(go.Scatter(x=sg, y=krog_b, mode="lines", name="KROG before HM"))
    fig_go.add_trace(go.Scatter(x=sg, y=krg_b, mode="lines", name="KRG before HM"))
    fig_go.add_trace(go.Scatter(x=sg, y=krog_a, mode="lines", name="KROG after HM", line=dict(dash="dash")))
    fig_go.add_trace(go.Scatter(x=sg, y=krg_a, mode="lines", name="KRG after HM", line=dict(dash="dash")))
    fig_go.update_layout(
        title="Gas-Oil Relative Permeabilities – Corey Function",
        xaxis_title="Sg",
        yaxis_title="Relative Permeability",
        template="plotly_white",
        height=520,
    )
    fig_go.update_xaxes(range=[0, 1], showgrid=True, gridcolor="#D5E5F7")
    fig_go.update_yaxes(range=[0, 1.05], showgrid=True, gridcolor="#D5E5F7")
    st.plotly_chart(fig_go, use_container_width=True)

with tab3:
    overlay_type = st.radio(
        "Overlay type",
        ["Water-Oil only", "Gas-Oil only", "Both"],
        horizontal=True,
    )

    if overlay_type == "Both":
        fig_overlay = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Water-Oil Before vs After", "Gas-Oil Before vs After"),
        )
    else:
        fig_overlay = make_subplots(
            rows=1,
            cols=1,
            subplot_titles=(f"{overlay_type} Before vs After",),
        )

    if overlay_type in ["Water-Oil only", "Both"]:
        c = 1
        fig_overlay.add_trace(go.Scatter(x=sw, y=krow_b, mode="lines", name="KROW before"), row=1, col=c)
        fig_overlay.add_trace(go.Scatter(x=sw, y=krw_b, mode="lines", name="KRW before"), row=1, col=c)
        fig_overlay.add_trace(go.Scatter(x=sw, y=krow_a, mode="lines", name="KROW after", line=dict(dash="dash")), row=1, col=c)
        fig_overlay.add_trace(go.Scatter(x=sw, y=krw_a, mode="lines", name="KRW after", line=dict(dash="dash")), row=1, col=c)
        fig_overlay.update_xaxes(title_text="Sw", range=[0, 1], row=1, col=c)
        fig_overlay.update_yaxes(title_text="RelPerm", range=[0, 1.05], row=1, col=c)

    if overlay_type in ["Gas-Oil only", "Both"]:
        c = 2 if overlay_type == "Both" else 1
        fig_overlay.add_trace(go.Scatter(x=sg, y=krog_b, mode="lines", name="KROG before"), row=1, col=c)
        fig_overlay.add_trace(go.Scatter(x=sg, y=krg_b, mode="lines", name="KRG before"), row=1, col=c)
        fig_overlay.add_trace(go.Scatter(x=sg, y=krog_a, mode="lines", name="KROG after", line=dict(dash="dash")), row=1, col=c)
        fig_overlay.add_trace(go.Scatter(x=sg, y=krg_a, mode="lines", name="KRG after", line=dict(dash="dash")), row=1, col=c)
        fig_overlay.update_xaxes(title_text="Sg", range=[0, 1], row=1, col=c)
        fig_overlay.update_yaxes(title_text="RelPerm", range=[0, 1.05], row=1, col=c)

    fig_overlay.update_layout(
        template="plotly_white",
        height=520,
        title_text="Relative Permeability Overlay",
        title_x=0.5,
    )
    fig_overlay.update_xaxes(showgrid=True, gridcolor="#D5E5F7")
    fig_overlay.update_yaxes(showgrid=True, gridcolor="#D5E5F7")
    st.plotly_chart(fig_overlay, use_container_width=True)

# =========================================================
# DataFrames and downloads
# =========================================================
history_df = pd.DataFrame({
    "Date": dates,
    "Time_Years": t_years,
    "Total_Liquid_Measured_m3d": total_liq_measured,
    "Total_Liquid_Model_m3d": total_liq_model,
    "Oil_Rate_Measured_m3d": oil_measured,
    "Oil_Rate_Model_m3d": oil_model,
    "Water_Rate_Measured_m3d": water_measured,
    "Water_Rate_Model_m3d": water_model,
    "Water_Cut_Measured_pct": wcut_measured,
    "Water_Cut_Model_pct": wcut_model,
    "GOR_Measured_m3m3": gor_measured,
    "GOR_Model_m3m3": gor_model,
    "Gas_Rate_Measured_10e3m3d": gas_measured,
    "Gas_Rate_Model_10e3m3d": gas_model,
    "Cumulative_Oil_Measured_MMm3": cum_oil_measured,
    "Cumulative_Oil_Model_MMm3": cum_oil_model,
    "Reservoir_Pressure_Measured_bar": pressure_measured,
    "Reservoir_Pressure_Model_bar": pressure_model,
})

relperm_wo_df = pd.DataFrame({
    "Sw": sw,
    "KRW_before_HM": krw_b,
    "KROW_before_HM": krow_b,
    "KRW_after_HM": krw_a,
    "KROW_after_HM": krow_a,
})

relperm_go_df = pd.DataFrame({
    "Sg": sg,
    "KRG_before_HM": krg_b,
    "KROG_before_HM": krog_b,
    "KRG_after_HM": krg_a,
    "KROG_after_HM": krog_a,
})

current_values = {k: st.session_state[k] for k in DEFAULTS.keys() if k in st.session_state}

controls_df = pd.DataFrame({
    "Parameter": list(current_values.keys()),
    "Value": list(current_values.values()),
})

with st.expander("Show generated data tables"):
    st.subheader("History Match Data – Metric Base")
    st.dataframe(history_df, use_container_width=True)

    st.subheader("Water-Oil RelPerm")
    st.dataframe(relperm_wo_df, use_container_width=True)

    st.subheader("Gas-Oil RelPerm")
    st.dataframe(relperm_go_df, use_container_width=True)

    st.subheader("Controls")
    st.dataframe(controls_df, use_container_width=True)

excel_file = make_excel(history_df, relperm_wo_df, relperm_go_df, controls_df)

st.download_button(
    label="Download Excel – Generic Field Material-Balanced Dataset",
    data=excel_file,
    file_name="Generic_Field_MaterialBalanced_HistoryMatch_RelPerm.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

json_payload = {
    "app": "Generic Oil & Gas Field Artificial History Match, PVT & RelPerm Generator",
    "version": "5.0",
    "active_field_type": "Oil Field",
    "values": current_values,
}

json_bytes = json.dumps(json_payload, indent=2).encode("utf-8")

st.download_button(
    label="Save Current Settings as JSON",
    data=json_bytes,
    file_name="Generic_Field_HistoryMatch_Settings.json",
    mime="application/json",
)
