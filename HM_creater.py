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
    "field_name": "Synthetic Oil Field",
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


# =========================================================
# App title
# =========================================================
st.title("Generic Field – Material-Balanced Artificial History Match & RelPerm Generator")
st.caption(
    "Synthetic field history generator with material-balance consistency. "
    "Water cut is capped below 100% by user control."
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

field_name = st.sidebar.text_input("Field name", key="field_name")
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

plot_start_year = st.sidebar.slider(
    "Plot start year",
    int(start_year),
    int(start_year + history_years - 1),
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

st.sidebar.header("Measured Field History Controls")

with st.sidebar.expander("Production Profile", expanded=True):
    initial_liquid_rate = st.slider("Initial liquid rate, m³/d", 50.0, 3000.0, step=10.0, key="initial_liquid_rate")
    peak_liquid_rate = st.slider("Peak liquid rate, m³/d", 100.0, 6000.0, step=10.0, key="peak_liquid_rate")
    plateau_years = st.slider("Plateau years", 0.0, 20.0, step=0.5, key="plateau_years")
    annual_decline_pct = st.slider("Annual decline, %", 1.0, 25.0, step=0.5, key="annual_decline_pct")
    late_redevelopment = st.checkbox("Add late redevelopment / infill hump", key="late_redevelopment")
    redevelopment_year = st.slider("Redevelopment timing, years from start", 1.0, float(history_years), step=0.5, key="redevelopment_year")
    redevelopment_strength = st.slider("Redevelopment strength", 0.0, 1.5, step=0.05, key="redevelopment_strength")
    liquid_noise_pct = st.slider("Measured liquid noise, %", 0.0, 20.0, step=0.5, key="liquid_noise_pct")

with st.sidebar.expander("Water Cut Profile", expanded=True):
    initial_water_cut_pct = st.slider("Initial water cut, %", 0.0, 60.0, step=1.0, key="initial_water_cut_pct")
    maximum_water_cut_pct = st.slider("Maximum water cut, %", 30.0, 95.0, step=1.0, key="maximum_water_cut_pct")
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
    "app": "Generic Field Material-Balanced Artificial History Match & RelPerm Generator",
    "version": "4.0",
    "values": current_values,
}

json_bytes = json.dumps(json_payload, indent=2).encode("utf-8")

st.download_button(
    label="Save Current Settings as JSON",
    data=json_bytes,
    file_name="Generic_Field_HistoryMatch_Settings.json",
    mime="application/json",
)
