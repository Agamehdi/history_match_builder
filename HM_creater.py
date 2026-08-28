import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Real Field History Match & RelPerm Dashboard",
    layout="wide"
)

# =========================================================
# DEFAULTS & SESSION STATE INITIALIZATION
# =========================================================
DEFAULTS = {
    "field_name": "Rumaila / Generic Asset",
    "seed": 2026,
    "start_year": 2010,
    "history_years": 20,
    "time_frequency": "Monthly",

    # Plot view
    "unit_system": "Metric",       # Metric / Field
    "x_axis_mode": "Date",        # Date / Years
    "plot_start_year": 2010,
    "plot_years": 20,

    # Measured field behavior controls
    "initial_liquid_rate": 1000.0,
    "peak_liquid_rate": 1800.0,
    "plateau_years": 4.0,
    "annual_decline_pct": 7.0,
    "late_redevelopment": True,
    "redevelopment_year": 10.0,
    "redevelopment_strength": 0.4,
    "liquid_noise_pct": 4.0,

    "initial_water_cut_pct": 10.0,
    "maximum_water_cut_pct": 85.0,
    "water_cut_rise_year": 6.0,
    "water_cut_rise_speed": 2.5,
    "water_cut_noise_pct": 2.0,

    "initial_gor": 80.0,
    "maximum_gor": 250.0,
    "gor_pressure_sensitivity": 1.2,
    "gor_noise_pct": 4.0,

    "initial_pressure_bar": 300.0,
    "abandonment_pressure_bar": 120.0,
    "aquifer_support": 0.3,
    "pressure_noise_bar": 2.5,

    # Model matching controls
    "liq_quality": 0.85,
    "liq_global_bias": 0,
    "liq_early_bias": 2,
    "liq_late_bias": -2,
    "liq_lag": 0,
    "liq_smoothness": 9,
    "liq_noise": 0.01,
    "liq_offzones": 3,
    "liq_offstrength": 0.04,

    "wcut_quality": 0.85,
    "wcut_global_bias": 0,
    "wcut_early_bias": -2,
    "wcut_late_bias": 2,
    "wcut_lag": 0,
    "wcut_smoothness": 9,
    "wcut_noise": 0.005,
    "wcut_offzones": 2,
    "wcut_offstrength": 0.02,

    "gor_quality": 0.80,
    "gor_global_bias": 0,
    "gor_early_bias": 3,
    "gor_late_bias": -2,
    "gor_lag": 0,
    "gor_smoothness": 11,
    "gor_noise": 0.01,
    "gor_offzones": 3,
    "gor_offstrength": 0.04,

    "prs_quality": 0.88,
    "prs_global_bias": 0,
    "prs_early_bias": 2,
    "prs_late_bias": -2,
    "prs_lag": 0,
    "prs_smoothness": 13,
    "prs_noise": 0.004,
    "prs_offzones": 2,
    "prs_offstrength": 0.015,

    # Display controls (Rənglər və ölçülər)
    "oil_dot_size": 7,
    "oil_dot_stride": 1,
    "oil_meas_color": "#1f77b4",
    "oil_model_color": "#d62728",

    "gas_dot_size": 7,
    "gas_dot_stride": 1,
    "gas_meas_color": "#1f77b4",
    "gas_model_color": "#d62728",

    "water_dot_size": 7,
    "water_dot_stride": 1,
    "water_meas_color": "#1f77b4",
    "water_model_color": "#d62728",

    "wcut_dot_size": 7,
    "wcut_dot_stride": 1,
    "wcut_meas_color": "#1f77b4",
    "wcut_model_color": "#d62728",

    "gor_dot_size": 7,
    "gor_dot_stride": 1,
    "gor_meas_color": "#1f77b4",
    "gor_model_color": "#d62728",

    "prs_dot_size": 7,
    "prs_dot_stride": 1,
    "prs_meas_color": "#1f77b4",
    "prs_model_color": "#d62728",

    "cum_dot_size": 7,
    "cum_dot_stride": 2,
    "cum_meas_color": "#1f77b4",
    "cum_model_color": "#d62728",

    # RelPerm default values (Water-Oil & Gas-Oil)
    "swc_b": 0.20, "sorw_b": 0.25, "krw_end_b": 0.80, "krow_end_b": 1.00, "nw_b": 2.5, "no_b": 2.0,
    "swc_a": 0.22, "sorw_a": 0.23, "krw_end_a": 0.70, "krow_end_a": 0.95, "nw_a": 3.0, "no_a": 2.2,
    "sgc_b": 0.05, "sorg_b": 0.20, "krg_end_b": 0.90, "krog_end_b": 1.00, "ng_b": 2.5, "nog_b": 2.0,
    "sgc_a": 0.06, "sorg_a": 0.22, "krg_end_a": 0.80, "krog_end_a": 0.90, "ng_a": 3.0, "nog_a": 2.2,
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =========================================================
# HELPER FUNCTIONS & MATH ENGINES
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
    days = pd.Series(dates).diff().dt.days.fillna(30).to_numpy()
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

def build_model_curve(measured, quality, global_bias, early_bias, late_bias, lag, smoothness, noise, offzones, offstrength, seed, min_value=0, max_value=None):
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

    distorted = add_smooth_offzones(distorted, measured, count=offzones, strength=offstrength, seed=seed + 101)

    smooth_measured = rolling_np(measured, max(3, smoothness // 2))
    model = quality * smooth_measured + (1 - quality) * distorted
    model = rolling_np(model, max(3, smoothness // 3))

    if max_value is not None:
        return np.clip(model, min_value, max_value)
    return np.clip(model, min_value, None)


def convert_display_units(unit_system, oil_m, oil_s, gas_m, gas_s, water_m, water_s, wcut_m, wcut_s, gor_m, gor_s, prs_m, prs_s, cum_m, cum_s):
    if unit_system == "Metric":
        return {
            "oil_meas": oil_m, "oil_model": oil_s,
            "gas_meas": gas_m, "gas_model": gas_s,
            "water_meas": water_m, "water_model": water_s,
            "wcut_meas": wcut_m, "wcut_model": wcut_s,
            "gor_meas": gor_m, "gor_model": gor_s,
            "prs_meas": prs_m, "prs_model": prs_s,
            "cum_meas": cum_m, "cum_model": cum_s,
            "oil_lbl": "Oil Rate, m³/d", "gas_lbl": "Gas Rate, 10³ m³/d",
            "water_lbl": "Water Rate, m³/d", "wcut_lbl": "Water Cut, %",
            "gor_lbl": "GOR, m³/m³", "prs_lbl": "Reservoir Pressure, bar",
            "cum_lbl": "Cumulative Oil, MMm³"
        }
    
    # Field Units (STB, psi, MMSTB, MMscf/d)
    M3_TO_STB = 6.28981
    KSM3D_TO_MMSCFD = 0.0353147
    M3M3_TO_SCFSTB = 5.615
    BAR_TO_PSI = 14.5038

    return {
        "oil_meas": oil_m * M3_TO_STB, "oil_model": oil_s * M3_TO_STB,
        "gas_meas": gas_m * KSM3D_TO_MMSCFD, "gas_model": gas_s * KSM3D_TO_MMSCFD,
        "water_meas": water_m * M3_TO_STB, "water_model": water_s * M3_TO_STB,
        "wcut_meas": wcut_m, "wcut_model": wcut_s,
        "gor_meas": gor_m * M3M3_TO_SCFSTB, "gor_model": gor_s * M3M3_TO_SCFSTB,
        "prs_meas": prs_m * BAR_TO_PSI, "prs_model": prs_s * BAR_TO_PSI,
        "cum_meas": cum_m * M3_TO_STB, "cum_model": cum_s * M3_TO_STB,
        "oil_lbl": "Oil Rate, STB/d", "gas_lbl": "Gas Rate, MMscf/d",
        "water_lbl": "Water Rate, STB/d", "wcut_lbl": "Water Cut, %",
        "gor_lbl": "GOR, scf/STB", "prs_lbl": "Reservoir Pressure, psi",
        "cum_lbl": "Cumulative Oil, MMSTB"
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
        return mask, x_values, "Elapsed Years"

    return mask, filtered_dates, "Date"


def add_history_trace(fig, row, col, x, measured, model, y_title, meas_color, model_color, dot_size, dot_stride, x_title="Date", show_legend=False):
    fig.add_trace(
        go.Scatter(
            x=x[::dot_stride], y=measured[::dot_stride],
            mode="markers", name="Measured Data",
            marker=dict(color=meas_color, size=dot_size, opacity=0.85, line=dict(color="black", width=0.5)),
            showlegend=show_legend
        ),
        row=row, col=col
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=model,
            mode="lines", name="Simulation Model",
            line=dict(color=model_color, width=2.5),
            showlegend=show_legend
        ),
        row=row, col=col
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
        lag = st.slider("Lag / lead, steps", -12, 12, step=1, key=f"{prefix}_lag")
        smoothness = st.slider("Model smoothness", 3, 35, step=2, key=f"{prefix}_smoothness")
        noise = st.slider("Low-freq noise", 0.0, 0.08, step=0.002, key=f"{prefix}_noise")
        offzones = st.slider("Off-zone count", 0, 8, step=1, key=f"{prefix}_offzones")
        offstrength = st.slider("Off-zone strength", 0.0, 0.25, step=0.005, key=f"{prefix}_offstrength")
    return {"quality": quality, "global_bias": global_bias, "early_bias": early_bias, "late_bias": late_bias, "lag": lag, "smoothness": smoothness, "noise": noise, "offzones": offzones, "offstrength": offstrength}


def display_control_group(title, prefix):
    with st.sidebar.expander(title, expanded=False):
        dot_size = st.slider("Measured dot size", 2, 18, step=1, key=f"{prefix}_dot_size")
        dot_stride = st.slider("Show every Nth point", 1, 12, step=1, key=f"{prefix}_dot_stride")
        meas_color = st.color_picker("Measured color", key=f"{prefix}_meas_color")
        model_color = st.color_picker("Model color", key=f"{prefix}_model_color")
    return {"dot_size": dot_size, "dot_stride": dot_stride, "meas_color": meas_color, "model_color": model_color}


def make_excel(history_df, relperm_wo_df, relperm_go_df, controls_df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        history_df.to_excel(writer, sheet_name="History_Match", index=False)
        relperm_wo_df.to_excel(writer, sheet_name="RelPerm_WaterOil", index=False)
        relperm_go_df.to_excel(writer, sheet_name="RelPerm_GasOil", index=False)
        controls_df.to_excel(writer, sheet_name="Controls", index=False)

        workbook = writer.book
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})
        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            ws.freeze_panes(1, 0)
            ws.set_column(0, 25, 20)
    output.seek(0)
    return output


# =========================================================
# UI HEADER & JSON MANAGEMENT
# =========================================================
st.title("🛢️ Advanced Field History Match & Corey RelPerm Dashboard")
st.markdown("Material-balance coupled production generator with fully customizable Plotly charts, interactive history match sliders, and Corey relative permeability modeling.")

st.sidebar.header("📁 Settings Management")
uploaded_json = st.sidebar.file_uploader("Load settings JSON", type=["json"])
if uploaded_json is not None and st.sidebar.button("Apply Loaded JSON"):
    loaded = json.load(uploaded_json)
    values = loaded.get("values", loaded)
    for k, v in values.items():
        if k in DEFAULTS:
            st.session_state[k] = v
    st.rerun()

if st.sidebar.button("Reset to Defaults"):
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.rerun()


# =========================================================
# SIDEBAR CONTROLS
# =========================================================
st.sidebar.header("⚙️ Field Setup")
field_name = st.sidebar.text_input("Field name", key="field_name")
seed = st.sidebar.number_input("Random seed", min_value=1, max_value=999999, step=1, key="seed")
start_year = st.sidebar.slider("Start year", 1950, 2030, step=1, key="start_year")
history_years = st.sidebar.slider("History length, years", 3, 50, step=1, key="history_years")
time_frequency = st.sidebar.radio("Frequency", ["Monthly", "Quarterly"], key="time_frequency")

st.sidebar.header("📊 Plot View Controls")
unit_system = st.sidebar.radio("Unit system", ["Metric", "Field"], key="unit_system")
x_axis_mode = st.sidebar.radio("X-axis mode", ["Date", "Years"], key="x_axis_mode")
plot_start_year = st.sidebar.slider("Plot start year", int(start_year), int(start_year + history_years - 1), key="plot_start_year", step=1)

max_plot_years = max(1, int(start_year + history_years - plot_start_year))
if st.session_state["plot_years"] > max_plot_years:
    st.session_state["plot_years"] = max_plot_years
plot_years = st.sidebar.slider("Display years", 1, max_plot_years, key="plot_years", step=1)

st.sidebar.header("📈 Measured Profiles")
with st.sidebar.expander("Production Profile", expanded=False):
    initial_liquid_rate = st.slider("Initial liquid rate, m³/d", 50.0, 3000.0, step=10.0, key="initial_liquid_rate")
    peak_liquid_rate = st.slider("Peak liquid rate, m³/d", 100.0, 6000.0, step=10.0, key="peak_liquid_rate")
    plateau_years = st.slider("Plateau years", 0.0, 15.0, step=0.5, key="plateau_years")
    annual_decline_pct = st.slider("Annual decline, %", 1.0, 25.0, step=0.5, key="annual_decline_pct")
    late_redevelopment = st.checkbox("Add redevelopment hump", key="late_redevelopment")
    redevelopment_year = st.slider("Redevelopment timing", 1.0, float(history_years), step=0.5, key="redevelopment_year")
    redevelopment_strength = st.slider("Redevelopment strength", 0.0, 1.5, step=0.05, key="redevelopment_strength")
    liquid_noise_pct = st.slider("Liquid noise, %", 0.0, 20.0, step=0.5, key="liquid_noise_pct")

with st.sidebar.expander("Water Cut Profile", expanded=False):
    initial_water_cut_pct = st.slider("Initial water cut, %", 0.0, 50.0, step=1.0, key="initial_water_cut_pct")
    maximum_water_cut_pct = st.slider("Maximum water cut, %", 30.0, 98.0, step=1.0, key="maximum_water_cut_pct")
    water_cut_rise_year = st.slider("Water cut rise timing", 0.5, float(history_years), step=0.5, key="water_cut_rise_year")
    water_cut_rise_speed = st.slider("Water cut rise speed", 0.5, 8.0, step=0.1, key="water_cut_rise_speed")
    water_cut_noise_pct = st.slider("WCUT noise, %", 0.0, 10.0, step=0.2, key="water_cut_noise_pct")

with st.sidebar.expander("GOR & Pressure Profile", expanded=False):
    initial_gor = st.slider("Initial GOR, m³/m³", 10.0, 500.0, step=5.0, key="initial_gor")
    maximum_gor = st.slider("Maximum GOR, m³/m³", 50.0, 1500.0, step=10.0, key="maximum_gor")
    gor_pressure_sensitivity = st.slider("GOR pressure sensitivity", 0.0, 3.0, step=0.1, key="gor_pressure_sensitivity")
    gor_noise_pct = st.slider("GOR noise, %", 0.0, 20.0, step=0.5, key="gor_noise_pct")
    initial_pressure_bar = st.slider("Initial pressure, bar", 100.0, 500.0, step=5.0, key="initial_pressure_bar")
    abandonment_pressure_bar = st.slider("Abandonment pressure, bar", 30.0, 300.0, step=5.0, key="abandonment_pressure_bar")
    aquifer_support = st.slider("Aquifer support", 0.0, 0.9, step=0.05, key="aquifer_support")
    pressure_noise_bar = st.slider("Pressure noise, bar", 0.0, 15.0, step=0.5, key="pressure_noise_bar")

st.sidebar.header("🎛️ Model Match Tuning")
liq_cfg = model_control_group("Liquid / Oil-Water", "liq")
wcut_cfg = model_control_group("Water Cut System", "wcut")
gor_cfg = model_control_group("GOR System", "gor")
prs_cfg = model_control_group("Pressure System", "prs")

st.sidebar.header("🎨 Display & Styling")
oil_disp = display_control_group("Oil Rate Styling", "oil")
gas_disp = display_control_group("Gas Rate Styling", "gas")
water_disp = display_control_group("Water Rate Styling", "water")
wcut_disp = display_control_group("Water Cut Styling", "wcut")
gor_disp = display_control_group("GOR Styling", "gor")
prs_disp = display_control_group("Pressure Styling", "prs")
cum_disp = display_control_group("Cumulative Oil Styling", "cum")


# =========================================================
# GENERATE SYNTHETIC HISTORY & MATERIAL BALANCE
# =========================================================
freq = "MS" if time_frequency == "Monthly" else "QS"
periods = int(history_years * (12 if time_frequency == "Monthly" else 4))
dates = pd.date_range(f"{int(start_year)}-01-01", periods=periods, freq=freq)
t_years = np.arange(periods) / (12 if time_frequency == "Monthly" else 4)
rng = np.random.default_rng(seed)

# Liquid
ramp = initial_liquid_rate + (peak_liquid_rate - initial_liquid_rate) * (1 - np.exp(-t_years / 0.9))
decline_factor = np.exp(-annual_decline_pct / 100.0 * np.maximum(t_years - plateau_years, 0))
total_liq_clean = ramp * decline_factor
if late_redevelopment:
    hump = redevelopment_strength * peak_liquid_rate * np.exp(-0.5 * ((t_years - redevelopment_year) / (history_years / 9.0)) ** 2)
    total_liq_clean += hump

total_liq_measured = total_liq_clean * (1 + rng.normal(0, liquid_noise_pct / 100.0, periods))
total_liq_measured = np.clip(total_liq_measured, 1, None)

# Water Cut
wcut_clean = initial_water_cut_pct + (maximum_water_cut_pct - initial_water_cut_pct) / (1 + np.exp(-(t_years - water_cut_rise_year) / water_cut_rise_speed))
wcut_measured = np.clip(wcut_clean + rng.normal(0, water_cut_noise_pct, periods), 0, maximum_water_cut_pct)

oil_measured = total_liq_measured * (1 - wcut_measured / 100.0)
water_measured = total_liq_measured * (wcut_measured / 100.0)
cum_oil_measured = calc_cumulative(oil_measured, dates)

# Pressure & GOR (Material balance coupled)
cum_norm = cum_oil_measured / max(cum_oil_measured.max(), 1e-9)
pressure_drop = (initial_pressure_bar - abandonment_pressure_bar) * (1 - aquifer_support)
pressure_clean = initial_pressure_bar - pressure_drop * (cum_norm ** 0.75)
pressure_measured = np.clip(pressure_clean + rng.normal(0, pressure_noise_bar, periods), abandonment_pressure_bar, initial_pressure_bar)

pressure_depletion = np.clip((initial_pressure_bar - pressure_measured) / max(initial_pressure_bar - abandonment_pressure_bar, 1e-9), 0, 1)
gor_clean = initial_gor + (maximum_gor - initial_gor) * (pressure_depletion ** gor_pressure_sensitivity)
gor_measured = np.clip(gor_clean * (1 + rng.normal(0, gor_noise_pct / 100.0, periods)), 1, maximum_gor)
gas_measured = oil_measured * gor_measured / 1000.0


# --- MODEL CURVES ---
total_liq_model = build_model_curve(total_liq_measured, seed=seed+1, min_value=0, **liq_cfg)
wcut_model = build_model_curve(wcut_measured, seed=seed+2, min_value=0, max_value=maximum_water_cut_pct, **wcut_cfg)
gor_model = build_model_curve(gor_measured, seed=seed+3, min_value=1, max_value=maximum_gor, **gor_cfg)

oil_model = total_liq_model * (1 - wcut_model / 100.0)
water_model = total_liq_model * (wcut_model / 100.0)
gas_model = oil_model * gor_model / 1000.0
cum_oil_model = calc_cumulative(oil_model, dates)

cum_model_norm = cum_oil_model / max(cum_oil_model.max(), 1e-9)
pressure_consistent = initial_pressure_bar - pressure_drop * (cum_model_norm ** 0.75)
pressure_model = build_model_curve(pressure_consistent, seed=seed+4, min_value=abandonment_pressure_bar, max_value=initial_pressure_bar, **prs_cfg)


# =========================================================
# DISPLAY UNITS & PLOTTING WINDOW
# =========================================================
display = convert_display_units(
    unit_system, oil_measured, oil_model, gas_measured, gas_model,
    water_measured, water_model, wcut_measured, wcut_model,
    gor_measured, gor_model, pressure_measured, pressure_model,
    cum_oil_measured, cum_oil_model
)

plot_mask, x_plot, x_title = make_plot_window(dates, x_axis_mode, plot_start_year, plot_years)


# =========================================================
# PLOT HISTORY MATCH (PLOTLY SUBPLOTS)
# =========================================================
history_fig = make_subplots(
    rows=4, cols=2,
    specs=[[{}, {}], [{}, {}], [{}, {}], [{"colspan": 2}, None]],
    subplot_titles=(
        "Oil Rate History Match", "Gas Rate History Match",
        "Water Rate History Match", "Water Cut History Match",
        "GOR History Match", "Reservoir Pressure History Match",
        "Cumulative Oil History Match"
    ),
    vertical_spacing=0.11,
    horizontal_spacing=0.08
)

add_history_trace(history_fig, 1, 1, x_plot, display["oil_meas"][plot_mask], display["oil_model"][plot_mask], display["oil_lbl"], oil_disp["meas_color"], oil_disp["model_color"], oil_disp["dot_size"], oil_disp["dot_stride"], x_title, show_legend=True)
add_history_trace(history_fig, 1, 2, x_plot, display["gas_meas"][plot_mask], display["gas_model"][plot_mask], display["gas_lbl"], gas_disp["meas_color"], gas_disp["model_color"], gas_disp["dot_size"], gas_disp["dot_stride"], x_title)
add_history_trace(history_fig, 2, 1, x_plot, display["water_meas"][plot_mask], display["water_model"][plot_mask], display["water_lbl"], water_disp["meas_color"], water_disp["model_color"], water_disp["dot_size"], water_disp["dot_stride"], x_title)
add_history_trace(history_fig, 2, 2, x_plot, display["wcut_meas"][plot_mask], display["wcut_model"][plot_mask], display["wcut_lbl"], wcut_disp["meas_color"], wcut_disp["model_color"], wcut_disp["dot_size"], wcut_disp["dot_stride"], x_title)
add_history_trace(history_fig, 3, 1, x_plot, display["gor_meas"][plot_mask], display["gor_model"][plot_mask], display["gor_lbl"], gor_disp["meas_color"], gor_disp["model_color"], gor_disp["dot_size"], gor_disp["dot_stride"], x_title)
add_history_trace(history_fig, 3, 2, x_plot, display["prs_meas"][plot_mask], display["prs_model"][plot_mask], display["prs_lbl"], prs_disp["meas_color"], prs_disp["model_color"], prs_disp["dot_size"], prs_disp["dot_stride"], x_title)
add_history_trace(history_fig, 4, 1, x_plot, display["cum_meas"][plot_mask], display["cum_model"][plot_mask], display["cum_lbl"], cum_disp["meas_color"], cum_disp["model_color"], cum_disp["dot_size"], cum_disp["dot_stride"], x_title)

history_fig.update_layout(
    title_text=f"{field_name} – History Match Dashboard ({unit_system} Units)",
    title_x=0.5, height=1300, template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=-0.05, xanchor="center", x=0.5)
)
history_fig.update_xaxes(showgrid=True, gridcolor="#EAEAEA")
history_fig.update_yaxes(showgrid=True, gridcolor="#EAEAEA")

st.plotly_chart(history_fig, use_container_width=True)


# =========================================================
# RELATIVE PERMEABILITY PLAYGROUND (COREY)
# =========================================================
st.header("🔬 Corey Relative Permeability Playground")

with st.sidebar.expander("Water-Oil Corey Controls", expanded=False):
    st.markdown("**Before History Match**")
    swc_b = st.slider("Swc (Before)", 0.05, 0.40, step=0.01, key="swc_b")
    sorw_b = st.slider("Sorw (Before)", 0.05, 0.40, step=0.01, key="sorw_b")
    krw_end_b = st.slider("Krw endpoint (Before)", 0.1, 1.0, step=0.01, key="krw_end_b")
    krow_end_b = st.slider("Krow endpoint (Before)", 0.1, 1.0, step=0.01, key="krow_end_b")
    nw_b = st.slider("nw (Before)", 1.0, 5.0, step=0.05, key="nw_b")
    no_b = st.slider("no (Before)", 1.0, 5.0, step=0.05, key="no_b")

    st.markdown("**After History Match**")
    swc_a = st.slider("Swc (After)", 0.05, 0.40, step=0.01, key="swc_a")
    sorw_a = st.slider("Sorw (After)", 0.05, 0.40, step=0.01, key="sorw_a")
    krw_end_a = st.slider("Krw endpoint (After)", 0.1, 1.0, step=0.01, key="krw_end_a")
    krow_end_a = st.slider("Krow endpoint (After)", 0.1, 1.0, step=0.01, key="krow_end_a")
    nw_a = st.slider("nw (After)", 1.0, 5.0, step=0.05, key="nw_a")
    no_a = st.slider("no (After)", 1.0, 5.0, step=0.05, key="no_a")

with st.sidebar.expander("Gas-Oil Corey Controls", expanded=False):
    st.markdown("**Before History Match**")
    sgc_b = st.slider("Sgc (Before)", 0.0, 0.25, step=0.01, key="sgc_b")
    sorg_b = st.slider("Sorg (Before)", 0.05, 0.40, step=0.01, key="sorg_b")
    krg_end_b = st.slider("Krg endpoint (Before)", 0.1, 1.0, step=0.01, key="krg_end_b")
    krog_end_b = st.slider("Krog endpoint (Before)", 0.1, 1.0, step=0.01, key="krog_end_b")
    ng_b = st.slider("ng (Before)", 1.0, 5.0, step=0.05, key="ng_b")
    nog_b = st.slider("nog (Before)", 1.0, 5.0, step=0.05, key="nog_b")

    st.markdown("**After History Match**")
    sgc_a = st.slider("Sgc (After)", 0.0, 0.25, step=0.01, key="sgc_a")
    sorg_a = st.slider("Sorg (After)", 0.05, 0.40, step=0.01, key="sorg_a")
    krg_end_a = st.slider("Krg endpoint (After)", 0.1, 1.0, step=0.01, key="krg_end_a")
    krog_end_a = st.slider("Krog endpoint (After)", 0.1, 1.0, step=0.01, key="krog_end_a")
    ng_a = st.slider("ng (After)", 1.0, 5.0, step=0.05, key="ng_a")
    nog_a = st.slider("nog (After)", 1.0, 5.0, step=0.05, key="nog_a")

sw = np.linspace(0, 1, 101)
sg = np.linspace(0, 1, 101)

krw_b, krow_b = corey_water_oil(sw, swc_b, sorw_b, krw_end_b, krow_end_b, nw_b, no_b)
krw_a, krow_a = corey_water_oil(sw, swc_a, sorw_a, krw_end_a, krow_end_a, nw_a, no_a)
krg_b, krog_b = corey_gas_oil(sg, sgc_b, sorg_b, krg_end_b, krog_end_b, ng_b, nog_b)
krg_a, krog_a = corey_gas_oil(sg, sgc_a, sorg_a, krg_end_a, krog_end_a, ng_a, nog_a)

tab1, tab2 = st.tabs(["Water-Oil RelPerm", "Gas-Oil RelPerm"])

with tab1:
    fig_wo = go.Figure()
    fig_wo.add_trace(go.Scatter(x=sw, y=krow_b, mode="lines", name="Krow (Before)"))
    fig_wo.add_trace(go.Scatter(x=sw, y=krw_b, mode="lines", name="Krw (Before)"))
    fig_wo.add_trace(go.Scatter(x=sw, y=krow_a, mode="lines", name="Krow (After)", line=dict(dash="dash")))
    fig_wo.add_trace(go.Scatter(x=sw, y=krw_a, mode="lines", name="Krw (After)", line=dict(dash="dash")))
    fig_wo.update_layout(title="Water-Oil Relative Permeability (Corey)", xaxis_title="Water Saturation (Sw)", yaxis_title="RelPerm", template="plotly_white", height=500)
    st.plotly_chart(fig_wo, use_container_width=True)

with tab2:
    fig_go = go.Figure()
    fig_go.add_trace(go.Scatter(x=sg, y=krog_b, mode="lines", name="Krog (Before)"))
    fig_go.add_trace(go.Scatter(x=sg, y=krg_b, mode="lines", name="Krg (Before)"))
    fig_go.add_trace(go.Scatter(x=sg, y=krog_a, mode="lines", name="Krog (After)", line=dict(dash="dash")))
    fig_go.add_trace(go.Scatter(x=sg, y=krg_a, mode="lines", name="Krg (After)", line=dict(dash="dash")))
    fig_go.update_layout(title="Gas-Oil Relative Permeability (Corey)", xaxis_title="Gas Saturation (Sg)", yaxis_title="RelPerm", template="plotly_white", height=500)
    st.plotly_chart(fig_go, use_container_width=True)


# =========================================================
# EXCEL & JSON EXPORTS
# =========================================================
history_df = pd.DataFrame({
    "Date": dates, "Time_Years": t_years,
    "Oil_Measured_m3d": oil_measured, "Oil_Model_m3d": oil_model,
    "Water_Measured_m3d": water_measured, "Water_Model_m3d": water_model,
    "Gas_Measured_10e3m3d": gas_measured, "Gas_Model_10e3m3d": gas_model,
    "WaterCut_Measured_pct": wcut_measured, "WaterCut_Model_pct": wcut_model,
    "GOR_Measured": gor_measured, "GOR_Model": gor_model,
    "Pressure_Measured_bar": pressure_measured, "Pressure_Model_bar": pressure_model,
    "CumOil_Measured_MMm3": cum_oil_measured, "CumOil_Model_MMm3": cum_oil_model
})

relperm_wo_df = pd.DataFrame({"Sw": sw, "Krw_Before": krw_b, "Krow_Before": krow_b, "Krw_After": krw_a, "Krow_After": krow_a})
relperm_go_df = pd.DataFrame({"Sg": sg, "Krg_Before": krg_b, "Krog_Before": krog_b, "Krg_After": krg_a, "Krog_After": krog_a})
current_values = {k: st.session_state[k] for k in DEFAULTS.keys() if k in st.session_state}
controls_df = pd.DataFrame({"Parameter": list(current_values.keys()), "Value": list(current_values.values())})

excel_file = make_excel(history_df, relperm_wo_df, relperm_go_df, controls_df)
st.download_button(
    label="📥 Download Complete Excel Dataset",
    data=excel_file,
    file_name="Field_HistoryMatch_Corey_RelPerm.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

json_bytes = json.dumps({"app": "Advanced Field HM & RelPerm", "values": current_values}, indent=2).encode("utf-8")
st.download_button(
    label="💾 Save Current Settings as JSON",
    data=json_bytes,
    file_name="Field_HM_Settings.json",
    mime="application/json"
)
