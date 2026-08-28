import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Zira Artificial History Match & RelPerm Generator",
    layout="wide"
)

# =========================================================
# Defaults
# =========================================================
DEFAULTS = {
    # Global
    "seed": 2024,

    # Model distortion controls: Liquid controls oil/water consistency
    "liq_quality": 0.82,
    "liq_global_bias": 0,
    "liq_early_bias": 4,
    "liq_late_bias": -3,
    "liq_lag": 0,
    "liq_smoothness": 13,
    "liq_noise": 0.012,
    "liq_offzones": 4,
    "liq_offstrength": 0.06,

    # WCUT controls
    "wcut_quality": 0.86,
    "wcut_global_bias": 0,
    "wcut_early_bias": -3,
    "wcut_late_bias": 2,
    "wcut_lag": 0,
    "wcut_smoothness": 11,
    "wcut_noise": 0.008,
    "wcut_offzones": 3,
    "wcut_offstrength": 0.035,

    # GOR controls
    "gor_quality": 0.78,
    "gor_global_bias": 0,
    "gor_early_bias": 5,
    "gor_late_bias": -4,
    "gor_lag": 0,
    "gor_smoothness": 13,
    "gor_noise": 0.015,
    "gor_offzones": 4,
    "gor_offstrength": 0.06,

    # Pressure controls
    "prs_quality": 0.84,
    "prs_global_bias": 0,
    "prs_early_bias": 2,
    "prs_late_bias": -2,
    "prs_lag": 0,
    "prs_smoothness": 17,
    "prs_noise": 0.006,
    "prs_offzones": 3,
    "prs_offstrength": 0.025,

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
    "prs_dot_stride": 3,
    "prs_meas_color": "#d97a6c",
    "prs_model_color": "#6f35a5",

    "cum_dot_size": 7,
    "cum_dot_stride": 3,
    "cum_meas_color": "#d97a6c",
    "cum_model_color": "#2fa84f",

    # Water-Oil Corey defaults
    "swc_b": 0.25,
    "sorw_b": 0.22,
    "krw_end_b": 0.95,
    "krow_end_b": 1.00,
    "nw_b": 2.60,
    "no_b": 2.10,

    "swc_a": 0.30,
    "sorw_a": 0.22,
    "krw_end_a": 0.60,
    "krow_end_a": 0.96,
    "nw_a": 3.80,
    "no_a": 3.20,

    # Gas-Oil Corey defaults
    "sgc_b": 0.05,
    "sorg_b": 0.25,
    "krg_end_b": 1.00,
    "krog_end_b": 1.00,
    "ng_b": 3.00,
    "nog_b": 2.00,

    "sgc_a": 0.08,
    "sorg_a": 0.28,
    "krg_end_a": 0.72,
    "krog_end_a": 0.96,
    "ng_a": 4.00,
    "nog_a": 2.80,
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =========================================================
# Helper functions
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


def add_smooth_offzones(model, measured, count, strength, seed):
    rng = np.random.default_rng(seed)
    n = len(model)
    idx = np.arange(n)
    scale = max(np.nanmax(measured) - np.nanmin(measured), 1.0)

    out = model.copy()

    for _ in range(int(count)):
        center = rng.integers(8, max(10, n - 8))
        width = rng.integers(6, 22)
        sign = rng.choice([-1, 1])
        amp = scale * strength * rng.uniform(0.55, 1.20)
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
        seed=seed + 101
    )

    # Blend with measured: high quality -> closer to measured, but still smooth
    smooth_measured = rolling_np(measured, max(3, smoothness // 2))
    model = quality * smooth_measured + (1 - quality) * distorted

    model = rolling_np(model, max(3, smoothness // 3))

    if max_value is not None:
        model = np.clip(model, min_value, max_value)
    else:
        model = np.clip(model, min_value, None)

    return model


def calc_cumulative(rate_m3d, dates):
    days = pd.Series(dates).diff().dt.days.fillna(0).to_numpy()
    cum_m3 = np.cumsum(rate_m3d * days)
    return cum_m3 / 1e6  # MMm3


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
    show_legend=False,
):
    x_meas = x[::dot_stride]
    y_meas = measured[::dot_stride]

    fig.add_trace(
        go.Scatter(
            x=x_meas,
            y=y_meas,
            mode="markers",
            name="Measured Data",
            marker=dict(
                color=measured_color,
                size=dot_size,
                opacity=0.80,
                line=dict(color="rgba(70,70,70,0.55)", width=0.8),
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
    fig.update_xaxes(title_text="DATE", row=row, col=col)


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


def slider(label, key, min_value, max_value, step):
    return st.slider(
        label,
        min_value,
        max_value,
        value=st.session_state[key],
        step=step,
        key=key,
    )


def number_input(label, key, min_value, max_value, step):
    return st.number_input(
        label,
        min_value=min_value,
        max_value=max_value,
        value=st.session_state[key],
        step=step,
        key=key,
    )


def color_picker(label, key):
    return st.color_picker(label, value=st.session_state[key], key=key)


def model_control_group(title, prefix):
    with st.sidebar.expander(title, expanded=False):
        quality = slider("Match quality", f"{prefix}_quality", 0.0, 1.0, 0.01)
        global_bias = slider("Global bias, %", f"{prefix}_global_bias", -30, 30, 1)
        early_bias = slider("Early-period bias, %", f"{prefix}_early_bias", -30, 30, 1)
        late_bias = slider("Late-period bias, %", f"{prefix}_late_bias", -30, 30, 1)
        lag = slider("Lag / lead, quarters", f"{prefix}_lag", -8, 8, 1)
        smoothness = slider("Model smoothness", f"{prefix}_smoothness", 3, 35, 2)
        noise = slider("Low-frequency noise", f"{prefix}_noise", 0.0, 0.08, 0.002)
        offzones = slider("Off-zone count", f"{prefix}_offzones", 0, 8, 1)
        offstrength = slider("Off-zone strength", f"{prefix}_offstrength", 0.0, 0.25, 0.005)

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
        dot_size = slider("Measured dot size", f"{prefix}_dot_size", 2, 18, 1)
        dot_stride = slider("Measured dot density: show every Nth point", f"{prefix}_dot_stride", 1, 12, 1)
        meas_color = color_picker("Measured dot color", f"{prefix}_meas_color")
        model_color = color_picker("Model line color", f"{prefix}_model_color")

    return {
        "dot_size": dot_size,
        "dot_stride": dot_stride,
        "meas_color": meas_color,
        "model_color": model_color,
    }


def make_excel(history_df, relperm_wo_df, relperm_go_df, controls_df):
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        history_df.to_excel(writer, sheet_name="History_Match", index=False)
        relperm_wo_df.to_excel(writer, sheet_name="RelPerm_WaterOil", index=False)
        relperm_go_df.to_excel(writer, sheet_name="RelPerm_GasOil", index=False)
        controls_df.to_excel(writer, sheet_name="Controls", index=False)

        workbook = writer.book
        header_fmt = workbook.add_format({
            "bold": True,
            "bg_color": "#D9EAF7",
            "border": 1,
        })

        for sheet_name, df in {
            "History_Match": history_df,
            "RelPerm_WaterOil": relperm_wo_df,
            "RelPerm_GasOil": relperm_go_df,
            "Controls": controls_df,
        }.items():
            ws = writer.sheets[sheet_name]
            ws.freeze_panes(1, 0)
            ws.set_column(0, len(df.columns) - 1, 22)

            for c, col in enumerate(df.columns):
                ws.write(0, c, col, header_fmt)

    output.seek(0)
    return output


# =========================================================
# Title
# =========================================================
st.title("Zira Field – Material-Balanced Artificial History Match & RelPerm Generator")
st.caption(
    "Synthetic curves for draft paper formatting only. "
    "Oil, gas, GOR, water rate, water cut, cumulative oil and pressure are kept physically consistent."
)

# =========================================================
# Sidebar JSON load
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
st.sidebar.header("Global Controls")
seed = number_input("Random seed", "seed", 1, 99999, 1)

st.sidebar.header("Model Consistency Controls")
st.sidebar.caption(
    "Liquid controls oil/water material balance. "
    "Gas is derived from Oil × GOR. "
    "Pressure is linked to cumulative offtake."
)

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
# Synthetic measured data with material balance consistency
# =========================================================
dates = pd.date_range("1956-01-01", "2024-12-01", freq="QS")
year = dates.year + (dates.dayofyear - 1) / 365.25
rng = np.random.default_rng(seed)

# Total liquid rate: early peak + decline + later secondary water-handling hump
early_liq_peak = 1650 * np.exp(-((year - 1962.0) / 1.55) ** 2)
liq_tail = 370 * np.exp(-0.052 * np.maximum(year - 1965, 0))
late_liq_hump = 420 * np.exp(-((year - 2010) / 7.8) ** 2)
small_ops = 40 * np.sin((year - 1956) / 1.8) * np.exp(-0.025 * np.maximum(year - 1965, 0))

total_liq_measured = early_liq_peak + liq_tail + late_liq_hump + small_ops
total_liq_measured += rng.normal(0, 28, len(dates))
total_liq_measured = np.clip(total_liq_measured, 3, None)

# Water cut: increasing mature field behavior
wcut_measured = 18 + 80 / (1 + np.exp(-(year - 1967.5) / 3.4))
wcut_measured += 8 * np.sin((year - 1957) / 2.5) * np.exp(-0.045 * np.maximum(year - 1967, 0))
wcut_measured += rng.normal(0, 3.2, len(dates))
wcut_measured = np.clip(wcut_measured, 5, 99)

# Oil and water from material balance
oil_measured = total_liq_measured * (1 - wcut_measured / 100)
water_measured = total_liq_measured * (wcut_measured / 100)

# GOR: high early solution-gas period, then declining/stabilized, no random unrealistic spikes
gor_measured = 650 + 3600 * np.exp(-((year - 1961.3) / 2.15) ** 2)
gor_measured += 260 * np.exp(-0.045 * np.maximum(year - 1966, 0))
gor_measured += 90 * np.sin((year - 1956) / 2.2) * np.exp(-0.02 * np.maximum(year - 1970, 0))
gor_measured += rng.normal(0, 55, len(dates))
gor_measured = np.clip(gor_measured, 80, None)

# Gas rate from Oil × GOR
# Unit: 10^3 m3/d. Example: oil m3/d * GOR m3/m3 / 1000
gas_measured = oil_measured * gor_measured / 1000

# Cumulative oil from oil rate
cum_oil_measured = calc_cumulative(oil_measured, dates)

# Pressure linked to cumulative oil/offtake
cum_norm = cum_oil_measured / max(cum_oil_measured.max(), 1e-6)
pressure_base = 145 - 43 * (cum_norm ** 0.72)
pressure_measured = pressure_base + 2.2 * np.sin((year - 1956) / 4.2)
pressure_measured += rng.normal(0, 1.8, len(dates))
pressure_measured = np.clip(pressure_measured, 88, 150)

# =========================================================
# Model curves – primary variables first
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
    min_value=1,
    max_value=99,
    **wcut_cfg
)

gor_model = build_model_curve(
    gor_measured,
    seed=seed + 3,
    min_value=20,
    max_value=None,
    **gor_cfg
)

# Derived model values preserving consistency
oil_model = total_liq_model * (1 - wcut_model / 100)
water_model = total_liq_model * (wcut_model / 100)
gas_model = oil_model * gor_model / 1000
cum_oil_model = calc_cumulative(oil_model, dates)

# Pressure model based on model cumulative oil/offtake, then slightly distorted
cum_model_norm = cum_oil_model / max(cum_oil_model.max(), 1e-6)
pressure_consistent_model = 145 - 43 * (cum_model_norm ** 0.72)
pressure_consistent_model += 1.2 * np.sin((year - 1956) / 5.0)

pressure_model = build_model_curve(
    pressure_consistent_model,
    seed=seed + 4,
    min_value=80,
    max_value=155,
    **prs_cfg
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
        "OIL_RATE, m³/d – DATE",
        "GAS_RATE, 10³ m³/d – DATE",
        "WATER_RATE, m³/d – DATE",
        "WCUT, % – DATE",
        "GOR, m³/m³ – DATE",
        "RESERVOIR PRESSURE, bar – DATE",
        "CUMULATIVE OIL, MMm³ – DATE",
    ),
    vertical_spacing=0.105,
    horizontal_spacing=0.08,
)

add_history_trace(
    history_fig, 1, 1,
    dates, oil_measured, oil_model,
    "OIL_RATE, m³/d",
    oil_disp["meas_color"], oil_disp["model_color"],
    oil_disp["dot_size"], oil_disp["dot_stride"],
    show_legend=True
)

add_history_trace(
    history_fig, 1, 2,
    dates, gas_measured, gas_model,
    "GAS_RATE, 10³ m³/d",
    gas_disp["meas_color"], gas_disp["model_color"],
    gas_disp["dot_size"], gas_disp["dot_stride"],
    show_legend=False
)

add_history_trace(
    history_fig, 2, 1,
    dates, water_measured, water_model,
    "WATER_RATE, m³/d",
    water_disp["meas_color"], water_disp["model_color"],
    water_disp["dot_size"], water_disp["dot_stride"],
    show_legend=False
)

add_history_trace(
    history_fig, 2, 2,
    dates, wcut_measured, wcut_model,
    "WCUT, %",
    wcut_disp["meas_color"], wcut_disp["model_color"],
    wcut_disp["dot_size"], wcut_disp["dot_stride"],
    show_legend=False
)

add_history_trace(
    history_fig, 3, 1,
    dates, gor_measured, gor_model,
    "GOR, m³/m³",
    gor_disp["meas_color"], gor_disp["model_color"],
    gor_disp["dot_size"], gor_disp["dot_stride"],
    show_legend=False
)

add_history_trace(
    history_fig, 3, 2,
    dates, pressure_measured, pressure_model,
    "P, bar",
    prs_disp["meas_color"], prs_disp["model_color"],
    prs_disp["dot_size"], prs_disp["dot_stride"],
    show_legend=False
)

add_history_trace(
    history_fig, 4, 1,
    dates, cum_oil_measured, cum_oil_model,
    "CUMULATIVE OIL, MMm³",
    cum_disp["meas_color"], cum_disp["model_color"],
    cum_disp["dot_size"], cum_disp["dot_stride"],
    show_legend=False
)

history_fig.update_layout(
    title_text="Artificial Zira History Match – Material-Balanced Synthetic Dataset",
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

st.plotly_chart(history_fig, use_container_width=True)

# =========================================================
# Material balance consistency checks
# =========================================================
st.subheader("Material Balance / Consistency Checks")

check_col1, check_col2, check_col3, check_col4 = st.columns(4)

oil_from_balance = total_liq_measured * (1 - wcut_measured / 100)
water_from_balance = total_liq_measured * wcut_measured / 100
gas_from_balance = oil_measured * gor_measured / 1000

check_col1.metric(
    "Max oil balance error",
    f"{np.max(np.abs(oil_measured - oil_from_balance)):.4f} m³/d"
)
check_col2.metric(
    "Max water balance error",
    f"{np.max(np.abs(water_measured - water_from_balance)):.4f} m³/d"
)
check_col3.metric(
    "Max gas/GOR error",
    f"{np.max(np.abs(gas_measured - gas_from_balance)):.4f} 10³ m³/d"
)
check_col4.metric(
    "Final Cum Oil",
    f"{cum_oil_measured[-1]:.3f} MMm³"
)

# =========================================================
# Relative permeability
# =========================================================
st.header("Relative Permeability – Corey Function Playground")

with st.sidebar.expander("Water-Oil Corey Controls", expanded=False):
    st.markdown("**Before History Match**")
    swc_b = slider("WO Before Swc", "swc_b", 0.05, 0.45, 0.01)
    sorw_b = slider("WO Before Sorw", "sorw_b", 0.05, 0.45, 0.01)
    krw_end_b = slider("WO Before Krw end", "krw_end_b", 0.10, 1.00, 0.01)
    krow_end_b = slider("WO Before Kro end", "krow_end_b", 0.10, 1.00, 0.01)
    nw_b = slider("WO Before nw", "nw_b", 1.0, 6.0, 0.05)
    no_b = slider("WO Before no", "no_b", 1.0, 6.0, 0.05)

    st.markdown("**After History Match**")
    swc_a = slider("WO After Swc", "swc_a", 0.05, 0.45, 0.01)
    sorw_a = slider("WO After Sorw", "sorw_a", 0.05, 0.45, 0.01)
    krw_end_a = slider("WO After Krw end", "krw_end_a", 0.10, 1.00, 0.01)
    krow_end_a = slider("WO After Kro end", "krow_end_a", 0.10, 1.00, 0.01)
    nw_a = slider("WO After nw", "nw_a", 1.0, 6.0, 0.05)
    no_a = slider("WO After no", "no_a", 1.0, 6.0, 0.05)

with st.sidebar.expander("Gas-Oil Corey Controls", expanded=False):
    st.markdown("**Before History Match**")
    sgc_b = slider("GO Before Sgc", "sgc_b", 0.00, 0.30, 0.01)
    sorg_b = slider("GO Before Sorg", "sorg_b", 0.05, 0.45, 0.01)
    krg_end_b = slider("GO Before Krg end", "krg_end_b", 0.10, 1.00, 0.01)
    krog_end_b = slider("GO Before Krog end", "krog_end_b", 0.10, 1.00, 0.01)
    ng_b = slider("GO Before ng", "ng_b", 1.0, 6.0, 0.05)
    nog_b = slider("GO Before nog", "nog_b", 1.0, 6.0, 0.05)

    st.markdown("**After History Match**")
    sgc_a = slider("GO After Sgc", "sgc_a", 0.00, 0.30, 0.01)
    sorg_a = slider("GO After Sorg", "sorg_a", 0.05, 0.45, 0.01)
    krg_end_a = slider("GO After Krg end", "krg_end_a", 0.10, 1.00, 0.01)
    krog_end_a = slider("GO After Krog end", "krog_end_a", 0.10, 1.00, 0.01)
    ng_a = slider("GO After ng", "ng_a", 1.0, 6.0, 0.05)
    nog_a = slider("GO After nog", "nog_a", 1.0, 6.0, 0.05)

sw = np.linspace(0, 1, 101)
sg = np.linspace(0, 1, 101)

krw_b, krow_b = corey_water_oil(sw, swc_b, sorw_b, krw_end_b, krow_end_b, nw_b, no_b)
krw_a, krow_a = corey_water_oil(sw, swc_a, sorw_a, krw_end_a, krow_end_a, nw_a, no_a)

krg_b, krog_b = corey_gas_oil(sg, sgc_b, sorg_b, krg_end_b, krog_end_b, ng_b, nog_b)
krg_a, krog_a = corey_gas_oil(sg, sgc_a, sorg_a, krg_end_a, krog_end_a, ng_a, nog_a)

tab1, tab2, tab3 = st.tabs([
    "Water-Oil RelPerm",
    "Gas-Oil RelPerm",
    "Overlay Before vs After"
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
# DataFrames
# =========================================================
history_df = pd.DataFrame({
    "Date": dates,
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

current_values = {k: st.session_state[k] for k in DEFAULTS.keys()}
controls_df = pd.DataFrame({
    "Parameter": list(current_values.keys()),
    "Value": list(current_values.values()),
})

with st.expander("Show generated data tables"):
    st.subheader("History Match Data")
    st.dataframe(history_df, use_container_width=True)

    st.subheader("Water-Oil RelPerm")
    st.dataframe(relperm_wo_df, use_container_width=True)

    st.subheader("Gas-Oil RelPerm")
    st.dataframe(relperm_go_df, use_container_width=True)

    st.subheader("Controls")
    st.dataframe(controls_df, use_container_width=True)

# =========================================================
# Downloads: Excel and JSON
# =========================================================
excel_file = make_excel(history_df, relperm_wo_df, relperm_go_df, controls_df)

st.download_button(
    label="Download Excel – Artificial Zira Material-Balanced Dataset",
    data=excel_file,
    file_name="Zira_MaterialBalanced_HistoryMatch_RelPerm.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

json_payload = {
    "app": "Zira Material-Balanced Artificial History Match & RelPerm Generator",
    "version": "2.0",
    "values": current_values,
}

json_bytes = json.dumps(json_payload, indent=2).encode("utf-8")

st.download_button(
    label="Save Current Settings as JSON",
    data=json_bytes,
    file_name="Zira_HistoryMatch_Settings.json",
    mime="application/json",
)
