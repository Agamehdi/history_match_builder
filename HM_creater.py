import io
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
# Helper functions
# =========================================================
def rolling_np(values, window):
    return pd.Series(values).rolling(window, min_periods=1, center=True).mean().to_numpy()


def ema_np(values, span):
    return pd.Series(values).ewm(span=span, adjust=False).mean().to_numpy()


def smooth_noise(n, rng, scale=1.0, smooth_window=9):
    raw = rng.normal(0, scale, n)
    return rolling_np(raw, smooth_window)


def make_realistic_model_from_measured(
    measured,
    quality=0.82,           # 1.0 => better match, 0.0 => poorer match
    bias_pct=0.0,           # global bias
    smoothness=11,          # higher => smoother model
    noise_factor=0.025,     # low-amplitude smooth noise
    mismatch_count=4,       # number of off-zones
    mismatch_strength=0.08, # amplitude of off-zones
    seed=2024,
):
    rng = np.random.default_rng(seed)
    measured = np.asarray(measured, dtype=float)
    n = len(measured)

    ymin = np.nanmin(measured)
    ymax = np.nanmax(measured)
    scale = max(ymax - ymin, 1.0)

    short_trend = rolling_np(measured, max(3, smoothness // 2))
    long_trend  = rolling_np(measured, smoothness)
    ema_trend   = ema_np(measured, max(4, smoothness))

    # Smooth backbone
    base_model = 0.45 * short_trend + 0.35 * long_trend + 0.20 * ema_trend

    # Smooth low-frequency noise
    lf_noise = smooth_noise(
        n,
        rng,
        scale=noise_factor * scale * (1.20 - 0.70 * quality),
        smooth_window=max(5, smoothness)
    )

    model = base_model * (1 + bias_pct / 100.0) + lf_noise

    # Add a few realistic mismatch windows
    mismatch_count = max(1, int(mismatch_count))
    idx = np.arange(n)
    for _ in range(mismatch_count):
        center = rng.integers(low=8, high=max(9, n - 8))
        width = rng.integers(low=5, high=16)
        sign = rng.choice([-1, 1])

        local_amp = scale * mismatch_strength * (1.10 - 0.55 * quality) * rng.uniform(0.6, 1.2)
        bump = np.exp(-0.5 * ((idx - center) / width) ** 2)
        model += sign * local_amp * bump

    # Pull somewhat toward measured trend without making it perfect
    pull = 0.55 + 0.30 * quality
    model = pull * model + (1 - pull) * long_trend

    # Final smoothing to suppress unrealistic spikes
    model = rolling_np(model, max(3, smoothness // 2))

    return np.clip(model, 0, None)


def add_history_trace(
    fig,
    row,
    col,
    x,
    measured,
    model,
    measured_name,
    model_name,
    y_title,
    measured_color,
    model_color,
    dot_size,
    dot_stride,
    show_legend=False,
):
    x_meas = x[::dot_stride]
    y_meas = measured[::dot_stride]

    # Measured = dots only
    fig.add_trace(
        go.Scatter(
            x=x_meas,
            y=y_meas,
            mode="markers",
            name=measured_name,
            marker=dict(
                color=measured_color,
                size=dot_size,
                opacity=0.80,
                line=dict(color="rgba(70,70,70,0.65)", width=0.8)
            ),
            showlegend=show_legend
        ),
        row=row,
        col=col
    )

    # Model = smooth line
    fig.add_trace(
        go.Scatter(
            x=x,
            y=model,
            mode="lines",
            name=model_name,
            line=dict(color=model_color, width=2.8),
            showlegend=show_legend
        ),
        row=row,
        col=col
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
            "border": 1
        })

        sheets = {
            "History_Match": history_df,
            "RelPerm_WaterOil": relperm_wo_df,
            "RelPerm_GasOil": relperm_go_df,
            "Controls": controls_df
        }

        for sheet_name, df in sheets.items():
            ws = writer.sheets[sheet_name]
            ws.freeze_panes(1, 0)
            ws.set_column(0, len(df.columns) - 1, 20)
            for c, col in enumerate(df.columns):
                ws.write(0, c, col, header_fmt)

    output.seek(0)
    return output


def variable_controls(label, key, measured_default, model_default, dot_default=7,
                      quality_default=0.84, smooth_default=11, noise_default=0.02,
                      mismatch_count_default=4, mismatch_strength_default=0.08,
                      dot_stride_default=2):
    with st.sidebar.expander(label, expanded=False):
        quality = st.slider(
            f"{label} match quality",
            0.0, 1.0, quality_default, 0.01,
            key=f"{key}_quality"
        )

        bias = st.slider(
            f"{label} model bias, %",
            -30, 30, 0, 1,
            key=f"{key}_bias"
        )

        smoothness = st.slider(
            f"{label} model smoothness",
            3, 31, smooth_default, 2,
            key=f"{key}_smoothness"
        )

        noise = st.slider(
            f"{label} low-frequency noise",
            0.000, 0.080, noise_default, 0.002,
            key=f"{key}_noise"
        )

        mismatch_count = st.slider(
            f"{label} off-zones count",
            1, 8, mismatch_count_default, 1,
            key=f"{key}_mismatch_count"
        )

        mismatch_strength = st.slider(
            f"{label} off-zones strength",
            0.01, 0.25, mismatch_strength_default, 0.01,
            key=f"{key}_mismatch_strength"
        )

        dot_size = st.slider(
            f"{label} measured dot size",
            2, 18, dot_default, 1,
            key=f"{key}_dot_size"
        )

        dot_stride = st.slider(
            f"{label} measured dot density (show every Nth point)",
            1, 12, dot_stride_default, 1,
            key=f"{key}_dot_stride"
        )

        measured_color = st.color_picker(
            f"{label} measured dots color",
            measured_default,
            key=f"{key}_measured_color"
        )

        model_color = st.color_picker(
            f"{label} model line color",
            model_default,
            key=f"{key}_model_color"
        )

    return (
        quality, bias, smoothness, noise,
        mismatch_count, mismatch_strength,
        dot_size, dot_stride,
        measured_color, model_color
    )


def unpack(ctrl):
    return {
        "quality": ctrl[0],
        "bias": ctrl[1],
        "smoothness": ctrl[2],
        "noise": ctrl[3],
        "mismatch_count": ctrl[4],
        "mismatch_strength": ctrl[5],
        "dot_size": ctrl[6],
        "dot_stride": ctrl[7],
        "measured_color": ctrl[8],
        "model_color": ctrl[9],
    }


# =========================================================
# App title
# =========================================================
st.title("Zira Field – Artificial History Match & RelPerm Generator")
st.caption(
    "Synthetic/artificial curves for draft paper formatting only. "
    "Replace with final Nexus/OFM/Excel export after real Zira model runs."
)

# =========================================================
# Sidebar global controls
# =========================================================
st.sidebar.header("Global Controls")

seed = st.sidebar.number_input(
    "Random seed",
    min_value=1,
    max_value=99999,
    value=2024,
    step=1
)

# =========================================================
# Create synthetic measured data
# =========================================================
dates = pd.date_range("1956-01-01", "2024-12-01", freq="QS")
year = dates.year + (dates.dayofyear - 1) / 365.25
rng = np.random.default_rng(seed)

# Oil Rate
oil_peak = 1250 * np.exp(-((year - 1962.2) / 1.55) ** 2)
oil_tail = 310 * np.exp(-0.075 * np.maximum(year - 1966, 0))
oil_measured = oil_peak + oil_tail
oil_measured += rng.normal(0, 25, len(dates))
oil_measured = np.clip(oil_measured, 0, None)

# Gas Rate
gas_peak = 4200 * np.exp(-((year - 1961.8) / 1.45) ** 2)
gas_tail = 520 * np.exp(-0.065 * np.maximum(year - 1965, 0))
gas_measured = gas_peak + gas_tail
gas_measured += rng.normal(0, 110, len(dates))
gas_measured = np.clip(gas_measured, 0, None)

# Water Rate
water_base = 90 * np.exp(-0.025 * np.maximum(year - 1962, 0))
water_hump = 70 * np.exp(-((year - 2010) / 8.5) ** 2)
water_measured = water_base + water_hump
water_measured += rng.normal(0, 9, len(dates))
water_measured = np.clip(water_measured, 0, None)

# Water Cut
wcut_measured = 20 + 78 / (1 + np.exp(-(year - 1967) / 3.2))
wcut_measured += 10 * np.sin((year - 1956) / 2.1) * np.exp(-0.035 * np.maximum(year - 1965, 0))
wcut_measured += rng.normal(0, 4.5, len(dates))
wcut_measured = np.clip(wcut_measured, 0, 100)

# GOR
gor_peak = 3900 * np.exp(-((year - 1961.5) / 2.25) ** 2)
gor_tail = 850 * np.exp(-0.055 * np.maximum(year - 1966, 0))
late_spikes = np.where(
    year > 1995,
    rng.choice([0, 0, 0, 250, 700, 1200, 1600], size=len(dates)),
    0
)
gor_measured = gor_peak + gor_tail + late_spikes
gor_measured += rng.normal(0, 170, len(dates))
gor_measured = np.clip(gor_measured, 0, None)

# Reservoir Pressure
pressure_trend = 142 - 45 * (1 - np.exp(-0.025 * np.maximum(year - 1956, 0)))
pressure_measured = pressure_trend + rng.normal(0, 2.8, len(dates))
pressure_measured = np.clip(pressure_measured, 85, 150)

# =========================================================
# Sidebar controls for history match
# =========================================================
st.sidebar.header("History Match Controls")
st.sidebar.caption("1.0 = near-ideal match, 0.0 = poor match")

oil_ctrl = variable_controls(
    "Oil Rate", "oil",
    measured_default="#d97a6c",
    model_default="#2fa84f",
    dot_default=7,
    quality_default=0.82,
    smooth_default=11,
    noise_default=0.015,
    mismatch_count_default=3,
    mismatch_strength_default=0.06,
    dot_stride_default=2
)

gas_ctrl = variable_controls(
    "Gas Rate", "gas",
    measured_default="#d97a6c",
    model_default="#d96adf",
    dot_default=7,
    quality_default=0.80,
    smooth_default=13,
    noise_default=0.020,
    mismatch_count_default=4,
    mismatch_strength_default=0.08,
    dot_stride_default=2
)

water_ctrl = variable_controls(
    "Water Rate", "water",
    measured_default="#d97a6c",
    model_default="#103d8f",
    dot_default=7,
    quality_default=0.76,
    smooth_default=9,
    noise_default=0.018,
    mismatch_count_default=5,
    mismatch_strength_default=0.10,
    dot_stride_default=2
)

wcut_ctrl = variable_controls(
    "Water Cut", "wcut",
    measured_default="#d97a6c",
    model_default="#e96be0",
    dot_default=7,
    quality_default=0.88,
    smooth_default=11,
    noise_default=0.010,
    mismatch_count_default=3,
    mismatch_strength_default=0.04,
    dot_stride_default=2
)

gor_ctrl = variable_controls(
    "GOR", "gor",
    measured_default="#d97a6c",
    model_default="#e96be0",
    dot_default=7,
    quality_default=0.72,
    smooth_default=11,
    noise_default=0.025,
    mismatch_count_default=5,
    mismatch_strength_default=0.10,
    dot_stride_default=2
)

pressure_ctrl = variable_controls(
    "Reservoir Pressure", "pressure",
    measured_default="#d97a6c",
    model_default="#6f35a5",
    dot_default=7,
    quality_default=0.86,
    smooth_default=15,
    noise_default=0.008,
    mismatch_count_default=3,
    mismatch_strength_default=0.03,
    dot_stride_default=3
)

oil_cfg = unpack(oil_ctrl)
gas_cfg = unpack(gas_ctrl)
water_cfg = unpack(water_ctrl)
wcut_cfg = unpack(wcut_ctrl)
gor_cfg = unpack(gor_ctrl)
pressure_cfg = unpack(pressure_ctrl)

# =========================================================
# Create model history-match curves
# =========================================================
oil_model = make_realistic_model_from_measured(
    oil_measured,
    quality=oil_cfg["quality"],
    bias_pct=oil_cfg["bias"],
    smoothness=oil_cfg["smoothness"],
    noise_factor=oil_cfg["noise"],
    mismatch_count=oil_cfg["mismatch_count"],
    mismatch_strength=oil_cfg["mismatch_strength"],
    seed=seed + 1
)

gas_model = make_realistic_model_from_measured(
    gas_measured,
    quality=gas_cfg["quality"],
    bias_pct=gas_cfg["bias"],
    smoothness=gas_cfg["smoothness"],
    noise_factor=gas_cfg["noise"],
    mismatch_count=gas_cfg["mismatch_count"],
    mismatch_strength=gas_cfg["mismatch_strength"],
    seed=seed + 2
)

water_model = make_realistic_model_from_measured(
    water_measured,
    quality=water_cfg["quality"],
    bias_pct=water_cfg["bias"],
    smoothness=water_cfg["smoothness"],
    noise_factor=water_cfg["noise"],
    mismatch_count=water_cfg["mismatch_count"],
    mismatch_strength=water_cfg["mismatch_strength"],
    seed=seed + 3
)

wcut_model = make_realistic_model_from_measured(
    wcut_measured,
    quality=wcut_cfg["quality"],
    bias_pct=wcut_cfg["bias"],
    smoothness=wcut_cfg["smoothness"],
    noise_factor=wcut_cfg["noise"],
    mismatch_count=wcut_cfg["mismatch_count"],
    mismatch_strength=wcut_cfg["mismatch_strength"],
    seed=seed + 4
)
wcut_model = np.clip(wcut_model, 0, 100)

gor_model = make_realistic_model_from_measured(
    gor_measured,
    quality=gor_cfg["quality"],
    bias_pct=gor_cfg["bias"],
    smoothness=gor_cfg["smoothness"],
    noise_factor=gor_cfg["noise"],
    mismatch_count=gor_cfg["mismatch_count"],
    mismatch_strength=gor_cfg["mismatch_strength"],
    seed=seed + 5
)

pressure_model = make_realistic_model_from_measured(
    pressure_measured,
    quality=pressure_cfg["quality"],
    bias_pct=pressure_cfg["bias"],
    smoothness=pressure_cfg["smoothness"],
    noise_factor=pressure_cfg["noise"],
    mismatch_count=pressure_cfg["mismatch_count"],
    mismatch_strength=pressure_cfg["mismatch_strength"],
    seed=seed + 6
)
pressure_model = np.clip(pressure_model, 80, 155)

# =========================================================
# Main history match plot
# =========================================================
history_fig = make_subplots(
    rows=3,
    cols=2,
    subplot_titles=(
        "OIL_RATE, m³/d – DATE",
        "GAS_RATE, Mm³/d – DATE",
        "WATER_RATE, m³/d – DATE",
        "WCUT, % – DATE",
        "GOR, m³/m³ – DATE",
        "RESERVOIR PRESSURE, bar – DATE",
    ),
    vertical_spacing=0.12,
    horizontal_spacing=0.08
)

add_history_trace(
    history_fig, 1, 1,
    dates, oil_measured, oil_model,
    "Measured Data", "Model",
    "OIL_RATE, m³/d",
    oil_cfg["measured_color"],
    oil_cfg["model_color"],
    oil_cfg["dot_size"],
    oil_cfg["dot_stride"],
    show_legend=True
)

add_history_trace(
    history_fig, 1, 2,
    dates, gas_measured, gas_model,
    "Measured Data", "Model",
    "GAS_RATE, Mm³/d",
    gas_cfg["measured_color"],
    gas_cfg["model_color"],
    gas_cfg["dot_size"],
    gas_cfg["dot_stride"],
    show_legend=False
)

add_history_trace(
    history_fig, 2, 1,
    dates, water_measured, water_model,
    "Measured Data", "Model",
    "WATER_RATE, m³/d",
    water_cfg["measured_color"],
    water_cfg["model_color"],
    water_cfg["dot_size"],
    water_cfg["dot_stride"],
    show_legend=False
)

add_history_trace(
    history_fig, 2, 2,
    dates, wcut_measured, wcut_model,
    "Measured Data", "Model",
    "WCUT, %",
    wcut_cfg["measured_color"],
    wcut_cfg["model_color"],
    wcut_cfg["dot_size"],
    wcut_cfg["dot_stride"],
    show_legend=False
)

add_history_trace(
    history_fig, 3, 1,
    dates, gor_measured, gor_model,
    "Measured Data", "Model",
    "GOR, m³/m³",
    gor_cfg["measured_color"],
    gor_cfg["model_color"],
    gor_cfg["dot_size"],
    gor_cfg["dot_stride"],
    show_legend=False
)

add_history_trace(
    history_fig, 3, 2,
    dates, pressure_measured, pressure_model,
    "Measured Data", "Model",
    "P, bar",
    pressure_cfg["measured_color"],
    pressure_cfg["model_color"],
    pressure_cfg["dot_size"],
    pressure_cfg["dot_stride"],
    show_legend=False
)

history_fig.update_layout(
    title_text="Artificial Zira History Match – Oil First Layout",
    title_x=0.5,
    height=1100,
    template="plotly_white",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.08,
        xanchor="center",
        x=0.5
    )
)

history_fig.update_xaxes(showgrid=True, gridcolor="#D5E5F7")
history_fig.update_yaxes(showgrid=True, gridcolor="#D5E5F7")

st.plotly_chart(history_fig, use_container_width=True)

# =========================================================
# Relative Permeability – Corey Function
# =========================================================
st.header("Relative Permeability – Corey Function Playground")

with st.sidebar.expander("Water-Oil Corey Controls", expanded=False):
    st.markdown("**Before History Match**")
    swc_b = st.slider("WO Before Swc", 0.05, 0.45, 0.25, 0.01)
    sorw_b = st.slider("WO Before Sorw", 0.05, 0.45, 0.22, 0.01)
    krw_end_b = st.slider("WO Before Krw end", 0.10, 1.00, 0.95, 0.01)
    krow_end_b = st.slider("WO Before Kro end", 0.10, 1.00, 1.00, 0.01)
    nw_b = st.slider("WO Before nw", 1.0, 6.0, 2.60, 0.05)
    no_b = st.slider("WO Before no", 1.0, 6.0, 2.10, 0.05)

    st.markdown("**After History Match**")
    swc_a = st.slider("WO After Swc", 0.05, 0.45, 0.30, 0.01)
    sorw_a = st.slider("WO After Sorw", 0.05, 0.45, 0.22, 0.01)
    krw_end_a = st.slider("WO After Krw end", 0.10, 1.00, 0.60, 0.01)
    krow_end_a = st.slider("WO After Kro end", 0.10, 1.00, 0.96, 0.01)
    nw_a = st.slider("WO After nw", 1.0, 6.0, 3.80, 0.05)
    no_a = st.slider("WO After no", 1.0, 6.0, 3.20, 0.05)

with st.sidebar.expander("Gas-Oil Corey Controls", expanded=False):
    st.markdown("**Before History Match**")
    sgc_b = st.slider("GO Before Sgc", 0.00, 0.30, 0.05, 0.01)
    sorg_b = st.slider("GO Before Sorg", 0.05, 0.45, 0.25, 0.01)
    krg_end_b = st.slider("GO Before Krg end", 0.10, 1.00, 1.00, 0.01)
    krog_end_b = st.slider("GO Before Krog end", 0.10, 1.00, 1.00, 0.01)
    ng_b = st.slider("GO Before ng", 1.0, 6.0, 3.00, 0.05)
    nog_b = st.slider("GO Before nog", 1.0, 6.0, 2.00, 0.05)

    st.markdown("**After History Match**")
    sgc_a = st.slider("GO After Sgc", 0.00, 0.30, 0.08, 0.01)
    sorg_a = st.slider("GO After Sorg", 0.05, 0.45, 0.28, 0.01)
    krg_end_a = st.slider("GO After Krg end", 0.10, 1.00, 0.72, 0.01)
    krog_end_a = st.slider("GO After Krog end", 0.10, 1.00, 0.96, 0.01)
    ng_a = st.slider("GO After ng", 1.0, 6.0, 4.00, 0.05)
    nog_a = st.slider("GO After nog", 1.0, 6.0, 2.80, 0.05)

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
        height=520
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
        height=520
    )
    fig_go.update_xaxes(range=[0, 1], showgrid=True, gridcolor="#D5E5F7")
    fig_go.update_yaxes(range=[0, 1.05], showgrid=True, gridcolor="#D5E5F7")
    st.plotly_chart(fig_go, use_container_width=True)

with tab3:
    overlay_type = st.radio(
        "Overlay type",
        ["Water-Oil only", "Gas-Oil only", "Both"],
        horizontal=True
    )

    if overlay_type == "Both":
        fig_overlay = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Water-Oil Before vs After", "Gas-Oil Before vs After")
        )
    else:
        fig_overlay = make_subplots(
            rows=1,
            cols=1,
            subplot_titles=(f"{overlay_type} Before vs After",)
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
        title_x=0.5
    )
    fig_overlay.update_xaxes(showgrid=True, gridcolor="#D5E5F7")
    fig_overlay.update_yaxes(showgrid=True, gridcolor="#D5E5F7")
    st.plotly_chart(fig_overlay, use_container_width=True)

# =========================================================
# Data tables
# =========================================================
history_df = pd.DataFrame({
    "Date": dates,
    "Oil_Rate_Measured_m3d": oil_measured,
    "Oil_Rate_Model_m3d": oil_model,
    "Gas_Rate_Measured_Mm3d": gas_measured,
    "Gas_Rate_Model_Mm3d": gas_model,
    "Water_Rate_Measured_m3d": water_measured,
    "Water_Rate_Model_m3d": water_model,
    "Water_Cut_Measured_pct": wcut_measured,
    "Water_Cut_Model_pct": wcut_model,
    "GOR_Measured_m3m3": gor_measured,
    "GOR_Model_m3m3": gor_model,
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

controls_df = pd.DataFrame({
    "Parameter": [
        "Seed",

        "Oil quality", "Oil bias", "Oil smoothness", "Oil noise", "Oil mismatch count", "Oil mismatch strength", "Oil dot stride",
        "Gas quality", "Gas bias", "Gas smoothness", "Gas noise", "Gas mismatch count", "Gas mismatch strength", "Gas dot stride",
        "Water quality", "Water bias", "Water smoothness", "Water noise", "Water mismatch count", "Water mismatch strength", "Water dot stride",
        "Water cut quality", "Water cut bias", "Water cut smoothness", "Water cut noise", "Water cut mismatch count", "Water cut mismatch strength", "Water cut dot stride",
        "GOR quality", "GOR bias", "GOR smoothness", "GOR noise", "GOR mismatch count", "GOR mismatch strength", "GOR dot stride",
        "Pressure quality", "Pressure bias", "Pressure smoothness", "Pressure noise", "Pressure mismatch count", "Pressure mismatch strength", "Pressure dot stride",

        "WO before Swc", "WO before Sorw", "WO before Krw end", "WO before Kro end", "WO before nw", "WO before no",
        "WO after Swc", "WO after Sorw", "WO after Krw end", "WO after Kro end", "WO after nw", "WO after no",

        "GO before Sgc", "GO before Sorg", "GO before Krg end", "GO before Krog end", "GO before ng", "GO before nog",
        "GO after Sgc", "GO after Sorg", "GO after Krg end", "GO after Krog end", "GO after ng", "GO after nog",
    ],
    "Value": [
        seed,

        oil_cfg["quality"], oil_cfg["bias"], oil_cfg["smoothness"], oil_cfg["noise"], oil_cfg["mismatch_count"], oil_cfg["mismatch_strength"], oil_cfg["dot_stride"],
        gas_cfg["quality"], gas_cfg["bias"], gas_cfg["smoothness"], gas_cfg["noise"], gas_cfg["mismatch_count"], gas_cfg["mismatch_strength"], gas_cfg["dot_stride"],
        water_cfg["quality"], water_cfg["bias"], water_cfg["smoothness"], water_cfg["noise"], water_cfg["mismatch_count"], water_cfg["mismatch_strength"], water_cfg["dot_stride"],
        wcut_cfg["quality"], wcut_cfg["bias"], wcut_cfg["smoothness"], wcut_cfg["noise"], wcut_cfg["mismatch_count"], wcut_cfg["mismatch_strength"], wcut_cfg["dot_stride"],
        gor_cfg["quality"], gor_cfg["bias"], gor_cfg["smoothness"], gor_cfg["noise"], gor_cfg["mismatch_count"], gor_cfg["mismatch_strength"], gor_cfg["dot_stride"],
        pressure_cfg["quality"], pressure_cfg["bias"], pressure_cfg["smoothness"], pressure_cfg["noise"], pressure_cfg["mismatch_count"], pressure_cfg["mismatch_strength"], pressure_cfg["dot_stride"],

        swc_b, sorw_b, krw_end_b, krow_end_b, nw_b, no_b,
        swc_a, sorw_a, krw_end_a, krow_end_a, nw_a, no_a,

        sgc_b, sorg_b, krg_end_b, krog_end_b, ng_b, nog_b,
        sgc_a, sorg_a, krg_end_a, krog_end_a, ng_a, nog_a,
    ]
})

with st.expander("Show generated data tables"):
    st.subheader("History Match Data")
    st.dataframe(history_df, use_container_width=True)

    st.subheader("Water-Oil RelPerm")
    st.dataframe(relperm_wo_df, use_container_width=True)

    st.subheader("Gas-Oil RelPerm")
    st.dataframe(relperm_go_df, use_container_width=True)

# =========================================================
# Excel download
# =========================================================
excel_file = make_excel(history_df, relperm_wo_df, relperm_go_df, controls_df)

st.download_button(
    label="Download Excel – Artificial Zira History Match & RelPerm",
    data=excel_file,
    file_name="Zira_Artificial_HistoryMatch_RelPerm_Playground.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
