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
# General helpers
# =========================================================
def rolling_np(values, window):
    return pd.Series(values).rolling(window, min_periods=1, center=True).mean().to_numpy()


def make_model_from_measured(
    measured,
    quality=0.85,
    bias_pct=0.0,
    noise_factor=0.05,
    smooth_window=5,
    seed=2024,
):
    """
    quality = 1.0 -> near ideal history match
    quality = 0.0 -> poor / smoothed / biased match
    """
    rng = np.random.default_rng(seed)

    measured = np.asarray(measured, dtype=float)
    scale = np.nanmax(measured) - np.nanmin(measured)
    if scale <= 0:
        scale = 1.0

    smooth = rolling_np(measured, smooth_window)

    poor_component = smooth * (1 + bias_pct / 100.0)
    poor_component += rng.normal(0, noise_factor * scale, len(measured))

    ideal_component = measured + rng.normal(0, 0.01 * scale, len(measured))

    model = quality * ideal_component + (1 - quality) * poor_component
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
    show_legend=False,
):
    # Measured = dots only
    fig.add_trace(
        go.Scatter(
            x=x,
            y=measured,
            mode="markers",
            name=measured_name,
            marker=dict(
                color=measured_color,
                size=dot_size,
                opacity=0.78,
                line=dict(color="rgba(80,80,80,0.7)", width=0.8)
            ),
            showlegend=show_legend
        ),
        row=row,
        col=col
    )

    # Model = line
    fig.add_trace(
        go.Scatter(
            x=x,
            y=model,
            mode="lines",
            name=model_name,
            line=dict(color=model_color, width=2.5),
            showlegend=show_legend
        ),
        row=row,
        col=col
    )

    fig.update_yaxes(title_text=y_title, row=row, col=col)
    fig.update_xaxes(title_text="DATE", row=row, col=col)


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

        for sheet_name, df in {
            "History_Match": history_df,
            "RelPerm_WaterOil": relperm_wo_df,
            "RelPerm_GasOil": relperm_go_df,
            "Controls": controls_df,
        }.items():
            ws = writer.sheets[sheet_name]
            ws.freeze_panes(1, 0)
            ws.set_column(0, len(df.columns) - 1, 18)

            for c, col in enumerate(df.columns):
                ws.write(0, c, col, header_fmt)

    output.seek(0)
    return output


# =========================================================
# Synthetic Zira-like measured history data
# =========================================================
st.title("Zira Field – Artificial History Match & RelPerm Generator")
st.caption(
    "Synthetic/artificial curves for draft paper formatting only. "
    "All final values should be replaced after real Zira model runs."
)

st.sidebar.header("Global Controls")

seed = st.sidebar.number_input(
    "Random seed",
    min_value=1,
    max_value=99999,
    value=2024,
    step=1
)

dates = pd.date_range("1956-01-01", "2024-12-01", freq="QS")
year = dates.year + (dates.dayofyear - 1) / 365.25
rng = np.random.default_rng(seed)

# Oil rate: early peak, rapid decline, long low tail
oil_peak = 1250 * np.exp(-((year - 1962.2) / 1.55) ** 2)
oil_tail = 310 * np.exp(-0.075 * np.maximum(year - 1966, 0))
oil_measured = oil_peak + oil_tail
oil_measured += rng.normal(0, 25, len(dates))
oil_measured = np.clip(oil_measured, 0, None)

# Gas rate: higher early peak, rapid decline
gas_peak = 4200 * np.exp(-((year - 1961.8) / 1.45) ** 2)
gas_tail = 520 * np.exp(-0.065 * np.maximum(year - 1965, 0))
gas_measured = gas_peak + gas_tail
gas_measured += rng.normal(0, 110, len(dates))
gas_measured = np.clip(gas_measured, 0, None)

# Water rate: follows oil period, then declines but noisy
water_base = 90 * np.exp(-0.025 * np.maximum(year - 1962, 0))
water_hump = 70 * np.exp(-((year - 2010) / 8.5) ** 2)
water_measured = water_base + water_hump
water_measured += rng.normal(0, 9, len(dates))
water_measured = np.clip(water_measured, 0, None)

# Water cut: rises toward mature high water-cut behavior
wcut_measured = 20 + 78 / (1 + np.exp(-(year - 1967) / 3.2))
wcut_measured += 10 * np.sin((year - 1956) / 2.1) * np.exp(-0.035 * np.maximum(year - 1965, 0))
wcut_measured += rng.normal(0, 4.5, len(dates))
wcut_measured = np.clip(wcut_measured, 0, 100)

# GOR: early peak, decline, late spikes
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

# Reservoir pressure: declining, sparse-like behavior but shown as continuous artificial measured dots
pressure_trend = 142 - 45 * (1 - np.exp(-0.025 * np.maximum(year - 1956, 0)))
pressure_measured = pressure_trend + rng.normal(0, 2.8, len(dates))
pressure_measured = np.clip(pressure_measured, 85, 150)


# =========================================================
# History match controls
# =========================================================
st.sidebar.header("History Match Controls")
st.sidebar.caption("1.0 = ideal match, 0.0 = poor match")

def variable_controls(label, key, measured_default, model_default, dot_default=7):
    with st.sidebar.expander(label, expanded=False):
        quality = st.slider(
            f"{label} match quality",
            0.0,
            1.0,
            0.86,
            0.01,
            key=f"{key}_quality"
        )
        bias = st.slider(
            f"{label} model bias, %",
            -50,
            50,
            0,
            1,
            key=f"{key}_bias"
        )
        noise = st.slider(
            f"{label} poor-match noise",
            0.00,
            0.35,
            0.08,
            0.01,
            key=f"{key}_noise"
        )
        smooth = st.slider(
            f"{label} poor-match smoothing",
            1,
            25,
            6,
            1,
            key=f"{key}_smooth"
        )
        dot_size = st.slider(
            f"{label} measured dot size",
            2,
            18,
            dot_default,
            1,
            key=f"{key}_dot_size"
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

    return quality, bias, noise, smooth, dot_size, measured_color, model_color


oil_ctrl = variable_controls("Oil Rate", "oil", "#e95d5d", "#00aa55", 8)
gas_ctrl = variable_controls("Gas Rate", "gas", "#e95d5d", "#ff77dd", 7)
water_ctrl = variable_controls("Water Rate", "water", "#e95d5d", "#0b3c8c", 7)
wcut_ctrl = variable_controls("Water Cut", "wcut", "#e95d5d", "#ff77dd", 7)
gor_ctrl = variable_controls("GOR", "gor", "#e95d5d", "#ff77dd", 7)
pressure_ctrl = variable_controls("Reservoir Pressure", "pressure", "#e95d5d", "#6f35a5", 8)


def unpack(ctrl):
    return {
        "quality": ctrl[0],
        "bias": ctrl[1],
        "noise": ctrl[2],
        "smooth": ctrl[3],
        "dot_size": ctrl[4],
        "measured_color": ctrl[5],
        "model_color": ctrl[6],
    }


oil_cfg = unpack(oil_ctrl)
gas_cfg = unpack(gas_ctrl)
water_cfg = unpack(water_ctrl)
wcut_cfg = unpack(wcut_ctrl)
gor_cfg = unpack(gor_ctrl)
pressure_cfg = unpack(pressure_ctrl)

oil_model = make_model_from_measured(
    oil_measured,
    oil_cfg["quality"],
    oil_cfg["bias"],
    oil_cfg["noise"],
    oil_cfg["smooth"],
    seed + 1
)
gas_model = make_model_from_measured(
    gas_measured,
    gas_cfg["quality"],
    gas_cfg["bias"],
    gas_cfg["noise"],
    gas_cfg["smooth"],
    seed + 2
)
water_model = make_model_from_measured(
    water_measured,
    water_cfg["quality"],
    water_cfg["bias"],
    water_cfg["noise"],
    water_cfg["smooth"],
    seed + 3
)
wcut_model = make_model_from_measured(
    wcut_measured,
    wcut_cfg["quality"],
    wcut_cfg["bias"],
    wcut_cfg["noise"],
    wcut_cfg["smooth"],
    seed + 4
)
wcut_model = np.clip(wcut_model, 0, 100)

gor_model = make_model_from_measured(
    gor_measured,
    gor_cfg["quality"],
    gor_cfg["bias"],
    gor_cfg["noise"],
    gor_cfg["smooth"],
    seed + 5
)

pressure_model = make_model_from_measured(
    pressure_measured,
    pressure_cfg["quality"],
    pressure_cfg["bias"],
    pressure_cfg["noise"],
    pressure_cfg["smooth"],
    seed + 6
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
    dates,
    oil_measured,
    oil_model,
    "Measured Oil",
    "Model Oil",
    "OIL_RATE, m³/d",
    oil_cfg["measured_color"],
    oil_cfg["model_color"],
    oil_cfg["dot_size"],
    True
)

add_history_trace(
    history_fig, 1, 2,
    dates,
    gas_measured,
    gas_model,
    "Measured Gas",
    "Model Gas",
    "GAS_RATE, Mm³/d",
    gas_cfg["measured_color"],
    gas_cfg["model_color"],
    gas_cfg["dot_size"],
    True
)

add_history_trace(
    history_fig, 2, 1,
    dates,
    water_measured,
    water_model,
    "Measured Water",
    "Model Water",
    "WATER_RATE, m³/d",
    water_cfg["measured_color"],
    water_cfg["model_color"],
    water_cfg["dot_size"],
    True
)

add_history_trace(
    history_fig, 2, 2,
    dates,
    wcut_measured,
    wcut_model,
    "Measured WCUT",
    "Model WCUT",
    "WCUT, %",
    wcut_cfg["measured_color"],
    wcut_cfg["model_color"],
    wcut_cfg["dot_size"],
    True
)

add_history_trace(
    history_fig, 3, 1,
    dates,
    gor_measured,
    gor_model,
    "Measured GOR",
    "Model GOR",
    "GOR, m³/m³",
    gor_cfg["measured_color"],
    gor_cfg["model_color"],
    gor_cfg["dot_size"],
    True
)

add_history_trace(
    history_fig, 3, 2,
    dates,
    pressure_measured,
    pressure_model,
    "Measured Pressure",
    "Model Pressure",
    "P, bar",
    pressure_cfg["measured_color"],
    pressure_cfg["model_color"],
    pressure_cfg["dot_size"],
    True
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
# Relative permeability Corey functions
# =========================================================
st.header("Relative Permeability – Corey Function Playground")

with st.sidebar.expander("Water-Oil Corey Controls", expanded=False):
    st.markdown("**Before History Match**")
    swc_b = st.slider("WO Before Swc", 0.05, 0.45, 0.25, 0.01)
    sorw_b = st.slider("WO Before Sorw", 0.05, 0.45, 0.22, 0.01)
    krw_end_b = st.slider("WO Before Krw end", 0.1, 1.0, 0.95, 0.01)
    krow_end_b = st.slider("WO Before Kro end", 0.1, 1.0, 1.00, 0.01)
    nw_b = st.slider("WO Before nw", 1.0, 6.0, 2.60, 0.05)
    no_b = st.slider("WO Before no", 1.0, 6.0, 2.10, 0.05)

    st.markdown("**After History Match**")
    swc_a = st.slider("WO After Swc", 0.05, 0.45, 0.30, 0.01)
    sorw_a = st.slider("WO After Sorw", 0.05, 0.45, 0.22, 0.01)
    krw_end_a = st.slider("WO After Krw end", 0.1, 1.0, 0.60, 0.01)
    krow_end_a = st.slider("WO After Kro end", 0.1, 1.0, 0.96, 0.01)
    nw_a = st.slider("WO After nw", 1.0, 6.0, 3.80, 0.05)
    no_a = st.slider("WO After no", 1.0, 6.0, 3.20, 0.05)

with st.sidebar.expander("Gas-Oil Corey Controls", expanded=False):
    st.markdown("**Before History Match**")
    sgc_b = st.slider("GO Before Sgc", 0.00, 0.30, 0.05, 0.01)
    sorg_b = st.slider("GO Before Sorg", 0.05, 0.45, 0.25, 0.01)
    krg_end_b = st.slider("GO Before Krg end", 0.1, 1.0, 1.00, 0.01)
    krog_end_b = st.slider("GO Before Krog end", 0.1, 1.0, 1.00, 0.01)
    ng_b = st.slider("GO Before ng", 1.0, 6.0, 3.00, 0.05)
    nog_b = st.slider("GO Before nog", 1.0, 6.0, 2.00, 0.05)

    st.markdown("**After History Match**")
    sgc_a = st.slider("GO After Sgc", 0.00, 0.30, 0.08, 0.01)
    sorg_a = st.slider("GO After Sorg", 0.05, 0.45, 0.28, 0.01)
    krg_end_a = st.slider("GO After Krg end", 0.1, 1.0, 0.72, 0.01)
    krog_end_a = st.slider("GO After Krog end", 0.1, 1.0, 0.96, 0.01)
    ng_a = st.slider("GO After ng", 1.0, 6.0, 4.00, 0.05)
    nog_a = st.slider("GO After nog", 1.0, 6.0, 2.80, 0.05)


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

    fig_overlay = make_subplots(
        rows=1,
        cols=2 if overlay_type == "Both" else 1,
        subplot_titles=(
            ["Water-Oil Before vs After", "Gas-Oil Before vs After"]
            if overlay_type == "Both"
            else [f"{overlay_type} Before vs After"]
        )
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
# Dataframes and Excel download
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
        "Oil quality", "Gas quality", "Water quality", "Water cut quality", "GOR quality", "Pressure quality",
        "WO before Swc", "WO before Sorw", "WO before Krw end", "WO before Kro end", "WO before nw", "WO before no",
        "WO after Swc", "WO after Sorw", "WO after Krw end", "WO after Kro end", "WO after nw", "WO after no",
        "GO before Sgc", "GO before Sorg", "GO before Krg end", "GO before Krog end", "GO before ng", "GO before nog",
        "GO after Sgc", "GO after Sorg", "GO after Krg end", "GO after Krog end", "GO after ng", "GO after nog",
    ],
    "Value": [
        seed,
        oil_cfg["quality"], gas_cfg["quality"], water_cfg["quality"], wcut_cfg["quality"], gor_cfg["quality"], pressure_cfg["quality"],
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

excel_file = make_excel(history_df, relperm_wo_df, relperm_go_df, controls_df)

st.download_button(
    label="Download Excel – Artificial Zira History Match & RelPerm",
    data=excel_file,
    file_name="Zira_Artificial_HistoryMatch_RelPerm_Playground.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
