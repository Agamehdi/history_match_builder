# app.py
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Zira Artificial History Match & RelPerm",
    layout="wide"
)

# -----------------------------
# Helper functions
# -----------------------------
def add_actual_model_trace(fig, row, col, x, actual, model, title, y_title):
    fig.add_trace(
        go.Scatter(
            x=x,
            y=actual,
            mode="lines",
            name="Actual",
            line=dict(color="black", width=2),
            showlegend=(row == 1 and col == 1)
        ),
        row=row,
        col=col
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=model,
            mode="lines",
            name="Model",
            line=dict(color="#ff77dd", width=2),
            showlegend=(row == 1 and col == 1)
        ),
        row=row,
        col=col
    )

    fig.update_xaxes(title_text="DATE", row=row, col=col)
    fig.update_yaxes(title_text=y_title, row=row, col=col)


def make_excel_download(history_df, relperm_wo_df, relperm_go_df):
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        history_df.to_excel(writer, index=False, sheet_name="History_Match")
        relperm_wo_df.to_excel(writer, index=False, sheet_name="RelPerm_WO")
        relperm_go_df.to_excel(writer, index=False, sheet_name="RelPerm_GO")

        workbook = writer.book

        header_fmt = workbook.add_format({
            "bold": True,
            "bg_color": "#D9EAF7",
            "border": 1
        })

        for sheet_name in ["History_Match", "RelPerm_WO", "RelPerm_GO"]:
            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes(1, 0)
            worksheet.set_column(0, 20, 18)

            df = {
                "History_Match": history_df,
                "RelPerm_WO": relperm_wo_df,
                "RelPerm_GO": relperm_go_df
            }[sheet_name]

            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_fmt)

    output.seek(0)
    return output


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Synthetic Zira Input Controls")

seed = st.sidebar.number_input("Random seed", min_value=1, max_value=9999, value=2024, step=1)
noise_level = st.sidebar.slider("Model noise level", 0.00, 0.30, 0.08, 0.01)

oil_scale = st.sidebar.slider("Oil rate scale", 0.5, 2.0, 1.0, 0.05)
gas_scale = st.sidebar.slider("Gas rate scale", 0.5, 2.0, 1.0, 0.05)
wcut_shift = st.sidebar.slider("Water cut shift, %", -20, 20, 0, 1)
gor_scale = st.sidebar.slider("GOR scale", 0.5, 2.0, 1.0, 0.05)

np.random.seed(seed)

# -----------------------------
# Artificial Zira-like history data
# -----------------------------
dates = pd.date_range("1956-01-01", "2024-12-01", freq="QS")
year = dates.year + (dates.dayofyear - 1) / 365.25
t = np.linspace(0, 1, len(dates))

# Gas rate: big early peak, rapid decline, low tail
gas_peak = 4300 * np.exp(-((year - 1961.5) / 1.6) ** 2)
gas_tail = 450 * np.exp(-0.07 * np.maximum(year - 1965, 0))
gas_actual = (gas_peak + gas_tail) * gas_scale
gas_actual += np.random.normal(0, 90, len(dates))
gas_actual = np.clip(gas_actual, 0, None)

gas_model = gas_actual * (1 + np.random.normal(0, noise_level, len(dates)))
gas_model = pd.Series(gas_model).rolling(2, min_periods=1).mean().to_numpy()
gas_model = np.clip(gas_model, 0, None)

# Oil rate: early high peak then long decline
oil_peak = 1300 * np.exp(-((year - 1961.8) / 1.9) ** 2)
oil_decline = 280 * np.exp(-0.10 * np.maximum(year - 1966, 0))
oil_actual = (oil_peak + oil_decline) * oil_scale
oil_actual += np.random.normal(0, 25, len(dates))
oil_actual = np.clip(oil_actual, 0, None)

oil_model = oil_actual * (1 + np.random.normal(0, noise_level, len(dates)))
oil_model = pd.Series(oil_model).rolling(2, min_periods=1).mean().to_numpy()
oil_model = np.clip(oil_model, 0, None)

# Water cut: increasing with fluctuations, similar to mature field
wcut_base = 20 + 78 / (1 + np.exp(-(year - 1965) / 3.0))
wcut_actual = wcut_base + 12 * np.sin((year - 1956) / 2.3) * np.exp(-0.03 * np.maximum(year - 1960, 0))
wcut_actual += np.random.normal(0, 5, len(dates)) + wcut_shift
wcut_actual = np.clip(wcut_actual, 0, 100)

wcut_model = wcut_actual * (1 + np.random.normal(0, noise_level / 2, len(dates)))
wcut_model = pd.Series(wcut_model).rolling(3, min_periods=1).mean().to_numpy()
wcut_model = np.clip(wcut_model, 0, 100)

# GOR: early peak, decline, late noisy spikes
gor_peak = 3800 * np.exp(-((year - 1961.0) / 2.5) ** 2)
gor_decline = 900 * np.exp(-0.06 * np.maximum(year - 1965, 0))
late_spikes = np.where(
    year > 1995,
    np.random.choice([0, 0, 0, 400, 900, 1600], size=len(dates)),
    0
)
gor_actual = (gor_peak + gor_decline + late_spikes) * gor_scale
gor_actual += np.random.normal(0, 180, len(dates))
gor_actual = np.clip(gor_actual, 0, None)

gor_model = gor_actual * (1 + np.random.normal(0, noise_level, len(dates)))
gor_model = pd.Series(gor_model).rolling(3, min_periods=1).mean().to_numpy()
gor_model = np.clip(gor_model, 0, None)

history_df = pd.DataFrame({
    "Date": dates,
    "Oil_Rate_Actual_m3d": oil_actual,
    "Oil_Rate_Model_m3d": oil_model,
    "Gas_Rate_Actual_Mm3d": gas_actual,
    "Gas_Rate_Model_Mm3d": gas_model,
    "Water_Cut_Actual_pct": wcut_actual,
    "Water_Cut_Model_pct": wcut_model,
    "GOR_Actual_m3m3": gor_actual,
    "GOR_Model_m3m3": gor_model,
})

# -----------------------------
# Relative permeability data
# -----------------------------
sw = np.linspace(0.25, 1.0, 31)
swc = 0.25
sor = 0.22
sew = np.clip((sw - swc) / (1 - swc - sor), 0, 1)

krw = 0.95 * sew ** 2.6
krow = 1.00 * (1 - sew) ** 2.1

relperm_wo_df = pd.DataFrame({
    "Sw": sw,
    "KRW": krw,
    "KROW": krow
})

sg = np.linspace(0.0, 0.78, 31)
sgc = 0.05
sorg = 0.25
seg = np.clip((sg - sgc) / (1 - sgc - sorg), 0, 1)

krg = 1.0 * seg ** 3.0
krog = 1.0 * (1 - seg) ** 2.0

relperm_go_df = pd.DataFrame({
    "Sg": sg,
    "KRG": krg,
    "KROG": krog
})

# -----------------------------
# App title
# -----------------------------
st.title("Zira Field – Artificial History Match & RelPerm Generator")
st.caption(
    "Synthetic/artificial curves for draft paper formatting only. "
    "Replace with final Nexus/OFM/Excel export after real Zira model runs."
)

# -----------------------------
# History match plot
# -----------------------------
fig_hm = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        "GAS_RATE, Mm³/d – DATE",
        "WCUT, % – DATE",
        "OIL_RATE, m³/d – DATE",
        "GOR, m³/m³ – DATE"
    ),
    vertical_spacing=0.16,
    horizontal_spacing=0.10
)

add_actual_model_trace(
    fig_hm, 1, 1,
    history_df["Date"],
    history_df["Gas_Rate_Actual_Mm3d"],
    history_df["Gas_Rate_Model_Mm3d"],
    "Gas Rate",
    "GAS_RATE, Mm³/d"
)

add_actual_model_trace(
    fig_hm, 1, 2,
    history_df["Date"],
    history_df["Water_Cut_Actual_pct"],
    history_df["Water_Cut_Model_pct"],
    "Water Cut",
    "WCUT, %"
)

add_actual_model_trace(
    fig_hm, 2, 1,
    history_df["Date"],
    history_df["Oil_Rate_Actual_m3d"],
    history_df["Oil_Rate_Model_m3d"],
    "Oil Rate",
    "OIL_RATE, m³/d"
)

add_actual_model_trace(
    fig_hm, 2, 2,
    history_df["Date"],
    history_df["GOR_Actual_m3m3"],
    history_df["GOR_Model_m3m3"],
    "GOR",
    "GOR, m³/m³"
)

fig_hm.update_layout(
    height=850,
    title_text="Figure 67: History Match for All Flow Units – Artificial Zira Dataset",
    title_x=0.5,
    template="plotly_white",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.12,
        xanchor="center",
        x=0.5
    )
)

fig_hm.update_xaxes(showgrid=True, gridcolor="#e6e6e6")
fig_hm.update_yaxes(showgrid=True, gridcolor="#e6e6e6")

st.plotly_chart(fig_hm, use_container_width=True)

# -----------------------------
# RelPerm plots
# -----------------------------
st.subheader("Relative Permeability Curves")

col1, col2 = st.columns(2)

with col1:
    fig_wo = go.Figure()
    fig_wo.add_trace(go.Scatter(
        x=relperm_wo_df["Sw"],
        y=relperm_wo_df["KRW"],
        mode="lines+markers",
        name="KRW"
    ))
    fig_wo.add_trace(go.Scatter(
        x=relperm_wo_df["Sw"],
        y=relperm_wo_df["KROW"],
        mode="lines+markers",
        name="KROW"
    ))
    fig_wo.update_layout(
        title="Water-Oil Relative Permeabilities",
        xaxis_title="Sw",
        yaxis_title="Relative Permeability",
        template="plotly_white",
        height=450
    )
    fig_wo.update_xaxes(range=[0, 1], showgrid=True, gridcolor="#e6e6e6")
    fig_wo.update_yaxes(range=[0, 1.05], showgrid=True, gridcolor="#e6e6e6")
    st.plotly_chart(fig_wo, use_container_width=True)

with col2:
    fig_go = go.Figure()
    fig_go.add_trace(go.Scatter(
        x=relperm_go_df["Sg"],
        y=relperm_go_df["KRG"],
        mode="lines+markers",
        name="KRG"
    ))
    fig_go.add_trace(go.Scatter(
        x=relperm_go_df["Sg"],
        y=relperm_go_df["KROG"],
        mode="lines+markers",
        name="KROG"
    ))
    fig_go.update_layout(
        title="Gas-Oil Relative Permeabilities",
        xaxis_title="Sg",
        yaxis_title="Relative Permeability",
        template="plotly_white",
        height=450
    )
    fig_go.update_xaxes(range=[0, 1], showgrid=True, gridcolor="#e6e6e6")
    fig_go.update_yaxes(range=[0, 1.05], showgrid=True, gridcolor="#e6e6e6")
    st.plotly_chart(fig_go, use_container_width=True)

# -----------------------------
# Tables
# -----------------------------
with st.expander("Show generated data tables"):
    st.write("History Match Data")
    st.dataframe(history_df, use_container_width=True)

    st.write("Water-Oil RelPerm")
    st.dataframe(relperm_wo_df, use_container_width=True)

    st.write("Gas-Oil RelPerm")
    st.dataframe(relperm_go_df, use_container_width=True)

# -----------------------------
# Excel download button
# -----------------------------
excel_file = make_excel_download(history_df, relperm_wo_df, relperm_go_df)

st.download_button(
    label="Download Artificial Zira Data as Excel",
    data=excel_file,
    file_name="Zira_Artificial_HistoryMatch_RelPerm.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)