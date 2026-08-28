import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Synthetic History Match Dashboard", layout="wide")

st.title("🛢️ Real Field Synthetic History Match & Subsurface QC Dashboard")
st.markdown("Material balansını qoruyan real yataq üçün sintetik history match və qrafiklər paneli.")

# --- SIDEBAR ---
st.sidebar.header("🎨 Styling & Control Panel")
meas_color = st.sidebar.color_picker("Measured Data Color", "#1f77b4")
meas_size = st.sidebar.slider("Measured Marker Size", 10, 100, 30)
meas_alpha = st.sidebar.slider("Measured Alpha", 0.1, 1.0, 0.8)

sim_color = st.sidebar.color_picker("Simulation Line Color", "#d62728")
sim_linewidth = st.sidebar.slider("Line Thickness", 0.5, 5.0, 2.0)
sim_linestyle = st.sidebar.selectbox("Line Style", ["-", "--", "-.", ":"], index=0)

mismatch_intensity = st.sidebar.slider("History Mismatch Intensity", 0.0, 3.0, 1.0)
pressure_drop_rate = st.sidebar.slider("MBAL Pressure Drop Factor", 5.0, 25.0, 12.5)

# --- DATA GENERATOR ---
@st.cache_data
def generate_data(mismatch_mult, p_drop_factor):
    np.random.seed(42)
    time_months = np.arange(1, 61)
    
    base_oil = 15000 * np.exp(-0.02 * time_months)
    oil_rate_sim = base_oil + np.random.normal(0, 300, len(time_months))
    oil_rate_meas = oil_rate_sim + np.random.normal(0, 700 * mismatch_mult, len(time_months))
    oil_rate_meas[20:30] += 1500 * mismatch_mult * np.sin(np.linspace(0, np.pi, 10))

    water_rate_sim = 2000 + 400 * time_months**0.7 + np.random.normal(0, 150, len(time_months))
    water_rate_meas = water_rate_sim + np.random.normal(0, 400 * mismatch_mult, len(time_months))

    gas_rate_sim = oil_rate_sim * (1.2 + 0.01 * time_months) + np.random.normal(0, 200, len(time_months))
    gas_rate_meas = gas_rate_sim + np.random.normal(0, 800 * mismatch_mult, len(time_months))

    total_liq_sim = oil_rate_sim + water_rate_sim
    wc_sim = (water_rate_sim / total_liq_sim) * 100
    wc_meas = np.clip(wc_sim + np.random.normal(0, 3 * mismatch_mult, len(time_months)), 0, 100)

    gor_sim = (gas_rate_sim / oil_rate_sim) * 1000
    gor_meas = gor_sim + np.random.normal(0, 150 * mismatch_mult, len(time_months))

    cum_oil_sim = np.cumsum(oil_rate_sim) / 1000.0
    cum_oil_meas = cum_oil_sim + np.cumsum(np.random.normal(0, 50, len(time_months))) / 1000.0

    initial_pressure = 4500.0
    pressure_sim = initial_pressure - (cum_oil_sim * p_drop_factor) + np.random.normal(0, 15, len(time_months))
    pressure_meas = pressure_sim + np.random.normal(0, 40 * mismatch_mult, len(time_months))
    pressure_meas[40:50] -= 120 * mismatch_mult  

    pvt_pressure = np.linspace(1000, 5000, 50)
    bo_sim = 1.05 + 0.00003 * pvt_pressure - 0.000000002 * pvt_pressure**2
    bo_meas = bo_sim + np.random.normal(0, 0.01, len(pvt_pressure))

    sw = np.linspace(0.2, 0.8, 40)
    krw_sim = 0.3 * ((sw - 0.2) / 0.6)**2.5
    kro_sim = 0.8 * ((0.8 - sw) / 0.6)**2.0
    krw_meas = np.clip(krw_sim + np.random.normal(0, 0.02, len(sw)), 0, 1)
    kro_meas = np.clip(kro_sim + np.random.normal(0, 0.02, len(sw)), 0, 1)

    return {
        "time": time_months,
        "oil_rate": (oil_rate_meas, oil_rate_sim),
        "water_rate": (water_rate_meas, water_rate_sim),
        "gas_rate": (gas_rate_meas, gas_rate_sim),
        "water_cut": (wc_meas, wc_sim),
        "gor": (gor_meas, gor_sim),
        "cum_oil": (cum_oil_meas, cum_oil_sim),
        "pressure": (pressure_meas, pressure_sim),
        "pvt_x": pvt_pressure,
        "bo": (bo_meas, bo_sim),
        "sw": sw,
        "krw": (krw_meas, krw_sim),
        "kro": (kro_meas, kro_sim)
    }

data = generate_data(mismatch_intensity, pressure_drop_rate)
t = data["time"]

fig, axes = plt.subplots(3, 3, figsize=(16, 12), dpi=150)
axes = axes.flatten()

def draw_plot(ax, x, meas, sim, title, ylabel):
    ax.scatter(x, meas, color=meas_color, s=meas_size, alpha=meas_alpha, label="Measured Data")
    ax.plot(x, sim, color=sim_color, linestyle=sim_linestyle, linewidth=sim_linewidth, label="Simulation Model")
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper right", fontsize=8)

draw_plot(axes[0], t, data["oil_rate"][0], data["oil_rate"][1], "Oil Production Rate", "Qo (STB/day)")
draw_plot(axes[1], t, data["water_rate"][0], data["water_rate"][1], "Water Production Rate", "Qw (STB/day)")
draw_plot(axes[2], t, data["gas_rate"][0], data["gas_rate"][1], "Gas Production Rate", "Qg (MSCF/day)")
draw_plot(axes[3], t, data["water_cut"][0], data["water_cut"][1], "Water Cut", "WCT (%)")
draw_plot(axes[4], t, data["gor"][0], data["gor"][1], "Gas-Oil Ratio (GOR)", "GOR (scf/stb)")
draw_plot(axes[5], t, data["cum_oil"][0], data["cum_oil"][1], "Cumulative Oil Production", "Np (MMSTB)")
draw_plot(axes[6], t, data["pressure"][0], data["pressure"][1], "Reservoir Pressure", "Pressure (psi)")

axes[7].scatter(data["pvt_x"], data["bo"][0], color="darkgreen", s=20, label="Bo Meas")
axes[7].plot(data["pvt_x"], data["bo"][1], color="darkgreen", lw=2, label="Bo Model")
axes[7].set_title("PVT: Formation Volume Factor (Bo)", fontsize=11, fontweight='bold')
axes[7].set_ylabel("Bo (bbl/stb)", fontsize=10)
axes[7].grid(True, alpha=0.4)
axes[7].legend(fontsize=8)

sw = data["sw"]
axes[8].plot(sw, data["krw"][1], color="blue", lw=2, label="Krw (Model)")
axes[8].scatter(sw, data["krw"][0], color="blue", s=15, alpha=0.6, label="Krw (Lab)")
axes[8].plot(sw, data["kro"][1], color="brown", lw=2, label="Kro (Model)")
axes[8].scatter(sw, data["kro"][0], color="brown", s=15, alpha=0.6, label="Kro (Lab)")
axes[8].set_title("Relative Permeability (Kr)", fontsize=11, fontweight='bold')
axes[8].set_ylabel("Kr", fontsize=10)
axes[8].set_xlabel("Water Saturation (Sw)", fontsize=10)
axes[8].grid(True, alpha=0.4)
axes[8].legend(fontsize=8, loc="center right")

plt.tight_layout()
st.pyplot(fig)
