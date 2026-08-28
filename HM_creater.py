import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. FULLY CUSTOMIZABLE SETTINGS (STYLING)
# ==========================================
STYLE_CONFIG = {
    # Ümumi qrafik parametrləri
    "figsize": (16, 12),
    "dpi": 300,
    "grid": True,
    "grid_alpha": 0.4,
    
    # Measured Data (Faktiki Məlumat - Nöqtələr) styles
    "meas_color": "#1f77b4",       # Məsələn: Tünd göy
    "meas_marker": "o",            # Nöqtə forması ('o', 's', '^', etc.)
    "meas_size": 25,               # Nöqtə ölçüsü
    "meas_alpha": 0.8,             # Şəffaflıq
    "meas_label": "Measured Data",
    
    # Simulation Line (Simulyasiya xətti) styles
    "sim_color": "#d62728",        # Məsələn: Qırmızı
    "sim_linestyle": "-",          # Xətt növü ('-', '--', '-.', ':')
    "sim_linewidth": 2.0,          # Xətt qalınlığı
    "sim_alpha": 0.9,
    "sim_label": "Simulation Model",
    
    # PVT & RelPerm xüsusi rəngləri
    "pvt_oil_col": "darkgreen",
    "pvt_gas_col": "goldenrod",
    "relperm_w_col": "blue",
    "relperm_o_col": "brown"
}

# ==========================================
# 2. SYNTHETIC DATA GENERATOR (MATERIAL BALANCE)
# ==========================================
def generate_synthetic_history_match():
    np.random.seed(42)  # Təkrarlana bilən nəticələr üçün
    
    # Zaman oxu (Məsələn: 60 ay - 5 illik tarixçə)
    time_months = np.arange(1, 61)
    
    # --- Hasilat dərəcələri (Rate) ---
    # Sabit plateau dövrü və sonra tabii decline (azalma)
    base_oil = 15000 * np.exp(-0.02 * time_months)
    oil_rate_sim = base_oil + np.random.normal(0, 300, len(time_months))
    # Measured data-ya bəzi yerlərdə qəsdən fərqlilik (mismatch) qatırıq
    oil_rate_meas = oil_rate_sim + np.random.normal(0, 700, len(time_months))
    # Bəzi nöqtələrdə (məsələn 20-30cu aylarda) uyğunsuzluğu daha da artırırıq
    oil_rate_meas[20:30] += 1500 * np.sin(np.linspace(0, np.pi, 10))

    # Water Rate (Su hasilatı - getdikcə artır)
    water_rate_sim = 2000 + 400 * time_months**0.7 + np.random.normal(0, 150, len(time_months))
    water_rate_meas = water_rate_sim + np.random.normal(0, 400, len(time_months))

    # Gas Rate (Qaz hasilatı - GOR dəyişiminə görə)
    gas_rate_sim = oil_rate_sim * (1.2 + 0.01 * time_months) + np.random.normal(0, 200, len(time_months))
    gas_rate_meas = gas_rate_sim + np.random.normal(0, 800, len(time_months))

    # Water Cut (%)
    total_liq_sim = oil_rate_sim + water_rate_sim
    wc_sim = (water_rate_sim / total_liq_sim) * 100
    wc_meas = wc_sim + np.random.normal(0, 3, len(time_months))
    wc_meas = np.clip(wc_meas, 0, 100) # 0-100% arası məhdudlaşdırma

    # GOR (Gas-Oil Ratio, scf/stb)
    gor_sim = (gas_rate_sim / oil_rate_sim) * 1000
    gor_meas = gor_sim + np.random.normal(0, 150, len(time_months))

    # Kümülatif Neft (Cum Oil, Mstb)
    cum_oil_sim = np.cumsum(oil_rate_sim) / 1000.0
    cum_oil_meas = cum_oil_sim + np.cumsum(np.random.normal(0, 50, len(time_months))) / 1000.0

    # Reservoir Pressure (Material balansına uyğun olaraq oftake-dən asılı düşüş)
    initial_pressure = 4500.0  psi = initial_pressure
    # Kümülatif oftake-ə proporsional təzyiq düşməsi (Material Balance prinsipi)
    pressure_sim = initial_pressure - (cum_oil_sim * 12.5) + np.random.normal(0, 15, len(time_months))
    pressure_meas = pressure_sim + np.random.normal(0, 40, len(time_months))
    # Mismatch nümunəsi: 40-50ci aylarda təzyiq uyğunsuzluğu
    pressure_meas[40:50] -= 120  

    # --- PVT Data ---
    pvt_pressure = np.linspace(1000, 5000, 50)
    # Bo (Formation Volume Factor)
    bo_sim = 1.05 + 0.00003 * pvt_pressure - 0.000000002 * pvt_pressure**2
    bo_meas = bo_sim + np.random.normal(0, 0.01, len(pvt_pressure))
    # Rs (Solution Gas Ratio)
    rs_sim = 0.1 * pvt_pressure**1.2
    rs_meas = rs_sim + np.random.normal(0, 20, len(pvt_pressure))

    # --- Relative Permeability Data ---
    sw = np.linspace(0.2, 0.8, 40) # Water saturation
    krw_sim = 0.3 * ((sw - 0.2) / 0.6)**2.5
    kro_sim = 0.8 * ((0.8 - sw) / 0.6)**2.0
    krw_meas = krw_sim + np.random.normal(0, 0.02, len(sw))
    kro_meas = kro_sim + np.random.normal(0, 0.02, len(sw))
    krw_meas = np.clip(krw_meas, 0, 1)
    kro_meas = np.clip(kro_meas, 0, 1)

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
        "rs": (rs_meas, rs_sim),
        "sw": sw,
        "krw": (krw_meas, krw_sim),
        "kro": (kro_meas, kro_sim)
    }

# ==========================================
# 3. PLOTTING ENGINE (FULLY CUSTOMIZABLE)
# ==========================================
def plot_history_match(data, cfg):
    fig, axes = plt.subplots(3, 3, figsize=cfg["figsize"], dpi=cfg["dpi"])
    axes = axes.flatten()

    def plot_pair(ax, x, meas, sim, title, ylabel, is_pvt_or_relperm=False):
        if not is_pvt_or_relperm:
            # Measured Data - Nöqtələr (Scatter)
            ax.scatter(x, meas, color=cfg["meas_color"], marker=cfg["meas_marker"], 
                       s=cfg["meas_size"], alpha=cfg["meas_alpha"], label=cfg["meas_label"])
            # Simulation - Xətt (Line)
            ax.plot(x, sim, color=cfg["sim_color"], linestyle=cfg["sim_linestyle"], 
                    linewidth=cfg["sim_linewidth"], alpha=cfg["sim_alpha"], label=cfg["sim_label"])
        else:
            # PVT və ya RelPerm xüsusi halları üçün
            ax.scatter(x, meas, color=cfg["meas_color"], s=cfg["meas_size"], alpha=cfg["meas_alpha"], label="Lab Data")
            ax.plot(x, sim, color=cfg["sim_color"], linewidth=cfg["sim_linewidth"], label="Model Fit")

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(cfg["grid"], alpha=cfg["grid_alpha"])
        ax.legend(loc="upper right", fontsize=8)

    t = data["time"]

    # 1. Oil Rate
    plot_pair(axes[0], t, data["oil_rate"][0], data["oil_rate"][1], "Oil Production Rate", "Qo (STB/day)")

    # 2. Water Rate
    plot_pair(axes[1], t, data["water_rate"][0], data["water_rate"][1], "Water Production Rate", "Qw (STB/day)")

    # 3. Gas Rate
    plot_pair(axes[2], t, data["gas_rate"][0], data["gas_rate"][1], "Gas Production Rate", "Qg (MSCF/day)")

    # 4. Water Cut
    plot_pair(axes[3], t, data["water_cut"][0], data["water_cut"][1], "Water Cut", "WCT (%)")

    # 5. GOR
    plot_pair(axes[4], t, data["gor"][0], data["gor"][1], "Gas-Oil Ratio (GOR)", "GOR (scf/stb)")

    # 6. Cumulative Oil
    plot_pair(axes[5], t, data["cum_oil"][0], data["cum_oil"][1], "Cumulative Oil Production", "Np (MMSTB)")

    # 7. Reservoir Pressure (Material balance oftake uyğun)
    plot_pair(axes[6], t, data["pressure"][0], data["pressure"][1], "Reservoir Pressure (Material Bal.)", "Pressure (psi)")

    # 8. PVT (Bo / Rs)
    pvt_x = data["pvt_x"]
    axes[7].scatter(pvt_x, data["bo"][0], color=cfg["pvt_oil_col"], s=20, label="Bo Meas")
    axes[7].plot(pvt_x, data["bo"][1], color=cfg["pvt_oil_col"], lw=2, label="Bo Model")
    axes[7].set_title("PVT: Formation Volume Factor (Bo)", fontsize=11, fontweight='bold')
    axes[7].set_ylabel("Bo (bbl/stb)", fontsize=10)
    axes[7].grid(cfg["grid"], alpha=cfg["grid_alpha"])
    axes[7].legend(fontsize=8)

    # 9. Relative Permeability
    sw = data["sw"]
    axes[8].plot(sw, data["krw"][1], color=cfg["relperm_w_col"], lw=2, label="Krw (Model)")
    axes[8].scatter(sw, data["krw"][0], color=cfg["relperm_w_col"], s=15, alpha=0.6, label="Krw (Lab)")
    axes[8].plot(sw, data["kro"][1], color=cfg["relperm_o_col"], lw=2, label="Kro (Model)")
    axes[8].scatter(sw, data["kro"][0], color=cfg["relperm_o_col"], s=15, alpha=0.6, label="Kro (Lab)")
    axes[8].set_title("Relative Permeability (Kr)", fontsize=11, fontweight='bold')
    axes[8].set_ylabel("Kr", fontsize=10)
    axes[8].set_xlabel("Water Saturation (Sw)", fontsize=10)
    axes[8].grid(cfg["grid"], alpha=cfg["grid_alpha"])
    axes[8].legend(fontsize=8, loc="center right")

    plt.suptitle("Real Field Synthetic History Match & Subsurface QC Dashboard", fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.show()

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    print("Sintetik məlumatlar material balansına uyğun olaraq yaradılır...")
    field_data = generate_synthetic_history_match()
    
    print("Qrafiklər fərdiləşdirilmiş parametrlərlə qurulur...")
    plot_history_match(field_data, STYLE_CONFIG)
