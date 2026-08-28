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
    """
    Internal model remains metric.
    This function only converts values for plotting.
    """

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

    # Field units
    M3_TO_STB = 6.28981
    M3M3_TO_SCFSTB = 5.615
    BAR_TO_PSI = 14.5038

    # gas_measured is currently 10³ m³/d
    # 1 thousand m³/d = 0.0353147 MMscf/d
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
    start_date = pd.Timestamp(f"{plot_start_year}-01-01")
    end_date = start_date + pd.DateOffset(years=int(plot_years))

    mask = (dates >= start_date) & (dates < end_date)

    if mask.sum() == 0:
        mask = np.ones(len(dates), dtype=bool)

    dates_filtered = dates[mask]

    if x_axis_mode == "Years":
        x_values = (dates_filtered - dates_filtered[0]).days / 365.25
        x_title = "YEARS"
    else:
        x_values = dates_filtered
        x_title = "DATE"

    return mask, x_values, x_title
