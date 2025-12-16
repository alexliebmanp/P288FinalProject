# %% 
#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Created on December 10, 2025 // @author: Sarah Shi """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
plt.rcParams.update({
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'axes.titlesize': 22,
    'axes.labelsize': 22,
    'pdf.fonttype': 42,
    'font.family': 'Avenir',
    'font.size': 12,
    'xtick.direction': 'in',  # Set x-tick direction to 'in'
    'ytick.direction': 'in',  # Set y-tick direction to 'in'
    'xtick.major.size': 5,    # Set x-tick length
    'ytick.major.size': 5,    # Set y-tick length
    'xtick.major.pad': 6.5,   # Set x-tick padding
    'ytick.major.pad': 6.5    # Set y-tick padding
})


# %%

interpolated = pd.read_csv(
    "processed_data/dataInterpolated.csv",
    index_col=0,
    parse_dates=True
)

# %% 

def plot_multimodal_chunk(df, station, event_col, start=None, end=None):
    from matplotlib.ticker import FixedLocator
    import matplotlib.dates as mdates
    import pandas as pd
    import matplotlib.pyplot as plt

    df = df.copy()
    df.index = pd.to_datetime(df.index)

    if start is not None and end is not None:
        chunk = df.loc[start:end]
    else:
        chunk = df

    fig, ax1 = plt.subplots(figsize=(12, 8))

    # CO2
    ax1.plot(chunk.index, chunk["CO2_ppm"],
             color="#5AADC7", linewidth=1.0, alpha=1, zorder=10)
    ax1.set_ylabel(r"CO$_2$ [ppm]", color="#5AADC7")
    ax1.tick_params(axis="y", labelcolor="#5AADC7")

    # RSAM
    ax2 = ax1.twinx()
    ax2.semilogy(chunk.index, chunk[station],
                 color="black", linewidth=1.0, alpha=0.8, zorder=30)
    ax2.set_ylabel("log(seismic/infrasound amplitude)", color="black")
    ax2.tick_params(axis="y", labelcolor="black")

    # Events
    event_times = chunk.index[chunk[event_col] > 0]
    ax1.set_xlim(chunk.index.min(), chunk.index.max())
    ymin, ymax = ax1.get_ylim()

    if len(event_times) > 0:
        ax1.vlines(
            event_times,
            ymin=ymin,
            ymax=ymax,
            colors="purple",
            alpha=0.1,
            linewidth=1,
            zorder=-30
        )

    # Ticks
    start_m = chunk.index.min().to_period("M").to_timestamp()
    end_m   = chunk.index.max().to_period("M").to_timestamp()
    month_ticks = pd.date_range(start_m, end_m, freq="MS")

    ax1.xaxis.set_major_locator(FixedLocator(mdates.date2num(month_ticks)))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

    ax1.set_title(f"CO2 and {station} Infrasound/Seismic Amplitude")
    plt.tight_layout()
    plt.show()

# %%

plot_multimodal_chunk(interpolated, "VPPC_RSAM", "Eruption_Activity",
                start="2024-01-15", end="2025-12-15")

# %% 

plot_multimodal_chunk(interpolated, "VPNC_RSAM", "Eruption_Activity",
                start="2024-01-15", end="2025-12-15")

# %% 

plot_multimodal_chunk(interpolated, "VPCC_RSAM", "Eruption_Activity",
                start="2024-01-15", end="2025-12-15")

# %% 

plot_multimodal_chunk(interpolated, "VPRS_RSAM", "Eruption_Activity",
                start="2024-01-15", end="2025-12-15")

# %%
