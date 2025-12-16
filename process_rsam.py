# %% 
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, pathlib
import calendar
from datetime import datetime as dt, timedelta as td

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import obspy
import pandas as pd
from obspy.signal.filter import envelope


def compute_rsam(
    st: obspy.Stream,
    starttime: dt,
    endtime: dt,
    window_length: int,
    overlap: int = 0,
) -> list[float]:
    """
    Compute the Real-time Seismic Amplitude Measurement for a given stream of seismic
    data.

    Parameters
    ----------
    st:
        Vertical component data stream for the selected station/network.
    starttime:
        Timestamp from which to compute the RSAM.
    endtime:
        Timestamp up to which to compute the RSAM.
    window_length:
        Timespan of measurement windows, in seconds.
    overlap:
        Define the number of seconds of overlap between consecutive measurement windows.

    Returns
    -------
     :
        List of computed RSAM values.

    """

    compute_from = starttime

    rsams = []
    while compute_from < endtime:
        st_window = st.slice(
            starttime=obspy.UTCDateTime(compute_from),
            endtime=obspy.UTCDateTime(compute_from) + window_length,
        )

        if (
            not st_window  # No data for this window
            or (
                len(st_window[0].data)
                != st_window[0].stats.sampling_rate * window_length + 1
            )  # Gaps during this window
            or any(tr.data.max() == tr.data.min() for tr in st_window)  # All data bad
        ):
            rsam_value = np.nan
        else:
            rsam_value = np.nanmean(envelope(st_window[0].data))

        rsams.append(rsam_value)
        compute_from += td(seconds=window_length - overlap)

    return rsams


def read_miniseed_archive(
    network: str,
    station: str,
    channels: str,
    starttime: dt,
    endtime: dt,
    archive: pathlib.Path | str,
    archive_fmt: str,
) -> obspy.Stream:
    """
    Read data from a waveform archive laid out like:

        {archive}/{station}/
            {network}.{station}..{channels}.D.{year}.{jday:03d}.mseed
    """

    archive = pathlib.Path(archive)
    st = obspy.Stream()

    read_from = starttime
    while read_from.date() <= endtime.date():
        rel_path = archive_fmt.format(
            network=network,
            station=station,
            channels=channels,
            year=read_from.year,
            jday=read_from.timetuple().tm_yday,
        )
        fpath = archive / rel_path

        if fpath.exists():
            print("Reading:", fpath)
            st += obspy.read(str(fpath))
        else:
            # optional: be noisy so you see missing days
            print("Missing:", fpath)

        read_from += td(days=1)

    if len(st) == 0:
        print(f"No data read for station {station}")
        return st

    # --- resample everything before merging ---
    target_fs = 100.0  # RSAM paper uses 100 Hz; adjust if needed

    for tr in st:
        # Resample to consistent sampling rate
        tr.resample(target_fs)
        tr.trim(tr.stats.starttime, tr.stats.endtime, nearest_sample=True)

    # --- merge after resampling ---
    st.merge(method=-1) #, fill_value=1e-4)

    # --- final trim to requested start/end ---
    st.trim(
        starttime=obspy.UTCDateTime(starttime),
        endtime=obspy.UTCDateTime(endtime),
    )

    return st

# %% 

def main():
    seismic_archive = "Seismic/OV"
    seismic_archive_format = (
        "{station}/"
        "{network}.{station}..{channels}.D.{year}.{jday:03d}.mseed"
    )

    stations = ["VPCC"]  # ["VPCC", "VPPC", "VPRS", "VPNC"]
    channels = "HHZ"

    start_year = 2024
    end_year   = 2025

    window_len = 600.0  # 600-s RSAM

    for station in stations:
        print(f"\n=== RSAM for {station} ===")

        for year in range(start_year, end_year + 1):
            print(f"\n--- Year {year} ---")

            for month in range(1, 13):
                # last day of this month (handles leap years correctly)
                last_day = calendar.monthrange(year, month)[1]
                starttime = dt(year, month, 1, 0, 0, 0)
                endtime   = dt(year, month, last_day, 23, 59, 59)

                print(f"\nMonth {year}-{month:02d}: {starttime} -> {endtime}")

                st = read_miniseed_archive(
                    network="OV",
                    station=station,
                    channels=channels,
                    starttime=starttime,
                    endtime=endtime,
                    archive=seismic_archive,
                    archive_fmt=seismic_archive_format,
                )

                print(st)
                print("Number of traces:", len(st))

                if len(st) == 0:
                    print(f"No data for {station} in {year}-{month:02d}, skipping.")
                    continue

                st.detrend("linear")
                st.detrend("demean")
                st = st.select(component="Z")

                st_filt = st.copy().filter("bandpass", freqmin=1.0, freqmax=10.0)
                st_filt.trim(obspy.UTCDateTime(starttime), obspy.UTCDateTime(endtime))

                rsam_vals = compute_rsam(
                    st_filt,
                    starttime,
                    endtime,
                    window_length=window_len,
                )
                times = [
                    starttime + td(seconds=window_len * i)
                    for i in range(len(rsam_vals))
                ]

                df_rsam = pd.DataFrame(
                    {
                        "timestamp": times,
                        "rsam": rsam_vals,
                        "station": station,
                        "channel": channels,
                        "year": year,
                        "month": month,
                    }
                )

                out_csv = f"{station}_RSAM_600s_{year}_{month:02d}.csv"
                df_rsam.to_csv(out_csv, index=False)
                print(f"Saved {len(df_rsam)} rows to {out_csv}")

                # Optional: quick sanity plot per month
                # plt.figure(figsize=(8, 4))
                # plt.semilogy(df_rsam["timestamp"], df_rsam["rsam"])
                # plt.title(f"{station} RSAM 600 s ({year}-{month:02d})")
                # plt.tight_layout()
                # plt.show()
                # plt.close()

                # dump big objects so they can be freed
                del st, st_filt, df_rsam, rsam_vals, times

    return


if __name__ == "__main__":
    main()


# %% 

stations = ["VPCC", "VPPC", "VPRS", "VPNC"]

for station in stations: 
    df_rsam = pd.read_csv(f"new_seismic/{station}_RSAM_600s_merged_m-1.csv")
    df_rsam["timestamp"] = pd.to_datetime(df_rsam["timestamp"])

    plt.figure(figsize=(10,5))
    plt.semilogy(df_rsam.timestamp, df_rsam.rsam, linewidth=0.5)

    # Reduce tick frequency
    import matplotlib.dates as mdates
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=14))  # show every 14 days
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.title(station)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# %% 

start_date = pd.to_datetime("2025-02-05")
end_date   = pd.to_datetime("2025-03-05")

for station in stations: 
    df_rsam = pd.read_csv(f"Seismic/{station}_RSAM_600s_merged.csv")
    df_rsam["timestamp"] = pd.to_datetime(df_rsam["timestamp"])

    # --- filter to desired range ---
    mask = (df_rsam["timestamp"] >= start_date) & (df_rsam["timestamp"] <= end_date)
    df_plot = df_rsam.loc[mask]

    plt.figure(figsize=(10,5))
    plt.semilogy(df_plot.timestamp, df_plot.rsam, linewidth=0.5)

    # Tick formatting
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    plt.title(f"{station} RSAM")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# %%
