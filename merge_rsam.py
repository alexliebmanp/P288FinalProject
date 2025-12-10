import glob
import pandas as pd
from pathlib import Path

def merge_monthly_rsam_for_station(station: str, outdir: str = "."):
    """
    Merge all monthly RSAM CSVs for one station into a single file.

    Expects files named like:
        {station}_RSAM_600s_YYYY_MM.csv
    """
    pattern = f"{station}_RSAM_600s_*.csv"
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No monthly files found for station {station} with pattern {pattern}")
        return

    print(f"Merging {len(files)} files for station {station}:")
    for f in files:
        print("   ", f)

    dfs = []
    for f in files:
        df = pd.read_csv(f, parse_dates=["timestamp"])
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # optional but sensible: sort by timestamp
    df_all = df_all.sort_values("timestamp").reset_index(drop=True)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{station}_RSAM_600s_merged_m-1.csv"

    df_all.to_csv(outpath, index=False)
    print(f"Saved merged CSV for {station} to {outpath}")


if __name__ == "__main__":
    stations = ["VPCC", "VPPC", "VPRS", "VPNC"]  # or whatever subset you used
    for sta in stations:
        merge_monthly_rsam_for_station(sta, outdir="new_seismic")
