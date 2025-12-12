#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from datetime import datetime as dt, timedelta as td

import obspy

def read_miniseed_archive_merge_only(
    network: str,
    station: str,
    channels: str,
    starttime: dt,
    endtime: dt,
    archive: Path | str,
    archive_fmt: str,
    merge_method: int = -1,      # 0 = no interpolation, 1 = interpolate small gaps
    verbose: bool = True,
) -> obspy.Stream:
    """
    Read miniSEED files from an archive layout like:

        {archive}/{station}/
            {network}.{station}..{channels}.{year}.{jday:03d}.mseed

    Then merge + trim, with no filtering or computation.
    """
    archive = Path(archive)
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
            if verbose:
                print("Reading:", fpath)
            st += obspy.read(str(fpath))
        else:
            if verbose:
                print("Missing:", fpath)

        read_from += td(days=1)

    if len(st) == 0:
        if verbose:
            print(f"No data read for station={station}, channels={channels}")
        return st

    # Merge (no detrend/filter). This may produce masked arrays if there are gaps.
    st.merge(method=merge_method)

    # Final trim to the exact requested window
    st.trim(
        starttime=obspy.UTCDateTime(starttime),
        endtime=obspy.UTCDateTime(endtime),
        nearest_sample=True,
    )

    return st


def main():
    # ---- CHANGE THESE FOR MAGNETIC ----
    magnetic_archive = Path("Magnetic/OV")

    # Example magnetic file name you likely have:
    # Magnetic/OV/VPCC/OV.VPCC..LFZ.D.2025.315.mseed
    magnetic_archive_format = (
        "{station}/"
        "{network}.{station}..{channels}.{year}.{jday:03d}.mseed"
    )

    network = "OV"
    stations = ["VPCC", "VPPC", "VPRS", "VPNC"]
    channels_list = ["LFE.D", "LFN.D", "LFZ.D"]

    # FULL 2-year span
    starttime = dt(2024, 1, 1, 0, 0, 0)
    endtime   = dt(2025, 12, 31, 23, 59, 59)

    out_dir = Path("processed_magnetic_merged")
    out_dir.mkdir(parents=True, exist_ok=True)

    for station in stations:
        for channels in channels_list:
            print(f"\n=== MERGE {station} {channels} {starttime.date()}–{endtime.date()} ===")

            st = read_miniseed_archive_merge_only(
                network=network,
                station=station,
                channels=channels,
                starttime=starttime,
                endtime=endtime,
                archive=magnetic_archive,
                archive_fmt=magnetic_archive_format,
                merge_method=-1, # 0 = keep gaps honest; 1 = interpolate small gaps
                verbose=True,
            )

            if len(st) == 0:
                print("No data, skipping.")
                continue

            out_mseed = out_dir / f"{station}_{channels.replace('.', '_')}_merged.mseed"
            st.write(str(out_mseed), format="MSEED")
            print("Wrote:", out_mseed)

            del st


if __name__ == "__main__":
    main()
