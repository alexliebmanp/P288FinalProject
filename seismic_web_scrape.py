#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from datetime import date, timedelta

import requests

BASE_URL = "https://avert-legacy.ldeo.columbia.edu/archive/miniseed"
NETWORK  = "OV"
STATIONS = ["VPPC", "VPNC", "VPCC", "VPRS"]
CHANNEL_DIR = "HHZ.D"   # directory + chunk in filename
LOCATION = ""           # empty → two dots in filename

# local root where you want the data
OUT_ROOT = Path("Seismic/OV")

# date range of interest
START_DATE = date(2024, 1, 1)
END_DATE   = date(2025, 12, 31) # inclusive

def build_url(year: int, jday: int, station: str) -> str:
    """
    Build full HTTP URL like:
    .../2024/OV/VPPC/HHZ.D/OV.VPPC..HHZ.D.2024.001
    """
    fname = f"{NETWORK}.{station}..{CHANNEL_DIR}.{year}.{jday:03d}"
    return f"{BASE_URL}/{year}/{NETWORK}/{station}/{CHANNEL_DIR}/{fname}"


def build_local_path(year: int, jday: int, station: str) -> Path:
    """
    Local path:
        Seismic/OV/<STATION>/OV.<STATION>..HHZ.D.<year>.<jday>.mseed
    """
    fname = f"{NETWORK}.{station}..{CHANNEL_DIR}.{year}.{jday:03d}.mseed"
    out_dir = OUT_ROOT / station
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / fname


def download_one(url: str, out_path: Path) -> bool:
    """
    Download a single file if present. Returns True if downloaded / exists,
    False if not found.
    """
    # Skip if already present
    if out_path.exists():
        print(f"[SKIP] {out_path} already exists")
        return True

    print(f"[GET] {url}")
    resp = requests.get(url, stream=True, timeout=30)

    if resp.status_code == 200:
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        size_mb = out_path.stat().st_size / 1024**2
        print(f"  -> saved {out_path} ({size_mb:.2f} MB)")
        return True
    else:
        print(f"  -> HTTP {resp.status_code}, no file")
        return False


def main():
    d = START_DATE
    n_total = 0
    n_ok = 0

    while d <= END_DATE:
        year = d.year
        jday = d.timetuple().tm_yday

        for sta in STATIONS:
            url = build_url(year, jday, sta)
            out_path = build_local_path(year, jday, sta)
            n_total += 1
            ok = download_one(url, out_path)
            if ok:
                n_ok += 1

        d += timedelta(days=1)

    print(f"\nDone. Tried {n_total} files, successfully present/downloaded {n_ok}.")


if __name__ == "__main__":
    main()
