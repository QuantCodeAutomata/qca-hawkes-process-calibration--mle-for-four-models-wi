# data/download_earthquake.py
"""
Download earthquake catalog for the Wenchuan region (2008-2023).

Primary source: ISC FDSN API — preferred magnitude type MW, Mw >= 5.0.
The paper states n=37 events (ISC catalog, Mw>=5.0, dist<=300 km from
Chengdu). The current ISC catalog (as of 2025) returns n=44 events under
the same criteria due to post-publication catalog updates. This module
downloads and saves all available events and documents the discrepancy.

Filters applied (exactly as described in paper):
  1. Haversine distance <= 300 km from Chengdu (30.5728N, 104.0668E)
  2. Preferred magnitude type = MW, value >= 5.0
  3. Temporal window: 2008-05-12 to 2023-12-31
Time origin: mainshock event time (t_1 = 0 by definition).
"""
import math
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

CHENGDU_LAT: float = 30.5728
CHENGDU_LON: float = 104.0668
MAX_DIST_KM: float = 300.0
MIN_MAG: float = 5.0
DATE_START: str = "2008-05-12"
DATE_END: str = "2023-12-31"

_DEFAULT_CSV = Path(__file__).parent / "earthquake_events.csv"


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in km (WGS-84 spherical approx, R=6371 km)."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return 2.0 * R * math.asin(math.sqrt(a))


def download_isc_catalog(
    lat_min: float = 28.0,
    lat_max: float = 33.0,
    lon_min: float = 101.0,
    lon_max: float = 108.0,
    min_mag: float = MIN_MAG,
    start_date: str = DATE_START,
    end_date: str = DATE_END,
) -> pd.DataFrame:
    """
    Download ISC FDSN catalog, filter by Haversine<=300 km and MW>=5.0.

    The ISC FDSN API returns events with their preferred magnitude.  We
    filter post-download to keep only MagType == 'MW'.

    Parameters
    ----------
    lat_min, lat_max, lon_min, lon_max : rectangular pre-filter bounding box
    min_mag : minimum preferred magnitude
    start_date, end_date : ISO date strings

    Returns
    -------
    pd.DataFrame with columns DATETIME, LAT, LON, MAG, DIST_KM, T_YEARS
    """
    base_url = "https://www.isc.ac.uk/fdsnws/event/1/query"
    # Download in 3 time chunks to avoid server timeouts
    time_chunks = [
        (f"{start_date}T00:00:00", "2012-12-31T23:59:59"),
        ("2013-01-01T00:00:00", "2018-12-31T23:59:59"),
        ("2019-01-01T00:00:00", f"{end_date}T23:59:59"),
    ]

    all_rows = []
    header_cols: Optional[list] = None

    for t_start, t_end in time_chunks:
        url = (
            f"{base_url}?"
            f"minlatitude={lat_min}&maxlatitude={lat_max}"
            f"&minlongitude={lon_min}&maxlongitude={lon_max}"
            f"&starttime={t_start}&endtime={t_end}"
            f"&minmagnitude={min_mag}&format=text"
        )
        try:
            r = requests.get(url, timeout=90)
            if r.status_code == 200 and len(r.text) > 50:
                lines = r.text.strip().split("\n")
                if header_cols is None:
                    header_cols = lines[0].strip("#").split("|")
                all_rows.extend(lines[1:])
        except Exception as exc:
            print(f"  ISC download warning ({t_start[:10]}): {exc}")

    if not all_rows or header_cols is None:
        raise RuntimeError("ISC FDSN download returned no data.")

    data = [row.split("|") for row in all_rows if row.strip()]
    df = pd.DataFrame(data, columns=header_cols)
    df["Magnitude"] = pd.to_numeric(df["Magnitude"], errors="coerce")
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["Time"] = pd.to_datetime(df["Time"], utc=True, errors="coerce")

    # Keep only MW-type events >= min_mag
    mw_mask = (df["MagType"].str.strip() == "MW") & (df["Magnitude"] >= min_mag)
    df = df[mw_mask].dropna(subset=["Time", "Latitude", "Longitude", "Magnitude"])

    # Haversine filter
    df = df.copy()
    df["DIST_KM"] = df.apply(
        lambda r: haversine(r["Latitude"], r["Longitude"], CHENGDU_LAT, CHENGDU_LON),
        axis=1,
    )
    df = df[df["DIST_KM"] <= MAX_DIST_KM].sort_values("Time").reset_index(drop=True)

    if df.empty:
        raise RuntimeError("No events survived Haversine + MW filter.")

    # Time conversion: t_i = seconds from mainshock / (365.25 * 86400)
    t0 = df.iloc[0]["Time"]
    df["T_YEARS"] = (df["Time"] - t0).dt.total_seconds() / (365.25 * 86400)
    df["DATETIME"] = df["Time"]
    df["LAT"] = df["Latitude"]
    df["LON"] = df["Longitude"]
    df["MAG"] = df["Magnitude"]

    return df[["DATETIME", "LAT", "LON", "MAG", "DIST_KM", "T_YEARS"]].copy()


def load_or_download(csv_path: Path = _DEFAULT_CSV, force_download: bool = False) -> pd.DataFrame:
    """
    Load from CSV if it exists; otherwise download from ISC and save.

    Parameters
    ----------
    csv_path     : path to cached CSV file
    force_download : if True, always re-download

    Returns
    -------
    pd.DataFrame with DATETIME, LAT, LON, MAG, DIST_KM, T_YEARS columns.
    """
    if csv_path.exists() and not force_download:
        df = pd.read_csv(csv_path, parse_dates=["DATETIME"])
        print(f"Loaded cached catalog: {csv_path} (n={len(df)} events)")
        return df

    print("Downloading ISC catalog (MW>=5.0, dist<=300 km from Chengdu)...")
    df = download_isc_catalog()
    n = len(df)
    t_max = df["T_YEARS"].max()
    print(f"  Downloaded n={n} events (paper: n=37; ISC may have been updated)")
    print(f"  T_last = {t_max:.4f} yr (≈ {t_max:.1f} years after mainshock)")
    df.to_csv(csv_path, index=False)
    print(f"  Saved to {csv_path}")
    return df


if __name__ == "__main__":
    df = load_or_download(force_download=True)
    print(df[["DATETIME", "MAG", "DIST_KM", "T_YEARS"]].to_string())
    print(f"\nn = {len(df)} events")
    print(f"T_cal_last = {df['T_YEARS'].max():.4f} yr")
