#!/usr/bin/env python3
"""Extract a real crop-label sample from the ESA WorldCereal point parquet.

Real, satellite-map-derived crop labels (3 classes: Maize, Winter Cereals, Spring
Cereals) for US corn-belt states and African countries. Output: a small CSV sample
committed to the repo, used by the real-label feasibility experiment.

    python3 scripts/make_worldcereal_sample.py
"""
import collections
import csv
import os
import struct

import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PARQ = os.path.join(ROOT, "data", "worldcereal", "worldcereal_points.geoparquet")
OUT = os.path.join(ROOT, "data", "samples", "worldcereal_real_sample.csv")

US_STATES = {"Iowa", "Illinois", "Kansas", "Nebraska", "Minnesota", "Indiana", "Missouri",
             "Ohio", "South Dakota", "North Dakota", "Wisconsin", "Michigan"}
AF_COUNTRIES = {"Ethiopia", "United Republic of Tanzania", "Ghana", "Kenya", "Nigeria",
                "Malawi", "Zambia", "Mozambique", "Uganda", "South Africa", "Burkina Faso",
                "Mali", "Zimbabwe"}
CROPS = ("Maize", "Winter Cereals", "Spring Cereals")
CAP = 450


def lonlat(wkb):
    try:
        return struct.unpack("<2d", wkb[5:21])
    except Exception:
        return (None, None)


def main():
    pf = pq.ParquetFile(PARQ)
    cnt = collections.Counter()
    rows = []
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["Crop", "Country (Admin 0)", "State (Admin 1)",
                                           "Continent", "geometry"]).to_pydict()
        for i in range(len(t["Crop"])):
            crop = t["Crop"][i]
            cont = t["Continent"][i]
            st = t["State (Admin 1)"][i]
            co = t["Country (Admin 0)"][i]
            if crop not in CROPS:
                continue
            if cont == "North America" and st in US_STATES:
                reg = "US"
            elif cont == "Africa" and co in AF_COUNTRIES:
                reg = "Africa"
            else:
                continue
            key = (reg, crop)
            if cnt[key] >= CAP:
                continue
            lon, lat = lonlat(t["geometry"][i])
            if lon is None or not (-180 < lon < 180 and -90 < lat < 90):
                continue
            cnt[key] += 1
            rows.append({"crop": crop, "region": reg, "country": co, "state": st,
                         "lat": round(lat, 4), "lon": round(lon, 4)})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["crop", "region", "country", "state", "lat", "lon"])
        w.writeheader()
        w.writerows(rows)
    print("saved", OUT, len(rows), "rows")
    print("region:", collections.Counter(r["region"] for r in rows))
    print("crop:", collections.Counter(r["crop"] for r in rows))
    print("US states:", collections.Counter(r["state"] for r in rows if r["region"] == "US").most_common(12))
    print("Africa:", collections.Counter(r["country"] for r in rows if r["region"] == "Africa").most_common(12))


if __name__ == "__main__":
    main()
