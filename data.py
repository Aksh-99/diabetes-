"""
NHANES Universal Downloader
----------------------------
Instead of hardcoding guessed URLs (which break every cycle, since each
cycle uses a different folder and filename convention), this script reads
the REAL file listing directly off the CDC's own data page for whichever
Component + Cycle you ask for, then downloads exactly those files.

Usage:
    pip install pandas requests beautifulsoup4 lxml

    python nhanes_downloader.py --component Laboratory --cycle 2021-2023
    python nhanes_downloader.py --component Demographics --cycle 2021-2023
    python nhanes_downloader.py --component Questionnaire --cycle 2021-2023 --filter DIQ
    python nhanes_downloader.py --component Examination --cycle 2021-2023 --filter BMX

--filter is optional: only download files whose name contains that substring
(case-insensitive). Without it, ALL files for that component+cycle download,
which for Dietary/Questionnaire can be dozens of files.

Output:
    nhanes_raw/<CYCLE>/<COMPONENT>/*.xpt   (raw downloaded files)
"""

import argparse
import os
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup

BASE = "https://wwwn.cdc.gov/Nchs/Nhanes/search/datapage.aspx"


def get_file_table(component: str, cycle: str):
    """Scrape the CDC data page and return list of (name, xpt_url, size_str)."""
    url = f"{BASE}?Component={component}&Cycle={cycle}"
    print(f"[fetch] {url}")
    resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")
    rows = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".xpt"):
            full_url = href if href.startswith("http") else f"https://wwwn.cdc.gov{href}"
            # e.g. "GHB_L Data [XPT - 169.9 KB]"
            text = a.get_text(strip=True)
            m = re.match(r"([A-Za-z0-9_]+)\s*Data", text)
            name = m.group(1) if m else os.path.basename(href).replace(".xpt", "").replace(".XPT", "")
            rows.append((name, full_url, text))

    if not rows:
        raise RuntimeError(
            f"No .xpt files found on {url}. The page structure may have changed, "
            f"or Component/Cycle values may be wrong. Check the URL in a browser."
        )
    return rows


def download_file(name, url, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.xpt")
    if os.path.exists(path) and os.path.getsize(path) > 5000:
        print(f"  [skip] {name} already downloaded ({os.path.getsize(path)} bytes)")
        return path

    r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    if r.status_code != 200 or len(r.content) < 5000:
        print(f"  [warn] {name} failed or too small ({r.status_code}, {len(r.content)} bytes) <- {url}")
        return None

    with open(path, "wb") as f:
        f.write(r.content)
    print(f"  [saved] {name} ({len(r.content)/1024:.1f} KB)")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--component", required=True,
                     choices=["Demographics", "Dietary", "Examination", "Laboratory", "Questionnaire", "LimitedAccess"])
    ap.add_argument("--cycle", required=True, help="e.g. 2021-2023, 2017-2020, 2017-2018")
    ap.add_argument("--filter", default=None, help="Only download files whose name contains this substring")
    ap.add_argument("--load", action="store_true", help="Also load each downloaded file into a pandas DataFrame and print shape")
    args = ap.parse_args()

    out_dir = os.path.join("nhanes_raw", args.cycle, args.component)
    rows = get_file_table(args.component, args.cycle)

    if args.filter:
        rows = [r for r in rows if args.filter.lower() in r[0].lower()]
        if not rows:
            print(f"No files matched filter '{args.filter}'. Available names:")
            for name, _, _ in get_file_table(args.component, args.cycle):
                print(f"  - {name}")
            return

    print(f"\nFound {len(rows)} matching file(s):")
    for name, url, text in rows:
        print(f"  - {name}: {text}")
    print()

    downloaded = []
    for name, url, _ in rows:
        path = download_file(name, url, out_dir)
        if path:
            downloaded.append((name, path))

    if args.load:
        print("\nLoading into pandas...")
        for name, path in downloaded:
            try:
                df = pd.read_sas(path, format="xport")
                print(f"  {name}: {df.shape[0]} rows, {df.shape[1]} cols -> {list(df.columns)[:8]}...")
            except Exception as e:
                print(f"  [error] Could not load {name}: {e}")

    print(f"\nDone. Files saved in: {out_dir}/")


if __name__ == "__main__":
    main()