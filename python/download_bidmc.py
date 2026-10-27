"""
download_bidmc.py - download the BIDMC PPG and Respiration Dataset (CSV version) from PhysioNet.

Only the files this project needs are downloaded (Signals + Numerics, ~53 x 2 files).
Dataset page: https://physionet.org/content/bidmc/1.0.0/   (open access, ODC-By licence)

Usage:
    python download_bidmc.py                 # all 53 recordings
    python download_bidmc.py --n 5           # just the first 5 (quick test)
"""
import argparse
import os
import urllib.request

BASE = "https://physionet.org/files/bidmc/1.0.0/bidmc_csv/"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=53, help="Number of recordings to download (1-53)")
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "bidmc_csv"))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for i in range(1, args.n + 1):
        for kind in ("Signals", "Numerics"):
            name = f"bidmc_{i:02d}_{kind}.csv"
            dest = os.path.join(args.out, name)
            if os.path.exists(dest):
                continue
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(BASE + name, dest)
    print(f"Done. Files are in {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
