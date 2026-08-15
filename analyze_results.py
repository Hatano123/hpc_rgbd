#!/usr/bin/env python3
import csv
import sys
from pathlib import Path

def best_map(csv_path):
    best = -1.0
    if not Path(csv_path).exists():
        return None
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            # try both possible column names
            for key in ("metrics/mAP50(B)", "metrics/mAP50", "metrics/mAP50-95(B)"):
                if key in r and r[key] != "":
                    try:
                        val = float(r[key])
                        if val > best:
                            best = val
                    except Exception:
                        pass
    return best if best >= 0 else None

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: analyze_results.py <results.csv> [<results2.csv> ...]")
        sys.exit(1)
    for p in sys.argv[1:]:
        b = best_map(p)
        print(f"{p}: best mAP50 = {b}")
