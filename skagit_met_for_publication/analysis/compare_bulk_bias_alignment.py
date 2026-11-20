"""Compare step-3 output before and after aligning all products to the PRISM day (issue #61).

Old: analysis/outputs/3_multi_product_bulk_bias_data.csv
     (step 3 run with --no-align: date labels used as-is)
New: analysis/outputs/3_multi_product_bulk_bias_data_prism_day.csv
     (Daymet/gridMET day d relabelled d+1; PNNL, CONUS404, UCLA rebuilt 12 UTC -> 12 UTC
      from hourly output)

Checks that both files hold the same events and that only Daymet/gridMET
columns changed, then reports, per product and metric (7-day total / max),
for all events and split into AR (ar_scale > 0) and non-AR events:
    Pearson r and Spearman rho with PRISM, median bias (product - PRISM),
    median absolute difference, and n. Old and new are computed on the same
    rows (where the product has a value in both versions, and PRISM exists).

Usage:
    python compare_bulk_bias_alignment.py
"""

import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD_CSV = os.path.join(BASE_DIR, "analysis/outputs/3_multi_product_bulk_bias_data.csv")
NEW_CSV = os.path.join(BASE_DIR, "analysis/outputs/3_multi_product_bulk_bias_data_prism_day.csv")

PRODUCTS = ["daymet", "gridmet", "pnnl", "conus", "ucla"]
SHIFTED = {"daymet", "gridmet", "pnnl", "conus", "ucla"}
METRICS = ["7d_tot", "7d_max"]


def stats(x, y):
    """Agreement statistics of product x against PRISM y, on rows where both exist."""
    ok = x.notna() & y.notna()
    x, y = x[ok], y[ok]
    return {
        "n": int(ok.sum()),
        "pearson": x.corr(y),
        "spearman": x.corr(y, method="spearman"),
        "med_bias": (x - y).median(),
        "med_absdiff": (x - y).abs().median(),
    }


def main():
    old = pd.read_csv(OLD_CSV, parse_dates=["date"])
    new = pd.read_csv(NEW_CSV, parse_dates=["date"])

    # 1. Same events (dedup depends only on PRISM, which is unchanged)
    assert old["date"].equals(new["date"]), "event dates differ between old and new CSV"
    print(f"Same {len(old)} event dates in both files.")

    # 2. Only shifted products should change
    print("\nColumns that changed (max |difference|, rows whose NaN status changed):")
    for col in old.columns.drop("date"):
        a, b = old[col], new[col]
        diff = (a - b).abs().max() if pd.api.types.is_numeric_dtype(a) else np.nan
        nan_changed = int((a.isna() != b.isna()).sum())
        changed = (diff > 1e-9) or nan_changed
        if changed or col.split("_")[0] in SHIFTED:
            flag = "" if col.split("_")[0] in SHIFTED else "   <-- UNEXPECTED"
            print(f"  {col:20s} max|diff| {diff:8.3f}   NaN changed {nan_changed}{flag}")

    # 3. Agreement with PRISM, old vs new
    subsets = {"all": slice(None), "AR": old["ar_scale"] > 0, "non-AR": old["ar_scale"] == 0}
    rows = []
    for subset, sel in subsets.items():
        for p in PRODUCTS:
            for m in METRICS:
                # same rows for old and new: where both versions have a value
                common = old[f"{p}_{m}"].notna() & new[f"{p}_{m}"].notna()
                for label, df in (("old", old), ("new", new)):
                    d = df.loc[sel][common[sel]]
                    s = stats(d[f"{p}_{m}"], d[f"prism_{m}"])
                    rows.append({"subset": subset, "product": p, "metric": m, "version": label, **s})
    res = pd.DataFrame(rows)

    wide = res.pivot_table(index=["subset", "metric", "product"], columns="version",
                           values=["pearson", "spearman", "med_bias", "med_absdiff", "n"], sort=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    for subset in subsets:
        for m in METRICS:
            t = wide.loc[(subset, m)]
            out = pd.DataFrame({
                "n": t[("n", "new")].astype(int),
                "r old": t[("pearson", "old")], "r new": t[("pearson", "new")],
                "rho old": t[("spearman", "old")], "rho new": t[("spearman", "new")],
                "bias old": t[("med_bias", "old")], "bias new": t[("med_bias", "new")],
                "|d| old": t[("med_absdiff", "old")], "|d| new": t[("med_absdiff", "new")],
            })
            print(f"\n=== {subset} events, {m} vs PRISM (bias, |d| in mm) ===")
            print(out.round(3).to_string())


if __name__ == "__main__":
    main()
