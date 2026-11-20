import glob
import pandas as pd

frames = []
for f in sorted(glob.glob("plots/plot_specific_ar_and_non_ar_events/timeseries_data/cumulative_precip_*.csv")):
    cum = pd.read_csv(f, parse_dates=["datetime"]).set_index("datetime")
    daily = cum.diff()
    daily.iloc[0] = cum.iloc[0]
    frames.append(daily)
daily = pd.concat(frames)
daily = daily[~daily.index.duplicated()].sort_index().asfreq("D")

print("Correlation of each product's daily rain with PRISM:")
for p in ["daymet", "gridmet", "conus", "ucla", "pnnl"]:
    same = daily[p].corr(daily["prism"])
    next_day = daily[p].corr(daily["prism"].shift(-1))
    print(f"  {p:8s} same date {same:.2f}   PRISM's next day {next_day:.2f}")

week = daily.rolling(7, min_periods=7).sum()
print("Median |7-day total - PRISM| (mm):")
for p in ["daymet", "gridmet"]:
    as_coded = (week[p] - week["prism"]).abs()
    shifted = (week[p] - week["prism"].shift(-1)).abs()
    ok = as_coded.notna() & shifted.notna()
    print(f"  {p:8s} as coded {as_coded[ok].median():.1f}   shifted one day {shifted[ok].median():.1f}   (n={ok.sum()})")