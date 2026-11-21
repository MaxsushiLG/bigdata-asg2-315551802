import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from features import ACTS, build_events, load_tables

ap = argparse.ArgumentParser()
ap.add_argument("--db", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--data-end", default="2026-10-15")
args = ap.parse_args()

events = build_events(load_tables(args.db))
events = events[events["t"] >= "2026-07-01"]
daily = events.groupby([events["t"].dt.floor("D"), "act"]).size().unstack(fill_value=0)
daily.columns = [ACTS[i] for i in daily.columns]
daily = daily.reindex(pd.date_range(daily.index.min(), "2026-11-20"), fill_value=0)
fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
daily.plot(ax=ax, linewidth=1.2)
ax.axvline(pd.Timestamp(args.data_end), color="black", linestyle="--")
ax.set_ylabel("events per day (9,000 cohort users)")
ax.set_xlabel("date (UTC)")
ax.set_title("Daily activity in the tables: drop in GH Archive coverage")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(args.out)
daily.to_csv(args.out.rsplit(".", 1)[0] + ".csv")
print(daily.sum(axis=1).describe())
