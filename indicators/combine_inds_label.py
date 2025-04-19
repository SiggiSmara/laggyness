import polars as pl

from rich.progress import track
from pathlib import Path

from common import (
    find_data_all_lazy,
    indicator_path,
    centrally_smoothed_path 
)

combined_path = indicator_path.parent / "stocks_1d_combined"
combined_path.mkdir(exist_ok=True)

# find the eligible tickers
all_tickers = list(centrally_smoothed_path.glob("*.parquet"))
my_tckrs = all_tickers

for label_ticker in track(my_tckrs, description="combining..."):
    ind_ticker = indicator_path / label_ticker.name
    q = (
        pl.scan_parquet(label_ticker).select(["date", "savgol_p2_11_rel_diff_5_trend"])
    )
    p = (
        pl.scan_parquet(ind_ticker)#.select(pcols)
    )
    # print(q.columns)
    # print(p.columns)
    pq = p.join(q, on="date", how="inner")
    pq.drop_nulls().collect(streaming=True).write_parquet(combined_path / label_ticker.name)

print(pq.collect_schema().names())