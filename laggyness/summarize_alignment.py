import polars as pl
import polars_ols as pls
from time import time
from pathlib import Path
import numpy as np
from rich.progress import track
import altair as alt
import matplotlib.pyplot as plt
import csv
from typing import List

from smoothers.moving_averages import SMA, WMA, EMA, HMA, EHMA, HullWEMA, TEMA #WEMA, 
from common import (
    clear_intermediates, 
    get_random_subset,
    find_data_all_lazy,
    simple_smoothers,
    periods,
    data_path,
    image_path,
    result_path,
    intermediate_path,
)


res_path = result_path / "smoother_raw_results.parquet"

if res_path.exists():
    # load the data
    allres = pl.scan_parquet(result_path / "smoother_raw_results.parquet")
else:
    # find the eligible tickers
    all_tickers = [ x for x in (intermediate_path / "mins" ).glob("*.parquet")]

    res = [pl.scan_parquet(ticker) for ticker in all_tickers]

    print("Concatinating the data...")
    my_df = pl.concat(res, how="vertical").drop_nans().drop_nulls().filter(
        ~pl.any_horizontal(pl.col(pl.Float64).is_infinite())
    )
    # print(my_df.collect())
    my_df.sink_parquet(result_path / "smoother_raw_results.parquet")
    allres = pl.scan_parquet(result_path / "smoother_raw_results.parquet")
    # print(allres.collect())
    allres.sink_csv(result_path / "smoother_raw_results.csv")

# for smid in simple_smoothers:
# for period in track(periods, description="Summarizing results for each period ..."):
    # for each of the periods, plot a histogra of n values for each smoother
period = 160
period_df = allres.filter(pl.col("period") == period).collect()
if period != 5:
    print(period)
    print(f"all data: {period_df.shape}")
    # print("data with more than one ticker/smoother combi:")
    # print(period_df.group_by([pl.col("ticker"), pl.col("smoother")]).len().filter(pl.col("len") > 1))

    print("data with n=0 for smoother HMA:") #and more than one ticker/smoother combi:
    null_only = period_df.filter([pl.col("shifted") == 0,])# pl.col("smoother") == 'HMA'].group_by([pl.col("smoother")]).len() #.group_by([pl.col("ticker"), pl.col("smoother")]).len().filter(pl.col("len") > 1)
    all_tickers = null_only.select(["smoother", "ticker"]) # sorted(null_only.get_column("ticker").to_list())

    print(all_tickers)

period_df = period_df.filter(pl.col("shifted") < 100)
uniform_jitter = alt.Chart(
    period_df, 
    title=f"Laggynesss, smoothing window = {period}",
    width=800,
).mark_bar(
    opacity=0.3,
    binSpacing=0,
    
).encode(
    x = alt.X('shifted:Q', title="Lag", ).bin(maxbins=100, minstep=1),  #scale=alt.Scale(domain=[0, 100])
    y = alt.Y('count()', title="Count of tickers").stack(None),
    color = alt.Color('smoother:N'),
)

uniform_jitter.save(str(image_path / f"smooth_summary_{period}.html"))
uniform_jitter.save(str(image_path / f"smooth_summary_{period}.png"))

summary = (
    allres.group_by([pl.col("period"), pl.col("smoother")])
    .agg(
        pl.col("shifted").mean().alias("lag_mean"),
        (pl.col("shifted").std() / pl.col("shifted").mean()).alias("lag_rsd"),
        pl.col("mase").mean().alias("mase_mean"),
        (pl.col("mase").std() / pl.col("mase").mean()).alias("mase_rsd"),
    ).with_columns(
        (pl.col("lag_mean") / pl.col("period")).alias("lag_relative_mean"),
        (pl.col("mase_mean") / pl.col("period")).alias("mase_relative_mean"),
    ).sort("period", "lag_relative_mean" )
).collect()

summary.write_csv(result_path / "smoother_summary.csv")

mase_graph = alt.Chart( 
    summary, 
    # width=800,
    # height=600,
).mark_bar().encode(
    x=alt.X('smoother:N', sort = '-y'),
    y=alt.Y('mase_mean:Q', title="mean MASE"),
    color=alt.Color('smoother:N', sort = '-y'),
    column='period:N'
)

mase_graph.save(str(image_path / f"mase_summary.html"))
mase_graph.save(str(image_path / f"mase_summary.png"))


lag_graph = alt.Chart(
    summary, 
    # width=800,
    # height=600,
).mark_bar().encode(
    x=alt.X('smoother:N', sort = '-y'),
    y=alt.Y('lag_relative_mean:Q', title="relative lag"),
    color=alt.Color('smoother:N', sort = '-y'),
    column='period:N'
)

lag_graph.save(str(image_path / f"lag_summary.html"))
lag_graph.save(str(image_path / f"lag_summary.png"))

# for period in track(periods, description="Summarizing results for each period ..."):
#     source = summary.filter(pl.col("period") == period)
#     mase_graph = alt.Chart(source).mark_bar().encode(
#         x='smoother:N',
#         y='mase_mean:Q',
#         color='smoother:N',
#         column='period:N'
#     )

#     mase_graph.save(str(image_path / f"mase_summary_{period}_meas.html"))
#     mase_graph.save(str(image_path / f"mase_summary_{period}_meas.png"))