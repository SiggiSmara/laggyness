import polars as pl
import polars_ols as pls
from time import time
from pathlib import Path
import numpy as np
from rich.progress import track
import altair as alt
import matplotlib.pyplot as plt
import csv
from typing import List,Tuple
import concurrent.futures

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


# find the eligible tickers
smoothed_path = data_path.parent / f"{data_path.name}_smoothed_vertical"
all_tickers = [ x for x in smoothed_path.glob("*.parquet")]

def process_combination(ticker, period, smid):
    q = ( 
        pl.scan_parquet(ticker)  
    )

    orig_df = q.filter(pl.col("smoother") =="original").rename({"data":"original"}).select(["original","date"])
    q = (
            q.join(orig_df, on="date", how="inner")
            .with_columns(
                pl.col("original").cast(pl.Float64)
            )
            .with_columns(
                orig_diff = pl.col("original").diff().abs()
            )
    )
    qsel_p = q.filter(pl.col("period") == period)
    smoothed = ( 
        qsel_p.filter(pl.col("smoother") == smid)
        .with_columns(
            (pl.col("original") - pl.col("data")).abs().alias("mase_0") ,
        )
    )
   
    for n in range(1, period):
        smoothed = (
            smoothed.with_columns(
                (pl.col("original") - pl.col("data").shift(-n)).abs().alias(f"mase_{n}"),
            )
        )
    smoothed = smoothed.mean()
    
    for n in range(period):
        smoothed = smoothed.with_columns(
            (pl.col(f"mase_{n}") / pl.col(f"orig_diff")).alias(f"mase_{n}")
        )
    
    # Hacky, but have not found a more performant way of doing this
    # Find the index of the minimum value in the mases list
    smoothed = smoothed.collect()
    
    mases = [smoothed.get_column(f"mase_{n}")[0] for n in range(period)]
    min_index = mases.index(min(mases))
    
    return pl.LazyFrame({
        "mase": smoothed.get_column(f"mase_{min_index}")
    }).with_columns(
        shifted = pl.lit(min_index, dtype=pl.Int64),
        period = pl.lit(period, dtype=pl.Int64),
        smoother = pl.lit(smid, dtype=pl.Utf8),
        ticker = pl.lit(ticker.stem, dtype=pl.Utf8),
    ).select(["ticker", "period","smoother","shifted","mase"])

res = []
shifters = []
for ticker in track(all_tickers, description="Aligning tickers and finding the minimum..."):
    mins = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_combination, ticker, period, smid) for period in periods for smid in simple_smoothers]
        for future in concurrent.futures.as_completed(futures):
            mins.append(future.result())
    
    pl.concat(mins, how="vertical").collect(streaming=True).write_parquet(
        intermediate_path / "mins" / f"{ticker.stem}.parquet"
    )
