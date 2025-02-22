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

from smoothers.moving_averages import SMA, WMA, EMA, HMA, EHMA, HullWEMA, TEMA #WEMA, 
from common import (
    clear_intermediates, 
    find_data_all_lazy,
    simple_smoothers,
    periods,
    data_path,
    image_path,
    result_path,
    intermediate_path,
)


# Mean absolute scaled error, https://en.wikipedia.org/wiki/Mean_absolute_scaled_error
def mase(*, orig:pl.Series, smoothed:pl.Series):
    diff = orig - smoothed
    mase = (diff.abs() / (orig.diff().abs().mean())).mean()
    return mase

# remove nulls at the start of the smoothed series
def align(*, orig:pl.Series, smoothed:pl.Series):
    smoothed_new = smoothed.drop_nulls()
    orig_new = orig[len(orig) - len(smoothed_new):]
    return orig_new, smoothed_new, len(orig) - len(smoothed_new)

# shift the smoothed series n periods into the past
def shift(*, orig:pl.Series, smoothed:pl.Series, n:int=1):
    smoothed_new = smoothed[n:]
    orig_new = orig[:-n] 
    return orig_new, smoothed_new


# find the eligible tickers
all_tickers, _ = find_data_all_lazy(track_on=False)

# clear some transient data
clear_intermediates()

# define some inital variables and values
t_start = time()
# all_tickers = [data_path / "NRSN.parquet", ]
for ticker in track(all_tickers, description="Smoothing tickers ..."):
    # use a lazyframe to read the data
    ticker = Path(ticker)
    curr_ticker= str(ticker.stem)
    q = (
        pl.scan_parquet(ticker)
        .rename({
            "stock": "ticker",
            "close": "original",
        })
        .select(["ticker", "original"])
    )

    orig = q.select("original").collect().get_column("original")
    # print(f"ticker: {curr_ticker}, length: {len(orig)}")
    sm_list = [q,]
    for smid, smoother in simple_smoothers.items():
        # print(f"smoother: {smid}")
        # ctick = pl.LazyFrame({
        #     "ticker": [curr_ticker for _ in range(len(orig))],
        #     "smoother": [smid for _ in range(len(orig))],
        # })
        ctick = pl.LazyFrame()
        for period in periods:
            # print(f"period: {period}, length: {len(orig)}")
            # smooth the data
            smoothed = smoother(src=orig, period=period)

            # add the smoothed data to the list of lazyframes
            # sm_list.append(ctick.with_columns(
            #     period = pl.lit(period, dtype=pl.Int64),
            #     data = smoothed,
            # ))
            sm_list.append( pl.LazyFrame({
                f"{smid}_{period}": smoothed
            }))

    
    pl.concat(sm_list, how="horizontal").collect(streaming=True).write_parquet(data_path.parent / "stocks_1d_smoothed_horizontal" / f"{curr_ticker}.parquet")
        

