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
    get_random_subset,
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

def lazy_mase(*, lazy_df:pl.LazyFrame, mean:float):
    # mean = lazy_df.select("original").collect().get_column("original").diff().abs().mean()
    mase = lazy_df.with_columns(
        diff = (pl.col("original") - pl.col("smoothed")).abs() 
    )
    return mase

# shift the smoothed series n periods into the past
def lazy_shift(*, lazy_df:pl.LazyFrame, n:int=1):
    lazy_df = lazy_df.with_columns(
        smoothed = pl.col("smoothed").shift(-n)
    ).drop_nulls()
    return lazy_df

# find the eligible tickers
shifted_path = intermediate_path / "vertically_shifteds" 
all_tickers = [ x for x in shifted_path.glob("*.parquet")]

res = []

for ticker in track(all_tickers, description="Finding the optimum alignment ..."):
    if ticker.stem == "ZIP":
        # read the data directly into a dataframe
        # orig = pl.read_parquet(ticker).get_column("close")
        min_ticker = []
        # use a lazyframe to read the data
        q = ( 
            pl.scan_parquet(ticker).filter(pl.col("smoother") != "original")
            # .drop_nulls()
            # .drop_nans() 
        )
        print(q.select("shifted").unique().collect())
        # for period in periods:
        #     qsel_p = q.filter(pl.col("period") == period)
        #     for smid in simple_smoothers:
        #         qsel_ps = qsel_p.filter(pl.col("smoother") == smid)
        #         min_mase = qsel_ps.group_by(["period","smoother"]).min()
        #         qsel_ps.join(min_mase, on=["period","smoother","mase"])
        #         # print(min_mase.collect())
        #         #.filter(pl.col("mase") == pl.col("mase").min())

        #         min_ticker.append(qsel_ps)

        # min_tickers = pl.concat(min_ticker, how="vertical")      
        
        # min_tickers.collect(streaming=True).write_parquet(
        #     intermediate_path / "mins" / f"{ticker.stem}.parquet"
        # )
        break
    
    