# set up some path constants
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
from scipy.signal import savgol_filter

my_path = Path(__file__).parent.parent
data_path = my_path / "data"
intermediate_path = data_path / "intermediate"
centrally_smoothed_path = data_path / "stocks_1d_centrally_smoothed"

centrally_smoothed_path.mkdir(exist_ok=True)

# smoothing windows
windows = [11, 17, 25]

# get list of all tickers
tickers = [ x for x in (data_path / "stocks_1d").glob("*.parquet")]


def centerSMA(*, data:pl.Series, period:int):
    sma = data.rolling_mean(window_size=period, center=True)
    return sma

def pl_savgol_filter(*, data:pl.Series, window:int, degree:int):
    data = savgol_filter(x=data.to_numpy(), window_length=window, polyorder=degree)
    return pl.Series("data", data)



def find_data_all_lazy(*, track_on:bool=True) -> Tuple[List, float]:
    paths = []
    avg_cnt = 0
    my_tickers = []
    for ticker in track(tickers, description="Finding tickers lazily...", disable=not track_on):
        # use a lazyframe to read the data
        q = (
            pl.scan_parquet(ticker)
            .with_columns(
                path = pl.lit(str(ticker))
            )
            .select([
                "path",
                "close"
            ])
        )
        my_tickers.append(q)
    ticker_df = pl.concat(my_tickers, how="vertical")
    ticker_df = (
        ticker_df
        .group_by("path").len(name="count")
        .filter(pl.col("count") > 500)
    ).collect()
    paths = ticker_df.get_column("path").to_list()
    avg_cnt = ticker_df.get_column("count").mean()
    return paths, avg_cnt