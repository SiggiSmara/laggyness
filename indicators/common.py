# set up some path constants
import polars as pl
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
indicator_path = data_path / "stocks_1d_indicators"
centrally_smoothed_path = data_path / "stocks_1d_centrally_smoothed"


indicator_path.mkdir(exist_ok=True)


# get list of all tickers
tickers = [ x for x in (data_path / "stocks_1d").glob("*.parquet")]



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