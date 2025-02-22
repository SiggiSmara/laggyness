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
smoothed_path = data_path.parent / "stocks_1d_smoothed_vertical" 
check_path = intermediate_path / "vertically_shifteds" 
all_data_paths = [x for x in check_path.glob("*.parquet")]
all_data = [pl.scan_parquet(x) for x in all_data_paths]

cnt = 0
for i, data in enumerate(all_data):
    ch_df = data.group_by(["period", "smoother", "shifted"]).len().collect()
    print(ch_df.columns)
    print(ch_df)
    break
    # if all_data_paths[i].stem in ("A", "ALZN"):
       
        # all_tickers = [data.filter((pl.col("period") == 0) & (pl.col("smoother") == "original")),]
        # all_tickers.append(data.filter(pl.col("period") == 160))
        # sum_df = pl.concat(all_tickers, how="vertical").collect()

        # my_fig = alt.Chart(sum_df).mark_line().encode(
        #     x="date",
        #     y="close",
        #     color="ticker"
        # )

        # print(summary_shape)
        # print(df.group_by([pl.col("smoother"),pl.col("period")]).len().shape)
        # print(df.filter(pl.col("n") == 0).shape)

print(cnt)

# smoothed_path = intermediate_path
# all_data = [pl.scan_parquet(x) for x in smoothed_path.glob("*.parquet")]

# for data in all_data:
#     df = data.collect()
#     # print(df.shape)
#     print(df.group_by([pl.col("smoother"),pl.col("period")]).len().shape)
#     print(df.filter(pl.col("n") == 0))
#     break