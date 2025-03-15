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


from common import (
    find_data_all_lazy,
    data_path,
    centrally_smoothed_path,
    windows,
    centerSMA,
    pl_savgol_filter,
)

# find the eligible tickers
all_tickers, _ = find_data_all_lazy(track_on=False)
trend_windows = [5, 7, 9]
def sw_smooth():
    for ticker in track(all_tickers, description="Smoothing tickers ..."):
        # use a lazyframe to read the data
        ticker = Path(ticker)
        q = (
            pl.read_parquet(ticker)
            .rename({
                "stock": "ticker",
                "close": "original",
            })
            .select(["ticker", "date", "original"])
        )
        for window in windows:
            orig_name = f"savgol_p2_{window}"
            # diff_name = f"{orig_name}_diff"
            rdiff_name = f"{orig_name}_rel_diff"
            # label_name = f"{rdiff_name}_label"
            # seven_trend_name = f"{rdiff_name}_7_trend"
            # small_name = f"{rdiff_name}_is_small"
            # pos_name = f"{rdiff_name}_is_positive"
            # neg_name = f"{rdiff_name}_is_negative"
            # posneg_name = f"{rdiff_name}_posneg"

            
            q = q.with_columns(
                pl.col("original").map_batches(lambda x: pl_savgol_filter(data=x, window=window, degree=2)).alias(orig_name)
            )
        q.drop_nulls().write_parquet(centrally_smoothed_path / f"{ticker.stem}.parquet")

def sw_label():
    for ticker in track(all_tickers, description="Labeling tickers ..."):
        ticker = Path(ticker)
        q = (
            pl.scan_parquet(centrally_smoothed_path / f"{ticker.stem}.parquet")
        )
        for window in windows:
            orig_name = f"savgol_p2_{window}"
            diff_name = f"{orig_name}_diff"
            rdiff_name = f"{orig_name}_rel_diff"
            # label_name = f"{rdiff_name}_label"
            # trend9_name = f"{rdiff_name}_9_trend"
            # trend7_name = f"{rdiff_name}_7_trend"
            # trend5_name = f"{rdiff_name}_5_trend"
            # label9_name = f"{rdiff_name}_label9"
            # label7_name = f"{rdiff_name}_label7"
            # label5_name = f"{rdiff_name}_label5"
            # small_name = f"{rdiff_name}_is_small"
            pos_name = f"{rdiff_name}_is_positive"
            neg_name = f"{rdiff_name}_is_negative"
            posneg_name = f"{rdiff_name}_posneg"
        
            
            q = q.with_columns(
                (pl.col(orig_name).diff()).alias(diff_name),
            ).with_columns(
                (pl.col(diff_name) / pl.col(orig_name).shift(-1)).alias(rdiff_name), 
            ).with_columns(
                    pl.col(rdiff_name).lt(0).alias(neg_name),
                    pl.col(rdiff_name).ge(0).alias(pos_name),
            ).with_columns(
                pl.when(pl.col(pos_name))
                    .then(pl.lit(1, dtype=pl.Int32))
                .when(pl.col(neg_name))
                    .then(pl.lit(0, dtype=pl.Int32))
                .otherwise(pl.lit(0, dtype=pl.Int32))
                .alias(posneg_name)
            )
            for tw in trend_windows:
                trend_name = f"{rdiff_name}_{tw}_trend"
                label_name = f"{rdiff_name}_label{tw}"
                index_name = f"{rdiff_name}_labelindex{tw}"
                q = q.with_columns(
                    pl.col(posneg_name).rolling_mean(window_size=tw, center=True).alias(trend_name)
                ).with_columns(
                    pl.when(pl.col(trend_name) >= 0.8)
                        .then(pl.lit(1, dtype=pl.Int32))
                    .when(pl.col(trend_name) <= 0.2)
                        .then(pl.lit(-1, dtype=pl.Int32))
                    .otherwise(pl.lit(0, dtype=pl.Int32))
                    .alias(label_name)
                ).with_columns(
                    pl.col(label_name).ne(pl.col(label_name).shift(1, fill_value=2)).cast(pl.Int32).cum_sum().alias(index_name)
                )
            # .with_columns(
            #     pl.col(posneg_name).rolling_mean(window_size=9, center=True).alias(trend9_name),
            #     pl.col(posneg_name).rolling_mean(window_size=7, center=True).alias(trend7_name),
            #     pl.col(posneg_name).rolling_mean(window_size=5, center=True).alias(trend5_name),

            # ).with_columns(
            #     pl.when(pl.col(trend9_name) >= 0.8)
            #         .then(pl.lit(1, dtype=pl.Int32))
            #     .when(pl.col(trend9_name) <= 0.2)
            #         .then(pl.lit(-1, dtype=pl.Int32))
            #     .otherwise(pl.lit(0, dtype=pl.Int32))
            #     .alias(label9_name),

            #     pl.when(pl.col(trend7_name) >= 0.8)
            #         .then(pl.lit(1, dtype=pl.Int32))
            #     .when(pl.col(trend7_name) <= 0.2)
            #         .then(pl.lit(-1, dtype=pl.Int32))
            #     .otherwise(pl.lit(0, dtype=pl.Int32))
            #     .alias(label7_name),

            #     pl.when(pl.col(trend5_name) >= 0.8)
            #         .then(pl.lit(1, dtype=pl.Int32))
            #     .when(pl.col(trend5_name) <= 0.2)
            #         .then(pl.lit(-1, dtype=pl.Int32))
            #     .otherwise(pl.lit(0, dtype=pl.Int32))
            #     .alias(label5_name),
            # )
            # .with_columns(
            #     pl.when(pl.col(seven_trend_name) > 0.8)
            #         .then(pl.lit("01_uptrend", dtype=pl.Utf8))
            #     .when(pl.col(seven_trend_name) < 0.2)
            #         .then(pl.lit("02_downtrend", dtype=pl.Utf8))
            #     .otherwise(pl.lit("03_unknown", dtype=pl.Utf8))
            #     .alias(label_name)
            # )
            
        q.collect(streaming=True).write_parquet(centrally_smoothed_path / f"{ticker.stem}.parquet")

if __name__ == "__main__":
    # sw_smooth()
    sw_label()