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
    trend_windows,
    centerSMA,
    pl_savgol_filter,
)

# find the eligible tickers
all_tickers, _ = find_data_all_lazy(track_on=False)

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
            sdiff_name = f"original_vs_{orig_name}_rel_diff"
            asdiff_name = f"original_vs_{orig_name}_absrel_diff"
            # diff_name = f"{orig_name}_diff"
            rdiff_name = f"{orig_name}_rel_diff"
            # pos_name = f"{rdiff_name}_is_positive"
            # neg_name = f"{rdiff_name}_is_negative"
            posneg_name = f"{rdiff_name}_posneg"
        
            
            q = q.with_columns(
                (pl.col("original") - pl.col(orig_name)).alias(sdiff_name)
            ).with_columns(
                pl.col(sdiff_name).abs().alias(asdiff_name)
            ).with_columns(
                (pl.col(orig_name).pct_change()).alias(rdiff_name)
            ).with_columns(
                pl.when(pl.col(rdiff_name) > 0)
                    .then(pl.lit(1, dtype=pl.Int32))
                .otherwise(pl.lit(0, dtype=pl.Int32))
                .alias(posneg_name)
            )
            for tw in trend_windows:
                trend_name = f"{rdiff_name}_{tw}_trend"
                label_name = f"{rdiff_name}_label{tw}"
                index_name = f"{rdiff_name}_labelindex{tw}"
                mase_name = f"{orig_name}_mase{tw}"
                q = q.with_columns(
                    pl.col(posneg_name).rolling_mean(window_size=tw, center=True).alias(trend_name),
                    (pl.col(asdiff_name).rolling_median(
                        window_size=tw, center=True
                    )/pl.col(rdiff_name).abs().rolling_median(
                        window_size=tw, center=True
                    )).alias(mase_name),
                ).with_columns(
                    pl.col(trend_name).cut(
                        breaks=[0.2, 0.8],
                        labels=["0", "0.5", "1"]
                    ).cast(pl.Float32).alias(label_name)
                ).with_columns(
                    pl.col(label_name).ne(pl.col(label_name).shift(1)).cum_sum().alias(index_name)
                )
            
        q.collect(streaming=True).drop_nulls().write_parquet(centrally_smoothed_path / f"{ticker.stem}.parquet")

def sw_label_final():
    for ticker in track(all_tickers, description="Labeling tickers ..."):
        ticker = Path(ticker)
        q = (
            # pl.read_parquet(centrally_smoothed_path / f"{ticker.stem}.parquet")
            pl.scan_parquet(centrally_smoothed_path / f"{ticker.stem}.parquet")
        )
        window = windows[0]
        tw = trend_windows[0]
        
        orig_name = f"savgol_p2_{window}"
        sdiff_name = f"original_vs_{orig_name}_rel_diff"
        asdiff_name = f"original_vs_{orig_name}_absrel_diff"
        rdiff_name = f"{orig_name}_rel_diff"
        posneg_name = f"{rdiff_name}_posneg"

        trend_name = f"{rdiff_name}_{tw}_trend"
        label_name = f"{rdiff_name}_label{tw}"
        index_name = f"{rdiff_name}_labelindex{tw}"
        mase_name = f"{orig_name}_mase{tw}"
        
        q = q.with_columns(
            (pl.col(orig_name).pct_change()).alias(rdiff_name),
            (pl.col("original") - pl.col(orig_name)).alias(sdiff_name),
            # (pl.col("original") - pl.col(orig_name)).abs().alias(asdiff_name),
            # (pl.col(orig_name).pct_change() > 0).cast(pl.Int32).alias(posneg_name)
        ).with_columns(
            pl.col(sdiff_name).abs().alias(asdiff_name),
            # pl.when(pl.col(rdiff_name) > 0)
            #     .then(pl.lit(1, dtype=pl.Int32))
            # .otherwise(pl.lit(0, dtype=pl.Int32))
            # .alias(posneg_name),
            (pl.col(rdiff_name) > 0).cast(pl.Int32).alias(posneg_name)
        ).with_columns(
            pl.col(posneg_name).rolling_mean(window_size=tw, center=True).alias(trend_name),
            (pl.col(asdiff_name).rolling_median(
                window_size=tw, center=True
            )/pl.col(rdiff_name).abs().rolling_median(
                window_size=tw, center=True
            )).alias(mase_name),
        # ).with_columns(
        #     pl.col(trend_name).cut(
        #         breaks=[0.2, 0.8],
        #         labels=["0", "0.5", "1"]
        #     ).cast(pl.Float32).alias(label_name)
        ).with_columns(
            pl.col(trend_name).ne(pl.col(trend_name).shift(1)).cum_sum().alias(index_name)
        )
            
        # q.drop_nulls().write_parquet(centrally_smoothed_path / f"{ticker.stem}.parquet")
        q.collect(streaming=True).drop_nulls().write_parquet(centrally_smoothed_path / f"{ticker.stem}.parquet")

if __name__ == "__main__":
    # sw_smooth()
    # sw_label()
    sw_label_final()