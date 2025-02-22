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
        diff = (pl.col("original") - pl.col("smoothed")).abs() / mean
    ).select("diff").collect().get_column("diff").mean()
    return mase

# shift the smoothed series n periods into the past
def lazy_shift(*, lazy_df:pl.LazyFrame, n:int=1):
    lazy_df = lazy_df.with_columns(
        smoothed = pl.col("smoothed").shift(-n)
    ).drop_nans()
    return lazy_df

# find the eligible tickers
smoothed_path = data_path.parent / f"{data_path.name}_smoothed_horizontal"
all_tickers = [ x for x in smoothed_path.glob("*.parquet")]

res = []
for ticker in track(all_tickers[0:100], description="Alignig tickers ..."):
    # print(ticker.stem)
    # read the data directly into a dataframe
    # orig = pl.read_parquet(ticker).get_column("close")
    
    # use a lazyframe to read the data
    q = ( 
        pl.scan_parquet(ticker)
         
    )
        # print(q.collect_schema().names())
    q2 = []
    q4 = []
    for period in periods:
        # print(period)
        my_ns = list(range(1, round(period / 2) + 2))
        for smid in simple_smoothers:            
            q = q.with_columns(
                pl.col(f"{smid}_{period}").shift(-n).alias(f"{smid}_{period}_shifted_{n}") for n in my_ns
            )
        q = q.drop_nulls()
        q = q.with_columns(
            pl.col("original").diff().abs().alias("original_delta")     
        )
        mean = q.select("original_delta").collect().get_column("original_delta").mean()
        sel_cols = []
        for smid in simple_smoothers:
            sel_cols += [f"{smid}_{period}_shifted_{n}_delta" for n in my_ns]
            q = q.with_columns(
                ((pl.col("original") - pl.col(f"{smid}_{period}_shifted_{n}")).abs() / mean).alias(f"{smid}_{period}_shifted_{n}_delta") for n in my_ns
                
            ).with_columns(
                ((pl.col("original") - pl.col(f"{smid}_{period}")).abs() / mean).alias(f"{smid}_{period}_shifted_0_delta"),
            )
            sel_cols += [f"{smid}_{period}_shifted_0_delta",]

            

        
        sel_cols += ["original",]
        # print(q.collect_schema().names())
        # print(sel_cols)
        qm = q.mean()  #.collect()
        # my_mase = qm.select(f"{smid}_{period}_shifted_0_delta").collect().get_column(f"{smid}_{period}_shifted_0_delta")
        my_periods = [period for _ in range(len(my_ns))]
        for smid in simple_smoothers:
            my_mase = qm.get_column(f"{smid}_{period}_shifted_0_delta")
            q4.append(
                pl.LazyFrame({
                    "smoother": smid,
                    "period": period,
                    "n": 0,
                    "mase": my_mase
                })
            )
        
        # for smid in simple_smoothers:
            qm.unpivot()
            q2.append(
                pl.LazyFrame({
                    "smoother": [smid for _ in range(len(my_ns))],
                    "period": my_periods,
                    "n": my_ns,
                    "mase": [qm.get_column(f"{smid}_{period}_shifted_{n}_delta")[0] for n in my_ns]
                })
            )
            q2.append(
                pl.LazyFrame().with_columns(
                    smoother = pl.lit(smid),
                    period = pl.lit(period, dtype=pl.Int64),
                    n = pl.lit(0, dtype=pl.Int64),
                    mase = pl.lit(my_mase)
                )
            )
    q2 = pl.concat(q2, how="vertical")
    # print(q2.collect())
    q4 = (
            pl.concat(q4, how="vertical")
            .select(["smoother", "period", "mase"])
            .rename({"mase":"orig_mase"})
        )
    # print(q4.collect())
    q3 = q2.group_by(["smoother", "period"]).min().select(["smoother", "period", "mase"])
    # print(q3.collect())
    q2 = q2.join(q3, on=["smoother", "period", "mase"])
    # print(q2.collect())
    q2 = (
        q2.join(q4, on=["smoother", "period"])
        .with_columns(
            ticker = pl.lit(str(ticker.stem))
        )
    )
    # q2.collect().write_parquet(intermediate_path / f"{ticker.stem}_smoother_results.parquet")
    res.append(q2)

# res.collect().write_parquet(intermediate_path / f"{ticker.stem}_smoother_results.parquet") #
pl.concat(res, how="vertical").collect().write_parquet(result_path / "smoother_raw_results.parquet") #
    # break

    
        

#                 mase_val = lazy_mase(
#                     lazy_df=lazy_shift(lazy_df=qq, n=n), 
#                     mean=orig_mean
#                 )
#                 if mase_val is None:
#                     print(lazy_shift(lazy_df=qq, n=n).collect())
#                     exit(1)
#                 if mase_val < mase_val1:
#                     mase_val1 = mase_val
#                     n1 = n

#                 # assuming monotonic decrease in mase in direction of best shift window
#                 # we don't need to do an exhaustive search, just go beyond the minimum
#                 if n > n1 + 3:
#                     break

#             # collect the results
#             res.append({
#                 "ticker":ticker.stem,
#                 "smoother":smid,
#                 "period":period,
#                 # "period2":"n/a",
#                 "n":n1,
#                 "mase":mase_val1,
#                 "orig_mase":orig_mase,
#             })

# pl.LazyFrame(res).sink_parquet(result_path / "smoother_raw_results.parquet")
# # prepare for the summary of the results
# smoothers = {}
# for r in res:
#     key = (r["smoother"], r["period"])
#     if key not in smoothers:
#         smoothers[key] = {"n":0, "mase":0, "orig_mase":r["orig_mase"]}
#     smoothers[key]["n"] += r["n"]
#     smoothers[key]["mase"] += r["mase"]

# tot_num = len(all_tickers) 

# # summarize the window shifting results
# res = []
# for key, val in smoothers.items():
#     # calculate averages
#     val["n"] /= tot_num
#     val["mase"] /= tot_num

#     # collect the results
#     res.append({
#         "smoother":key[0],
#         "period":key[1],
#         # "period2":key[2],
#         "best_shift_window_avg":val["n"],
#         "best_shift_mase_avg":val["mase"],
#         "orig_mase": val["orig_mase"]
#     })

# # write out the results
# with open(result_path / "smoother_results.csv", "w") as f:
#     dicf = csv.DictWriter(f, fieldnames=res[0].keys())
#     dicf.writeheader()
#     dicf.writerows(res)
