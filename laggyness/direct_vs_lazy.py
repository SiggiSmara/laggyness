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
import timeit
import time
import json
import altair as alt
from typing import List,Tuple

from smoothers.moving_averages import SMA, WMA, EMA, HMA, EHMA, HullWEMA, TEMA #WEMA, 

from common import (
    clear_intermediates, 
    get_random_subset,
    simple_smoothers,
    periods,
    data_path,
    image_path,
    result_path,
    intermediate_path,
)

my_path = Path(__file__).parent.parent
data_path = my_path / "data" / "stocks_1d"
image_path = my_path / "data" / "images"
result_path = my_path / "data" / "results"
intermediate_path = my_path / "data" / "intermediate"

# get list of all tickers
tickers = [ x for x in data_path.glob("*.parquet")]

# get a random sample of tickers
rng = np.random.default_rng()
sampling_perc = 0.025
sample_tickers = rng.choice(tickers, round(sampling_perc*len(tickers)), replace=False)
len_tickers = len(tickers)

# define the smoothers to be tested
simple_smoothers = {"SMA":SMA, "WMA":WMA, "EMA":EMA, "HMA":HMA, "EHMA":EHMA, "TEMA":TEMA }

# define the smoothing periods to be tested
periods = [5, 10, 20, 40, 80, 160]

clear_intermediates()
sample_tickers = get_random_subset(sampling_ratio=0.0025)

def find_data_direct(*, sample_tickers:list, track_on:bool=True) -> Tuple[List, float]:
    # read the data directly into a dataframe
    paths = []
    avg_cnt = 0
    for ticker in track(sample_tickers, description="Finding tickers directly...", disable=not track_on):
        q = pl.read_parquet(ticker).get_column("close").len()
        if q > 500:
            paths.append(ticker)
            avg_cnt += q
    
    return paths, avg_cnt/len(paths)


def collect_data_direct(*, sample_tickers:list, paths:list = None, track_on:bool=True):
    if paths is None:
        paths, _ = find_data_direct(sample_tickers=sample_tickers, track_on=track_on)

    final_df = pl.DataFrame({
        "ticker": pl.Series([], dtype=pl.String), 
        "smoother": pl.Series([], dtype=pl.String),
        "period": pl.Series([], dtype=pl.Int64),
        "data": pl.Series([], dtype=pl.Float64),
    })

    for ticker in paths:
        df = (
            pl.read_parquet(ticker)
            .rename({
                "stock": "ticker",
                "close": "data",
            })
            .with_columns(
                smoother = pl.lit("original"),
                period = pl.lit(0, dtype=pl.Int64),
            )
            .select(["ticker", "smoother", "period", "data"])
        )
        final_df = pl.concat([final_df, df], how="vertical")
    return final_df



def find_data_lazy(*, sample_tickers:list, track_on:bool=True) -> Tuple[List, float]:
    paths = []
    avg_cnt = 0
    for ticker in track(sample_tickers, description="Finding tickers lazily...", disable=not track_on):
        # use a lazyframe to read the data
        q = (
            pl.scan_parquet(ticker)
            .select([
                "close"
            ])
            .count()
        ).collect().get_column("close")[0]
        if q > 500:
            paths.append(ticker)
            avg_cnt += q
        
    return paths, avg_cnt/len(paths)

def find_data_all_lazy(*, sample_tickers:List[Path], track_on:bool=True) -> Tuple[List, float]:
    paths = []
    avg_cnt = 0
    tickers = []
    for ticker in track(sample_tickers, description="Finding tickers lazily...", disable=not track_on):
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
        tickers.append(q)
    ticker_df = pl.concat(tickers, how="vertical")
    ticker_df = (
        ticker_df
        .group_by("path").len(name="count")
        .filter(pl.col("count") > 500)
    ).collect()
    paths = ticker_df.get_column("path").to_list()
    avg_cnt = ticker_df.get_column("count").mean()
    return paths, avg_cnt


def collect_data_lazy(*, sample_tickers:list, paths:list = None, track_on:bool=True):
    if paths is None:
        paths, _ = find_data_lazy(sample_tickers=sample_tickers, track_on=track_on)

    final_df = pl.LazyFrame({
        "ticker": pl.Series([], dtype=pl.String), 
        "smoother": pl.Series([], dtype=pl.String),
        "period": pl.Series([], dtype=pl.Int64),
        "data": pl.Series([], dtype=pl.Float64),
    })

    for ticker in track(paths, description="Collecting lazy data...", disable=not track_on):
        df = (
            pl.scan_parquet(ticker)
            .rename({
                "stock": "ticker",
                "close": "data",
            })
            .with_columns(
                smoother = pl.lit("original"),
                period = pl.lit(0, dtype=pl.Int64),
            )
            .select(["ticker", "smoother", "period", "data"])
        )
        final_df = pl.concat([final_df, df], how="vertical")
    return final_df.collect()

def collect_data_all_lazy(*, sample_tickers:list, paths:list = None, track_on:bool=True):
    if paths is None:
        paths, _ = find_data_lazy(sample_tickers=sample_tickers, track_on=track_on)

    tickers = []
    for ticker in track(paths, description="Collecting lazy data...", disable=not track_on):
        df = (
            pl.scan_parquet(ticker)
            .rename({
                "stock": "ticker",
                "close": "data",
            })
            .with_columns(
                smoother = pl.lit("original"),
                period = pl.lit(0, dtype=pl.Int64),
            )
            .select(["ticker", "smoother", "period", "data"])
        )
        tickers.append(df)
    final_df = pl.concat(tickers, how="vertical")
    return final_df.collect()



def collect_data_lazy_intermed_sink(*, sample_tickers:list, paths:list = None, track_on:bool=True):
    if paths is None:
        paths, _ = find_data_lazy(sample_tickers=sample_tickers, track_on=track_on)

    sinks = []
    for ticker in track(paths, description="Collecting lazy data...", disable=not track_on):
        df = (
            pl.scan_parquet(ticker)
            .rename({
                "stock": "ticker",
                "close": "data",
            })
            .with_columns(
                smoother = pl.lit("original"),
                period = pl.lit(0, dtype=pl.Int64),
            )
            .select(["ticker", "smoother", "period", "data"])
        )
        df.sink_parquet(intermediate_path / f"{ticker.stem}.parquet")
        sinks.append(pl.scan_parquet(intermediate_path / f"{ticker.stem}.parquet"))
    return pl.concat(sinks, how="vertical").collect()


def collect_data_lazy_intermed_memory(*, sample_tickers:list, paths:list = None, track_on:bool=True):
    if paths is None:
        paths, _ = find_data_lazy(sample_tickers=sample_tickers, track_on=track_on)

    sinks = []
    for ticker in track(paths, description="Collecting lazy data...", disable=not track_on):
        df = (
            pl.scan_parquet(ticker)
            .rename({
                "stock": "ticker",
                "close": "data",
            })
            .with_columns(
                smoother = pl.lit("original"),
                period = pl.lit(0, dtype=pl.Int64),
            )
            .select(["ticker", "smoother", "period", "data"])
        )
        sinks.append(df.collect(streaming=True))
    return pl.concat(sinks, how="vertical")


def collect_data_lazy_intermed_stream(*, sample_tickers:list, paths:list = None, track_on:bool=True):
    if paths is None:
        paths, _ = find_data_lazy(sample_tickers=sample_tickers, track_on=track_on)

    sinks = []
    for ticker in track(paths, description="Collecting lazy data...", disable=not track_on):
        df = (
            pl.scan_parquet(ticker)
            .rename({
                "stock": "ticker",
                "close": "data",
            })
            .with_columns(
                smoother = pl.lit("original"),
                period = pl.lit(0, dtype=pl.Int64),
            )
            .select(["ticker", "smoother", "period", "data"])
        )
        df.collect(streaming=True).write_parquet(intermediate_path / f"{ticker.stem}.parquet")
        sinks.append(pl.scan_parquet(intermediate_path / f"{ticker.stem}.parquet"))
    return pl.concat(sinks, how="vertical").collect()


def collect_data_direct_intermed_memory(*, sample_tickers:list, paths:list = None, track_on:bool=True):
    if paths is None:
        paths, _ = find_data_direct(sample_tickers=sample_tickers, track_on=track_on)
    sinks = []
    for ticker in paths:
        df = (
            pl.read_parquet(ticker)
            .rename({
                "stock": "ticker",
                "close": "data",
            })
            .with_columns(
                smoother = pl.lit("original"),
                period = pl.lit(0, dtype=pl.Int64),
            )
            .select(["ticker", "smoother", "period", "data"])
        )
        sinks.append(df)
    return pl.concat(sinks, how="vertical")


def collect_data_direct_intermed(*, sample_tickers:list, paths:list = None, track_on:bool=True):
    if paths is None:
        paths, _ = find_data_direct(sample_tickers=sample_tickers, track_on=track_on)
    sinks = []
    for ticker in paths:
        df = (
            pl.read_parquet(ticker)
            .rename({
                "stock": "ticker",
                "close": "data",
            })
            .with_columns(
                smoother = pl.lit("original"),
                period = pl.lit(0, dtype=pl.Int64),
            )
            .select(["ticker", "smoother", "period", "data"])
        )
        df.write_parquet(intermediate_path / f"{ticker.stem}.parquet")
        sinks.append(pl.read_parquet(intermediate_path / f"{ticker.stem}.parquet"))
    return pl.concat(sinks, how="vertical")





def smooth_data_direct(*, sample_tickers:list, track_on:bool=True):
    paths, avg_cnt = find_data_direct(sample_tickers=sample_tickers, track_on=track_on)
    final_df = pl.DataFrame({
        "ticker": pl.Series([], dtype=pl.String), 
        "smoother": pl.Series([], dtype=pl.String),
        "period": pl.Series([], dtype=pl.Int64),
        "data": pl.Series([], dtype=pl.Float64),
    })

    for ticker in paths:
        df = (
            pl.read_parquet(ticker)
            .rename({
                "stock": "ticker",
                "close": "data",
            })
            .with_columns(
                smoother = pl.lit("original"),
                period = pl.lit(0, dtype=pl.Int64),
            )
            .select(["ticker", "smoother", "period", "data"])
        )
        final_df = pl.concat([final_df, df], how="vertical")
        orig = df.get_column("data")

        for smid, smoother in simple_smoothers.items():
            for period in periods:
                smooth_df = pl.DataFrame({
                    "data": smoother(src=orig, period=period),
                }).with_columns(
                    ticker = pl.lit(ticker.stem ),
                    smoother = pl.lit(smid),
                    period = pl.lit(period, dtype=pl.Int64),
                ).select(["ticker", "smoother", "period", "data"])
                final_df = pl.concat([final_df, smooth_df], how="vertical")
    return final_df

def smooth_data_mixed(*, sample_tickers:list):
    paths, avg_cnt = find_data_lazy(sample_tickers=sample_tickers, track_on=False)
    orig_data = collect_data_lazy(sample_tickers=sample_tickers, paths=paths, track_on=False)
    

    final_df = pl.LazyFrame({
        "ticker": pl.Series([], dtype=pl.String), 
        "smoother": pl.Series([], dtype=pl.String),
        "period": pl.Series([], dtype=pl.Int64),
        "data": pl.Series([], dtype=pl.Float64),
    })
    for ticker in track(paths, description="Smoothing data...", disable=True):
        orig = orig_data.filter(
            pl.col("ticker") == ticker.stem
        ).get_column("data")
        for smid, smoother in simple_smoothers.items():
            for period in periods:
                smooth_df = pl.LazyFrame({
                    "data": smoother(src=orig, period=period),
                }).with_columns(
                    ticker = pl.lit(ticker.stem ),
                    smoother = pl.lit(smid),
                    period = pl.lit(period, dtype=pl.Int64),
                ).select(["ticker", "smoother", "period", "data"])
                final_df = pl.concat([final_df, smooth_df], how="vertical")
    
    return pl.concat([final_df.collect(), orig_data], how="vertical")


def plot_results(*, source:pl.DataFrame, title:str):
    # copied and adapted from https://altair-viz.github.io/gallery/multiline_tooltip.html

    # Create a selection that chooses the nearest point & selects, x-value
    nearest = alt.selection_point(
        nearest=True, on="pointerover",
        fields=["sampling_size"], empty=False
    )

    # The basic line
    line = alt.Chart(source).mark_line(interpolate="basis").encode(
        x=alt.X("sampling_size:Q", title="Number of Parquet files read"),
        y=alt.Y("avg_exec_time:Q", title="Avg execution time (s)"),
        color="function:N"
    )

    # Transparent selectors across the chart. This is what tells us
    # the x-value of the cursor
    selectors = alt.Chart(source).mark_point().encode(
        x="sampling_size:Q",
        opacity=alt.value(0),
    ).add_params(
        nearest
    )
    when_near = alt.when(nearest)

    # Draw points on the line, and highlight, selection
    points = line.mark_point().encode(
        opacity=when_near.then(alt.value(1)).otherwise(alt.value(0))
    )

    # Draw text labels near the points, and highlight, selection
    text = line.mark_text(align="left", dx=5, dy=-10).encode(
        text=when_near.then("avg_exec_time:Q").otherwise(alt.value(" "))
    )

    # Draw a rule at the location of the selection
    rules = alt.Chart(source).mark_rule(color="gray").encode(
        x="sampling_size:Q",
    ).transform_filter(
        nearest
    )

    # Put the five layers into a chart and bind the data
    chart = alt.layer(
        line, selectors, points, rules, text
    ).properties(
        width=600, height=300, title=title
    )
    return chart

def create_one_res(*, func:str, sampling_ratio:float, avg_exec_time:float):
    return {
        "function": func,
        "sampling_size": round(sampling_ratio * len_tickers),
        "avg_exec_time": round(avg_exec_time,2),
    }

def measure_execution_time(*, func, sample_tickers, no, paths:list = None):
    if paths is not None:
        t_start = time.time()
        for j in range(no):
            func(sample_tickers=sample_tickers[j], paths=paths[j], track_on=False)
        t_end = time.time()
    else:
        t_start = time.time()
        for j in range(no):
            func(sample_tickers=sample_tickers[j], track_on=False)
        t_end = time.time()
    return (t_end - t_start) / no

def fibonacci(n):
    fibonacci_cache = {0: 0, 1: 1}
    def fill_fib(n):
        if n not in fibonacci_cache:
            fibonacci_cache[n] = fill_fib(n-1) + fill_fib(n-2)
        return fibonacci_cache[n]
    fill_fib(n)
    
    # return all values, ensuring that the 1 only appears once
    return [value for key, value in fibonacci_cache.items() if key != 1]

def test_find_data():
    res = []
    samp_ratio = 0.0025
    max_n = 13
    num_seq = fibonacci(max_n) + [380]

    print(num_seq)
    no = 10
    funcs = {
        "find_lazy" : find_data_lazy,
        "find_all_lazy" : find_data_all_lazy,
        "find_direct" : find_data_direct,
    }
    for i in num_seq:
        samp_ratio2 = samp_ratio * (1 + i)
        samps = [get_random_subset(sampling_ratio=samp_ratio2) for _ in range(no)]
        for func in funcs:
            get_func = measure_execution_time(func=funcs[func], sample_tickers=samps, no=no)
            print(f"{func}, {no} calls, sampling ratio {round(samp_ratio2,3)} ({round(samp_ratio2 * len_tickers)}): {get_func:.3f} s")
            res.append(create_one_res(func=func, sampling_ratio=samp_ratio2, avg_exec_time=get_func))

    res_name = "selection_direct_vs_lazy"
    res_df = pl.DataFrame(res)
    res_df.write_csv(result_path / f"{res_name}.csv")
    res_df_zoom1 = res_df.filter(pl.col("sampling_size") < 3000)
    res_df_zoom2 = res_df.filter(pl.col("sampling_size") < 500)
    my_plots = {
        "final":{
            "df":res_df,
            "title":"",
        },
        "final_zoom1":{
            "df":res_df_zoom1,
            "title":"",
        },
        "final_zoom2":{
            "df":res_df_zoom2,
            "title":"",
        },
        "initial":{
            "df":res_df.filter(pl.col("function").is_in(["find_lazy","find_direct"])),
            "title":" naive",
        },
        "initial_zoom1":{
            "df":res_df_zoom1.filter(pl.col("function").is_in(["find_lazy","find_direct"])),
            "title":" naive",
        },
        "initial_zoom2":{
            "df":res_df_zoom2.filter(pl.col("function").is_in(["find_lazy","find_direct"])),
            "title":" naive",
        },
    }
    for one_plot in my_plots:
        res_plot = plot_results(source=my_plots[one_plot]["df"], title=f"Performance of{my_plots[one_plot]['title']} lazy vs direct data selection")
        res_plot.save(result_path / f"{res_name}_{one_plot}.png")
        res_plot.save(result_path / f"{res_name}_{one_plot}.html")
        res_plot.save(result_path / f"{res_name}_{one_plot}.json")

   

def test_collect_data():
    res = []
    samp_ratio = 0.0025
    max_n = 13
    num_seq = fibonacci(max_n) + [380]
    # num_seq = fibonacci(4) 

    # max_n = 12
    # num_seq = fibonacci(max_n) 
    # max_n = 8
    # num_seq = fibonacci(max_n) 
    # max_n = 6
    # num_seq = fibonacci(max_n) 
    print(num_seq)
    no = 1
    funcs = {
        "collect_lazy" : collect_data_lazy,
        "collect_all_lazy" : collect_data_all_lazy,
        "collect_direct" : collect_data_direct,
        "collect_lazy_sink" : collect_data_lazy_intermed_sink,
        "collect_lazy_stream" : collect_data_lazy_intermed_stream,
        "collect_lazy_memory" : collect_data_lazy_intermed_memory,
        "collect_direct_intermed" : collect_data_direct_intermed,
        "collect_direct_intermed_memory" : collect_data_direct_intermed_memory,
    }
    for i in num_seq:
        samp_ratio2 = samp_ratio * (1 + i)
        samps = [get_random_subset(sampling_ratio=samp_ratio2) for _ in range(no)]
        paths = []
        for samp in samps:
            p = find_data_lazy(sample_tickers=samp, track_on=False)
            paths.append(p[0])
        for func in funcs:
            get_func = measure_execution_time(func=funcs[func], sample_tickers=samps, paths=paths, no=no)
            print(f"{func}, {no} calls, sampling ratio {round(samp_ratio2,3)} ({round(samp_ratio2 * len_tickers)}): {get_func:.3f} s")
            res.append(create_one_res(func=func, sampling_ratio=samp_ratio2, avg_exec_time=get_func))

       
    # create some plots of the data
    res_name = "collect_direct_vs_lazy"
    res_df = pl.DataFrame(res)
    res_df.write_csv(result_path / f"{res_name}.csv")
    res_df_zoom1 = res_df.filter(pl.col("sampling_size") < 3000)
    res_df_zoom2 = res_df.filter(pl.col("sampling_size") < 500)
    my_plots = {
        "final":{
            "df":res_df,
            "title":"",
        },
        "initial":{
            "df":res_df.filter(pl.col("function").is_in(["collect_lazy","collect_direct"])),
            "title":" naive",
        },
        "second":{
            "df":res_df.filter(pl.col("function").is_in(["collect_all_lazy"]) == False),
            "title":" more naive",
        },
        "zoom1_final":{
            "df":res_df_zoom1,
            "title":"",
        },
        "zoom1_first":{
            "df":res_df_zoom1.filter(pl.col("function").is_in(["collect_lazy","collect_direct"])),
            "title":" naive",
        },
        "zoom1_second":{
            "df":res_df_zoom1.filter(pl.col("function").is_in(["collect_all_lazy"]) == False),
            "title":" more naive",
        },
        "zoom2_final":{
            "df":res_df_zoom2,
            "title":"",
        },
        "zoom2_first":{
            "df":res_df_zoom2.filter(pl.col("function").is_in(["collect_lazy","collect_direct"])),
            "title":" naive",
        },
        "zoom2_second":{
            "df":res_df_zoom2.filter(pl.col("function").is_in(["collect_all_lazy"]) == False),
            "title":" more naive",
        },
    }
    for one_plot in my_plots:
        res_plot = plot_results(source=my_plots[one_plot]["df"], title=f"Performance of{my_plots[one_plot]['title']} lazy vs direct data collection")
        res_plot.save(result_path / f"{res_name}_{one_plot}.png")
        res_plot.save(result_path / f"{res_name}_{one_plot}.html")
        res_plot.save(result_path / f"{res_name}_{one_plot}.json")



# generate the data
test_find_data()
test_collect_data()