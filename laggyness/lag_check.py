import polars as pl
import polars_ols as pls
from time import time
from pathlib import Path
import numpy as np
from rich.progress import track
import altair as alt
import matplotlib.pyplot as plt
import csv

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

def join_longframe(*, orig:list, period:int, smoother:str, ticker:str, df_sink:pl.LazyFrame = None) -> None:
    smooth_df = pl.LazyFrame({
        "period": [period for _ in range(len(orig))],
        "smoother": [smoother for _ in range(len(orig))],
        "ticker": [ticker for _ in range(len(orig))],
        "data": orig,
    })
    if df_sink is not None:
        df_sink = pl.concat([smooth_df, df_sink])
    else:
        df_sink = smooth_df
    
    return df_sink

def join_longframe_inmem(*, orig:list, period:int, smoother:str, ticker:str, df_sink:pl.DataFrame = None) -> None:
    smooth_df = pl.DataFrame({
        "period": [period for _ in range(len(orig))],
        "smoother": [smoother for _ in range(len(orig))],
        "ticker": [ticker for _ in range(len(orig))],
        "data": orig,
    })
    if df_sink is not None:
        df_sink = pl.concat([smooth_df, df_sink])
    else:
        df_sink = smooth_df
    
    return df_sink

def join_longdict(*, orig:pl.Series, period:int, smoother:str, ticker:str, dict_sink:dict = None) -> dict:
    smooth_df = {
        "period": [period for _ in range(len(orig))],
        "smoother": [smoother for _ in range(len(orig))],
        "ticker": [ticker for _ in range(len(orig))],
        "data": orig.to_list(),
    }
    if dict_sink is not None:
        for key, val in smooth_df.items():
            dict_sink[key].extend(val)
    else:
        dict_sink = smooth_df
    
    return dict_sink

def save_long(*, df_sink:pl.LazyFrame, ticker:str) -> None:
    df_sink.collect(streaming=True).write_parquet(intermediate_path / f"{ticker}.parquet")

def save_long_inmem(*, df_sink:pl.DataFrame, ticker:str) -> None:
    df_sink.write_parquet(intermediate_path / f"{ticker}.parquet")

def save_long_dic(*, dict_sink:dict, ticker:str) -> None:
    df_sink = pl.DataFrame(dict_sink)
    df_sink.write_parquet(intermediate_path / f"{ticker}.parquet")

def plot_outlier(*, q:pl.LazyFrame) -> None:
    df = q.collect()
    df = df.with_columns(
        row_num = pl.Series([x for x in range(df.shape[0])]),
    )
    title = f"{ticker}: original vs {sm1} vs {sm2}"
    subtitle = f"window = {period}"
    my_path = image_path / f"{sm1}_{sm2}_{period}_{ticker}_time.png"
    
    # plot the original and smoothed series
    fig, ax = plt.subplots(figsize=(13,13))
    ax.plot(df["row_num"], df["original"], label="original")
    ax.plot(df["row_num"], df[sm1], label=sm1)
    ax.plot(df["row_num"], df[sm2], label=sm2)

    # Hacky way to get the title and subtitle to show up
    # Add main title
    fig.suptitle(title, fontsize=font_size+10)
    # Add subtitle
    ax.set_title(subtitle, fontsize=font_size)
    
    ax.set_xlabel("trading day", fontsize=font_size)
    ax.set_ylabel("close", fontsize=font_size)
    ax.legend(fontsize=font_size)

    # Set tick font size
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.tick_params(axis='both', which='minor', labelsize=font_size - 10)

    fig.savefig(my_path)
    plt.close(fig)

    # ok plot the sm1 vs sm2 as a scatter plot
    title = f"{ticker}: correlation {sm1} vs {sm2}"
    subtitle=f"window = {period}, slope={round(coeffs.to_series(0)[0],3)}, intercept={round(coeffs.to_series(1)[0],3)}"
    my_path = image_path / f"{sm1}_{sm2}_{period}_{ticker}_inter.png"
    fig, ax = plt.subplots(figsize=(13,13))
    ax.scatter(df[sm1], df[sm2])

    # Hacky way to get the title and subtitle to show up
    # Add main title
    fig.suptitle(title, fontsize=font_size+10)
    # Add subtitle
    ax.set_title(subtitle, fontsize=font_size)

    ax.set_xlabel(sm1, fontsize=font_size)
    ax.set_ylabel(sm2, fontsize=font_size)
    
    # Set tick font size
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.tick_params(axis='both', which='minor', labelsize=font_size - 10)

    fig.savefig(my_path)
    plt.close(fig)

def plot_scatter_overview(*, df:pl.DataFrame, sm1:str, sm2:str, period:int) -> None:
    # first plot, scatter plot of the smoothed data from the two smoothers
    title = f"{sm1} vs {sm2}"
    subtitle=f"{tot_num} randomly selected tickers, window = {period}"
    my_path = image_path / f"{sm1}_{sm2}_{period}.png"

    fig, ax = plt.subplots(figsize=(13,13))
    ax.scatter(final_df[sm1], final_df[sm2])
    ax.set_xlabel(sm1, fontsize=font_size)
    ax.set_ylabel(sm2, fontsize=font_size)

    # Hacky way to get the title and subtitle to show up
    # Add main title
    fig.suptitle(title, fontsize=font_size+10)
    # Add subtitle
    ax.set_title(subtitle, fontsize=font_size)

    # Set tick font size
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.tick_params(axis='both', which='minor', labelsize=font_size - 10)
    
    fig.savefig(my_path)
    plt.close(fig)

def plot_histogram_overview(*, df_hist:pl.DataFrame, sm1:str, sm2:str, period:int) -> None:
     # second plot, histograms of the slopes and intercepts
    title = "\n".join((f"Correlation histograms {sm1} vs {sm2}", f" window = {period}"))
    my_path = image_path / f"{sm1}_{sm2}_{period}_hist.png"

    fig, axs = plt.subplots(ncols=2, figsize=(15,13))
    axs[0].hist(df_hist["slope"], bins=100)
    # axs[0].hist([x["slope"] for x in df_hist], bins=100)
    axs[0].set_title("slope", fontsize=font_size)

    axs[1].hist(df_hist["intercept"], bins=50)
    # axs[1].hist([x["intercept"] for x in df_hist], bins=50)
    axs[1].set_title("intercept", fontsize=font_size)

    # Set the overall title
    fig.suptitle(title, fontsize=font_size+10)

    # Set tick font size
    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=font_size)
        ax.tick_params(axis='both', which='minor', labelsize=font_size - 10)

    plt.subplots_adjust(wspace=0.25)

    fig.savefig(my_path)
    plt.close(fig)


# get a random sample of tickers
sample_tickers = get_random_subset(sampling_ratio=0.025)

# define some inital variables and values
t_start = time()
skipped = 0
res = []
smooth_data = {}
smooth_ticker_data = {}
orig_ticker_data = {}
# trading_days_all = 0
trading_days_included = []

# clear some transient data
clear_intermediates()


for ticker in track(sample_tickers, description="Smoothing tickers ..."):
    # read the data directly into a dataframe
    # orig = pl.read_parquet(ticker).get_column("close")
    
    # use a lazyframe to read the data
    q = (
        pl.scan_parquet(ticker)
        .select([
            "close"
        ])
    )
    orig = q.collect().get_column("close")

    # check to see if there are enough trading days to do the moving averages
    # trading_days_all+=1
    if orig.shape[0] < 500:
        skipped += 1
        continue

    # collect the number of trading days included to calculate an average for the summary later
    trading_days_included.append(orig.shape[0])

    #
    # define the dataframe for the smoothed data
    #

    # the lazyframe way
    # df_smooth = join_longframe(orig, 0, "original", ticker.stem)

    # the dataframe way
    df_smooth = join_longframe_inmem(orig=orig, period=0, smoother="original", ticker=ticker.stem)

    # the dictionary way
    # df_smooth = join_longdict(orig, 0, "original", ticker.stem)
    

    
    for smid, smoother in simple_smoothers.items():
        for period in periods:

            # smooth the data
            smoothed = smoother(src=orig, period=period)

            #
            # grow the dataframe with the smoothed data
            #

            # the lazyframy way
            # df_smooth = join_longframe(orig=smoothed, period=period, smoother=smid, ticker=ticker.stem, df_sink=df_smooth)

            # the dataframe way
            df_smooth = join_longframe_inmem(orig=smoothed, period=period, smoother=smid, ticker=ticker.stem, df_sink=df_smooth)

            # the dictionary way
            # df_smooth = join_longdict(orig=smoothed, period=period, smoother=smid, ticker=ticker.stem, dict_sink=df_smooth)

            # align the original and smoothed data (remove nulls at the start)
            orig_aligned, smoothed_aligned, delta_shift = align(orig=orig, smoothed=smoothed)

            # set initial values for the best shift window
            orig_mase = mase(orig=orig_aligned, smoothed=smoothed_aligned)
            mase_val1 = mase(orig=orig_aligned, smoothed=smoothed_aligned)
            n1 = 0

            # look for the best shift window
            for n in range(1, period):
                orig_new, smoothed_new = shift(orig=orig_aligned, smoothed=smoothed_aligned, n=n)
                mase_val = mase(orig=orig_new, smoothed=smoothed_new)
                if mase_val is None:
                    print((len(orig_new), len(smoothed_new)))
                    exit(1)
                if mase_val < mase_val1:
                    mase_val1 = mase_val
                    n1 = n

                # assuming monotonic decrease in mase in direction of best shift window
                # we don't need to do an exhaustive search, just go beyond the minimum
                if n > n1 + 3:
                    break

            # collect the results
            res.append({
                "ticker":ticker.stem,
                "smoother":smid,
                "period1":period,
                "period2":"n/a",
                "n":n1,
                "mase":mase_val1,
                "orig_mase":orig_mase,
            })
        
       
    # #
    # #
    # # testing a newly published MA: HullWEMA that was supposed to be zero lag
    # # but it doesn't seem to be, thus it's not included in the final results
    # # change the if False: to if True: to include it
    # # 
    # # watch out this code is not maintained and may not work, only left here for historical reasons
    # #
    # #
    # if False:
    #     for smid2, smoother in complex_smoothers.items():
    #         for period1 in periods:
                
    #             for period2 in [ x for x in periods if x<=period1 ]:
    
    #                 smoothed = smoother(orig, period1=period1, period2=period2)
    #                 smooth_df = smooth_df.with_columns(
    #                     pl.Series(name=f"{smid}_{period1}_{period2}", values=smoothed)
    #                 )
    #                 orig_aligned, smoothed_aligned, delta_shift = align(orig, smoothed)

    #                 orig_mase = mase(orig_aligned, smoothed_aligned)
    #                 mase_val1 = mase(orig_aligned, smoothed_aligned)
    #                 n1 = 0
    #                 for n in range(1, max(period1, period2)):
    #                     orig_new, smoothed_new = shift(orig, smoothed, n)
    #                     mase_val = mase(orig_new, smoothed_new)
    #                     if mase_val < mase_val1:
    #                         mase_val1 = mase_val
    #                         n1 = n
    #                     if n > n1 + 3:
    #                         break
    #                 res.append({
    #                     "ticker":ticker.stem,
    #                     "smoother":smid2,
    #                     "period1":period1,
    #                     "period2":period2,
    #                     "n":n1,
    #                     "mase":mase_val1,
    #                     "orig_mase":orig_mase,
    #                 })


    #
    # Save the smoothed data to a parquet file
    #

    # save the lazyframe data to a parquet file
    # by collecting the data first
    # save_long(df_sink=df_smooth, ticker=ticker.stem)

    # save the dataframe data to a parquet file
    # direct save
    save_long_inmem(df_sink=df_smooth, ticker=ticker.stem)

    # save a dictionary of the smoothed data to a parquet file
    # converting first to a dataframe
    # save_long_dic(dict_sink=df_smooth, ticker=ticker.stem)

    # create the dictionary of lazyframes for the smoothed data
    smooth_data[ticker.stem] = pl.scan_parquet(intermediate_path / f"{ticker.stem}.parquet")
t_smoothend = time()

# prepare for the summary of the results
smoothers = {}
for r in res:
    key = (r["smoother"], r["period1"], r["period2"])
    if key not in smoothers:
        smoothers[key] = {"n":0, "mase":0, "orig_mase":r["orig_mase"]}
    smoothers[key]["n"] += r["n"]
    smoothers[key]["mase"] += r["mase"]

tot_num = len(sample_tickers) - skipped

# summarize the window shifting results
res = []
for key, val in smoothers.items():
    # calculate averages
    val["n"] /= tot_num
    val["mase"] /= tot_num

    # collect the results
    res.append({
        "smoother":key[0],
        "period1":key[1],
        "period2":key[2],
        "best_shift_window_avg":val["n"],
        "best_shift_mase_avg":val["mase"],
        "orig_mase": val["orig_mase"]
    })

# write out the results
with open(result_path / "smoother_results.csv", "w") as f:
    dicf = csv.DictWriter(f, fieldnames=res[0].keys())
    dicf.writeheader()
    dicf.writerows(res)

# set the base plotting font size
font_size = 30

# build up a list of the combinations to be compared
smoother_comparison = ["TEMA", "HMA", "EHMA"]
smoother_combis = []
for i in range(len(smoother_comparison)-1):
    for j in range(i+1, len(smoother_comparison)):
        smoother_combis.append((smoother_comparison[i], smoother_comparison[j]))
        

# create the a dictionary of lazyframes pointing to the unsmoothed data
origs = {}
for ticker, sm_df in smooth_data.items():
    origs[ticker] = (
        sm_df
        .filter(pl.col("ticker") == ticker)
        .filter(pl.col("period") == 0)
        .with_columns(pl.col("data").alias("original"))
        .select("original")
    )

t_comp_start = time()
for sm1, sm2 in smoother_combis:
    for period in track(periods, description=f"Comparing {sm1} vs {sm2} ..."):
        # initialize the final dataframe for the scatter plot for the period
        final_df = pl.DataFrame({
            sm1: pl.Series([]), 
            sm2: pl.Series([]),
        })

        # set the list of results for the linear regression to an empty list for the period
        smoother_corrs = []

        for ticker in smooth_data:
            # select the lazyframe for the ticker
            sm_df = smooth_data[ticker]
            
            # select the smoother data for the two smoothers and the period
            sm_data = {
                sm1: sm_df.filter(pl.col("smoother") == sm1).filter(pl.col("period") == period),
                sm2: sm_df.filter(pl.col("smoother") == sm2).filter(pl.col("period") == period),
            }

            # select the ticker and rename the data columns to the smoother names
            q = pl.concat([
                sm_data[sm1].filter(pl.col("ticker") == ticker).with_columns(pl.col("data").alias(sm1)).select(sm1),
                sm_data[sm2].filter(pl.col("ticker") == ticker).with_columns(pl.col("data").alias(sm2)).select(sm2),
            ], how="horizontal")
            
            # collect and drop the nulls at the beginning (due to the smoothing)
            scaled_df = q.collect().drop_nulls()

            # scale the data from zero to one
            scaled_df = scaled_df.select((pl.all()-pl.all().min()) / (pl.all().max()-pl.all().min()))
            
            # do linear regression to find outliers
            cfs = scaled_df.select(
                pl.col(sm2)
                .least_squares.ols(pl.col(sm1), add_intercept=True, mode="coefficients")
                .alias("coeffs")
            ).unnest(
                "coeffs"
            ).row(0)


            # check if the regression was successful
            # if cfs[0] is not None:
            # check for (arbitrarily thresholded) extreme outliers
            # if (abs(1 - cfs[0]) > 0.2 or abs(cfs[1]) > 0.2): 
            #     df_outlier = pl.concat([q, origs[ticker]], how="horizontal")

            #     # send the lazyframe to a function that will plot the outlier data
            #     plot_outlier(q=df_outlier)

            # add the slope and intercept to the list of results
            smoother_corrs.append({"ticker":ticker, "slope":cfs[0], "intercept":cfs[1]})

            # collect the scaled data for a summary plot
            final_df = pl.concat([scaled_df.select(pl.col([sm1,sm2])), final_df])
        
        
        # first plot, scatter plot of the smoothed data from the two smoothers
        plot_scatter_overview(df=final_df, sm1=sm1, sm2=sm2, period=period)

        # second plot, histograms of the slopes and intercepts
        plot_histogram_overview(df_hist=pl.DataFrame(smoother_corrs), sm1=sm1, sm2=sm2, period=period)
        


t_end = time()
tot_combis = len(smoother_combis) * len(periods) * len(trading_days_included)
tot_smoothers = len(simple_smoothers) * len(periods) * len(trading_days_included)
print("".join([
    f"Time taken to process {len(sample_tickers)} tickers \n",
    f"of which {len(trading_days_included)} had more than 500 trading days and thus included in the analysis\n",
    f"with the average trading days being {round(sum(trading_days_included)/len(trading_days_included),2)}\n",
    f"Initial smoothing of the data for each ticker, for {len(periods)} windows of smoothing and {len(simple_smoothers)} number of moving averages ",
    f"for total smoothing operations of {round(tot_smoothers,2)}\n"
     f"took {round(t_smoothend - t_start, 2)} seconds or {round(1000 * (t_smoothend - t_start)/tot_smoothers, 2 )} sec/1000 moving averages\n",
    f"pairwise comparing {len(smoother_combis)} smoothers ",
    f"and {len(periods)} windows of smoothing, for a total of {tot_combis} comparisons\n",
    f"took: {round(t_end - t_comp_start, 2)} seconds or { round(1000 * (t_end - t_comp_start)/tot_combis, 2)} sec/1000 comparison\n",
    ])
)
