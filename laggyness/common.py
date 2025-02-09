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

from smoothers.moving_averages import SMA, WMA, EMA, HMA, EHMA, HullWEMA, TEMA #WEMA, 

my_path = Path(__file__).parent.parent
data_path = my_path / "data" / "stocks_1d"
image_path = my_path / "data" / "images"
result_path = my_path / "data" / "results"
intermediate_path = my_path / "data" / "intermediate"

# define the smoothers to be tested
simple_smoothers = {"SMA":SMA, "WMA":WMA, "EMA":EMA, "HMA":HMA, "EHMA":EHMA, "TEMA":TEMA }

# define the smoothing periods to be tested
periods = [5, 10, 20, 40, 80, 160]

# get list of all tickers
tickers = [ x for x in data_path.glob("*.parquet")]

def clear_intermediates():
    imgs = image_path.glob("*.png")
    for img in imgs:
        img.unlink()
    smalls = intermediate_path.glob("*.parquet")
    for small in smalls:
        if small.stem != "all_smoothed":
            small.unlink()


def get_random_subset(*, sampling_ratio:float = 0.0025):
    # get a random sample of tickers
    rng = np.random.default_rng()
    return rng.choice(tickers, round(sampling_ratio*len(tickers)), replace=False)