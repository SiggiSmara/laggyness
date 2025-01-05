import polars as pl
from pathlib import Path
import numpy as np
from rich.progress import track

from smoothers.moving_averages import SMA, WMA, EMA, HMA, EHMA, WEMA, HullWEMA, TEMA

# Mean absolute scaled error, https://en.wikipedia.org/wiki/Mean_absolute_scaled_error
def mase(orig:pl.Series, smoothed:pl.Series):
    diff = orig - smoothed
    mase = (diff.abs() / (orig.diff().abs().mean())).mean()
    return mase

# remove nulls at the start of the smoothed series
def align(orig:pl.Series, smoothed:pl.Series):
    smoothed_new = smoothed.drop_nulls()
    orig_new = orig[len(orig) - len(smoothed_new):]
    return orig_new, smoothed_new

# shift the smoothed series n periods into the past
def shift(orig:pl.Series, smoothed:pl.Series, n:int=1):
    smoothed_new = smoothed.shift(-n).drop_nulls()
    orig_new = orig[:(len(smoothed_new)-len(orig))]
    # print(smoothed_new.head(10))
    # print(smoothed_new.tail(10))
    # print(len(orig), len(smoothed))
    # print(len(orig_new), len(smoothed_new))
    return orig_new, smoothed_new

rng = np.random.default_rng()
data_path = Path(__file__).parent.parent / "data" / "stocks_1d"
tickers = [ x for x in data_path.glob("*.parquet")]
# get a representative sample of tickers
sample_tickers = rng.choice(tickers, round(0.01*len(tickers)), replace=False)

simple_smoothers = {"SMA":SMA, "WMA":WMA, "EMA":EMA, "HMA":HMA, "EHMA":EHMA, "TEMA":TEMA }
complex_smoothers = {"WEMA":WEMA, "HullWEMA":HullWEMA, }
periods = [5, 10, 20, 40, 80, 160]
skipped = 0
for ticker in track(sample_tickers):
    print(f"Processing {ticker.stem}")
    df = pl.read_parquet(ticker)
    if df.shape[0] < 500:
        skipped += 1
        continue
    # print(df.shape)
    orig = df["close"]
    res = []
    for smid, smoother in simple_smoothers.items():
        for period in periods:
            # print(f"Processing {ticker.stem} {smid} {period}")
            smoothed = smoother(orig, period=period)
            # print(len(orig), len(smoothed))
            orig_aligned, smoothed_aligned = align(orig, smoothed)
            mase_val1 = mase(orig_aligned, smoothed_aligned)
            n1 = period
            for n in range(1,period):
                # print(n)
                orig_new, smoothed_new = shift(orig_aligned, smoothed_aligned, n)
                mase_val = mase(orig_new, smoothed_new)
                if mase_val is None:
                    print((len(orig_new), len(smoothed_new)))
                    exit(1)
                if mase_val < mase_val1:
                    mase_val1 = mase_val
                    n1 = n
                if n > n1 + 3:
                    break
            res.append({
                "ticker":ticker.stem,
                "smoother":smid,
                "period1":period,
                "period2":0,
                "n":n1,
                "mase":mase_val1
            })
    for smid, smoother in complex_smoothers.items():
        for period1 in periods:
            for period2 in periods:
                smoothed = smoother(orig, period1=period1, period2=period2)
                orig_aligned, smoothed_aligned = align(orig, smoothed)
                mase_val1 = mase(orig_aligned, smoothed_aligned)
                n1 = max(period1, period2)
                for n in range(1, max(period1, period2)):
                    orig_new, smoothed_new = shift(orig, smoothed, n)
                    mase_val = mase(orig_new, smoothed_new)
                    if mase_val < mase_val1:
                        mase_val1 = mase_val
                        n1 = n
                    if n > n1 + 3:
                        break
                res.append({
                    "ticker":ticker.stem,
                    "smoother":smid,
                    "period1":period1,
                    "period2":period2,
                    "n":n1,
                    "mase":mase_val1
                })

smoothers = {}
for r in res:
    key = (r["smoother"], r["period1"], r["period2"])
    if key not in smoothers:
        smoothers[key] = {"n":0, "mase":0}
    smoothers[key]["n"] += r["n"]
    smoothers[key]["mase"] += r["mase"]

for key, val in smoothers.items():
    val["n"] /= len(sample_tickers) - skipped
    val["mase"] /= len(sample_tickers) - skipped
    print(f"{key} n average:{val['n']} mase average:{val['mase']}")

print(f"Skipped {skipped} tickers")
print(f"Processed {len(sample_tickers) - skipped} tickers")
