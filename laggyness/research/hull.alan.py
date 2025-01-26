"""Interesting strategy and moving averages code from one the greats, Alan Hull.
"""
import polars as pl

# Weighted Moving Average
def WMA(src:pl.Series, period:int):
    weights = pl.Series([i+1 for i in range(period)])
    return src.rolling_mean(window_size=period, weights=weights)

# Hull Moving Average, ALan Hull.  
# A zero lag MA
# https://alanhull.com/hull-moving-average
def HMA(src:pl.Series, period:int):
    wma_half = WMA(src, period // 2)
    wma_full = WMA(src, period)
    hma_input = 2 * wma_half - wma_full
    hma = WMA(hma_input[period-1:], int(period**0.5))
    res = pl.select(pl.repeat(None, period-1, dtype=pl.Float64)).to_series()
    res.append(hma)
    return res

# https://alanhull.com/hull-roar-indicator
def HULL_ROAR(src:pl.Series, period:int):
    pass