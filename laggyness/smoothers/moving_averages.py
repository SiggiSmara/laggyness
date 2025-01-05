import polars as pl

# Simple Moving Average
def SMA(src:pl.Series, period:int):
    return src.rolling_mean(window_size=period)

# Weighted Moving Average
def WMA(src:pl.Series, period:int):
    weights = pl.Series([i+1 for i in range(period)])
    return src.rolling_mean(window_size=period, weights=weights)

# Exponential Moving Average
def EMA(src:pl.Series, alpha:float=0.5, period:int=None):
    if period:
        alpha = 2 / (period + 1)
    return src.ewm_mean(alpha=alpha)

# Hull Moving Average, from one of the greats, ALan Hull.  
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

# Exponential HMA, same as HMA but with EMA instead of WMA
# author: ???
def EHMA(src:pl.Series, period:int):
    alpha =  2 / (period + 1)
    alpha2 = 2 / (period // 2 + 1)
    alpha3 = 2 / (int(period**0.5) + 1)
    ema_half = EMA(src, alpha=alpha2)
    ema_full = EMA(src, alpha=alpha)
    ehma_input = 2 * ema_half - ema_full
    ehma = EMA(ehma_input[period-1:], alpha3)
    res = pl.select(pl.repeat(None, period-1, dtype=pl.Float64)).to_series()
    res.append(ehma)
    return res

# TEMA - Triple Exponential Moving Average, a zero lag MA
# author: Patrick Mulloy 
# https://www.tradingtechnologies.com/xtrader-help/x-study/technical-indicator-definitions/triple-exponential-moving-average-tema/
def TEMA(src:pl.Series, alpha:float = 0.5, period:int=None):
    if period:
        alpha = 2 / (period + 1)
    ema1 = EMA(src, alpha=alpha)
    ema2 = EMA(ema1, alpha=alpha)
    ema3 = EMA(ema2, alpha=alpha)
    return 3 * ema1 - 3 * ema2 + ema3

# Weighted Exponential Moving Average
def WEMA(src:pl.Series, period1:int, alpha:float = 0.5, period2:int = None):
    if period2:
        wma_alpha = 2 / (period2 + 1)
    return EMA(WMA(src, period1), alpha=wma_alpha)

# Hull Weighted Exponential Moving Average, a zero lag MA
# based on the HMA
# https://github.com/senghansun/Hull-WEMA
def HullWEMA(src:pl.Series, period1:int, alpha:float = 0.5, period2:int = None):
    if period2:
        wema_alpha = 2 / (period2 + 1)
    return EMA(HMA(src, period1), alpha=wema_alpha)


