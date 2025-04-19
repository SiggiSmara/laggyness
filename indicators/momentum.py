import polars as pl
import polars_ols as pls  

import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

def true_price(df:pl.DataFrame) -> pl.Series:
    """Calculate the true price from high, low and close prices.  Requires 'high', 'low' and 'close' columns.
    """
    return (df.get_column("high") + df.get_column("low") + df.get_column("close")) / 3

def n_momentum(src:pl.Series, period:int = 14) -> pl.Series:
    """Calculate the n-day momentum of a price series"""
    return src.pct_change(period)

def average_true_range(df:pl.DataFrame, period:int = 14) -> pl.Series:
    """Calculate the average true range of a price series. Requires 'high', 'low' and 'close' columns.
    """
    df = pl.DataFrame().with_columns(
        (df.get_column("high") - df.get_column("low")).alias("high_low"),
        (df.get_column("high") - df.get_column("close").shift()).abs().alias("high_close"),
        (df.get_column("low") - df.get_column("close").shift()).abs().alias("low_close")
    ).with_columns(
        pl.max_horizontal(["high_low", "high_close", "low_close"]).alias("max")
    )
    tr = df.get_column("max")
    return tr.rolling_mean(window_size=period)

def plus_di(df:pl.DataFrame, period:int = 14) -> pl.Series:
    """Calculate the +DI of a price series. Requires 'high', 'low' and 'close' columns.
    """
    tr = average_true_range(df, period)
    plus_dm = (df.get_column("high") - df.get_column("high").shift()).clip(lower_bound=0)
    plus_di = (plus_dm.rolling_mean(window_size=period) / tr.rolling_mean(window_size=period)) * 100
    return plus_di

def minus_di(df:pl.DataFrame, period:int = 14) -> pl.Series:
    """Calculate the -DI of a price series. Requires 'high', 'low' and 'close' columns.
    """
    tr = average_true_range(df, period)
    minus_dm = (df.get_column("low").shift() - df.get_column("low")).clip(lower_bound=0)
    minus_di = (minus_dm.rolling_mean(window_size=period) / tr.rolling_mean(window_size=period)) * 100
    return minus_di

def adx(df:pl.DataFrame, period:int = 14) -> pl.Series:
    """Calculate the ADX of a price series. Requires 'plus_di_{period}' and 'minus_di_{period}' columns
    or 'high' and 'low' columns.
    """
    if f"plus_di_{period}" not in df.columns:
        plus_di_val = plus_di(df, period)
    else:
        plus_di_val = df.get_column(f"plus_di_{period}")

    if f"minus_di_{period}" not in df.columns:
        minus_di_val = minus_di(df, period)
    else:
        minus_di_val = df.get_column(f"minus_di_{period}")
    
    adx = (plus_di_val - minus_di_val).abs() / (plus_di_val + minus_di_val) * 100
    return adx.rolling_mean(window_size=period)

def rsi(src:pl.Series, period:int = 14) -> pl.Series:
    """Calculates the RSI index. Works for any price series."""
    delta = src.diff()
    up = delta.clone().clip(lower_bound=0).ewm_mean(span=period)
    down = delta.clone().clip(upper_bound=0).abs().ewm_mean(span=period)
    return 100 - (100 / (1 + (up / down)))

def cci(df:pl.DataFrame, period:int = 20) -> pl.Series:
    """Calculate the Commodity Channel Index (CCI) of a price series. 
    Requires 'true_price' column or 'high', 'low' and 'close' columns.
    """
    if f"true_price" not in df.columns:
        pt = true_price(df)
    else:   
        pt = df.get_column("true_price")
    
    sma = pt.rolling_mean(window_size=period)
    mad = (pt - sma).abs().rolling_mean(window_size=period)
    return (pt - sma) / (0.015 * mad)

def stochastic_percent_k(df:pl.DataFrame, period:int = 14) -> pl.Series:
    """Calculate the n-day %K of a price series. 
    Requires columns 'high', 'low' and 'close'.
    """
    min_low = df.get_column("low").rolling_min(window_size=period)
    max_high = df.get_column("high").rolling_max(window_size=period)
    return (df.get_column("close") - min_low) / (max_high - min_low) * 100


def stochastic_oschilator(df:pl.DataFrame, period:int = 14) -> pl.Series:
    """Calculate the Stochastic Oscillator of a price series.
    Requires 'stochastic_percent_k_{period}' column or columns 'high', 'low' and 'close'.
    """
    if f"stochastic_percent_k_{period}" not in df.columns:
        pt = stochastic_percent_k(df, period=period)
    else:   
        pt = df.get_column(f"stochastic_percent_k_{period}")
    
    return pt.rolling_median(window_size=3)

def stochastic_oscilator_signal(df:pl.DataFrame, period:int = 14) -> pl.Series:
    """Calculate the Stochastic Oscillator signal of a price series.
    Requires 'stochastic_oschilator_{period}' column or columns 'high', 'low' and 'close'.
    """
    if f"stochastic_percent_k_{period}" not in df.columns:
        pk = stochastic_percent_k(df, period=period)
    else:
        pk = df.get_column(f"stochastic_percent_k_{period}")

    if f"stochastic_oschilator_{period}" not in df.columns:
        so = stochastic_oschilator(df, period=period)
    else:   
        so = df.get_column(f"stochastic_oschilator_{period}")
    
    # Calculate the Stochastic signal as:
    #   when the pk is larger that so and the so is <= 20 then buy (1)
    #   when the pk is smaller that so and the so is => 80 then sell (0)
    #   otherwise do nothing (0.5)
    return pl.when((so <=20) & (pk > so) ).then(1).when((so >= 80) & (pk < so)).then(0).otherwise(0.5)


def stochastic_oschilator_slow(df:pl.DataFrame, period:int = 14) -> pl.Series:
    """Calculate the Slow Stochastic Oscillator of a price series.  Looks first for
    the normal stochastic oscillator column (stochastic_oschilator_{period}) and if not found,
    looks for the %K column (stochastic_percent_k_{period}). Based on what is found it calculates 
    the slow stochastic oscillator.
    If neither stocahstic columns are present requires 'high', 'low' and 'close' columns.
    """
    if f"sto_osch_{period}" not in df.columns:
        if f"stochastic_percent_k_{period}" not in df.columns:
            pt = stochastic_percent_k(df, period=period)
        else:   
            pt = df.get_column(f"stochastic_percent_k_{period}")
        pt = pt.rolling_median(window_size=3)
    else:   
        pt = df.get_column(f"stochastic_oschilator_{period}")
    
    return pt.rolling_median(window_size=3)



def EMA(*, src:pl.Series, alpha:float=0.5, period:int=None) -> pl.Series:
    """Calculate the Exponential Moving Average of a price series.
    """
    if period:
        alpha = 2 / (period + 1)
    return src.ewm_mean(alpha=alpha)


def TEMA(*, src:pl.Series, alpha:float = 0.5, period:int=None) -> pl.Series:
    """TEMA - Triple Exponential Moving Average, a zero lag MA
    author: Patrick Mulloy 
    https://www.tradingtechnologies.com/xtrader-help/x-study/technical-indicator-definitions/triple-exponential-moving-average-tema/
    """
    if period:
        alpha = 2 / (period + 1)
    ema1 = EMA(src=src, alpha=alpha)
    ema2 = EMA(src=ema1, alpha=alpha)
    ema3 = EMA(src=ema2, alpha=alpha)
    return 3 * ema1 - 3 * ema2 + ema3

def relative_zero_lag_macd(*, src:pl.Series, slow_period:int=26, fast_period:int = 12, signal_period:int = 9) -> pl.Series:
    """Calculate the relative MACD signal of a price series based on 
    zero lag TEMA.
    """
    macd = TEMA(src=src, period=fast_period) - TEMA(src=src, period=slow_period)
    signal = TEMA(src=macd, period=signal_period)
    return (macd - signal) / signal

