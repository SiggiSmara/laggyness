"""Interesting strategies and signal processing code stemming
from one of the greats, John Ehlers and his company Mesa Software.
"""

import polars as pl

# https://www.mesasoftware.com/papers/A%20Thinking%20Man's%20MACD.pdf
def TMMACD(src:pl.Series, fastperiod:int, slowperiod:int, signalperiod:int):
    # macd = EMA(src, alpha=2/(fastperiod+1)) - EMA(src, alpha=2/(slowperiod+1))
    # signal = EMA(macd, alpha=2/(signalperiod+1))
    # return macd, signal
    pass

# FRAMA, https://www.mesasoftware.com/papers/FRAMA.pdf
def FRAMA(src:pl.Series, period:int):
    # return EMA(src, alpha=1/period)
    pass

# MESA Adaptive Moving Average (MAMA),https://www.mesasoftware.com/papers/MAMA.pdf
def MAMA(src:pl.Series, fastlimit:float, slowlimit:float):
    # return MAMA(src, fastlimit, slowlimit)
    pass

# zero lag exponential moving average, https://www.mesasoftware.com/papers/ZeroLag.pdf
def ZLEMA(src:pl.Series, period:int):
    # return EMA(2 * src - EMA(src, period), alpha=2/(period+1))
    pass