import polars as pl
import polars_ds as pds

from rich.progress import track
from pathlib import Path

from common import (
    find_data_all_lazy,
    indicator_path
)

from momentum import (
    true_price,
    n_momentum,
    relative_zero_lag_macd,
    plus_di,
    minus_di,
    adx,
    rsi,
    cci,
    stochastic_percent_k,
    stochastic_oschilator,
    stochastic_oschilator_slow,
    TEMA
)

cols = [
    "high",
    "low",
    "close",
    "open",
    "volume"
]


periods = [5, 9, 14, 20]
periods_momentum = [1, 5,  9, 14, 20]

series_indicators = {}
for col in cols:
    for ind in [rsi, n_momentum, TEMA, relative_zero_lag_macd]:
        if ind == n_momentum:
            sel_periods = periods_momentum
        else:
            sel_periods = periods
        for period in sel_periods:
            # for col in cols:
            if ind == TEMA:
                name = f"{col}_{ind.__name__}_reldiff_{period}"
            else:
                name = f"{col}_{ind.__name__}_{period}"

            series_indicators[name] = {
                "func": ind,
                "series": col,
                "period": period
            }

periods = [5,  9, 14, 20]
dataframe_indicators = {
    "true_price": {"func": true_price, "period": None},
}
for ind in [
    minus_di,
    plus_di,
    adx,
    cci,
    stochastic_percent_k,
    stochastic_oschilator,
    stochastic_oschilator_slow,
]:
    for period in periods:
        dataframe_indicators[f"{ind.__name__}_{period}"] = {
            "func": ind,
            "period": period
        }


# find the eligible tickers
all_tickers, _ = find_data_all_lazy(track_on=False)

def calc_series(
    *, df:pl.DataFrame,
) -> pl.DataFrame:
    """Calculate the series indicator."""
    for ind, item in series_indicators.items():
        if "TEMA" in ind:
            df = df.with_columns(
                item["func"](src=df.get_column(item["series"]), period=item["period"]).pct_change().alias(ind)
            )
        elif "relative_zero_lag_macd" in ind:
            df = df.with_columns(
                item["func"](src=df.get_column(item["series"]), fast_period=item["period"], slow_period=round(item["period"]*26.0/12.0), signal_period=round(item["period"]*9.0/12.0)).alias(ind)
            )
        else:   
            df = df.with_columns(
                item["func"](df.get_column(item["series"]), item["period"]).alias(ind)
            )
    return df

def calc_dataframe(
    *, df:pl.DataFrame,
) -> pl.DataFrame:
    """Calculate the dataframe indicator."""
    for ind, item in dataframe_indicators.items():
        if ind == "true_price":
            df = df.with_columns(
                item["func"](df).alias(ind)
            )
        else:
            df = df.with_columns(
                item["func"](df, item["period"]).alias(ind)
            )
    return df

my_tckrs = [all_tickers[i] for i in range(0,len(all_tickers), 1000)]
my_tckrs = [allx for allx in all_tickers if Path(allx).stem in ["SPY", "AAPL", "AMZN", "GOOGL", "NFLX"]]
my_tckrs = all_tickers

for check_path in track(my_tckrs, description="Calculating indicators..."):
    ticker = Path(check_path)
    q = (
        pl.read_parquet(ticker)
    )
    
    q = calc_dataframe(df=q)
    q = calc_series(df=q)

    curr_cols = q.columns
    q = q.with_row_index("idx")
    
    diff_cols = []
    for col in cols:
        diff_name = f"{col}_reldiff"
        diff_cols.append(diff_name)
        q = q.with_columns(
            pl.col(col).pct_change().fill_null(0).alias(diff_name),
        )
        for period in periods:
            slope_name = f"{col}_perc_slope_{period}"
            curr_cols.append(slope_name)
            q = q.with_columns(
                pds.rolling_lin_reg(
                    "idx",
                    target=diff_name,
                    window_size=period,
                    add_bias=True,
                ).alias("linreg")
            ).unnest(
                "linreg"
            ).with_columns(
                pl.col("coeffs").list.first().alias(slope_name),
            ).select(curr_cols + diff_cols + ["idx",])
            
    q.select(curr_cols).drop_nulls().write_parquet(indicator_path / ticker.name)