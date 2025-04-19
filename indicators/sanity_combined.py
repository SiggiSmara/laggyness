
import polars as pl
import numpy as np
from rich.progress import track
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from common import (
    find_data_all_lazy,
    data_path,
    indicator_path
)
# check_path = centrally_smoothed_path / "AMN.parquet"

images_path = data_path / "intermediate" / "combined" / "images"
images_path.mkdir(exist_ok=True, parents=True)

# find the eligible tickers
combined_path = indicator_path.parent / "stocks_1d_combined"
all_tickers = list(combined_path.glob("*.parquet"))
all_tickers = np.random.default_rng().choice(all_tickers, size=10, replace=False)
cols = [
    "high",
    "low",
    "close",
    "open",
    "volume"
]


def make_scatter_plot(
    *,
    df:pl.DataFrame,
    y_col:str,
    title:str,
    filename:str,
    x_col:str,
):
    fig, ax = plt.subplots(figsize=(16, 8), dpi=300)
    my_ycol = df.get_column(y_col)
    my_xcol = df.get_column(x_col)
    # Normalize the values to the range [0, 1]
    my_ycol = (my_ycol - my_ycol.min()) / (my_ycol.max() - my_ycol.min())
    my_xcol = (my_xcol - my_xcol.min()) / (my_xcol.max() - my_xcol.min())
    

    ax.scatter(my_xcol, my_ycol)
    # ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filename) 
    plt.close(fig)

def make_plot(
    *,
    df:pl.DataFrame,
    df_col:str,
    title:str,
    filename:str,
    x_col:str="date",
):
    fig, ax = plt.subplots(figsize=(16, 8), dpi=300)
    my_col = df.get_column(df_col)
    my_label = df.get_column("label")
    # Normalize the values to the range [0, 1]
    my_col = (my_col - my_col.min()) / (my_col.max() - my_col.min())
    # # Create masks for values outside the desired range
    # less_than_mask = my_col < 0.2
    # greater_than_mask = my_col > 0.8
    # # set values outside the range to 0.2 and 0.8
    # my_col = my_col.set(less_than_mask, 0.2)
    # my_col = my_col.set(greater_than_mask, 0.8)
    # # Normalize the values to the range [0, 1]
    # my_col = (my_col - my_col.min()) / (my_col.max() - my_col.min())

    ax.plot(df.get_column(x_col), my_col, label=df_col)
    ax.plot(df.get_column(x_col), my_label, label="label")
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filename) 
    plt.close(fig)


for i in track(all_tickers, description="Plotting result..."):
    check_path = Path(i)
    q = pl.read_parquet(check_path).drop_nulls().rename(
        {"savgol_p2_11_rel_diff_5_trend": "label"}
    ).filter(pl.col("date") > datetime(year=2023, month=1, day=1))
    
    # print(q.columns)
    # print(q.head())
    print(check_path.stem)

    
    window = 5
    indicators = [
        # "close",
        f"close_rsi_{window}", 
        # "close_n_momentum{window}",
        f"close_TEMA_reldiff_{window}",
        f"close_relative_zero_lag_macd_{window}",
        # "volume_rsi", 
        # "volume_n_momentum",
        # "volume_TEMA",
        # "volume_relative_zero_lag_macd_{window}",
        f"minus_di_{window}",
        f"plus_di_{window}",
        f"adx_{window}",
        f"cci_{window}",
        f"stochastic_percent_k_{window}",
        f"stochastic_oschilator_{window}",
        # "stochastic_oschilator_slow",
        f"close_perc_slope_{window}",
        f"close_perc_slope_9",
        f"close_perc_slope_14",
    ]
    for ind in indicators:
        print(ind)
        if "rsi" in ind:
            # print("recalc rsi")
            # print(q.get_column(ind).max())
            # print(q.get_column(ind).min())
            # print(q.get_column(ind).median())
            q = q.with_columns(
                pl.when(pl.col(ind) > 70).then(
                    1
                ).when(pl.col(ind) < 30).then(
                    0
                ).otherwise(
                    0.5
                ).alias(ind)
            )
        elif "zero_lag_macd" in ind:
            # print("recalc zero_lag_macd")
            q = q.with_columns(
                pl.when(pl.col(ind) > 0).then(
                    1
                ).when(pl.col(ind) < 0).then(
                    0
                ).otherwise(
                    0.5
                ).alias(ind)
            )
        elif "plus_di_" in ind:
            make_scatter_plot(
                df=q, 
                y_col=ind, 
                title=f"{check_path.stem}_{ind} pos_minus_{window}",
                filename=images_path / f"{check_path.stem}_pos_minus_{window}.png",
                x_col=f"minus_di_{window}"
            )
        elif "perc_slope_" in ind:
            my_col = q.get_column(ind)
            # my_median = my_col.median()
            abs_med_err = (my_col - my_col.median()).abs().median()
            # print((my_median, abs_med_err))
            q = q.with_columns(
                pl.when(pl.col(ind) > (0 + 1.5 * abs_med_err)).then(
                    1
                ).when(pl.col(ind) < (0 - 1.5 * abs_med_err)).then(
                    0
                ).otherwise(
                    0.5
                ).alias(ind)
            )
        # if ind in ["rsi", "n_momentum"]:
        #     for col in cols:
        #         df_col_cols = [df_col for df_col in df_cols if col in df_col]
        #         make_plot(
        #             df=q, 
        #             df_col_cols=df_col_cols, 
        #             title=f"{check_path.stem}_{col}_{ind}",
        #             filename=images_path / f"{check_path.stem}_{col}_{ind}.png"
        #         )
        # else:
        make_plot(
            df=q, 
            df_col=ind, 
            title=f"{check_path.stem}_{ind}",
            filename=images_path / f"{check_path.stem}_{ind}.png"
        )
        
            


    