
import polars as pl
from rich.progress import track
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from common import (
    find_data_all_lazy,
    data_path,
    indicator_path
)
# check_path = centrally_smoothed_path / "AMN.parquet"

images_path = data_path / "intermediate" / "indicators" / "images"
images_path.mkdir(exist_ok=True, parents=True)

# find the eligible tickers
all_tickers = list(indicator_path.glob("*.parquet"))

cols = [
    "high",
    "low",
    "close",
    "open",
    "volume"
]


def make_plot(
    *,
    df:pl.DataFrame,
    df_col_cols:list,
    title:str,
    filename:str,
):
    fig, ax = plt.subplots(figsize=(16, 8), dpi=300)
    for df_col in df_col_cols:
        ax.plot(df.get_column("date"), df.get_column(df_col), label=df_col)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filename) 
    plt.close(fig)


for i in track(all_tickers, description="Plotting result..."):
    check_path = Path(i)
    q = pl.read_parquet(check_path).drop_nulls()
    
    # print(q.columns)
    # print(q.head())
    print(check_path.stem)

    excl_cols = ["close", "volume"]

    indicators = [
        # "close",
        "close_rsi", 
        "close_n_momentum",
        "close_TEMA",
        "close_relative_zero_lag_macd",
        "volume_rsi", 
        "volume_n_momentum",
        "volume_TEMA",
        "volume_relative_zero_lag_macd",
        "minus_di",
        "plus_di",
        "adx",
        "cci",
        # "stochastic_percent_k",
        "stochastic_oschilator",
        "stochastic_oschilator_slow",
        "slope_perc"
    ]
    p = q.select(
        [col for col in q.columns if any(ind in col for ind in indicators) and "9" in col  ], 
    ).corr()
    fig, ax = plt.subplots(figsize=(16, 8), dpi=300)
    sns.heatmap(
        p,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        xticklabels=p.columns,
        yticklabels=p.columns,
        ax=ax
    )
    fig.tight_layout()
    fig.savefig(images_path / f"{check_path.stem}_correlation_heatmap.png")
    plt.close(fig)

    # indicators = [
    #     "rsi", 
    #     "n_momentum",
    #     "true_price",
    #     "minus_di",
    #     "plus_di",
    #     "adx",
    #     "cci",
    #     "stochastic_percent_k",
    #     "stochastic_oschilator",
    #     "stochastic_oschilator_slow",
    # ]

    for ind in indicators:
        print(ind)
        df_cols = [col for col in q.columns if ind in col]
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
            df_col_cols=df_cols, 
            title=f"{check_path.stem}_{ind}",
            filename=images_path / f"{check_path.stem}_{ind}.png"
        )
        
            


    