import polars as pl
from rich.progress import track
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score,
    accuracy_score, average_precision_score, roc_auc_score,
    r2_score, mean_squared_error
)
from sklearn.model_selection import train_test_split

# Example dataset (replace with your data)
from sklearn.datasets import make_classification

# Example dataset (replace with your actual data)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from common import (
    trainsets_path,
)

def make_boxplots(df: pl.DataFrame, x_col: str, y_cols: list[str], title: str, filename: str):
    pdf = df #.to_pandas()
    fig, axes = plt.subplots(ncols=len(y_cols), figsize=(16, 8), dpi=300)
    for i, col in enumerate(y_cols):
        sns.boxplot(x=pdf[x_col], y=pdf[col], ax=axes[i])
        axes[i].set_title(col)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)

def make_histogram(df: pl.DataFrame, x_col: str, y_cols: list[str], title: str, filename: str):
    # print(df.schema)
    pdf = df #.filter((pl.col(x_col) <= 0.2) | (pl.col(x_col) >= 0.8) )
    fig, axes = plt.subplots(ncols=len(y_cols), figsize=(16, 8), dpi=300)
    for i, col in enumerate(y_cols):
        sns.histplot(data=pdf, x=col, hue=x_col, ax=axes[i], kde=True)
        axes[i].set_title(col)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)

def make_plot(
    *,
    df:pl.DataFrame,
    x_col:str,
    y_cols:list[str],
    title:str,
    filename:str,
):
    fig, ax = plt.subplots(ncols=len(y_cols), figsize=(16, 8), dpi=300)
    for i, df_col in enumerate(y_cols):
        # ax[i].(df.get_column(x_col), df.get_column(df_col), label=df_col)
        ax[i].violinplot(df.get_column(df_col),
                  showmeans=False,
                  showmedians=True)
        ax[i].set_title(df_col)
    # ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filename) 
    plt.close(fig)



# 
train_tickers = [ x for x in (trainsets_path).glob("*.parquet")]


results = []  # To store performance metrics

windows = [5, 9, 14, 20]

for tr_path in track(train_tickers, "training...."):
    q = pl.read_parquet(tr_path).with_columns(
        (
            pl.col(f"stochastic_oschilator_{wi}")/(pl.col(f"stochastic_oschilator_slow_{wi}") + 1e-7)
        ).alias(f"stochastic_oschilator_ratio_{wi}") for wi in windows
    ).with_columns(
        ((
            pl.col(f'close_perc_slope_{wi}') + pl.col(f'open_perc_slope_{wi}') + pl.col(f'high_perc_slope_{wi}') + pl.col(f'low_perc_slope_{wi}')
        )/ 4).alias(f'avg_perc_slope_{wi}') for wi in windows
    ).drop_nulls(
    ).drop_nans(
    ).filter(
        ~pl.any_horizontal(pl.selectors.numeric().is_infinite())
    )

    cols = q.select(pl.selectors.numeric()).columns
    cols = [ c for c in cols if c not in ["index", "label"] ]
    corr_cols = [ f"corr_{c}" for c in cols ]
    p = q.select(
        pl.corr("label", pl.col(c)).alias(f"corr_{c}") for c in cols 
    ).drop_nans()
    
    r = q.with_columns(pl.col("stock").cast(pl.Categorical)).group_by("stock").agg(
        pl.corr("label", pl.col(c)).alias(f"corr_{c}") for c in cols
    ).drop_nans()

    long_r = r.unpivot(
        index = "stock",
        variable_name="corr_cols",
        value_name="corr_value"
    )


    long_p = p.unpivot(
        on=None,  # All columns will be unpivoted
        variable_name="column_name",  # Name for the column containing original column names
        value_name="value"  # Name for the column containing values
    ).sort(
        by="value", descending=True
    )

    long_p = pl.concat(
        [
            long_p.top_k(3, by="value"),
            # long_p.bottom_k(3, by="value"),
        ]
    )
    # print(long_p.head(12))
    best_cols = [x[5:] for x in long_p.get_column("column_name").to_list()]

    make_boxplots( df=q, x_col="label", y_cols=best_cols, title=tr_path.stem, filename=f"{tr_path.stem}_boxplot.png")
    make_histogram( df=q, x_col="label", y_cols=best_cols, title=tr_path.stem, filename=f"{tr_path.stem}_histogram.png")

    best_of_best = ["corr_close_rsi_5", "corr_minus_di_5"]

    # First find the max value and index for each category1
    long_r = long_r.with_row_index("row_idx")
    max_indices = long_r.group_by("stock").agg(
        pl.col("corr_value").arg_max().alias("max_idx")
    )

    # Then join back to get the full row with the maximum value
    long_r_best_pos = long_r.sort(
        ["stock", pl.col("corr_value")],
        descending=[False, True]
    ).group_by("stock").agg([
        pl.col("corr_cols").first(),
        pl.col("corr_value").first(),
    ]).sort(
        by="corr_value", 
        descending=True
    )
    long_r_best_pos.write_csv(
        f"long_r_best_pos_{tr_path.stem}.csv"
    )
    
    print(long_r_best_pos.head())

    
    