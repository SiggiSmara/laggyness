import polars as pl
from datetime import date
import altair as alt
from rich.progress import track
from pathlib import Path
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
# from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.colors as pColors
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np

from common import (
    find_data_all_lazy,
    data_path,
    centrally_smoothed_path,
    windows,
    trend_windows,
    intermediate_path,
)
# check_path = centrally_smoothed_path / "AMN.parquet"

images_path = intermediate_path / "labeling" / "images"

# find the eligible tickers
all_tickers = list(centrally_smoothed_path.glob("*.parquet"))
# subset = all_tickers[list(range(0,len(all_tickers), 100))]
graph_ending = "png"
ch_width = 1600
ch_height = 600
mark_point_size = 3

# trend_windows = [5, 7, 9]

brush = alt.selection_interval(encodings=['x'])
my_tckrs = [all_tickers[i] for i in range(0,len(all_tickers), 1000)]
my_tckrs = [allx for allx in all_tickers if Path(allx).stem in ["SPY", "AAPL", "AMZN", "GOOGL", "NFLX"]]

rnd_tckrs = list(np.random.choice(all_tickers, 10))
my_tckrs = my_tckrs + rnd_tckrs
my_tckrs = list(set(my_tckrs))
my_tckrs = [Path(x) for x in my_tckrs]
# print(my_tckrs)
# for i in track(range(0,len(all_tickers), 1000), description="Plotting result..."):
#     check_path = Path(all_tickers[i])
for check_path in track(my_tckrs, description="Plotting result..."):
    q = pl.read_parquet(check_path).filter(pl.col("date") > date(year=2023, month=1, day=1))
    
    for window in windows:
        orig_name = f"savgol_p2_{window}"
        rdiff_name = f"{orig_name}_rel_diff"
        asdiff_name = f"original_vs_{orig_name}_absrel_diff"
        sdiff_name = f"original_vs_{orig_name}_rel_diff"
        posneg_name = f"{rdiff_name}_posneg"
        mase_names = [f"{orig_name}_mase{tw}" for tw in trend_windows]
        
        x = q.get_column('date')
        x_num = mdates.date2num(x)
        y = q.get_column('original')
        # ysmooth = q.get_column(orig_name)
        # xtra_trend_windows = [20,40,80]
        # for tw in xtra_trend_windows:
        #     trend_name = f"{rdiff_name}_{tw}_trend"
        #     q = q.with_columns(
        #         pl.col(posneg_name).rolling_mean(window_size=tw, center=True).alias(trend_name),
        #     )
        xtra_trend_windows = []
        for tw in trend_windows + xtra_trend_windows:
            trend_name = f"{rdiff_name}_{tw}_trend"
            # unique_vals = np.unique(q.get_column(trend_name).to_numpy(), return_counts=True)
            # print(unique_vals)
            label_name = f"{rdiff_name}_label{tw}"
            ylabel = q.get_column(trend_name)
            # unique_vals = np.unique(ylabel, return_counts=True)
            # print(unique_vals)
            # Create segments
            points = np.array([x_num, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            # Build collection
            # cmap = pColors.ListedColormap(["red", "blue", "green"])
            # bounds = [-0.1, 0.25, 0.75, 1.1]  # Slightly around the target values
            # norm = pColors.BoundaryNorm(bounds, cmap.N)

            cmap = plt.cm.RdYlGn
            norm = pColors.Normalize(vmin=0, vmax=1)    

            plot_pt = 4
            lc = LineCollection(segments, cmap=cmap, norm=norm)
            lc.set_array(ylabel)
            lc.set_linewidth(plot_pt)
            lc.set_alpha(0.5)
            
            # Plot
            fig, ax = plt.subplots(figsize=(16, 8), dpi=300) #, facecolor="#f0f0f0"
            # ax.set_facecolor("#000000")
            ax.scatter(x, y, s=plot_pt*2, alpha=0.5, c=ylabel, cmap=cmap, norm=norm)
            ax.add_collection(lc)
            ax.set_xlim(x.min(), x.max())
            ax.set_ylim(y.min(), y.max())
            ax.set(title=f'Sustained {tw}-trading day regimes of {check_path.stem}')
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, ticks=[0, 0.5, 1])
            cbar.ax.set_yticklabels(["Bearish", "Undefined", "Bullish"])

            # Format the x-axis as dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(images_path / f"{check_path.stem}_regimes_{window}_{tw}.{graph_ending}")
            plt.close(fig)
    

    # the approach from https://medium.com/@spencer13luck/time-series-regime-analysis-in-python-ffdc7aa005dd 
    # is not generalizable for individual stocks
    # n_regimes = 2
    # # mod_kns = sm.tsa.MarkovRegression(q.get_column("original_diff").to_pandas(), k_regimes=n_regimes, trend='n', switching_variance=True)
    # my_data = q.get_column("original_diff").to_pandas()
    # mod_kns = sm.tsa.MarkovRegression(my_data.iloc[1:], k_regimes=n_regimes, exog=my_data.iloc[:-1], switching_variance=True)
    # # mod_akns = sm.tsa.MarkovAutoregression(q.get_column("original").to_pandas(), k_regimes=n_regimes, order=2, trend="ct", switching_variance=False)
    # # mod_kns = sm.tsa.MarkovRegression(q.get_column(rdiff_name).to_pandas(), k_regimes=3, trend='n', switching_variance=True)
    # res_kns = mod_kns.fit()
    # # res_akns = mod_akns.fit()
    # print(check_path.stem)
    # # print(res_kns.summary())

    # fig, axes = plt.subplots(n_regimes, figsize=(10,7))
    # for i in range(n_regimes):
    #     ax = axes[i]
    #     ax.plot(res_kns.smoothed_marginal_probabilities[i])
    #     ax.grid(False)
    #     ax.set(title=f'Smoothed probability of the {i}-regime')
   
    # fig.tight_layout()
    # ax.grid(False)
    # fig.savefig(images_path / f"{check_path.stem}_markov.{graph_ending}")
    # plt.close(fig)

    # low_var = list(res_kns.smoothed_marginal_probabilities[0])
    # high_var = list(res_kns.smoothed_marginal_probabilities[1])
    # # mid_var = list(res_kns.smoothed_marginal_probabilities[2])

        
    # regime_list = []
    # threshold = 0.8
    # for i in range(0, len(low_var)):
    #     if low_var[i] > threshold:
    #         regime_list.append(0)
    #     elif high_var[i] > threshold:
    #         regime_list.append(1)
    #     else:
    #         regime_list.append(0.5)
    
    # regime_df = pd.DataFrame()
    # regime_df['regimes'] = regime_list
    # # print(regime_df.head(20))

    # sns.set_theme(font_scale=1.5)
    # df = q.to_pandas()
    # df = df.iloc[len(df)-len(regime_list):]
    # regimes = (pd.DataFrame(regime_list, columns=['regimes'], index=df.index)
    #         .join(df, how='inner')
    #         .reset_index(drop=False)
    #         # .rename(columns={'index':'Date'})
    #         )
    # # print(regimes.head(20))

    # import warnings
    # warnings.filterwarnings("ignore")
    # colors = 'green', 'yellow', 'red', 'blue'
    # sns.set_style("whitegrid")
    # order = [0, 0.5, 1 ]
    # fg = sns.FacetGrid(data=regimes, hue='regimes', hue_order=order,
    #                 palette=colors, aspect=1.31, height=12)
    # fg.map(plt.scatter, 'date', "original", alpha=0.8).add_legend()
    # sns.despine(offset=10)
    # fg.figure.suptitle(f'{check_path.stem} regimes', fontsize=24, fontweight='demi')
    # fg.savefig(images_path / f"{check_path.stem}_regimes.{graph_ending}")
        