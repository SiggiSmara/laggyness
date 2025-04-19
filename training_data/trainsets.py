import polars  as pl
import numpy as np
from rich.progress import track
from datetime import date

from common import (
    tickers,
    trainsets_path,
    combined_path,
)
label = "savgol_p2_11_rel_diff_5_trend"

sel_cols =['stock', 'date', 'open', 'high', 'low', 'close', 'volume', 'true_price', 'minus_di_5', 'minus_di_9', 
'minus_di_14', 'minus_di_20', 'plus_di_5', 'plus_di_9', 'plus_di_14', 'plus_di_20', 'adx_5', 'adx_9', 'adx_14', 'adx_20', 
'cci_5', 'cci_9', 'cci_14', 'cci_20', 'stochastic_percent_k_5', 'stochastic_percent_k_9', 'stochastic_percent_k_14', 
'stochastic_percent_k_20', 'stochastic_oschilator_5', 'stochastic_oschilator_9', 'stochastic_oschilator_14', 
'stochastic_oschilator_20', 'stochastic_oschilator_slow_5', 'stochastic_oschilator_slow_9', 
'stochastic_oschilator_slow_14', 'stochastic_oschilator_slow_20', 'high_rsi_5', 'high_rsi_9', 'high_rsi_14', 'high_rsi_20',
'high_n_momentum_1', 'high_n_momentum_5', 'high_n_momentum_9', 'high_n_momentum_14', 'high_n_momentum_20', 
'high_TEMA_reldiff_5', 'high_TEMA_reldiff_9', 'high_TEMA_reldiff_14', 'high_TEMA_reldiff_20', 
'high_relative_zero_lag_macd_5', 'high_relative_zero_lag_macd_9', 'high_relative_zero_lag_macd_14', 
'high_relative_zero_lag_macd_20', 'low_rsi_5', 'low_rsi_9', 'low_rsi_14', 'low_rsi_20', 'low_n_momentum_1', 
'low_n_momentum_5', 'low_n_momentum_9', 'low_n_momentum_14', 'low_n_momentum_20', 'low_TEMA_reldiff_5', 
'low_TEMA_reldiff_9', 'low_TEMA_reldiff_14', 'low_TEMA_reldiff_20', 'low_relative_zero_lag_macd_5', 
'low_relative_zero_lag_macd_9', 'low_relative_zero_lag_macd_14', 'low_relative_zero_lag_macd_20', 'close_rsi_5', 
'close_rsi_9', 'close_rsi_14', 'close_rsi_20', 'close_n_momentum_1', 'close_n_momentum_5', 'close_n_momentum_9', 
'close_n_momentum_14', 'close_n_momentum_20', 'close_TEMA_reldiff_5', 'close_TEMA_reldiff_9', 'close_TEMA_reldiff_14', 
'close_TEMA_reldiff_20', 'close_relative_zero_lag_macd_5', 'close_relative_zero_lag_macd_9', 
'close_relative_zero_lag_macd_14', 'close_relative_zero_lag_macd_20', 'open_rsi_5', 'open_rsi_9', 'open_rsi_14', 
'open_rsi_20', 'open_n_momentum_1', 'open_n_momentum_5', 'open_n_momentum_9', 'open_n_momentum_14', 'open_n_momentum_20', 
'open_TEMA_reldiff_5', 'open_TEMA_reldiff_9', 'open_TEMA_reldiff_14', 'open_TEMA_reldiff_20', 
'open_relative_zero_lag_macd_5', 'open_relative_zero_lag_macd_9', 'open_relative_zero_lag_macd_14', 
'open_relative_zero_lag_macd_20', 'volume_rsi_5', 'volume_rsi_9', 'volume_rsi_14', 'volume_rsi_20', 'volume_n_momentum_1', 
'volume_n_momentum_5', 'volume_n_momentum_9', 'volume_n_momentum_14', 'volume_n_momentum_20', 'volume_TEMA_reldiff_5', 
'volume_TEMA_reldiff_9', 'volume_TEMA_reldiff_14', 'volume_TEMA_reldiff_20', 'volume_relative_zero_lag_macd_5', 
'volume_relative_zero_lag_macd_9', 'volume_relative_zero_lag_macd_14', 'volume_relative_zero_lag_macd_20', 
'high_perc_slope_5', 'high_perc_slope_9', 'high_perc_slope_14', 'high_perc_slope_20', 'low_perc_slope_5', 
'low_perc_slope_9', 'low_perc_slope_14', 'low_perc_slope_20', 'close_perc_slope_5', 'close_perc_slope_9', 
'close_perc_slope_14', 'close_perc_slope_20', 'open_perc_slope_5', 'open_perc_slope_9', 'open_perc_slope_14', 
'open_perc_slope_20', 'volume_perc_slope_5', 'volume_perc_slope_9', 'volume_perc_slope_14', 'volume_perc_slope_20', 
'savgol_p2_11_rel_diff_5_trend']
# collect a list of parquet files
all_tickers = list(tickers)

# iterate n times:
n = 10
last_dtypes = None
dtypes = None
columns = None
last_columns = None
for i in track(range(n), f"Generating {n} equally distributed trainsets..."):
    # sample parquet files (5% ? 10%)
    parq_rate = 0.05
    sampled_tickers = np.random.choice(all_tickers, size=int(len(all_tickers) * parq_rate), replace=False)

    # put them together in a dataframe
    df_list = []
    df_columns = []
    for ticker in sampled_tickers:
        
        df = pl.scan_parquet(ticker).filter(
            pl.col("date") > date(year=2020, month=1, day=1)
        ).select(sel_cols).rename(
            { label:"label"}
        )
        
        # columns = df.collect_schema().names()
        # dtypes = df.dtypes
        # columns = df.columns
        # if last_dtypes is None:
        #     last_dtypes = dtypes
        #     last_columns = columns
        # else:
        #     for i, dt in enumerate(dtypes):
        #         if dt != last_dtypes[i]:
        #             raise ValueError(f"Data type mismatch: {dt} != {last_dtypes[i]} for column {columns[i]}")
        # print(f"{ticker.stem} has {columns[0]} as first column")
        # print(columns)
        # if len(df_columns) == 0:
        #     df_columns = columns
        # else:
        #     for col1 in df_columns:
        #         if col1 not in columns:
        #             raise ValueError(f"Column mismatch: {col1} not found in incoming columns")
        #     for col2 in columns:
        #         if col2 not in df_columns:
        #             raise ValueError(f"Column mismatch: {col2} not found in existing columns")
        df_list.append(df)
    combined_df = pl.concat(df_list, how="vertical").collect()
    
    # Create a random mask directly
    # print(combined_df.select(pl.len()).collect().item())
    mask = np.random.default_rng().random(combined_df.select(pl.len()).item()) < 0.8  # 80% train, 20% test
    # mask = np.random.default_rng().random(combined_df.select(pl.len()).collect().item()) < 0.8  # 80% train, 20% test
    # mask = np.random.default_rng().random(combined_df.len()) < 0.8  # 80% train, 20% test
    
    
    # Split into train and test sets, first get the test set
    combined_df = combined_df.with_row_index()
    within_df_sampleratio = 0.25
    test_df = combined_df.filter(
        ~pl.Series(mask)
    ).sample(
        fraction=within_df_sampleratio
    ).with_columns(
        pl.lit("testing").alias("data_type")
    )
   
    # then get the training set
    train_df = combined_df.filter(
        pl.Series(mask),
    ).with_columns(
        pl.lit("training").alias("data_type")
    )

    # build the label statistics
    label_counts = train_df.group_by("label").len()
    # label_counts = train_df.group_by("label").count().collect()
    
    # equal sampling for each label
    train_count = round(label_counts.get_column("len").min())
    label_counts = label_counts.with_columns(
        (train_count / pl.col("len")).alias("sampling_ratio")
    )
    
    resampled = []
    for row in label_counts.iter_rows(named=True):
        resampled.append(
            train_df.filter(pl.col("label") == row["label"]).sample(
                fraction=within_df_sampleratio*row["sampling_ratio"]
            ),
        )
    train_resampled_df = pl.concat(
        resampled, 
        how="vertical"
    )
    
    final_df = pl.concat([train_resampled_df, test_df], how="vertical")
    
    # save the dataframe to parquet for further processing
    final_df.write_parquet(trainsets_path / f"trainset_{i}.parquet")
    # print(final_df.group_by(["data_type", "label"]).len())
    # final_df.collect(streaming=True).write_parquet(trainsets_path / f"trainset_{i}.parquet")
