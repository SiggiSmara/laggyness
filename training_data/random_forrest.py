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


from common import (
    trainsets_path,
)

sel_types = [
    "minus_di", "plus_di", "adx", "cci", "stochastic_oschilator",
    "stochastic_oschilator_slow", "high_rsi", "high_n_momentum",
    "high_TEMA_reldiff", "high_relative_zero_lag_macd", "low_rsi",
    "low_n_momentum", "low_TEMA_reldiff", "low_relative_zero_lag_macd",
    "close_rsi", "close_n_momentum", "close_TEMA_reldiff",
    "close_relative_zero_lag_macd", "open_rsi", "open_n_momentum",
    "open_TEMA_reldiff", "open_relative_zero_lag_macd", "volume_rsi",
    "volume_n_momentum", "volume_TEMA_reldiff", "volume_relative_zero_lag_macd",
    "high_perc_slope", "low_perc_slope", "close_perc_slope",
    "open_perc_slope", "volume_perc_slope"
]

sel_types2 = np.random.choice(sel_types, 10, replace=False)

sel_cols =['open', 'high', 'low', 'close', 'volume', 'true_price', 'minus_di_5', 'minus_di_9', 
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
'open_perc_slope_20', 'volume_perc_slope_5', 'volume_perc_slope_9', 'volume_perc_slope_14', 'volume_perc_slope_20', ]


sel_types = [
    "minus_di", "plus_di", "adx", "cci", "stochastic_oschilator",
    "stochastic_oschilator_slow", "high_rsi", "high_n_momentum",
    "high_TEMA_reldiff", "high_relative_zero_lag_macd", "low_rsi",
    "low_n_momentum", "low_TEMA_reldiff", "low_relative_zero_lag_macd",
    "close_rsi", "close_n_momentum", "close_TEMA_reldiff",
    "close_relative_zero_lag_macd", "open_rsi", "open_n_momentum",
    "open_TEMA_reldiff", "open_relative_zero_lag_macd", "volume_rsi",
    "volume_n_momentum", "volume_TEMA_reldiff", "volume_relative_zero_lag_macd",
    "high_perc_slope", "low_perc_slope", "close_perc_slope",
    "open_perc_slope", "volume_perc_slope"
]

sel_types2 = np.random.choice(sel_types, 10, replace=False)
sel_types2 = [
    "minus_di", "plus_di", "adx", "cci", "stochastic_oschilator",
    "close_n_momentum", "close_perc_slope", "close_rsi", "close_TEMA_reldiff", "close_relative_zero_lag_macd",
    "volume_n_momentum", "volume_perc_slope", "volume_rsi", "volume_TEMA_reldiff", "volume_relative_zero_lag_macd",
    "high_n_momentum", "high_perc_slope", "high_rsi", "high_TEMA_reldiff", "high_relative_zero_lag_macd",
    "low_n_momentum", "low_perc_slope", "low_rsi", "low_TEMA_reldiff", "low_relative_zero_lag_macd",
    "open_n_momentum", "open_perc_slope", "open_rsi", "open_TEMA_reldiff", "open_relative_zero_lag_macd",
]
sel_cols2 = []
for stype in sel_types2:
    # print(stype)
    sel_cols2.append(np.random.choice([x for x in sel_cols if stype in x], 1)[0])


sel_type = "perc_slope"
sel_cols2 = [x for x in sel_cols if sel_type in x] 
sel_cols2 = ['avg_perc_slope_5', 'avg_perc_slope_9', 'avg_perc_slope_14', 'avg_perc_slope_20']

best_cols = [
    'minus_di_5',
    'minus_di_9',
    # 'plus_di_5',
    # 'adx_20',
    'cci_5',
    'cci_9',
    # 'stochastic_oschilator_5',
    # 'close_n_momentum_5',
    # 'close_perc_slope_9',
    # 'close_perc_slope_14',
    'close_rsi_5',
    # 'close_TEMA_reldiff_5',
    # 'close_relative_zero_lag_macd_5',
    # 'high_n_momentum_5',
    # 'high_perc_slope_9',
    # 'high_perc_slope_14',
    # 'high_rsi_5',
    # 'high_TEMA_reldiff_5',
    # 'high_relative_zero_lag_macd_5',
    # 'low_n_momentum_5',
    # 'low_perc_slope_9',
    # 'low_perc_slope_14',
    # 'low_rsi_5',
    # 'low_TEMA_reldiff_5',
    # 'low_relative_zero_lag_macd_5',
    # 'open_n_momentum_5',
    # 'open_perc_slope_14',
    # 'open_rsi_5',
    # 'open_TEMA_reldiff_5',
    # 'open_relative_zero_lag_macd_5',
    # 'stochastic_oschilator_ratio_5',
    'stochastic_percent_k_5',
    # 'avg_perc_slope_9',
    
]
sel_cols2 = best_cols
# windows = [5, 9, 14, 20]
# sel_cols2 = [f"stochastic_oschilator_ratio_{wi}" for wi in windows]
print(sel_cols2)
print(len(sel_cols2))
# sel_cols2 = np.random.choice(sel_cols, 20, replace=False)
# sel_cols2 = ['open', 'high', 'low', 'close', 'volume', 'true_price', 'minus_di_5', 'minus_di_9',

# collect a list of parquet files
# iterate n times:
# sample parquet files (5% ? 10%)
# put them together in a dataframe
# build the label statistics
# decide on the overall sampling ratio for the training data
# split of test data (10-20% of the number of trainining data), this is not normalized
# for training data sample each label so that the sampled data is equally distributed
# join training and test data in a dataframe and label the data with training or testing
# save the dataframe to parquet for further processing

# 
train_tickers = [ x for x in (trainsets_path).glob("*.parquet")]

# Parameters for random sampling
n_iterations = 5  # Number of random samples
sample_size = 10000  # Size of each sample
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
    # print(q.schema)
    # q = q.filter(
    #     ~pl.any_horizontal(pl.selectors.numeric().is_infinite())
    # )
    train_set = q.filter(pl.col("data_type") == "training")
    test_set = q.filter(pl.col("data_type") == "testing")

    X_train = train_set.select(sel_cols2).to_numpy()
    # print(X_train[~np.isfinite(X_train)])
    # X_train[~np.isfinite(X_train)] = 0
    y_train = [ str(x) for x in train_set.get_column("label").to_list() ]
    # y_train = train_set.get_column("label").to_numpy()

    X_test = test_set.select(sel_cols2).to_numpy()
    # X_test[~np.isfinite(X_test)] = 0
    y_test = [str(x) for x in test_set.get_column("label").to_list()]
    # y_test = test_set.get_column("label").to_numpy()

    # print(X_train.shape, y_train.shape)
    
    for i in range(1):
    
        # Train the model
        rf = RandomForestClassifier(random_state=i)
        # rf = RandomForestRegressor(random_state=i)
        rf.fit(X_train, y_train)
        
        # Predictions
        y_pred = rf.predict(X_test)
        y_probs = rf.predict_proba(X_test)
        
        # # Evaluate metrics
        r2 = r2_score(y_test, y_pred)
        # mse = mean_squared_error(y_test, y_pred)
        # rmse = np.sqrt(mse)
        accuracy = accuracy_score(y_test, y_pred)
        ap_score = average_precision_score(y_test, y_probs, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        auc_score = roc_auc_score(y_test, y_probs, multi_class='ovr')
        results.append({
            'iteration': i, 
            # 'rmse': rmse,
            'accuracy': accuracy, 
            'average_precision': ap_score, 
            'f1_score': f1,
            'auc_score': auc_score
        })

        f_i = list(zip(sel_cols2,rf.feature_importances_))
        f_i.sort(key = lambda x : x[1], reverse=True)
        print(f_i)

    # Aggregate results
    # avg_rmse = np.mean([r['rmse'] for r in results])
    avg_accuracy = np.mean([r['accuracy'] for r in results])
    avg_ap_score = np.mean([r['average_precision'] for r in results])
    avg_f1 = np.mean([r['f1_score'] for r in results])
    avg_auc = np.mean([r['auc_score'] for r in results])

    # print(f"Average RMSE: {avg_rmse:.2f}")
    print(f"Average Accuracy: {avg_accuracy:.2f}")
    print(f"Average Precision Score: {avg_ap_score:.2f}")
    print(f"Average F1 Score: {avg_f1:.2f}")
    print(f"Average AUC Score: {avg_auc:.2f}")





# X, y = make_classification(n_classes=6, weights=[0.6, 0.1, 0.1, 0.1, 0.05, 0.05], n_informative=10, n_samples=1000, random_state=42)

# # Split the dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Train the Random Forest model
# rf = RandomForestClassifier(random_state=42)
# rf.fit(X_train, y_train)

# # Make predictions
# y_pred = rf.predict(X_test)

# # Evaluation metrics
# print("Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred))

# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# # Optional: Check specific metrics like F1-score or balanced accuracy
# f1 = f1_score(y_test, y_pred, average='weighted')
# accuracy = accuracy_score(y_test, y_pred)
# print(f"\nF1-Score (weighted): {f1:.2f}")
# print(f"Accuracy: {accuracy:.2f}")



# # different metrics

# # Get probabilities for each class (needed for AUC and AP)
# y_probs = rf.predict_proba(X_test)

# # For multiclass AUC, use the 'ovr' (one-vs-rest) approach
# auc_score = roc_auc_score(y_test, y_probs, multi_class='ovr')
# print(f"Multiclass AUC Score: {auc_score:.2f}")

# # Average Precision Score (binary or multiclass)
# ap_score = average_precision_score(y_test, y_probs, average='weighted')
# print(f"Weighted Average Precision Score: {ap_score:.2f}")