import pandas as pd
import numpy as np
from pandas.core.common import random_state

df_full = pd.read_parquet("simulator/data/full_dataset.parquet")
def engineer_features(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # time-based features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # time since previous transaction, per user (in hours)
    df["prev_timestamp"] = df.groupby("user_id")["timestamp"].shift(1)
    df["time_since_prev_txn_hrs"] = (
        (df["timestamp"] - df["prev_timestamp"]).dt.total_seconds() / 3600
    )
    df["time_since_prev_txn_hrs"] = df["time_since_prev_txn_hrs"].fillna(-1)  # first txn per user

    # rolling velocity: count of txns in last 24h and 7d, per user
    df = df.set_index("timestamp")
    df["txn_count_24h"] = (
        df.groupby("user_id")["amount"]
        .rolling("24h").count()
        .reset_index(level=0, drop=True)
    )
    df["txn_count_7d"] = (
        df.groupby("user_id")["amount"]
        .rolling("7d").count()
        .reset_index(level=0, drop=True)
    )
    df = df.reset_index()

    # rolling average amount over user's last 5 txns (excluding current)
    df["rolling_avg_amount"] = (
        df.groupby("user_id")["amount"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    df["rolling_avg_amount"] = df["rolling_avg_amount"].fillna(df["amount"])

    # deviation from user's own historical average (expanding, excludes current txn)
    df["user_hist_mean"] = df.groupby("user_id")["amount"].transform(lambda x: x.shift(1).expanding().mean())
    df["user_hist_std"] = df.groupby("user_id")["amount"].transform(lambda x: x.shift(1).expanding().std())
    df["amount_zscore"] = (df["amount"] - df["user_hist_mean"]) / df["user_hist_std"]
    df["amount_zscore"] = df["amount_zscore"].fillna(0).replace([np.inf, -np.inf], 0)

    # device/country deviation from user's most common historical value
    df["most_common_device"] = df.groupby("user_id")["device"].transform(
        lambda x: x.shift(1).mode().iloc[0] if not x.shift(1).mode().empty else x.iloc[0]
    )
    df["device_mismatch"] = (df["device"] != df["most_common_device"]).astype(int)

    df["most_common_country"] = df.groupby("user_id")["country"].transform(
        lambda x: x.shift(1).mode().iloc[0] if not x.shift(1).mode().empty else x.iloc[0]
    )
    df["country_mismatch"] = (df["country"] != df["most_common_country"]).astype(int)

    # categorical encoding (integer indices for embedding layers)
    df["device_idx"] = pd.factorize(df["device"])[0]
    df["country_idx"] = pd.factorize(df["country"])[0]

    # cleanup helper columns
    df = df.drop(columns=["prev_timestamp", "most_common_device", "most_common_country"])

    return df

df_features=engineer_features(df_full)
df_features.to_parquet("simulator/data/features.parquet")





