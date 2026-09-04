import pandas as pd

def make_holdout_split(df,holdout_type,legit_sample_size):
    train_pool=df[df["attack_type"]!=holdout_type]
    test_fraud=df[df["attack_type"]==holdout_type]
    legit_sample=train_pool[train_pool["attack_type"]=="None"].sample(n=legit_sample_size,random_state=42)
    train_pool=train_pool.drop(legit_sample.index)
    test_df=pd.concat([test_fraud,legit_sample],ignore_index=True)
    return train_pool,test_df
def prepare_xy(df):
    y=df["is_fraud"]
    cols_to_drop=["user_id", "timestamp", "device", "country", "attack_type", "burst_id", "identity_id", "is_fraud"]
    x=df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    return x,y