from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
from features import df_features
import torch
y=df_features['is_fraud']
x=df_features.drop(columns=["is_fraud"])
non_feature_cols = ["user_id", "timestamp", "device", "country", "attack_type", "burst_id", "identity_id"]
x = x.drop(columns=[c for c in non_feature_cols if c in x.columns])
num_cols = ["amount", "hour", "day_of_week", "time_since_prev_txn_hrs",
                "txn_count_24h", "txn_count_7d", "rolling_avg_amount",
                "amount_zscore", "device_mismatch", "country_mismatch"]
cat_cols = ["device_idx", "country_idx"]
class dataset(Dataset):
    def __init__(self, x, y):
        self.numeric = torch.tensor(x[num_cols].values, dtype=torch.float32)
        self.categorical = torch.tensor(x[cat_cols].values, dtype=torch.long)
        self.labels = torch.tensor(y.values, dtype=torch.float32)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.numeric[idx], self.categorical[idx], self.labels[idx]
def load_data(x_train,y_train,x_test,y_test,batch_size=256):
    train_fraud_data=dataset(x_train,y_train)
    train_df=DataLoader(train_fraud_data,batch_size=batch_size,shuffle=True)
    test_fraud_data=dataset(x_test,y_test)
    test_df=DataLoader(test_fraud_data,batch_size=batch_size,shuffle=False)
    return train_df,test_df


