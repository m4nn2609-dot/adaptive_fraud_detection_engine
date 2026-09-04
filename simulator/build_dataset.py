import base_gen,simulator.injectors.fraud_ring,simulator.injectors.id_fraud
from simulator.base_gen import df_transactions
from simulator.injectors.fraud_ring import df_coordinated_burst
from simulator.injectors.id_fraud import df_synthetic_identity
import os
import pandas as pd
os.makedirs("data",exist_ok=True)

def generate_legit_transactions():
    df_transactions.to_parquet("data/legit.parquet")
    return base_gen.df_transactions
def generate_coordinated_burst():
    df_coordinated_burst.to_parquet('data/coordinated_burst.parquet')
    return simulator.injectors.fraud_ring.df_coordinated_burst
def generate_synthetic_identity():
    df_synthetic_identity.to_parquet("data/synthetic_identity.parquet")
    return simulator.injectors.id_fraud.df_synthetic_identity


generate_synthetic_identity();generate_coordinated_burst();generate_legit_transactions()
df_legit = pd.read_parquet("data/legit.parquet")
df_burst = pd.read_parquet("data/coordinated_burst.parquet")
df_synthetic = pd.read_parquet("data/synthetic_identity.parquet")
df_legit["attack_type"] = "None"
df_full = pd.concat([df_legit, df_burst, df_synthetic], ignore_index=True)
df_full.to_parquet("data/full_dataset.parquet")