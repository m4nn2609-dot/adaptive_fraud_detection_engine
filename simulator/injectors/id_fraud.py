import random
import faker
import datetime
import numpy as np
import pandas as pd


n_identities=random.randint(100,300)
fake=faker.Faker()
fake_user_data={}
for i in range(n_identities):
    user_id=fake.uuid4()
    user_country=fake.country()
    creation_date=fake.date_between("-5y","today")
    device=random.choice(["android","iphone","desktop(windows)","desktop(mac)","linux"])
    min_transaction=random.randint(20,200)
    max_transaction=random.randint(min_transaction+50,min_transaction+2000)

    cultivation_duration=random.randint(30,180)
    cultivation_end= creation_date + datetime.timedelta(days=cultivation_duration)
    fake_user_data[user_id] = {"user_country": user_country, "creation_date": creation_date, "user_device": device,
                               "user_transaction_min": min_transaction, "user_transaction_max": max_transaction,
                               "cultivation_end":cultivation_end
                               }
all_synthetic_transactions=[]
for fake_user,fake_data in fake_user_data.items():
    num_normal_txns=random.randint(15,40)
    bust_out_num=random.randint(1,5)
    anomalous_range_min,anomalous_range_max=fake_data["user_transaction_min"]*0.1,fake_data["user_transaction_max"]*20
    for i in range(bust_out_num):
        amount = random.uniform(anomalous_range_min, anomalous_range_max)

        offset_days = random.randint(0, 3)
        timestamp = fake_data["cultivation_end"] + datetime.timedelta(days=offset_days)

        all_synthetic_transactions.append({
            "user_id": fake_user,
            "timestamp": timestamp,
            "amount": amount,
            "device": fake_data["user_device"],
            "country": fake_data["user_country"],
            "is_fraud": 1,
            "attack_type": "synthetic_identity",
            "identity_id": fake_user
        })

    for i in range(num_normal_txns):
        centre_point = (fake_data["user_transaction_min"] + fake_data["user_transaction_max"]) / 2
        spread = (fake_data["user_transaction_max"] - fake_data["user_transaction_min"]) / 4

        window_days = (fake_data["cultivation_end"] - fake_data["creation_date"]).days
        offset = random.randint(0, window_days)
        timestamp = fake_data["creation_date"] + datetime.timedelta(days=offset)

        amount = np.random.normal(centre_point, spread)
        amount = np.clip(amount, fake_data["user_transaction_min"], fake_data["user_transaction_max"])

        all_synthetic_transactions.append({
            "user_id": fake_user,
            "timestamp": timestamp,
            "amount": amount,
            "device": fake_data["user_device"],
            "country": fake_data["user_country"],
            "is_fraud": 0,
            "attack_type": "synthetic_identity",
            "identity_id": fake_user
        })
df_synthetic_identity = pd.DataFrame(all_synthetic_transactions)