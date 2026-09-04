import pandas as pd
import numpy as np
import faker
import random
import datetime

fake = faker.Faker()

n_bursts = 150
min_accounts, max_accounts = 5, 20
time_window_minutes = 30
amount_range = (5, 50)
shared_device_prob = 0.5

overall_start = datetime.date.today() - datetime.timedelta(days=365)
overall_end = datetime.date.today()
overall_days = (overall_end - overall_start).days

devices = ["android", "iphone", "desktop(windows)", "desktop(mac)", "linux"]

burst_transactions = []

for burst_id in range(n_bursts):
    burst_day_offset = random.randint(0, overall_days)
    burst_start = overall_start + datetime.timedelta(
        days=burst_day_offset,
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59)
    )

    n_accounts = random.randint(min_accounts, max_accounts)
    shared_device = random.random() < shared_device_prob
    ring_device = random.choice(devices)
    ring_country = fake.country()

    for i in range(n_accounts):
        fake_user_id = fake.uuid4()
        offset_minutes = random.randint(0, time_window_minutes)
        timestamp = burst_start + datetime.timedelta(minutes=offset_minutes)
        amount = round(random.uniform(*amount_range), 2)

        if shared_device:
            device = ring_device
            country = ring_country
        else:
            device = random.choice(devices)
            country = fake.country()

        burst_transactions.append({
            "user_id": fake_user_id,
            "timestamp": timestamp,
            "amount": amount,
            "device": device,
            "country": country,
            "is_fraud": 1,
            "attack_type": "coordinated_burst",
            "burst_id": burst_id
        })

df_coordinated_burst = pd.DataFrame(burst_transactions)