import pandas as pd
import numpy as np
import faker
from faker.providers import BaseProvider as provider
import random
import json
import datetime
fake=faker.Faker()
class my_provider(provider):
    def device(self):
        return random.choice(["android","iphone","desktop(windows)","desktop(mac)","linux"])
    def range(self):
        lo=random.randint(2,200)
        hi=random.randint(lo+50,lo+2000)
        return lo,hi
base_provider=my_provider(fake)
user_data={}
for i in range(2500):
    user_id=fake.uuid4()
    user_country=fake.country()
    creation_date=fake.date_between("-5y","today")#date as a datetime
    user_device=base_provider.device()
    user_transaction_min,user_transaction_max=base_provider.range()

    user_data[user_id]={"user_country":user_country,"creation_date":creation_date,"user_device":user_device,
                        "user_transaction_min":user_transaction_min,"user_transaction_max":user_transaction_max
                        }

df=pd.DataFrame.from_dict(user_data,orient='index')
dev=0.3
all_transactions=[]
today=datetime.date.today()

for user,profile in user_data.items():
    num=random.randint(20,100)
    days_active = (today-profile["creation_date"]).days
    for i in range(num):

        centre_point=(profile["user_transaction_max"]+profile["user_transaction_min"])/2
        spread=(profile["user_transaction_max"]-profile["user_transaction_min"])/4
        offset=random.randint(0,days_active)
        timestamp=profile["creation_date"]+datetime.timedelta(days=offset)
        amount = np.random.normal(centre_point, spread)
        amount = np.clip(amount, profile["user_transaction_min"], profile["user_transaction_max"])

        if random.random()<dev:
            device=random.choice(["android", "iphone", "desktop(windows)", "desktop(mac)", "linux"])
            country=fake.country()
        else:
            device = profile["user_device"]
            country = profile["user_country"]
        all_transactions.append({
            "user_id": user,
            "timestamp": timestamp,
            "amount": amount,
            "device": device,
            "country": country,
            "is_fraud": 0,
            "attack_type":"None",
            "days_active":days_active
        })

df_transactions = pd.DataFrame(all_transactions)

