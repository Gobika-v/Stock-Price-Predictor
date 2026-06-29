import pandas as pd
import numpy as np

np.random.seed(42)

days = 500

dates = pd.date_range("2023-01-01", periods=days)

price = 100

open_prices = []
high_prices = []
low_prices = []
close_prices = []
volumes = []

for i in range(days):

    open_price = price + np.random.uniform(-2, 2)

    high_price = open_price + np.random.uniform(0, 5)

    low_price = open_price - np.random.uniform(0, 5)

    close_price = np.random.uniform(low_price, high_price)

    volume = np.random.randint(100000, 1000000)

    open_prices.append(round(open_price, 2))
    high_prices.append(round(high_price, 2))
    low_prices.append(round(low_price, 2))
    close_prices.append(round(close_price, 2))
    volumes.append(volume)

    price = close_price

df = pd.DataFrame({
    "Date": dates,
    "Open": open_prices,
    "High": high_prices,
    "Low": low_prices,
    "Close": close_prices,
    "Volume": volumes
})

df.to_csv("data/stock_data.csv",  index=False)

print("Dataset Generated Successfully!")
print(df.head())
