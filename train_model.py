import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------
# Create models folder
# ----------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("images", exist_ok=True)

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("data/stock_data.csv")

print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Info")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

# ----------------------------
# Feature Engineering
# ----------------------------

# Previous Day Close
df["Prev_Close"] = df["Close"].shift(1)

# Moving Average (5 days)
df["MA5"] = df["Close"].rolling(window=5).mean()

# Moving Average (10 days)
df["MA10"] = df["Close"].rolling(window=10).mean()

# Daily Return
df["Return"] = df["Close"].pct_change()

# Volatility
df["Volatility"] = df["Return"].rolling(window=5).std()

# Remove NaN values
df.dropna(inplace=True)

# ----------------------------
# Features and Target
# ----------------------------

X = df[
    [
        "Open",
        "High",
        "Low",
        "Volume",
        "Prev_Close",
        "MA5",
        "MA10",
        "Return",
        "Volatility",
    ]
]

y = df["Close"]

# ----------------------------
# Train Test Split
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ----------------------------
# Models
# ----------------------------

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    ),
}

results = []

best_model = None
best_score = -999

# ----------------------------
# Training Loop
# ----------------------------

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    mae = mean_absolute_error(y_test, prediction)

    mse = mean_squared_error(y_test, prediction)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, prediction)

    print("MAE :", round(mae, 3))
    print("RMSE:", round(rmse, 3))
    print("R2  :", round(r2, 3))

    results.append([name, mae, rmse, r2])

    joblib.dump(
        model,
        f"models/{name.replace(' ','_')}.pkl"
    )

    if r2 > best_score:
        best_score = r2
        best_model = model

# ----------------------------
# Save Best Model
# ----------------------------

joblib.dump(best_model, "models/best_model.pkl")

print("\nBest Model Saved Successfully!")

# ----------------------------
# Results Table
# ----------------------------

result_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "MAE",
        "RMSE",
        "R2"
    ]
)

print("\nModel Comparison")
print(result_df)

# ----------------------------
# Save Results
# ----------------------------

result_df.to_csv(
    "models/model_results.csv",
    index=False
)

# ----------------------------
# Plot R2 Comparison
# ----------------------------

plt.figure(figsize=(8, 5))

plt.bar(result_df["Model"], result_df["R2"])

plt.title("Model Comparison (R² Score)")
plt.ylabel("R² Score")

plt.tight_layout()

plt.savefig("images/model_comparison.png")

plt.show()

# ----------------------------
# Actual vs Predicted
# ----------------------------

prediction = best_model.predict(X_test)

plt.figure(figsize=(7, 7))

plt.scatter(
    y_test,
    prediction
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")

plt.tight_layout()

plt.savefig("images/actual_vs_predicted.png")

plt.show()

print("\nTraining Completed Successfully!")
