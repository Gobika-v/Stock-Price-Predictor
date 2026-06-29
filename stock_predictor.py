
import joblib
import numpy as np

# ----------------------------
# Load Best Model
# ----------------------------
try:
    model = joblib.load("models/best_model.pkl")
except FileNotFoundError:
    print("Error: best_model.pkl not found.")
    print("Run train_model.py first.")
    exit()

print("=" * 50)
print("        STOCK PRICE PREDICTOR")
print("=" * 50)

# ----------------------------
# Get User Input
# ----------------------------

try:
    open_price = float(input("Open Price        : "))
    high_price = float(input("High Price        : "))
    low_price = float(input("Low Price         : "))
    volume = int(input("Volume            : "))
    prev_close = float(input("Previous Close    : "))
    ma5 = float(input("Moving Average(5) : "))
    ma10 = float(input("Moving Average(10): "))
    daily_return = float(input("Daily Return      : "))
    volatility = float(input("Volatility        : "))

except ValueError:
    print("\nInvalid input! Please enter numeric values only.")
    exit()

# ----------------------------
# Prepare Input
# ----------------------------

features = np.array([[
    open_price,
    high_price,
    low_price,
    volume,
    prev_close,
    ma5,
    ma10,
    daily_return,
    volatility
]])

# ----------------------------
# Prediction
# ----------------------------

prediction = model.predict(features)

print("\n" + "=" * 50)
print(f"Predicted Closing Price : {prediction[0]:.2f}")
print("=" * 50)
