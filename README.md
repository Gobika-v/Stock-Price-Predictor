# 📈 Stock Price Predictor

A Machine Learning project that predicts stock closing prices using multiple regression algorithms and provides an interactive web interface built with Streamlit.

---

# 🚀 Features

* Generate 500+ days of synthetic stock market data
* Perform feature engineering
* Train multiple regression models
* Compare model performance
* Predict stock closing prices
* Interactive Streamlit dashboard
* Download prediction results as CSV
* Save and load trained models

---

# 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Plotly
* Streamlit
* Joblib

---

# 📂 Project Structure

```text
Stock-Price-Predictor/
│
├── data/
│   └── stock_data.csv
│
├── models/
│   ├── Linear_Regression.pkl
│   ├── Random_Forest.pkl
│   ├── Gradient_Boosting.pkl
│   ├── best_model.pkl
│   └── model_results.csv
│
├── images/
│   ├── model_comparison.png
│   └── actual_vs_predicted.png
│
├── data_generator.py
├── train_model.py
├── stock_predictor.py
├── app.py
├── utils.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 📊 Dataset

The project generates **500+ days** of synthetic stock market data containing:

* Date
* Open Price
* High Price
* Low Price
* Close Price
* Volume

---

# ⚙️ Feature Engineering

The following features are created before training:

* Previous Close Price
* 5-Day Moving Average (MA5)
* 10-Day Moving Average (MA10)
* Daily Return
* Volatility

---

# 🤖 Machine Learning Models

1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

The best-performing model is automatically saved as:

```
models/best_model.pkl
```

---

# 📈 Evaluation Metrics

The models are evaluated using:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

---



# ▶️ Usage

Generate the dataset:


python data_generator.py


Train the models:


python train_model.py


Run the CLI application:


python stock_predictor.py


Launch the Streamlit web app:


streamlit run app.py



