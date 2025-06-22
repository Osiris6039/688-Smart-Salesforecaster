
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import datetime

# --- User Auth ---
def login():
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
        else:
            st.sidebar.error("Invalid credentials")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()

# --- Main App ---
st.title("📈 Smart AI Forecasting App")
DATA_PATH = os.path.join(os.path.dirname(__file__), "stored_data.csv")

def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH, parse_dates=["date"])
    return pd.DataFrame(columns=["date", "sales", "customers", "weather", "addons"])

def save_data(data):
    data.sort_values("date", inplace=True)
    data.to_csv(DATA_PATH, index=False)

def manual_input_form():
    with st.form("input_form"):
        date = st.date_input("Date")
        sales = st.number_input("Sales", step=100.0)
        customers = st.number_input("Customers", step=1)
        weather = st.selectbox("Weather", ["sunny", "rainy", "cloudy"])
        addons = st.number_input("Add-on Sales", step=100.0)
        submitted = st.form_submit_button("Add Entry")
        if submitted:
            new_entry = pd.DataFrame([[date, sales, customers, weather, addons]],
                                     columns=["date", "sales", "customers", "weather", "addons"])
            current_data = load_data()
            updated = pd.concat([current_data, new_entry], ignore_index=True)
            save_data(updated)
            st.success("Entry added and data updated!")

# --- Upload CSV ---
uploaded_file = st.file_uploader("Upload CSV (date, sales, customers, weather, addons)", type="csv")
if uploaded_file:
    df_upload = pd.read_csv(uploaded_file, parse_dates=["date"])
    save_data(df_upload)
    st.success("CSV uploaded and data updated!")

# --- Manual Input ---
manual_input_form()

# --- Display Data ---
data = load_data()
if not data.empty:
    st.subheader("🗂 Existing Data")
    st.dataframe(data)

    if st.button("Clear All Data"):
        save_data(pd.DataFrame(columns=data.columns))
        st.success("All data cleared!")
        st.stop()

    # Delete Row
    index_to_delete = st.number_input("Enter Row Index to Delete", min_value=0, max_value=len(data)-1, step=1)
    if st.button("Delete Row"):
        data.drop(index=index_to_delete, inplace=True)
        save_data(data)
        st.success(f"Row {index_to_delete} deleted.")

# --- Forecasting ---
if not data.empty:
    st.subheader("🔮 Forecast for Next 10 Days")

    df = data.copy()
    df["weather"] = df["weather"].astype("category").cat.codes
    df.fillna(0, inplace=True)

    features = ["sales", "customers", "weather", "addons"]
    df["target_sales"] = df["sales"].shift(-1)
    df["target_customers"] = df["customers"].shift(-1)
    df.dropna(inplace=True)

    X = df[features]
    y_sales = df["target_sales"]
    y_customers = df["target_customers"]

    X_train, X_test, y_train_sales, y_test_sales = train_test_split(X, y_sales, test_size=0.2, random_state=42)
    _, _, y_train_customers, y_test_customers = train_test_split(X, y_customers, test_size=0.2, random_state=42)

    model_sales = xgb.XGBRegressor()
    model_customers = xgb.XGBRegressor()
    model_sales.fit(X_train, y_train_sales)
    model_customers.fit(X_train, y_train_customers)

    last_row = df[features].iloc[-1].values.reshape(1, -1)
    future_dates = [df["date"].max() + datetime.timedelta(days=i) for i in range(1, 11)]

    forecast_sales = []
    forecast_customers = []

    for _ in range(10):
        pred_sales = model_sales.predict(last_row)[0]
        pred_customers = model_customers.predict(last_row)[0]
        forecast_sales.append(pred_sales)
        forecast_customers.append(pred_customers)
        next_row = last_row.copy()
        next_row[0, 0] = pred_sales
        next_row[0, 1] = pred_customers
        last_row = next_row

    forecast_df = pd.DataFrame({
        "date": future_dates,
        "forecast_sales": np.round(forecast_sales, 2),
        "forecast_customers": np.round(forecast_customers, 0)
    })

    st.line_chart(forecast_df.set_index("date"))

    st.dataframe(forecast_df)

    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Forecast CSV", csv, "forecast.csv", "text/csv")
else:
    st.warning("Please upload or input at least 2 days of data to generate a forecast.")
