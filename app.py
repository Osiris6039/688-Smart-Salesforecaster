import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import os

# --- Authentication ---
def login():
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if username == "admin" and password == "admin123":
        return True
    else:
        return False

# --- Data Load and Save ---
DATA_FILE = "data.csv"
if not os.path.exists(DATA_FILE):
    pd.DataFrame(columns=["Date", "Sales", "Customers", "Weather", "AddOns", "Event", "PastEventSales", "PastEventCustomers"]).to_csv(DATA_FILE, index=False)

@st.cache_data(ttl=600)
def load_data():
    return pd.read_csv(DATA_FILE, parse_dates=["Date"])

def save_data(df):
    df.to_csv(DATA_FILE, index=False)

# --- Main App ---
if login():
    st.title("AI Forecasting: Sales and Customers")

    df = load_data()

    # --- Input New Data ---
    st.subheader("Manual Data Entry")
    with st.form("entry_form"):
        date = st.date_input("Date")
        sales = st.number_input("Sales", 0)
        customers = st.number_input("Customers", 0)
        weather = st.text_input("Weather")
        addons = st.number_input("Add-on Sales", 0)
        event = st.text_input("Future Event (optional)")
        past_sales = st.number_input("Last Year Event Sales (optional)", 0)
        past_customers = st.number_input("Last Year Event Customers (optional)", 0)
        submitted = st.form_submit_button("Submit")
        if submitted:
            new_row = pd.DataFrame([[date, sales, customers, weather, addons, event, past_sales, past_customers]],
                                   columns=df.columns)
            df = pd.concat([df, new_row], ignore_index=True)
            df = df.sort_values("Date")
            save_data(df)
            st.success("Data added successfully!")

    # --- Display Data with Delete Option ---
    st.subheader("Current Data")
    for i, row in df.iterrows():
        st.write(f"{row['Date'].date()} - Sales: {row['Sales']}, Customers: {row['Customers']}, Event: {row['Event']}")
        if st.button(f"Delete {i}"):
            df = df.drop(i).reset_index(drop=True)
            save_data(df)
            st.rerun()

    # --- Forecasting ---
    st.subheader("10-Day Forecast")
    df = df.sort_values("Date")
    df["Day"] = df["Date"].dt.dayofyear

    features = ["Day", "AddOns", "PastEventSales", "PastEventCustomers"]
    X = df[features]
    y_sales = df["Sales"]
    y_customers = df["Customers"]

    model_sales = XGBRegressor()
    model_customers = RandomForestRegressor()
    model_sales.fit(X, y_sales)
    model_customers.fit(X, y_customers)

    today = datetime.now()
    forecast_days = [today + timedelta(days=i) for i in range(1, 11)]
    forecast_data = pd.DataFrame({
        "Date": forecast_days,
        "Day": [d.timetuple().tm_yday for d in forecast_days],
        "AddOns": [0]*10,
        "PastEventSales": [0]*10,
        "PastEventCustomers": [0]*10
    })

    forecast_sales = model_sales.predict(forecast_data[features])
    forecast_customers = model_customers.predict(forecast_data[features])
    forecast_data["ForecastSales"] = forecast_sales
    forecast_data["ForecastCustomers"] = forecast_customers

    st.line_chart(forecast_data[["ForecastSales", "ForecastCustomers"]])

    st.download_button("Download Forecast", forecast_data.to_csv(index=False), file_name="forecast.csv")
else:
    st.warning("Please log in to access the app.")