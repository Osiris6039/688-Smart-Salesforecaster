
import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import os

# Initialize data
DATA_FILE = "data.csv"
EVENT_FILE = "future_events.csv"

def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE, parse_dates=["date"])
    return pd.DataFrame(columns=["date", "sales", "customers", "weather", "addons"])

def load_events():
    if os.path.exists(EVENT_FILE):
        return pd.read_csv(EVENT_FILE, parse_dates=["event_date"])
    return pd.DataFrame(columns=["event_date", "event_name", "prev_sales", "prev_customers"])

def save_data(df):
    df.sort_values("date", inplace=True)
    df.to_csv(DATA_FILE, index=False)

def save_events(df):
    df.sort_values("event_date", inplace=True)
    df.to_csv(EVENT_FILE, index=False)

def forecast(df):
    df = df.copy()
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["addons"] = df["addons"].fillna(0)

    features = ["dayofweek", "month", "weather", "addons"]

    df = pd.get_dummies(df, columns=["weather"])
    features = list(set(df.columns) - set(["date", "sales", "customers"]))

    X = df[features]
    y_sales = df["sales"]
    y_customers = df["customers"]

    model_sales = xgb.XGBRegressor()
    model_customers = xgb.XGBRegressor()

    model_sales.fit(X, y_sales)
    model_customers.fit(X, y_customers)

    last_date = df["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=10)

    forecast_input = pd.DataFrame({
        "date": future_dates,
        "dayofweek": future_dates.dayofweek,
        "month": future_dates.month,
        "addons": 0
    })

    common_weather = df["weather"].mode()[0]
    forecast_input["weather"] = common_weather
    forecast_input = pd.get_dummies(forecast_input, columns=["weather"])

    for col in set(X.columns) - set(forecast_input.columns):
        forecast_input[col] = 0
    forecast_input = forecast_input[X.columns]

    forecast_input["forecast_sales"] = model_sales.predict(forecast_input)
    forecast_input["forecast_customers"] = model_customers.predict(forecast_input)

    return forecast_input

# App UI
st.set_page_config(page_title="Smart Sales & Customer Forecaster", layout="wide")
st.title("📊 Smart AI Sales & Customer Forecaster")

st.sidebar.header("🔐 Login")
password = st.sidebar.text_input("Enter password", type="password")
if password != "688":
    st.warning("Please enter the correct password to access the app.")
    st.stop()

st.success("Login successful!")

df = load_data()
events = load_events()

tab1, tab2, tab3 = st.tabs(["📥 Input Data", "📆 Add Future Event", "📈 Forecast"])

with tab1:
    st.subheader("Add Past Data")
    with st.form("data_form", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        date = col1.date_input("Date")
        sales = col2.number_input("Sales", min_value=0.0)
        customers = col3.number_input("Customers", min_value=0)
        weather = st.selectbox("Weather", ["Sunny", "Rainy", "Cloudy", "Stormy"])
        addons = st.number_input("Add-on Sales (Irregular)", min_value=0.0)
        submitted = st.form_submit_button("Add Data")
        if submitted:
            df = pd.concat([df, pd.DataFrame([[date, sales, customers, weather, addons]], columns=df.columns)])
            save_data(df)
            st.success("Data added!")

    if not df.empty:
        st.markdown("### Existing Data")
        for i in df.index:
            st.write(f"{df.loc[i, 'date'].date()} | Sales: ₱{df.loc[i, 'sales']} | Customers: {df.loc[i, 'customers']}")
            if st.button(f"❌ Delete {df.loc[i, 'date'].date()}", key=f"del_{i}"):
                df = df.drop(i)
                save_data(df)
                st.rerun()

with tab2:
    st.subheader("Add Future Reference Event")
    with st.form("event_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        event_date = col1.date_input("Event Date")
        event_name = col2.text_input("Event Name")
        prev_sales = st.number_input("Last Year Sales", min_value=0.0)
        prev_customers = st.number_input("Last Year Customers", min_value=0)
        added = st.form_submit_button("Save Event")
        if added:
            events = pd.concat([events, pd.DataFrame([[event_date, event_name, prev_sales, prev_customers]], columns=events.columns)])
            save_events(events)
            st.success("Event added!")

    if not events.empty:
        st.markdown("### Stored Future Events")
        for i in events.index:
            st.write(f"{events.loc[i, 'event_date'].date()} - {events.loc[i, 'event_name']}")
            if st.button(f"❌ Delete {events.loc[i, 'event_name']}", key=f"del_evt_{i}"):
                events = events.drop(i)
                save_events(events)
                st.rerun()

with tab3:
    st.subheader("🔮 Forecasting Results")
    if len(df) < 10:
        st.warning("Please input at least 10 days of data for accurate forecasting.")
    else:
        result = forecast(df)
        st.write(result[["date", "forecast_sales", "forecast_customers"]])
        st.line_chart(result.set_index("date")[["forecast_sales", "forecast_customers"]])
        csv = result.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Forecast CSV", csv, "forecast.csv", "text/csv")
