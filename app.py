import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib
import os

st.set_page_config(page_title="AI Sales & Customer Forecaster", layout="wide")

st.title("📊 AI Sales & Customer Forecaster")

if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=['date', 'sales', 'customers', 'weather', 'addons'])

if 'future_events' not in st.session_state:
    st.session_state.future_events = pd.DataFrame(columns=['future_date', 'event_name', 'prev_sales', 'prev_customers'])

with st.expander("➕ Add Sales Data"):
    with st.form("data_form"):
        date = st.date_input("Date")
        sales = st.number_input("Sales", min_value=0.0)
        customers = st.number_input("Customers", min_value=0)
        weather = st.text_input("Weather")
        addons = st.number_input("Add-ons (if any)", min_value=0.0)
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.session_state.data.loc[len(st.session_state.data)] = [date, sales, customers, weather, addons]
            st.success("Data added!")

st.subheader("📋 Historical Data")
if not st.session_state.data.empty:
    edited_data = st.data_editor(st.session_state.data.sort_values("date").reset_index(drop=True), num_rows="dynamic")
    st.session_state.data = edited_data
else:
    st.info("No data added yet.")

with st.expander("📅 Future Events Reference"):
    with st.form("event_form"):
        future_date = st.date_input("Future Event Date")
        event_name = st.text_input("Event Name")
        prev_sales = st.number_input("Previous Sales on Event", min_value=0.0)
        prev_customers = st.number_input("Previous Customers on Event", min_value=0)
        event_submit = st.form_submit_button("Add Event")
        if event_submit:
            st.session_state.future_events.loc[len(st.session_state.future_events)] = [future_date, event_name, prev_sales, prev_customers]
            st.success("Event added!")

    st.dataframe(st.session_state.future_events)

st.subheader("📈 Forecast (Next 10 Days)")

def prepare_data(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    return df

def train_and_forecast(df, target):
    df = prepare_data(df)
    X = df[["dayofweek", "month", "day", "addons"]]
    y = df[target]
    model = XGBRegressor()
    model.fit(X, y)

    future_dates = [datetime.now().date() + timedelta(days=i) for i in range(1, 11)]
    future_df = pd.DataFrame({
        "date": future_dates,
        "dayofweek": [d.weekday() for d in future_dates],
        "month": [d.month for d in future_dates],
        "day": [d.day for d in future_dates],
        "addons": [0] * 10
    })

    y_pred = model.predict(future_df[["dayofweek", "month", "day", "addons"]])
    future_df["forecast_" + target] = y_pred
    return future_df[["date", "forecast_" + target]]

if not st.session_state.data.empty and len(st.session_state.data) >= 10:
    sales_forecast = train_and_forecast(st.session_state.data, "sales")
    customers_forecast = train_and_forecast(st.session_state.data, "customers")

    forecast = pd.merge(sales_forecast, customers_forecast, on="date")
    st.line_chart(forecast.set_index("date"))

    csv = forecast.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Forecast CSV", csv, "forecast.csv", "text/csv")
else:
    st.warning("Need at least 10 data entries to generate forecast.")
