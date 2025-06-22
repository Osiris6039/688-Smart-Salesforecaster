
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import base64
import os

# Login System
def check_login(username, password):
    return username == "admin" and password == "admin123"

# Load or initialize data
def load_data(file_path='data.csv'):
    if os.path.exists(file_path):
        return pd.read_csv(file_path, parse_dates=['Date'])
    else:
        return pd.DataFrame(columns=['Date', 'Sales', 'Customers', 'Weather', 'Add-on Sales'])

def save_data(df, file_path='data.csv'):
    df.to_csv(file_path, index=False)

# Forecasting Model
def train_forecast_model(df):
    df = df.sort_values('Date')
    df['Add-on Sales'] = df['Add-on Sales'].fillna(0)
    df['Weather'] = df['Weather'].astype('category').cat.codes

    features = ['Weather', 'Add-on Sales']
    X = df[features]
    y_sales = df['Sales']
    y_customers = df['Customers']

    X_train, X_test, y_sales_train, y_sales_test = train_test_split(X, y_sales, test_size=0.2, random_state=42)
    _, _, y_customers_train, y_customers_test = train_test_split(X, y_customers, test_size=0.2, random_state=42)

    model_sales = xgb.XGBRegressor()
    model_customers = xgb.XGBRegressor()

    model_sales.fit(X_train, y_sales_train)
    model_customers.fit(X_train, y_customers_train)

    return model_sales, model_customers

# Generate future data
def generate_future_inputs(df, n_days=10):
    last_row = df.iloc[-1]
    future_dates = [df['Date'].max() + timedelta(days=i) for i in range(1, n_days + 1)]
    future_data = pd.DataFrame({
        'Date': future_dates,
        'Weather': last_row['Weather'],
        'Add-on Sales': 0  # assume no add-ons unless manually added
    })
    future_data['Weather'] = future_data['Weather'].astype('category').cat.codes
    return future_data

# Main app
def main():
    st.title("📈 AI Sales & Customer Forecaster (10 Days)")

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        with st.form("Login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login = st.form_submit_button("Login")
            if login and check_login(username, password):
                st.session_state.authenticated = True
                st.experimental_rerun()
            elif login:
                st.error("Invalid credentials")
        return

    data = load_data()

    st.subheader("📤 Upload CSV")
    uploaded_file = st.file_uploader("Upload your data file", type=["csv"])
    if uploaded_file:
        new_data = pd.read_csv(uploaded_file, parse_dates=['Date'])
        data = pd.concat([data, new_data], ignore_index=True)
        data.drop_duplicates(subset="Date", keep="last", inplace=True)
        data = data.sort_values("Date")
        save_data(data)
        st.success("Data uploaded successfully")

    st.subheader("📝 Manual Data Entry")
    with st.form("manual_entry"):
        date = st.date_input("Date")
        sales = st.number_input("Sales", 0)
        customers = st.number_input("Customers", 0)
        weather = st.selectbox("Weather", ["Sunny", "Rainy", "Cloudy"])
        addons = st.number_input("Add-on Sales", 0)
        submit = st.form_submit_button("Add Entry")
        if submit:
            new_row = pd.DataFrame([[date, sales, customers, weather, addons]],
                                   columns=data.columns)
            data = pd.concat([data, new_row], ignore_index=True)
            data = data.drop_duplicates(subset="Date", keep="last")
            data = data.sort_values("Date")
            save_data(data)
            st.success("Data added successfully")

    st.subheader("🗃️ Data Records")
    if not data.empty:
        for i, row in data.iterrows():
            cols = st.columns([6, 1])
            cols[0].write(f"{row['Date'].date()} | Sales: {row['Sales']} | Cust: {row['Customers']} | Weather: {row['Weather']} | Add-ons: {row['Add-on Sales']}")
            if cols[1].button("❌", key=f"del_{i}"):
                data.drop(i, inplace=True)
                data = data.sort_values("Date")
                save_data(data)
                st.experimental_rerun()

    st.subheader("🔮 Forecasting")
    if len(data) >= 10:
        model_sales, model_customers = train_forecast_model(data)
        future_data = generate_future_inputs(data)

        pred_sales = model_sales.predict(future_data[['Weather', 'Add-on Sales']])
        pred_customers = model_customers.predict(future_data[['Weather', 'Add-on Sales']])

        forecast_df = future_data.copy()
        forecast_df['Forecasted Sales'] = pred_sales
        forecast_df['Forecasted Customers'] = pred_customers

        st.line_chart(forecast_df[['Forecasted Sales', 'Forecasted Customers']])
        st.dataframe(forecast_df[['Date', 'Forecasted Sales', 'Forecasted Customers']])

        csv = forecast_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="forecast.csv">📥 Download Forecast CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

    else:
        st.warning("Please upload at least 10 records to enable forecasting.")

if __name__ == "__main__":
    main()
