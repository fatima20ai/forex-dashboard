import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor     
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(page_title="Forex Pro Dashboard", layout="wide")

# Styling
st.markdown(""" <style> .main { background-color: #f0f2f6; } </style> """, unsafe_allow_html=True)

#@st.cache_data
def load_data():
    df = pd.read_csv('currency_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')
    if df['Price'].dtype == 'object':
        df['Price'] = df['Price'].str.replace(',', '').astype(float)
    return df

try:
    df = load_data()

    # --- SIDEBAR ---
    st.sidebar.title("⚙️ Control Panel")
    st.sidebar.info("Adjust settings to update graphs and predictions.")
    date_range = st.sidebar.date_input("Filter Data Range", [df['Date'].min(), df['Date'].max()])
    prediction_days = st.sidebar.slider("Days to Predict Future", 7, 90, 30)

    # --- HEADER SECTION ---
    st.title("💹 USD to PKR Computational Finance Project")
    st.markdown("Dashboard | Statistical Indicators | AI Predictions")
    st.divider()

    # Metrics
    last_price = df['Price'].iloc[-1]
    avg_price = df['Price'].mean()
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Latest Rate", f"{last_price:.2f} ₨")
    m2.metric("Average Rate", f"{avg_price:.2f} ₨")
    m3.metric("Highest (All Time)", f"{df['Price'].max():.2f} ₨")
    
    # Trend Indicator
    ma50_val = df['Price'].rolling(window=50).mean().iloc[-1]
    if last_price > ma50_val:
        m4.success("Trend: Bullish (Up)")
    else:
        m4.warning("Trend: Bearish (Down)")

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["📊 Market Analysis", "🤖 Future Forecasting", "📄 Dataset Preview"])

    with tab1:
        st.subheader("1. Price Trend & Moving Average (50-Day)")
        df['MA50'] = df['Price'].rolling(window=50).mean()
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df['Date'], y=df['Price'], name="Market Price", line=dict(color='#1f77b4')))
        fig1.add_trace(go.Scatter(x=df['Date'], y=df['MA50'], name="50-Day Indicator", line=dict(color='#ff7f0e', dash='dash')))
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        st.subheader(f"AI Forecast for Next {prediction_days} Days")
        
        # --- FIX: Prediction Logic ---
        # We are using only the last 600 days of data to ensure more accurate predictions
        train_df = df.tail(600).copy() 
        train_df['Days_Count'] = range(len(train_df))
        
        X = np.array(train_df['Days_Count']).reshape(-1, 1)
        y = np.array(train_df['Price'])
        
        # random forest is using for currency trend
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        last_idx = train_df['Days_Count'].iloc[-1]
        future_X = np.array(range(last_idx + 1, last_idx + prediction_days + 1)).reshape(-1, 1)
        forecast = model.predict(future_X)
        
        future_dates = [df['Date'].iloc[-1] + timedelta(days=i) for i in range(1, prediction_days + 1)]
        
        fig_pred = go.Figure()
        # Actual Data (Pichle 3 mahine)
        fig_pred.add_trace(go.Scatter(x=df['Date'].tail(90), y=df['Price'].tail(90), name="Recent History"))
        # Future Prediction
        fig_pred.add_trace(go.Scatter(x=future_dates, y=forecast, name="AI Prediction", line=dict(color='red', width=3, dash='dot')))
        
        st.plotly_chart(fig_pred, use_container_width=True)
        st.success(f"AI Target: approx {forecast[-1]:.2f} PKR by {future_dates[-1].strftime('%d %B %Y')}")

    with tab3:
        st.dataframe(df.sort_values(by='Date', ascending=False), use_container_width=True)

except Exception as e:
    st.error(f"Error: {e}")
