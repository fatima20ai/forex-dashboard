import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(page_title="Forex Pro Dashboard", layout="wide")

# Styling
st.markdown(""" <style> .main { background-color: #f0f2f6; } </style> """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    # Aapki file ka naam 'currency_data.csv' hona chahiye
    df = pd.read_csv('currency_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')
    # Clean price column if it contains commas
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
    st.markdown("Dashboard | Statistical Indicators | ML Predictions")
    st.divider()

    # --- METRICS SECTION (Requirement 1: Values) ---
    last_price = df['Price'].iloc[-1]
    avg_price = df['Price'].mean()
    max_price = df['Price'].max()
    min_price = df['Price'].min()
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Latest Rate", f"{last_price:.2f} ₨")
    m2.metric("Average Rate", f"{avg_price:.2f} ₨")
    m3.metric("Highest (All Time)", f"{max_price:.2f} ₨")
    
    # Trend Indicator Logic
    if last_price > df['Price'].rolling(window=50).mean().iloc[-1]:
        m4.success("Trend: Bullish (Up)")
    else:
        m4.warning("Trend: Bearish (Down)")

    # --- TABS SECTION ---
    tab1, tab2, tab3 = st.tabs(["📊 Market Analysis", "🤖 Future Forecasting", "📄 Dataset Preview"])

    with tab1:
        # Requirement 1 & 2: Main Dashboard Graph with MA Indicator
        st.subheader("1. Price Trend & Moving Average (50-Day)")
        df['MA50'] = df['Price'].rolling(window=50).mean()
        
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df['Date'], y=df['Price'], name="Market Price", line=dict(color='#1f77b4', width=2)))
        fig1.add_trace(go.Scatter(x=df['Date'], y=df['MA50'], name="50-Day Indicator", line=dict(color='#ff7f0e', dash='dash')))
        fig1.update_layout(height=450, template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig1, use_container_width=True)

        # Additional Graphs for "Strong" Project
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("2. Market Volatility (Risk)")
            df['Volatility'] = df['Price'].pct_change().rolling(window=21).std() * 100
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df['Date'], y=df['Volatility'], name="Volatility", fill='tozeroy', line=dict(color='purple')))
            fig2.update_layout(height=350, template="plotly_white")
            st.plotly_chart(fig2, use_container_width=True)

        with col_b:
            st.subheader("3. Price Frequency (Histogram)")
            fig3 = go.Figure()
            fig3.add_trace(go.Histogram(x=df['Price'], nbinsx=40, marker_color='#2ca02c', opacity=0.7))
            fig3.update_layout(height=350, template="plotly_white")
            st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        # Requirement 2: Prediction using ML Formula
        st.subheader(f"Linear Forecast for Next {prediction_days} Days")
        
        # Prepare Data for Prediction
        df['Days_Count'] = (df['Date'] - df['Date'].min()).dt.days
        X = np.array(df['Days_Count']).reshape(-1, 1)
        y = np.array(df['Price'])
        
        model = LinearRegression()
        model.fit(X, y)
        
        last_day = df['Days_Count'].iloc[-1]
        future_X = np.array(range(last_day + 1, last_day + prediction_days + 1)).reshape(-1, 1)
        forecast = model.predict(future_X)
        
        # Forecast Visual
        future_dates = [df['Date'].iloc[-1] + timedelta(days=i) for i in range(1, prediction_days + 1)]
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=future_dates, y=forecast, name="Predicted Path", mode='lines+markers', line=dict(color='red', width=3)))
        fig_pred.update_layout(height=500, template="plotly_white", xaxis_title="Timeline", yaxis_title="Price (PKR)")
        st.plotly_chart(fig_pred, use_container_width=True)
        
        st.success(f"Model Prediction: USD/PKR might reach approx {forecast[-1]:.2f} by {future_dates[-1].strftime('%d %B %Y')}")

    with tab3:
        # Requirement 3: Real Online Dataset
        st.subheader("Verified Dataset (2010 - 2024)")
        st.write("Source: Real-world Forex Historical Data")
        st.dataframe(df.sort_values(by='Date', ascending=False), use_container_width=True)

except Exception as e:
    st.error(f"Attention: {e}")
    st.warning("Please make sure 'currency_data.csv' is in the same folder as this script.")