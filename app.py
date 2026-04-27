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
    
    # 1. FIXED: Sidebar date selection
    date_selection = st.sidebar.date_input("Filter Data Range", [df['Date'].min(), df['Date'].max()])
    prediction_days = st.sidebar.slider("Days to Predict Future", 7, 90, 30)

    # 2. FIXED: Filter application logic
    if isinstance(date_selection, (list, tuple)) and len(date_selection) == 2:
        start_date, end_date = date_selection
        # Hum 'filtered_df' bana rahay hain jo sirf select ki hui dates dikhayega
        filtered_df = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)].copy()
    else:
        filtered_df = df.copy()

    # --- HEADER SECTION ---
    st.title("💹 USD to PKR Computational Finance Project")
    st.divider()

    # Metrics (Using filtered_df so they update)
    last_price = filtered_df['Price'].iloc[-1] if not filtered_df.empty else 0
    avg_price = filtered_df['Price'].mean() if not filtered_df.empty else 0
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Latest Rate", f"{last_price:.2f} ₨")
    m2.metric("Average Rate", f"{avg_price:.2f} ₨")
    m3.metric("Highest (Selected)", f"{filtered_df['Price'].max():.2f} ₨" if not filtered_df.empty else "0 ₨")
    
    # Trend Indicator
    ma50_series = filtered_df['Price'].rolling(window=50).mean()
    ma50_val = ma50_series.iloc[-1] if len(ma50_series) > 0 else 0
    if last_price > ma50_val:
        m4.success("Trend: Bullish")
    else:
        m4.warning("Trend: Bearish")

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["📊 Market Analysis", "🤖 Future Forecasting", "📄 Dataset Preview"])

    with tab1:
        # Graph 1: Price Trend
        st.subheader("1. Price Trend & Moving Average (50-Day)")
        filtered_df['MA50'] = filtered_df['Price'].rolling(window=50).mean()
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['Price'], name="Market Price"))
        fig1.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['MA50'], name="50-Day Indicator", line=dict(dash='dash')))
        st.plotly_chart(fig1, use_container_width=True)

        # Graph 2 & 3: Volatility & Histogram (In Columns)
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("2. Market Volatility (Risk)")
            # Daily returns percentage change
            filtered_df['Volatility'] = filtered_df['Price'].pct_change().rolling(window=21).std() * 100
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['Volatility'], name="Volatility", fill='tozeroy', line=dict(color='purple')))
            fig2.update_layout(height=350)
            st.plotly_chart(fig2, use_container_width=True)

        with col_b:
            st.subheader("3. Price Frequency (Histogram)")
            fig3 = go.Figure()
            fig3.add_trace(go.Histogram(x=filtered_df['Price'], nbinsx=40, marker_color='#2ca02c', opacity=0.7))
            fig3.update_layout(height=350)
            st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        st.subheader(f"AI Forecast for Next {prediction_days} Days")
        
        # Training Model (Uses last 600 days of ORIGINAL data)
        train_df = df.tail(600).copy() 
        train_df['Days_Count'] = range(len(train_df))
        X = np.array(train_df['Days_Count']).reshape(-1, 1)
        y = np.array(train_df['Price'])
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        last_idx = train_df['Days_Count'].iloc[-1]
        future_X = np.array(range(last_idx + 1, last_idx + prediction_days + 1)).reshape(-1, 1)
        forecast = model.predict(future_X)
        
        future_dates = [df['Date'].iloc[-1] + timedelta(days=i) for i in range(1, prediction_days + 1)]
        
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=filtered_df['Date'].tail(90), y=filtered_df['Price'].tail(90), name="History"))
        fig_pred.add_trace(go.Scatter(x=future_dates, y=forecast, name="AI Prediction", line=dict(color='red', width=3, dash='dot')))
        st.plotly_chart(fig_pred, use_container_width=True)
        st.success(f"AI Target: approx {forecast[-1]:.2f} PKR")

    with tab3:
        st.dataframe(filtered_df.sort_values(by='Date', ascending=False), use_container_width=True)

except Exception as e:
    st.error(f"Error: {e}")
