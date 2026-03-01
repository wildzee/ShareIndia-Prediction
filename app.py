import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.pipeline_utils import run_full_pipeline

st.set_page_config(page_title="Share India Predictor", layout="wide", page_icon="📈")

# Load full list of NSE stocks
@st.cache_data
def load_nse_tickers():
    try:
        # Load from the local file we downloaded
        df = pd.read_csv('nse_tickers.csv')
        df['Display'] = df['NAME OF COMPANY'] + ' (' + df['SYMBOL'] + ')'
        
        # Create a dictionary of Display Name -> Ticker symbol with .NS suffix
        ticker_dict = dict(zip(df['Display'], df['SYMBOL'] + '.NS'))
        return ticker_dict
    except Exception as e:
        st.error("Failed to load NSE tickers.")
        return {"Share India Securities (SHAREINDIA)": "SHAREINDIA.NS"}

NSE_TICKERS = load_nse_tickers()

st.title("📈 Indian Stock Price Predictor (Long-Term)")
st.caption("A multi-modal Ensemble ML prediction system (LSTM + XGBoost) and RandomForest Direction Predictor running locally.")

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Autocomplete selectbox
    selected_company = st.selectbox(
        "Search or Select a Company", 
        options=list(NSE_TICKERS.keys()), 
        index=0
    )
    
    ticker = NSE_TICKERS[selected_company]
    st.info(f"**Selected Ticker:** `{ticker}`")
    
    st.markdown("---")
    st.write("Click below to fetch the latest data, build features, train the AI models, and generate a prediction.")
    
    run_button = st.button("🚀 Run Prediction Pipeline", type="primary", use_container_width=True)

if run_button:
    with st.spinner(f"Running full ML pipeline for {selected_company} (This takes 1-2 minutes)..."):
        try:
            forecast, signal, master_df, rf_signal, rf_precision, rf_predictions = run_full_pipeline(ticker, selected_company)
            
            st.success("✅ Pipeline completed successfully!")
            st.toast("Prediction Generated!", icon="🎯")

            # --- Metrics Row ---
            st.subheader("📊 Latest AI Forecast & Signals")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Price", f"₹{forecast['current_price']}")
                
                signal_color = "normal"
                if signal == "BUY 🟢":
                    signal_color = "inverse"
                elif signal == "SELL 🔴":
                    signal_color = "inverse" # Streamlit doesn't natively support red for positive, but we style text
                    
                st.metric("LSTM+XGBoost Target Signal", signal)
            
            with col2:
                st.metric(
                    "1-Month Target", 
                    f"₹{forecast['monthly_target']}", 
                    f"{forecast['monthly_upside']}%"
                )
                
                st.metric(
                    "RF Direction Signal (Next Day)", 
                    rf_signal
                )
                
            with col3:
                st.metric(
                    "1-Year Target", 
                    f"₹{forecast['yearly_target']}", 
                    f"{forecast['yearly_upside']}%"
                )
                
                st.metric(
                    "RF Backtest Precision",
                    f"{rf_precision:.1%}" if rf_precision > 0 else "N/A"
                )

            st.markdown("---")
            
            # --- Charts ---
            st.subheader(f"{selected_company} — Historical Output & Indicators")
            
            # Chart 1: Price and MAs
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(x=master_df.index, y=master_df['close'], name='Close Price', line=dict(color='blue')))
            if 'ema_50' in master_df.columns:
                fig_price.add_trace(go.Scatter(x=master_df.index, y=master_df['ema_50'], name='EMA 50', line=dict(color='orange', dash='dot')))
            if 'ema_200' in master_df.columns:
                fig_price.add_trace(go.Scatter(x=master_df.index, y=master_df['ema_200'], name='EMA 200', line=dict(color='red', dash='dot')))
            
            fig_price.update_layout(title="Closing Price & Moving Averages", hovermode="x unified")
            st.plotly_chart(fig_price, use_container_width=True)
            
            # Chart 2: RSI
            if 'rsi' in master_df.columns:
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=master_df.index, y=master_df['rsi'], name='RSI (14)', line=dict(color='purple')))
                fig_rsi.add_hline(y=70, line_dash='dash', line_color='red', annotation_text='Overbought')
                fig_rsi.add_hline(y=30, line_dash='dash', line_color='green', annotation_text='Oversold')
                fig_rsi.update_layout(title="Relative Strength Index (RSI)", height=300)
                st.plotly_chart(fig_rsi, use_container_width=True)
                
            # Chart 3: RF Rolling Accuracy
            if not rf_predictions.empty:
                st.subheader("🌲 RandomForest Walk-Forward Backtest")
                correct = (rf_predictions["Target"] == rf_predictions["Predictions"]).astype(int)
                rolling_acc = correct.rolling(50, min_periods=10).mean()
                
                fig_rf = go.Figure()
                fig_rf.add_trace(go.Scatter(x=rolling_acc.index, y=rolling_acc, name='50-Day Rolling Accuracy', line=dict(color='green')))
                fig_rf.add_hline(y=0.5, line_dash='dash', line_color='red', annotation_text='50% Baseline (Coin Flip)')
                fig_rf.update_layout(
                    title=f"Walk-Forward Direction Precision (Historical Avg: {rf_precision:.1%})", 
                    height=300,
                    yaxis_title="Accuracy"
                )
                st.plotly_chart(fig_rf, use_container_width=True)

        except Exception as e:
            st.error(f"Failed to run pipeline: {str(e)}")
            st.exception(e)

else:
    st.info("👈 Please select a company from the sidebar and click **Run Prediction Pipeline**.")
