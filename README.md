# Share India Long-Term Stock Predictor

This project contains a complete machine learning pipeline to predict the long-term price targets of **Share India Securities Limited (NSE: SHAREINDIA)**. It uses a hybrid ensemble model of LSTM (Deep Learning), XGBoost (Gradient Boosting), and a standalone RandomForest classifier for next-day direction prediction, incorporating historical price data, fundamentals, technical indicators, and news sentiment.

## Architecture & Data Stack
All data is retrieved from 100% free, zero-login sources:
- **Price Data:** Yahoo Finance (`yfinance`)
- **Fundamentals:** Yahoo Finance (`yfinance`)
- **News Headlines:** Google News RSS
- **Technical Indicators:** Native Pandas calculations (EMA, MACD, RSI, Bollinger Bands, ATR)
- **Sentiment Scoring:** Local VADER + FinBERT analysis

## Project Structure
```text
ShareIndia-Predictor/
├── 01_data_ingestion.ipynb       # Fetch OHLCV, fundamentals, news -> /data/raw/
├── 02_feature_engineering.ipynb  # Compute indicators, sentiment -> /data/processed/
├── 03_eda.ipynb                  # Exploratory analysis & correlation plots
├── 04_model_training.ipynb       # Train LSTM + XGBoost -> save to /models/
├── 05_prediction.ipynb           # Generate forecasts + Buy/Hold/Sell signal
├── 06_backtest.ipynb             # Simulate historical signals -> backtest report
├── predict_shareindia_rf.py      # Standalone RF direction backtesting script (NVDA Style)
├── app.py                        # Streamlit web application
├── config.yaml                   # Core configuration
├── requirements.txt              # Standardized dependencies
└── utils/
    ├── data_utils.py             # Data fetching and TA features
    ├── model_utils.py            # Model training and prediction
    ├── pipeline_utils.py         # App pipeline orchestration
    └── rf_utils.py               # RandomForest feature engineering and backtest logic
```

## Setup & Installation

**Prerequisites:** Python 3.9+

**1. Clone or generate the directory structure.**
Be sure to run setup commands inside `ShareIndia-Predictor/`.

**2. Create a virtual environment and install dependencies:**
```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Running the Application (Web UI)

The easiest way to use the predictor is via the interactive **Streamlit Web Application**, which provides an autocomplete search for all NSE stocks and runs the entire AI pipeline on-the-fly.

```bash
# Start the web server
streamlit run app.py
```
1. Open the provided `Local URL` in your browser.
2. Search for any valid NSE company (e.g. `Reliance Industries`).
3. Click **Run Prediction Pipeline** to trigger real-time data ingestion, multi-horizon model training (LSTM/XGBoost), and the RandomForest direction backtest simultaneously.

### Running the Standalone RF Evaluator
If you only want to test the RandomForest next-day direction model (adapted from the NVDA approach) and view its historical accuracy plot in the terminal:
```bash
python predict_shareindia_rf.py
```

---

## Running the Pipeline (Manual / Jupyter)

If you prefer to run the raw pipeline sequentially for research/debugging:

```bash
# Data Ingestion
jupyter nbconvert --to notebook --execute 01_data_ingestion.ipynb

# Feature Engineering
jupyter nbconvert --to notebook --execute 02_feature_engineering.ipynb

# EDA (optional visual check)
jupyter nbconvert --to notebook --execute 03_eda.ipynb

# Model Training (Trains LSTM Array & XGBoost MultiOutputRegressors)
jupyter nbconvert --to notebook --execute 04_model_training.ipynb

# Prediction & Signals
jupyter nbconvert --to notebook --execute 05_prediction.ipynb

# Backtesting
jupyter nbconvert --to notebook --execute 06_backtest.ipynb
```

## Outputs

The prediction pipeline outputs:
- **`current_price`**: the last fetched closing price
- **`monthly_target` / `yearly_target`**: predicted future prices (from LSTM/XGBoost)
- **`monthly_upside` / `yearly_upside`**: expected returns in percentages (from LSTM/XGBoost)
- **`LSTM+XGBoost Target Signal`**: `BUY`, `HOLD`, or `SELL` based on long-term predicted upside, RSI, and MACD.
- **`RF Direction Signal`**: binary prediction of whether tomorrow's close will be `UP ↑` or `DOWN ↓`.
- **`RF Backtest Precision`**: historical win rate (%) of the 50-day rolling Walk-Forward backtest for the selected stock.
