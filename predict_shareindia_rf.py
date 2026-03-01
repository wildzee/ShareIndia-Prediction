#!/usr/bin/env python3
"""
Share India Securities (NSE: SHAREINDIA) — RandomForest Direction Predictor
===========================================================================
Adapted from the NVDA prediction notebook.

This script uses a RandomForestClassifier to predict whether SHAREINDIA's
closing price will go UP or DOWN the next trading day.

Features:
  - Rolling close ratios (2, 5, 60, 250, 1000 day windows)
  - Trend counts (rolling sum of up-day targets)
  - Walk-forward backtesting with expanding training window
  - Precision score evaluation

Usage:
  python predict_shareindia_rf.py
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

warnings.filterwarnings("ignore")

# ─── Configuration ───────────────────────────────────────────────────────────

TICKER = "SHAREINDIA.NS"
STOCK_NAME = "Share India Securities"

# RandomForest hyperparameters
N_ESTIMATORS = 300        # More trees to compensate for smaller dataset
MIN_SAMPLES_SPLIT = 25    # Low split threshold for ~1,000 usable rows
RANDOM_STATE = 1

# Rolling feature horizons (days)
HORIZONS = [2, 5, 20, 60, 250]  # No 1000 — SHAREINDIA only has ~1,348 days

# Walk-forward backtest parameters
# SHAREINDIA has ~1,700 trading days — much less than NVDA's ~6,200
# so we use smaller windows
BACKTEST_START = 300      # Start backtesting after 300 rows
BACKTEST_STEP = 50        # Step forward 50 rows per iteration

# Basic predictors (same as NVDA notebook)
BASIC_PREDICTORS = ["Close", "Volume", "Open", "High", "Low"]

# Output paths
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")
PLOT_PATH = os.path.join(OUTPUT_DIR, "rf_backtest_results.png")


# ─── Helper Functions ────────────────────────────────────────────────────────

def fetch_data(ticker: str) -> pd.DataFrame:
    """Fetch maximum available historical data from Yahoo Finance."""
    print(f"\n{'='*60}")
    print(f"  Fetching data for {ticker}...")
    print(f"{'='*60}")

    stock = yf.Ticker(ticker)
    df = stock.history(period="max")

    if df.empty:
        raise ValueError(f"No data returned for ticker: {ticker}")

    print(f"  ✅ Downloaded {len(df)} trading days")
    print(f"  📅 Date range: {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  💰 Latest close: ₹{df['Close'].iloc[-1]:.2f}")

    return df


def engineer_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create tomorrow's close and binary target (1=up, 0=down)."""
    df = df.copy()
    df["Tomorrow"] = df["Close"].shift(-1)
    df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)
    return df


def engineer_rolling_features(df: pd.DataFrame, horizons: list) -> tuple:
    """
    Create rolling ratio and trend features for each horizon.
    
    For each horizon h:
      - Close_Ratio_{h} = Close / rolling(h).mean(Close)
      - Trend_{h} = shift(1).rolling(h).sum(Target)
    """
    df = df.copy()
    new_predictors = []

    for horizon in horizons:
        rolling_averages = df.rolling(horizon).mean()

        # Close ratio: how far is current close from the h-day mean?
        ratio_col = f"Close_Ratio_{horizon}"
        df[ratio_col] = df["Close"] / rolling_averages["Close"]

        # Trend: how many of the last h days went up?
        trend_col = f"Trend_{horizon}"
        df[trend_col] = df.shift(1).rolling(horizon).sum()["Target"]

        new_predictors += [ratio_col, trend_col]

    # Drop NaN rows caused by rolling windows
    before = len(df)
    df = df.dropna()
    after = len(df)
    print(f"  📊 Dropped {before - after} rows (rolling warmup) → {after} rows remain")

    return df, new_predictors


def predict(train: pd.DataFrame, test: pd.DataFrame,
            predictors: list, model) -> pd.DataFrame:
    """Train model on train set and predict on test set."""
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


def backtest(data: pd.DataFrame, model, predictors: list,
             start: int, step: int) -> pd.DataFrame:
    """Walk-forward backtest with expanding training window."""
    all_predictions = []

    total_steps = (data.shape[0] - start) // step
    print(f"\n{'='*60}")
    print(f"  Walk-Forward Backtest")
    print(f"{'='*60}")
    print(f"  Total rows: {data.shape[0]}")
    print(f"  Start: {start} | Step: {step} | Iterations: {total_steps}")

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i + step)].copy()
        if len(test) == 0:
            break
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)

    result = pd.concat(all_predictions)
    print(f"  ✅ Backtest complete — {len(result)} predictions generated")
    return result


def evaluate(predictions: pd.DataFrame, stock_name: str):
    """Print evaluation metrics."""
    precision = precision_score(predictions["Target"], predictions["Predictions"])

    target_dist = predictions["Target"].value_counts() / predictions.shape[0]
    pred_dist = predictions["Predictions"].value_counts()

    print(f"\n{'='*60}")
    print(f"  {stock_name} — RandomForest Direction Prediction Results")
    print(f"{'='*60}")
    print(f"  📏 Precision Score: {precision:.4f}")
    print(f"     (1.0 = perfect | 0.5 = coin flip | >0.5 = useful)")
    print()
    print(f"  📊 Prediction Distribution:")
    for val, count in pred_dist.items():
        label = "UP ↑" if val == 1 else "DOWN ↓"
        print(f"       {label}: {count} ({count/len(predictions)*100:.1f}%)")
    print()
    print(f"  📊 Actual Target Distribution:")
    for val, pct in target_dist.items():
        label = "UP ↑" if val == 1 else "DOWN ↓"
        print(f"       {label}: {pct*100:.1f}%")
    print(f"{'='*60}\n")

    return precision


def plot_results(predictions: pd.DataFrame, stock_name: str, save_path: str):
    """Save backtest results plot."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle(f"{stock_name} — RF Direction Prediction Backtest",
                 fontsize=14, fontweight="bold")

    # Top: Target vs Predictions overlay
    axes[0].plot(predictions.index, predictions["Target"],
                 label="Actual", alpha=0.6, linewidth=0.8, color="steelblue")
    axes[0].plot(predictions.index, predictions["Predictions"],
                 label="Predicted", alpha=0.6, linewidth=0.8, color="coral")
    axes[0].set_ylabel("Direction (1=UP, 0=DOWN)")
    axes[0].legend(loc="upper left")
    axes[0].set_title("Actual vs Predicted Direction")

    # Bottom: Rolling precision (50-day window)
    correct = (predictions["Target"] == predictions["Predictions"]).astype(int)
    rolling_acc = correct.rolling(50, min_periods=10).mean()
    axes[1].plot(rolling_acc.index, rolling_acc, color="green", linewidth=1)
    axes[1].axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="50% baseline")
    axes[1].set_ylabel("Rolling Accuracy (50-day)")
    axes[1].set_xlabel("Date")
    axes[1].legend(loc="upper left")
    axes[1].set_title("Rolling 50-Day Accuracy")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  📈 Plot saved → {save_path}")
    plt.close()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    # 1. Fetch data
    data = fetch_data(TICKER)

    # 2. Engineer target
    data = engineer_target(data)

    # 3. Build model
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        min_samples_split=MIN_SAMPLES_SPLIT,
        random_state=RANDOM_STATE,
    )

    # 4. Engineer rolling features
    print(f"\n{'='*60}")
    print(f"  Engineering rolling features...")
    print(f"  Horizons: {HORIZONS}")
    print(f"{'='*60}")
    data, new_predictors = engineer_rolling_features(data, HORIZONS)

    # Use all predictors (basic + rolling)
    all_predictors = BASIC_PREDICTORS + new_predictors
    print(f"  Total predictors: {len(all_predictors)}")
    for p in all_predictors:
        print(f"    • {p}")

    # 5. Check if we have enough data
    if len(data) < BACKTEST_START + BACKTEST_STEP:
        print(f"\n  ⚠️  Only {len(data)} rows after feature engineering.")
        print(f"      Need at least {BACKTEST_START + BACKTEST_STEP}.")
        print(f"      Adjusting backtest parameters...")
        adjusted_start = max(int(len(data) * 0.5), 100)
        adjusted_step = max(int(len(data) * 0.1), 25)
        print(f"      New start={adjusted_start}, step={adjusted_step}")
    else:
        adjusted_start = BACKTEST_START
        adjusted_step = BACKTEST_STEP

    # 6. Run backtest
    predictions = backtest(data, model, all_predictors,
                           start=adjusted_start, step=adjusted_step)

    # 7. Evaluate
    precision = evaluate(predictions, STOCK_NAME)

    # 8. Plot
    plot_results(predictions, STOCK_NAME, PLOT_PATH)

    # 9. Today's signal (latest prediction)
    print(f"{'='*60}")
    print(f"  📡 Latest Signal")
    print(f"{'='*60}")

    # Retrain on all data and predict today's direction
    model.fit(data[all_predictors], data["Target"])
    today_pred = model.predict(data[all_predictors].iloc[-1:])
    direction = "UP ↑ (BUY signal)" if today_pred[0] == 1 else "DOWN ↓ (SELL signal)"
    latest_close = data["Close"].iloc[-1]
    latest_date = data.index[-1].strftime("%Y-%m-%d")

    print(f"  Date       : {latest_date}")
    print(f"  Close      : ₹{latest_close:.2f}")
    print(f"  Tomorrow   : {direction}")
    print(f"  Confidence : Based on {N_ESTIMATORS}-tree RF with {len(all_predictors)} features")
    print(f"  Precision  : {precision:.1%} (backtest)")
    print(f"{'='*60}")
    print(f"\n  ⚠️  This is for educational/research purposes only.")
    print(f"      Not financial advice. Always do your own analysis.\n")


if __name__ == "__main__":
    main()
