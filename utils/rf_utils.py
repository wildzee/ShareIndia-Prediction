import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import warnings

warnings.filterwarnings("ignore")

def engineer_rf_features(df):
    """
    Creates rolling ratio and trend features for the RF direction predictor.
    Adapted from the NVDA prediction notebook approach.
    """
    # Create tomorrow's close and binary target
    df = df.copy()
    df["Tomorrow"] = df["close"].shift(-1)
    df["Target"] = (df["Tomorrow"] > df["close"]).astype(int)
    
    # Map pandas-ta column names back to Title Case for the model consistency
    # since data_utils.py lowercases them
    mapping = {
        'close': 'Close', 'volume': 'Volume', 
        'open': 'Open', 'high': 'High', 'low': 'Low'
    }
    df = df.rename(columns=mapping)

    horizons = [2, 5, 20, 60, 250]
    new_predictors = []

    for horizon in horizons:
        rolling_averages = df.rolling(horizon).mean()
        
        ratio_col = f"Close_Ratio_{horizon}"
        df[ratio_col] = df["Close"] / rolling_averages["Close"]
        
        trend_col = f"Trend_{horizon}"
        df[trend_col] = df.shift(1).rolling(horizon).sum()["Target"]
        
        new_predictors += [ratio_col, trend_col]

    # Clean missing values from rolling periods
    df = df.dropna(subset=new_predictors)
    
    return df, ["Close", "Volume", "Open", "High", "Low"] + new_predictors


def predict_rf(train, test, predictors, model):
    """Helper to train and predict for walk-forward validation"""
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    return pd.concat([test["Target"], preds], axis=1)


def backtest_rf(data, model, predictors, start=300, step=50):
    """Runs a walk-forward expanding window backtest"""
    all_predictions = []
    
    # Safety fallback if dataset is very small
    if len(data) < start + step:
        start = max(int(len(data) * 0.5), 100)
        step = max(int(len(data) * 0.1), 25)

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        if len(test) == 0:
            continue
        predictions = predict_rf(train, test, predictors, model)
        all_predictions.append(predictions)
        
    if not all_predictions:
        return pd.DataFrame()
        
    return pd.concat(all_predictions)


def run_rf_pipeline(df):
    """
    Main entry point for pipeline_utils.py.
    Returns:
        rf_signal: str ("BUY" or "SELL")
        precision: float (historical precision score)
        predictions: pd.DataFrame (backtest results for plotting)
    """
    model = RandomForestClassifier(n_estimators=300, min_samples_split=25, random_state=1)
    
    # 1. Build features
    rf_data, predictors = engineer_rf_features(df)
    
    # 2. Backtest for historical accuracy
    predictions = backtest_rf(rf_data, model, predictors)
    precision = 0.0
    if not predictions.empty:
        precision = precision_score(predictions["Target"], predictions["Predictions"])
        
    # 3. Train on complete dataset for today's signal
    model.fit(rf_data[predictors], rf_data["Target"])
    
    # Get last row (which contains today's predictors)
    latest_row = rf_data[predictors].iloc[-1:]
    today_pred = model.predict(latest_row)[0]
    
    signal = "UP ↑" if today_pred == 1 else "DOWN ↓"
    
    return signal, precision, predictions
