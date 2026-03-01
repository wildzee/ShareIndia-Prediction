import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import xgboost as xgb
import optuna
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
import joblib

def build_lstm(sequence_length, n_features, config):
    units = config['model']['lstm']['units']
    model = Sequential([
        LSTM(units[0], return_sequences=True, input_shape=(sequence_length, n_features)),
        Dropout(config['model']['lstm']['dropout']),
        BatchNormalization(),
        LSTM(units[1], return_sequences=False),
        Dropout(config['model']['lstm']['dropout']),
        Dense(32, activation='relu'),
        Dense(2)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config['model']['lstm']['learning_rate']),
        loss='huber'
    )
    return model

def create_sequences(df, feature_cols, target_cols, seq_len):
    X, y = [], []
    valid_df = df.dropna(subset=target_cols)
    for i in range(seq_len, len(valid_df)):
        X.append(valid_df[feature_cols].iloc[i-seq_len:i].values)
        y.append(valid_df[target_cols].iloc[i])
    return np.array(X), np.array(y)

def train_lstm(df, config):
    seq_len = config['model']['lstm']['sequence_length']
    target_cols = ['target_1m', 'target_1y']
    # Explicitly exclude the target relative columns so LSTM doesn't cheat using tomorrow's answer
    feature_cols = [c for c in df.columns if c not in target_cols]

    X, y = create_sequences(df, feature_cols, target_cols, seq_len)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_lstm(seq_len, X_train.shape[2], config)

    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True),
        ReduceLROnPlateau(patience=7, factor=0.5)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=config['model']['lstm']['epochs'],
        batch_size=config['model']['lstm']['batch_size'],
        callbacks=callbacks,
        verbose=1
    )

    model.save("models/lstm_shareindia.keras")
    print("LSTM saved -> models/lstm_shareindia.keras")
    return model, history, X_test, y_test

def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'n_estimators':    trial.suggest_int('n_estimators', 200, 1000),
        'max_depth':       trial.suggest_int('max_depth', 3, 9),
        'learning_rate':   trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample':       trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha':       trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda':      trial.suggest_float('reg_lambda', 0.0, 2.0)
    }
    model = MultiOutputRegressor(xgb.XGBRegressor(**params, random_state=42))
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return mean_absolute_error(y_val, preds, multioutput='uniform_average')

def train_xgboost(df, config, n_trials=50):
    target_cols = ['target_1m', 'target_1y']
    valid_df = df.dropna(subset=target_cols)
    feature_cols = [c for c in df.columns if c not in target_cols]
    X = valid_df[feature_cols].values
    y = valid_df[target_cols].values

    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda t: objective(t, X_train, y_train, X_val, y_val),
                   n_trials=n_trials, show_progress_bar=True)

    best_model = MultiOutputRegressor(xgb.XGBRegressor(**study.best_params, random_state=42))
    best_model.fit(X_train, y_train)

    joblib.dump(best_model, "models/xgboost_shareindia.joblib")
    print(f"XGBoost saved -> models/xgboost_shareindia.joblib")
    print(f"Best MAE: {study.best_value:.4f}")
    return best_model, study.best_params

def ensemble_predict(lstm_model, xgb_model, X_lstm, X_xgb, config):
    lstm_preds = lstm_model.predict(X_lstm).flatten()
    xgb_preds  = xgb_model.predict(X_xgb)
    w_lstm = config['model']['ensemble']['lstm_weight']
    w_xgb  = config['model']['ensemble']['xgboost_weight']
    return w_lstm * lstm_preds + w_xgb * xgb_preds

def predict_price_targets(lstm_model, xgb_model, df, config, scaler):
    seq_len = config['model']['lstm']['sequence_length']
    target_cols = ['target_1m', 'target_1y']
    feature_cols = [c for c in df.columns if c not in target_cols]

    X_seq = df[feature_cols].values[-seq_len:]
    X_seq = X_seq.reshape(1, seq_len, -1)
    X_tab = df[feature_cols].values[-1:] 

    pred_raw = ensemble_predict(lstm_model, xgb_model, X_seq, X_tab, config)[0]

    current_price = df['close'].iloc[-1]
    
    # Model explicitly predicts exact forward multi-horizon returns.
    # We apply a conservatism penalty (0.6x multiplier) because long-term 
    # historical geometry often overshoots realistic future large-cap returns.
    monthly_upside = pred_raw[0] * 0.60
    yearly_upside = pred_raw[1] * 0.60
    
    # Strictly constrain anomalies from overextending
    monthly_upside = max(min(monthly_upside, 0.15), -0.20)  # Max +15% / -20% in a month
    yearly_upside = max(min(yearly_upside, 0.40), -0.50)    # Max +40% / -50% in a year

    return {
        "current_price":  round(current_price, 2),
        "monthly_target": round(current_price * (1 + monthly_upside), 2),
        "yearly_target":  round(current_price * (1 + yearly_upside), 2),
        "monthly_upside": round(monthly_upside * 100, 2),
        "yearly_upside":  round(yearly_upside * 100, 2),
    }

def generate_signal(forecast, df, config):
    rsi   = df['rsi'].iloc[-1]
    macd  = df['macd'].iloc[-1]
    macd_s = df['macd_signal'].iloc[-1]
    upside = forecast['monthly_upside'] / 100

    cfg = config['signals']
    macd_bullish = macd > macd_s
    macd_bearish = macd < macd_s

    # SELL if expected upside is poor/negative or the stock is heavily overbought and breaking down
    if upside < cfg['sell_threshold'] or (rsi > cfg['rsi_overbought'] and macd_bearish):
        signal = "SELL"
    # BUY if expected upside is solid AND (momentum is bullish OR it's oversold)
    elif upside > cfg['buy_threshold'] and (macd_bullish or rsi < cfg['rsi_oversold']):
        signal = "BUY"
    else:
        signal = "HOLD"

    return signal

def run_backtest(df, config):
    capital    = config['backtest']['initial_capital']
    pos_size   = config['backtest']['position_size']
    portfolio  = [capital]
    trades     = []
    position   = None

    signals_df = df.copy()
    signals_df['signal'] = 'HOLD'

    signals_df.loc[
        (signals_df['rsi'] < 60) & (signals_df['macd'] > signals_df['macd_signal']),
        'signal'
    ] = 'BUY'
    signals_df.loc[
        (signals_df['rsi'] > 75) & (signals_df['macd'] < signals_df['macd_signal']),
        'signal'
    ] = 'SELL'

    for i, row in signals_df.iterrows():
        if row['signal'] == 'BUY' and position is None:
            shares   = (capital * pos_size) // row['close']
            position = {'entry': row['close'], 'shares': shares, 'date': i}
            capital -= shares * row['close']
        elif row['signal'] == 'SELL' and position is not None:
            pnl     = (row['close'] - position['entry']) * position['shares']
            capital += position['shares'] * row['close']
            trades.append({'entry': position['entry'], 'exit': row['close'],
                           'pnl': pnl, 'return': pnl / (position['entry'] * position['shares'])})
            position = None
        total_val = capital + (position['shares'] * row['close'] if position else 0)
        portfolio.append(total_val)

    returns  = pd.Series(portfolio).pct_change().dropna()
    sharpe   = (returns.mean() / returns.std()) * (252 ** 0.5) if returns.std() != 0 else 0
    drawdown = (pd.Series(portfolio) / pd.Series(portfolio).cummax() - 1).min()
    win_rate = sum(1 for t in trades if t['pnl'] > 0) / max(len(trades), 1)
    total_ret = (portfolio[-1] / portfolio[0] - 1) * 100

    print(f"Total Return    : {total_ret:.1f}%")
    print(f"Sharpe Ratio    : {sharpe:.2f}")
    print(f"Max Drawdown    : {drawdown*100:.1f}%")
    
    return pd.DataFrame(trades), pd.Series(portfolio)
