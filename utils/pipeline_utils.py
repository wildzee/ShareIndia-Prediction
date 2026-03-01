import os
from utils.data_utils import load_config, fetch_ohlcv, fetch_fundamentals, fetch_news, build_sentiment_series, build_master_features
from utils.model_utils import train_lstm, train_xgboost, predict_price_targets, generate_signal
from utils.rf_utils import run_rf_pipeline

def run_full_pipeline(ticker, company_name):
    print(f"--- Starting Pipeline for {ticker} ({company_name}) ---")
    config = load_config()
    
    # Override config with the user-selected stock
    config['stock']['ticker'] = ticker
    config['stock']['name'] = company_name

    # Step 1: Ingestion
    print("Fetching data...")
    ohlcv_df = fetch_ohlcv(config)
    fundamentals, quarterly = fetch_fundamentals(config)
    news_df = fetch_news(query=f"{company_name} OR {ticker}", max_articles=50)

    # Step 2: Feature Engineering
    print("Computing features and sentiment...")
    sentiment_df = build_sentiment_series(news_df)
    master_df, scaler = build_master_features(ohlcv_df, quarterly, sentiment_df)

    # Step 3: Model Training
    print("Training models...")
    if len(master_df) < 400:
        raise ValueError(f"Not enough historical data! {ticker} only has {len(master_df)} trading days of data on record. The AI's 1-year prediction engine requires at least 400 days of past data to train successfully. Please select an older, more established stock.")
        
    # Using 3 trials for XGBoost to keep training time fast in the web app
    lstm_model, lstm_history, X_test, y_test = train_lstm(master_df, config)
    xgb_model, xgb_params = train_xgboost(master_df, config, n_trials=3)

    # Step 4: Prediction
    print("Generating forecast...")
    forecast = predict_price_targets(lstm_model, xgb_model, master_df, config, scaler)
    signal = generate_signal(forecast, master_df, config)
    
    print("Running RandomForest Direction Predictor...")
    rf_signal, rf_precision, rf_predictions = run_rf_pipeline(master_df)

    return forecast, signal, master_df, rf_signal, rf_precision, rf_predictions
