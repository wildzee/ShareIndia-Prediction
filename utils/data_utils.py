import yfinance as yf
import pandas as pd
import numpy as np
import yaml
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
import feedparser
from datetime import datetime

vader = SentimentIntensityAnalyzer()

def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def fetch_ohlcv(config):
    ticker = config['stock']['ticker']
    start  = config['data']['start_date']
    end    = config['data']['end_date']
    if end == "today":
        end = pd.Timestamp.today().strftime("%Y-%m-%d")

    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    df.index.name = "date"
    df.dropna(inplace=True)

    out = f"{config['data']['raw_dir']}/ohlcv.csv"
    df.to_csv(out)
    print(f"OHLCV saved -> {out}  ({len(df)} rows)")
    return df

def fetch_fundamentals(config):
    ticker = yf.Ticker(config['stock']['ticker'])
    info   = ticker.info

    fundamentals = {
        'pe_ratio':        info.get('trailingPE'),
        'eps':             info.get('trailingEps'),
        'book_value':      info.get('bookValue'),
        'debt_to_equity':  info.get('debtToEquity'),
        'roe':             info.get('returnOnEquity'),
        'revenue':         info.get('totalRevenue'),
        'market_cap':      info.get('marketCap'),
        'profit_margin':   info.get('profitMargins'),
    }

    # Quarterly income statement for Revenue Growth QoQ
    quarterly = ticker.quarterly_financials.T
    quarterly.index = pd.to_datetime(quarterly.index)
    quarterly = quarterly.sort_index()

    out = f"{config['data']['raw_dir']}/fundamentals.csv"
    pd.DataFrame([fundamentals]).to_csv(out, index=False)
    quarterly.to_csv(f"{config['data']['raw_dir']}/quarterly_financials.csv")
    print(f"Fundamentals saved -> {out}")
    return fundamentals, quarterly

def fetch_news(query="Share India OR SHAREINDIA OR NSE:SHAREINDIA", max_articles=100):
    encoded = query.replace(" ", "+")
    url = f"https://news.google.com/rss/search?q={encoded}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)

    articles = []
    for entry in feed.entries[:max_articles]:
        articles.append({
            "date": datetime(*entry.published_parsed[:6]).strftime("%Y-%m-%d"),
            "title": entry.title,
            "source": entry.source.get("title", "") if hasattr(entry, "source") else "",
            "link": entry.link
        })

    df = pd.DataFrame(articles)
    df.to_csv("data/raw/news.csv", index=False)
    print(f"News saved -> data/raw/news.csv  ({len(df)} articles)")
    return df

def add_technical_indicators(df):
    df['ema_20']  = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50']  = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['sma_10']  = df['close'].rolling(window=10).mean()
    df['sma_30']  = df['close'].rolling(window=30).mean()

    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    df['rsi'] = 100 - (100 / (1 + rs))

    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

    sma_20 = df['close'].rolling(window=20).mean()
    std_20 = df['close'].rolling(window=20).std()
    df['bb_upper'] = sma_20 + 2 * std_20
    df['bb_middle'] = sma_20
    df['bb_lower'] = sma_20 - 2 * std_20

    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift()).abs()
    tr3 = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()

    df['daily_return'] = df['close'].pct_change()
    df['log_return']   = np.log(df['close'] / df['close'].shift(1))

    return df

def score_headline(text):
    return round(vader.polarity_scores(text)['compound'], 4)

def build_sentiment_series(news_df):
    if len(news_df) == 0:
        return pd.DataFrame(columns=['date', 'sentiment', 'sentiment_ma7']).set_index('date')

    news_df['sentiment'] = news_df['title'].apply(score_headline)
    daily = news_df.groupby('date')['sentiment'].mean().reset_index()
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.set_index('date').sort_index()
    daily['sentiment_ma7'] = daily['sentiment'].rolling(7, min_periods=1).mean()
    return daily

def build_master_features(ohlcv_df, fundamentals_df, sentiment_df):
    df = ohlcv_df.copy()
    df = add_technical_indicators(df)

    fundamentals_df.index = pd.to_datetime(fundamentals_df.index)
    df = df.join(fundamentals_df.resample('D').ffill(), how='left')

    df = df.join(sentiment_df, how='left')
    df['sentiment'] = df['sentiment'].fillna(0)
    df['sentiment_ma7'] = df['sentiment_ma7'].ffill()

    df.dropna(subset=['ema_200', 'rsi', 'macd'], inplace=True)
    
    # Forward fill missing fundamentals & indicator gaps, then fill 0
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)
    
    # Target 1: 21 trading days (1 month) forward return
    df['target_1m'] = df['close'].pct_change(periods=21).shift(-21)
    
    # Target 2: 252 trading days (1 year) forward return
    df['target_1y'] = df['close'].pct_change(periods=252).shift(-252)

    feature_cols = [c for c in df.columns if c not in ['close', 'open', 'high', 'low', 'target_1m', 'target_1y']]
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    df.to_csv("data/processed/features.csv")
    print(f"Master features saved -> data/processed/features.csv  ({len(df)} rows, {len(df.columns)} cols)")
    return df, scaler
