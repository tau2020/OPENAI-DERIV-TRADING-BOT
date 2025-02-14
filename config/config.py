import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# -------------------- Deriv API Configuration --------------------
DERIV_APP_ID = os.getenv("DERIV_APP_ID")
DERIV_API_TOKEN = os.getenv("DERIV_API_TOKEN")

if DERIV_APP_ID is None or DERIV_API_TOKEN is None:
    raise ValueError("DERIV_APP_ID and DERIV_API_TOKEN must be set in environment variables.")

DERIV_API_URL = f"wss://ws.binaryws.com/websockets/v3?app_id={DERIV_APP_ID}"

# -------------------- API Keys --------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
EODHD_API_KEY = os.getenv("EODHD_API_KEY")
POLYGONIO_API_KEY = os.getenv("POLYGONIO_API_KEY")

# -------------------- Feature Flags --------------------
# Convert the FETCH_CANDLES_FOR_VOLUME value to a boolean.
FETCH_CANDLES_FOR_VOLUME = os.getenv("FETCH_CANDLES_FOR_VOLUME", "True").lower() in ("true", "1", "yes")

# -------------------- Trading Parameters --------------------
SYMBOL = os.getenv("SYMBOL", "R_10")
AMOUNT = float(os.getenv("AMOUNT", "10"))
DURATION = int(os.getenv("DURATION", "180"))
CURRENCY = os.getenv("CURRENCY", "USD")

# Data window for strategy indicators (e.g. number of candles to maintain)
DATA_WINDOW = int(os.getenv("DATA_WINDOW", "500"))

# -------------------- Strategy Parameters --------------------
MAX_TICK_DATA_POINTS = int(os.getenv("MAX_TICK_DATA_POINTS", "1000"))
CHART_UPDATE_INTERVAL = int(os.getenv("CHART_UPDATE_INTERVAL", "60"))
STRATEGY_EVALUATION_INTERVAL = int(os.getenv("STRATEGY_EVALUATION_INTERVAL", "15"))
MIN_DATA_POINTS_FOR_STRATEGY = int(os.getenv("MIN_DATA_POINTS_FOR_STRATEGY", "200"))
STOP_LOSS = float(os.getenv("STOP_LOSS", "5"))
TAKE_PROFIT = float(os.getenv("TAKE_PROFIT", "10"))
MONITOR_INTERVAL = int(os.getenv("MONITOR_INTERVAL", "5"))

# -------------------- Historical Data Configuration --------------------
HISTORICAL_DATA_DURATION = int(os.getenv("HISTORICAL_DATA_DURATION", "600"))
GRANULARITY = int(os.getenv("GRANULARITY", "600"))
TICKS_HISTORY_COUNT = int(os.getenv("TICKS_HISTORY_COUNT", "5000"))

# -------------------- Trade Limits --------------------
MAX_STAKE = float(os.getenv("MAX_STAKE", "2000"))
# The risk manager expects a count for consecutive losses. Use an integer.
MAX_CONSECUTIVE_LOSSES = int(os.getenv("MAX_CONSECUTIVE_LOSSES", "10"))
DAILY_LOSS_LIMIT_PERCENT = float(os.getenv("DAILY_LOSS_LIMIT_PERCENT", "0.05"))
DAILY_PROFIT_TARGET_PERCENT = float(os.getenv("DAILY_PROFIT_TARGET_PERCENT", "0.03"))
# Maximum daily drawdown as a fraction (e.g. 3% drawdown)
MAX_DAILY_DRAWDOWN = float(os.getenv("MAX_DAILY_DRAWDOWN", "0.03"))
# Maximum risk per trade (if used by signal generation or risk manager)
MAX_RISK_PER_TRADE = float(os.getenv("MAX_RISK_PER_TRADE", "0.02"))

# -------------------- Logging Configuration --------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "logs/trading_bot.log")

# -------------------- API Rate Limiting --------------------
MAX_API_CALLS_PER_SECOND = int(os.getenv("MAX_API_CALLS_PER_SECOND", "5"))
