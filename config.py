# config.py
import os

# Telegram
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"

# Risk Management
TOTAL_CASH = 10000
MAX_POSITIONS = 1
TAKE_PROFIT_POINTS = 40
MAX_HOLD_MINUTES = 120  # 2 hours
THETA_THRESHOLD = 0.25  # Max acceptable theta decay ratio
MIN_OI = 100000         # Minimum Open Interest for liquidity

# MongoDB
MONGO_URI = "mongodb+srv://singhrajeev1470_db_user:kaPh8sxuaVFWWsSr@cluster0.mtmtbrr.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0&tls=true"

# Time Rules
START_TIME = (9, 30)   # Start scanning at 9:30 AM
EXIT_TIME = (15, 0)    # Force exit by 3:00 PM
