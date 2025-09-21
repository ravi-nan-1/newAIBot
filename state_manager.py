# state_manager.py
import json
import os

STATE_FILE = "active_trades_state.json"

def load_active_trades():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_active_trade(symbol, trade_data):
    trades = load_active_trades()
    trades[symbol] = trade_data
    with open(STATE_FILE, 'w') as f:
        json.dump(trades, f, indent=4, default=str)

def remove_active_trade(symbol):
    trades = load_active_trades()
    if symbol in trades:
        del trades[symbol]
        with open(STATE_FILE, 'w') as f:
            json.dump(trades, f, indent=4, default=str)

def get_active_trade(symbol):
    trades = load_active_trades()
    return trades.get(symbol, None)
