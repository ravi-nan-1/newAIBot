# main.py
import ssl
import certifi
import pyotp
import os
import pandas as pd
import datetime as dt
import numpy as np
import requests
import math
import threading
import pytz
import time
import joblib
from pymongo import MongoClient
from sklearn.ensemble import RandomForestClassifier
import pandas_ta as ta
from py5paisa import FivePaisaClient
from py5paisa.order import Order, OrderType, Exchange

# Import local modules
from telegram_alert import send_telegram_message
from state_manager import load_active_trades, save_active_trade, remove_active_trade, get_active_trade
import config

# SSL Context
ssl_context = ssl.create_default_context(cafile=certifi.where())

# Timezone
UTC = pytz.timezone('Asia/Kolkata')

# Load Tickers
ticker_path = os.path.join(os.path.dirname(__file__), "tickers.json")
if os.path.exists(ticker_path):
    with open(ticker_path, "r") as f:
        TICKERS = json.load(f)
else:
    raise FileNotFoundError("tickers.json not found!")

# Initialize 5Paisa Client
import auth
client = FivePaisaClient(cred=auth.cred)
client.get_totp_session(auth.client_id, pyotp.TOTP(auth.token).now(), auth.pin)

# MongoDB
client1 = MongoClient(config.MONGO_URI, tls=True, tlsCAFile=certifi.where())
db = client1['AlgoTradingDB']

# Load AI Model
MODEL_PATH = "trade_filter_model.pkl"
if os.path.exists(MODEL_PATH):
    AI_MODEL = joblib.load(MODEL_PATH)
    FEATURES = ['RSI', 'ADX', 'VO', 'TII', 'TII_Signal1', 'TII_Signal2',
                'BB_width', 'EMA', 'EMA20', 'EMA50', 'upper_wick', 'lower_wick', 'body']
else:
    AI_MODEL = None
    FEATURES = []

# Read Instrument Master
instrument_df = pd.read_csv('ScripMasterfno.csv')
instrument_df = instrument_df[(instrument_df.Exch == 'N')]

# Utility Functions
def scripcode_lookup(instrument, symbol):
    try:
        return instrument[instrument.Name == symbol].ScripCode.values[0]
    except:
        return -1

def opt_exp(ticker):
    dates = instrument_df[
        (instrument_df.SymbolRoot == ticker) & ((instrument_df.ScripType == 'CE') | (instrument_df.ScripType == 'PE'))]
    dates = dates['Expiry'].unique().tolist()
    dates = [dt.datetime.strptime(date, '%Y-%m-%d') for date in dates]
    today = dt.datetime.today()
    future_dates = [date for date in dates if date > today]
    if future_dates:
        return future_dates[0].strftime('%d %b %Y')
    return None

def process_expiry_date(date_str):
    timestamp = int(date_str.split('(')[1].split('+')[0])
    date = dt.datetime.utcfromtimestamp(timestamp / 1000.0)
    return timestamp, date.strftime('%d %b %Y')

def volume_oscillator(df, fast=14, slow=28):
    ema_fast = df['Volume'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Volume'].ewm(span=slow, adjust=False).mean()
    vo = ((ema_fast - ema_slow) / ema_slow) * 100
    return vo

# Fetch Option Data with Greeks
def fetch_option_data(option_string):
    parts = option_string.split()
    ticker = parts[0]
    expiry = f"{parts[1]} {parts[2]} {parts[3]}"
    opttype = parts[4]
    strike = float(parts[5])

    expiry1 = opt_exp("NIFTY")
    if not expiry1:
        return None

    a = client.get_expiry("N", ticker)
    spot_price = a['lastrate'][0]['LTP'] if 'lastrate' in a and len(a['lastrate']) > 0 else 0

    expiry_list = pd.DataFrame(a['Expiry'])
    expiry_list['Timestamp'], expiry_list['Format'] = zip(*expiry_list['ExpiryDate'].apply(process_expiry_date))
    timestamp_row = expiry_list[expiry_list.Format == expiry1]
    if timestamp_row.empty:
        return None
    timestamp = timestamp_row.Timestamp.values[0]

    option_chain = client.get_option_chain("N", ticker, timestamp)
    option_chain = pd.DataFrame(option_chain['Options'])
    option_chain = option_chain[(option_chain.CPType == opttype) & (option_chain.LastRate != 0)]
    option_chain = option_chain[option_chain.StrikeRate == strike]

    if option_chain.empty:
        return None

    startTime = dt.datetime.today()
    date_obj = dt.datetime.strptime(expiry, "%d %b %Y")
    daysToExpiry = max((date_obj - startTime).days, 1)

    opt_data = pd.DataFrame({
        'SPOT': [spot_price],
        'STRIKE': [strike],
        f'{opttype}_LTP': option_chain['LastRate'].values,
        'OI': option_chain['OpenInterest'].values,
        'SYMBOL': option_chain['Name'].values
    })

    Delta, Gamma, Theta, IV = [], [], [], []
    r = 10

    for i in range(len(opt_data)):
        c = mb.BS([opt_data['SPOT'][i], opt_data['STRIKE'][i], r, daysToExpiry],
                  callPrice=opt_data[f'{opttype}_LTP'][i]) if opttype == 'CE' else \
            mb.BS([opt_data['SPOT'][i], opt_data['STRIKE'][i], r, daysToExpiry],
                  putPrice=opt_data[f'{opttype}_LTP'][i])
        civ = c.impliedVolatility
        cg = mb.BS([opt_data['SPOT'][i], opt_data['STRIKE'][i], r, daysToExpiry], volatility=civ)

        if opttype == 'CE':
            Delta.append(cg.callDelta * 100)
            Theta.append(cg.callTheta)
        else:
            Delta.append(cg.putDelta * 100)
            Theta.append(cg.putTheta)

        Gamma.append(cg.gamma * 100)
        IV.append(civ)

    opt_data[f'{opttype}_Delta'] = Delta
    opt_data[f'{opttype}_Gamma'] = Gamma
    opt_data[f'{opttype}_Theta'] = Theta
    opt_data['Implied_Volatility'] = IV
    opt_data["inserted_at"] = dt.datetime.now(UTC).strftime("%d-%b-%Y %I:%M%p")

    collection_name = 'NIFTY_CE' if opttype == 'CE' else 'NIFTY_PE'
    db[collection_name].insert_many(opt_data.to_dict('records'))

    return opt_data

# SuperTrend + AI Signal Engine
def super_trend(symbol, data):
    parts = symbol.split()
    opttype = parts[4]

    # Indicators
    data['EMA'] = ta.ema(data['Close'], length=5)
    data['EMA20'] = ta.ema(data['Close'], length=20)
    data['EMA3'] = ta.ema(data['Close'], length=3)
    data['EMA50'] = ta.ema(data['Close'], length=50)
    data['box_high'] = data['High'].rolling(5).max()
    data['box_low'] = data['Low'].rolling(5).min()
    data['ADX'] = ta.adx(data['High'], data['Low'], data['Close'], length=14)['ADX_14']
    data['RSI'] = ta.rsi(data['Close'], length=14)
    data['VO'] = volume_oscillator(data, fast=10, slow=20)

    bb = ta.bbands(data['Close'], length=20, std=1)
    data['BB_upper'] = bb['BBU_20_2.0_2.0']
    data['BB_lower'] = bb['BBL_20_2.0_2.0']
    data['BB_width'] = data['BB_upper'] - data['BB_lower']

    data['body'] = data['Close'] - data['Open']
    data['range'] = data['High'] - data['Low']
    data['upper_wick'] = data['High'] - data[['Close', 'Open']].max(axis=1)
    data['lower_wick'] = data[['Close', 'Open']].min(axis=1) - data['Low']

    src = data['Close']
    per = 34
    eper = 5
    eper2 = 21
    av = src.rolling(per).mean()
    dev = src - av
    udev = dev.where(dev > 0, 0).abs()
    ddev = dev.where(dev < 0, 0).abs()
    sudev = udev.rolling(per // 2).sum()
    sddev = ddev.rolling(per // 2).sum()
    data['TII'] = (100 * sudev) / (sudev + sddev)
    data['TII_Signal1'] = data['TII'].ewm(span=eper, adjust=False).mean()
    data['TII_Signal2'] = data['TII'].ewm(span=eper2, adjust=False).mean()

    # Signal Conditions
    strong_bullish_candle = (
        (data['body'] > 0) &
        (data['body'] > 0.6 * data['range']) &
        (data['upper_wick'] < 0.3 * data['body']) &
        (data['lower_wick'] < 0.3 * data['body'])
    )
    tii_filter = (
        (data['TII'] > data['TII_Signal1']) &
        (data['TII'] > 2) &
        (data['RSI'] > 50)
    )

    data['st_sig'] = np.where(tii_filter | strong_bullish_candle, 1, 0)
    data['signal_reason'] = np.where(tii_filter, 'Branch1_TII_Bullish',
                                     np.where(strong_bullish_candle, 'Branch2_StrongCandle', ''))

    # AI Filter
    if AI_MODEL is not None and all(f in data.columns for f in FEATURES):
        data[FEATURES] = data[FEATURES].fillna(0)
        predictions = AI_MODEL.predict(data[FEATURES])
        prediction_proba = AI_MODEL.predict_proba(data[FEATURES])[:, 1]
        data['ai_confidence'] = prediction_proba
        data['final_signal'] = np.where((data['st_sig'] == 1) & (prediction_proba > 0.6), 1, 0)
    else:
        data['final_signal'] = data['st_sig']

    inserted_at = dt.datetime.now(UTC).strftime("%d-%b-%Y %I:%M%p")
    data['inserted_at'] = inserted_at

    # Merge with Option Data
    last_row_main = data.tail(1).copy()
    opt_data = fetch_option_data(symbol)
    if opt_data is not None and not opt_data.empty:
        last_row_opt = opt_data.tail(1).copy()
        last_row_opt['inserted_at'] = inserted_at
        prefix = "CE_" if opttype == 'CE' else "PE_"
        last_row_opt_prefixed = last_row_opt.add_prefix(prefix)

        merged_row = pd.concat([last_row_main.reset_index(drop=True),
                                last_row_opt_prefixed.reset_index(drop=True)], axis=1)

        # Add Greeks features
        if opttype == 'CE':
            merged_row['theta_ratio'] = abs(merged_row['CE_CE_Theta'].iloc[0]) / merged_row['CE_CE_LTP'].iloc[0] if merged_row['CE_CE_LTP'].iloc[0] > 0 else 999
            merged_row['delta'] = merged_row['CE_CE_Delta'].iloc[0]
            merged_row['oi'] = merged_row['CE_OI'].iloc[0]
            merged_row['iv_percentile'] = merged_row['CE_Implied_Volatility'].iloc[0]
        else:
            merged_row['theta_ratio'] = abs(merged_row['PE_PE_Theta'].iloc[0]) / merged_row['PE_PE_LTP'].iloc[0] if merged_row['PE_PE_LTP'].iloc[0] > 0 else 999
            merged_row['delta'] = merged_row['PE_PE_Delta'].iloc[0]
            merged_row['oi'] = merged_row['PE_OI'].iloc[0]
            merged_row['iv_percentile'] = merged_row['PE_Implied_Volatility'].iloc[0]

        collection = 'All_NIFTY_CE' if opttype == 'CE' else 'All_NIFTY_PE'
        db[collection].insert_many(merged_row.to_dict('records'))

    return data[['final_signal', 'signal_reason']]

# Real-Time Monitoring Engine
def monitor_trade_dynamic(symbol, current_spot, current_option_data, trade_state):
    instructions = []
    action = "HOLD"

    entry_price = trade_state['entry_price']
    current_sl = trade_state['current_sl']
    target_price = trade_state['target_price']
    greeks = trade_state['greeks']
    entry_time = dt.datetime.strptime(trade_state['entry_time'], "%d-%b-%Y %I:%M%p")
    time_elapsed = (dt.datetime.now() - entry_time).total_seconds() / 60

    # 1. STOP LOSS HIT
    if current_spot <= current_sl:
        instructions.append("🚨 SPOT HIT STOP LOSS → EXIT NOW")
        action = "EXIT"
        return "\n".join(instructions), action

    # 2. TARGET HIT
    if current_spot >= target_price:
        instructions.append("🎯 TARGET HIT → EXIT FULL POSITION")
        action = "EXIT"
        return "\n".join(instructions), action

    # 3. TIME DECAY KILL SWITCH
    current_theta_ratio = current_option_data.get('theta_ratio', 0)
    if current_theta_ratio > config.THETA_THRESHOLD:
        instructions.append(f"⏳ THETA DECAY HIGH ({current_theta_ratio:.1%}) → CONSIDER EARLY EXIT")

    # 4. OI DROP → Smart money exiting?
    current_oi = current_option_data.get('oi', 0)
    if current_oi < greeks.get('entry_oi', 0) * 0.8:
        instructions.append("📉 OI DROPPED 20%+ → BEARISH SENTIMENT")

    # 5. DELTA COLLAPSING
    current_delta = current_option_data.get('delta', 50)
    if current_delta < greeks.get('entry_delta', 50) * 0.7:
        instructions.append("🔻 DELTA COLLAPSING → LOSS OF MOMENTUM")

    # 6. TRAILING SL (based on profit)
    profit_pct = (current_spot - entry_price) / entry_price
    if profit_pct > 0.02:  # 2% profit
        new_sl = entry_price + (profit_pct * entry_price * 0.5)  # Trail 50% of profit
        if new_sl > current_sl:
            trade_state['current_sl'] = new_sl
            instructions.append(f"🛡 SL TRAILED TO ₹{new_sl:.2f}")

    # 7. MAX HOLD TIME
    if time_elapsed > config.MAX_HOLD_MINUTES:
        instructions.append("⏰ MAX HOLD TIME REACHED → EXIT NOW")
        action = "EXIT"

    if not instructions:
        instructions.append("✅ HOLD – TREND STILL VALID")

    instructions.append(f"📊 Spot: ₹{current_spot} | SL: ₹{current_sl} | TGT: ₹{target_price}")
    return "\n".join(instructions), action

# WebSocket Streaming
spot_prices = {ticker: None for ticker in TICKERS}
req_list = [{"Exch": "N", "ExchType": "D", "ScripCode": str(scripcode_lookup(instrument_df, ticker))} for ticker in TICKERS]

def on_message(ws, message):
    global spot_prices
    try:
        data = json.loads(message)
        if isinstance(data, list) and len(data) > 0:
            token = data[0].get('Token')
            last_rate = data[0].get('LastRate')
            matching = instrument_df[instrument_df['ScripCode'] == token]
            if not matching.empty:
                symbol = matching.iloc[0]['Name']
                spot_prices[symbol] = last_rate
    except Exception as e:
        print("WebSocket Error:", e)

# UI Pusher
def send_to_ui(symbol: str, price: float):
    try:
        res = requests.post("https://algotrading-1-dluo.onrender.com/update-ticker", json={"symbol": symbol, "price": price})
        if res.status_code != 200:
            print(f"UI Update Failed for {symbol}: {res.status_code}")
    except Exception as e:
        print(f"UI Error: {e}")

# Main Bot Loop
def main():
    print("🚀 Starting Enhanced AI Options Bot...")
    
    # Start WebSocket
    req_data = client.Request_Feed('mf', 's', req_list)
    client.connect(req_data)
    streaming_thread = threading.Thread(target=lambda: client.receive_data(on_message))
    streaming_thread.daemon = True
    streaming_thread.start()

    time.sleep(5)

    # Wait for market open
    now = dt.datetime.now(UTC)
    market_start = now.replace(hour=config.START_TIME[0], minute=config.START_TIME[1], second=0, microsecond=0)
    if now < market_start:
        wait_sec = (market_start - now).total_seconds()
        print(f"⏳ Waiting {int(wait_sec)} seconds until market open...")
        time.sleep(wait_sec)

    print("✅ Market Open — Bot Active!")

    while dt.datetime.now(UTC).time() < dt.time(config.EXIT_TIME[0], config.EXIT_TIME[1]):
        try:
            for symbol in TICKERS:
                current_spot = spot_prices.get(symbol)
                if current_spot is None:
                    continue

                send_to_ui(symbol, current_spot)

                # Fetch latest data
                data_fut = get_cash_market_data(symbol, '3m')
                if len(data_fut) < 10:
                    continue

                data_fut.drop(data_fut.tail(1).index, inplace=True)
                signal_data = super_trend(symbol, data_fut)

                if signal_data is None or len(signal_data) == 0:
                    continue

                latest_signal = signal_data['final_signal'].iloc[-1]

                # ENTRY LOGIC
                if latest_signal == 1:
                    active_trades = load_active_trades()
                    if len(active_trades) >= config.MAX_POSITIONS:
                        print("🛑 Max positions reached.")
                        continue
                    if symbol in active_trades:
                        print(f"🔁 {symbol} already in active trade.")
                        continue

                    # Get latest option data
                    coll_name = "All_NIFTY_CE" if "CE" in symbol else "All_NIFTY_PE"
                    cursor = db[coll_name].find({"CE_SYMBOL" if "CE" in symbol else "PE_SYMBOL": symbol}).sort("inserted_at", -1).limit(1)
                    opt_docs = list(cursor)
                    if not opt_docs:
                        continue

                    opt_data = opt_docs[0]
                    theta_ratio = opt_data.get('theta_ratio', 999)
                    oi = opt_data.get('CE_OI' if "CE" in symbol else 'PE_OI', 0)
                    delta = opt_data.get('CE_CE_Delta' if "CE" in symbol else 'PE_PE_Delta', 50)

                    if theta_ratio > config.THETA_THRESHOLD or oi < config.MIN_OI or delta < 40:
                        print(f"❌ Rejected: Theta={theta_ratio:.2%}, OI={oi}, Delta={delta}")
                        continue

                    # Enter Trade
                    entry_price = current_spot
                    target_price = entry_price + config.TAKE_PROFIT_POINTS
                    stop_loss = entry_price - 10
                    qty = 75  # Fixed for Nifty

                    trade_state = {
                        "symbol": symbol,
                        "entry_price": entry_price,
                        "entry_time": dt.datetime.now(UTC).strftime("%d-%b-%Y %I:%M%p"),
                        "target_price": target_price,
                        "current_sl": stop_loss,
                        "qty": qty,
                        "greeks": {
                            "entry_delta": delta,
                            "entry_theta_ratio": theta_ratio,
                            "entry_oi": oi,
                            "entry_iv": opt_data.get('CE_Implied_Volatility' if "CE" in symbol else 'PE_Implied_Volatility', 0),
                        },
                        "status": "OPEN"
                    }

                    save_active_trade(symbol, trade_state)
                    msg = f"🚀 *NEW TRADE*\n{symbol}\nENTRY: ₹{entry_price}\nSL: ₹{stop_loss} | TGT: ₹{target_price}\nDELTA: {delta:.1f} | THETA: {theta_ratio:.1%}"
                    send_telegram_message(msg)
                    print(msg)

                # MONITOR EXISTING TRADE
                active_trade = get_active_trade(symbol)
                if active_trade and active_trade['status'] == 'OPEN':
                    coll_name = "All_NIFTY_CE" if "CE" in symbol else "All_NIFTY_PE"
                    cursor = db[coll_name].find({"CE_SYMBOL" if "CE" in symbol else "PE_SYMBOL": symbol}).sort("inserted_at", -1).limit(1)
                    opt_docs = list(cursor)
                    if not opt_docs:
                        continue

                    opt_data = opt_docs[0]
                    instruction, action = monitor_trade_dynamic(symbol, current_spot, opt_data, active_trade)

                    send_telegram_message(f"📢 *UPDATE [{dt.datetime.now(UTC).strftime('%H:%M')}]*\n{instruction}")
                    print(instruction)

                    if action == "EXIT":
                        pnl = (current_spot - active_trade['entry_price']) * active_trade['qty']
                        msg = f"✅ *TRADE CLOSED*\n{symbol}\nP/L: ₹{pnl:.2f}"
                        send_telegram_message(msg)
                        remove_active_trade(symbol)

            time.sleep(60)  # Run every minute

        except Exception as e:
            error_msg = f"🔥 ERROR: {str(e)}"
            print(error_msg)
            send_telegram_message(error_msg)
            time.sleep(30)

    print("🔚 Market Closed — Bot Stopped.")

if __name__ == "__main__":
    main()
