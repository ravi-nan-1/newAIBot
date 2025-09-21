# telegram_alert.py
import requests
import os

def tele_msg(msg, parse_mode="Markdown"):
    try:
        from Telegram_token import telegram_token, chat_id
    except ImportError:
        telegram_token = os.getenv("TELEGRAM_TOKEN", "YOUR_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")

    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': msg,
        'parse_mode': parse_mode
    }
    try:
        response = requests.post(url, data=payload)
        if response.status_code != 200:
            print("⚠️ Telegram Alert Failed:", response.text)
    except Exception as e:
        print("⚠️ Telegram Error:", e)
