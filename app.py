from flask import Flask, render_template
import yfinance as yf
import numpy as np
from scipy.stats import norm

app = Flask(__name__)

# Black-Scholes Formula for Call Option
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + sigma**2 / 2.) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Fetch live SPX data
def fetch_spx():
    spx = yf.Ticker("^GSPC")
    data = spx.history(period='15d', interval='1d')
    latest_close = data['Close'][-1]
    atr = np.mean(data['High'] - data['Low'])
    return latest_close, atr

@app.route('/')
def index():
    S, atr = fetch_spx()

    # Strategy Calculation
    safe_level = S + (atr * 1.5)

    # Round to nearest 5
    short_strike = 5 * round(safe_level / 5)
    long_strike = short_strike + 20  # 20-point spread

    # Black-Scholes Parameters
    T = 1/252  # 1 trading day expiration
    r = 0.05  # Approximate risk-free rate
    sigma = (atr / S) * np.sqrt(252)  # Approximate annual volatility

    short_premium = black_scholes_call(S, short_strike, T, r, sigma)
    long_premium = black_scholes_call(S, long_strike, T, r, sigma)
    credit = short_premium - long_premium

    result = {
        'SPX Price': round(S, 2),
        'ATR': round(atr, 2),
        'Short Strike': short_strike,
        'Long Strike': long_strike,
        'Premium Collected ($)': round(credit * 100, 2)
    }

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
