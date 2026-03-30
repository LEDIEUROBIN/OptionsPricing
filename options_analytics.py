from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import norm


def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {k: 0.0 for k in ["Price", "Delta", "Gamma", "Theta", "Vega", "Vanna", "Charm", "Volga"]}
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        theta = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * pdf_d1 * np.sqrt(T) / 100
    vanna = -pdf_d1 * d2 / sigma
    charm = -pdf_d1 * (2 * r * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T)) / 365
    volga = vega * d1 * d2 / sigma
    return {
        "Price": max(0, price),
        "Delta": delta,
        "Gamma": gamma,
        "Theta": theta,
        "Vega": vega,
        "Vanna": vanna,
        "Charm": charm,
        "Volga": volga,
    }


def _clean_quote_value(value):
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(val) or val <= 0:
        return None
    return val


def get_option_quote_price(row: pd.Series, price_mode: str = 'mid'):
    """Retourne un prix exploitable a partir des quotes Yahoo."""
    bid = _clean_quote_value(row.get('bid'))
    ask = _clean_quote_value(row.get('ask'))
    last = _clean_quote_value(row.get('lastPrice'))
    mid = (bid + ask) / 2 if bid is not None and ask is not None else None

    if price_mode == 'buy':
        candidates = (ask, mid, last, bid)
    elif price_mode == 'sell':
        candidates = (bid, mid, last, ask)
    else:
        candidates = (mid, last, bid, ask)

    for candidate in candidates:
        if candidate is not None:
            return float(candidate)
    return None


def compute_iv_rank(current_iv_pct: float, hv_series: pd.Series):
    """
    IV Rank = % du temps ou IV etait inferieure a l'IV actuelle sur 52 semaines.
    Utilise HV 30j rolling comme proxy de l'IV historique.
    """
    if hv_series.empty:
        return None, None, None
    iv_min = float(hv_series.min())
    iv_max = float(hv_series.max())
    iv_rank = (current_iv_pct - iv_min) / (iv_max - iv_min) * 100 if iv_max > iv_min else 50.0
    iv_pct = float((hv_series < current_iv_pct).mean() * 100)
    return round(iv_rank, 1), round(iv_pct, 1), round(iv_min, 1), round(iv_max, 1)


def compute_implied_move(calls_df, puts_df, spot):
    """
    Implied Move = (ATM Call + ATM Put) x 0.85.
    Retourne le move en $ et en %.
    """
    try:
        calls_c = calls_df.dropna(subset=['strike']).copy()
        puts_c = puts_df.dropna(subset=['strike']).copy()
        atm_call_row = calls_c.iloc[(calls_c['strike'] - spot).abs().argsort()[:1]]
        atm_put_row = puts_c.iloc[(puts_c['strike'] - spot).abs().argsort()[:1]]
        atm_call_price = get_option_quote_price(atm_call_row.iloc[0], 'mid')
        atm_put_price = get_option_quote_price(atm_put_row.iloc[0], 'mid')
        if atm_call_price is None or atm_put_price is None:
            return None, None, None
        straddle = atm_call_price + atm_put_price
        move_usd = straddle * 0.85
        move_pct = move_usd / spot * 100
        return round(move_usd, 2), round(move_pct, 2), round(straddle, 2)
    except Exception:
        return None, None, None


def compute_skew_data(calls_df, puts_df, spot, T, r_rate):
    """Calcule le skew de volatilite implicite par strike."""
    try:
        calls = calls_df.dropna(subset=['impliedVolatility', 'strike']).copy()
        puts = puts_df.dropna(subset=['impliedVolatility', 'strike']).copy()
        calls = calls[(calls['strike'] > spot * 0.75) & (calls['strike'] < spot * 1.25)]
        puts = puts[(puts['strike'] > spot * 0.75) & (puts['strike'] < spot * 1.25)]
        calls['moneyness'] = (calls['strike'] - spot) / spot * 100
        puts['moneyness'] = (puts['strike'] - spot) / spot * 100
        calls['iv_pct'] = calls['impliedVolatility'] * 100
        puts['iv_pct'] = puts['impliedVolatility'] * 100

        atm_mask_c = calls['moneyness'].abs() < 3
        atm_iv = calls[atm_mask_c]['iv_pct'].mean() if atm_mask_c.sum() > 0 else calls['iv_pct'].median()

        otm_put = puts[puts['moneyness'] < -4]['iv_pct'].mean() if (puts['moneyness'] < -4).sum() > 0 else None
        otm_call = calls[calls['moneyness'] > 4]['iv_pct'].mean() if (calls['moneyness'] > 4).sum() > 0 else None
        if otm_put is not None and otm_call is not None and not np.isnan(otm_put) and not np.isnan(otm_call):
            skew_val = otm_put - otm_call
        else:
            skew_val = None

        put_25d = puts[puts['moneyness'].between(-8, -3)]['iv_pct'].mean()
        call_25d = calls[calls['moneyness'].between(3, 8)]['iv_pct'].mean()
        if not np.isnan(put_25d) and not np.isnan(call_25d):
            risk_reversal = call_25d - put_25d
        else:
            risk_reversal = None

        return calls, puts, atm_iv, skew_val, risk_reversal, otm_put, otm_call
    except Exception:
        return None, None, None, None, None, None, None


def compute_pop(spot, strike, T, r_rate, iv, option_type, direction='long', premium=None):
    """
    Probability of Profit basee sur la distribution log-normale BSM.
    Long Call profitable si S_exp > strike + prime.
    Long Put profitable si S_exp < strike - prime.
    """
    try:
        fair_value = calculate_greeks(spot, strike, T, r_rate, iv, option_type)['Price']
        price = fair_value if premium is None else float(premium)
        if direction == 'long':
            if option_type == 'call':
                breakeven = strike + price
                d = (np.log(spot / breakeven) + (r_rate - 0.5 * iv**2) * T) / (iv * np.sqrt(T))
                pop = float(norm.cdf(d)) * 100
            else:
                breakeven = strike - price
                d = (np.log(spot / breakeven) + (r_rate - 0.5 * iv**2) * T) / (iv * np.sqrt(T))
                pop = float(norm.cdf(-d)) * 100
        else:
            if option_type == 'call':
                breakeven = strike + price
                d = (np.log(spot / breakeven) + (r_rate - 0.5 * iv**2) * T) / (iv * np.sqrt(T))
                pop = float(norm.cdf(-d)) * 100
            else:
                breakeven = strike - price
                d = (np.log(spot / breakeven) + (r_rate - 0.5 * iv**2) * T) / (iv * np.sqrt(T))
                pop = float(norm.cdf(d)) * 100
        model_edge = ((fair_value - price) if direction == 'long' else (price - fair_value)) * 100
        return round(pop, 1), round(price, 2), round(model_edge, 2), round(fair_value, 2)
    except Exception:
        return None, None, None, None


def compute_options_flow(calls_df, puts_df, spot, T, r_rate):
    """
    Detecte les trades anormaux : Volume/OI eleve, gros notionnel, OTM suspects.
    Retourne un DataFrame des signaux classes par importance.
    """
    try:
        calls = calls_df.dropna(subset=['impliedVolatility', 'volume', 'openInterest']).copy()
        puts = puts_df.dropna(subset=['impliedVolatility', 'volume', 'openInterest']).copy()
        calls['type'] = 'CALL'
        puts['type'] = 'PUT'
        df = pd.concat([calls, puts], ignore_index=True)
        df = df[df['volume'] > 0].copy()

        df['vol_oi_ratio'] = df['volume'] / df['openInterest'].replace(0, np.nan)
        df['notional'] = df['volume'] * df['lastPrice'].fillna(0) * 100
        df['moneyness'] = (df['strike'] - spot) / spot * 100
        df['is_otm'] = ((df['type'] == 'CALL') & (df['strike'] > spot)) | ((df['type'] == 'PUT') & (df['strike'] < spot))
        df['iv_pct'] = df['impliedVolatility'] * 100

        vol_mean = df['volume'].mean()
        vol_std = df['volume'].std()
        df['z_volume'] = (df['volume'] - vol_mean) / (vol_std + 1e-9)

        df['anomaly_score'] = (
            df['z_volume'].clip(0, 5) * 2
            + df['vol_oi_ratio'].fillna(0).clip(0, 20) * 1.5
            + df['is_otm'].astype(int) * 1
        )

        df = df[df['volume'] >= df['volume'].median()]
        df = df.sort_values('anomaly_score', ascending=False).head(20)

        def heat(score):
            if score > 12:
                return '!!!'
            if score > 7:
                return '!!'
            if score > 3:
                return '!'
            return ''

        df['heat'] = df['anomaly_score'].apply(heat)

        def interpret(row):
            if row['type'] == 'CALL':
                if row['strike'] > spot * 1.05:
                    return 'Pari haussier agressif (call OTM)'
                if row['strike'] < spot * 0.95:
                    return 'Couverture ou spread (call ITM)'
                return 'Pari haussier ATM'
            if row['strike'] < spot * 0.95:
                return 'Pari baissier agressif (put OTM)'
            if row['strike'] > spot * 1.05:
                return 'Couverture ou spread (put ITM)'
            return 'Protection baissiere ATM'

        df['interpretation'] = df.apply(interpret, axis=1)

        return df[
            [
                'type',
                'strike',
                'moneyness',
                'volume',
                'openInterest',
                'vol_oi_ratio',
                'notional',
                'iv_pct',
                'heat',
                'anomaly_score',
                'interpretation',
            ]
        ].reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def calc_gamma_bs(S, K, T, sigma, r=0.04):
    """Calcule le gamma Black-Scholes."""
    try:
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0.0
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))
    except Exception:
        return 0.0


def compute_roll_metrics(
    curr_row: pd.Series,
    new_row: pd.Series,
    current_expiry: str,
    new_expiry: str,
    strike: float,
    option_type: str,
    spot: float,
    r_rate: float,
    position_side: str = 'long',
):
    curr_side = 'sell' if position_side == 'long' else 'buy'
    new_side = 'buy' if position_side == 'long' else 'sell'
    curr_exec = get_option_quote_price(curr_row, curr_side)
    new_exec = get_option_quote_price(new_row, new_side)
    if curr_exec is None or new_exec is None:
        return None

    curr_iv = float(curr_row['impliedVolatility'])
    new_iv = float(new_row['impliedVolatility'])

    T_curr = max(1 / 365, (datetime.strptime(current_expiry, "%Y-%m-%d") - datetime.now()).days / 365.0)
    T_new = max(1 / 365, (datetime.strptime(new_expiry, "%Y-%m-%d") - datetime.now()).days / 365.0)

    sign = 1 if position_side == 'long' else -1
    curr_theta = calculate_greeks(spot, strike, T_curr, r_rate, curr_iv, option_type)['Theta'] * sign
    new_theta = calculate_greeks(spot, strike, T_new, r_rate, new_iv, option_type)['Theta'] * sign
    curr_delta = calculate_greeks(spot, strike, T_curr, r_rate, curr_iv, option_type)['Delta'] * sign
    new_delta = calculate_greeks(spot, strike, T_new, r_rate, new_iv, option_type)['Delta'] * sign

    roll_cost = new_exec - curr_exec if position_side == 'long' else curr_exec - new_exec
    days_gained = (datetime.strptime(new_expiry, "%Y-%m-%d") - datetime.strptime(current_expiry, "%Y-%m-%d")).days

    return {
        'curr_price': curr_exec,
        'new_price': new_exec,
        'roll_cost': roll_cost,
        'days_gained': days_gained,
        'curr_iv': round(curr_iv * 100, 1),
        'new_iv': round(new_iv * 100, 1),
        'curr_theta': curr_theta,
        'new_theta': new_theta,
        'curr_delta': curr_delta,
        'new_delta': new_delta,
        'iv_change': round((new_iv - curr_iv) * 100, 2),
        'position_side': position_side,
    }


__all__ = [
    'calc_gamma_bs',
    'calculate_greeks',
    'compute_implied_move',
    'compute_iv_rank',
    'compute_options_flow',
    'compute_pop',
    'compute_roll_metrics',
    'compute_skew_data',
    'get_option_quote_price',
]
