import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import options_analytics as analytics
warnings.filterwarnings('ignore')

# Shared analytics live in options_analytics.py so app.py can stay UI-focused.

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    return analytics.calculate_greeks(S, K, T, r, sigma, option_type)

# ============================================================
# DATA FETCHING
# ============================================================

@st.cache_data(ttl=300)
def get_market_context():
    """Recupere VIX, taux 10Y et les principaux indices/matieres premieres."""
    try:
        tnx = yf.Ticker("^TNX").history(period="1d")['Close'].iloc[-1] / 100
        vix = yf.Ticker("^VIX").history(period="1d")['Close'].iloc[-1]
    except:
        tnx, vix = 0.0425, 20.0

    market_tickers = {
        'SPY':  ('S&P 500',  '📈', '#00e5ff'),
        'QQQ':  ('NASDAQ',   '💻', '#b06fff'),
        'DIA':  ('DOW',      '🏭', '#f5a623'),
        'IWM':  ('RUSSEL',   '📊', '#3fb950'),
        'GLD':  ('OR',       '🥇', '#f5a623'),
        'USO':  ('PÉTROLE',  '🛢️', '#ff4b6e'),
        'TLT':  ('BONDS 20Y','🏦', '#00e5ff'),
        'UUP':  ('USD',      '💵', '#3fb950'),
        'BTC-USD': ('BITCOIN','₿', '#f7931a'),
    }
    data = {}
    for sym, (label, icon, color) in market_tickers.items():
        try:
            h = yf.Ticker(sym).history(period="2d")
            if len(h) >= 2:
                prev  = float(h['Close'].iloc[-2])
                last  = float(h['Close'].iloc[-1])
                chg   = (last - prev) / prev * 100
            elif len(h) == 1:
                last  = float(h['Close'].iloc[-1])
                chg   = 0.0
            else:
                continue
            data[sym] = {'label': label, 'icon': icon, 'color': color,
                         'price': last, 'chg': chg}
        except:
            pass
    return float(tnx), float(vix), data

@st.cache_data(ttl=300)
def get_stock_data(ticker: str):
    stock        = yf.Ticker(ticker)
    info         = stock.info
    hist         = stock.history(period="6mo")
    spot         = float(hist['Close'].iloc[-1])
    expiry_dates = list(stock.options)
    return info, hist, spot, expiry_dates

@st.cache_data(ttl=300)
def get_option_chain(ticker: str, expiry: str):
    chain = yf.Ticker(ticker).option_chain(expiry)
    return chain.calls, chain.puts

@st.cache_data(ttl=3600)
def get_iv_history(ticker: str):
    """Récupère IV ATM historique sur 52 semaines via historique de prix + HV comme proxy."""
    try:
        hist_1y = yf.Ticker(ticker).history(period="1y")
        log_ret = np.log(hist_1y['Close'] / hist_1y['Close'].shift(1)).dropna()
        # Calcul rolling HV 30j comme proxy IV historique
        hv_series = log_ret.rolling(30).std() * np.sqrt(252) * 100
        return hv_series.dropna()
    except:
        return pd.Series(dtype=float)

@st.cache_data(ttl=1800)
def get_earnings_date(ticker: str):
    """
    Recupere la prochaine date d'annonce de resultats.
    Essaie plusieurs methodes en cascade car l'API yfinance change souvent.
    """
    stock = yf.Ticker(ticker)
    today = datetime.now().date()

    # ── Methode 1 : info (earningsTimestamp) ─────────────────
    try:
        info = stock.info
        for key in ('earningsTimestamp', 'earningsTimestampStart', 'earningsTimestampEnd'):
            ts = info.get(key)
            if ts and isinstance(ts, (int, float)) and ts > 0:
                d = datetime.fromtimestamp(ts).date()
                if d >= today:
                    return d
    except Exception:
        pass

    # ── Methode 2 : calendar (dict ou DataFrame) ─────────────
    try:
        cal = stock.calendar
        if cal is not None:
            # Nouveau format : dict
            if isinstance(cal, dict):
                for key in ('Earnings Date', 'earningsDate', 'earnings_date'):
                    val = cal.get(key)
                    if val is not None:
                        if not isinstance(val, (list, tuple)):
                            val = [val]
                        dates = []
                        for v in val:
                            try:
                                d = pd.to_datetime(v)
                                if hasattr(d, 'date'):
                                    dates.append(d.date() if not callable(d.date) else d.date())
                                else:
                                    dates.append(d)
                            except Exception:
                                pass
                        future = [d for d in dates if d >= today]
                        if future:
                            return sorted(future)[0]
            # Ancien format : DataFrame
            elif isinstance(cal, pd.DataFrame):
                for key in ('Earnings Date', 'Earnings High', 'Earnings Low'):
                    if key in cal.index:
                        for val in cal.loc[key]:
                            try:
                                d = pd.to_datetime(val)
                                if not pd.isna(d):
                                    dd = d.date() if hasattr(d, "date") else d
                                    if dd >= today:
                                        return dd
                            except Exception:
                                pass
    except Exception:
        pass

    # ── Methode 3 : earnings_dates (index futur) ─────────────
    try:
        ed = stock.earnings_dates
        if ed is not None and not ed.empty:
            now_ts = pd.Timestamp.now(tz='UTC')
            idx = ed.index
            # Normalise timezone
            if idx.tzinfo is None:
                future_idx = idx[idx.normalize() >= pd.Timestamp(today)]
            else:
                future_idx = idx[idx >= now_ts]
            if not future_idx.empty:
                return future_idx[-1].date()
    except Exception:
        pass

    # ── Methode 4 : quarterly earnings history → estimer la prochaine ──
    try:
        qe = stock.quarterly_earnings
        if qe is not None and not qe.empty:
            # Les earnings sont typiquement trimestriels (~91 jours)
            last_str = str(qe.index[0])
            last_date = pd.to_datetime(last_str).date()
            next_est  = last_date + timedelta(days=91)
            if next_est >= today:
                return next_est
    except Exception:
        pass

    # ── Methode 5 : fast_info ────────────────────────────────
    try:
        fi = stock.fast_info
        for attr in ('earnings_date', 'next_earnings_date'):
            val = getattr(fi, attr, None)
            if val is not None:
                d = pd.to_datetime(val).date()
                if d >= today:
                    return d
    except Exception:
        pass

    return None

# ============================================================
# IV RANK & IMPLIED MOVE
# ============================================================

def compute_iv_rank(current_iv_pct: float, hv_series: pd.Series):
    """
    IV Rank = % du temps où IV était INFÉRIEURE à l'IV actuelle sur 52 semaines.
    Utilise HV 30j rolling comme proxy de l'IV historique.
    """
    return analytics.compute_iv_rank(current_iv_pct, hv_series)

def get_option_quote_price(row: pd.Series, price_mode: str = 'mid'):
    """Retourne un prix exploitable a partir des quotes Yahoo."""
    return analytics.get_option_quote_price(row, price_mode)

def compute_implied_move(calls_df, puts_df, spot):
    """
    Implied Move = (ATM Call + ATM Put) × 0.85
    Retourne le move en $ et en %.
    """
    return analytics.compute_implied_move(calls_df, puts_df, spot)

# ============================================================
# SKEW ANALYSIS
# ============================================================

def compute_skew_data(calls_df, puts_df, spot, T, r_rate):
    """Calcule le skew de volatilité implicite par strike."""
    return analytics.compute_skew_data(calls_df, puts_df, spot, T, r_rate)


def chart_skew(calls, puts, spot):
    """Graphique smile/skew de volatilité implicite."""
    fig = go.Figure()
    calls_s = calls.sort_values('strike')
    puts_s  = puts.sort_values('strike')

    fig.add_trace(go.Scatter(
        x=calls_s['moneyness'], y=calls_s['iv_pct'],
        name='IV Calls', mode='lines+markers',
        line=dict(color='#00e5ff', width=2.5),
        marker=dict(size=5, color='#00e5ff')))
    fig.add_trace(go.Scatter(
        x=puts_s['moneyness'], y=puts_s['iv_pct'],
        name='IV Puts', mode='lines+markers',
        line=dict(color='#ff4b6e', width=2.5),
        marker=dict(size=5, color='#ff4b6e')))

    fig.add_vline(x=0, line_dash='dot', line_color='rgba(255,255,255,0.35)', line_width=1.5,
                  annotation=dict(text='ATM', font=dict(color='#ffffff', size=11),
                                  bgcolor='rgba(0,0,0,0.6)'))
    fig.add_vrect(x0=-3, x1=3, fillcolor='rgba(255,255,255,0.02)',
                  line_width=0, annotation_text='Zone ATM',
                  annotation_font=dict(color='#606878', size=10))
    fig.update_layout(**LAYOUT, height=340,
                      xaxis={**AXIS, 'title_text': 'Moneyness (% du spot)'},
                      yaxis={**AXIS, 'title_text': 'IV Implicite (%)'})
    return fig


# ============================================================
# PROBABILITY OF PROFIT
# ============================================================

def compute_pop(spot, strike, T, r_rate, iv, option_type, direction='long', premium=None):
    """
    Probability of Profit basée sur la distribution log-normale BSM.
    Long Call profitable si S_exp > strike + prime
    Long Put  profitable si S_exp < strike - prime
    """
    return analytics.compute_pop(spot, strike, T, r_rate, iv, option_type, direction, premium)


def chart_pop_distribution(spot, strike, T, r_rate, iv, option_type, premium=None, direction='long'):
    """Distribution log-normale du prix à expiration avec zones profit/perte."""
    x = np.linspace(spot * 0.6, spot * 1.4, 300)
    mu    = np.log(spot) + (r_rate - 0.5*iv**2)*T
    sigma = iv * np.sqrt(T)
    pdf   = (1 / (x * sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*((np.log(x)-mu)/sigma)**2)
    pdf   = pdf / pdf.max()  # normalise pour affichage

    price = calculate_greeks(spot, strike, T, r_rate, iv, option_type)['Price'] if premium is None else float(premium)
    be    = (strike + price) if option_type == 'call' else (strike - price)
    is_long = direction == 'long'

    if option_type == 'call':
        profit_mask = x >= be if is_long else x <= be
    else:
        profit_mask = x <= be if is_long else x >= be
    loss_mask = ~profit_mask

    fig = go.Figure()
    # Zone perte
    x_loss = x[loss_mask]
    y_loss = pdf[loss_mask]
    fig.add_trace(go.Scatter(x=x_loss, y=y_loss, fill='tozeroy',
        fillcolor='rgba(255,75,110,0.15)', line=dict(color='rgba(255,75,110,0)', width=0),
        name='Zone perte', showlegend=True))
    # Zone profit
    x_prof = x[profit_mask]
    y_prof = pdf[profit_mask]
    fig.add_trace(go.Scatter(x=x_prof, y=y_prof, fill='tozeroy',
        fillcolor='rgba(63,185,80,0.15)', line=dict(color='rgba(63,185,80,0)', width=0),
        name='Zone profit', showlegend=True))
    # Courbe complète
    fig.add_trace(go.Scatter(x=x, y=pdf, line=dict(color='#00e5ff', width=2),
        name='Distribution prix', showlegend=False))
    # Lignes verticales
    fig.add_vline(x=spot, line_dash='dot', line_color='rgba(255,255,255,0.5)', line_width=1.5,
                  annotation=dict(text='Spot', font=dict(color='#fff', size=10)))
    fig.add_vline(x=strike, line_dash='dash', line_color='#f5a623', line_width=1.5,
                  annotation=dict(text='Strike', font=dict(color='#f5a623', size=10)))
    fig.add_vline(x=be, line_dash='dash', line_color='#3fb950', line_width=1.5,
                  annotation=dict(text='Break-even', font=dict(color='#3fb950', size=10)))
    fig.update_layout(**LAYOUT, height=280,
                      xaxis={**AXIS, 'title_text': 'Prix à expiration ($)'},
                      yaxis={**AXIS, 'title_text': 'Probabilité (normalisée)',
                             'showticklabels': False},
                      showlegend=True)
    return fig


# ============================================================
# OPTIONS FLOW
# ============================================================

def compute_options_flow(calls_df, puts_df, spot, T, r_rate):
    """
    Détecte les trades anormaux : Volume/OI élevé, gros notionnel, OTM suspects.
    Retourne un DataFrame des signaux classés par importance.
    """
    return analytics.compute_options_flow(calls_df, puts_df, spot, T, r_rate)


# ============================================================
# ROLL ANALYZER
# ============================================================

def compute_roll(ticker, current_expiry, new_expiry, strike, option_type, spot, r_rate, position_side='long'):
    """
    Calcule le coût / crédit du roll d'une position vers une nouvelle échéance.
    """
    try:
        curr_calls, curr_puts = get_option_chain(ticker, current_expiry)
        new_calls,  new_puts  = get_option_chain(ticker, new_expiry)
        curr_df = curr_calls if option_type == 'call' else curr_puts
        new_df  = new_calls  if option_type == 'call' else new_puts

        curr_row = curr_df[curr_df['strike'] == strike]
        new_row  = new_df[new_df['strike'] == strike]
        if curr_row.empty or new_row.empty:
            return None

        return analytics.compute_roll_metrics(
            curr_row.iloc[0], new_row.iloc[0], current_expiry, new_expiry,
            strike, option_type, spot, r_rate, position_side
        )
    except Exception:
        return None

# ============================================================
# SHARED LAYOUT DEFAULTS
# ============================================================

LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(10,14,20,0.95)',
    font=dict(family='Inter, sans-serif', color='#e0e6f0', size=12),
    margin=dict(l=10, r=10, t=36, b=10),
    legend=dict(orientation='h', y=1.08, x=0,
                bgcolor='rgba(0,0,0,0)', font=dict(color='#c9d1d9', size=11))
)
AXIS = dict(
    gridcolor='rgba(255,255,255,0.06)',
    zerolinecolor='rgba(255,255,255,0.12)',
    tickfont=dict(color='#8b949e', size=11),
    title_font=dict(color='#c9d1d9', size=12),
    linecolor='rgba(255,255,255,0.1)',
)

# ============================================================
# CHARTS
# ============================================================

def chart_ohlcv(hist):
    h = hist.copy()
    h['MA20'] = h['Close'].rolling(20).mean()
    h['MA50'] = h['Close'].rolling(50).mean()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28], vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(
        x=h.index, open=h['Open'], high=h['High'], low=h['Low'], close=h['Close'],
        increasing=dict(line=dict(color='#00e5ff', width=1), fillcolor='rgba(0,229,255,0.7)'),
        decreasing=dict(line=dict(color='#ff4b6e', width=1), fillcolor='rgba(255,75,110,0.7)'),
        name='Prix', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=h.index, y=h['MA20'],
                             line=dict(color='#f5a623', width=1.5), name='MA 20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=h.index, y=h['MA50'],
                             line=dict(color='#b06fff', width=1.5), name='MA 50'), row=1, col=1)
    bar_colors = ['rgba(0,229,255,0.55)' if c >= o else 'rgba(255,75,110,0.55)'
                  for c, o in zip(h['Close'], h['Open'])]
    fig.add_trace(go.Bar(x=h.index, y=h['Volume'], marker_color=bar_colors,
                         name='Volume', showlegend=False), row=2, col=1)
    fig.update_layout(**LAYOUT, height=440, xaxis_rangeslider_visible=False)
    fig.update_xaxes(**AXIS)
    fig.update_yaxes(**AXIS)
    fig.update_yaxes(title_text='Prix ($)', row=1, col=1, title_font=dict(color='#c9d1d9', size=12))
    fig.update_yaxes(title_text='Volume',   row=2, col=1, title_font=dict(color='#c9d1d9', size=12))
    return fig


def chart_vol_surface(ticker, expiry_dates, spot, r_rate, option_type='call'):
    """
    Construit la surface de volatilite implicite 3D + retourne les donnees enrichies
    pour l'affichage des KPIs (term structure, smile, contango/backwardation).
    Retourne: (fig, surf_meta) ou surf_meta est un dict de stats utiles.
    """
    k_grid = np.linspace(0.82, 1.18, 28)
    expiries_label, iv_rows, dte_list = [], [], []

    for exp in expiry_dates[:12]:
        try:
            calls, puts = get_option_chain(ticker, exp)
            data = (calls if option_type == 'call' else puts).dropna(subset=['impliedVolatility']).copy()
            data = data[(data['impliedVolatility'] > 0.01) & (data['impliedVolatility'] < 5.0)]
            data = data[(data['strike'] > spot*0.75) & (data['strike'] < spot*1.25)]
            if len(data) < 5: continue
            ks  = data['strike'].values / spot
            ivs = data['impliedVolatility'].values * 100
            order = np.argsort(ks)
            ks, ivs = ks[order], ivs[order]
            # Interpolation avec extrapolation plate aux bords
            row = np.interp(k_grid, ks, ivs,
                            left=float(ivs[0]), right=float(ivs[-1]))
            dte = max(1, (datetime.strptime(exp, "%Y-%m-%d") - datetime.now()).days)
            expiries_label.append(exp)
            dte_list.append(dte)
            iv_rows.append(row)
        except:
            continue

    if len(iv_rows) < 2:
        return go.Figure(), {}

    z_arr = np.array(iv_rows)

    # ── Meta donnees pour les KPIs ────────────────────────────
    atm_idx  = int(np.argmin(np.abs(k_grid - 1.0)))
    atm_ivs  = z_arr[:, atm_idx]           # IV ATM par echeance
    otm_put_idx  = int(np.argmin(np.abs(k_grid - 0.90)))
    otm_call_idx = int(np.argmin(np.abs(k_grid - 1.10)))

    term_slope = float(atm_ivs[-1] - atm_ivs[0]) if len(atm_ivs) >= 2 else 0.0
    smile_skew = float(z_arr[0, otm_put_idx] - z_arr[0, otm_call_idx])
    iv_min, iv_max = float(z_arr.min()), float(z_arr.max())
    iv_atm_near  = float(atm_ivs[0])
    iv_atm_far   = float(atm_ivs[-1])
    structure    = "BACKWARDATION" if term_slope < -1.0 else ("CONTANGO" if term_slope > 1.0 else "PLATE")
    struct_color = "#ff4b6e" if structure == "BACKWARDATION" else ("#3fb950" if structure == "CONTANGO" else "#f5a623")
    struct_desc  = ("Court terme > Long terme — stress immediat") if structure == "BACKWARDATION" else                    ("Court terme < Long terme — marche calme") if structure == "CONTANGO" else                    "Court terme ≈ Long terme — sans signal fort"

    surf_meta = {
        "expiries": expiries_label,
        "dte_list": dte_list,
        "atm_ivs": atm_ivs.tolist(),
        "iv_min": iv_min, "iv_max": iv_max,
        "iv_atm_near": iv_atm_near, "iv_atm_far": iv_atm_far,
        "term_slope": term_slope,
        "smile_skew": smile_skew,
        "structure": structure,
        "struct_color": struct_color,
        "struct_desc": struct_desc,
        "z_arr": z_arr,
        "k_grid": k_grid,
    }

    # ── Colorscale professionnelle (froid → chaud) ────────────
    colorscale = [
        [0.00, '#0d1b3e'], [0.15, '#1a3a7e'],
        [0.35, '#0066cc'], [0.55, '#00aaff'],
        [0.72, '#00e5ff'], [0.85, '#ffe066'],
        [1.00, '#ff4b6e'],
    ]

    scene_axis = dict(
        backgroundcolor='rgba(7,11,18,0.0)',
        gridcolor='rgba(255,255,255,0.08)',
        zerolinecolor='rgba(255,255,255,0.12)',
        tickfont=dict(color='#8b949e', size=9, family='JetBrains Mono'),
        title_font=dict(color='#c9d1d9', size=11),
        showbackground=True,
    )

    # Axe Y : labels echeances raccourcis (Mmm YY)
    y_labels_short = []
    for e in expiries_label:
        try:
            d = datetime.strptime(e, "%Y-%m-%d")
            y_labels_short.append(d.strftime("%b %y"))
        except:
            y_labels_short.append(e)

    # X labels : moneyness en %
    x_tickvals = [i for i, k in enumerate(k_grid) if abs(k - round(k*10)/10) < 0.01]
    x_ticktext  = [f"{k_grid[i]*100:.0f}%" for i in x_tickvals]

    fig = go.Figure(data=[go.Surface(
        z=z_arr,
        x=list(range(len(k_grid))),
        y=list(range(len(expiries_label))),
        colorscale=colorscale,
        opacity=0.96,
        lighting=dict(ambient=0.75, diffuse=0.85, roughness=0.4,
                      specular=0.5, fresnel=0.3),
        lightposition=dict(x=1000, y=1000, z=2000),
        contours=dict(
            z=dict(show=True, usecolormap=True,
                   highlightcolor='rgba(255,255,255,0.3)', project_z=False,
                   width=1),
        ),
        colorbar=dict(
            title=dict(text='IV  (%)', font=dict(color='#c9d1d9', size=12,
                        family='JetBrains Mono'), side='right'),
            tickfont=dict(color='#8b949e', size=10, family='JetBrains Mono'),
            thickness=12, len=0.65, x=1.0,
            bgcolor='rgba(7,11,18,0.8)',
            bordercolor='rgba(255,255,255,0.08)', borderwidth=1,
        ),
        hovertemplate=(
            "<b>Moneyness :</b> %{x}<br>"
            "<b>Echeance  :</b> %{y}<br>"
            "<b>IV        :</b> %{z:.1f}%<extra></extra>"
        ),
    )])

    fig.update_layout(
        scene=dict(
            xaxis=dict(**scene_axis,
                       title=dict(text='Moneyness (Strike / Spot)',
                                  font=dict(color='#c9d1d9', size=11)),
                       tickvals=x_tickvals, ticktext=x_ticktext),
            yaxis=dict(**scene_axis,
                       title=dict(text='Echeance',
                                  font=dict(color='#c9d1d9', size=11)),
                       tickvals=list(range(len(expiries_label))),
                       ticktext=y_labels_short),
            zaxis=dict(**scene_axis,
                       title=dict(text='Vol Implicite (%)',
                                  font=dict(color='#c9d1d9', size=11))),
            bgcolor='rgba(7,11,18,0.97)',
            aspectmode='manual',
            aspectratio=dict(x=1.7, y=1.1, z=0.65),
            camera=dict(
                eye=dict(x=1.6, y=-1.5, z=0.9),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        template='plotly_dark',
        height=540,
        margin=dict(l=0, r=20, t=10, b=0),
        font=dict(color='#e0e6f0', family='Inter'),
    )
    return fig, surf_meta


def chart_hv_iv(hist, data, spot):
    log_ret = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
    hv30 = log_ret.rolling(30).std() * np.sqrt(252) * 100
    hv60 = log_ret.rolling(60).std() * np.sqrt(252) * 100
    atm_mask = (data['strike'] > spot*0.95) & (data['strike'] < spot*1.05)
    atm_iv = data[atm_mask]['impliedVolatility'].mean() * 100 if len(data[atm_mask]) > 0 else np.nan
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hv30, name='HV 30j',
                             line=dict(color='#f5a623', width=2),
                             fill='tozeroy', fillcolor='rgba(245,166,35,0.06)'))
    fig.add_trace(go.Scatter(x=hist.index, y=hv60, name='HV 60j',
                             line=dict(color='#b06fff', width=2),
                             fill='tozeroy', fillcolor='rgba(176,111,255,0.06)'))
    if not np.isnan(atm_iv):
        fig.add_hline(y=atm_iv, line_dash='dash', line_color='#00e5ff', line_width=1.5,
                      annotation=dict(text=f'IV ATM : {atm_iv:.1f}%',
                                      font=dict(color='#00e5ff', size=12),
                                      bgcolor='rgba(0,14,30,0.85)',
                                      bordercolor='#00e5ff', borderwidth=1,
                                      xanchor='left', x=0.01))
    fig.update_layout(**LAYOUT, height=300, xaxis={**AXIS},
                      yaxis={**AXIS, 'title_text': 'Volatilite (%)'})
    return fig


def chart_bsm_vs_market(plot_data, spot):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_data['strike'], y=plot_data['lastPrice'],
        name='Prix Marche', line=dict(color='#00e5ff', width=2.5),
        mode='lines+markers', marker=dict(size=4, color='#00e5ff')))
    fig.add_trace(go.Scatter(x=plot_data['strike'], y=plot_data['BS_Price'],
        name='BSM Theorique', line=dict(color='#ff4b6e', width=2, dash='dot')))
    fig.add_vline(x=spot, line_dash='dot', line_color='rgba(255,255,255,0.4)', line_width=1.5,
                  annotation=dict(text='Spot', font=dict(color='#ffffff', size=11),
                                  bgcolor='rgba(0,0,0,0.6)', bordercolor='rgba(255,255,255,0.3)', borderwidth=1))
    fig.update_layout(**LAYOUT, height=290,
                      xaxis={**AXIS, 'title_text': 'Strike ($)'},
                      yaxis={**AXIS, 'title_text': 'Prime ($)'})
    return fig


def chart_pnl_multiscenario(spot, sel_strike, premium_total, nb_contrats, T, r_rate, iv, option_type, direction='long'):
    x_range  = np.linspace(sel_strike*0.70, sel_strike*1.30, 100)
    horizons = [("A expiration", 0, '#00e5ff'), ("T-15j", 15/365, '#f5a623'), ("T-30j", 30/365, '#b06fff')]
    fig = go.Figure()
    is_long = direction == 'long'
    for label, dt, color in horizons:
        t_rem = max(1e-6, T - dt)
        if dt == 0:
            option_values = [max(0, x-sel_strike if option_type=='call' else sel_strike-x) * 100 * nb_contrats
                             for x in x_range]
        else:
            option_values = [calculate_greeks(x, sel_strike, t_rem, r_rate, iv, option_type)['Price'] * 100 * nb_contrats
                             for x in x_range]
        y = [(val - premium_total) if is_long else (premium_total - val) for val in option_values]
        fig.add_trace(go.Scatter(x=x_range, y=y, name=label, line=dict(color=color, width=2.5)))
    fig.add_hline(y=0, line_dash='dash', line_color='rgba(255,75,110,0.6)', line_width=1.5)
    fig.add_vline(x=spot, line_dash='dot', line_color='rgba(255,255,255,0.4)', line_width=1.5,
                  annotation=dict(text='Spot', font=dict(color='#ffffff', size=11),
                                  bgcolor='rgba(0,0,0,0.6)', bordercolor='rgba(255,255,255,0.2)', borderwidth=1))
    fig.update_layout(**LAYOUT, height=310,
                      xaxis={**AXIS, 'title_text': 'Prix sous-jacent ($)'},
                      yaxis={**AXIS, 'title_text': 'P/L ($)'})
    return fig


def calc_gamma_bs(S, K, T, sigma, r=0.04):
    """Calcule le gamma Black-Scholes — utilisé quand Yahoo ne fournit pas gamma."""
    return analytics.calc_gamma_bs(S, K, T, sigma, r)


def chart_open_interest(calls_raw, puts_raw, spot, T=0.1, r_rate=0.04):
    calls = calls_raw[['strike','openInterest']].copy()
    puts  = puts_raw[['strike','openInterest']].copy()

    # Essaye d'abord la colonne gamma de Yahoo Finance
    # Si absente ou quasi-nulle (SPY, etc.), recalcule via Black-Scholes
    def get_gamma_series(df_raw, df_filt, option_type):
        has_gamma = ('gamma' in df_raw.columns and
                     df_raw['gamma'].notna().sum() > 0 and
                     df_raw['gamma'].abs().sum() > 1e-6)
        if has_gamma:
            return df_raw.loc[df_filt.index, 'gamma'].fillna(0).values if hasattr(df_filt, 'index') else df_raw['gamma'].fillna(0).reindex(df_filt.index, fill_value=0).values
        # Recalcul BSM
        iv_col = 'impliedVolatility'
        gammas = []
        for _, row in df_filt.iterrows():
            iv = row[iv_col] if iv_col in df_filt.columns else 0.2
            if pd.isna(iv) or iv <= 0:
                iv = 0.2
            gammas.append(calc_gamma_bs(spot, row['strike'], T, iv, r_rate))
        return gammas

    calls = calls[(calls['strike'] > spot*0.7) & (calls['strike'] < spot*1.3)].copy()
    puts  = puts[(puts['strike']  > spot*0.7) & (puts['strike']  < spot*1.3)].copy()

    # Merge IV pour le calcul BSM
    if 'impliedVolatility' in calls_raw.columns:
        calls = calls.merge(calls_raw[['strike','impliedVolatility']].drop_duplicates('strike'),
                            on='strike', how='left')
    else:
        calls['impliedVolatility'] = 0.2

    if 'impliedVolatility' in puts_raw.columns:
        puts = puts.merge(puts_raw[['strike','impliedVolatility']].drop_duplicates('strike'),
                          on='strike', how='left')
    else:
        puts['impliedVolatility'] = 0.2

    # Gamma : Yahoo ou BSM
    has_gamma_calls = ('gamma' in calls_raw.columns and
                       calls_raw['gamma'].notna().sum() > 0 and
                       calls_raw['gamma'].abs().sum() > 1e-6)
    has_gamma_puts  = ('gamma' in puts_raw.columns and
                       puts_raw['gamma'].notna().sum() > 0 and
                       puts_raw['gamma'].abs().sum() > 1e-6)

    if has_gamma_calls:
        calls = calls.merge(calls_raw[['strike','gamma']].drop_duplicates('strike'),
                            on='strike', how='left')
        calls['gamma'] = calls['gamma'].fillna(0)
    else:
        calls['gamma'] = calls.apply(
            lambda r: calc_gamma_bs(spot, r['strike'], T,
                                    r['impliedVolatility'] if pd.notna(r['impliedVolatility']) else 0.2,
                                    r_rate), axis=1)

    if has_gamma_puts:
        puts = puts.merge(puts_raw[['strike','gamma']].drop_duplicates('strike'),
                          on='strike', how='left')
        puts['gamma'] = puts['gamma'].fillna(0)
    else:
        puts['gamma'] = puts.apply(
            lambda r: calc_gamma_bs(spot, r['strike'], T,
                                    r['impliedVolatility'] if pd.notna(r['impliedVolatility']) else 0.2,
                                    r_rate), axis=1)

    calls['GEX'] =  calls['gamma'] * calls['openInterest'].fillna(0) * 100 * spot
    puts['GEX']  = -puts['gamma']  * puts['openInterest'].fillna(0)  * 100 * spot
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08,
                        subplot_titles=['Open Interest par Strike', 'Gamma Exposure (GEX)'])
    fig.add_trace(go.Bar(x=calls['strike'], y=calls['openInterest'], name='OI Calls',
                         marker_color='rgba(0,229,255,0.75)'), row=1, col=1)
    fig.add_trace(go.Bar(x=puts['strike'],  y=puts['openInterest'],  name='OI Puts',
                         marker_color='rgba(255,75,110,0.75)'), row=1, col=1)
    fig.add_trace(go.Bar(x=calls['strike'], y=calls['GEX'], name='GEX Calls',
                         marker_color='rgba(0,229,255,0.6)'), row=2, col=1)
    fig.add_trace(go.Bar(x=puts['strike'],  y=puts['GEX'],  name='GEX Puts',
                         marker_color='rgba(255,75,110,0.6)'), row=2, col=1)
    for row in [1,2]:
        fig.add_vline(x=spot, line_dash='dash', line_color='rgba(255,255,255,0.35)', line_width=1.5,
                      annotation=dict(text='Spot', font=dict(color='#ffffff', size=10),
                                      bgcolor='rgba(0,0,0,0.5)'), row=row, col=1)
    fig.update_layout(**LAYOUT, barmode='relative', height=540)
    fig.update_xaxes(**AXIS)
    fig.update_yaxes(**AXIS)
    for ann in fig.layout.annotations:
        ann.font.color = '#e0e6f0'
        ann.font.size  = 13
    return fig, calls, puts


def chart_payoff_strategy(x, y, spot, label):
    pos = [max(0, v) for v in y]
    neg = [min(0, v) for v in y]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=pos, fill='tozeroy', fillcolor='rgba(0,229,255,0.10)',
                             line=dict(color='#00e5ff', width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=neg, fill='tozeroy', fillcolor='rgba(255,75,110,0.10)',
                             line=dict(color='#ff4b6e', width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=y, name=label, line=dict(color='#00e5ff', width=2.5)))
    fig.add_hline(y=0, line_dash='dash', line_color='rgba(255,75,110,0.6)', line_width=1.5)
    fig.add_vline(x=spot, line_dash='dot', line_color='rgba(255,255,255,0.4)', line_width=1.5,
                  annotation=dict(text='Spot', font=dict(color='#ffffff', size=11),
                                  bgcolor='rgba(0,0,0,0.6)', bordercolor='rgba(255,255,255,0.2)', borderwidth=1))
    fig.update_layout(**LAYOUT, height=360,
                      xaxis={**AXIS, 'title_text': 'Prix sous-jacent ($)'},
                      yaxis={**AXIS, 'title_text': 'P/L ($)'},
                      title=dict(text=f'<b>{label}</b>', font=dict(color='#e0e6f0', size=13), x=0, xanchor='left'))
    return fig

# ============================================================
# CSS
# ============================================================

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #070b12 !important;
    color: #c9d1d9;
}
.stApp { background: radial-gradient(ellipse at 20% 0%, #0a1628 0%, #070b12 60%) !important; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1320 0%, #0a0f1a 100%) !important;
    border-right: 1px solid rgba(0,229,255,0.08) !important;
}
section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

h1 { font-size:1.9rem !important; font-weight:700 !important; color:#ffffff !important; }
h2 { color:#e0e6f0 !important; font-weight:600 !important; }
h3 { color:#c9d1d9 !important; }

.stTabs [data-baseweb="tab-list"] {
    gap:4px; background:rgba(255,255,255,0.03); border-radius:10px;
    padding:4px; border:1px solid rgba(255,255,255,0.06);
}
.stTabs [data-baseweb="tab"] {
    border-radius:8px !important; padding:8px 18px !important;
    color:#8b949e !important; font-size:0.82rem !important;
    font-weight:500 !important; letter-spacing:0.04em !important;
    background:transparent !important;
}
.stTabs [aria-selected="true"] {
    background:linear-gradient(135deg,rgba(0,229,255,0.18),rgba(0,140,255,0.12)) !important;
    color:#00e5ff !important; border:1px solid rgba(0,229,255,0.25) !important;
}
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background:linear-gradient(135deg,#0d1320,#0a0f1a) !important;
    border:1px solid rgba(0,229,255,0.15) !important;
    border-radius:10px !important; color:#e0e6f0 !important;
    box-shadow:0 0 0 0 transparent !important;
    transition:border-color 0.2s, box-shadow 0.2s !important;
}
.stSelectbox > div > div:focus-within,
.stNumberInput > div > div:focus-within,
.stTextInput > div > div:focus-within {
    border-color:rgba(0,229,255,0.5) !important;
    box-shadow:0 0 0 3px rgba(0,229,255,0.08) !important;
}
.stNumberInput > div,
.stNumberInput > div > div[data-baseweb="input"],
.stNumberInput [data-baseweb="base-input"] {
    border:none !important; box-shadow:none !important;
    background:transparent !important;
}
.stNumberInput > div > div {
    background:linear-gradient(135deg,#0d1320,#0a0f1a) !important;
    border:1px solid rgba(0,229,255,0.15) !important;
    border-radius:10px !important;
    overflow:hidden;
}
.stNumberInput > div > div:focus-within {
    border-color:rgba(0,229,255,0.45) !important;
    box-shadow:0 0 0 3px rgba(0,229,255,0.07) !important;
}
/* Kill ALL red/default borders Streamlit injects */
.stNumberInput * { outline:none !important; }
.stNumberInput > div > div > div { border:none !important; box-shadow:none !important; }
.stNumberInput button {
    background:rgba(0,229,255,0.08) !important;
    border:none !important; border-left:1px solid rgba(0,229,255,0.12) !important;
    color:#00e5ff !important; border-radius:0 !important; font-size:1rem !important;
    transition:background 0.15s !important;
}
.stNumberInput button:hover { background:rgba(0,229,255,0.2) !important; }
.stNumberInput button:first-of-type {
    border-left:none !important; border-right:1px solid rgba(0,229,255,0.12) !important;
}
.stSelectbox label, .stNumberInput label,
.stTextInput label, .stRadio label {
    color:#8b949e !important; font-size:0.75rem !important;
    font-weight:500 !important; letter-spacing:0.05em !important;
    text-transform:uppercase !important;
}
.stTextInput > div > div > input::placeholder { color:#404858 !important; }

[data-testid="stMetric"] {
    background:linear-gradient(135deg,#0d1320,#111827);
    border:1px solid rgba(255,255,255,0.07);
    border-radius:12px; padding:14px 18px !important;
}
[data-testid="stMetricLabel"] { color:#8b949e !important; font-size:0.75rem !important; text-transform:uppercase; letter-spacing:0.07em; }
[data-testid="stMetricValue"] { color:#e0e6f0 !important; font-size:1.3rem !important; font-family:'JetBrains Mono' !important; }
[data-testid="stMetricDelta"] { font-size:0.78rem !important; }
hr { border-color:rgba(255,255,255,0.06) !important; }

.page-header {
    display:flex; align-items:center; gap:16px;
    padding:20px 0 12px; border-bottom:1px solid rgba(0,229,255,0.15); margin-bottom:24px;
}
.ticker-badge {
    background:linear-gradient(135deg,rgba(0,229,255,0.15),rgba(0,140,255,0.08));
    border:1px solid rgba(0,229,255,0.3); border-radius:8px;
    padding:6px 14px; font-family:'JetBrains Mono'; font-size:1rem; color:#00e5ff; font-weight:600;
}
.company-name { font-size:1.55rem; font-weight:700; color:#ffffff; }
.sector-tag   { font-size:0.75rem; color:#8b949e; margin-top:2px; }

.profile-box {
    background:linear-gradient(135deg,#0a0f1a 0%,#0d1320 100%);
    padding:18px 22px; border-radius:12px;
    border:1px solid rgba(255,255,255,0.07); border-left:3px solid #00e5ff;
    color:#a0aab8; font-size:0.84rem; line-height:1.75; margin-bottom:24px;
}

.kpi-row { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:20px; }
.kpi-card {
    background:linear-gradient(135deg,#0d1320 0%,#111827 100%);
    border:1px solid rgba(255,255,255,0.07);
    border-radius:12px; padding:16px 18px; flex:1; min-width:110px;
}
.kpi-label { color:#606878; font-size:0.68rem; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px; }
.kpi-value { color:#e0e6f0; font-size:1.3rem; font-weight:700; font-family:'JetBrains Mono'; }
.kpi-sub   { color:#4a8aff; font-size:0.72rem; margin-top:4px; }

.section-label {
    display:flex; align-items:center; gap:8px;
    font-size:0.7rem; font-weight:600; letter-spacing:0.14em;
    text-transform:uppercase; color:#606878; margin:24px 0 10px;
}
.section-label::after {
    content:''; flex:1; height:1px;
    background:linear-gradient(90deg,rgba(255,255,255,0.08),transparent);
}

.greek-row { display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin:10px 0; }
.greek-card {
    background:#0d1320; border:1px solid rgba(255,255,255,0.07);
    border-radius:10px; padding:14px 10px; text-align:center;
}
.g-name  { color:#606878; font-size:0.65rem; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:6px; }
.g-value { color:#00e5ff; font-size:1.2rem; font-weight:600; font-family:'JetBrains Mono'; }

.greek-row-2 { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin:10px 0; }
.greek-card-2 {
    background:#090d16; border:1px solid rgba(255,255,255,0.05);
    border-radius:10px; padding:12px 10px; text-align:center;
}
.g-name-2  { color:#505868; font-size:0.62rem; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:5px; }
.g-value-2 { color:#b06fff; font-size:1.05rem; font-weight:600; font-family:'JetBrains Mono'; }

.signal-card {
    background:#0a0f1a; border:1px solid rgba(255,255,255,0.07);
    border-radius:10px; padding:14px 18px; margin:10px 0;
    font-size:0.83rem; color:#c9d1d9; line-height:1.6;
}

/* EARNINGS ALERT */
.earnings-alert {
    background:linear-gradient(135deg, rgba(245,166,35,0.12), rgba(245,100,35,0.06));
    border:1px solid rgba(245,166,35,0.4); border-radius:12px;
    padding:16px 20px; margin:12px 0;
}
.earnings-alert .ea-title { color:#f5a623; font-size:0.72rem; font-weight:700;
    text-transform:uppercase; letter-spacing:0.12em; margin-bottom:8px; }
.earnings-alert .ea-date  { color:#ffffff; font-size:1.4rem; font-weight:700;
    font-family:'JetBrains Mono'; }
.earnings-alert .ea-days  { color:#f5a623; font-size:0.85rem; margin-top:4px; }
.earnings-alert .ea-warn  { color:#ff9f43; font-size:0.78rem; margin-top:8px;
    padding-top:8px; border-top:1px solid rgba(245,166,35,0.2); }

/* IV RANK */
.ivrank-box {
    background:linear-gradient(135deg,#0d1320,#111827);
    border:1px solid rgba(255,255,255,0.07); border-radius:12px; padding:18px;
}
.ivrank-bar-bg {
    background:rgba(255,255,255,0.07); border-radius:999px;
    height:10px; margin:10px 0;
}
.ivrank-bar-fill {
    height:10px; border-radius:999px;
    background:linear-gradient(90deg,#3fb950,#f5a623,#ff4b6e);
    transition: width 0.4s ease;
}
.ivrank-labels { display:flex; justify-content:space-between;
    font-size:0.68rem; color:#606878; margin-top:4px; }

/* IMPLIED MOVE */
.move-box {
    background:linear-gradient(135deg, rgba(0,229,255,0.07), rgba(0,100,200,0.05));
    border:1px solid rgba(0,229,255,0.2); border-radius:12px; padding:18px 20px;
}
.move-title { color:#00e5ff; font-size:0.7rem; font-weight:700;
    text-transform:uppercase; letter-spacing:0.12em; margin-bottom:10px; }
.move-value { color:#ffffff; font-size:2rem; font-weight:700; font-family:'JetBrains Mono'; }
.move-sub   { color:#8b949e; font-size:0.78rem; margin-top:4px; }
.move-range {
    display:flex; justify-content:space-between; margin-top:12px;
    padding-top:10px; border-top:1px solid rgba(0,229,255,0.12);
}
.move-range .mr-item { text-align:center; }
.move-range .mr-label { color:#606878; font-size:0.65rem; text-transform:uppercase; }
.move-range .mr-val   { color:#e0e6f0; font-size:0.95rem; font-weight:600; font-family:'JetBrains Mono'; }
.move-range .up       { color:#3fb950; }
.move-range .dn       { color:#ff4b6e; }

.move-commentary {
    background:rgba(0,0,0,0.3); border-radius:8px; padding:12px 14px;
    margin-top:12px; font-size:0.82rem; color:#c9d1d9; line-height:1.65;
    border-left:3px solid #00e5ff;
}

/* ROLL ANALYZER */
.roll-box {
    background:linear-gradient(135deg,#0d1320,#0a0f1a);
    border:1px solid rgba(176,111,255,0.2); border-radius:12px; padding:20px;
}
.roll-title { color:#b06fff; font-size:0.7rem; font-weight:700;
    text-transform:uppercase; letter-spacing:0.12em; margin-bottom:16px; }
.roll-table { width:100%; border-collapse:collapse; font-size:0.85rem; }
.roll-table th { color:#606878; font-size:0.68rem; text-transform:uppercase;
    letter-spacing:0.1em; padding:4px 8px; text-align:center; }
.roll-table td { padding:10px 8px; text-align:center; color:#e0e6f0;
    font-family:'JetBrains Mono'; border-top:1px solid rgba(255,255,255,0.04); }
.roll-table .highlight { color:#b06fff; font-weight:700; }
.roll-cost-pos { color:#3fb950; font-size:1.2rem; font-weight:700; font-family:'JetBrains Mono'; }
.roll-cost-neg { color:#ff4b6e; font-size:1.2rem; font-weight:700; font-family:'JetBrains Mono'; }
.roll-verdict {
    background:rgba(176,111,255,0.07); border:1px solid rgba(176,111,255,0.2);
    border-radius:8px; padding:12px 14px; margin-top:12px;
    font-size:0.82rem; color:#c9d1d9; line-height:1.65;
}

.pnl-table { width:100%; border-collapse:collapse; font-size:0.85rem; color:#c9d1d9; }
.pnl-table td { padding:8px 4px; }
.pnl-table tr { border-bottom:1px solid rgba(255,255,255,0.04); }
.pnl-table .lbl { color:#606878; }
.pnl-table .val { text-align:right; font-family:'JetBrains Mono'; font-weight:500; }
.pnl-table .total td { border-top:1px solid rgba(255,255,255,0.12); padding-top:12px; font-weight:600; }
.pos { color:#3fb950; } .neg { color:#ff4b6e; }

.vix-widget {
    background:linear-gradient(135deg,rgba(0,229,255,0.07),rgba(0,80,180,0.05));
    border:1px solid rgba(0,229,255,0.15); border-radius:12px;
    padding:16px 14px; text-align:center; margin-bottom:6px;
}
.vix-label { color:#606878; font-size:0.65rem; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:4px; }
.vix-value { color:#00e5ff; font-size:2.2rem; font-weight:700; font-family:'JetBrains Mono'; line-height:1; }
.vix-sub   { color:#8b949e; font-size:0.68rem; margin-top:4px; }

/* SIDEBAR TICKER INPUT */
.sb-section-title {
    color:#606878; font-size:0.62rem; font-weight:700; letter-spacing:0.14em;
    text-transform:uppercase; margin:14px 0 6px; display:flex; align-items:center; gap:6px;
}
.sb-section-title::after { content:''; flex:1; height:1px;
    background:linear-gradient(90deg,rgba(255,255,255,0.08),transparent); }

/* MARKET TICKER CARDS */
.mkt-card {
    display:flex; align-items:center; justify-content:space-between;
    background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.05);
    border-radius:8px; padding:8px 10px; margin-bottom:5px;
    transition:border-color 0.2s;
}
.mkt-card:hover { border-color:rgba(0,229,255,0.15); }
.mkt-left  { display:flex; align-items:center; gap:8px; }
.mkt-icon  { font-size:0.9rem; }
.mkt-label { color:#8b949e; font-size:0.7rem; font-weight:500; letter-spacing:0.05em; }
.mkt-price { color:#e0e6f0; font-size:0.82rem; font-weight:600;
    font-family:'JetBrains Mono',monospace; }
.mkt-chg-pos { color:#3fb950; font-size:0.72rem; font-family:'JetBrains Mono',monospace; font-weight:600; }
.mkt-chg-neg { color:#ff4b6e; font-size:0.72rem; font-family:'JetBrains Mono',monospace; font-weight:600; }
.mkt-chg-neu { color:#8b949e; font-size:0.72rem; font-family:'JetBrains Mono',monospace; }

/* FEAR/GREED gauge */
.fg-bar-bg { background:rgba(255,255,255,0.06); border-radius:999px; height:7px; margin:6px 0 2px; }
.fg-bar-fill { height:7px; border-radius:999px;
    background:linear-gradient(90deg,#ff4b6e 0%,#f5a623 40%,#3fb950 80%,#00e5ff 100%); }
.fg-labels { display:flex; justify-content:space-between;
    font-size:0.6rem; color:#505868; margin-top:2px; }

.strat-kpi-row { display:flex; gap:10px; margin-top:14px; }
.strat-kpi {
    flex:1; background:#0d1320; border:1px solid rgba(255,255,255,0.07);
    border-radius:10px; padding:14px; text-align:center;
}
.strat-kpi .sk-label { color:#606878; font-size:0.68rem; text-transform:uppercase; letter-spacing:0.1em; }
.strat-kpi .sk-value { font-size:1.25rem; font-weight:700; font-family:'JetBrains Mono'; margin-top:4px; }

/* STRATEGY SELECTOR */
.strat-selector {
    display:grid; grid-template-columns:repeat(5,1fr); gap:8px; margin-bottom:20px;
}
.strat-card-desc {
    background:linear-gradient(135deg,rgba(10,15,26,0.98),rgba(13,19,32,0.98));
    border:1px solid rgba(255,255,255,0.07); border-radius:14px;
    padding:20px 22px; margin-bottom:16px;
}
.scd-header { display:flex; align-items:center; gap:14px; margin-bottom:14px; }
.scd-icon { font-size:1.8rem; }
.scd-title { font-size:1.05rem; font-weight:700; color:#ffffff; }
.scd-subtitle { font-size:0.72rem; color:#8b949e; margin-top:2px; }
.scd-tags { display:flex; gap:6px; flex-wrap:wrap; margin-bottom:14px; }
.scd-tag {
    font-size:0.62rem; padding:3px 10px; border-radius:999px; font-weight:600;
    letter-spacing:0.06em; text-transform:uppercase;
}
.tag-bull { background:rgba(63,185,80,0.12); color:#3fb950; border:1px solid rgba(63,185,80,0.25); }
.tag-bear { background:rgba(255,75,110,0.12); color:#ff4b6e; border:1px solid rgba(255,75,110,0.25); }
.tag-neut { background:rgba(245,166,35,0.12); color:#f5a623; border:1px solid rgba(245,166,35,0.25); }
.tag-info { background:rgba(0,229,255,0.10); color:#00e5ff; border:1px solid rgba(0,229,255,0.2); }
.tag-risk { background:rgba(176,111,255,0.10); color:#b06fff; border:1px solid rgba(176,111,255,0.2); }
.scd-body { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
.scd-section { }
.scd-section-title { color:#606878; font-size:0.62rem; text-transform:uppercase;
    letter-spacing:0.1em; margin-bottom:6px; font-weight:600; }
.scd-text { color:#c9d1d9; font-size:0.8rem; line-height:1.65; }
.scd-steps { list-style:none; padding:0; margin:0; }
.scd-steps li { color:#c9d1d9; font-size:0.78rem; line-height:1.7;
    padding-left:14px; position:relative; }
.scd-steps li::before { content:"→"; position:absolute; left:0; color:#00e5ff; }
.scd-ideal { background:rgba(0,229,255,0.05); border:1px solid rgba(0,229,255,0.12);
    border-radius:8px; padding:10px 12px; margin-top:14px;
    font-size:0.78rem; color:#c9d1d9; line-height:1.6; }
.scd-ideal b { color:#00e5ff; }

/* VOL SURFACE */
.surf-wrap {
    background:linear-gradient(135deg,rgba(7,11,18,0.98),rgba(10,15,26,0.98));
    border:1px solid rgba(0,229,255,0.10); border-radius:16px;
    padding:0; overflow:hidden; margin-bottom:4px;
}
.surf-header {
    display:flex; align-items:center; justify-content:space-between;
    padding:16px 22px 14px; border-bottom:1px solid rgba(255,255,255,0.06);
}
.surf-title { font-size:0.95rem; font-weight:700; color:#ffffff; }
.surf-subtitle { font-size:0.7rem; color:#8b949e; margin-top:2px; }
.surf-badge {
    font-size:0.62rem; font-weight:700; padding:4px 12px;
    border-radius:999px; letter-spacing:0.08em; text-transform:uppercase;
}
.surf-kpi-row {
    display:grid; grid-template-columns:repeat(5,1fr);
    gap:0; border-bottom:1px solid rgba(255,255,255,0.05);
}
.surf-kpi {
    padding:14px 16px; border-right:1px solid rgba(255,255,255,0.05);
    text-align:center;
}
.surf-kpi:last-child { border-right:none; }
.surf-kpi-label {
    color:#606878; font-size:0.62rem; text-transform:uppercase;
    letter-spacing:0.1em; font-weight:600; margin-bottom:5px;
}
.surf-kpi-value {
    color:#e0e6f0; font-size:1.1rem; font-weight:700;
    font-family:'JetBrains Mono',monospace; line-height:1;
}
.surf-kpi-sub { color:#606878; font-size:0.65rem; margin-top:4px; }
.surf-chart-area { padding:0 8px 8px; }
.surf-legend {
    display:flex; align-items:center; justify-content:space-between;
    padding:10px 22px 14px; border-top:1px solid rgba(255,255,255,0.05);
}
.surf-legend-item { display:flex; align-items:center; gap:7px;
    font-size:0.7rem; color:#8b949e; }
.surf-legend-dot { width:10px; height:10px; border-radius:50%; flex-shrink:0; }
.surf-hint {
    font-size:0.65rem; color:#404858; text-align:center;
    padding:0 0 10px; letter-spacing:0.04em;
}
.term-bar-wrap { padding:12px 22px 16px; }
.term-bar-title { color:#606878; font-size:0.62rem; text-transform:uppercase;
    letter-spacing:0.1em; font-weight:600; margin-bottom:10px; }
.term-bar-row { display:flex; align-items:center; gap:10px; margin-bottom:6px; }
.term-bar-label { color:#8b949e; font-size:0.68rem; font-family:'JetBrains Mono';
    width:52px; text-align:right; flex-shrink:0; }
.term-bar-track { flex:1; height:8px; background:rgba(255,255,255,0.05);
    border-radius:999px; overflow:hidden; }
.term-bar-fill { height:8px; border-radius:999px;
    background:linear-gradient(90deg,#0066cc,#00e5ff); }
.term-bar-val { color:#c9d1d9; font-size:0.68rem; font-family:'JetBrains Mono';
    width:46px; }

/* SKEW */
.skew-box { background:linear-gradient(135deg,#0d1320,#0a0f1a);
    border:1px solid rgba(255,255,255,0.07); border-radius:14px; padding:18px 20px; }
.skew-kpi-row { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-top:14px; }
.skew-kpi { background:#090d16; border:1px solid rgba(255,255,255,0.05);
    border-radius:10px; padding:12px 10px; text-align:center; }
.sk2-label { color:#606878; font-size:0.64rem; text-transform:uppercase;
    letter-spacing:0.1em; margin-bottom:5px; }
.sk2-value { font-size:1.1rem; font-weight:700; font-family:'JetBrains Mono',monospace; }

/* POP */
.pop-ring-row { display:flex; gap:14px; flex-wrap:wrap; margin:14px 0; }
.pop-card { flex:1; min-width:140px; background:#0d1320;
    border:1px solid rgba(255,255,255,0.07); border-radius:12px;
    padding:16px 14px; text-align:center; }
.pop-label { color:#606878; font-size:0.64rem; text-transform:uppercase;
    letter-spacing:0.1em; margin-bottom:8px; }
.pop-value { font-size:2rem; font-weight:700; font-family:'JetBrains Mono',monospace; }
.pop-sub   { font-size:0.72rem; color:#8b949e; margin-top:4px; }
.pop-bar-bg { background:rgba(255,255,255,0.06); border-radius:999px;
    height:6px; margin:8px 0 4px; }
.pop-bar-fill { height:6px; border-radius:999px; }
.ev-box { background:linear-gradient(135deg,rgba(0,229,255,0.06),rgba(0,80,200,0.04));
    border:1px solid rgba(0,229,255,0.15); border-radius:10px;
    padding:14px 16px; margin-top:12px; }
.ev-label { color:#606878; font-size:0.64rem; text-transform:uppercase;
    letter-spacing:0.1em; margin-bottom:6px; }
.ev-value { font-size:1.4rem; font-weight:700; font-family:'JetBrains Mono',monospace; }
.ev-breakdown { font-size:0.74rem; color:#8b949e; margin-top:4px; line-height:1.6; }

/* FLOW */
.flow-table { width:100%; border-collapse:collapse; font-size:0.82rem; }
.flow-table th { color:#606878; font-size:0.65rem; text-transform:uppercase;
    letter-spacing:0.1em; padding:8px 10px; text-align:left;
    border-bottom:1px solid rgba(255,255,255,0.06); }
.flow-table td { padding:9px 10px; color:#e0e6f0;
    border-bottom:1px solid rgba(255,255,255,0.03); }
.flow-table tr:hover td { background:rgba(0,229,255,0.03); }
.flow-badge { display:inline-block; padding:3px 9px; border-radius:999px;
    font-size:0.65rem; font-weight:700; letter-spacing:0.08em; }
.flow-call { background:rgba(0,229,255,0.12); color:#00e5ff;
    border:1px solid rgba(0,229,255,0.25); }
.flow-put  { background:rgba(255,75,110,0.12); color:#ff4b6e;
    border:1px solid rgba(255,75,110,0.25); }
.flow-fire { color:#f5a623; font-size:0.9rem; }
.flow-summary-row { display:grid; grid-template-columns:repeat(3,1fr);
    gap:10px; margin-bottom:16px; }
.flow-summary-card { background:#0d1320; border:1px solid rgba(255,255,255,0.07);
    border-radius:10px; padding:14px; text-align:center; }
.fsc-label { color:#606878; font-size:0.64rem; text-transform:uppercase;
    letter-spacing:0.1em; margin-bottom:6px; }
.fsc-value { font-size:1.3rem; font-weight:700; font-family:'JetBrains Mono',monospace; }

/* ANALYSIS CARDS */
.analysis-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:16px 0; }
.analysis-card {
    background:linear-gradient(135deg,#0d1320,#0a0f1a);
    border:1px solid rgba(255,255,255,0.07);
    border-radius:12px; padding:16px 18px;
}
.analysis-card:hover { border-color:rgba(0,229,255,0.15); }
.ac-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; }
.ac-title { color:#606878; font-size:0.65rem; text-transform:uppercase; letter-spacing:0.12em; font-weight:600; }
.ac-badge {
    font-size:0.62rem; font-weight:700; letter-spacing:0.08em;
    padding:3px 8px; border-radius:999px; text-transform:uppercase;
}
.badge-bull { background:rgba(63,185,80,0.15); color:#3fb950; border:1px solid rgba(63,185,80,0.3); }
.badge-bear { background:rgba(255,75,110,0.15); color:#ff4b6e; border:1px solid rgba(255,75,110,0.3); }
.badge-neut { background:rgba(245,166,35,0.15); color:#f5a623; border:1px solid rgba(245,166,35,0.3); }
.badge-info { background:rgba(0,229,255,0.10); color:#00e5ff; border:1px solid rgba(0,229,255,0.2); }
.ac-value { font-size:1.5rem; font-weight:700; font-family:'JetBrains Mono',monospace; color:#e0e6f0; margin-bottom:6px; }
.ac-sub   { font-size:0.75rem; color:#8b949e; margin-bottom:10px; }
.ac-comment { font-size:0.78rem; color:#c9d1d9; line-height:1.6;
    background:rgba(0,0,0,0.2); border-radius:6px; padding:10px 12px;
    border-left:2px solid rgba(255,255,255,0.1); }

.synthesis-box {
    background:linear-gradient(135deg, rgba(10,15,26,0.95), rgba(13,19,32,0.95));
    border:1px solid rgba(255,255,255,0.08); border-radius:14px;
    padding:22px 24px; margin-top:20px;
}
.synthesis-title {
    font-size:0.68rem; font-weight:700; text-transform:uppercase;
    letter-spacing:0.14em; color:#606878; margin-bottom:16px;
}
.synthesis-score-row { display:flex; align-items:center; gap:20px; margin-bottom:16px; }
.synthesis-score {
    font-size:2.8rem; font-weight:700; font-family:'JetBrains Mono',monospace;
    line-height:1;
}
.synthesis-label { font-size:1rem; font-weight:600; }
.synthesis-desc  { font-size:0.82rem; color:#8b949e; margin-top:4px; line-height:1.5; }
.synthesis-signals { display:flex; flex-wrap:wrap; gap:8px; margin-top:14px; }
.sig-pill {
    font-size:0.72rem; padding:5px 12px; border-radius:999px; font-weight:500;
}
.sig-bull { background:rgba(63,185,80,0.12);  color:#3fb950; border:1px solid rgba(63,185,80,0.25); }
.sig-bear { background:rgba(255,75,110,0.12); color:#ff4b6e; border:1px solid rgba(255,75,110,0.25); }
.sig-neut { background:rgba(245,166,35,0.12); color:#f5a623; border:1px solid rgba(245,166,35,0.25); }
</style>
"""

# ============================================================
# APP MAIN
# ============================================================

current_10y, current_vix, market_data = get_market_context()

st.set_page_config(layout="wide", page_title="Quantum Options Terminal", page_icon="💎")
st.markdown(CSS, unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    # ── Logo / titre ──────────────────────────────────────────
    st.markdown("""
    <div style="padding:10px 0 16px;border-bottom:1px solid rgba(0,229,255,0.12);margin-bottom:14px">
        <div style="font-size:1.05rem;font-weight:700;color:#fff;letter-spacing:-0.01em">
            💎 Quantum Options
        </div>
        <div style="font-size:0.65rem;color:#606878;margin-top:2px;letter-spacing:0.06em">
            TERMINAL · DONNÉES EN QUASI TEMPS RÉEL
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Ticker + taux ─────────────────────────────────────────
    st.markdown('<div class="sb-section-title">Analyse</div>', unsafe_allow_html=True)
    ticker = st.text_input("Symbole", value="AAPL",
                           placeholder="AAPL, TSLA, SPY, QQQ…",
                           label_visibility="collapsed").upper().strip()
    r_rate = st.number_input("Taux sans risque (US 10Y)",
                              value=float(current_10y), format="%.4f", step=0.001)

    # ── VIX + Fear/Greed proxy ────────────────────────────────
    vix_col  = "#3fb950" if current_vix < 15 else ("#f5a623" if current_vix < 25 else "#ff4b6e")
    vix_mood = "Calme" if current_vix < 15 else ("Modere" if current_vix < 25 else ("Eleve" if current_vix < 35 else "Extreme"))
    vix_pct  = min(100, (current_vix / 50) * 100)

    st.markdown(f'''
    <div class="vix-widget" style="margin-top:12px">
        <div class="vix-label">Indice VIX — Peur du Marche</div>
        <div class="vix-value" style="color:{vix_col}">{current_vix:.2f}</div>
        <div class="vix-sub" style="color:{vix_col}">{vix_mood}</div>
        <div class="fg-bar-bg" style="margin-top:8px">
            <div class="fg-bar-fill" style="width:{vix_pct:.0f}%"></div>
        </div>
        <div class="fg-labels">
            <span>Faible</span><span>Normal</span><span>Stress</span><span>Crise</span>
        </div>
        <div style="color:#606878;font-size:0.65rem;margin-top:6px">
            US 10Y · {current_10y*100:.2f}% &nbsp;|&nbsp; Delai ~15min
        </div>
    </div>''', unsafe_allow_html=True)

    # ── Marchés en temps réel ─────────────────────────────────
    st.markdown('<div class="sb-section-title" style="margin-top:16px">Marches</div>',
                unsafe_allow_html=True)

    if market_data:
        groups = [
            ("Indices", ["SPY","QQQ","DIA","IWM"]),
            ("Matieres premieres", ["GLD","USO"]),
            ("Macro", ["TLT","UUP"]),
            ("Crypto", ["BTC-USD"]),
        ]
        for group_name, syms in groups:
            group_html = f'''<div style="color:#505868;font-size:0.6rem;text-transform:uppercase;
                letter-spacing:0.1em;margin:8px 0 4px">{group_name}</div>'''
            for sym in syms:
                if sym not in market_data:
                    continue
                d = market_data[sym]
                chg_class = "mkt-chg-pos" if d['chg'] > 0 else ("mkt-chg-neg" if d['chg'] < 0 else "mkt-chg-neu")
                arrow = "▲" if d['chg'] > 0 else ("▼" if d['chg'] < 0 else "—")
                price_fmt = f"${d['price']:,.2f}" if d['price'] < 1000 else f"${d['price']:,.0f}"
                group_html += f'''
                <div class="mkt-card">
                    <div class="mkt-left">
                        <span class="mkt-icon">{d['icon']}</span>
                        <div>
                            <div class="mkt-label">{d['label']}</div>
                            <div class="mkt-price">{price_fmt}</div>
                        </div>
                    </div>
                    <div class="{chg_class}">{arrow} {abs(d['chg']):.2f}%</div>
                </div>'''
            st.markdown(group_html, unsafe_allow_html=True)
    else:
        st.caption("Donnees marche indisponibles.")

    # ── Surface de vol ────────────────────────────────────────
    st.markdown('<div class="sb-section-title" style="margin-top:14px">Surface Vol</div>',
                unsafe_allow_html=True)
    vol_surface_type = st.radio("Type", ['call', 'put'], horizontal=True,
                                 label_visibility="collapsed")

    st.markdown('''
    <div style="margin-top:14px;padding-top:12px;border-top:1px solid rgba(255,255,255,0.05);
        font-size:0.62rem;color:#404858;text-align:center;line-height:1.6">
        Yahoo Finance · ~15 min delay<br>
        Quantum Options Terminal v2.0
    </div>''', unsafe_allow_html=True)

if not ticker:
    st.info("Entrez un symbole dans la barre laterale.")
    st.stop()

# ── Load data ────────────────────────────────────────────────
try:
    info, hist, spot, expiry_dates_all = get_stock_data(ticker)
except Exception as e:
    st.error(f"Impossible de charger **{ticker}** : {e}")
    st.stop()

long_name = info.get('longName', ticker)
sector    = info.get('sector', 'N/A')
industry  = info.get('industry', 'N/A')
summary   = info.get('longBusinessSummary', '')
prev      = info.get('previousClose', spot)
chg       = spot - prev
chg_pct   = chg/prev*100 if prev else 0
mkt_cap   = info.get('marketCap', 0)
pe        = info.get('trailingPE', None)
beta      = info.get('beta', None)
log_ret   = np.log(hist['Close']/hist['Close'].shift(1)).dropna()
hv30_val  = log_ret.rolling(30).std().iloc[-1] * np.sqrt(252) * 100

# ── Header ───────────────────────────────────────────────────
chg_col = '#3fb950' if chg >= 0 else '#ff4b6e'
st.markdown(f'''
<div class="page-header">
    <div class="ticker-badge">{ticker}</div>
    <div>
        <div class="company-name">{long_name}</div>
        <div class="sector-tag">{sector} &middot; {industry}</div>
    </div>
    <div style="margin-left:auto;text-align:right">
        <div style="font-size:1.6rem;font-weight:700;font-family:JetBrains Mono,monospace;color:#fff">${spot:.2f}</div>
        <div style="font-size:0.82rem;color:{chg_col}">{chg:+.2f} ({chg_pct:+.2f}%)</div>
    </div>
</div>''', unsafe_allow_html=True)

if summary:
    st.markdown(f'<div class="profile-box">{summary[:520]}{"..." if len(summary)>520 else ""}</div>',
                unsafe_allow_html=True)

# ── KPIs ─────────────────────────────────────────────────────
st.markdown(f"""
<div class="kpi-row">
  <div class="kpi-card">
    <div class="kpi-label">Market Cap</div>
    <div class="kpi-value">${mkt_cap/1e9:.1f}B</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">P/E Ratio</div>
    <div class="kpi-value">{f"{pe:.1f}" if pe else "N/A"}</div>
    <div class="kpi-sub">Trailing</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Beta</div>
    <div class="kpi-value">{f"{beta:.2f}" if beta else "N/A"}</div>
    <div class="kpi-sub">vs S&P 500</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">HV 30j</div>
    <div class="kpi-value">{hv30_val:.1f}%</div>
    <div class="kpi-sub">Realisee</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">VIX</div>
    <div class="kpi-value">{current_vix:.2f}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── OHLCV ────────────────────────────────────────────────────
st.markdown('<div class="section-label">Historique Prix et Volume - 6 mois</div>', unsafe_allow_html=True)
st.plotly_chart(chart_ohlcv(hist), use_container_width=True)

# ── Vol Surface ──────────────────────────────────────────────
if expiry_dates_all and len(expiry_dates_all) >= 2:
    with st.spinner("Construction de la surface de volatilite..."):
        fig_surf, surf_meta = chart_vol_surface(ticker, expiry_dates_all, spot, r_rate, vol_surface_type)

    if fig_surf.data and surf_meta:
        sm = surf_meta
        # ── Structure badge ───────────────────────────────────
        struct_bg = {
            "BACKWARDATION": "rgba(255,75,110,0.15)",
            "CONTANGO":      "rgba(63,185,80,0.15)",
            "PLATE":         "rgba(245,166,35,0.15)",
        }.get(sm["structure"], "rgba(245,166,35,0.15)")
        struct_border = sm["struct_color"]
        opt_label = "CALLS" if vol_surface_type == "call" else "PUTS"

        # ── Smile skew label ──────────────────────────────────
        skew_v = sm["smile_skew"]
        if skew_v > 3:
            skew_label = f"PUT SKEW +{skew_v:.1f}pts"
            skew_col   = "#ff4b6e"
        elif skew_v < -3:
            skew_label = f"CALL SKEW {skew_v:.1f}pts"
            skew_col   = "#3fb950"
        else:
            skew_label = f"SMILE NEUTRE {skew_v:+.1f}pts"
            skew_col   = "#f5a623"

        # ── Term slope label ──────────────────────────────────
        ts = sm["term_slope"]
        ts_col = "#ff4b6e" if ts < -1 else ("#3fb950" if ts > 1 else "#f5a623")
        ts_label = f"{ts:+.1f}pts"

        st.markdown(f"""
        <div class="surf-wrap">
            <div class="surf-header">
                <div>
                    <div class="surf-title">📐 Surface de Volatilite Implicite
                        <span style="font-size:0.7rem;color:#606878;font-weight:400;margin-left:8px">
                            {opt_label} · {len(sm['expiries'])} echeances
                        </span>
                    </div>
                    <div class="surf-subtitle">
                        Chaque point = IV implicite du marche pour un strike et une echeance donnee
                    </div>
                </div>
                <span class="surf-badge"
                    style="background:{struct_bg};color:{struct_border};
                           border:1px solid {struct_border}40;">
                    {sm['structure']}
                </span>
            </div>
            <div class="surf-kpi-row">
                <div class="surf-kpi">
                    <div class="surf-kpi-label">IV ATM Court Terme</div>
                    <div class="surf-kpi-value" style="color:#00e5ff">{sm['iv_atm_near']:.1f}%</div>
                    <div class="surf-kpi-sub">{sm['expiries'][0] if sm['expiries'] else 'N/A'}</div>
                </div>
                <div class="surf-kpi">
                    <div class="surf-kpi-label">IV ATM Long Terme</div>
                    <div class="surf-kpi-value" style="color:#b06fff">{sm['iv_atm_far']:.1f}%</div>
                    <div class="surf-kpi-sub">{sm['expiries'][-1] if sm['expiries'] else 'N/A'}</div>
                </div>
                <div class="surf-kpi">
                    <div class="surf-kpi-label">Pente Term Structure</div>
                    <div class="surf-kpi-value" style="color:{ts_col}">{ts_label}</div>
                    <div class="surf-kpi-sub">{sm['struct_desc'][:28]}...</div>
                </div>
                <div class="surf-kpi">
                    <div class="surf-kpi-label">Smile / Skew 1M</div>
                    <div class="surf-kpi-value" style="color:{skew_col}">{skew_label}</div>
                    <div class="surf-kpi-sub">Put 90% vs Call 110%</div>
                </div>
                <div class="surf-kpi">
                    <div class="surf-kpi-label">Plage IV globale</div>
                    <div class="surf-kpi-value" style="color:#e0e6f0;font-size:0.9rem">
                        {sm['iv_min']:.1f}% – {sm['iv_max']:.1f}%
                    </div>
                    <div class="surf-kpi-sub">min / max surface</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── 3D Chart ──────────────────────────────────────────
        st.plotly_chart(fig_surf, use_container_width=True)

        # ── Term Structure Bar Chart (2D complementaire) ──────
        if len(sm["expiries"]) >= 3:
            atm_max = max(sm["atm_ivs"]) if sm["atm_ivs"] else 1
            bars_html = '<div class="term-bar-wrap"><div class="term-bar-title">📊 Term Structure — IV ATM par echeance</div>'
            for exp_lbl, iv_val, dte_val in zip(sm["expiries"], sm["atm_ivs"], sm["dte_list"]):
                pct = min(100, (iv_val / (atm_max * 1.05)) * 100)
                try:
                    d = datetime.strptime(exp_lbl, "%Y-%m-%d")
                    short = d.strftime("%d %b %y")
                except:
                    short = exp_lbl
                color = "#ff4b6e" if iv_val == max(sm["atm_ivs"]) else ("#00e5ff" if iv_val == min(sm["atm_ivs"]) else "#0088cc")
                bars_html += f"""
                <div class="term-bar-row">
                    <div class="term-bar-label">{short}</div>
                    <div class="term-bar-track">
                        <div class="term-bar-fill" style="width:{pct:.0f}%;background:{color};opacity:0.85"></div>
                    </div>
                    <div class="term-bar-val">{iv_val:.1f}%</div>
                    <div style="color:#505868;font-size:0.62rem;width:36px">{dte_val}j</div>
                </div>"""
            bars_html += '</div>'

            # ── Lecture de la structure ────────────────────────
            struct_explain = {
                "BACKWARDATION": (
                    "⚠️ <b style='color:#ff4b6e'>Backwardation</b> — La vol court terme est supérieure à la vol long terme. "
                    "Signe de stress ou d'evenement imminent (earnings, FDA, macro). "
                    "Les options court terme sont survalorisées — opportunite de les vendre si le risque est connu."
                ),
                "CONTANGO": (
                    "✅ <b style='color:#3fb950'>Contango</b> — La vol court terme est inferieure à la vol long terme. "
                    "Structure normale de marche calme. Le marche anticipe plus d'incertitude a long terme. "
                    "Favorable aux strategies de vente d'options court terme (theta positif)."
                ),
                "PLATE": (
                    "➡️ <b style='color:#f5a623'>Structure plate</b> — La vol implicite est similaire sur toutes les echeances. "
                    "Aucun signal fort de stress ou d'anticipation directionnelle. "
                    "Surveiller un changement de regime qui pourrait creer des opportunites."
                ),
            }.get(sm["structure"], "")

            st.markdown(bars_html + f"""
            <div style="margin:0 0 6px;padding:12px 22px;
                background:rgba(255,255,255,0.02);border-top:1px solid rgba(255,255,255,0.05)">
                <div style="font-size:0.78rem;color:#c9d1d9;line-height:1.7">{struct_explain}</div>
            </div>
            <div class="surf-hint">
                🖱 Cliquez-glissez pour pivoter · Molette pour zoomer · Double-clic pour reinitialiser
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="surf-hint">🖱 Cliquez-glissez pour pivoter · Molette pour zoomer</div>',
                        unsafe_allow_html=True)
    else:
        st.warning("Donnees insuffisantes pour construire la surface de volatilite.")
else:
    st.warning("Moins de 2 echeances disponibles pour ce ticker.")

# ── Options par échéance ─────────────────────────────────────
st.markdown('<div class="section-label">Analyse par Echeance</div>', unsafe_allow_html=True)

if not expiry_dates_all:
    st.warning(f"Aucune option listee pour {ticker}.")
    st.stop()

expiry      = st.selectbox("Echeance", expiry_dates_all)
T           = max(1/365, (datetime.strptime(expiry, "%Y-%m-%d") - datetime.now()).days / 365.0)
days_to_exp = max(0, (datetime.strptime(expiry, "%Y-%m-%d") - datetime.now()).days)

try:
    chain_calls, chain_puts = get_option_chain(ticker, expiry)
except Exception as e:
    st.error(f"Erreur chaine d'options : {e}")
    st.stop()

# ============================================================
# SECTION : EARNINGS ALERT + IV RANK + IMPLIED MOVE
# ============================================================

st.markdown('<div class="section-label">Intelligence Options — Earnings / IV Rank / Implied Move</div>',
            unsafe_allow_html=True)

col_earn, col_ivr, col_move = st.columns([1, 1, 1.2])

# ── Earnings Alert ───────────────────────────────────────────
with col_earn:
    with st.spinner("Recherche earnings..."):
        earn_date = get_earnings_date(ticker)

    if earn_date:
        days_to_earn = (earn_date - datetime.now().date()).days
        earn_before_expiry = (earn_date <= datetime.strptime(expiry, "%Y-%m-%d").date())

        if days_to_earn <= 0:
            earn_label = "Annonce recente"
            earn_color = "#8b949e"
            urgency    = ""
        elif days_to_earn <= 7:
            earn_label = f"Dans {days_to_earn} jour{'s' if days_to_earn>1 else ''}"
            earn_color = "#ff4b6e"
            urgency    = "IMMINENT"
        elif days_to_earn <= 30:
            earn_label = f"Dans {days_to_earn} jours"
            earn_color = "#f5a623"
            urgency    = "PROCHE"
        else:
            earn_label = f"Dans {days_to_earn} jours"
            earn_color = "#3fb950"
            urgency    = ""

        warn_txt = ""
        if earn_before_expiry and days_to_earn > 0:
            warn_txt = "⚠️ Les earnings tombent AVANT l'expiration selectionnee — IV probablement gonflée, prime elevee."
        elif not earn_before_expiry and days_to_earn > 0:
            warn_txt = "✅ Les earnings sont APRES l'expiration — pas d'impact direct sur cette echeance."

        st.markdown(f'''
        <div class="earnings-alert">
            <div class="ea-title">📅 Prochains Earnings {f"· <span style='color:{earn_color};font-size:0.7rem;font-weight:700'>{urgency}</span>" if urgency else ""}</div>
            <div class="ea-date" style="color:{earn_color}">{earn_date.strftime("%d %b %Y")}</div>
            <div class="ea-days" style="color:{earn_color}">{earn_label}</div>
            {f'<div class="ea-warn">{warn_txt}</div>' if warn_txt else ""}
        </div>''', unsafe_allow_html=True)
    else:
        st.markdown('''
        <div class="earnings-alert" style="border-color:rgba(139,148,158,0.3)">
            <div class="ea-title" style="color:#8b949e">📅 Earnings</div>
            <div style="color:#8b949e;font-size:0.85rem;margin-top:6px">
                Date non trouvee via Yahoo Finance pour ce ticker.<br>
                <span style="font-size:0.75rem;color:#505868">
                Essayez de changer d'echeance ou verifiez sur Nasdaq.com / Earnings Whispers.
                </span>
            </div>
        </div>''', unsafe_allow_html=True)

# ── IV Rank ──────────────────────────────────────────────────
with col_ivr:
    hv_series = get_iv_history(ticker)
    atm_calls = chain_calls.dropna(subset=['impliedVolatility'])
    atm_puts  = chain_puts.dropna(subset=['impliedVolatility'])

    if not atm_calls.empty:
        atm_c_row = atm_calls.iloc[(atm_calls['strike']-spot).abs().argsort()[:1]]
        current_iv_pct = float(atm_c_row['impliedVolatility'].values[0]) * 100
    else:
        current_iv_pct = hv30_val

    iv_result = compute_iv_rank(current_iv_pct, hv_series)
    if iv_result[0] is not None:
        iv_rank, iv_pct, iv_min, iv_max = iv_result
        bar_color = "#3fb950" if iv_rank < 30 else ("#ff4b6e" if iv_rank > 70 else "#f5a623")
        rank_signal = ("Vendre des options — IV elevee" if iv_rank > 70
                       else ("Acheter des options — IV basse" if iv_rank < 30
                             else "Zone neutre"))
        st.markdown(f'''
        <div class="ivrank-box">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
                <span style="color:#8b949e;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em">IV Rank 52 sem.</span>
                <span style="color:{bar_color};font-size:1.5rem;font-weight:700;font-family:JetBrains Mono,monospace">{iv_rank:.0f}%</span>
            </div>
            <div class="ivrank-bar-bg">
                <div class="ivrank-bar-fill" style="width:{min(100,iv_rank):.0f}%"></div>
            </div>
            <div class="ivrank-labels"><span>0 (faible)</span><span>50</span><span>100 (elevee)</span></div>
            <div style="margin-top:10px;padding-top:10px;border-top:1px solid rgba(255,255,255,0.06)">
                <div style="color:#606878;font-size:0.65rem;text-transform:uppercase;margin-bottom:4px">IV Percentile</div>
                <div style="color:#e0e6f0;font-size:0.9rem;font-family:JetBrains Mono,monospace">{iv_pct:.0f}%
                    <span style="color:#606878;font-size:0.72rem"> · IV actuelle {current_iv_pct:.1f}%</span></div>
                <div style="color:#606878;font-size:0.68rem;margin-top:4px">Min {iv_min:.1f}% / Max {iv_max:.1f}%</div>
            </div>
            <div style="margin-top:10px;background:rgba(0,0,0,0.2);border-radius:6px;padding:8px 10px;
                color:{bar_color};font-size:0.75rem;font-weight:600">
                {rank_signal}
            </div>
        </div>''', unsafe_allow_html=True)
    else:
        st.info("IV Rank indisponible.")

# ── Implied Move ─────────────────────────────────────────────
with col_move:
    move_usd, move_pct, straddle = compute_implied_move(chain_calls, chain_puts, spot)

    if move_usd is not None:
        price_up  = round(spot + move_usd, 2)
        price_dn  = round(spot - move_usd, 2)

        # Commentaire contextuel
        earn_context = ""
        if earn_date:
            days_to_earn = (earn_date - datetime.now().date()).days
            if 0 < days_to_earn <= days_to_exp:
                earn_context = (
                    f"Les earnings sont dans {days_to_earn} jours, AVANT cette expiration. "
                    f"Le marche anticipe un mouvement de ±{move_pct:.1f}% — "
                    f"probablement principalement du a l'annonce de resultats. "
                    f"La prime du straddle ATM (${straddle:.2f}) reflète cette incertitude.")
            else:
                earn_context = (
                    f"Pas d'earnings avant cette expiration. "
                    f"Le mouvement implique de ±{move_pct:.1f}% reflete la volatilite courante.")

        if move_pct > 10:
            move_comment = f"Mouvement ELEVE ({move_pct:.1f}%) — marche tres incertain, options cheres."
        elif move_pct > 5:
            move_comment = f"Mouvement modere ({move_pct:.1f}%) — volatilite dans la norme."
        else:
            move_comment = f"Mouvement faible ({move_pct:.1f}%) — options bon marche, faible incertitude."

        earn_html = f"<b>Contexte earnings :</b> {earn_context}<br><br>" if earn_context else ""
        move_html = f'''
        <div class="move-box">
            <div class="move-title">Implied Move a Expiration</div>
            <div class="move-value">±{move_pct:.2f}%</div>
            <div class="move-sub">±${move_usd:.2f} · Straddle ATM: ${straddle:.2f}</div>
            <div class="move-range">
                <div class="mr-item">
                    <div class="mr-label">Borne haute</div>
                    <div class="mr-val up">${price_up:.2f}</div>
                </div>
                <div class="mr-item">
                    <div class="mr-label">Spot actuel</div>
                    <div class="mr-val">${spot:.2f}</div>
                </div>
                <div class="mr-item">
                    <div class="mr-label">Borne basse</div>
                    <div class="mr-val dn">${price_dn:.2f}</div>
                </div>
            </div>
            <div class="move-commentary">
                {earn_html}<b>Signal :</b> {move_comment}<br><br><b>Note :</b> borne estimee avec l'heuristique 85% du straddle ATM.
            </div>
        </div>'''
        st.markdown(move_html, unsafe_allow_html=True)
    else:
        st.warning("Implied Move indisponible.")

# ============================================================
# ONGLETS PRINCIPAUX
# ============================================================

tab_c, tab_p, tab_oi, tab_skew, tab_pop, tab_flow, tab_strat, tab_roll = st.tabs(
    ["CALLS", "PUTS", "OI & GEX", "SKEW", "PROBABILITES", "OPTIONS FLOW", "STRATEGIES", "ROLL ANALYZER"])

for tab, data_raw, o_type in zip([tab_c, tab_p], [chain_calls, chain_puts], ['call','put']):
    with tab:
        data = data_raw.dropna(subset=['impliedVolatility','strike']).copy()
        if data.empty:
            st.warning("Aucune donnee."); continue

        atm_row  = data.iloc[(data['strike']-spot).abs().argsort()[:1]]
        atm_iv   = atm_row['impliedVolatility'].values[0]
        iv_vs_hv = atm_iv*100 - hv30_val
        sig_col  = "#ff4b6e" if iv_vs_hv > 5 else ("#3fb950" if iv_vs_hv < -5 else "#f5a623")
        sig_txt  = ("IV > HV - options potentiellement surevaluees (favorable vendeur)"
                    if iv_vs_hv > 5 else
                    ("IV < HV - options potentiellement sous-evaluees (favorable acheteur)"
                     if iv_vs_hv < -5 else "IV appr. HV - valorisation neutre"))

        st.markdown('<div class="section-label">Volatilite Historique vs Implicite</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(chart_hv_iv(hist, data, spot), use_container_width=True)
        st.markdown(f'''
        <div class="signal-card" style="border-left:3px solid {sig_col}">
            <b style="color:{sig_col}">Signal IV/HV</b> &middot; {sig_txt}
            <span style="color:{sig_col};font-family:monospace;margin-left:8px">{iv_vs_hv:+.1f}%</span>
        </div>''', unsafe_allow_html=True)

        st.markdown('<div class="section-label">Prix BSM vs Prix Marche</div>', unsafe_allow_html=True)
        mask = (data['strike'] > spot*0.80) & (data['strike'] < spot*1.20)
        pd_  = data[mask].copy()
        pd_['BS_Price'] = pd_.apply(
            lambda x: calculate_greeks(spot, x.strike, T, r_rate, x.impliedVolatility, o_type)['Price'], axis=1)
        st.plotly_chart(chart_bsm_vs_market(pd_, spot), use_container_width=True)

        greeks = calculate_greeks(spot, atm_row['strike'].values[0], T, r_rate, atm_iv, o_type)
        st.markdown(f'<div class="section-label">Greeks ATM - Strike ${atm_row["strike"].values[0]:.0f} - {days_to_exp} jours</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class="greek-row">
          <div class="greek-card"><div class="g-name">Delta</div><div class="g-value">{greeks['Delta']:.3f}</div></div>
          <div class="greek-card"><div class="g-name">Gamma</div><div class="g-value">{greeks['Gamma']:.4f}</div></div>
          <div class="greek-card"><div class="g-name">Theta /j</div><div class="g-value">{greeks['Theta']:.3f}</div></div>
          <div class="greek-card"><div class="g-name">Vega</div><div class="g-value">{greeks['Vega']:.3f}</div></div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-label">Greeks 2e Ordre</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="greek-row-2">
          <div class="greek-card-2"><div class="g-name-2">Vanna</div><div class="g-value-2">{greeks['Vanna']:.4f}</div></div>
          <div class="greek-card-2"><div class="g-name-2">Charm</div><div class="g-value-2">{greeks['Charm']:.5f}</div></div>
          <div class="greek-card-2"><div class="g-name-2">Volga</div><div class="g-value-2">{greeks['Volga']:.4f}</div></div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-label">Simulateur P/L</div>', unsafe_allow_html=True)
        c_in, c_res = st.columns([1, 1.3])
        with c_in:
            sel_strike  = st.selectbox("Strike", sorted(data['strike'].unique()), key=f"sk_{o_type}")
            nb_contrats = st.number_input("Contrats (x100)", min_value=1, value=1, key=f"nb_{o_type}")
            target_p    = st.number_input("Prix cible a expiration ($)",
                                           value=round(float(spot*1.1), 2), key=f"tp_{o_type}")
            direction   = st.radio("Position", ["Long (Acheteur)", "Short (Vendeur)"],
                                   horizontal=True, key=f"dir_{o_type}")

        row_s = data[data['strike'] == sel_strike]
        if row_s.empty:
            st.warning("Strike introuvable."); continue

        price_buy  = get_option_quote_price(row_s.iloc[0], 'mid')
        if price_buy is None:
            st.warning("Prime de marche indisponible pour ce strike."); continue
        iv_sel     = row_s['impliedVolatility'].values[0]
        is_long    = "Long" in direction
        premium_total = nb_contrats * price_buy * 100
        val_finale = nb_contrats * max(0, (target_p-sel_strike) if o_type=='call' else (sel_strike-target_p)) * 100
        pnl        = (val_finale - premium_total) if is_long else (premium_total - val_finale)
        breakeven  = (sel_strike+price_buy) if o_type=='call' else (sel_strike-price_buy)
        pc         = "pos" if pnl >= 0 else "neg"
        entry_label = "Debit initial" if is_long else "Credit initial"
        value_label = "Valeur a expiration" if is_long else "Cout de rachat a expiration"
        risk_row = ""
        roi_label = "ROI sur prime"
        if is_long:
            roi_display = f"{(pnl/premium_total*100):.2f}%" if premium_total > 0 else "N/A"
            roi_class = pc
        elif o_type == 'put':
            max_loss = nb_contrats * max(sel_strike - price_buy, 0) * 100
            risk_row = f"<tr><td class=\"lbl\">Risque max</td><td class=\"val\">${max_loss:,.2f}</td></tr>"
            roi_label = "ROI / risque max"
            roi_display = f"{(pnl/max_loss*100):.2f}%" if max_loss > 0 else "N/A"
            roi_class = pc if max_loss > 0 else ""
        else:
            risk_row = '<tr><td class="lbl">Risque max</td><td class="val">Illimite</td></tr>'
            roi_label = "ROI"
            roi_display = "N/A"
            roi_class = ""

        with c_res:
            st.markdown(f"""
            <table class="pnl-table">
              <tr><td class="lbl">Prime par contrat</td><td class="val">${price_buy:.2f}</td></tr>
              <tr><td class="lbl">{entry_label}</td><td class="val">${premium_total:,.2f}</td></tr>
              <tr><td class="lbl">{value_label}</td><td class="val">${val_finale:,.2f}</td></tr>
              <tr><td class="lbl">Point mort</td><td class="val">${breakeven:.2f}</td></tr>
              {risk_row}
              <tr class="total"><td>P/L Net</td><td class="val {pc}">${pnl:,.2f}</td></tr>
              <tr><td class="lbl">{roi_label}</td><td class="val {roi_class}">{roi_display}</td></tr>
            </table>""", unsafe_allow_html=True)

        st.plotly_chart(
            chart_pnl_multiscenario(spot, sel_strike, premium_total, nb_contrats, T, r_rate, iv_sel, o_type,
                                    direction='long' if is_long else 'short'),
            use_container_width=True)

# ── OI & GEX ─────────────────────────────────────────────────
with tab_oi:
    st.markdown('<div class="section-label">Open Interest et Gamma Exposure</div>', unsafe_allow_html=True)
    try:
        fig_oi, oi_c, oi_p = chart_open_interest(chain_calls, chain_puts, spot, T=T, r_rate=r_rate)
        st.plotly_chart(fig_oi, use_container_width=True)

        # ── Calculs ──────────────────────────────────────────────
        total_c = chain_calls['openInterest'].fillna(0).sum()
        total_p = chain_puts['openInterest'].fillna(0).sum()
        pcr     = total_p / total_c if total_c > 0 else 0

        gex_df   = pd.concat([oi_c[['strike','GEX']], oi_p[['strike','GEX']]]).groupby('strike')['GEX'].sum().reset_index()
        dominant = gex_df.loc[gex_df['GEX'].abs().idxmax(), 'strike']
        total_gex_calls = oi_c['GEX'].sum()
        total_gex_puts  = oi_p['GEX'].sum()
        net_gex         = total_gex_calls + total_gex_puts

        # Concentration OI : où est le max OI calls vs puts ?
        max_oi_call_strike = chain_calls.loc[chain_calls['openInterest'].fillna(0).idxmax(), 'strike'] if not chain_calls.empty else spot
        max_oi_put_strike  = chain_puts.loc[chain_puts['openInterest'].fillna(0).idxmax(), 'strike']  if not chain_puts.empty else spot
        call_wall = max_oi_call_strike  # résistance
        put_wall  = max_oi_put_strike   # support

        # ── Signaux individuels ───────────────────────────────────
        # PCR
        if pcr > 1.5:
            pcr_badge = "badge-bear"; pcr_bias = "BEARISH"; pcr_col = "#ff4b6e"
            pcr_comment = (f"PCR de {pcr:.2f} — très élevé. La demande de puts dépasse largement les calls. "
                           f"Signe de protection ou de paris baissiers massifs. "
                           f"Attention : un PCR extrême peut aussi signaler un plancher de peur (contrarian bullish).")
        elif pcr > 1.2:
            pcr_badge = "badge-bear"; pcr_bias = "LÉGÈREMENT BEARISH"; pcr_col = "#ff4b6e"
            pcr_comment = (f"PCR de {pcr:.2f} — au-dessus de 1. Plus de puts que de calls en circulation. "
                           f"Les participants se protègent davantage qu'ils ne spéculent à la hausse. "
                           f"Signal modérément baissier ou prudent.")
        elif pcr < 0.5:
            pcr_badge = "badge-bull"; pcr_bias = "TRÈS BULLISH"; pcr_col = "#3fb950"
            pcr_comment = (f"PCR de {pcr:.2f} — très bas. Forte dominance des calls. "
                           f"Les traders anticipent une hausse marquée ou spéculent agressivement. "
                           f"Peut signaler une exubérance excessive — surveiller un retournement.")
        elif pcr < 0.8:
            pcr_badge = "badge-bull"; pcr_bias = "BULLISH"; pcr_col = "#3fb950"
            pcr_comment = (f"PCR de {pcr:.2f} — inférieur à 0.8. Les calls dominent. "
                           f"Le marché positionne davantage sur la hausse que sur la baisse. "
                           f"Signal haussier de sentiment.")
        else:
            pcr_badge = "badge-neut"; pcr_bias = "NEUTRE"; pcr_col = "#f5a623"
            pcr_comment = (f"PCR de {pcr:.2f} — dans la zone neutre (0.8–1.2). "
                           f"Équilibre relatif entre acheteurs de calls et de puts. "
                           f"Pas de signal directionnel fort — attendre une cassure.")

        # GEX net
        if net_gex > 0:
            gex_badge = "badge-bull"; gex_bias = "STABILISANT"; gex_col = "#3fb950"
            gex_comment = (f"GEX net positif (${net_gex/1e6:.1f}M). Les market makers sont long gamma. "
                           f"Ils vendent quand le prix monte et achètent quand il baisse → "
                           f"effet d'ancrage autour du strike dominant (${dominant:.0f}). "
                           f"Environnement à faible volatilité réalisée probable.")
        else:
            gex_badge = "badge-bear"; gex_bias = "DÉSTABILISANT"; gex_col = "#ff4b6e"
            gex_comment = (f"GEX net négatif (${net_gex/1e6:.1f}M). Les market makers sont short gamma. "
                           f"Ils achètent quand le prix monte et vendent quand il baisse → "
                           f"ils amplifient les mouvements. "
                           f"Environnement propice aux mouvements brusques et à la volatilité élevée.")

        # Call Wall / Put Wall
        cw_dist = ((call_wall - spot) / spot) * 100
        pw_dist = ((put_wall  - spot) / spot) * 100
        if call_wall > spot:
            wall_bias = "BULLISH" if abs(cw_dist) < abs(pw_dist) else "BEARISH"
            wall_comment = (f"Call Wall (résistance) à ${call_wall:.0f} ({cw_dist:+.1f}% du spot) — "
                            f"forte concentration de calls, les MMs hedgent en vendant au-dessus. "
                            f"Put Wall (support) à ${put_wall:.0f} ({pw_dist:+.1f}% du spot) — "
                            f"les MMs achètent en dessous pour se couvrir.")
            wall_badge = "badge-bull" if wall_bias == "BULLISH" else "badge-bear"
            wall_col   = "#3fb950" if wall_bias == "BULLISH" else "#ff4b6e"
        else:
            wall_bias    = "BEARISH"; wall_badge = "badge-bear"; wall_col = "#ff4b6e"
            wall_comment = (f"Call Wall à ${call_wall:.0f} ({cw_dist:+.1f}%) — en dessous du spot, "
                            f"résistance déjà franchie ou repositionnement. "
                            f"Put Wall à ${put_wall:.0f} ({pw_dist:+.1f}%) — "
                            f"support potentiel si le prix recule.")

        # ── Affichage des 3 cartes ────────────────────────────────
        st.markdown(f'''
        <div class="analysis-grid">

          <div class="analysis-card" style="border-left:3px solid {pcr_col}">
            <div class="ac-header">
              <span class="ac-title">Put / Call Ratio</span>
              <span class="ac-badge {pcr_badge}">{pcr_bias}</span>
            </div>
            <div class="ac-value" style="color:{pcr_col}">{pcr:.3f}</div>
            <div class="ac-sub">{total_c:,.0f} calls · {total_p:,.0f} puts</div>
            <div class="ac-comment">{pcr_comment}</div>
          </div>

          <div class="analysis-card" style="border-left:3px solid {gex_col}">
            <div class="ac-header">
              <span class="ac-title">GEX Net</span>
              <span class="ac-badge {gex_badge}">{gex_bias}</span>
            </div>
            <div class="ac-value" style="color:{gex_col}">${net_gex/1e6:.1f}M</div>
            <div class="ac-sub">Strike dominant : ${dominant:.0f}</div>
            <div class="ac-comment">{gex_comment}</div>
          </div>

          <div class="analysis-card" style="border-left:3px solid {wall_col}">
            <div class="ac-header">
              <span class="ac-title">Call Wall / Put Wall</span>
              <span class="ac-badge {wall_badge}">{wall_bias}</span>
            </div>
            <div class="ac-value" style="color:#e0e6f0">${call_wall:.0f} / ${put_wall:.0f}</div>
            <div class="ac-sub">Résistance / Support OI majeurs</div>
            <div class="ac-comment">{wall_comment}</div>
          </div>

        </div>''', unsafe_allow_html=True)

        # ── Synthèse globale ──────────────────────────────────────
        bull_signals = []
        bear_signals = []
        neut_signals = []

        if pcr < 0.8:  bull_signals.append("PCR haussier")
        elif pcr > 1.2: bear_signals.append("PCR baissier")
        else:           neut_signals.append("PCR neutre")

        if net_gex > 0: bull_signals.append("GEX stabilisant")
        else:           bear_signals.append("GEX déstabilisant")

        if call_wall > spot and cw_dist > 0:
            if abs(cw_dist) > 3: neut_signals.append(f"Call Wall loin (+{cw_dist:.1f}%)")
            else: bull_signals.append(f"Call Wall proche (+{cw_dist:.1f}%)")
        else:
            bear_signals.append("Call Wall sous le spot")

        if put_wall < spot:
            if abs(pw_dist) > 3: neut_signals.append(f"Put Wall loin ({pw_dist:.1f}%)")
            else: bull_signals.append(f"Put Wall proche ({pw_dist:.1f}%)")

        score = len(bull_signals) - len(bear_signals)
        if score >= 2:
            synth_color = "#3fb950"; synth_label = "BULLISH"; synth_icon = "↑"
            synth_desc = ("Les signaux OI/GEX sont majoritairement haussiers. "
                          "La structure options favorise une continuation à la hausse à court terme.")
        elif score <= -2:
            synth_color = "#ff4b6e"; synth_label = "BEARISH"; synth_icon = "↓"
            synth_desc = ("Les signaux OI/GEX sont majoritairement baissiers. "
                          "La structure options suggère une pression vendeuse ou une demande de protection élevée.")
        elif score == 1:
            synth_color = "#3fb950"; synth_label = "LÉGÈREMENT BULLISH"; synth_icon = "↗"
            synth_desc = "Légère dominance haussière — pas de signal fort, surveiller l'évolution de l'OI."
        elif score == -1:
            synth_color = "#ff4b6e"; synth_label = "LÉGÈREMENT BEARISH"; synth_icon = "↘"
            synth_desc = "Légère dominance baissière — prudence, pas de signal fort."
        else:
            synth_color = "#f5a623"; synth_label = "NEUTRE / MIXTE"; synth_icon = "→"
            synth_desc = ("Signaux contradictoires ou équilibrés. "
                          "Le marché options ne donne pas de direction claire — attendre un catalyseur.")

        pills_html = ""
        for s in bull_signals: pills_html += f'<span class="sig-pill sig-bull">✓ {s}</span>'
        for s in bear_signals: pills_html += f'<span class="sig-pill sig-bear">✗ {s}</span>'
        for s in neut_signals: pills_html += f'<span class="sig-pill sig-neut">~ {s}</span>'

        st.markdown(f'''
        <div class="synthesis-box">
            <div class="synthesis-title">Synthese — Lecture globale du flux options</div>
            <div class="synthesis-score-row">
                <div class="synthesis-score" style="color:{synth_color}">{synth_icon}</div>
                <div>
                    <div class="synthesis-label" style="color:{synth_color}">{synth_label}</div>
                    <div class="synthesis-desc">{synth_desc}</div>
                </div>
            </div>
            <div class="synthesis-signals">{pills_html}</div>
        </div>''', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Erreur OI/GEX : {e}")

# ── Strategies ────────────────────────────────────────────────
with tab_strat:
    st.markdown('<div class="section-label">Strategies sur Options — Payoff & Pedagogie</div>', unsafe_allow_html=True)

    # ── Dictionnaire des fiches pedagogiques ─────────────────
    STRAT_INFO = {
        "Bull Call Spread": {
            "icon": "📈", "color": "#3fb950",
            "subtitle": "Strategie haussiere a cout reduit",
            "tags": [("HAUSSIER","tag-bull"),("DEBIT","tag-info"),("RISQUE LIMITE","tag-risk"),("GAIN PLAFONNE","tag-neut")],
            "principe": (
                "Acheter un call a strike bas (ITM ou ATM) et vendre simultanement un call a strike plus "
                "eleve (OTM). La prime recue sur le call vendu reduit le cout total de la position. "
                "Le gain est plafonné a la difference entre les deux strikes moins le debit paye."
            ),
            "construction": [
                "BUY Call K1 (proche du prix actuel)",
                "SELL Call K2 (au-dessus du prix actuel)",
                "Debit net = Prix Call K1 − Prix Call K2",
                "Break-even = K1 + Debit paye",
            ],
            "ideal": (
                "<b>Contexte ideal :</b> Vous anticipez une hausse modérée et reguliere. "
                "L'action a un catalyseur haussier mais vous ne voulez pas payer la pleine prime d'un call sec. "
                "IV elevee → le call vendu reduit plus efficacement le cout. "
                "A eviter si vous pensez que le titre peut exploser bien au-dela de K2."
            ),
            "warn": "<b>Risque :</b> Gain maximum plafonné a K2. Si le titre monte tres fort, vous perdez le surplus vs un call seul.",
        },
        "Bear Put Spread": {
            "icon": "📉", "color": "#ff4b6e",
            "subtitle": "Strategie baissiere a cout reduit",
            "tags": [("BAISSIER","tag-bear"),("DEBIT","tag-info"),("RISQUE LIMITE","tag-risk"),("GAIN PLAFONNE","tag-neut")],
            "principe": (
                "Acheter un put a strike eleve (ITM ou ATM) et vendre simultanement un put a strike "
                "plus bas (OTM). Le put vendu reduit le cout de la protection. Profitable si le titre "
                "baisse significativement avant l'expiration."
            ),
            "construction": [
                "BUY Put K1 (proche du prix actuel ou au-dessus)",
                "SELL Put K2 (en-dessous du prix actuel)",
                "Debit net = Prix Put K1 − Prix Put K2",
                "Break-even = K1 − Debit paye",
            ],
            "ideal": (
                "<b>Contexte ideal :</b> Anticipation d'une baisse moderee a forte. Earnings negatifs attendus, "
                "deterioration des fondamentaux, marche en distribution. "
                "L'IV elevee fait monter le put vendu → reduit encore plus le cout. "
                "Alternative moins chere a l'achat de put sec."
            ),
            "warn": "<b>Risque :</b> Gain limite a K1 − K2. Si le titre s'effondre bien sous K2, vous ne profitez pas de la totalite de la baisse.",
        },
        "Long Straddle": {
            "icon": "⚡", "color": "#f5a623",
            "subtitle": "Pari sur un mouvement fort dans n'importe quelle direction",
            "tags": [("NEUTRE","tag-neut"),("DEBIT","tag-info"),("VOLATILITE","tag-risk"),("ILLIMITE","tag-bull")],
            "principe": (
                "Acheter simultanement un call et un put au meme strike (ATM) et a la meme echeance. "
                "La position est profitable si le titre bouge suffisamment dans l'une ou l'autre direction "
                "pour couvrir le cout total des deux primes. C'est un pari pur sur la volatilite realisee."
            ),
            "construction": [
                "BUY Call K (ATM — au prix actuel)",
                "BUY Put K (meme strike, meme echeance)",
                "Cout total = Prime Call + Prime Put",
                "Break-even haut = K + Cout | Break-even bas = K − Cout",
            ],
            "ideal": (
                "<b>Contexte ideal :</b> Avant un evenement binaire majeur (earnings, FDA, resultat judiciaire, "
                "decision Fed). Vous savez qu'il va se passer QUELQUE CHOSE mais pas dans quel sens. "
                "IV faible avant l'evenement → le straddle coute moins cher. "
                "A acheter AVANT la compression de volatilite, pas apres."
            ),
            "warn": "<b>Risque :</b> Si le titre ne bouge pas assez, les deux options perdent de la valeur par theta. Tres sensible a la baisse de la vol implicite apres l'evenement (vol crush).",
        },
        "Long Strangle": {
            "icon": "🌀", "color": "#b06fff",
            "subtitle": "Pari sur un grand mouvement, a moindre cout",
            "tags": [("NEUTRE","tag-neut"),("DEBIT","tag-info"),("MOINS CHER","tag-bull"),("GRAND MOUVEMENT","tag-risk")],
            "principe": (
                "Acheter un call OTM et un put OTM a des strikes differents (ecarts au-dela du prix actuel). "
                "Moins cher qu'un straddle car les deux options sont hors-de-la-monnaie, mais necessite "
                "un mouvement encore plus fort pour etre profitable. Excellent ratio risque/recompense "
                "si un mouvement violent est attendu."
            ),
            "construction": [
                "BUY Put Kp (en-dessous du prix — OTM)",
                "BUY Call Kc (au-dessus du prix — OTM)",
                "Cout total = Prime Put + Prime Call (inferieur au straddle)",
                "Break-even haut = Kc + Cout | Break-even bas = Kp − Cout",
            ],
            "ideal": (
                "<b>Contexte ideal :</b> Meme situations que le straddle (earnings, evenements macro) "
                "mais quand la vol implicite est deja elevee et le straddle trop cher. "
                "Le strangle offre une marge de securite supplementaire et un cout reduit. "
                "Ideal si vous pensez que le titre peut faire un gap de ±10%+."
            ),
            "warn": "<b>Risque :</b> Necessite un mouvement encore plus grand que le straddle pour etre profitable. Les deux options perdent 100% si le titre reste dans la range Kp-Kc.",
        },
        "Iron Condor": {
            "icon": "🦅", "color": "#00e5ff",
            "subtitle": "Vendre de la volatilite — encaisser un credit si le marche reste calme",
            "tags": [("NEUTRE","tag-neut"),("CREDIT","tag-bull"),("IV ELEVEE","tag-risk"),("RISQUE LIMITE","tag-info")],
            "principe": (
                "Vendre un put spread (sell put K2, buy put K1 plus bas) ET vendre un call spread "
                "(sell call K3, buy call K4 plus haut) simultanement. On encaisse un credit net. "
                "La position est profitable si le titre reste entre K2 et K3 a l'expiration. "
                "C'est la strategie des vendeurs de volatilite par excellence."
            ),
            "construction": [
                "BUY Put K1 (protection basse — OTM loin)",
                "SELL Put K2 (proche du prix — OTM modere)",
                "SELL Call K3 (proche du prix — OTM modere)",
                "BUY Call K4 (protection haute — OTM loin)",
                "Credit net = Primes vendues − Primes achetees",
                "Profit max = Credit encaisse (si prix entre K2 et K3)",
            ],
            "ideal": (
                "<b>Contexte ideal :</b> IV Rank eleve (> 50%) — vous vendez de la vol chere. "
                "Marche lateral attendu, apres un evenement deja passe (post-earnings). "
                "Actions avec catalyseurs deja digeres et faible momentum directionnel. "
                "Theta positif : chaque jour qui passe joue en votre faveur."
            ),
            "warn": "<b>Risque :</b> Un mouvement violent dans un sens casse un des spreads. Perte max = largeur du spread − credit recu. Eviter en periode de forte tendance ou avant earnings.",
        },
    }

    # ── Selecteur de strategie stylise ───────────────────────
    strat_list = list(STRAT_INFO.keys())
    strat_icons = [STRAT_INFO[s]["icon"] for s in strat_list]

    col_sel = st.columns(len(strat_list))
    if "selected_strat" not in st.session_state:
        st.session_state.selected_strat = strat_list[0]

    for i, (col, name) in enumerate(zip(col_sel, strat_list)):
        info = STRAT_INFO[name]
        is_active = st.session_state.selected_strat == name
        border = f"border:2px solid {info['color']};" if is_active else "border:1px solid rgba(255,255,255,0.07);"
        bg = f"background:rgba(0,0,0,0.3);" if is_active else "background:rgba(13,19,32,0.6);"
        col.markdown(f"""
        <div style="{bg}{border}border-radius:12px;padding:12px 8px;text-align:center;
            cursor:pointer;transition:all 0.2s;margin-bottom:4px;">
            <div style="font-size:1.5rem">{info['icon']}</div>
            <div style="font-size:0.62rem;color:{'#fff' if is_active else '#8b949e'};
                font-weight:{'700' if is_active else '400'};margin-top:4px;line-height:1.3">
                {name.replace(' ','<br>')}
            </div>
        </div>""", unsafe_allow_html=True)
        if col.button("Selectionner", key=f"strat_btn_{i}", use_container_width=True):
            st.session_state.selected_strat = name
            st.rerun()

    strat = st.session_state.selected_strat
    info  = STRAT_INFO[strat]

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Fiche pedagogique ─────────────────────────────────────
    tags_html = "".join(f'<span class="scd-tag {t[1]}">{t[0]}</span>' for t in info["tags"])
    steps_html = "".join(f"<li>{s}</li>" for s in info["construction"])
    st.markdown(f"""
    <div class="strat-card-desc">
        <div class="scd-header">
            <div class="scd-icon">{info['icon']}</div>
            <div>
                <div class="scd-title" style="color:{info['color']}">{strat}</div>
                <div class="scd-subtitle">{info['subtitle']}</div>
            </div>
        </div>
        <div class="scd-tags">{tags_html}</div>
        <div class="scd-body">
            <div class="scd-section">
                <div class="scd-section-title">💡 Principe</div>
                <div class="scd-text">{info['principe']}</div>
            </div>
            <div class="scd-section">
                <div class="scd-section-title">🔧 Construction</div>
                <ul class="scd-steps">{steps_html}</ul>
            </div>
        </div>
        <div class="scd-ideal">{info['ideal']}</div>
        <div class="scd-warn">{info['warn']}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Configurateur de strikes ──────────────────────────────
    st.markdown('<div class="section-label" style="margin-top:10px">Configurateur</div>', unsafe_allow_html=True)
    sa   = sorted(chain_calls['strike'].unique())
    aidx = int(np.argmin(np.abs(np.array(sa)-spot)))
    try:
        if strat == "Bull Call Spread":
            c1, c2 = st.columns(2)
            K1 = c1.selectbox("🟢 Buy Call — Strike bas (ITM/ATM)", sa, index=max(0, aidx-1))
            K2 = c2.selectbox("🔴 Sell Call — Strike haut (OTM)", sa, index=min(len(sa)-1, aidx+1))
            iv1 = chain_calls[chain_calls['strike']==K1]['impliedVolatility'].values
            iv2 = chain_calls[chain_calls['strike']==K2]['impliedVolatility'].values
            if not (len(iv1) and len(iv2)): raise ValueError("Strike introuvable pour la strategie selectionnee.")
            p1 = calculate_greeks(spot, K1, T, r_rate, iv1[0], 'call')['Price']
            p2 = calculate_greeks(spot, K2, T, r_rate, iv2[0], 'call')['Price']
            cost = p1 - p2
            x = np.linspace(spot*0.7, spot*1.3, 120)
            y = [max(0,xi-K1) - max(0,xi-K2) - cost for xi in x]
            lbl = f"Bull Call Spread {K1:.0f}/{K2:.0f} | Debit: ${cost:.2f}"
        elif strat == "Bear Put Spread":
            c1, c2 = st.columns(2)
            K1 = c1.selectbox("🟢 Buy Put — Strike haut (ATM/ITM)", sa, index=min(len(sa)-1, aidx+1))
            K2 = c2.selectbox("🔴 Sell Put — Strike bas (OTM)", sa, index=max(0, aidx-1))
            iv1 = chain_puts[chain_puts['strike']==K1]['impliedVolatility'].values
            iv2 = chain_puts[chain_puts['strike']==K2]['impliedVolatility'].values
            if not (len(iv1) and len(iv2)): raise ValueError("Strike introuvable pour la strategie selectionnee.")
            p1 = calculate_greeks(spot, K1, T, r_rate, iv1[0], 'put')['Price']
            p2 = calculate_greeks(spot, K2, T, r_rate, iv2[0], 'put')['Price']
            cost = p1 - p2
            x = np.linspace(spot*0.7, spot*1.3, 120)
            y = [max(0,K1-xi) - max(0,K2-xi) - cost for xi in x]
            lbl = f"Bear Put Spread {K1:.0f}/{K2:.0f} | Debit: ${cost:.2f}"
        elif strat == "Long Straddle":
            K = st.selectbox("⚡ Strike ATM — Call + Put au meme prix", sa, index=aidx)
            ivc = chain_calls[chain_calls['strike']==K]['impliedVolatility'].values
            ivp = chain_puts[chain_puts['strike']==K]['impliedVolatility'].values
            if not (len(ivc) and len(ivp)): raise ValueError("Strike introuvable pour la strategie selectionnee.")
            pc_ = calculate_greeks(spot, K, T, r_rate, ivc[0], 'call')['Price']
            pp_ = calculate_greeks(spot, K, T, r_rate, ivp[0], 'put')['Price']
            cost = pc_ + pp_
            x = np.linspace(spot*0.7, spot*1.3, 120)
            y = [max(0,xi-K) + max(0,K-xi) - cost for xi in x]
            lbl = f"Long Straddle {K:.0f} | Cout total: ${cost:.2f}"
        elif strat == "Long Strangle":
            c1, c2 = st.columns(2)
            Kp = c1.selectbox("🔵 Buy Put OTM — sous le prix", sa, index=max(0, aidx-2))
            Kc = c2.selectbox("🔵 Buy Call OTM — au-dessus du prix", sa, index=min(len(sa)-1, aidx+2))
            ivc = chain_calls[chain_calls['strike']==Kc]['impliedVolatility'].values
            ivp = chain_puts[chain_puts['strike']==Kp]['impliedVolatility'].values
            if not (len(ivc) and len(ivp)): raise ValueError("Strike introuvable pour la strategie selectionnee.")
            pc_ = calculate_greeks(spot, Kc, T, r_rate, ivc[0], 'call')['Price']
            pp_ = calculate_greeks(spot, Kp, T, r_rate, ivp[0], 'put')['Price']
            cost = pc_ + pp_
            x = np.linspace(spot*0.7, spot*1.3, 120)
            y = [max(0,xi-Kc) + max(0,Kp-xi) - cost for xi in x]
            lbl = f"Long Strangle {Kp:.0f}/{Kc:.0f} | Cout total: ${cost:.2f}"
        elif strat == "Iron Condor":
            c1, c2, c3, c4 = st.columns(4)
            Kp1 = c1.selectbox("Buy Put (protection basse)",  sa, index=max(0, aidx-3))
            Kp2 = c2.selectbox("Sell Put (OTM modere bas)",   sa, index=max(0, aidx-1))
            Kc1 = c3.selectbox("Sell Call (OTM modere haut)", sa, index=min(len(sa)-1, aidx+1))
            Kc2 = c4.selectbox("Buy Call (protection haute)", sa, index=min(len(sa)-1, aidx+3))
            def giv(df, k): return df[df['strike']==k]['impliedVolatility'].values[0]
            pbp = calculate_greeks(spot, Kp1, T, r_rate, giv(chain_puts,  Kp1), 'put')['Price']
            psp = calculate_greeks(spot, Kp2, T, r_rate, giv(chain_puts,  Kp2), 'put')['Price']
            psc = calculate_greeks(spot, Kc1, T, r_rate, giv(chain_calls, Kc1), 'call')['Price']
            pbc = calculate_greeks(spot, Kc2, T, r_rate, giv(chain_calls, Kc2), 'call')['Price']
            credit = (psp-pbp) + (psc-pbc)
            x = np.linspace(spot*0.7, spot*1.3, 120)
            y = [credit - max(0,Kp2-xi) + max(0,Kp1-xi) - max(0,xi-Kc1) + max(0,xi-Kc2) for xi in x]
            lbl = f"Iron Condor {Kp1:.0f}/{Kp2:.0f}/{Kc1:.0f}/{Kc2:.0f} | Credit: ${credit:.2f}"

        st.plotly_chart(chart_payoff_strategy(x, y, spot, lbl), use_container_width=True)
        mp = max(y); ml = min(y)
        rr = abs(mp/ml) if ml != 0 else float('inf')
        be_vals = [x[i] for i in range(1,len(y)) if (y[i-1]*y[i]) < 0]
        be_str = "  |  ".join([f"${b:.2f}" for b in be_vals]) if be_vals else "N/A"
        st.markdown(f"""
        <div class="strat-kpi-row">
          <div class="strat-kpi">
            <div class="sk-label">Profit Maximum</div>
            <div class="sk-value pos">${mp:.2f}</div>
            <div style="color:#606878;font-size:0.65rem;margin-top:3px">par contrat (×100 actions)</div>
          </div>
          <div class="strat-kpi">
            <div class="sk-label">Perte Maximum</div>
            <div class="sk-value neg">${ml:.2f}</div>
            <div style="color:#606878;font-size:0.65rem;margin-top:3px">risque maximum engage</div>
          </div>
          <div class="strat-kpi">
            <div class="sk-label">Ratio R/R</div>
            <div class="sk-value" style="color:#e0e6f0">{rr:.2f}x</div>
            <div style="color:#606878;font-size:0.65rem;margin-top:3px">gain/perte potentiel</div>
          </div>
          <div class="strat-kpi">
            <div class="sk-label">Break-even(s)</div>
            <div class="sk-value" style="color:#f5a623;font-size:0.95rem">{be_str}</div>
            <div style="color:#606878;font-size:0.65rem;margin-top:3px">prix neutre a expiration</div>
          </div>
        </div>""", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Erreur strategie : {e}")

# ============================================================
# ONGLET SKEW
# ============================================================

with tab_skew:
    st.markdown('<div class="section-label">Volatility Smile / Skew — Structure de la volatilite implicite</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="signal-card" style="border-left:3px solid #00e5ff;margin-bottom:16px">'
        '<b style="color:#00e5ff">Qu\'est-ce que le Skew ?</b><br>'
        'Le skew montre si le marche paie plus cher les options d\'un cote (puts OTM vs calls OTM). '
        'Un skew penche vers les puts = demande de protection asymetrique = peur latente. '
        'Un smile symetrique = marche equilibre. Un skew inverse = speculation haussiere agressive.'
        '</div>', unsafe_allow_html=True)

    try:
        sk_calls, sk_puts, atm_iv, skew_val, rr, otm_put_iv, otm_call_iv = compute_skew_data(
            chain_calls, chain_puts, spot, T, r_rate)

        if sk_calls is not None:
            st.plotly_chart(chart_skew(sk_calls, sk_puts, spot), use_container_width=True)

            skew_sign  = skew_val if skew_val else 0
            skew_col   = "#ff4b6e" if skew_sign > 3 else ("#3fb950" if skew_sign < -3 else "#f5a623")
            skew_label = "BAISSIER (fear skew)" if skew_sign > 3 else ("HAUSSIER (greed skew)" if skew_sign < -3 else "NEUTRE")
            rr_col     = "#ff4b6e" if (rr and rr < 0) else ("#3fb950" if (rr and rr > 0) else "#8b949e")

            if skew_sign > 5:
                skew_comment = (f"Skew tres prononce ({skew_sign:.1f}pts). Le marche paie nettement plus cher "
                                f"les puts OTM que les calls OTM. Signal de peur ou d'anticipation de chute violente. "
                                f"Favorable aux vendeurs de puts ou acheteurs de call spreads.")
            elif skew_sign > 2:
                skew_comment = (f"Skew modere ({skew_sign:.1f}pts). Legere preference pour la protection baissiere. "
                                f"Normal sur la plupart des actions — les puts OTM sont structurellement plus chers. "
                                f"Surveiller une accentuation comme signal d'alarme.")
            elif skew_sign < -2:
                skew_comment = (f"Skew inverse ({skew_sign:.1f}pts). Les calls OTM coutent plus cher que les puts OTM — rare. "
                                f"Signal d'euphorie ou d'anticipation de squeeze haussier. "
                                f"Favorable aux acheteurs de puts ou vendeurs de calls.")
            else:
                skew_comment = (f"Skew quasi-symetrique ({skew_sign:.1f}pts). "
                                f"Le marche ne manifeste pas de biais directionnel fort via les options. "
                                f"Pas de signal de peur ni d'euphorie visible dans la structure de vol.")

            if rr:
                rr_comment = (f"Risk Reversal de {rr:.2f}pts. "
                              + ("Les calls 25-delta sont plus chers -> sentiment haussier." if rr > 0
                                 else "Les puts 25-delta sont plus chers -> sentiment baissier ou couverture."))
            else:
                rr_comment = "Risk Reversal indisponible."

            otm_put_str  = f"{otm_put_iv:.1f}%"  if otm_put_iv  else "N/A"
            otm_call_str = f"{otm_call_iv:.1f}%" if otm_call_iv else "N/A"
            skew_str     = f"{skew_sign:+.1f}pts" if skew_val   else "N/A"
            skew_badge   = "badge-bear" if skew_sign > 2 else ("badge-bull" if skew_sign < -2 else "badge-neut")
            rr_badge     = "badge-bull" if rr and rr > 0 else "badge-bear"
            rr_str       = f"{rr:+.2f}pts" if rr else "N/A"

            st.markdown(f"""
<div class="skew-box">
  <div class="skew-kpi-row">
    <div class="skew-kpi"><div class="sk2-label">IV ATM</div>
      <div class="sk2-value" style="color:#00e5ff">{atm_iv:.1f}%</div></div>
    <div class="skew-kpi"><div class="sk2-label">IV Put OTM</div>
      <div class="sk2-value" style="color:#ff4b6e">{otm_put_str}</div></div>
    <div class="skew-kpi"><div class="sk2-label">IV Call OTM</div>
      <div class="sk2-value" style="color:#3fb950">{otm_call_str}</div></div>
    <div class="skew-kpi"><div class="sk2-label">Skew Put-Call</div>
      <div class="sk2-value" style="color:{skew_col}">{skew_str}</div></div>
  </div>
</div>""", unsafe_allow_html=True)

            col_s1, col_s2 = st.columns(2)
            col_s1.markdown(f"""
<div class="analysis-card" style="border-left:3px solid {skew_col};margin-top:12px">
  <div class="ac-header">
    <span class="ac-title">Lecture du Skew</span>
    <span class="ac-badge {skew_badge}">{skew_label}</span>
  </div>
  <div class="ac-comment">{skew_comment}</div>
</div>""", unsafe_allow_html=True)
            col_s2.markdown(f"""
<div class="analysis-card" style="border-left:3px solid {rr_col};margin-top:12px">
  <div class="ac-header">
    <span class="ac-title">Risk Reversal 25-Delta</span>
    <span class="ac-badge {rr_badge}">{rr_str}</span>
  </div>
  <div class="ac-comment">{rr_comment}</div>
</div>""", unsafe_allow_html=True)
        else:
            st.warning("Donnees insuffisantes pour calculer le skew.")
    except Exception as e:
        st.error(f"Erreur Skew : {e}")


# ============================================================
# ONGLET PROBABILITES (PoP + EDGE BSM)
# ============================================================

with tab_pop:
    st.markdown('<div class="section-label">Probability of Profit & Edge BSM</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="signal-card" style="border-left:3px solid #b06fff;margin-bottom:16px">'
        '<b style="color:#b06fff">Comment lire ces metriques ?</b><br>'
        'La <b>PoP</b> est la probabilite statistique (modele BSM) d\'etre profitable a expiration. '
        'L\'<b>Edge BSM</b> compare la prime de marche a la valeur theorique Black-Scholes. '
        'Le signe de l\'edge indique surtout si la prime parait bon marche ou chere face au modele. '
        'Ce n\'est pas une esperance mathematique complete, mais un repere de pricing.'
        '</div>', unsafe_allow_html=True)

    try:
        pop_col1, pop_col2, pop_col3 = st.columns(3)
        with pop_col1:
            pop_o_type = st.radio("Type", ["call", "put"], horizontal=True, key="pop_type")
        with pop_col2:
            pop_data   = (chain_calls if pop_o_type == 'call' else chain_puts).dropna(
                          subset=['impliedVolatility','strike'])
            pop_strikes = sorted(pop_data['strike'].unique())
            pop_atm_idx = int(np.argmin(np.abs(np.array(pop_strikes) - spot)))
            pop_strike  = st.selectbox("Strike", pop_strikes, index=pop_atm_idx, key="pop_strike")
        with pop_col3:
            pop_dir = st.radio("Direction", ["Long", "Short"], horizontal=True, key="pop_dir")

        pop_row = pop_data[pop_data['strike'] == pop_strike]
        if not pop_row.empty:
            pop_market_price = get_option_quote_price(pop_row.iloc[0], 'mid')
            if pop_market_price is None:
                raise ValueError("Prime de marche indisponible pour ce strike.")
            pop_iv = float(pop_row['impliedVolatility'].values[0])
            pop_val, pop_price, edge_val, fair_value = compute_pop(
                spot, pop_strike, T, r_rate, pop_iv, pop_o_type, pop_dir.lower(), premium=pop_market_price)
            if pop_val is not None:
                pop_color  = "#3fb950" if pop_val >= 50 else "#ff4b6e"
                edge_color = "#3fb950" if edge_val >= 0 else "#ff4b6e"
                loss_prob  = round(100 - pop_val, 1)
                breakeven  = round((pop_strike + pop_price) if pop_o_type == 'call'
                                   else (pop_strike - pop_price), 2)
                be_dist    = round((breakeven - spot) / spot * 100, 2)

                st.plotly_chart(chart_pop_distribution(spot, pop_strike, T, r_rate,
                                                       pop_iv, pop_o_type, premium=pop_price,
                                                       direction=pop_dir.lower()),
                                use_container_width=True)

                c1, c2, c3 = st.columns(3)
                bar_color = "#3fb950" if pop_val >= 50 else "#ff4b6e"
                c1.markdown(f"""
<div class="pop-card">
  <div class="pop-label">Probability of Profit</div>
  <div class="pop-value" style="color:{pop_color}">{pop_val:.1f}%</div>
  <div class="pop-bar-bg"><div class="pop-bar-fill" style="width:{pop_val:.0f}%;background:{bar_color}"></div></div>
  <div class="pop-sub">Perte : {loss_prob:.1f}%</div>
</div>""", unsafe_allow_html=True)

                c2.markdown(f"""
<div class="pop-card">
  <div class="pop-label">Break-even</div>
  <div class="pop-value" style="color:#e0e6f0;font-size:1.5rem">${breakeven:.2f}</div>
  <div class="pop-sub">{be_dist:+.2f}% du spot</div>
  <div class="pop-sub" style="margin-top:6px">Prime marche : ${pop_price:.2f}</div>
</div>""", unsafe_allow_html=True)

                c3.markdown(f"""
<div class="pop-card">
  <div class="pop-label">Edge BSM</div>
  <div class="pop-value" style="color:{edge_color};font-size:1.5rem">${edge_val:.2f}</div>
  <div class="pop-sub">{'Prime sous la juste valeur' if edge_val >= 0 else 'Prime au-dessus du modele'}</div>
  <div class="pop-sub" style="margin-top:6px">Fair value : ${fair_value:.2f}</div>
</div>""", unsafe_allow_html=True)

                if pop_val >= 65:
                    pop_comment = (f"PoP de {pop_val:.1f}% — position statistiquement favorable. "
                                   f"Plus de 2 chances sur 3 d'etre profitable a expiration selon BSM. "
                                   f"Attention : une PoP elevee sur un long correspond souvent a un ITM ou ATM couteux.")
                elif pop_val >= 50:
                    pop_comment = (f"PoP de {pop_val:.1f}% — legerement en faveur. "
                                   f"Juste au-dessus du pile ou face. Le trade a besoin d'un mouvement modere.")
                elif pop_val >= 35:
                    pop_comment = (f"PoP de {pop_val:.1f}% — position defavorable statistiquement. "
                                   f"Typique d'un achat d'option OTM speculatif. Necessite un mouvement fort. "
                                   f"A prendre uniquement si la conviction directionnelle est forte.")
                else:
                    pop_comment = (f"PoP de {pop_val:.1f}% — tres faible probabilite de profit. "
                                   f"Option tres OTM ou echeance courte. Perte probable mais gain potentiel eleve.")

                edge_comment = (f"Edge BSM de {'+' if edge_val >= 0 else ''}{edge_val:.2f}$ par contrat. "
                                + ("La prime de marche est inferieure a la valeur theorique du modele. "
                                   "Cela favorise plutot l'acheteur si vous faites confiance aux hypotheses BSM."
                                   if edge_val >= 0 else
                                   "La prime de marche est superieure a la valeur theorique du modele. "
                                   "Cela favorise plutot le vendeur, ou indique que le modele sous-estime le risque."))

                st.markdown(f"""
<div class="ev-box" style="margin-top:4px">
  <div class="ev-label">Analyse PoP & Edge BSM</div>
  <div class="ev-breakdown">{pop_comment}<br><br>{edge_comment}</div>
</div>""", unsafe_allow_html=True)

                st.markdown('<div class="section-label" style="margin-top:24px">PoP sur tous les strikes</div>',
                            unsafe_allow_html=True)
                pop_table_data = []
                for s in pop_strikes:
                    r_s = pop_data[pop_data['strike'] == s]
                    if r_s.empty: continue
                    iv_s = float(r_s['impliedVolatility'].values[0])
                    market_price_s = get_option_quote_price(r_s.iloc[0], 'mid')
                    if market_price_s is None:
                        continue
                    p, pr, edge_s, _ = compute_pop(spot, s, T, r_rate, iv_s, pop_o_type,
                                                   pop_dir.lower(), premium=market_price_s)
                    if p is not None:
                        pop_table_data.append({
                            'Strike': f"${s:.0f}",
                            'Moneyness': f"{(s-spot)/spot*100:+.1f}%",
                            'Prime': f"${pr:.2f}",
                            'IV': f"{iv_s*100:.1f}%",
                            'PoP': f"{p:.1f}%",
                            'Edge BSM ($)': f"${edge_s:.2f}",
                        })
                if pop_table_data:
                    st.dataframe(pd.DataFrame(pop_table_data), use_container_width=True, hide_index=True)
        else:
            st.warning("Donnees introuvables pour ce strike.")
    except Exception as e:
        st.error(f"Erreur Probabilites : {e}")


# ============================================================
# ONGLET OPTIONS FLOW
# ============================================================

with tab_flow:
    st.markdown('<div class="section-label">Options Flow — Detection des trades anormaux</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="signal-card" style="border-left:3px solid #f5a623;margin-bottom:16px">'
        '<b style="color:#f5a623">Comment lire le flux options ?</b><br>'
        'Un volume anormalement eleve par rapport a l\'OI (ratio Vol/OI eleve) signale une nouvelle position. '
        'Les gros trades sur des options OTM sont souvent des paris directionnels forts. '
        'Les feux indiquent le niveau d\'anomalie — plus il y en a, plus le trade sort de l\'ordinaire.'
        '</div>', unsafe_allow_html=True)

    try:
        flow_df = compute_options_flow(chain_calls, chain_puts, spot, T, r_rate)

        if not flow_df.empty:
            n_calls     = int((flow_df['type'] == 'CALL').sum())
            n_puts      = int((flow_df['type'] == 'PUT').sum())
            total_notio = float(flow_df['notional'].sum())
            top_row     = flow_df.iloc[0]

            flow_bias     = "BULLISH" if n_calls > n_puts * 1.3 else ("BEARISH" if n_puts > n_calls * 1.3 else "MIXTE")
            flow_bias_col = "#3fb950" if flow_bias == "BULLISH" else ("#ff4b6e" if flow_bias == "BEARISH" else "#f5a623")

            top_type   = top_row['type']
            top_strike = top_row['strike']
            top_heat   = top_row['heat']
            top_vol    = int(top_row['volume'])
            top_oi     = int(top_row['openInterest'])

            st.markdown(f"""
<div class="flow-summary-row">
  <div class="flow-summary-card" style="border-left:3px solid {flow_bias_col}">
    <div class="fsc-label">Biais du Flux</div>
    <div class="fsc-value" style="color:{flow_bias_col}">{flow_bias}</div>
    <div style="color:#8b949e;font-size:0.72rem;margin-top:4px">{n_calls} signaux calls · {n_puts} puts</div>
  </div>
  <div class="flow-summary-card">
    <div class="fsc-label">Notionnel Total</div>
    <div class="fsc-value" style="color:#00e5ff">${total_notio/1e6:.2f}M</div>
    <div style="color:#8b949e;font-size:0.72rem;margin-top:4px">Top {len(flow_df)} trades anormaux</div>
  </div>
  <div class="flow-summary-card">
    <div class="fsc-label">Plus Gros Signal</div>
    <div class="fsc-value" style="color:#f5a623">{top_heat} {top_type} ${top_strike:.0f}</div>
    <div style="color:#8b949e;font-size:0.72rem;margin-top:4px">Vol {top_vol:,} · OI {top_oi:,}</div>
  </div>
</div>""", unsafe_allow_html=True)

            # Tableau
            rows_html = ""
            for _, row in flow_df.iterrows():
                badge    = 'flow-call' if row['type'] == 'CALL' else 'flow-put'
                vol_oi_s = f"{row['vol_oi_ratio']:.1f}x" if pd.notna(row['vol_oi_ratio']) else "N/A"
                notio_s  = f"${row['notional']/1e3:.0f}K" if row['notional'] < 1e6 else f"${row['notional']/1e6:.2f}M"
                m_col    = "#3fb950" if row['moneyness'] > 0 else "#ff4b6e"
                rows_html += (
                    f"<tr>"
                    f"<td><span class='flow-badge {badge}'>{row['type']}</span></td>"
                    f"<td>${row['strike']:.0f}</td>"
                    f"<td style='color:{m_col}'>{row['moneyness']:+.1f}%</td>"
                    f"<td style='font-family:JetBrains Mono,monospace'>{int(row['volume']):,}</td>"
                    f"<td style='font-family:JetBrains Mono,monospace;color:#8b949e'>{int(row['openInterest']):,}</td>"
                    f"<td style='color:#f5a623'>{vol_oi_s}</td>"
                    f"<td>{notio_s}</td>"
                    f"<td style='color:#b06fff'>{row['iv_pct']:.1f}%</td>"
                    f"<td style='font-size:1rem'>{row['heat']}</td>"
                    f"<td style='color:#c9d1d9;font-size:0.78rem'>{row['interpretation']}</td>"
                    f"</tr>"
                )

            st.markdown(f"""
<table class="flow-table">
  <thead><tr>
    <th>Type</th><th>Strike</th><th>Moneyness</th>
    <th>Volume</th><th>OI</th><th>Vol/OI</th>
    <th>Notionnel</th><th>IV</th><th>Signal</th><th>Interpretation</th>
  </tr></thead>
  <tbody>{rows_html}</tbody>
</table>""", unsafe_allow_html=True)

            # Narrative
            st.markdown('<div class="section-label" style="margin-top:20px">Lecture du flux</div>',
                        unsafe_allow_html=True)

            if flow_bias == "BULLISH":
                narrative = (f"Le flux options est domine par les <b style='color:#00e5ff'>CALLS</b> "
                             f"({n_calls} signaux vs {n_puts} puts). "
                             f"Les participants achetenent principalement des calls, souvent OTM — signe d'anticipation "
                             f"haussiere ou de couverture de shorts. Notionnel total : ${total_notio/1e6:.2f}M.")
            elif flow_bias == "BEARISH":
                narrative = (f"Le flux est domine par les <b style='color:#ff4b6e'>PUTS</b> "
                             f"({n_puts} signaux vs {n_calls} calls). "
                             f"Forte demande de protection ou de paris baissiers. "
                             f"Peut signaler une anticipation de correction ou une couverture de longs. "
                             f"Notionnel : ${total_notio/1e6:.2f}M.")
            else:
                narrative = (f"Flux mixte — {n_calls} signaux calls et {n_puts} puts. "
                             f"Pas de biais directionnel clair. Le marche est partage ou des strategies "
                             f"non-directionnelles (straddles, condors) sont en jeu. "
                             f"Notionnel total : ${total_notio/1e6:.2f}M.")

            top_call_rows = flow_df[flow_df['type']=='CALL']
            top_put_rows  = flow_df[flow_df['type']=='PUT']
            if not top_call_rows.empty:
                tc = top_call_rows.iloc[0]
                narrative += (f"<br><br>Trade call le plus anormal : <b>${tc['strike']:.0f}</b> "
                              f"({tc['moneyness']:+.1f}%) — Volume {int(tc['volume']):,} "
                              f"· OI {int(tc['openInterest']):,} · ratio {tc['vol_oi_ratio']:.1f}x {tc['heat']}")
            if not top_put_rows.empty:
                tp = top_put_rows.iloc[0]
                narrative += (f"<br>Trade put le plus anormal : <b>${tp['strike']:.0f}</b> "
                              f"({tp['moneyness']:+.1f}%) — Volume {int(tp['volume']):,} "
                              f"· OI {int(tp['openInterest']):,} · ratio {tp['vol_oi_ratio']:.1f}x {tp['heat']}")

            st.markdown(f"""
<div class="signal-card" style="border-left:3px solid {flow_bias_col};margin-top:8px">
  {narrative}
</div>""", unsafe_allow_html=True)
        else:
            st.warning("Donnees de flux insuffisantes pour cette echeance.")
    except Exception as e:
        st.error(f"Erreur Options Flow : {e}")



# ============================================================
# ONGLET ROLL ANALYZER
# ============================================================

with tab_roll:
    st.markdown('<div class="section-label">Roll Analyzer — Evaluer le cout de roulement d\'une position</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="signal-card" style="margin-bottom:20px;border-left:3px solid #b06fff">
        <b style="color:#b06fff">A quoi sert le Roll ?</b><br>
        Roller une option = fermer la position actuelle et en ouvrir une nouvelle sur une echeance plus
        lointaine (ou un strike different). Utile pour eviter l'expiration, prolonger une these,
        ou collecter plus de theta. L'outil calcule le cout/credit net du roll et compare les Greeks.
    </div>""", unsafe_allow_html=True)

    if len(expiry_dates_all) < 2:
        st.warning("Moins de 2 echeances disponibles pour effectuer un roll.")
    else:
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            roll_current = st.selectbox("Echeance actuelle", expiry_dates_all,
                                         index=0, key="roll_curr")
        with rc2:
            future_expiries = [e for e in expiry_dates_all if e > roll_current]
            if not future_expiries:
                st.warning("Aucune echeance future disponible.")
                st.stop()
            roll_new = st.selectbox("Roller vers", future_expiries,
                                     index=0, key="roll_new")
        with rc3:
            roll_type = st.radio("Type d'option", ["call", "put"],
                                  horizontal=True, key="roll_type")
            roll_side = st.radio("Position", ["Long", "Short"],
                                 horizontal=True, key="roll_side")

        # Charger les strikes disponibles pour l'échéance actuelle
        try:
            roll_curr_calls, roll_curr_puts = get_option_chain(ticker, roll_current)
            roll_df = roll_curr_calls if roll_type == 'call' else roll_curr_puts
            roll_df = roll_df.dropna(subset=['impliedVolatility','strike'])
            roll_strikes = sorted(roll_df['strike'].unique())
            roll_atm_idx = int(np.argmin(np.abs(np.array(roll_strikes)-spot)))

            roll_strike = st.selectbox("Strike a roller", roll_strikes,
                                        index=roll_atm_idx, key="roll_strike")

            if st.button("Calculer le Roll", type="primary"):
                with st.spinner("Calcul en cours..."):
                    result = compute_roll(ticker, roll_current, roll_new,
                                          roll_strike, roll_type, spot, r_rate,
                                          position_side=roll_side.lower())

                if result is None:
                    st.error("Impossible de calculer le roll — donnees manquantes pour ce strike/echeance.")
                else:
                    rc_sign = "pos" if result['roll_cost'] <= 0 else "neg"
                    rc_label = "CREDIT (vous recevez)" if result['roll_cost'] <= 0 else "DEBIT (vous payez)"
                    rc_color = "#3fb950" if result['roll_cost'] <= 0 else "#ff4b6e"

                    # Verdict automatique
                    if result['roll_cost'] <= 0:
                        verdict = (f"Roll en CREDIT de ${abs(result['roll_cost']):.2f} — vous recevez de l'argent "
                                   f"pour prolonger la position de {result['days_gained']} jours supplementaires. "
                                   f"Generalement favorable si la these est toujours valide.")
                    else:
                        daily_cost = result['roll_cost'] / max(1, result['days_gained'])
                        verdict = (f"Roll en DEBIT de ${result['roll_cost']:.2f} — vous payez "
                                   f"${daily_cost:.3f}/jour pour {result['days_gained']} jours supplementaires. "
                                   f"Justifie si la position a besoin de temps pour se developper.")

                    theta_gain = (result['new_theta'] - result['curr_theta']) * result['days_gained']
                    theta_color = "#3fb950" if theta_gain >= 0 else "#ff4b6e"
                    position_label = "LONG" if result.get('position_side') == 'long' else "SHORT"
                    rc_label = f"{position_label} · {rc_label}"

                    st.markdown(f"""
                    <div class="roll-box">
                        <div class="roll-title">Resultat du Roll — {roll_type.upper()} ${roll_strike:.0f}</div>
                        <div style="text-align:center;margin-bottom:20px">
                            <div style="color:#606878;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em">Cout / Credit du Roll</div>
                            <div class="roll-cost-{'neg' if result['roll_cost']>0 else 'pos'}" style="font-size:2rem;margin-top:6px">
                                {'−' if result['roll_cost']>0 else '+'}${abs(result['roll_cost']):.2f}
                            </div>
                            <div style="color:{rc_color};font-size:0.78rem;margin-top:4px">{rc_label}</div>
                        </div>
                        <table class="roll-table">
                            <tr>
                                <th></th>
                                <th style="color:#8b949e">Echeance actuelle<br><span style="color:#606878;font-size:0.65rem">{roll_current}</span></th>
                                <th style="color:#b06fff">Nouvelle echeance<br><span style="color:#606878;font-size:0.65rem">{roll_new}</span></th>
                            </tr>
                            <tr>
                                <td style="color:#606878;text-align:left">Prix d'execution</td>
                                <td>${result['curr_price']:.2f}</td>
                                <td class="highlight">${result['new_price']:.2f}</td>
                            </tr>
                            <tr>
                                <td style="color:#606878;text-align:left">IV implicite</td>
                                <td>{result['curr_iv']:.1f}%</td>
                                <td class="highlight">{result['new_iv']:.1f}%
                                    <span style="color:{'#ff4b6e' if result['iv_change']>0 else '#3fb950'};font-size:0.75rem">
                                    ({'+' if result['iv_change']>0 else ''}{result['iv_change']:.1f}%)</span>
                                </td>
                            </tr>
                            <tr>
                                <td style="color:#606878;text-align:left">Delta position</td>
                                <td>{result['curr_delta']:.3f}</td>
                                <td class="highlight">{result['new_delta']:.3f}</td>
                            </tr>
                            <tr>
                                <td style="color:#606878;text-align:left">Theta position /jour</td>
                                <td>{result['curr_theta']:.4f}</td>
                                <td class="highlight">{result['new_theta']:.4f}</td>
                            </tr>
                            <tr>
                                <td style="color:#606878;text-align:left">Jours gagnes</td>
                                <td>—</td>
                                <td class="highlight" style="color:#00e5ff">+{result['days_gained']} jours</td>
                            </tr>
                        </table>
                        <div style="margin-top:14px;display:flex;gap:10px">
                            <div style="flex:1;background:rgba(0,0,0,0.2);border-radius:8px;padding:10px;text-align:center">
                                <div style="color:#606878;font-size:0.65rem;text-transform:uppercase">Gain Theta total</div>
                                <div style="color:{theta_color};font-size:1rem;font-family:JetBrains Mono,monospace;margin-top:4px">
                                    ${theta_gain:.2f}
                                </div>
                            </div>
                            <div style="flex:1;background:rgba(0,0,0,0.2);border-radius:8px;padding:10px;text-align:center">
                                <div style="color:#606878;font-size:0.65rem;text-transform:uppercase">Variation IV</div>
                                <div style="color:{'#ff4b6e' if result['iv_change']>0 else '#3fb950'};font-size:1rem;font-family:JetBrains Mono,monospace;margin-top:4px">
                                    {'+' if result['iv_change']>0 else ''}{result['iv_change']:.1f}%
                                </div>
                            </div>
                        </div>
                        <div class="roll-verdict">
                            <b style="color:#b06fff">Verdict :</b> {verdict}
                        </div>
                    </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Erreur Roll Analyzer : {e}")
