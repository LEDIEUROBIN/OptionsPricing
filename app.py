import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# BLACK-SCHOLES ENGINE
# ============================================================

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {k: 0.0 for k in ["Price","Delta","Gamma","Theta","Vega","Vanna","Charm","Volga"]}
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    if option_type == 'call':
        price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (-(S*pdf_d1*sigma)/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2)) / 365
    else:
        price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        theta = (-(S*pdf_d1*sigma)/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2)) / 365
    gamma = pdf_d1 / (S*sigma*np.sqrt(T))
    vega  = S*pdf_d1*np.sqrt(T) / 100
    vanna = -pdf_d1*d2 / sigma
    charm = -pdf_d1*(2*r*T - d2*sigma*np.sqrt(T)) / (2*T*sigma*np.sqrt(T)) / 365
    volga = vega*d1*d2 / sigma
    return {"Price": max(0,price), "Delta": delta, "Gamma": gamma,
            "Theta": theta, "Vega": vega, "Vanna": vanna, "Charm": charm, "Volga": volga}

# ============================================================
# DATA FETCHING
# ============================================================

@st.cache_data(ttl=300)
def get_market_context():
    try:
        tnx = yf.Ticker("^TNX").history(period="1d")['Close'].iloc[-1] / 100
        vix = yf.Ticker("^VIX").history(period="1d")['Close'].iloc[-1]
        return float(tnx), float(vix)
    except:
        return 0.0425, 20.0

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

@st.cache_data(ttl=3600)
def get_earnings_date(ticker: str):
    """Récupère la prochaine date d'annonce de résultats."""
    try:
        stock = yf.Ticker(ticker)
        cal = stock.calendar
        if cal is not None and not cal.empty:
            # calendar peut être un DataFrame ou dict selon la version yfinance
            if isinstance(cal, pd.DataFrame):
                if 'Earnings Date' in cal.index:
                    val = cal.loc['Earnings Date'].values
                    dates = [pd.to_datetime(v) for v in val if pd.notna(v)]
                    future = [d for d in dates if d > pd.Timestamp.now(tz=d.tzinfo)]
                    return future[0].date() if future else None
            elif isinstance(cal, dict):
                earn = cal.get('Earnings Date', [])
                if earn:
                    dates = [pd.to_datetime(d) for d in earn if pd.notna(d)]
                    future = [d for d in dates if d.date() > datetime.now().date()]
                    return future[0].date() if future else None
        # fallback: earnings_dates
        ed = stock.earnings_dates
        if ed is not None and not ed.empty:
            future = ed[ed.index.normalize() > pd.Timestamp.now().normalize()]
            if not future.empty:
                return future.index[0].date()
    except:
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
    if hv_series.empty:
        return None, None, None
    iv_min = float(hv_series.min())
    iv_max = float(hv_series.max())
    iv_rank = (current_iv_pct - iv_min) / (iv_max - iv_min) * 100 if iv_max > iv_min else 50.0
    iv_pct  = float((hv_series < current_iv_pct).mean() * 100)
    return round(iv_rank, 1), round(iv_pct, 1), round(iv_min, 1), round(iv_max, 1)

def compute_implied_move(calls_df, puts_df, spot):
    """
    Implied Move = (ATM Call + ATM Put) × 0.85
    Retourne le move en $ et en %.
    """
    try:
        calls_c = calls_df.dropna(subset=['lastPrice','strike']).copy()
        puts_c  = puts_df.dropna(subset=['lastPrice','strike']).copy()
        atm_call_row = calls_c.iloc[(calls_c['strike']-spot).abs().argsort()[:1]]
        atm_put_row  = puts_c.iloc[(puts_c['strike']-spot).abs().argsort()[:1]]
        atm_call_price = float(atm_call_row['lastPrice'].values[0])
        atm_put_price  = float(atm_put_row['lastPrice'].values[0])
        straddle = atm_call_price + atm_put_price
        move_usd = straddle * 0.85
        move_pct = move_usd / spot * 100
        return round(move_usd, 2), round(move_pct, 2), round(straddle, 2)
    except:
        return None, None, None

# ============================================================
# ROLL ANALYZER
# ============================================================

def compute_roll(ticker, current_expiry, new_expiry, strike, option_type, spot, r_rate):
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

        curr_bid = float(curr_row['bid'].values[0]) if 'bid' in curr_row.columns else float(curr_row['lastPrice'].values[0])
        new_ask  = float(new_row['ask'].values[0])  if 'ask' in new_row.columns  else float(new_row['lastPrice'].values[0])
        curr_iv  = float(curr_row['impliedVolatility'].values[0])
        new_iv   = float(new_row['impliedVolatility'].values[0])

        T_curr = max(1/365, (datetime.strptime(current_expiry, "%Y-%m-%d") - datetime.now()).days / 365.0)
        T_new  = max(1/365, (datetime.strptime(new_expiry,     "%Y-%m-%d") - datetime.now()).days / 365.0)

        curr_theta = calculate_greeks(spot, strike, T_curr, r_rate, curr_iv, option_type)['Theta']
        new_theta  = calculate_greeks(spot, strike, T_new,  r_rate, new_iv,  option_type)['Theta']
        curr_delta = calculate_greeks(spot, strike, T_curr, r_rate, curr_iv, option_type)['Delta']
        new_delta  = calculate_greeks(spot, strike, T_new,  r_rate, new_iv,  option_type)['Delta']

        roll_cost   = new_ask - curr_bid      # positif = on paie, négatif = on reçoit
        days_gained = (datetime.strptime(new_expiry, "%Y-%m-%d") - datetime.strptime(current_expiry, "%Y-%m-%d")).days

        return {
            'curr_price': curr_bid,
            'new_price':  new_ask,
            'roll_cost':  roll_cost,
            'days_gained': days_gained,
            'curr_iv':    round(curr_iv*100, 1),
            'new_iv':     round(new_iv*100, 1),
            'curr_theta': curr_theta,
            'new_theta':  new_theta,
            'curr_delta': curr_delta,
            'new_delta':  new_delta,
            'iv_change':  round((new_iv - curr_iv)*100, 2),
        }
    except Exception as e:
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
    k_grid = np.linspace(0.80, 1.20, 25)
    expiries_label, iv_rows = [], []
    for exp in expiry_dates[:10]:
        try:
            calls, puts = get_option_chain(ticker, exp)
            data = (calls if option_type == 'call' else puts).dropna(subset=['impliedVolatility']).copy()
            data = data[(data['strike'] > spot*0.72) & (data['strike'] < spot*1.28)]
            if len(data) < 4: continue
            ks  = data['strike'].values / spot
            ivs = data['impliedVolatility'].values * 100
            order = np.argsort(ks)
            ks, ivs = ks[order], ivs[order]
            row = np.interp(k_grid, ks, ivs)
            expiries_label.append(exp)
            iv_rows.append(row)
        except:
            continue
    if len(iv_rows) < 2:
        return go.Figure()
    z_arr = np.array(iv_rows)
    scene_axis = dict(
        backgroundcolor='rgba(10,14,22,0.0)',
        gridcolor='rgba(255,255,255,0.10)',
        zerolinecolor='rgba(255,255,255,0.15)',
        tickfont=dict(color='#c9d1d9', size=10),
        title_font=dict(color='#e0e6f0', size=12),
    )
    fig = go.Figure(data=[go.Surface(
        z=z_arr, x=k_grid, y=list(range(len(expiries_label))),
        colorscale=[[0.0,'#0a0f1e'],[0.25,'#1a3a6e'],[0.5,'#0080ff'],
                    [0.75,'#00d4ff'],[1.0,'#ffffff']],
        opacity=1.0,
        lighting=dict(ambient=0.7, diffuse=0.8, roughness=0.5, specular=0.3),
        colorbar=dict(
            title=dict(text='IV (%)', font=dict(color='#e0e6f0', size=13)),
            tickfont=dict(color='#c9d1d9', size=11), thickness=14, len=0.7)
    )])
    fig.update_layout(
        scene=dict(
            xaxis=dict(**scene_axis, title=dict(text='Strike/Spot', font=dict(color='#e0e6f0', size=12))),
            yaxis=dict(**scene_axis, title=dict(text='Echeance', font=dict(color='#e0e6f0', size=12)),
                       tickvals=list(range(len(expiries_label))), ticktext=expiries_label),
            zaxis=dict(**scene_axis, title=dict(text='IV (%)', font=dict(color='#e0e6f0', size=12))),
            bgcolor='rgba(10,14,22,0.95)', aspectmode='manual',
            aspectratio=dict(x=1.6, y=1.2, z=0.7),
        ),
        paper_bgcolor='rgba(0,0,0,0)', template='plotly_dark',
        height=520, margin=dict(l=0, r=0, t=10, b=10), font=dict(color='#e0e6f0'),
    )
    return fig


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


def chart_pnl_multiscenario(spot, sel_strike, investi, nb_contrats, T, r_rate, iv, option_type):
    x_range  = np.linspace(sel_strike*0.70, sel_strike*1.30, 100)
    horizons = [("A expiration", 0, '#00e5ff'), ("T-15j", 15/365, '#f5a623'), ("T-30j", 30/365, '#b06fff')]
    fig = go.Figure()
    for label, dt, color in horizons:
        t_rem = max(1e-6, T - dt)
        if dt == 0:
            y = [(max(0, x-sel_strike if option_type=='call' else sel_strike-x)*100*nb_contrats) - investi for x in x_range]
        else:
            y = [(calculate_greeks(x, sel_strike, t_rem, r_rate, iv, option_type)['Price']*100*nb_contrats) - investi for x in x_range]
        fig.add_trace(go.Scatter(x=x_range, y=y, name=label, line=dict(color=color, width=2.5)))
    fig.add_hline(y=0, line_dash='dash', line_color='rgba(255,75,110,0.6)', line_width=1.5)
    fig.add_vline(x=spot, line_dash='dot', line_color='rgba(255,255,255,0.4)', line_width=1.5,
                  annotation=dict(text='Spot', font=dict(color='#ffffff', size=11),
                                  bgcolor='rgba(0,0,0,0.6)', bordercolor='rgba(255,255,255,0.2)', borderwidth=1))
    fig.update_layout(**LAYOUT, height=310,
                      xaxis={**AXIS, 'title_text': 'Prix sous-jacent ($)'},
                      yaxis={**AXIS, 'title_text': 'P/L ($)'})
    return fig


def chart_open_interest(calls_raw, puts_raw, spot):
    calls = calls_raw[['strike','openInterest']].copy()
    puts  = puts_raw[['strike','openInterest']].copy()
    calls['gamma'] = calls_raw['gamma'].fillna(0) if 'gamma' in calls_raw.columns else 0.0
    puts['gamma']  = puts_raw['gamma'].fillna(0)  if 'gamma' in puts_raw.columns  else 0.0
    calls = calls[(calls['strike'] > spot*0.7) & (calls['strike'] < spot*1.3)]
    puts  = puts[(puts['strike']  > spot*0.7) & (puts['strike']  < spot*1.3)]
    calls['GEX'] =  calls['gamma'].fillna(0)*calls['openInterest'].fillna(0)*100*spot
    puts['GEX']  = -puts['gamma'].fillna(0)*puts['openInterest'].fillna(0)*100*spot
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
    background:#0d1320 !important;
    border:1px solid rgba(255,255,255,0.1) !important;
    border-radius:8px !important; color:#e0e6f0 !important;
}
.stSelectbox label, .stNumberInput label,
.stTextInput label, .stRadio label { color:#8b949e !important; font-size:0.8rem !important; }

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
    border:1px solid rgba(0,229,255,0.15); border-radius:12px; padding:14px; text-align:center;
}
.vix-label { color:#606878; font-size:0.68rem; letter-spacing:0.12em; text-transform:uppercase; }
.vix-value { color:#00e5ff; font-size:2rem; font-weight:700; font-family:'JetBrains Mono'; }

.strat-kpi-row { display:flex; gap:10px; margin-top:14px; }
.strat-kpi {
    flex:1; background:#0d1320; border:1px solid rgba(255,255,255,0.07);
    border-radius:10px; padding:14px; text-align:center;
}
.strat-kpi .sk-label { color:#606878; font-size:0.68rem; text-transform:uppercase; letter-spacing:0.1em; }
.strat-kpi .sk-value { font-size:1.25rem; font-weight:700; font-family:'JetBrains Mono'; margin-top:4px; }
</style>
"""

# ============================================================
# APP MAIN
# ============================================================

current_10y, current_vix = get_market_context()

st.set_page_config(layout="wide", page_title="Quantum Options Terminal", page_icon="💎")
st.markdown(CSS, unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Parametres")
    ticker = st.text_input("Symbole", value="AAPL",
                           placeholder="Ex: AAPL, TSLA, SPY").upper().strip()
    r_rate = st.number_input("Taux US 10Y", value=float(current_10y), format="%.4f", step=0.001)
    st.divider()
    st.markdown(f'''
    <div class="vix-widget">
        <div class="vix-label">Indice VIX</div>
        <div class="vix-value">{current_vix:.2f}</div>
        <div style="color:#606878;font-size:0.7rem;margin-top:2px;">10Y · {current_10y*100:.2f}%</div>
    </div>''', unsafe_allow_html=True)
    st.divider()
    st.markdown("**Surface de volatilite**")
    vol_surface_type = st.radio("Type", ['call', 'put'], horizontal=True,
                                 label_visibility="collapsed")
    st.divider()
    st.caption("Yahoo Finance · delai ~15 min")

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
st.markdown('<div class="section-label">Surface de Volatilite Implicite</div>', unsafe_allow_html=True)
if expiry_dates_all and len(expiry_dates_all) >= 2:
    with st.spinner("Calcul de la surface..."):
        fig_surf = chart_vol_surface(ticker, expiry_dates_all, spot, r_rate, vol_surface_type)
    if fig_surf.data:
        st.plotly_chart(fig_surf, use_container_width=True)
        st.caption("Cliquez-glissez pour pivoter · Molette pour zoomer · Double-clic pour reinitialiser")
    else:
        st.warning("Donnees insuffisantes pour construire la surface.")
else:
    st.warning("Moins de 2 echeances disponibles.")

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
            <div style="color:#8b949e;font-size:0.85rem;margin-top:6px">Date non disponible pour ce ticker.</div>
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
                    <div class="mr-val up">+${price_up:.2f}</div>
                </div>
                <div class="mr-item">
                    <div class="mr-label">Spot actuel</div>
                    <div class="mr-val">${spot:.2f}</div>
                </div>
                <div class="mr-item">
                    <div class="mr-label">Borne basse</div>
                    <div class="mr-val dn">-${price_dn:.2f}</div>
                </div>
            </div>
            <div class="move-commentary">
                {earn_html}<b>Signal :</b> {move_comment}
            </div>
        </div>'''
        st.markdown(move_html, unsafe_allow_html=True)
    else:
        st.warning("Implied Move indisponible.")

# ============================================================
# ONGLETS PRINCIPAUX
# ============================================================

tab_c, tab_p, tab_oi, tab_strat, tab_roll = st.tabs(
    ["CALLS", "PUTS", "OI & GEX", "STRATEGIES", "ROLL ANALYZER"])

for tab, data_raw, o_type in zip([tab_c, tab_p], [chain_calls, chain_puts], ['call','put']):
    with tab:
        data = data_raw.dropna(subset=['impliedVolatility','lastPrice']).copy()
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

        price_buy  = row_s['lastPrice'].values[0]
        iv_sel     = row_s['impliedVolatility'].values[0]
        sign       = 1 if "Long" in direction else -1
        investi    = nb_contrats * price_buy * 100
        val_finale = nb_contrats * max(0, (target_p-sel_strike) if o_type=='call' else (sel_strike-target_p)) * 100
        pnl        = sign * (val_finale - investi)
        roi        = pnl/investi*100 if investi > 0 else 0
        breakeven  = (sel_strike+price_buy) if o_type=='call' else (sel_strike-price_buy)
        pc         = "pos" if pnl >= 0 else "neg"

        with c_res:
            st.markdown(f"""
            <table class="pnl-table">
              <tr><td class="lbl">Prime par contrat</td><td class="val">${price_buy:.2f}</td></tr>
              <tr><td class="lbl">Capital engage</td><td class="val">${investi:,.2f}</td></tr>
              <tr><td class="lbl">Valeur a expiration</td><td class="val">${val_finale:,.2f}</td></tr>
              <tr><td class="lbl">Point mort</td><td class="val">${breakeven:.2f}</td></tr>
              <tr class="total"><td>P/L Net</td><td class="val {pc}">${pnl:,.2f}</td></tr>
              <tr><td class="lbl">ROI</td><td class="val {pc}">{roi:.2f}%</td></tr>
            </table>""", unsafe_allow_html=True)

        st.plotly_chart(
            chart_pnl_multiscenario(spot, sel_strike, investi, nb_contrats, T, r_rate, iv_sel, o_type),
            use_container_width=True)

# ── OI & GEX ─────────────────────────────────────────────────
with tab_oi:
    st.markdown('<div class="section-label">Open Interest et Gamma Exposure</div>', unsafe_allow_html=True)
    try:
        fig_oi, oi_c, oi_p = chart_open_interest(chain_calls, chain_puts, spot)
        st.plotly_chart(fig_oi, use_container_width=True)
        total_c = chain_calls['openInterest'].sum()
        total_p = chain_puts['openInterest'].sum()
        pcr     = total_p/total_c if total_c > 0 else 0
        pcr_col = "#ff4b6e" if pcr > 1.2 else ("#3fb950" if pcr < 0.8 else "#f5a623")
        pcr_txt = ("Sentiment baissier (exces de puts)" if pcr > 1.2
                   else ("Sentiment haussier (exces de calls)" if pcr < 0.8 else "Sentiment neutre"))
        col1, col2 = st.columns(2)
        col1.metric("Put / Call Ratio", f"{pcr:.3f}",
                    delta=f"Calls {total_c:,}  Puts {total_p:,}", delta_color="off")
        col2.markdown(f'''
        <div class="signal-card" style="border-left:3px solid {pcr_col}">
            <b style="color:{pcr_col}">Signal P/C Ratio</b> &middot; {pcr_txt}
        </div>''', unsafe_allow_html=True)
        gex_df   = pd.concat([oi_c[['strike','GEX']], oi_p[['strike','GEX']]]).groupby('strike')['GEX'].sum().reset_index()
        dominant = gex_df.loc[gex_df['GEX'].abs().idxmax(), 'strike']
        st.markdown(f'''
        <div class="signal-card" style="border-left:3px solid #b06fff; margin-top:8px">
            <b style="color:#b06fff">Strike GEX dominant</b> &middot; ${dominant:.0f}
            — Les market makers tendent a defendre ce niveau.
        </div>''', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Erreur OI/GEX : {e}")

# ── Strategies ────────────────────────────────────────────────
with tab_strat:
    st.markdown('<div class="section-label">Payoff de Strategies Combinees</div>', unsafe_allow_html=True)
    strat = st.selectbox("Strategie", [
        "Bull Call Spread", "Bear Put Spread", "Long Straddle", "Long Strangle", "Iron Condor"])
    sa   = sorted(chain_calls['strike'].unique())
    aidx = int(np.argmin(np.abs(np.array(sa)-spot)))
    try:
        if strat == "Bull Call Spread":
            c1, c2 = st.columns(2)
            K1 = c1.selectbox("Buy Call (bas)",   sa, index=max(0, aidx-1))
            K2 = c2.selectbox("Sell Call (haut)", sa, index=min(len(sa)-1, aidx+1))
            iv1 = chain_calls[chain_calls['strike']==K1]['impliedVolatility'].values
            iv2 = chain_calls[chain_calls['strike']==K2]['impliedVolatility'].values
            if not (len(iv1) and len(iv2)): st.warning("Strike introuvable"); st.stop()
            p1 = calculate_greeks(spot, K1, T, r_rate, iv1[0], 'call')['Price']
            p2 = calculate_greeks(spot, K2, T, r_rate, iv2[0], 'call')['Price']
            cost = p1 - p2
            x = np.linspace(spot*0.7, spot*1.3, 120)
            y = [max(0,xi-K1) - max(0,xi-K2) - cost for xi in x]
            lbl = f"Bull Call Spread {K1:.0f}/{K2:.0f} | Cout: ${cost:.2f}"
        elif strat == "Bear Put Spread":
            c1, c2 = st.columns(2)
            K1 = c1.selectbox("Buy Put (haut)", sa, index=min(len(sa)-1, aidx+1))
            K2 = c2.selectbox("Sell Put (bas)", sa, index=max(0, aidx-1))
            iv1 = chain_puts[chain_puts['strike']==K1]['impliedVolatility'].values
            iv2 = chain_puts[chain_puts['strike']==K2]['impliedVolatility'].values
            if not (len(iv1) and len(iv2)): st.warning("Strike introuvable"); st.stop()
            p1 = calculate_greeks(spot, K1, T, r_rate, iv1[0], 'put')['Price']
            p2 = calculate_greeks(spot, K2, T, r_rate, iv2[0], 'put')['Price']
            cost = p1 - p2
            x = np.linspace(spot*0.7, spot*1.3, 120)
            y = [max(0,K1-xi) - max(0,K2-xi) - cost for xi in x]
            lbl = f"Bear Put Spread {K1:.0f}/{K2:.0f} | Cout: ${cost:.2f}"
        elif strat == "Long Straddle":
            K = st.selectbox("Strike ATM", sa, index=aidx)
            ivc = chain_calls[chain_calls['strike']==K]['impliedVolatility'].values
            ivp = chain_puts[chain_puts['strike']==K]['impliedVolatility'].values
            if not (len(ivc) and len(ivp)): st.warning("Strike introuvable"); st.stop()
            pc_ = calculate_greeks(spot, K, T, r_rate, ivc[0], 'call')['Price']
            pp_ = calculate_greeks(spot, K, T, r_rate, ivp[0], 'put')['Price']
            cost = pc_ + pp_
            x = np.linspace(spot*0.7, spot*1.3, 120)
            y = [max(0,xi-K) + max(0,K-xi) - cost for xi in x]
            lbl = f"Long Straddle {K:.0f} | Cout: ${cost:.2f}"
        elif strat == "Long Strangle":
            c1, c2 = st.columns(2)
            Kp = c1.selectbox("Buy Put (OTM)",  sa, index=max(0, aidx-2))
            Kc = c2.selectbox("Buy Call (OTM)", sa, index=min(len(sa)-1, aidx+2))
            ivc = chain_calls[chain_calls['strike']==Kc]['impliedVolatility'].values
            ivp = chain_puts[chain_puts['strike']==Kp]['impliedVolatility'].values
            if not (len(ivc) and len(ivp)): st.warning("Strike introuvable"); st.stop()
            pc_ = calculate_greeks(spot, Kc, T, r_rate, ivc[0], 'call')['Price']
            pp_ = calculate_greeks(spot, Kp, T, r_rate, ivp[0], 'put')['Price']
            cost = pc_ + pp_
            x = np.linspace(spot*0.7, spot*1.3, 120)
            y = [max(0,xi-Kc) + max(0,Kp-xi) - cost for xi in x]
            lbl = f"Long Strangle {Kp:.0f}/{Kc:.0f} | Cout: ${cost:.2f}"
        elif strat == "Iron Condor":
            c1, c2, c3, c4 = st.columns(4)
            Kp1 = c1.selectbox("Buy Put",   sa, index=max(0, aidx-3))
            Kp2 = c2.selectbox("Sell Put",  sa, index=max(0, aidx-1))
            Kc1 = c3.selectbox("Sell Call", sa, index=min(len(sa)-1, aidx+1))
            Kc2 = c4.selectbox("Buy Call",  sa, index=min(len(sa)-1, aidx+3))
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
        st.markdown(f"""
        <div class="strat-kpi-row">
          <div class="strat-kpi"><div class="sk-label">Profit Maximum</div><div class="sk-value pos">${mp:.2f}</div></div>
          <div class="strat-kpi"><div class="sk-label">Perte Maximum</div><div class="sk-value neg">${ml:.2f}</div></div>
          <div class="strat-kpi"><div class="sk-label">Ratio R/R</div><div class="sk-value" style="color:#e0e6f0">{rr:.2f}x</div></div>
        </div>""", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Erreur strategie : {e}")

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

        # Charger les strikes disponibles pour l'échéance actuelle
        try:
            roll_curr_calls, roll_curr_puts = get_option_chain(ticker, roll_current)
            roll_df = roll_curr_calls if roll_type == 'call' else roll_curr_puts
            roll_df = roll_df.dropna(subset=['impliedVolatility','lastPrice'])
            roll_strikes = sorted(roll_df['strike'].unique())
            roll_atm_idx = int(np.argmin(np.abs(np.array(roll_strikes)-spot)))

            roll_strike = st.selectbox("Strike a roller", roll_strikes,
                                        index=roll_atm_idx, key="roll_strike")

            if st.button("Calculer le Roll", type="primary"):
                with st.spinner("Calcul en cours..."):
                    result = compute_roll(ticker, roll_current, roll_new,
                                          roll_strike, roll_type, spot, r_rate)

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
                                <td style="color:#606878;text-align:left">Prix (bid/ask)</td>
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
                                <td style="color:#606878;text-align:left">Delta</td>
                                <td>{result['curr_delta']:.3f}</td>
                                <td class="highlight">{result['new_delta']:.3f}</td>
                            </tr>
                            <tr>
                                <td style="color:#606878;text-align:left">Theta /jour</td>
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
                                <div style="color:{'#3fb950' if theta_gain<0 else '#ff4b6e'};font-size:1rem;font-family:JetBrains Mono,monospace;margin-top:4px">
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
