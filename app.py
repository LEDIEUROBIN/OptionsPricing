import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime
import plotly.graph_objects as go

# --- MOTEUR DE CALCUL BLACK-SCHOLES ---
def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {"Price": 0.0, "Delta": 0.0, "Gamma": 0.0, "Theta": 0.0, "Vega": 0.0}
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = (S * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))
        delta = norm.cdf(d1)
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2))
    else:
        price = (K * np.exp(-r * T) * norm.cdf(-d2)) - (S * norm.cdf(-d1))
        delta = norm.cdf(d1) - 1
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2))

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    
    return {
        "Price": max(0, price), 
        "Delta": delta, 
        "Gamma": gamma, 
        "Theta": theta / 365, 
        "Vega": vega / 100
    }

# --- DONNÉES MARCHÉ ---
@st.cache_data(ttl=300)
def get_market_context():
    try:
        tnx = yf.Ticker("^TNX").history(period="1d")['Close'].iloc[-1] / 100
        vix = yf.Ticker("^VIX").history(period="1d")['Close'].iloc[-1]
        return tnx, vix
    except: 
        return 0.0425, 20.0

current_10y, current_vix = get_market_context()

# --- CONFIGURATION UI ---
st.set_page_config(layout="wide", page_title="Quantum Terminal V4.1", page_icon="💎")

st.markdown("""
    <style>
    .profile-box { background-color: #1c2128; padding: 20px; border-radius: 12px; border: 1px solid #30363d; border-left: 5px solid #00d4ff; margin-bottom: 20px; color: #e6edf3; }
    .vix-box { background: linear-gradient(90deg, #00d4ff 0%, #005f73 100%); padding: 15px; border-radius: 10px; color: white; text-align: center; }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Contrôle")
    ticker = st.text_input("Symbole Boursier", value="AAPL").upper()
    r_rate = st.number_input("Taux US10Y", value=float(current_10y), format="%.4f")
    st.divider()
    st.markdown(f'<div class="vix-box"><b>Indice VIX</b><br><span style="font-size:25px;">{current_vix:.2f}</span></div>', unsafe_allow_html=True)

if ticker:
    stock = yf.Ticker(ticker)
    try:
        info = stock.info
        spot = stock.history(period="1d")['Close'].iloc[-1]

        # 1. EN-TÊTE & DESCRIPTION (Réintégrée)
        st.title(f"🚀 {info.get('longName', ticker)}")
        st.markdown(f'''
            <div class="profile-box">
                <b>Secteur :</b> {info.get("sector", "N/A")} | <b>Industrie :</b> {info.get("industry", "N/A")}<br><br>
                {info.get("longBusinessSummary", "Aucune description disponible.")[:600]}...
            </div>
        ''', unsafe_allow_html=True)
        
        # 2. MÉTRIQUES GLOBALES
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Prix Spot", f"${spot:.2f}")
        m2.metric("Market Cap", f"{info.get('marketCap', 0)/1e9:.1f}B")
        m3.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A')}")
        m4.metric("Indice VIX", f"{current_vix:.2f}")

        # 3. OPTIONS
        st.divider()
        expiry_dates = stock.options
        if expiry_dates:
            expiry = st.selectbox("📅 Choisir l'Échéance", expiry_dates)
            T = max(1/365, (datetime.strptime(expiry, "%Y-%m-%d") - datetime.now()).days / 365.0)
            chain = stock.option_chain(expiry)
            
            tab_c, tab_p, tab_oi = st.tabs(["📈 CALL OPTIONS", "📉 PUT OPTIONS", "📊 OPEN INTEREST"])
            
            for tab, data, o_type in zip([tab_c, tab_p], [chain.calls, chain.puts], ['call', 'put']):
                with tab:
                    data = data.dropna(subset=['impliedVolatility', 'lastPrice'])
                    
                    # Graphique BSM
                    st.subheader("📊 Comparaison Modèle vs Marché")
                    mask = (data['strike'] > spot * 0.8) & (data['strike'] < spot * 1.2)
                    plot_data = data[mask].copy()
                    plot_data['BS_Price'] = plot_data.apply(lambda x: calculate_greeks(spot, x.strike, T, r_rate, x.impliedVolatility, o_type)['Price'], axis=1)
                    st.line_chart(plot_data.set_index('strike')[['lastPrice', 'BS_Price']], color=["#00d4ff", "#ff4b4b"])

                    # GRECQUES ATM
                    atm_strike_row = data.iloc[(data['strike']-spot).abs().argsort()[:1]]
                    greeks = calculate_greeks(spot, atm_strike_row['strike'].values[0], T, r_rate, atm_strike_row['impliedVolatility'].values[0], o_type)
                    
                    g1, g2, g3, g4 = st.columns(4)
                    g1.metric("Delta (Δ)", f"{greeks['Delta']:.3f}")
                    g2.metric("Gamma (Γ)", f"{greeks['Gamma']:.4f}")
                    g3.metric("Theta (Θ)", f"{greeks['Theta']:.3f}/j")
                    g4.metric("Vega (ν)", f"{greeks['Vega']:.3f}")

                    # --- SIMULATEUR DE P/L (Réintégré et Amélioré) ---
                    st.divider()
                    st.subheader("💰 Simulateur de Profit & Perte")
                    c_in, c_res = st.columns([1, 1.2])
                    
                    with c_in:
                        sel_strike = st.selectbox("Strike", data['strike'].unique(), key=f"s_{o_type}")
                        nb_contrats = st.number_input("Nombre de contrats", value=1, min_value=1, key=f"n_{o_type}")
                        target_p = st.number_input("Prix cible à l'échéance ($)", value=float(spot*1.1), key=f"t_{o_type}")
                        # On récupère le prix actuel du contrat sélectionné
                        price_buy = data[data['strike'] == sel_strike]['lastPrice'].values[0]
                    
                    with c_res:
                        investi = nb_contrats * price_buy * 100
                        # Valeur intrinsèque à l'échéance
                        val_finale = nb_contrats * max(0, (target_p - sel_strike) if o_type == 'call' else (sel_strike - target_p)) * 100
                        pnl = val_finale - investi
                        roi = (pnl / investi * 100) if investi > 0 else 0
                        
                        res_df = pd.DataFrame({
                            "Métrique": ["Investissement total", "Valeur à l'échéance", "P/L Net", "ROI %"],
                            "Valeur": [f"${investi:,.2f}", f"${val_finale:,.2f}", f"${pnl:,.2f}", f"{roi:.2f}%"]
                        })
                        st.table(res_df.set_index("Métrique"))

                        # Mini Payoff Plot
                        x_range = np.linspace(sel_strike * 0.7, sel_strike * 1.3, 50)
                        y_pnl = [(max(0, x - sel_strike if o_type=='call' else sel_strike - x) * 100 * nb_contrats) - investi for x in x_range]
                        fig = go.Figure(data=go.Scatter(x=x_range, y=y_pnl, fill='tozeroy', line=dict(color='#00d4ff')))
                        fig.add_hline(y=0, line_dash="dash", line_color="#ff4b4b")
                        fig.update_layout(height=180, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
                        st.plotly_chart(fig, use_container_width=True)

            # 4. ONGLET OPEN INTEREST
            with tab_oi:
                st.subheader("Concentration de l'Open Interest (Liquidité)")
                oi_calls = chain.calls[['strike', 'openInterest']].copy(); oi_calls['Type'] = 'Call'
                oi_puts = chain.puts[['strike', 'openInterest']].copy(); oi_puts['Type'] = 'Put'
                oi_df = pd.concat([oi_calls, oi_puts])
                oi_plot = oi_df[(oi_df['strike'] > spot * 0.7) & (oi_df['strike'] < spot * 1.3)]

                fig_oi = go.Figure()
                fig_oi.add_trace(go.Bar(x=oi_plot[oi_plot['Type']=='Call']['strike'], y=oi_plot[oi_plot['Type']=='Call']['openInterest'], name='Calls', marker_color='#00d4ff'))
                fig_oi.add_trace(go.Bar(x=oi_plot[oi_plot['Type']=='Put']['strike'], y=oi_plot[oi_plot['Type']=='Put']['openInterest'], name='Puts', marker_color='#ff4b4b'))
                fig_oi.add_vline(x=spot, line_dash="dash", line_color="white", annotation_text="PRIX SPOT")
                fig_oi.update_layout(barmode='group', height=400, template="plotly_dark")
                st.plotly_chart(fig_oi, use_container_width=True)

    except Exception as e:
        st.error(f"Erreur : {e}")