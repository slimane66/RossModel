#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 21:20:55 2025

@author: slimane
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title='Seasonality & Climate — Malaria R0', layout='wide')
st.title('Seasonality & Climate — $R_0(t)$, Vector Capacity, and Sensitivity')

st.caption(
    "Cette page modélise l’effet de la température **T(t)** et de l’humidité **RH(t)** via "
    "le taux de piqûre **a(T)**, la densité **m(T,RH)**, la survie, le temps d’incubation extrinsèque **τ(T)** "
    "et donc **$R_0(t)$** et la **vector capacity**. "
    "Elle peut aussi essayer de **charger des fonctions** depuis ton notebook."
)

# =========================
# Chargement optionnel de fonctions depuis un notebook (.ipynb)
# =========================
st.sidebar.header('Avancé : Charger des fonctions depuis un .ipynb')
use_nb = st.sidebar.checkbox('Utiliser des fonctions depuis un .ipynb (exécuter les cellules)', value=False)
nb_path = st.sidebar.text_input(
    'Chemin du notebook',
    value='/mnt/data/malaria_kermack_SEI_T_RH_widgets_seasonality(3).ipynb'
)
nb_ns = {}

def _load_functions_from_ipynb(path: str):
    """Charge et exécute les cellules code d’un notebook pour récupérer ses fonctions."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        local_ns = {}
        for cell in nb.get('cells', []):
            if cell.get('cell_type') == 'code':
                src = ''.join(cell.get('source', []))
                try:
                    exec(src, local_ns, local_ns)  # ⚠️ exécute le code du notebook
                except Exception:
                    # On ignore les erreurs cellule par cellule pour ne pas bloquer la page
                    pass
        return local_ns
    except Exception:
        return {}

if use_nb:
    nb_ns = _load_functions_from_ipynb(nb_path)
    if nb_ns:
        st.sidebar.success('Notebook chargé. Si des fonctions comme `a_of_T`, `m_of_TRH`, '
                           '`tau_of_T`, `vector_capacity`, `compute_R0_from_TRH` existent, elles seront utilisées.')
    else:
        st.sidebar.warning("Impossible de charger le notebook ou fonctions indisponibles — on utilisera les défauts.")

# =========================
# 1) Données climat
# =========================
st.subheader('1) Données climat')
tab_csv, tab_synth = st.tabs(['Upload CSV (date, T, RH)', 'Climat saisonnier synthétique'])

with tab_csv:
    up = st.file_uploader('CSV avec colonnes: date (YYYY-MM-DD), T (°C), RH (%)', type=['csv'], key='clim_csv')
    if up is not None:
        df = pd.read_csv(up)
        if not {'date','T','RH'}.issubset(df.columns):
            st.error('CSV obligatoire: date, T, RH')
            st.stop()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        dates = df['date'].values
        T = df['T'].astype(float).values
        RH = np.clip(df['RH'].astype(float).values/100.0, 0, 1)
    else:
        dates, T, RH = None, None, None
        st.info("Pas encore de CSV envoyé.")

with tab_synth:
    days = st.number_input('Nombre de jours', value=365, step=1, min_value=30, max_value=1825)
    start_date = st.date_input('Date de début', value=pd.to_datetime('2020-01-01'))
    t = np.arange(days, dtype=float)

    T_mean = st.slider('Température moyenne (°C)', 10.0, 35.0, 26.0, 0.1)
    T_amp  = st.slider('Amplitude saisonnière T (°C)', 0.0, 15.0, 6.0, 0.1)
    T_phi  = st.slider('Déphasage T (jours)', -180.0, 180.0, 0.0, 1.0)

    RH_mean = st.slider('Humidité relative moyenne (0..1)', 0.0, 1.0, 0.6, 0.01)
    RH_amp  = st.slider('Amplitude saisonnière RH (0..1)', 0.0, 0.5, 0.2, 0.01)
    RH_phi  = st.slider('Déphasage RH (jours)', -180.0, 180.0, -45.0, 1.0)

    T_syn  = T_mean + T_amp*np.sin(2*np.pi*(t - T_phi)/365.0)
    RH_syn = np.clip(RH_mean + RH_amp*np.sin(2*np.pi*(t - RH_phi)/365.0), 0, 1)
    dates_syn = pd.date_range(start_date, periods=days, freq='D').values

    colT, colRH = st.columns(2)
    with colT:
        st.markdown('**Température (synthétique)**')
        figT, axT = plt.subplots(figsize=(7,2.8))
        axT.plot(dates_syn, T_syn, lw=2)
        axT.set_ylabel('T (°C)')
        axT.set_xlabel('Date')
        st.pyplot(figT, clear_figure=True)
    with colRH:
        st.markdown('**Humidité relative (synthétique)**')
        figR, axR = plt.subplots(figsize=(7,2.8))
        axR.plot(dates_syn, RH_syn, lw=2)
        axR.set_ylabel('RH (0..1)')
        axR.set_xlabel('Date')
        st.pyplot(figR, clear_figure=True)

# Choix de la source climat à utiliser
if (dates is not None) and st.checkbox('Utiliser le CSV uploadé', value=False):
    dates_use, T_use, RH_use = dates, T, RH
else:
    dates_use, T_use, RH_use = dates_syn, T_syn, RH_syn

# =========================
# 2) Paramètres & liaisons climat → transmission
# =========================
st.subheader('2) Paramètres de transmission et liaisons climat')

colL, colR = st.columns(2)
with colL:
    m_base = st.slider('m base (moustiques / humain)', 0.1, 50.0, 8.0, 0.1)
    a_base = st.slider('a base (piqûres / moustique / jour)', 0.01, 1.0, 0.25, 0.01)
    b = st.slider('b : moustique → humain', 0.0, 1.0, 0.3, 0.01)
    c = st.slider('c : humain → moustique', 0.0, 1.0, 0.5, 0.01)
with colR:
    mu = st.slider('μ : mortalité moustique (1/jour)', 0.01, 0.5, 0.1, 0.01)
    g  = st.slider('γ : guérison humain (1/jour)', 0.01, 0.5, 0.1, 0.01)
    tau_scale = st.slider('Facteur d’échelle de τ(T) (sensibilité)', 0.25, 2.5, 1.0, 0.05)

# Fonctions défaut (tu peux les remplacer par celles de ton notebook)
def a_of_T_default(T):
    T0, sig = 26.0, 6.0
    return a_base * np.exp(-((T - T0)**2) / (2*sig**2))

def m_of_TRH_default(T, RH):
    T0, sig = 26.0, 6.0
    bell = np.exp(-((T - T0)**2) / (2*sig**2))
    return m_base * (0.4 + 0.6*np.clip(RH,0,1)) * bell

def tau_of_T_default(T):
    # τ(T) ≈ 111/(T - 14.7) (coupée à [5, 30] jours), puis mise à l’échelle
    T_eff = np.maximum(T, 14.71)
    tau = 111.0 / (T_eff - 14.7)
    tau = np.clip(tau, 5.0, 30.0)
    return tau_scale * tau

def daily_survival_default(mu):
    # p = e^{-μ}
    return np.exp(-mu)

# Raccord éventuel aux fonctions du notebook
a_of_T = a_of_T_default
m_of_TRH = m_of_TRH_default
tau_of_T = tau_of_T_default
vector_capacity_fn = None
compute_R0_fn = None

if use_nb and nb_ns:
    a_of_T = nb_ns.get('a_of_T', a_of_T)
    m_of_TRH = nb_ns.get('m_of_TRH', m_of_TRH)
    tau_of_T = nb_ns.get('tau_of_T', tau_of_T)
    vector_capacity_fn = nb_ns.get('vector_capacity', None)
    compute_R0_fn = nb_ns.get('compute_R0_from_TRH', None)

# Séries temporelles
T_use = np.array(T_use)
RH_use = np.array(RH_use)

a_t   = a_of_T(T_use)
m_t   = m_of_TRH(T_use, RH_use)
tau_t = tau_of_T(T_use)
p_t   = np.full_like(tau_t, daily_survival_default(mu))
n_t   = np.maximum(1.0, tau_t)  # en jours

# Vector capacity C(t) par défaut : C = (m a^2 p^n) / (-ln p)
def vector_capacity_default(m, a, p, n):
    denom = -np.log(np.maximum(1e-9, p))
    return (m * a**2 * (p**n)) / np.maximum(1e-9, denom)

if vector_capacity_fn is None:
    C_t = vector_capacity_default(m_t, a_t, p_t, n_t)
else:
    try:
        C_t = vector_capacity_fn(m_t, a_t, p_t, n_t)
    except Exception:
        C_t = vector_capacity_default(m_t, a_t, p_t, n_t)

# R0(t) par défaut (Ross–Macdonald) avec survie ~ e^{-μ τ}
def R0_default(m, a, b, c, g, mu, tau):
    denom = (g * mu)
    surv = np.exp(-mu * tau)
    return (m * (a**2) * b * c * surv) / np.maximum(1e-9, denom)

if compute_R0_fn is None:
    R0_t = R0_default(m_t, a_t, b, c, g, mu, tau_t)
else:
    try:
        # Signature suggérée côté notebook
        R0_t = compute_R0_fn(
            T=T_use, RH=RH_use, m_base=m_base, a_base=a_base,
            b=b, c=c, g=g, mu=mu, tau_scale=tau_scale
        )
    except Exception:
        R0_t = R0_default(m_t, a_t, b, c, g, mu, tau_t)

# =========================
# 3) Résultats & graphiques
# =========================
st.subheader('3) Sorties')

c1, c2 = st.columns(2)
with c1:
    st.markdown('**R₀(t)**')
    figR0, axR0 = plt.subplots(figsize=(7,3.0))
    axR0.plot(dates_use, R0_t, lw=2)
    axR0.axhline(1.0, ls='--')
    axR0.set_ylabel('R0(t)')
    axR0.set_xlabel('Date')
    st.pyplot(figR0, clear_figure=True)

with c2:
    st.markdown('**Vector capacity C(t)**')
    figC, axC = plt.subplots(figsize=(7,3.0))
    axC.plot(dates_use, C_t, lw=2)
    axC.set_ylabel('C(t)')
    axC.set_xlabel('Date')
    st.pyplot(figC, clear_figure=True)

# Sensibilité à τ(T)
st.markdown('**Sensibilité : facteur sur τ(T)**')
s_factors = [0.5, 1.0, 1.5, 2.0]
figS, axS = plt.subplots(figsize=(10,3.0))
for s in s_factors:
    # On repart de la τ par défaut pour ne pas cumuler l’échelle
    T_eff = np.maximum(T_use, 14.71)
    tau_base = np.clip(111.0 / (T_eff - 14.7), 5.0, 30.0)
    tau_s = tau_base * s
    R0_s = R0_default(m_t, a_t, b, c, g, mu, tau_s)
    axS.plot(dates_use, R0_s, label=f'τ×{s:0.1f}')
axS.axhline(1.0, ls='--')
axS.set_ylabel('R0(t)')
axS.set_xlabel('Date')
axS.legend()
st.pyplot(figS, clear_figure=True)

# =========================
# 4) Export
# =========================
st.subheader('4) Export')
out = pd.DataFrame({
    'date': pd.to_datetime(dates_use),
    'T_C': T_use,
    'RH': RH_use,
    'a(T)': a_t,
    'm(T,RH)': m_t,
    'tau(T)': tau_t,
    'p': p_t,
    'C(t)': C_t,
    'R0(t)': R0_t,
})
st.download_button('Télécharger CSV', data=out.to_csv(index=False), file_name='seasonality_outputs.csv')
st.dataframe(out.head(15))

st.caption(
    "Astuce : si ton notebook définit `a_of_T`, `m_of_TRH`, `tau_of_T`, `vector_capacity`, "
    "ou `compute_R0_from_TRH`, cette page les utilisera quand l’option est cochée dans la sidebar."
)
