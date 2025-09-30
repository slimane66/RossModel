# streamlit_app.py
# -----------------------------------------------
# Seasonal malaria host–vector demo (Streamlit)
# Sliders in sidebar • 2x2 matplotlib figures
# Logo displayed at the bottom of the page
# -----------------------------------------------

import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe for Streamlit
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Seasonal R0 — Malaria (IPT-BIMS)", layout="wide")

# =============== Helpers (self-contained) ===============

def seasonal_cos(t, mean, amp, period=365.0, phase=0.0):
    """Cosine seasonality around 'mean' with relative amplitude 'amp'."""
    return mean * (1.0 + amp * np.cos(2*np.pi*(t - phase)/period))

def briere_a(T, a_opt=0.30):
    """
    Simple temperature-dependent biting rate proxy.
    Brière-like bell between Tmin=12°C and Tmax=38°C (peak near 28–30°C).
    Returns a daily biting rate in ~[~0, ~a_opt].
    """
    Tmin, Tmax = 12.0, 38.0
    T = np.asarray(T)
    x = np.clip(T, Tmin, Tmax)
    arch = (x - Tmin) * (Tmax - x)
    arch = arch - arch.min()
    if arch.max() > 0:
        arch = arch / arch.max()
    a = a_opt * arch
    return np.maximum(a, 1e-6)

def mu_v_from_T_RH(T, RH):
    """
    Vector mortality (per day) depending on Temperature & Relative Humidity.
    Lower at warm/moist, higher at very hot/dry. Clipped for stability.
    """
    T = np.asarray(T); RH = np.asarray(RH)
    temp_penalty = 0.25 / (1.0 + np.exp(-(T - 35.0)))   # rises after ~35°C
    hum_penalty  = 0.20 * np.clip(0.6 - RH, 0.0, 1.0)   # penalty if RH<0.6
    mu = 0.08 + temp_penalty + hum_penalty
    return np.clip(mu, 0.02, 0.5)

def R0_with_EIP_exact_params(a, b, c, m, mu_v, gamma_h, mu_h, tau):
    """
    Ross–Macdonald-style R0(t) proxy with EIP survival:
    R0 = [a^2 * b * c * m * exp(-mu_v * tau)] / [mu_v * (gamma_h + mu_h)]
    """
    a = np.asarray(a); mu_v = np.asarray(mu_v)
    num = (a**2) * b * c * m * np.exp(-mu_v * tau)
    den = np.maximum(mu_v * (gamma_h + mu_h), 1e-12)
    return num / den

# =============== Sidebar controls ===============
st.sidebar.title("Seasonality controls")

tau        = st.sidebar.slider("tau (days, EIP)", 3.0, 25.0, 10.0, 0.5)
T_mean     = st.sidebar.slider("T_mean (°C)", 15.0, 35.0, 28.0, 0.5)
T_amp      = st.sidebar.slider("T_amp (relative)", 0.0, 0.9, 0.15, 0.05)
RH_mean    = st.sidebar.slider("RH_mean", 0.20, 0.90, 0.60, 0.02)
RH_amp     = st.sidebar.slider("RH_amp (relative)", 0.0, 0.9, 0.30, 0.05)
a_opt      = st.sidebar.slider("a_opt (peak biting rate)", 0.10, 0.80, 0.30, 0.01)
m0         = st.sidebar.slider("m0 (vector/host ratio baseline)", 0.5, 30.0, 5.0, 0.5)
mu_v0      = st.sidebar.slider("mu_v0 (baseline for link)", 0.02, 0.30, 0.10, 0.005)
link_m     = st.sidebar.checkbox("m linked to mu_v (stabilize Nv)", value=True)
tmax       = st.sidebar.slider("tmax (days)", 90, 1095, 365, 15)

# Optional epi params (fixed for clarity)
b       = 0.3
c       = 0.3
gamma_h = 1/14
mu_h    = 1/(70*365)

# --- Sidebar branding ---
st.sidebar.markdown("---")
st.sidebar.subheader("Branding")
# Your local logo file (as requested)
# Résolution robuste du chemin du logo (gère script à la racine ou dans /pages)
PREFERRED_LOGO2 = "image/logo_bims.png"
PREFERRED_LOGO = "image/logo_IPT.png"

def resolve_logo_path(preferred: str) :
    # 1) Chemin relatif au répertoire de travail courant (CWD)
    if os.path.exists(preferred):
        return preferred
    # 2) Chemin relatif au dossier du script courant
    try:
        here = Path(__file__).resolve().parent
        cand = here / preferred
        if cand.exists():
            return str(cand)
        # 3) Si le script est dans /pages, remonter d’un niveau
        cand = (here / ".." / preferred).resolve()
        if cand.exists():
            return str(cand)
    except Exception:
        pass
    return None

logo_path = resolve_logo_path(PREFERRED_LOGO)
logo_path2 = resolve_logo_path(PREFERRED_LOGO2)

# Affichage du logo (header, sidebar, ou footer — à toi de choisir l’endroit)
if logo_path:
    st.image(logo_path, width=100, caption="IPT-BIMS")
else:
    # Aide au débogage non bloquante
    st.warning(
        "Logo introuvable. Vérifie que le fichier **image/logo_bims.png** existe bien "
        "et que le chemin correspond. Si besoin, change PREFERRED_LOGO."
    )
    # Pour t’aider à vérifier où tu es et ce que Streamlit voit :
    st.caption("Debug: répertoire courant = " + os.getcwd())

# =============== Simulation ===============
st.title("Seasonal malaria dynamics & R₀(t) — IPT-BIMS")

# Init conditions
Sh, Ih, Rh = 999.0, 1.0, 0.0
Sv, Ev, Iv = 4999.0, 0.0, 1.0
y = np.array([Sh, Ih, Rh, Sv, Ev, Iv], dtype=float)

dt = 0.5
Tgrid = np.arange(0, tmax + dt, dt)

Ih_series, Nh_series = [], []
a_series, m_series, mu_series, R_series = [], [], [], []

for t in Tgrid:
    Tt  = seasonal_cos(t, mean=T_mean, amp=T_amp, period=365.0)
    RHt = np.clip(seasonal_cos(t, mean=RH_mean, amp=RH_amp, period=365.0), 0.01, 0.99)

    a_t    = briere_a(Tt, a_opt=a_opt)
    mu_v_t = mu_v_from_T_RH(Tt, RHt)

    m_t = m0 * (mu_v0 / mu_v_t) if link_m else m0

    Sh, Ih, Rh, Sv, Ev, Iv = y
    Nh = Sh + Ih + Rh
    Nv = Sv + Ev + Iv

    lam_h = a_t * b * m_t * (Iv / max(Nv, 1e-9))
    lam_v = a_t * c * (Ih / max(Nh, 1e-9))
    sigma_v  = 1.0 / tau
    Lambda_v = mu_v_t * 5000.0

    dSh = - lam_h * Sh - mu_h * Sh
    dIh =   lam_h * Sh - (gamma_h + mu_h) * Ih
    dRh =   gamma_h * Ih - mu_h * Rh

    dSv =   Lambda_v - lam_v * Sv - mu_v_t * Sv
    dEv =   lam_v * Sv - (mu_v_t + sigma_v) * Ev
    dIv =   sigma_v * Ev - mu_v_t * Iv

    y = y + dt * np.array([dSh, dIh, dRh, dSv, dEv, dIv], dtype=float)
    y = np.maximum(y, 0.0)

    Ih_series.append(y[1]); Nh_series.append(y[0] + y[1] + y[2])
    a_series.append(a_t); m_series.append(m_t); mu_series.append(mu_v_t)
    R_series.append(R0_with_EIP_exact_params(a_t, b, c, m_t, mu_v_t, gamma_h, mu_h, tau))

Ih_series = np.array(Ih_series); Nh_series = np.array(Nh_series)
a_series  = np.array(a_series);  m_series  = np.array(m_series)
mu_series = np.array(mu_series); R_series  = np.array(R_series)

# =============== Plot (2x2) ===============
fig, axs = plt.subplots(2, 2, figsize=(11, 8))

axs[0, 0].plot(Tgrid, Ih_series / np.maximum(Nh_series, 1e-12))
axs[0, 0].set_xlabel("Time (days)"); axs[0, 0].set_ylabel("I_h / N_h")
axs[0, 0].set_title("Seasonal prevalence")

axs[0, 1].plot(Tgrid, a_series, label="a(t)")
axs[0, 1].plot(Tgrid, m_series, label="m(t)")
axs[0, 1].set_xlabel("Time (days)"); axs[0, 1].set_ylabel("a(t), m(t)")
axs[0, 1].set_title("Seasonal drivers"); axs[0, 1].legend()

axs[1, 0].plot(Tgrid, mu_series)
axs[1, 0].set_xlabel("Time (days)"); axs[1, 0].set_ylabel("mu_v(t)")
axs[1, 0].set_title("Vector mortality over time")

axs[1, 1].plot(Tgrid, R_series)
axs[1, 1].axhline(1.0, ls="--")
axs[1, 1].set_xlabel("Time (days)"); axs[1, 1].set_ylabel("R0(t)")
axs[1, 1].set_title("Instantaneous R0(t) proxy")

plt.tight_layout()
st.pyplot(fig, clear_figure=True)

# =============== Footer / Logo (bottom) ===============
st.divider()
cols = st.columns([1, 3, 1])  # center the logo
with cols[1]:
    if logo_path and os.path.exists(logo_path):
        st.image(logo_path2, use_container_width=False, caption="Developed by IPT-BIMS — 2025")
    else:
        # message non bloquant si le fichier n'est pas trouvé
        st.caption("Developed by IPT-BIMS — 2025")
        if logo_path is None:
            st.info("Logo introuvable : vérifie que 'image/logo_bims.png' existe dans le dépôt ou ajuste le chemin.")

