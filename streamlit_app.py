# streamlit_app.py — Dashboard 2x2 + deux logos (haut: BIMS, bas: IPT)
# streamlit_app.py — Dashboard 2x2 + deux logos + BRUIT sur T & RH
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sûr pour Streamlit
import matplotlib.pyplot as plt
import streamlit as st

# 1) Page config — DOIT venir avant tout autre st.*
st.set_page_config(page_title="Malaria SEI — Web Interface", layout="wide")

# ---------------------------
# Utilitaires : logos
# ---------------------------
LOGO_TOP = "image/logo_bims.png"   # header
LOGO_BOTTOM = "image/logo_IPT.jpg" # footer

def resolve_logo_path(path: str):
    """Résolution robuste pour supporter racine et /pages."""
    if os.path.exists(path):
        return path
    try:
        here = Path(__file__).resolve().parent
        cand = here / path
        if cand.exists():
            return str(cand)
        cand = (here / ".." / path).resolve()
        if cand.exists():
            return str(cand)
    except Exception:
        pass
    return None

logo_top_path = resolve_logo_path(LOGO_TOP)
logo_bottom_path = resolve_logo_path(LOGO_BOTTOM)

# ---------------------------
# Titre + logo (header)
# ---------------------------
st.title("Malaria Host–Vector Models — Web Interface")
st.caption("Drop-in UI you can connect to your `malaria_kermack_SEI_full_fused.ipynb` functions later.")

cols_head = st.columns([1, 2, 1])  # centre le logo top
with cols_head[1]:
    if logo_top_path and os.path.exists(logo_top_path):
        st.image(logo_top_path, use_container_width=True, caption="Developed by IPT-BIMS — 2025")
    else:
        st.caption("Developed by IPT-BIMS — 2025")
        if logo_top_path is None:
            st.info("Logo BIMS introuvable : vérifie que 'image/logo_bims.png' existe.")

with st.expander("How to plug into your notebook"):
    st.markdown(r"""
- This app currently uses a **lightweight Ross–Macdonald** implementation to compute time-varying \(R_0(t)\) and a simple SEI host proxy.
- To integrate your notebook:
  1. Export your core functions (e.g., `compute_R0_from_TRH`, `simulate_SEI`, `vector_capacity`) into a Python file (e.g., `malaria_core.py`).
  2. Replace the placeholder functions below with imports from your module.
  3. (Optional) To import directly from a notebook, consider **`nbclient`**/**`papermill`** workflows.
""")

# ---------------------------
# Helpers — placeholders (remplace par tes fonctions calibrées si besoin)
# ---------------------------

def _beta_T(T):
    """Biting/competence proxy vs temperature (smooth bell)."""
    T0, sig = 26.0, 6.0
    return np.exp(-((T - T0)**2) / (2 * sig**2))

def _mosq_survival_RH(RH):
    """Survival proxy vs relative humidity (0..1)."""
    RH = np.clip(RH, 0, 1)
    return 0.4 + 0.6*RH  # placeholder simple

def ross_macdonald_R0(m, a, b, c, g, mu, tau):
    """
    Ross–Macdonald R0 with EIP survival:
        R0 = (m * a^2 * b * c * exp(-mu * tau)) / (mu * g)
    (ici on néglige μ_h par rapport à g côté hôte)
    """
    m  = np.asarray(m)
    a  = np.asarray(a)
    mu = np.asarray(mu)
    num = m * (a**2) * b * c * np.exp(-mu * tau)
    den = np.maximum(mu * g, 1e-12)
    return num / den

def ross_macdonald_R0_time(T, RH, m_base, a_base, b, c, g, mu, tau):
    """Temperature- and humidity-modulated R0(t)."""
    beta = _beta_T(T)              # module a(t)
    surv = _mosq_survival_RH(RH)   # module m(t)
    m_t = m_base * surv
    a_t = a_base * beta
    return ross_macdonald_R0(m_t, a_t, b, c, g, mu, tau), m_t, a_t

def simulate_SEI(t_grid, beta_hv, gamma, N, I0=10, E0=0):
    """
    Minimal host-side SEI (S, E, I) with force proxy in beta_hv(t).
    """
    dt = np.diff(t_grid, prepend=t_grid[0])
    S = np.zeros_like(t_grid, dtype=float)
    E = np.zeros_like(t_grid, dtype=float)
    I = np.zeros_like(t_grid, dtype=float)

    S[0] = N - I0 - E0
    E[0] = E0
    I[0] = I0
    sigma = 1/10  # incubation ~10 days

    for k in range(1, len(t_grid)):
        dtk = max(dt[k], 1e-9)
        lam = beta_hv[k-1] * I[k-1] / max(N, 1e-9)
        dS = -lam * S[k-1]
        dE = lam * S[k-1] - sigma*E[k-1]
        dI = sigma*E[k-1] - gamma*I[k-1]

        S[k] = max(S[k-1] + dS*dtk, 0.0)
        E[k] = max(E[k-1] + dE*dtk, 0.0)
        I[k] = max(I[k-1] + dI*dtk, 0.0)

    return S, E, I

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("Inputs")

st.sidebar.markdown("### Climate (optional upload)")
datafile = st.sidebar.file_uploader("Upload CSV with columns: date (YYYY-MM-DD), T (°C), RH (%).",
                                    type=["csv"])

default_days = 365*3
if datafile is None:
    # Synthetic climate
    t = np.arange(default_days, dtype=float)
    T_base = 26 + 6*np.sin(2*np.pi*t/365)
    RH_base = 0.6 + 0.2*np.sin(2*np.pi*(t-45)/365)
    dates = pd.date_range("2020-01-01", periods=default_days, freq="D")
else:
    df = pd.read_csv(datafile)
    required = {"date", "T", "RH"}
    if not required.issubset(set(df.columns)):
        st.sidebar.error("CSV must contain columns: date, T, RH")
        st.stop()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    dates = df["date"].values
    T_base = df["T"].values
    RH_base = np.clip(df["RH"].values/100.0, 0, 1)  # 0..1
    t = np.arange(len(dates), dtype=float)

# --- NEW: bruit sur T & RH ---
st.sidebar.markdown("---")
st.sidebar.markdown("### Noise on climate")
noise_intensity = st.sidebar.slider("Noise intensity (0–1)", 0.0, 1.0, 0.2, 0.05)
noise_seed = st.sidebar.number_input("Random seed", value=0, step=1)

# Calcule bruit
rng = np.random.default_rng(int(noise_seed))
sigma_T = noise_intensity * 2.0     # °C
sigma_RH = noise_intensity * 0.15   # absolute (0..1)
noise_T = rng.normal(0.0, sigma_T, size=len(t))
noise_RH = rng.normal(0.0, sigma_RH, size=len(t))

# Applique bruit
T = T_base + noise_T
RH = np.clip(RH_base + noise_RH, 0.0, 1.0)

st.sidebar.caption("σ_T = {:.2f} °C, σ_RH = {:.2f}".format(sigma_T, sigma_RH))

st.sidebar.markdown("---")
st.sidebar.markdown("### Vector & transmission params")
m_base = st.sidebar.slider("m: mosquitoes per human", 0.1, 50.0, 8.0, 0.1)
a_base = st.sidebar.slider("a: bites per mosquito per day", 0.01, 1.0, 0.25, 0.01)
b      = st.sidebar.slider("b: mosquito → human transmission prob.", 0.0, 1.0, 0.3, 0.01)
c      = st.sidebar.slider("c: human → mosquito transmission prob.", 0.0, 1.0, 0.5, 0.01)
mu     = st.sidebar.slider("μ: mosquito death rate (1/day)", 0.01, 0.5, 0.1, 0.01)
tau    = st.sidebar.slider("τ: EIP (days)", 3.0, 25.0, 10.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.markdown("### SEI host dynamics")
g  = st.sidebar.slider("γ: human recovery rate (1/day)", 0.01, 0.5, 0.1, 0.01)
N  = st.sidebar.number_input("Population size N", value=100000, step=1000)
I0 = st.sidebar.number_input("Initial Infectious I0", value=10, step=1)
E0 = st.sidebar.number_input("Initial Exposed E0", value=0, step=1)

# ---------------------------
# Compute R0(t) and simulate
# ---------------------------
R0_t, m_t, a_t = ross_macdonald_R0_time(T, RH, m_base, a_base, b, c, g, mu, tau)

# Force of infection proxy scaled by R0(t)
beta_hv = R0_t * g
S, E, I = simulate_SEI(np.arange(len(dates), dtype=float), beta_hv, g, N, I0=I0, E0=E0)

# ---------------------------
# Dashboard 2x2 — UNE seule figure
# ---------------------------
fig, axs = plt.subplots(2, 2, figsize=(13, 8))

# (1) R0(t)
axs[0, 0].plot(dates, R0_t, lw=2, label="R₀(t)")
axs[0, 0].axhline(1.0, ls="--", color="gray")
axs[0, 0].set_title("R₀ over time")
axs[0, 0].set_ylabel("R₀(t)")
axs[0, 0].set_xlabel("Date")
axs[0, 0].legend()

# (2) SEI host compartments
axs[0, 1].plot(dates, S, label="S")
axs[0, 1].plot(dates, E, label="E")
axs[0, 1].plot(dates, I, label="I")
axs[0, 1].set_title("SEI — host compartments")
axs[0, 1].set_ylabel("Counts")
axs[0, 1].set_xlabel("Date")
axs[0, 1].legend()

# (3) a(t) & m(t) avec axe log pour m
ax3 = axs[1, 0]
ax3.plot(dates, a_t, label="a(t) — biting rate", color="tab:blue")
ax3.set_ylabel("a(t)")
ax3.set_xlabel("Date")
ax3.set_title("Vector drivers")

ax3b = ax3.twinx()
ax3b.plot(dates, m_t, label="m(t) — mosquitoes per human", color="tab:orange")
ax3b.set_yscale("log")
ax3b.set_ylabel("m(t) [log]")

lns = ax3.get_lines() + ax3b.get_lines()
labels = [l.get_label() for l in lns]
ax3.legend(lns, labels, loc="upper right")

# (4) Climate T & RH — montre BASE vs BRUITÉ
ax4 = axs[1, 1]
lns1 = ax4.plot(dates, T, label="T noisy (°C)")
ax4.plot(dates, T_base, label="T base (°C)", linestyle="--", color="gray", alpha=0.8)
ax4.set_ylabel("T (°C)")
ax4.set_xlabel("Date")
ax4.set_title("Climate (base vs noisy)")
ax4b = ax4.twinx()
lns2 = ax4b.plot(dates, RH, label="RH noisy (0–1)")
ax4b.plot(dates, RH_base, label="RH base (0–1)", linestyle="--", color="gray", alpha=0.8)
ax4b.set_ylabel("RH (0–1)")
# Légende combinée
lns = lns1 + lns2
labels = [l.get_label() for l in lns]
ax4.legend(lns, labels, loc="upper right")

plt.tight_layout()
st.pyplot(fig, clear_figure=True)

# ---------------------------
# Downloads
# ---------------------------
with st.expander("Download data as CSV"):
    out = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "R0": R0_t, "a_t": a_t, "m_t": m_t,
        "S": S, "E": E, "I": I,
        "T_base": T_base, "RH_base": RH_base,
        "T_noisy": T, "RH_noisy": RH
    })
    st.download_button("Download dashboard_data.csv",
                       data=out.to_csv(index=False),
                       file_name="dashboard_data.csv")

# ---------------------------
# Footer — logo IPT centré
# ---------------------------
st.divider()
cols = st.columns([1, 3, 1])  # centre le logo bottom
with cols[1]:
    if logo_bottom_path and os.path.exists(logo_bottom_path):
        st.image(logo_bottom_path, use_container_width=True, caption="Institut Pasteur de Tunis")
    else:
        st.caption("Institut Pasteur de Tunis")
        if logo_bottom_path is None:
            st.info("Logo IPT introuvable : vérifie que 'image/logo_IPT.jpg' existe.")

