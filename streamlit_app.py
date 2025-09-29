
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ---------------------------
# App metadata
# ---------------------------
st.set_page_config(page_title="Malaria SEI — Web Interface", layout="wide")

st.title("Malaria Host–Vector Models — Web Interface")
st.caption("Drop-in UI you can connect to your `malaria_kermack_SEI_full_fused.ipynb` functions later.")

with st.expander("How to plug into your notebook"):
    st.markdown("""
    - This app currently uses a **lightweight Ross–Macdonald** implementation to compute time-varying \\(R_0(t)\\) and simple SEI dynamics.
    - To integrate your notebook:
        1. Export your core functions (e.g., `compute_R0_from_TRH`, `simulate_SEI`, `vector_capacity`) into a Python file (e.g., `malaria_core.py`).
        2. Replace the placeholder functions below with imports from your module.
        3. (Optional) If you prefer to import directly from the notebook, try packages like **`importnb`** or **`nbclient`** to execute and access its symbols at runtime.
    """)

# ---------------------------
# Helpers — placeholder physics you can swap with your own
# ---------------------------

def _beta_T(T):
    "Biting/competence proxy vs temperature (smooth bell)."
    T0, sig = 26.0, 6.0
    return np.exp(-((T - T0)**2) / (2 * sig**2))

def _mosq_survival_RH(RH):
    "Survival proxy vs relative humidity (0..1)."
    RH = np.clip(RH, 0, 1)
    return 0.4 + 0.6*RH  # simple linear placeholder

def ross_macdonald_R0(m, a, b, c, g, mu):
    """
    Classic Ross–Macdonald R0 (scalar):
        R0 = (m * a^2 * b * c * e^{-mu * tau}) / (g * mu)
    Here g = human recovery rate, mu = mosquito death rate.
    """
    denom = (g * mu) if (g > 0 and mu > 0) else np.nan
    return (m * (a**2) * b * c) / denom

def ross_macdonald_R0_time(T, RH, m_base, a_base, b, c, g, mu):
    """
    Temperature- and humidity-modulated R0(t).
    """
    beta = _beta_T(T)
    surv = _mosq_survival_RH(RH)
    m_t = m_base * surv
    a_t = a_base * beta
    return ross_macdonald_R0(m_t, a_t, b, c, g, mu)

def simulate_SEI(t_grid, beta_hv, gamma, N, I0=10, E0=0):
    """
    Minimal host-side SEI (S, E, I) with vector force proxy in beta_hv(t).
    dS = -beta_hv * S / N * I_v? (proxy uses beta_hv directly)
    dE = beta_hv * S/N - sigma*E
    dI = sigma*E - gamma*I
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

        S[k] = S[k-1] + dS*dtk
        E[k] = E[k-1] + dE*dtk
        I[k] = I[k-1] + dI*dtk

        S[k] = max(S[k], 0.0)
        E[k] = max(E[k], 0.0)
        I[k] = max(I[k], 0.0)

    return S, E, I

# ---------------------------
# Sidebar controls
# ---------------------------

st.sidebar.header("Inputs")

st.sidebar.markdown("### Climate (optional upload)")
datafile = st.sidebar.file_uploader("Upload CSV with columns: date (YYYY-MM-DD), T (°C), RH (%).",
                                    type=["csv"])

default_days = 365
if datafile is None:
    # synthetic climate
    t = np.arange(default_days)
    T = 26 + 6*np.sin(2*np.pi*t/365)
    RH = 0.6 + 0.2*np.sin(2*np.pi*(t-45)/365)
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
    T = df["T"].values
    RH = (df["RH"].values)/100.0  # convert to 0..1 if given in %
    RH = np.clip(RH, 0, 1)
    t = np.arange(len(dates))

st.sidebar.markdown("---")
st.sidebar.markdown("### Vector & transmission params")
m_base = st.sidebar.slider("m: mosquitoes per human", 0.1, 50.0, 8.0, 0.1)
a_base = st.sidebar.slider("a: bites per mosquito per day", 0.01, 1.0, 0.25, 0.01)
b = st.sidebar.slider("b: mosquito → human transmission prob.", 0.0, 1.0, 0.3, 0.01)
c = st.sidebar.slider("c: human → mosquito transmission prob.", 0.0, 1.0, 0.5, 0.01)
mu = st.sidebar.slider("μ: mosquito death rate (1/day)", 0.01, 0.5, 0.1, 0.01)
g = st.sidebar.slider("γ: human recovery rate (1/day)", 0.01, 0.5, 0.1, 0.01)

st.sidebar.markdown("---")
st.sidebar.markdown("### SEI host dynamics")
N = st.sidebar.number_input("Population size N", value=100000, step=1000)
I0 = st.sidebar.number_input("Initial Infectious I0", value=10, step=1)
E0 = st.sidebar.number_input("Initial Exposed E0", value=0, step=1)

# ---------------------------
# Compute R0(t) and simulate
# ---------------------------

RH01 = RH if datafile is not None else np.clip(RH, 0, 1)
R0_t = ross_macdonald_R0_time(T, RH01, m_base, a_base, b, c, g, mu)

# Force of infection proxy scaled by R0(t)
beta_hv = R0_t * g

t_days = t.astype(float)
S, E, I = simulate_SEI(t_days, beta_hv, g, N, I0=I0, E0=E0)

# ---------------------------
# Layout & plots
# ---------------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("R0 over time")
    fig1, ax1 = plt.subplots(figsize=(7,3.0))
    ax1.plot(dates, R0_t, lw=2)
    ax1.axhline(1.0, ls="--")
    ax1.set_ylabel("R0(t)")
    ax1.set_xlabel("Date")
    st.pyplot(fig1, clear_figure=True)

    with st.expander("Download R0(t) as CSV"):
        out = pd.DataFrame({"date": pd.to_datetime(dates), "R0": R0_t})
        st.download_button("Download R0.csv", data=out.to_csv(index=False), file_name="R0_time_series.csv")

with col2:
    st.subheader("SEI — host compartments")
    fig2, ax2 = plt.subplots(figsize=(7,3.0))
    ax2.plot(dates, S, label="S")
    ax2.plot(dates, E, label="E")
    ax2.plot(dates, I, label="I")
    ax2.set_ylabel("Counts")
    ax2.set_xlabel("Date")
    ax2.legend()
    st.pyplot(fig2, clear_figure=True)

st.markdown("---")
st.subheader("Climate inputs (preview)")
clim = pd.DataFrame({"date": pd.to_datetime(dates), "T_C": T, "RH": RH01})
st.dataframe(clim.head(20))

st.markdown("Tip: Replace the placeholder functions with your calibrated ones to match **Tunis** climate and your field data.")
