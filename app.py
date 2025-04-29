
import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Constants
k_f = 97.2
k_r = 13.1
K_MBA = 0.49
K_ERY = 101.2
K_AP = 19.6
K_ABT = 39.4
K_eq = 830

def batch_model(t, y, E1, E2):
    HPA, GA, ERY, MBA, ABT, AP = y
    denominator_tk = (0.49 * HPA * (1 + HPA/50) + 0.5 * GA * (1 + GA/50) +
                     HPA * GA + (0.5/2.5) * GA * ERY + (0.5*1.2)/0.8 * ERY)
    V = (k_f * E1 * HPA * GA) / denominator_tk

    numerator_ta = k_f * k_r * E2 * (MBA * ERY - (AP * ABT)/K_eq)
    denominator_ta = (k_r * K_MBA * ERY + k_f * K_ERY * MBA +
                     k_f * ERY * MBA + (k_f * K_AP * ABT)/K_eq +
                     (k_f * K_ABT * AP)/K_eq + (k_f * AP * ABT)/K_eq)
    v = numerator_ta / denominator_ta

    dHPA = -V
    dGA = -V
    dERY = V - v
    dMBA = -v
    dABT = +v
    dAP = +v
    return [dHPA, dGA, dERY, dMBA, dABT, dAP]

# Streamlit UI
st.title("ABT Production Simulator (Batch Mode)")

E1 = st.slider("Enzyme E1 concentration (mM)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
E2 = st.slider("Enzyme E2 concentration (mM)", min_value=0.1, max_value=2.0, value=0.3, step=0.1)

y0 = [50, 50, 0, 20, 0, 0]
t_span = [0, 120]
solution = solve_ivp(
    batch_model,
    t_span,
    y0,
    args=(E1, E2),
    method='LSODA',
    dense_output=True
)

t_plot = np.linspace(0, 120, 100)
ABT = solution.sol(t_plot)[4]

fig, ax = plt.subplots()
ax.plot(t_plot, ABT, 'b-', label='[ABT]')
ax.set_xlabel("Time (min)")
ax.set_ylabel("[ABT] (mM)")
ax.set_title("ABT Concentration Over Time")
ax.legend()
st.pyplot(fig)
