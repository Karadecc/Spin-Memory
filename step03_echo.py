"""Step 3 — echo memory: why a plain Hahn echo fails, and the two-pi-pulse
"silenced echo" protocol.

Figures:
 (a) Hahn echo with the resonator kept on resonance: after the pi pulse the
     ensemble is inverted and amplifies — the echo carries MORE energy than the input
 (b) two-pi protocol at C = 1: absorb, pi, first echo silenced by detuning the
     resonator, second pi, second echo emitted with the spins back in the ground state
 (c) memory efficiency vs C, compared with [4C/(1+C)^2]^2 exp(-2T/T2)
 (d) memory efficiency vs storage time for several T2

Limit of the model: it is linear and mean-field (classical amplitudes). It shows the
gain of an inverted ensemble but not the spontaneous-emission noise that comes with
it — which is exactly why gain is unacceptable for a QUANTUM memory.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from spinmem import (discretise_gaussian, g_for_C_gaussian, gamma_from_T2,
                     run_sequence)

# Toy parameters (MHz, us)
kappa, Gamma_inh = 1.0, 4.0
x, rho = discretise_gaussian(Gamma_inh, x_max=12, dx=0.02)   # recurrence at 50 us
tau_p, t0 = 1.5, 4.5         # input pulse FWHM and centre
t_pi1 = t0 + 5.0             # first pi pulse
delta_off = 20.0             # resonator detuning used to silence the first echo (MHz)


def gaussian_pulse(tau, tc):
    A = (tau * np.sqrt(np.pi / (4 * np.log(2))))**-0.5
    return lambda t: A * np.exp(-2 * np.log(2) * (t - tc)**2 / tau**2)


a_in = gaussian_pulse(tau_p, t0)


def energy(t, f, lo, hi):
    m = (t >= lo) & (t < hi)
    return np.trapezoid(np.abs(f[m])**2, t[m])


def hahn(C, T2=200.0, t_end=t0 + 14):
    gamma = gamma_from_T2(T2)
    g = g_for_C_gaussian(C, kappa, Gamma_inh, gamma)
    segs = [dict(t0=0, t1=t_pi1, sz=-1, delta_c=0),
            dict(t0=t_pi1, t1=t_end, sz=+1, delta_c=0, pi_before=True)]
    return run_sequence(segs, a_in, x, rho, g, kappa, 0, gamma)


def two_pi(C, T_store=20.0, T2=200.0):
    """Storage time T_store = t_out - t0; the output echo comes at t0 + 2(t_pi2 - t_pi1)."""
    gamma = gamma_from_T2(T2)
    g = g_for_C_gaussian(C, kappa, Gamma_inh, gamma)
    t_pi2 = t_pi1 + T_store / 2
    t_out = t0 + T_store
    # the second pi pulse must come after the (silenced) first echo, at t0 + 2(t_pi1 - t0)
    assert t_pi2 > t0 + 2 * (t_pi1 - t0) + 2 * tau_p, "storage time too short for this sequence"
    segs = [dict(t0=0, t1=t_pi1, sz=-1, delta_c=0),
            dict(t0=t_pi1, t1=t_pi2, sz=+1, delta_c=delta_off, pi_before=True),
            dict(t0=t_pi2, t1=t_out + 5, sz=-1, delta_c=0, pi_before=True)]
    t, a, ao, Es, S, D = run_sequence(segs, a_in, x, rho, g, kappa, 0, gamma)
    absorbed = 1 - energy(t, ao, 0, t_pi1)
    leak = energy(t, ao, t_pi1, t_pi2)          # what escapes at the silenced echo
    eta = energy(t, ao, t_pi2, t_out + 5)       # retrieved echo
    return dict(t=t, ao=ao, Es=Es, S=S, D=D, absorbed=absorbed, leak=leak, eta=eta,
                t_pi2=t_pi2, t_out=t_out)


fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))

# (a) Hahn echo in an inverted ensemble
for C, col in [(0.1, "C0"), (0.3, "C1"), (0.6, "C3")]:
    t, a, ao, Es, S, D = hahn(C)
    E_echo = energy(t, ao, t_pi1, t[-1])
    ax[0, 0].semilogy(t, np.abs(ao)**2 + 1e-9, color=col,
                      label=f"C = {C}: echo energy = {E_echo:.2f} × input")
    print(f"Hahn, C = {C}: absorbed {1 - energy(t, ao, 0, t_pi1):.3f}, echo energy {E_echo:.3f}")
ax[0, 0].semilogy(t, np.abs(a_in(t))**2, "k", lw=1, label="input (1 photon)")
ax[0, 0].axvline(t_pi1, color="gray", ls=":")
ax[0, 0].text(t_pi1 + 0.2, 3e-5, "π", color="gray")
ax[0, 0].set(xlabel="time (µs)", ylabel="|a_out|² (photons/µs)", ylim=(1e-5, 30),
             title="(a) Hahn echo, resonator on: the inverted ensemble amplifies")
ax[0, 0].legend(fontsize=8, loc="upper left")

# (b) two-pi protocol, C = 1
r = two_pi(1.0)
t = r["t"]
axb = ax[0, 1]
axb.axvspan(t_pi1, r["t_pi2"], color="0.9", label=f"resonator detuned by {delta_off:.0f} MHz")
axb.plot(t, np.abs(a_in(t))**2, "k", lw=1, label="input")
axb.plot(t, np.abs(r["ao"])**2, "C0", label="output")
for tp in (t_pi1, r["t_pi2"]):
    axb.axvline(tp, color="C3", ls=":")
    axb.text(tp + 0.2, 0.02, "π", color="C3")
axb.annotate("silenced echo", xy=(t0 + 10, 0.01), xytext=(t0 + 6.5, 0.15),
             arrowprops=dict(arrowstyle="->", color="0.4"), fontsize=8, color="0.4")
axb.annotate(f"retrieved: η = {r['eta']:.2f}", xy=(r["t_out"] - 0.6, 0.35), xytext=(r["t_out"] - 12, 0.3),
             arrowprops=dict(arrowstyle="->", color="0.4"), fontsize=8, color="0.4")
axb.set(xlabel="time (µs)", ylabel="power (photons/µs)", ylim=(0, 0.7),
        title="(b) two π pulses + silenced first echo, C = 1, T₂ = 200 µs")
axb.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.3, 0.99))
print(f"two-pi, C = 1: absorbed {r['absorbed']:.4f}, leak at silenced echo {r['leak']:.1e}, "
      f"eta {r['eta']:.4f}, A^2 exp(-2T/T2) = {r['absorbed']**2*np.exp(-2*20/200):.4f}")

# (c) efficiency vs C
T, T2 = 20.0, 200.0
Cc = np.logspace(-1.3, 1.3, 300)
ax[1, 0].semilogx(Cc, (4 * Cc / (1 + Cc)**2)**2 * np.exp(-2 * T / T2), "k:",
                  label="[4C/(1+C)²]² e^{-2T/T₂}")
Cs = [0.1, 0.2, 0.5, 1, 2, 5, 10]
res = [two_pi(C, T, T2) for C in Cs]
ax[1, 0].semilogx(Cs, [q["eta"] for q in res], "o", label="time domain")
ax[1, 0].semilogx(Cs, [q["absorbed"]**2 * np.exp(-2 * T / T2) for q in res], "x", color="C1",
                  label="A² e^{-2T/T₂}, A = measured absorption")
ax[1, 0].set(xlabel="cooperativity C", ylabel="memory efficiency η",
             title=f"(c) η vs C   (T = {T:.0f} µs, T₂ = {T2:.0f} µs)", ylim=(0, 1))
ax[1, 0].legend(fontsize=8)
for C, q in zip(Cs, res):
    print(f"C = {C:>4}: eta = {q['eta']:.4f}, A^2 e^-2T/T2 = {q['absorbed']**2*np.exp(-2*T/T2):.4f}")

# (d) efficiency vs storage time
Ts = np.array([20, 30, 40, 50])   # below ~17 us the 2nd pi pulse would hit the 1st echo
Tl = np.linspace(17, 52, 100)
for T2, col in [(50, "C3"), (100, "C1"), (200, "C0")]:
    qs = [two_pi(1.0, Ti, T2) for Ti in Ts]
    A2 = np.mean([q["absorbed"] for q in qs])**2
    ax[1, 1].plot(Tl, A2 * np.exp(-2 * Tl / T2), color=col, lw=1)
    ax[1, 1].plot(Ts, [q["eta"] for q in qs], "o", color=col, label=f"T₂ = {T2} µs")
ax[1, 1].set(xlabel="storage time T (µs)", ylabel="memory efficiency η",
             title="(d) C = 1: points = simulation, lines = A² e^{-2T/T₂}", ylim=(0, 1))
ax[1, 1].legend(fontsize=8)

fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/step03_echo.png", dpi=150)
