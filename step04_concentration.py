"""Step 4 — the concentration / coherence trade-off, and why clock transitions help.

Toy scalings with the spin concentration n (ppm). They are ILLUSTRATIVE, chosen to
show the mechanism, not fitted to any real material:

    g_ens^2  = F * g1^2 * n          (more spins in the mode; F = coupling enhancement,
                                      e.g. better filling factor from the deposition)
    Gamma_inh = Gamma_0 + a * n      (static dipolar broadening by the other spins)
    1/T2      = 1/T2_0  + b * n      (instantaneous + spectral diffusion)

A clock transition is first-order insensitive to magnetic fluctuations, so a and b are
strongly reduced (here by a factor 100, arbitrary).

Since W(0) ~ g_ens^2 / Gamma_inh, C saturates at large n:
    C_max = C(n -> infinity), independent of n.
If C_max < 1, impedance matching is impossible at any concentration.

Memory efficiency (validated in step 3):  eta = [4C/(1+C)^2]^2 exp(-2T/T2).

Figures:
 (a) C(n) for both transitions, with their saturation values
 (b) eta(n) at T = 20 us, with a few full time-domain simulations (step 3 protocol)
 (c) best eta (optimised over n) vs coupling enhancement F
 (d) best eta vs storage time
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from spinmem import (W_gaussian, gamma_from_T2, discretise_gaussian, run_sequence)

kappa = 1.0                   # MHz, fixed resonator linewidth
g1 = 0.3                      # MHz, ensemble coupling at n = 1 ppm (F = 1)
Gamma0, T2_0 = 1.0, 1000.0    # MHz, us: low-concentration limits
transitions = {
    "magnetic transition": dict(a=1.0, b=1e-2, color="C3"),    # a: MHz/ppm, b: 1/us/ppm
    "clock transition":    dict(a=1e-2, b=1e-4, color="C0"),
}
n = np.logspace(-2, 4, 600)   # ppm


def params(n, a, b, F=1.0):
    g = np.sqrt(F * g1**2 * n)
    Gi = Gamma0 + a * n
    T2 = 1 / (1 / T2_0 + b * n)
    return g, Gi, T2


def coop(n, a, b, F=1.0):
    g, Gi, T2 = params(n, a, b, F)
    return 2 * W_gaussian(0.0, g, Gi, gamma_from_T2(T2)).real / kappa


def eta(n, a, b, T, F=1.0):
    C = coop(n, a, b, F)
    T2 = params(n, a, b, F)[2]
    return (4 * C / (1 + C)**2)**2 * np.exp(-2 * T / T2)


def C_max(a, F=1.0):
    # n -> infinity: Gamma_inh ~ a n, gamma negligible; Gaussian W(0) = g^2 sqrt(pi/2)/s
    s_per_n = a / (2 * np.sqrt(2 * np.log(2)))
    return 2 * F * g1**2 * np.sqrt(np.pi / 2) / s_per_n / kappa


# ---- full time-domain check (two-pi protocol of step 3) ------------------
def simulate_eta(n_ppm, a, b, T=20.0, tau_p=1.5, t_gap=5.0):
    """Full simulation; returns the retrieved energy. t_gap = t_pi1 - t0."""
    g, Gi, T2 = params(n_ppm, a, b)
    gamma = gamma_from_T2(T2)
    x, rho = discretise_gaussian(Gi, x_max=max(12, 3 * Gi), dx=0.02)
    t0 = 3 * tau_p
    A = (tau_p * np.sqrt(np.pi / (4 * np.log(2))))**-0.5
    a_in = lambda t: A * np.exp(-2 * np.log(2) * (t - t0)**2 / tau_p**2)
    t1, t2, tout = t0 + t_gap, t0 + t_gap + T / 2, t0 + T
    segs = [dict(t0=0, t1=t1, sz=-1, delta_c=0),
            dict(t0=t1, t1=t2, sz=+1, delta_c=max(20, 4 * Gi), pi_before=True),
            dict(t0=t2, t1=tout + 5, sz=-1, delta_c=0, pi_before=True)]
    segs[-1]["t1"] = tout + 3 * tau_p
    t, _, ao, *_ = run_sequence(segs, a_in, x, rho, g, kappa, 0, gamma)
    m = t >= t2
    return np.trapezoid(np.abs(ao[m])**2, t[m])


fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))
T = 20.0

for name, p in transitions.items():
    a, b, col = p["a"], p["b"], p["color"]
    C = coop(n, a, b)
    Cm = C_max(a)
    ax[0, 0].loglog(n, C, color=col, label=f"{name}  (C_max = {Cm:.2g})")
    ax[0, 0].axhline(Cm, color=col, ls=":", lw=0.8)

    e = eta(n, a, b, T)
    i = np.argmax(e)
    ax[0, 1].semilogx(n, e, color=col, label=f"{name.split()[0]}: η_max = {e[i]:.2f}\n  at n ≈ {n[i]:.2g} ppm")
    ax[0, 1].plot(n[i], e[i], "*", color=col, ms=10)
    print(f"{name}: C_max = {Cm:.3f}, eta_max = {e[i]:.3f} at n = {n[i]:.3g} ppm, "
          f"C there = {C[i]:.2f}, T2 there = {params(n[i], a, b)[2]:.0f} us")

    # time-domain points
    n_chk = [0.3, 3, 30] if name.startswith("clock") else [0.3, 1, 3]
    for nc in n_chk:
        es = simulate_eta(nc, a, b, T)
        ea = eta(nc, a, b, T)
        print(f"   n = {nc:>5} ppm: simulated eta = {es:.4f}, formula = {ea:.4f}")
        ax[0, 1].plot(nc, es, "o", mfc="none", color=col)

    Fs = np.logspace(-1, 2, 120)
    ax[1, 0].semilogx(Fs, [eta(n, a, b, T, F).max() for F in Fs], color=col, label=name)

    Ts = np.linspace(1, 200, 200)
    ax[1, 1].plot(Ts, [eta(n, a, b, Ti).max() for Ti in Ts], color=col, label=name)

ax[0, 0].axhline(1, color="k", lw=0.8)
ax[0, 0].text(1.5e-2, 1.15, "C = 1 (impedance matching)", fontsize=8)
ax[0, 0].set(xlabel="spin concentration n (ppm)", ylabel="cooperativity C",
             title="(a) C saturates when Γ_inh ∝ n", ylim=(1e-2, 1e2))
ax[0, 0].legend(fontsize=8, loc="lower right")

ax[0, 1].plot([], [], "ko", mfc="none", label="full simulation,\n  1.5 µs pulse")
ax[0, 1].set(xlabel="spin concentration n (ppm)", ylabel="memory efficiency η",
             title=f"(b) η(n), storage time T = {T:.0f} µs", ylim=(0, 1))
ax[0, 1].legend(fontsize=8, loc="center right")

ax[1, 0].axvline(1, color="gray", ls=":", lw=0.8)
ax[1, 0].set(xlabel="coupling enhancement F (g₁² → F g₁²)", ylabel="best η over n",
             title=f"(c) what a stronger spin-resonator coupling buys (T = {T:.0f} µs)", ylim=(0, 1))
ax[1, 0].legend(fontsize=8)

ax[1, 1].set(xlabel="storage time T (µs)", ylabel="best η over n",
             title="(d) best efficiency vs storage time", ylim=(0, 1))
ax[1, 1].legend(fontsize=8)

fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/step04_concentration.png", dpi=150)

# The formula assumes slow pulses. With 1.5 us pulses the simulated points differ by
# up to ~10 % (finite bandwidth). Check that the gap closes with a 5 us pulse:
print("slow-pulse check (tau_p = 5 us, T = 48 us):")
for name, p in transitions.items():
    for nc in [0.3, 3]:
        es = simulate_eta(nc, p["a"], p["b"], T=48.0, tau_p=5.0, t_gap=12.0)
        print(f"   {name}, n = {nc} ppm: simulated {es:.4f}, formula {eta(nc, p['a'], p['b'], 48.0):.4f}")
