"""Step 2 — absorption of a microwave pulse by the spin ensemble, in the time domain.

Figures:
 (a) input / reflected pulse and energy transferred to the spins, at C = 1 and C = 10
 (b) absorbed fraction vs C for a long pulse: time domain vs 4C/(1+C)^2
 (c) absorbed fraction vs pulse duration at C = 1: time domain vs a frequency-domain
     prediction (pulse spectrum x steady-state absorption from step 1)
 (d) the discretisation trap: a coarse spin grid rephases by itself after 1/dx
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from spinmem import (discretise_gaussian, g_for_C_gaussian, simulate,
                     W_gaussian, reflection)

# Toy parameters (MHz, us)
kappa, kappa_int = 1.0, 0.0
Gamma_inh, gamma = 4.0, 0.01
x, rho = discretise_gaussian(Gamma_inh, x_max=12, dx=0.02)   # recurrence at 1/dx = 50 us


def gaussian_pulse(tau, t0):
    """Normalised pulse: |a_in|^2 has FWHM tau and integrates to 1 (one photon)."""
    A = (tau * np.sqrt(np.pi / (4 * np.log(2))))**-0.5
    return lambda t: A * np.exp(-2 * np.log(2) * (t - t0)**2 / tau**2)


def run(C, tau, grid=(x, rho), t_extra=3.0, n=3000):
    xx, rr = grid
    g = g_for_C_gaussian(C, kappa, Gamma_inh, gamma)
    t0 = 3 * tau
    t = np.linspace(0, t0 + 3 * tau + t_extra, n)
    a_in = gaussian_pulse(tau, t0)
    a, a_out, Es, _ = simulate(t, a_in, xx, rr, g, kappa - kappa_int, kappa_int, gamma)
    absorbed = 1 - np.trapezoid(np.abs(a_out)**2, t)
    return t, a_in(t), a, a_out, Es, absorbed


def absorbed_freq_domain(C, tau):
    """Same quantity from step 1: weight 1 - |r(nu)|^2 by the pulse power spectrum."""
    g = g_for_C_gaussian(C, kappa, Gamma_inh, gamma)
    nu = np.linspace(-40, 40, 40001)
    sigma_t = tau / (2 * np.sqrt(2 * np.log(2))) * np.sqrt(2)   # std of a_in(t)
    S = np.exp(-(2 * np.pi * nu * sigma_t)**2)                   # |FT a_in|^2
    # Use the continuous line here: on the discrete grid, gamma < dx and the steady
    # state would see a comb of separate narrow lines instead of a continuum.
    r = reflection(-nu, W_gaussian(-nu, g, Gamma_inh, gamma), kappa - kappa_int, kappa_int)
    return np.trapezoid(S * (1 - np.abs(r)**2), nu) / np.trapezoid(S, nu)


fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))

# (a) time traces
axE = ax[0, 0].twinx()
for C, ls in [(1, "-"), (10, "--")]:
    t, ain, a, aout, Es, A = run(C, tau=2.0)
    if C == 1:
        ax[0, 0].plot(t, np.abs(ain)**2, "k", lw=1, label="|a_in|²")
        # energy balance: E_in - E_out = E_spins(T) + |a(T)|^2 + gamma-loss + kappa_int-loss
        w = 2 * np.pi
        lost = np.trapezoid(w * gamma * Es + w * kappa_int * np.abs(a)**2, t)
        balance = A - (Es[-1] + abs(a[-1])**2 + lost)
        print(f"energy balance residual (C=1): {balance:.1e}")
    ax[0, 0].plot(t, np.abs(aout)**2, ls, color="C0", label=f"|a_out|², C={C}")
    axE.plot(t, Es, ls, color="C1", label=f"energy in spins, C={C}")
ax[0, 0].set(xlabel="time (µs)", ylabel="power (photons / µs)",
             title="(a) a 2 µs, one-photon pulse: C = 1 vs C = 10")
axE.set_ylabel("energy in the spins (photons)", color="C1")
axE.set_ylim(0, 1.05)
h1, l1 = ax[0, 0].get_legend_handles_labels()
h2, l2 = axE.get_legend_handles_labels()
ax[0, 0].legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")

# (b) absorption vs C, long pulse
Cs_curve = np.logspace(-1.5, 1.5, 300)
ax[0, 1].semilogx(Cs_curve, 4 * Cs_curve / (1 + Cs_curve)**2, "k:", label="4C/(1+C)²")
Cs = [0.1, 0.3, 1, 3, 10]
A_td = []
for C in Cs:
    A_td.append(run(C, tau=3.0)[-1])
    print(f"C = {C:>4}: absorbed (time domain, 3 us pulse) = {A_td[-1]:.4f},"
          f"  4C/(1+C)^2 = {4*C/(1+C)**2:.4f}")
ax[0, 1].semilogx(Cs, A_td, "o", label="time domain, 3 µs pulse")
ax[0, 1].set(xlabel="cooperativity C", ylabel="absorbed fraction",
             title="(b) long pulse: impedance matching", ylim=(0, 1.05))
ax[0, 1].legend()

# (c) absorption vs pulse duration at C = 1
taus = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2])
A_t = np.array([run(1, tau)[-1] for tau in taus])
taus_f = np.logspace(np.log10(0.03), np.log10(5), 60)
A_f = [absorbed_freq_domain(1, tau) for tau in taus_f]
ax[1, 0].semilogx(taus_f, A_f, "-", label="frequency domain (step 1)")
ax[1, 0].semilogx(taus, A_t, "o", label="time domain")
ax[1, 0].set(xlabel="pulse duration, FWHM (µs)", ylabel="absorbed fraction",
             title="(c) C = 1: the pulse must be slower than κ", ylim=(0, 1.05))
ax[1, 0].legend()
for tau, At in zip(taus, A_t):
    print(f"tau = {tau:>4} us: time domain {At:.4f}, frequency domain {absorbed_freq_domain(1, tau):.4f}")

# (d) discretisation trap
coarse = discretise_gaussian(Gamma_inh, x_max=12, dx=0.25)   # recurrence at 4 us
for grid, lab in [(coarse, "coarse grid, dx = 0.25 MHz"), ((x, rho), "fine grid, dx = 0.02 MHz")]:
    t, ain, a, aout, Es, A = run(1, tau=0.5, grid=grid, t_extra=12, n=6000)
    ax[1, 1].semilogy(t, np.abs(aout)**2 + 1e-12, label=lab)
ax[1, 1].set(xlabel="time (µs)", ylabel="|a_out|²", ylim=(1e-8, 10),
             title="(d) coarse grid: spurious echoes every 1/dx = 4 µs")
ax[1, 1].legend()

fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/step02_pulse.png", dpi=150)
