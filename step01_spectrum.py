"""Step 1 — steady-state reflection spectrum of a resonator coupled to a spin ensemble.

Figures:
 (a) |r|^2 on resonance for increasing cooperativity C (Lorentzian spins)
 (b) fraction of an incoming resonant tone absorbed BY THE SPINS vs C,
     for several internal losses kappa_int
 (c) |r| map vs probe frequency and spin frequency: anticrossing at strong coupling
 (d) Lorentzian vs Gaussian inhomogeneous line at equal FWHM and equal g_ens,
     with a brute-force discretised check (dots)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from spinmem import (W_lorentzian, W_gaussian, W_numeric, reflection,
                     g_for_cooperativity)

# Toy parameters, in MHz (frequencies / 2pi)
kappa = 1.0      # total resonator linewidth
Gamma = 4.0      # spin linewidth (inhomogeneous + homogeneous)
gamma = 0.01     # homogeneous linewidth, i.e. T2 ~ 30 us; negligible here

nu = np.linspace(-8, 8, 4001)   # probe detuning from the resonator (MHz)
dc = -nu                         # omega_c - omega

fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))

# (a) cooperativity sweep, cavity and spins resonant, overcoupled port
for C in [0, 0.3, 1, 3, 10]:
    g = g_for_cooperativity(C, kappa, Gamma)
    r = reflection(dc, W_lorentzian(-nu, g, Gamma), kappa_ext=kappa, kappa_int=0)
    ax[0, 0].plot(nu, np.abs(r)**2, label=f"C = {C}")
ax[0, 0].set(xlabel="probe detuning (MHz)", ylabel="|r|²",
             title="(a) reflection vs cooperativity (κ_int = 0)", xlim=(-6, 6))
ax[0, 0].legend()

# (b) on-resonance absorption by the spins
Cs = np.logspace(-2, 2, 400)
for frac in [0, 0.1, 0.3]:
    k_int = frac * kappa
    k_ext = kappa - k_int
    g = g_for_cooperativity(Cs, kappa, Gamma)
    r0 = reflection(0.0, W_lorentzian(0.0, g, Gamma), k_ext, k_int)
    absorbed = 1 - np.abs(r0)**2
    to_spins = absorbed * (kappa * Cs) / (k_int + kappa * Cs)  # share of the dissipation that goes into the spins
    ax[0, 1].semilogx(Cs, to_spins, label=f"κ_int/κ = {frac}")
ax[0, 1].semilogx(Cs, 4 * Cs / (1 + Cs)**2, "k:", label="4C/(1+C)²")
ax[0, 1].set(xlabel="cooperativity C", ylabel="fraction absorbed by the spins",
             title="(b) impedance matching", ylim=(0, 1.05))
ax[0, 1].legend()

# (c) anticrossing: narrow Gaussian spins, strong coupling
g_strong, Gi = 3.0, 1.0
nus = np.linspace(-10, 10, 401)          # spin detuning from the resonator
NU, NUS = np.meshgrid(nu, nus)
# kappa_int > 0 here: with a lossless resonator the polaritons sit in the Gaussian tails,
# barely damped ("cavity protection"), and |r| ~ 1 everywhere — only the phase moves.
r = reflection(-NU, W_gaussian(NUS - NU, g_strong, Gi, gamma), kappa_ext=0.5, kappa_int=0.5)
ax[1, 0].pcolormesh(nu, nus, np.abs(r), shading="auto", cmap="viridis")
ax[1, 0].plot(nus, nus, "w--", lw=0.6)
ax[1, 0].set(xlabel="probe detuning (MHz)", ylabel="spin detuning (MHz)",
             title=f"(c) |r|, g_ens = {g_strong}, Γ_inh = {Gi}, κ_int = κ/2 (MHz)", xlim=(-8, 8))

# (d) Lorentzian vs Gaussian, same FWHM, same g_ens (C = 3 in the Lorentzian sense)
g = g_for_cooperativity(3, kappa, Gamma)
rL = reflection(dc, W_lorentzian(-nu, g, Gamma), kappa, 0)
rG = reflection(dc, W_gaussian(-nu, g, Gamma, 0.2), kappa, 0)  # gamma = 0.2 so the discretised grid resolves it
x = np.linspace(-40, 40, 8001)                       # discretised spin offsets
s = Gamma / (2 * np.sqrt(2 * np.log(2)))
rho = np.exp(-x**2 / (2 * s**2)) / (np.sqrt(2 * np.pi) * s)
nu_chk = nu[::100]
rN = reflection(-nu_chk, W_numeric(-nu_chk, g, rho, x, gamma=0.2), kappa, 0)
rG_chk = reflection(-nu_chk, W_gaussian(-nu_chk, g, Gamma, 0.2), kappa, 0)
ax[1, 1].plot(nu, np.abs(rL)**2, label="Lorentzian")
ax[1, 1].plot(nu, np.abs(rG)**2, label="Gaussian (Faddeeva, γ=0.2)")
ax[1, 1].plot(nu_chk, np.abs(rN)**2, "o", ms=4, mfc="none", label="Gaussian, discretised sum")
ax[1, 1].set(xlabel="probe detuning (MHz)", ylabel="|r|²",
             title="(d) shape of the inhomogeneous line matters", xlim=(-6, 6))
ax[1, 1].legend()

fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/step01_spectrum.png", dpi=150)

# ---- sanity checks -------------------------------------------------------
g1 = g_for_cooperativity(1, kappa, Gamma)
r_C1 = reflection(0.0, W_lorentzian(0.0, g1, Gamma), kappa, 0)
print(f"|r|^2 at resonance, C = 1, kappa_int = 0 : {abs(r_C1)**2:.2e}  (expect 0)")

for C in [0.3, 3, 10]:
    g = g_for_cooperativity(C, kappa, Gamma)
    r0 = reflection(0.0, W_lorentzian(0.0, g, Gamma), kappa, 0)
    print(f"C = {C:>4}: |r0|^2 = {abs(r0)**2:.4f}, ((C-1)/(C+1))^2 = {((C-1)/(C+1))**2:.4f}")

err = np.max(np.abs(rN - rG_chk))
print(f"max |r_discretised - r_Faddeeva| = {err:.1e}")

# splitting at zero spin detuning in panel (c): two dips separated by ~2 g_ens
row = np.abs(r[len(nus) // 2])
left, right = nu < 0, nu > 0
split = nu[right][np.argmin(row[right])] - nu[left][np.argmin(row[left])]
print(f"vacuum Rabi splitting: {split:.2f} MHz  (2 g_ens = {2*g_strong:.2f} MHz)")
