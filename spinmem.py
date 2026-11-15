"""Core functions for the spin-ensemble / resonator toy model.

All rates and detunings are ordinary frequencies (e.g. MHz); the 2*pi factors cancel
because the equations are linear in the rates. kappa, gamma, Gamma are FWHM-type
energy decay rates (as they appear in a Lorentzian |r|^2 dip).
"""
import numpy as np
from scipy.special import wofz


def W_lorentzian(delta_s, g_ens, Gamma):
    """Spin self-energy for a Lorentzian spin line (Gamma = Gamma_inh + gamma).

    delta_s = omega_s - omega (spin centre minus probe), array-like.
    """
    return g_ens**2 / (1j * delta_s + Gamma / 2)


def W_gaussian(delta_s, g_ens, Gamma_inh, gamma):
    """Spin self-energy for a Gaussian inhomogeneous line of FWHM Gamma_inh,
    each spin having homogeneous linewidth gamma (a Voigt profile).

    Uses the Faddeeva function: for a Gaussian of std s,
    int rho(x) dx / (gamma/2 + i(x + delta_s)) = sqrt(pi/2)/s * w(z),
    with z = (-delta_s + i gamma/2) / (sqrt(2) s).
    """
    s = Gamma_inh / (2 * np.sqrt(2 * np.log(2)))
    z = (-np.asarray(delta_s) + 1j * gamma / 2) / (np.sqrt(2) * s)
    return g_ens**2 * np.sqrt(np.pi / 2) / s * wofz(z)


def W_numeric(delta_s, g_ens, rho, x, gamma):
    """Brute-force self-energy: sum over a discretised spin distribution.

    rho : density sampled on grid x (normalised so that sum(rho)*dx = 1).
    This is what the time-domain simulation (step 2) will use, so it is worth
    checking it against the closed forms.
    """
    dx = x[1] - x[0]
    d = np.asarray(delta_s)[..., None]
    return g_ens**2 * np.sum(rho * dx / (1j * (x + d) + gamma / 2), axis=-1)


def reflection(detuning_c, W, kappa_ext, kappa_int):
    """Reflection coefficient of a single-port resonator.

    detuning_c = omega_c - omega ; W = spin self-energy at the same probe points.
    """
    kappa = kappa_ext + kappa_int
    return 1 - kappa_ext / (1j * detuning_c + kappa / 2 + W)


def cooperativity(g_ens, kappa, Gamma):
    return 4 * g_ens**2 / (kappa * Gamma)


def g_for_cooperativity(C, kappa, Gamma):
    return np.sqrt(C * kappa * Gamma / 4)


# ---------------------------------------------------------------------------
# Step 2: time domain
# ---------------------------------------------------------------------------
from scipy.integrate import solve_ivp


def discretise_gaussian(Gamma_inh, x_max, dx):
    """Spin offsets x (MHz) on a uniform grid and normalised Gaussian density rho.

    Beware: a grid of spacing dx rephases by itself after t_rec = 1/dx (us),
    producing a spurious echo. Keep every simulation shorter than t_rec.
    """
    x = np.arange(-x_max, x_max + dx / 2, dx)
    s = Gamma_inh / (2 * np.sqrt(2 * np.log(2)))
    rho = np.exp(-x**2 / (2 * s**2))
    rho /= rho.sum() * dx
    return x, rho


def g_for_C_gaussian(C, kappa, Gamma_inh, gamma):
    """g_ens giving cooperativity C for a Gaussian line, with C defined through the
    on-resonance self-energy: kappa/2 -> kappa/2 (1 + C), i.e. C = 2 W(0) / kappa.
    Then the on-resonance absorption is exactly 4C/(1+C)^2, as for a Lorentzian.
    """
    W1 = W_gaussian(0.0, 1.0, Gamma_inh, gamma).real
    return np.sqrt(C * kappa / (2 * W1))


def simulate(t, a_in, x, rho, g_ens, kappa_ext, kappa_int, gamma, b0=None):
    """Integrate the resonator + discretised spin ensemble in the frame rotating
    at the resonator frequency (spins centred on the resonator).

    Parameters are ordinary frequencies in MHz; t in us; a_in(t) in sqrt(photons/us).
    Internally everything is converted to angular rates (x 2 pi).

        da/dt   = -(kappa/2) a - i sum_j g_j b_j + sqrt(kappa_ext) a_in
        db_j/dt = -(i Delta_j + gamma/2) b_j - i g_j a,   g_j = g_ens sqrt(rho_j dx)

    Returns a(t), a_out(t), spin energy sum|b_j|^2 (t), and the final state.
    """
    w = 2 * np.pi
    k, kext = w * (kappa_ext + kappa_int), w * kappa_ext
    gam = w * gamma
    dx = x[1] - x[0]
    gj = w * g_ens * np.sqrt(rho * dx)
    Dj = w * x
    M = len(x)

    def rhs(tt, y):
        a, b = y[0], y[1:]
        da = -(k / 2) * a - 1j * np.dot(gj, b) + np.sqrt(kext) * a_in(tt)
        db = -(1j * Dj + gam / 2) * b - 1j * gj * a
        return np.concatenate(([da], db))

    y0 = np.zeros(M + 1, complex)
    if b0 is not None:
        y0[1:] = b0
    sol = solve_ivp(rhs, (t[0], t[-1]), y0, t_eval=t, rtol=1e-8, atol=1e-10,
                    max_step=0.02)
    a = sol.y[0]
    a_out = a_in(t) - np.sqrt(kext) * a
    E_spins = np.sum(np.abs(sol.y[1:])**2, axis=0)
    return a, a_out, E_spins, sol.y[:, -1]


# ---------------------------------------------------------------------------
# Step 3: inversion, pi pulses, tunable resonator
# ---------------------------------------------------------------------------
def gamma_from_T2(T2):
    """Homogeneous linewidth (MHz) from the coherence time T2 (us), defined so that
    a spin amplitude decays as exp(-t/T2): gamma_angular = 2/T2."""
    return 1.0 / (np.pi * T2)


def evolve(t, y0, a_in, x, rho, g_ens, kappa_ext, kappa_int, gamma,
           sz=-1, delta_c=0.0, max_step=0.02):
    """Like simulate(), but returns the full state and allows
       sz = -1 (spins in the ground state) or +1 (inverted, after a pi pulse),
       delta_c = resonator detuning from the spin line centre (MHz).

    From the Heisenberg equation d(sigma-)/dt = ... + i g sigma_z a, the sign of the
    spin->resonator back-action flips when the ensemble is inverted:
        db_j/dt = -(i Delta_j + gamma/2) b_j + i sz g_j a
    With sz = +1 the spins add gain instead of loss: kappa/2 (1 + C) -> kappa/2 (1 - C).
    """
    w = 2 * np.pi
    k, kext, gam, dc = w * (kappa_ext + kappa_int), w * kappa_ext, w * gamma, w * delta_c
    dx = x[1] - x[0]
    gj = w * g_ens * np.sqrt(rho * dx)
    Dj = w * x

    def rhs(tt, y):
        a, b = y[0], y[1:]
        da = -(1j * dc + k / 2) * a - 1j * np.dot(gj, b) + np.sqrt(kext) * a_in(tt)
        db = -(1j * Dj + gam / 2) * b + 1j * sz * gj * a
        return np.concatenate(([da], db))

    sol = solve_ivp(rhs, (t[0], t[-1]), np.asarray(y0, complex), t_eval=t,
                    rtol=1e-8, atol=1e-10, max_step=max_step)
    return sol.y


def pi_pulse(y):
    """Ideal, instantaneous pi pulse about x on every spin: sigma- -> sigma+,
    i.e. b_j -> conj(b_j). The resonator field is left untouched."""
    y = y.copy()
    y[1:] = np.conj(y[1:])
    return y


def run_sequence(segments, a_in, x, rho, g_ens, kappa_ext, kappa_int, gamma,
                 pts_per_us=200):
    """Chain evolution segments. Each segment is a dict with
         t0, t1           : start / end time (us)
         sz               : -1 or +1
         delta_c          : resonator detuning (MHz)
         pi_before (bool) : apply an ideal pi pulse at t0
    Returns t, a(t), a_out(t), spin-energy(t), sz(t), delta_c(t).
    """
    M = len(x)
    y = np.zeros(M + 1, complex)
    T, A, E, S, D = [], [], [], [], []
    for seg in segments:
        if seg.get("pi_before", False):
            y = pi_pulse(y)
        n = max(int((seg["t1"] - seg["t0"]) * pts_per_us), 10)
        t = np.linspace(seg["t0"], seg["t1"], n)
        Y = evolve(t, y, a_in, x, rho, g_ens, kappa_ext, kappa_int, gamma,
                   sz=seg["sz"], delta_c=seg["delta_c"])
        y = Y[:, -1]
        sl = slice(1, None) if T else slice(None)   # avoid duplicated boundary points
        T.append(t[sl]); A.append(Y[0, sl])
        E.append(np.sum(np.abs(Y[1:, sl])**2, axis=0))
        S.append(np.full(len(t[sl]), seg["sz"])); D.append(np.full(len(t[sl]), seg["delta_c"]))
    t, a = np.concatenate(T), np.concatenate(A)
    a_out = a_in(t) - np.sqrt(2 * np.pi * kappa_ext) * a
    return t, a, a_out, np.concatenate(E), np.concatenate(S), np.concatenate(D)
