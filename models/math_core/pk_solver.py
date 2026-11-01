"""Pharmacokinetic solvers — 1-compartment and structural 2-compartment models.

1-compartment models (from drug_parameters.json):
    Oral: C(t) = (F·Dose·ka) / (Vd·(ka-ke)) · (e^{-ke·t} - e^{-ka·t})
    IV:   C(t) = (Dose/Vd) · e^{-ke·t}
    IIV:  θ_i = θ_pop · exp(η_i), η_i ~ N(0, ω²)

Structural 2-compartment PK/PD (from pkpd_core.py canonical):
    dA_d/dt = -ka·A_d
    dA_c/dt = ka·A_d - CL/V1·A_c - Q/V1·A_c + Q/V2·A_p
    dA_p/dt = Q/V1·A_c - Q/V2·A_p
    dC_e/dt = ke0·(C - Ce)
    E(X) = Emax·X^γ / (EC50^γ + X^γ)
    dH/dt = h₀·exp(β_tox·ln(1+C/C50_tox))
"""
from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
from dataclasses import dataclass
from scipy.integrate import solve_ivp


@dataclass
class PKParams:
    ka_per_hr: float       # absorption rate constant (oral only)
    ke_per_hr: float       # elimination rate constant
    vd_l: float            # volume of distribution (total, not per-kg)
    bioavailability: float  # F
    dose_mg: float
    route: str             # "oral" or "iv"
    tau_hr: float = 24.0   # dosing interval


class PKSolver:
    """Deterministic PK concentration computations."""

    @staticmethod
    def single_dose_oral(p: PKParams, t: np.ndarray) -> np.ndarray:
        """Concentration-time curve after a single oral dose.
        C(t) = (F·Dose·ka) / (Vd·(ka-ke)) · (e^{-ke·t} - e^{-ka·t})
        """
        coeff = (p.bioavailability * p.dose_mg * p.ka_per_hr) / (
            p.vd_l * (p.ka_per_hr - p.ke_per_hr)
        )
        return coeff * (np.exp(-p.ke_per_hr * t) - np.exp(-p.ka_per_hr * t))

    @staticmethod
    def single_dose_iv(p: PKParams, t: np.ndarray) -> np.ndarray:
        """Concentration after IV bolus: C(t) = (Dose/Vd)·e^{-ke·t}"""
        return (p.dose_mg / p.vd_l) * np.exp(-p.ke_per_hr * t)

    @staticmethod
    def steady_state_trough_oral(p: PKParams) -> float:
        """Css_trough at steady state for oral dosing."""
        coeff = (p.bioavailability * p.dose_mg * p.ka_per_hr) / (
            p.vd_l * (p.ka_per_hr - p.ke_per_hr)
        )
        term_ke = np.exp(-p.ke_per_hr * p.tau_hr) / (1 - np.exp(-p.ke_per_hr * p.tau_hr))
        term_ka = np.exp(-p.ka_per_hr * p.tau_hr) / (1 - np.exp(-p.ka_per_hr * p.tau_hr))
        return float(coeff * (term_ke - term_ka))

    @staticmethod
    def steady_state_trough_iv(p: PKParams) -> float:
        """Css_trough at steady state for IV bolus."""
        return float(
            (p.dose_mg / p.vd_l) * np.exp(-p.ke_per_hr * p.tau_hr)
            / (1 - np.exp(-p.ke_per_hr * p.tau_hr))
        )

    @staticmethod
    def concentration_curve(p: PKParams, t: np.ndarray) -> np.ndarray:
        """Route-dispatched single-dose concentration."""
        if p.route == "iv":
            return PKSolver.single_dose_iv(p, t)
        return PKSolver.single_dose_oral(p, t)

    @staticmethod
    def steady_state_trough(p: PKParams) -> float:
        if p.route == "iv":
            return PKSolver.steady_state_trough_iv(p)
        return PKSolver.steady_state_trough_oral(p)

    @staticmethod
    def apply_iiv(
        pop_value: float, omega: float, rng: np.random.Generator | None = None,
    ) -> float:
        """Individual parameter: θ_i = θ_pop · exp(η), η ~ N(0, ω²)."""
        rng = rng or np.random.default_rng()
        eta = rng.normal(0, omega)
        return pop_value * np.exp(eta)

    @staticmethod
    def multi_dose_oral(p: PKParams, t: np.ndarray, n_doses: int) -> np.ndarray:
        """Superposition of n doses given at intervals τ."""
        C = np.zeros_like(t)
        for k in range(n_doses):
            t_shifted = t - k * p.tau_hr
            mask = t_shifted >= 0
            C[mask] += PKSolver.single_dose_oral(p, t_shifted[mask])
        return C


# ═══════════════════════════════════════════════════════════════════════
# Structural 2-compartment PK/PD (canonical from pkpd_core.py)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class StructuralPKPDParams:
    ka: float = 1.10
    cl: float = 4.50
    v1: float = 32.0
    q: float = 6.00
    v2: float = 58.0
    f: float = 0.72
    ke0: float = 0.35
    emax: float = 1.0
    ec50: float = 0.85
    gamma: float = 1.6
    absorption: Literal["oral", "iv"] = "oral"
    n_compartments: int = 2
    pd_link: Literal["direct", "effect_site"] = "effect_site"


@dataclass
class DoseEvent:
    time: float
    amount: float
    route: Literal["oral", "iv_bolus"] = "oral"


def sigmoid_emax(x: float | np.ndarray, emax: float, ec50: float,
                 gamma: float) -> float | np.ndarray:
    x = np.maximum(x, 0.0)
    xg = np.power(x, gamma)
    return emax * xg / (np.power(ec50, gamma) + xg)


def simulate_structural_pkpd(
    params: StructuralPKPDParams,
    doses: Sequence[DoseEvent],
    t_end: float,
    n_grid: int = 601,
) -> dict:
    oral = params.absorption == "oral"
    two_comp = params.n_compartments == 2
    effect_site = params.pd_link == "effect_site"
    p = params
    n_states = 1 + int(oral) + int(two_comp) + int(effect_site)
    SI_AD = 0 if oral else -1
    SI_AC = 1 if oral else 0
    SI_AP = SI_AC + 1 if two_comp else -1
    SI_CE = (SI_AP if two_comp else SI_AC) + 1 if effect_site else -1

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        dy = np.zeros_like(y)
        Ac = max(y[SI_AC], 0.0)
        C = Ac / p.v1
        inflow = 0.0
        if oral:
            Ad = max(y[SI_AD], 0.0)
            dy[SI_AD] = -p.ka * Ad
            inflow += p.ka * Ad
        clearance = p.cl * C
        dAc = inflow - clearance
        if two_comp:
            Ap = max(y[SI_AP], 0.0)
            k12 = p.q / p.v1
            k21 = p.q / p.v2
            dAc += -k12 * Ac + k21 * Ap
            dy[SI_AP] = k12 * Ac - k21 * Ap
        dy[SI_AC] = dAc
        if effect_site:
            Ce = y[SI_CE]
            dy[SI_CE] = p.ke0 * (C - Ce)
        return dy

    t_grid = np.linspace(0.0, t_end, n_grid)
    y0 = np.zeros(n_states)
    conc_arr = np.zeros(n_grid)
    effect_arr = np.zeros(n_grid)
    ce_arr = np.zeros(n_grid)

    dose_times = sorted([(d.time, d) for d in doses], key=lambda x: x[0])
    event_times = [d.time for d in doses] + [t_end]
    y = y0.copy()

    for seg_idx in range(len(dose_times)):
        dt, dose = dose_times[seg_idx]
        if dose.route == "oral" and oral:
            y[SI_AD] += dose.amount * p.f
        else:
            y[SI_AC] += dose.amount
        t_start = dt
        t_stop = dose_times[seg_idx + 1][0] if seg_idx + 1 < len(dose_times) else t_end
        if t_stop <= t_start:
            continue
        mask = (t_grid >= t_start) & (t_grid <= t_stop)
        t_seg = t_grid[mask]
        if len(t_seg) < 2:
            continue
        sol = solve_ivp(rhs, (t_seg[0], t_seg[-1]), y, t_eval=t_seg,
                        method="RK45", rtol=1e-6, atol=1e-9)
        for i, ti in enumerate(np.where(mask)[0]):
            state = sol.y[:, i]
            C = max(state[SI_AC], 0.0) / p.v1
            conc_arr[ti] = C
            driver = state[SI_CE] if effect_site and SI_CE >= 0 else C
            ce_arr[ti] = driver
            effect_arr[ti] = float(sigmoid_emax(driver, p.emax, p.ec50, p.gamma))
        y = sol.y[:, -1]

    return {
        "time": t_grid.tolist(),
        "concentration": conc_arr.tolist(),
        "effect_site": ce_arr.tolist(),
        "effect": effect_arr.tolist(),
    }


def toxicity_hazard(conc: np.ndarray, time: np.ndarray,
                    h0: float = 2e-4, beta_tox: float = 2.2,
                    c50_tox: float = 1.6) -> dict:
    conc = np.asarray(conc, dtype=float)
    time = np.asarray(time, dtype=float)
    rate = h0 * np.exp(beta_tox * np.log1p(conc / c50_tox))
    H = np.cumsum(rate * np.concatenate([[0], np.diff(time)]))
    p_dlt = 1.0 - np.exp(-H)
    return {
        "hazard_rate": rate.tolist(),
        "cumulative_hazard": H.tolist(),
        "p_dlt": p_dlt.tolist(),
    }
