#!/usr/bin/env python3
"""
Decline curve analysis and gas material balance.

Six decline models fitted in log-rate space and ranked by AICc, a bootstrap
P90/P50/P10 band under the reserves exceedance convention, flow-regime
diagnostics, WOR / Ershaghi X-plot water analysis, and gas material balance
(p/z, Havlena-Odeh, Fetkovich aquifer) with a consistency guard.

Units throughout
    t        months on production
    q        volume per day (bbl/d or Mcf/d)
    cum      volume (bbl or Mcf) -- rate integrated over days, not months
    Di, Dmin nominal decline, 1/month (use nom_from_eff() to convert from
             an effective annual percentage)
    P        psia
    Gp       MMscf

Use as a library
    from dca import fit, eur, bootstrap_eur, percentile, load_csv

    s = load_csv("data/ecmc-myers-19-11HA.csv")
    f = fit("hyperbolic", s.t, s.q)
    print(f.p, f.r2)
    print(eur("hyperbolic", f.p, qab=3).eur)

Use as a CLI
    python dca.py data/ecmc-myers-19-11HA.csv --qab 3
    python dca.py data/well-A-26-oil.csv --to 72 --qab 50 --plot fit.png
    python dca.py data/reservoir-B-3-pressure.csv --mb --temp 180 --sg 0.735

See docs/methods.md for the derivations and the reasoning behind each choice.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

__all__ = [
    "DPM", "MODELS", "Model", "FitResult", "EurResult", "Series",
    "nom_from_eff", "eff_from_nom",
    "fit", "fit_from", "eur", "cumulative", "aicc",
    "bootstrap_eur", "percentile",
    "loss_ratio_b", "loglog_slope", "find_plateau", "extrapolation_multiple",
    "wor_analysis", "chan_derivative",
    "z_factor", "material_balance", "fetkovich_fit",
    "load_csv", "load_pressure_csv",
]

DPM = 30.4375  # days per average month


# ----------------------------------------------------------------------------
# decline rate conversions
# ----------------------------------------------------------------------------

def nom_from_eff(d_eff: float) -> float:
    """Effective annual decline (fraction) -> nominal annual decline.

    "40% decline" means 40% *effective*, i.e. nominal 0.51/yr. Getting this
    backwards is a routine source of 15-20% errors.
    """
    if not 0 <= d_eff < 1:
        raise ValueError("effective decline must be in [0, 1)")
    return -math.log(1.0 - d_eff)


def eff_from_nom(d_nom: float) -> float:
    """Nominal decline -> effective decline over the same period."""
    return 1.0 - math.exp(-d_nom)


# ----------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Model:
    key: str
    name: str
    params: tuple                       # free parameter names, in order
    rate: Callable                      # (p: dict, t: array) -> array
    lo: Dict[str, float]
    hi: Dict[str, float]
    guess: Callable                     # (q0, decl) -> dict
    cum: Optional[Callable] = None      # closed-form cumulative, else Simpson
    note: str = ""


def _arr(t):
    return np.asarray(t, dtype=float)


def _q_exponential(p, t):
    return p["qi"] * np.exp(-p["Di"] * _arr(t))


def _q_harmonic(p, t):
    return p["qi"] / (1.0 + p["Di"] * _arr(t))


def _q_hyperbolic(p, t):
    t = _arr(t)
    b = p["b"]
    if b < 1e-6:
        return p["qi"] * np.exp(-p["Di"] * t)
    base = 1.0 + b * p["Di"] * t
    out = np.where(base <= 0, 0.0, p["qi"] * np.power(np.maximum(base, 1e-300), -1.0 / b))
    return out


def _q_modhyp(p, t):
    """Hyperbolic until the nominal decline falls to Dmin, exponential after.

    The switch time is solved rather than assumed, which is what makes the
    curve C1: Di/(1 + b*Di*ts) = Dmin  =>  ts = (Di/Dmin - 1)/(b*Di).
    """
    t = _arr(t)
    b, Dmin = p["b"], p["Dmin"]
    if b < 1e-6:
        return p["qi"] * np.exp(-p["Di"] * t)
    ts = (p["Di"] / Dmin - 1.0) / (b * p["Di"])
    if not ts > 0:                       # already below terminal decline at t=0
        return p["qi"] * np.exp(-Dmin * t)
    qs = p["qi"] * (1.0 + b * p["Di"] * ts) ** (-1.0 / b)
    hyp = p["qi"] * np.power(np.maximum(1.0 + b * p["Di"] * t, 1e-300), -1.0 / b)
    exp = qs * np.exp(-Dmin * np.maximum(t - ts, 0.0))
    return np.where(t <= ts, hyp, exp)


def _q_duong(p, t):
    tt = np.maximum(_arr(t), 1e-4)
    m, a = p["m"], p["a"]
    return p["q1"] * np.power(tt, -m) * np.exp((a / (1.0 - m)) * (np.power(tt, 1.0 - m) - 1.0))


def _q_ple(p, t):
    tt = np.maximum(_arr(t), 0.0)
    return p["qhi"] * np.exp(-p["Dinf"] * tt - p["D1"] * np.power(tt, p["n"]))


def _q_sepd(p, t):
    tt = np.maximum(_arr(t), 0.0)
    return p["qi"] * np.exp(-np.power(tt / p["tau"], p["n"]))


def _cum_exponential(p, t):
    return DPM * (p["qi"] - float(_q_exponential(p, t))) / p["Di"]


def _cum_harmonic(p, t):
    return DPM * (p["qi"] / p["Di"]) * math.log(1.0 + p["Di"] * t)


def _cum_hyperbolic(p, t):
    b = p["b"]
    if b < 1e-6 or abs(b - 1.0) < 1e-6:
        return None                     # degenerate; fall through to Simpson
    q = float(_q_hyperbolic(p, t))
    return DPM * (p["qi"] / (p["Di"] * (1.0 - b))) * (1.0 - (q / p["qi"]) ** (1.0 - b))


def _cum_duong(p, t):
    tt = max(t, 1e-4)
    m, a = p["m"], p["a"]
    return DPM * (p["q1"] / a) * math.exp((a / (1.0 - m)) * (tt ** (1.0 - m) - 1.0))


MODELS: Dict[str, Model] = {
    "exponential": Model(
        key="exponential", name="Arps exponential", params=("qi", "Di"),
        rate=_q_exponential, cum=_cum_exponential,
        lo={"qi": 1e-6, "Di": 1e-6}, hi={"qi": 1e9, "Di": 3.0},
        guess=lambda q0, decl: {"qi": q0, "Di": decl},
        note="Arps b = 0; finite EUR without any terminal-decline assumption.",
    ),
    "harmonic": Model(
        key="harmonic", name="Arps harmonic (b=1)", params=("qi", "Di"),
        rate=_q_harmonic, cum=_cum_harmonic,
        lo={"qi": 1e-6, "Di": 1e-6}, hi={"qi": 1e9, "Di": 3.0},
        guess=lambda q0, decl: {"qi": q0, "Di": decl},
        note="Arps b = 1; cumulative diverges logarithmically, so EUR needs a limit.",
    ),
    "hyperbolic": Model(
        key="hyperbolic", name="Arps hyperbolic", params=("qi", "Di", "b"),
        rate=_q_hyperbolic, cum=_cum_hyperbolic,
        lo={"qi": 1e-6, "Di": 1e-6, "b": 0.0}, hi={"qi": 1e9, "Di": 3.0, "b": 2.0},
        guess=lambda q0, decl: {"qi": q0, "Di": decl, "b": 0.9},
        note="EUR is infinite for b >= 1; the abandonment rate is doing the work.",
    ),
    "modhyp": Model(
        key="modhyp", name="Modified hyperbolic", params=("qi", "Di", "b"),
        rate=_q_modhyp, cum=None,
        lo={"qi": 1e-6, "Di": 1e-6, "b": 0.0}, hi={"qi": 1e9, "Di": 3.0, "b": 2.0},
        guess=lambda q0, decl: {"qi": q0, "Di": decl, "b": 1.1},
        note="Dmin is a policy choice held fixed during the fit, not an estimate.",
    ),
    "duong": Model(
        key="duong", name="Duong", params=("q1", "a", "m"),
        rate=_q_duong, cum=_cum_duong,
        lo={"q1": 1e-6, "a": 1e-4, "m": 1.0001}, hi={"q1": 1e9, "a": 20.0, "m": 4.0},
        guess=lambda q0, decl: {"q1": q0, "a": 1.0, "m": 1.2},
        note="Fracture-dominated flow; m -> 1 means the assumption has expired.",
    ),
    "ple": Model(
        key="ple", name="Power-law exponential", params=("qhi", "Dinf", "D1", "n"),
        rate=_q_ple, cum=None,
        lo={"qhi": 1e-6, "Dinf": 0.0, "D1": 1e-6, "n": 0.05},
        hi={"qhi": 1e9, "Dinf": 0.05, "D1": 10.0, "n": 1.5},
        guess=lambda q0, decl: {"qhi": q0, "Dinf": 1e-4, "D1": 0.3, "n": 0.5},
        note="Dinf supplies a terminal decline from the functional form itself.",
    ),
    "sepd": Model(
        key="sepd", name="Stretched exponential", params=("qi", "tau", "n"),
        rate=_q_sepd, cum=None,
        lo={"qi": 1e-6, "tau": 1e-3, "n": 0.05}, hi={"qi": 1e9, "tau": 1e5, "n": 2.0},
        guess=lambda q0, decl: {"qi": q0, "tau": 6.0, "n": 0.5},
        note="Bounded EUR by construction; systematically the conservative case.",
    ),
}


def rate(key: str, p: dict, t) -> np.ndarray:
    """Rate at time t (months) for model `key` with parameters `p`."""
    return MODELS[key].rate(p, t)


def _simpson(f: Callable, a: float, b: float, n: int = 600) -> float:
    if n % 2:
        n += 1
    x = np.linspace(a, b, n + 1)
    y = np.asarray(f(x), dtype=float)
    w = np.ones(n + 1)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return float(np.sum(w * y) * (b - a) / (3.0 * n))


def cumulative(key: str, p: dict, t: float) -> float:
    """Cumulative volume from 0 to t months.

    Closed form for exponential, harmonic, hyperbolic and Duong; composite
    Simpson for PLE and SEPD, which have no elementary integral.
    """
    if t <= 0:
        return 0.0
    m = MODELS[key]
    if m.cum is not None:
        v = m.cum(p, t)
        if v is not None and math.isfinite(v):
            return float(v)
    return DPM * _simpson(lambda x: m.rate(p, x), 0.0, float(t))


def cum_between(key: str, p: dict, t0: float, t1: float) -> float:
    """Volume produced between t0 and t1 months."""
    if t1 <= t0:
        return 0.0
    return DPM * _simpson(lambda x: MODELS[key].rate(p, x), float(t0), float(t1), 400)


# ----------------------------------------------------------------------------
# fitting
# ----------------------------------------------------------------------------

@dataclass
class FitResult:
    key: str
    p: Dict[str, float]
    sse: float                  # sum of squared residuals of ln q
    rmse_log: float
    r2: float                   # in log space
    n: int
    k: int                      # free parameters
    aicc: float

    def rate(self, t) -> np.ndarray:
        return MODELS[self.key].rate(self.p, t)

    def __repr__(self) -> str:
        ps = " ".join(f"{k}={v:.4g}" for k, v in self.p.items())
        return f"<FitResult {self.key} {ps} R2={self.r2:.4f} AICc={self.aicc:.1f}>"


def _clamp(v, lo, hi):
    return min(max(v, lo), hi)


def _to_z(p: dict, m: Model) -> np.ndarray:
    """Logit-transform bounded parameters onto the whole real line.

    Bounds then hold by construction: no infeasible trial point, no penalty
    term, and no discontinuity in the objective at a bound -- which matters,
    because b often sits near one.
    """
    z = []
    for k in m.params:
        v = _clamp(p[k], m.lo[k], m.hi[k])
        z.append(math.log((v - m.lo[k] + 1e-12) / (m.hi[k] - v + 1e-12)))
    return np.array(z)


def _from_z(z: Sequence[float], m: Model) -> dict:
    out = {}
    for i, k in enumerate(m.params):
        e = 1.0 / (1.0 + math.exp(-float(np.clip(z[i], -700, 700))))
        out[k] = m.lo[k] + e * (m.hi[k] - m.lo[k])
    return out


def _objective(m: Model, t: np.ndarray, lq: np.ndarray, dmin: Optional[float]):
    """SSE of ln q.

    Equal *proportional* weighting: a 10% miss at 50 bbl/d counts the same as a
    10% miss at 5000. Least squares in rate space would let the first quarter
    set a thirty-year forecast.
    """
    def f(z):
        p = _from_z(z, m)
        if dmin is not None:
            p["Dmin"] = dmin
        with np.errstate(all="ignore"):
            qm = m.rate(p, t)
            if not np.all(np.isfinite(qm)) or np.any(qm <= 0):
                return 1e12
            r = lq - np.log(qm)
            s = float(np.dot(r, r))
        return s if math.isfinite(s) else 1e12
    return f


_SHAPE_KEY = {"hyperbolic": "b", "modhyp": "b", "duong": "m", "ple": "n", "sepd": "n"}
_SHAPE_GRID = {
    "hyperbolic": (0.3, 0.7, 1.1, 1.6),
    "modhyp": (0.3, 0.7, 1.1, 1.6),
    "duong": (1.05, 1.2, 1.4, 1.8),
    "ple": (0.2, 0.4, 0.7, 1.0),
    "sepd": (0.2, 0.4, 0.7, 1.0),
}


def _starts(m: Model, q0: float, decl: float) -> List[dict]:
    g = m.guess(q0, decl)
    out = [g]
    shape = _SHAPE_KEY.get(m.key)
    if shape:
        for v in _SHAPE_GRID[m.key]:
            for dm in (0.5, 1.0, 2.0):
                s = dict(g)
                s[shape] = v
                if "Di" in s:
                    s["Di"] = _clamp(decl * dm, 1e-5, 2.9)
                if "tau" in s:
                    s["tau"] = 3.0 * dm
                if "D1" in s:
                    s["D1"] = _clamp(decl * dm, 1e-5, 9.0)
                out.append(s)
    else:
        for dm in (0.5, 1.0, 2.0):
            s = dict(g)
            s["Di"] = _clamp(decl * dm, 1e-5, 2.9)
            out.append(s)
    return out


def fit(key: str, t, q, dmin: Optional[float] = None) -> FitResult:
    """Fit one model to (t, q) by minimising the SSE of ln q.

    Nelder-Mead on logit-transformed parameters, multi-started over the shape
    parameter (b, m or n) because the objective is *not* convex in it -- a
    single start from a plausible b lands in a wrong local basin often enough
    on real data to matter.

    dmin: terminal nominal decline, 1/month, for the modified hyperbolic only.
          Held FIXED during the fit; the history has not reached it and so
          cannot identify it. Defaults to 6%/yr effective.
    """
    if key not in MODELS:
        raise KeyError(f"unknown model {key!r}; have {sorted(MODELS)}")
    m = MODELS[key]
    t = _arr(t)
    q = _arr(q)
    if t.size != q.size:
        raise ValueError("t and q must be the same length")
    if t.size < len(m.params) + 1:
        raise ValueError(f"{key} needs at least {len(m.params) + 1} points, got {t.size}")
    if np.any(q <= 0):
        raise ValueError("all rates must be positive -- drop zero months, the date gaps carry them")

    if key == "modhyp" and dmin is None:
        dmin = nom_from_eff(0.06) / 12.0

    n = t.size
    q0 = float(np.max(q))
    span = max(float(t[-1] - t[0]), 1e-3)
    decl = _clamp(math.log(max(q[0], 1e-9) / max(q[-1], 1e-9)) / span, 0.005, 0.6)
    lq = np.log(np.maximum(q, 1e-9))

    obj = _objective(m, t, lq, dmin if key == "modhyp" else None)

    best_z, best_f = None, math.inf
    for s in _starts(m, q0, decl):
        z0 = _to_z(s, m)
        for maxiter in (2000, 800):     # second pass restarts a collapsed simplex
            r = minimize(obj, z0, method="Nelder-Mead",
                         options={"maxiter": maxiter, "xatol": 1e-8, "fatol": 1e-12})
            z0 = r.x
        if r.fun < best_f:
            best_f, best_z = float(r.fun), r.x

    p = _from_z(best_z, m)
    if key == "modhyp":
        p["Dmin"] = dmin
    sst = float(np.sum((lq - lq.mean()) ** 2))
    k = len(m.params)
    return FitResult(
        key=key, p=p, sse=best_f, rmse_log=math.sqrt(best_f / n),
        r2=(1.0 - best_f / sst) if sst > 0 else 0.0,
        n=n, k=k, aicc=aicc(best_f, n, k),
    )


def fit_from(key: str, t, q, p0: dict, dmin: Optional[float] = None) -> dict:
    """Refit seeded from a known solution, no multi-start.

    This is the bootstrap's inner loop. A resample of the same data will not
    land in a distant basin, so the multi-start is wasted work there -- and
    skipping it is the difference between 0.4 s and 30 s for 300 replicates.
    """
    m = MODELS[key]
    t, q = _arr(t), _arr(q)
    lq = np.log(np.maximum(q, 1e-9))
    if key == "modhyp" and dmin is None:
        dmin = p0.get("Dmin", nom_from_eff(0.06) / 12.0)
    obj = _objective(m, t, lq, dmin if key == "modhyp" else None)
    r = minimize(obj, _to_z(p0, m), method="Nelder-Mead",
                 options={"maxiter": 900, "xatol": 1e-7, "fatol": 1e-10})
    p = _from_z(r.x, m)
    if key == "modhyp":
        p["Dmin"] = dmin
    return p


def aicc(sse: float, n: int, k: int) -> float:
    """Small-sample-corrected Akaike information criterion.

    Ranks fit to the HISTORY. It has no view on extrapolation: two models
    within dAICc < 2 can differ by a factor of two in EUR. The correction
    matters here because PLE has four parameters and monthly histories are
    often 20-40 points.
    """
    a = n * math.log(sse / n) + 2 * k
    return a + 2 * k * (k + 1) / (n - k - 1) if (n - k - 1) > 0 else a


# ----------------------------------------------------------------------------
# EUR
# ----------------------------------------------------------------------------

@dataclass
class EurResult:
    eur: float
    t_end: float                # months
    q_end: float

    @property
    def years(self) -> float:
        return self.t_end / 12.0


def eur(key: str, p: dict, qab: float, max_years: float = 50.0) -> EurResult:
    """Integrate to the economic limit `qab` (per day) or the horizon.

    For b >= 1 the hyperbolic integral does not converge, so the horizon cap is
    what makes the number finite. That truncation is doing real work -- report
    it rather than hiding it.
    """
    m = MODELS[key]
    tmax = max_years * 12.0
    if float(m.rate(p, tmax)) > qab:
        t_end = tmax
    else:
        lo, hi = 0.0, tmax
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if float(m.rate(p, mid)) > qab:
                lo = mid
            else:
                hi = mid
        t_end = hi
    return EurResult(eur=cumulative(key, p, t_end), t_end=t_end,
                     q_end=float(m.rate(p, t_end)))


# ----------------------------------------------------------------------------
# probabilistic EUR
# ----------------------------------------------------------------------------

def bootstrap_eur(key: str, t, q, base: dict, qab: float, n: int = 300,
                  max_years: float = 50.0, dmin: Optional[float] = None,
                  seed: Optional[int] = None) -> np.ndarray:
    """Bootstrap the log-residuals of `base` into an EUR distribution.

    Residuals are resampled, NOT the (t, q) pairs -- resampling pairs breaks
    the time ordering, which is exactly the information a decline model uses.

    Returns a sorted array. Pass it to percentile(); note the exceedance
    convention (see that function).

    What this covers: sampling uncertainty in the fit, given THIS model and
    THIS window. What it does not: model uncertainty (usually larger -- compare
    the six EURs), window choice (often larger still), and anything
    operational. Treat it as a floor on uncertainty, not a total.
    """
    m = MODELS[key]
    t, q = _arr(t), _arr(q)
    rng = np.random.default_rng(seed)
    qb = m.rate(base, t)
    res = np.log(np.maximum(q, 1e-9)) - np.log(np.maximum(qb, 1e-9))

    out = []
    for _ in range(n):
        qs = qb * np.exp(rng.choice(res, size=t.size, replace=True))
        try:
            p = fit_from(key, t, qs, base, dmin)
            e = eur(key, p, qab, max_years).eur
        except (ValueError, FloatingPointError, OverflowError):
            continue
        if math.isfinite(e) and e > 0:
            out.append(e)
    return np.sort(np.array(out))


def percentile(sorted_values, f: float) -> float:
    """Linear-interpolated percentile of a sorted array.

    NOTE the convention this serves. Petroleum reserves use EXCEEDANCE
    probabilities, so P90 is the LOW case -- the volume 90% of outcomes exceed
    -- and P90 < P50 < P10. The P90 is therefore percentile(dist, 0.10).
    This is inverted relative to statistical usage; label it when reporting.
    """
    v = np.asarray(sorted_values, dtype=float)
    if v.size == 0:
        return float("nan")
    i = (v.size - 1) * f
    lo, hi = math.floor(i), math.ceil(i)
    return float(v[lo] + (v[hi] - v[lo]) * (i - lo))


def pxx(sorted_values) -> Dict[str, float]:
    """{'P90': low, 'P50': median, 'P10': high} under the exceedance convention."""
    return {
        "P90": percentile(sorted_values, 0.10),
        "P50": percentile(sorted_values, 0.50),
        "P10": percentile(sorted_values, 0.90),
    }


# ----------------------------------------------------------------------------
# diagnostics -- run these BEFORE believing any fit
# ----------------------------------------------------------------------------

def nominal_decline(t, q, window: int = 7) -> np.ndarray:
    """Local nominal decline D(t) = -dln(q)/dt, by centred moving regression.

    A raw point-to-point derivative of noisy monthly data is unusable -- with
    10% scatter it recovers b = 0.37 where the truth is 1.1. Regressing ln q
    over a centred window of `window` points first recovers it to within a few
    percent, which is what makes the b cross-check below worth quoting.
    """
    t, q = _arr(t), _arr(q)
    n = t.size
    w = max(3, min(int(window) | 1, n if n % 2 else n - 1))
    lq = np.log(np.maximum(q, 1e-9))
    half = w // 2
    d = np.full(n, np.nan)
    for i in range(n):
        a = max(0, min(i - half, n - w))
        d[i] = -np.polyfit(t[a:a + w], lq[a:a + w], 1)[0]
    return d


@dataclass
class LossRatio:
    """b from the loss ratio, with the spread that says whether to believe it."""
    b: float
    spread: float               # inter-quartile spread across smoothing windows
    n: int                      # periods that survived the filters
    reliable: bool

    def agrees_with(self, b_fit: Optional[float]) -> Optional[bool]:
        """Does the fitted b agree? None when the loss ratio is not reliable."""
        if b_fit is None or not self.reliable:
            return None
        return abs(b_fit - self.b) <= max(0.4, 1.5 * self.spread)

    def __repr__(self):
        tag = "reliable" if self.reliable else "NOT reliable"
        return f"<LossRatio b={self.b:.2f} +/-{self.spread:.2f} n={self.n} {tag}>"


def _theil_sen(x, y) -> Optional[float]:
    """Median of pairwise slopes. Least squares on a loss ratio is not stable
    enough to quote: on a real intervened well it swings by more than 1.0 on
    nothing but a change of smoothing window."""
    n = len(x)
    s = [(y[j] - y[i]) / (x[j] - x[i])
         for i in range(n) for j in range(i + 1, n) if x[j] > x[i]]
    return float(np.median(s)) if s else None


def loss_ratio_b(t, q, tail: float = 0.6,
                 windows: Sequence[int] = (7, 11, 15, 21)) -> Optional[LossRatio]:
    """b from the slope of the loss ratio 1/D against t.

    This is the highest-yield check available, because it estimates b
    INDEPENDENTLY of the fit:

        agreement  -> the fit is describing reservoir depletion
        divergence -> it is describing operations, and the extrapolation will
                      be wrong in a direction the fit statistics cannot see

    But b here is a DIFFERENTIATED quantity, and on noisy monthly data it is
    imprecise: on a synthetic b = 1.1 well with 10% scatter the estimate ranges
    over roughly 0.3 to 1.6 depending only on the smoothing window. So this
    does not return a bare number to be quoted. It estimates over several
    windows, reports the spread between them, and marks itself unreliable when
    that spread is wide. An unreliable loss ratio is not a failure -- it is the
    finding that this well does not have a well-determined b, which is itself
    a reason to distrust an extrapolation that depends on one.

    Periods where the rate is rising, or where the local decline passes near
    zero, are dropped: a well being worked over has no meaningful loss ratio,
    and 1/D explodes there.
    """
    t, q = _arr(t), _arr(q)
    if t.size < 8:
        return None

    ests, used = [], 0
    for w in windows:
        if w > t.size:
            continue
        d = nominal_decline(t, q, w)
        ok = np.isfinite(d) & (d > 1e-4)
        ok[:2] = False
        if ok.sum() < 5:
            continue
        ok &= d > 0.08 * float(np.median(d[ok]))
        if ok.sum() < 5:
            continue
        tt, inv = t[ok], 1.0 / d[ok]
        cut = tt[int(tt.size * (1.0 - tail))]
        sel = tt >= cut
        if sel.sum() < 4:
            sel = np.ones(tt.size, dtype=bool)
        s = _theil_sen(tt[sel], inv[sel])
        if s is not None:
            ests.append(s)
            used = max(used, int(sel.sum()))

    if len(ests) < 2:
        return None
    b = float(np.median(ests))
    spread = float(np.percentile(ests, 75) - np.percentile(ests, 25))
    return LossRatio(b=b, spread=spread, n=used, reliable=spread <= 0.35)


def loglog_slope(t, q, tail: float = 0.5) -> Optional[float]:
    """Slope of log q against log t over the trailing part of the record.

    -1/2 is transient linear flow (fracture-dominated), -1/4 bilinear, -1
    boundary-dominated. Linear flow means Arps' boundary-dominated assumption
    does not hold yet: prefer Duong or PLE.
    """
    t, q = _arr(t), _arr(q)
    ok = (t > 0) & (q > 0)
    if ok.sum() < 6:
        return None
    lt, lq = np.log(t[ok]), np.log(q[ok])
    cut = lt[int(len(lt) * (1.0 - tail))]
    sel = lt >= cut
    if sel.sum() < 4:
        sel = np.ones_like(lt, dtype=bool)
    return float(np.polyfit(lt[sel], lq[sel], 1)[0])


@dataclass
class Plateau:
    end_index: int
    q_plateau: float
    months: float


def find_plateau(q, within: float = 0.90, min_months: int = 4) -> Optional[Plateau]:
    """Detect a facility-, quota- or compressor-constrained plateau.

    A plateau contains NO decline information, so the fit must start after it.
    The reference rate is the median of the top three (spike-robust), and a run
    that is short relative to the record is rejected -- that is an early flush
    feature or a late rebound, not a plateau.

    When one is found, re-reference time to the first fitted period. Without
    that, qi is the model extrapolated back to a fictitious t=0: on one real
    well held on plateau to month 17, the unreferenced fit reported
    qi = 2.4 million bbl/d.
    """
    q = _arr(q)
    if q.size < 4:
        return None
    top = np.sort(q)[::-1][:3]
    qp = float(top[1]) if top.size == 3 else float(top[0])
    hits = np.where(q >= within * qp)[0]
    if hits.size == 0:
        return None
    last, run = int(hits[-1]), int(hits.size)
    if last < min_months - 1 or run < min_months:
        return None
    if run < 0.6 * (last + 1):          # must be an early feature
        return None
    return Plateau(end_index=last, q_plateau=qp, months=float(last))


def extrapolation_multiple(remaining: float, cum_to_date: float) -> tuple:
    """remaining / cumulative, and what it means.

    This is the check that decides whether decline analysis alone is enough:

        < 0.5x   most of the volume is already history      -> DCA is fine
        0.5 - 1x comparable to everything the well has done  -> sanity-check it
        > 1x     the curve claims more than ever demonstrated
                 -> an INDEPENDENT in-place volume is required

    A hyperbolic knows nothing about how much fluid is in the tank. Past 1x it
    is no longer the binding constraint; the reservoir is.
    """
    if cum_to_date <= 0:
        return float("nan"), "no production history"
    m = remaining / cum_to_date
    if m < 0.5:
        v = "DCA alone is sufficient"
    elif m <= 1.0:
        v = "sanity-check against an in-place volume"
    else:
        v = "an independent in-place volume is REQUIRED"
    return m, v


# ----------------------------------------------------------------------------
# water analysis
# ----------------------------------------------------------------------------

def x_of(wor):
    """Ershaghi X = ln(WOR) + 1 + 1/WOR.

    From Buckley-Leverett fractional flow via the Welge tangent, for a
    log-linear relative-permeability ratio. X vs Np is linear where the semilog
    WOR plot curves.

    X is U-shaped with its minimum at WOR = 1 (dX/dWOR = 1/WOR - 1/WOR^2), so
    BELOW 50% WATER CUT IT IS DOUBLE-VALUED and any straight line through it is
    an artefact of which branch the points landed on. The 50% floor is
    structural, not conservatism.
    """
    w = np.asarray(wor, dtype=float)
    return np.log(w) + 1.0 + 1.0 / w


@dataclass
class WorResult:
    n: int
    wor_now: float
    fw_now: float
    cum_oil: float
    cum_water: float
    breakthrough_index: Optional[int]
    semilog: Optional[dict]             # {'slope','intercept','r2','n','np_at_limit'}
    xplot: Optional[dict]
    np_semilog: float
    np_xplot: float
    ec_water_cut: float

    def __repr__(self):
        return (f"<WorResult fw={self.fw_now:.1%} "
                f"Np@{self.ec_water_cut:.0%}wc semilog={self.np_semilog:,.0f} "
                f"xplot={self.np_xplot:,.0f}>")


def _linreg(x, y) -> dict:
    x, y = _arr(x), _arr(y)
    if x.size < 2:
        return {}
    m, b = np.polyfit(x, y, 1)
    yh = m * x + b
    ss = float(np.sum((y - yh) ** 2))
    st = float(np.sum((y - y.mean()) ** 2))
    return {"slope": float(m), "intercept": float(b),
            "r2": (1.0 - ss / st) if st > 0 else 0.0, "n": int(x.size)}


def wor_analysis(cum_oil, oil, water, ec_water_cut: float = 0.95,
                 fw_floor: float = 0.50) -> Optional[WorResult]:
    """Water-oil ratio extrapolation: semilog and Ershaghi X-plot.

    Both forecast oil recovery at an economic water cut, independent of any
    rate-time decline. For a water-drive reservoir this is usually the more
    reliable number, because the well dies of water rather than of pressure.

    A wide gap between the WOR answer and the Arps answer is information: the
    rate decline and the water trend are telling different stories about what
    kills this well.
    """
    cum_oil, oil, water = _arr(cum_oil), _arr(oil), _arr(water)
    ok = (oil > 0) & (water >= 0) & np.isfinite(cum_oil)
    if ok.sum() < 6:
        return None
    cum_oil, oil, water = cum_oil[ok], oil[ok], water[ok]
    wor = water / oil
    fw = water / (water + oil)
    wor_ec = ec_water_cut / (1.0 - ec_water_cut)

    bt = None
    for i in range(len(fw) - 1):
        if fw[i] >= 0.05 and fw[i + 1] >= 0.05:
            bt = i
            break

    sel_s = (wor >= 0.01) & (np.arange(len(wor)) >= (bt if bt is not None else 0))
    semi = _linreg(cum_oil[sel_s], np.log(np.maximum(wor[sel_s], 1e-12))) if sel_s.sum() >= 4 else None

    sel_x = fw >= fw_floor
    xp = _linreg(x_of(wor[sel_x]), cum_oil[sel_x]) if sel_x.sum() >= 4 else None

    np_semi = float("nan")
    if semi and semi.get("slope", 0) > 0:
        np_semi = (math.log(wor_ec) - semi["intercept"]) / semi["slope"]
    np_x = float("nan")
    if xp:
        np_x = xp["slope"] * float(x_of(wor_ec)) + xp["intercept"]

    for d, v in ((semi, np_semi), (xp, np_x)):
        if d is not None:
            d["np_at_limit"] = v

    return WorResult(
        n=int(len(wor)), wor_now=float(wor[-1]), fw_now=float(fw[-1]),
        cum_oil=float(cum_oil[-1]), cum_water=float(np.sum(water)),
        breakthrough_index=bt, semilog=semi, xplot=xp,
        np_semilog=np_semi, np_xplot=np_x, ec_water_cut=ec_water_cut,
    )


def chan_derivative(t, wor):
    """Chan (1995) WOR' = d(WOR)/dt, for the coning-vs-channelling read.

    On a log-log plot against time:
        WOR' flattening or turning down -> CONING. The cone reaches a
            pseudo-steady shape; rate reduction or a deeper completion helps.
        WOR' continuing to rise -> CHANNELLING or multilayer breakthrough.
            Rate reduction will not help; this needs conformance work.

    The distinction changes the recommendation, which is why the derivative's
    noise is worth tolerating.
    """
    t, wor = _arr(t), _arr(wor)
    return np.gradient(wor, t)


# ----------------------------------------------------------------------------
# gas material balance
# ----------------------------------------------------------------------------

_DAK = (0.3265, -1.0700, -0.5339, 0.01569, -0.05165,
        0.5475, -0.7361, 0.1844, 0.1056, 0.6134, 0.7210)


def z_factor(p_psia: float, t_degf: float, sg: float) -> float:
    """Gas compressibility factor: Dranchuk-Abou-Kassem on Sutton pseudo-criticals.

    Sutton rather than Standing because the pseudo-criticals here must cover
    wet gas with condensate. DAK because it is good to ~0.5% over the pressure
    range of interest. The iteration is damped, since the undamped step can
    overshoot into negative reduced density at high pressure.
    """
    if not p_psia > 0:
        return 1.0
    ppc = 756.8 - 131.0 * sg - 3.6 * sg * sg
    tpc = 169.2 + 349.5 * sg - 74.0 * sg * sg
    tpr = (t_degf + 459.67) / tpc
    ppr = p_psia / ppc
    z = 1.0
    for _ in range(300):
        rr = 0.27 * ppr / (z * tpr)
        zn = (1.0
              + (_DAK[0] + _DAK[1] / tpr + _DAK[2] / tpr**3 + _DAK[3] / tpr**4
                 + _DAK[4] / tpr**5) * rr
              + (_DAK[5] + _DAK[6] / tpr + _DAK[7] / tpr**2) * rr * rr
              - _DAK[8] * (_DAK[6] / tpr + _DAK[7] / tpr**2) * rr**5
              + _DAK[9] * (1.0 + _DAK[10] * rr * rr) * (rr * rr / tpr**3)
              * math.exp(-_DAK[10] * rr * rr))
        if not math.isfinite(zn):
            return z
        if abs(zn - z) < 1e-10:
            return zn
        z += 0.5 * (zn - z)
    return z


def condensate_gas_equivalent(api: float) -> float:
    """scf of gas equivalent per stock-tank barrel of condensate.

        Mo = 5954/(API - 8.811),  GE = 133000 * gamma_o / Mo

    Omitting this understates Gp in a gas-condensate reservoir, which biases
    OGIP low and remaining high -- the wrong direction for a reserves estimate.
    """
    if api <= 8.811:
        raise ValueError("API must exceed 8.811")
    gamma_o = 141.5 / (131.5 + api)
    mo = 5954.0 / (api - 8.811)
    return 133000.0 * gamma_o / mo


@dataclass
class MBResult:
    points: pd.DataFrame                # per-survey: P, z, p/z, Gp, Eg, F, F/Eg
    ogip_pz: float                      # MMscf, straight-line intercept
    pz_i: float
    r2: float
    gp_now: float
    recoverable: float                  # to abandonment pressure, MMscf
    remaining: float
    recovery_factor: float              # %
    ho_rise: float                      # F/Eg last / first
    min_f_over_eg: float                # Bcf
    volumetric: bool
    impossible: bool                    # G <= min(F/Eg) violated
    fetkovich: Optional[dict] = None

    @property
    def drive(self) -> str:
        if self.impossible:
            return "inconsistent -- check the reference pressure"
        return "volumetric" if self.volumetric else "water drive / pressure support"

    def __repr__(self):
        return (f"<MBResult drive={self.drive!r} OGIP(p/z)={self.ogip_pz/1000:.1f} Bcf "
                f"F/Eg rise={self.ho_rise:.2f}x>")


def material_balance(pressure_psia, gp_mmscf, t_degf: float, sg: float,
                     condensate_mbbl=None, water_mbbl=None,
                     gas_equivalent_scf_per_bbl: float = 0.0,
                     p_abandon: float = 1000.0,
                     skip_early: int = 0,
                     fit_aquifer: bool = True) -> MBResult:
    """Gas material balance: p/z, Havlena-Odeh drive diagnosis, Fetkovich aquifer.

    p/z vs Gp is straight for a volumetric tank and the Gp-axis intercept is
    OGIP. But a water-driven reservoir holds pressure UP, which flattens the
    trend and inflates that intercept -- the classic overestimate. So the
    straight line is never the discriminator.

    Havlena-Odeh is: F = G*Eg + We. Plot F/Eg against Gp --

        flat   => We ~ 0, volumetric, and the level IS G
        rising => water influx, and the p/z intercept is too high

    CONSISTENCY GUARD: since We >= 0, material balance requires G <= min(F/Eg).
    If the data imply an OGIP below what has already been produced, they are
    inconsistent and no aquifer model can rescue them. In practice this catches
    a wrong reference pressure -- an initial survey taken AFTER first production
    makes pi too low, every Eg too small, and F/Eg too large everywhere.

    skip_early: drop this many of the earliest surveys and re-reference. Use it
    when the guard fires and the first survey is the suspect one.
    """
    P = _arr(pressure_psia)
    Gs = _arr(gp_mmscf)
    C = _arr(condensate_mbbl) if condensate_mbbl is not None else np.zeros_like(P)
    W = _arr(water_mbbl) if water_mbbl is not None else np.zeros_like(P)
    if not (P.size == Gs.size == C.size == W.size):
        raise ValueError("all material balance inputs must be the same length")
    if P.size < 3:
        raise ValueError("need at least 3 pressure surveys")

    order = np.argsort(Gs)
    P, Gs, C, W = P[order], Gs[order], C[order], W[order]
    if skip_early:
        P, Gs, C, W = P[skip_early:], Gs[skip_early:], C[skip_early:], W[skip_early:]

    TR = t_degf + 459.67
    z = np.array([z_factor(float(p), t_degf, sg) for p in P])
    pz = P / z
    # wellstream basis: add the gas equivalent of produced condensate
    G = Gs + (C * 1000.0 * gas_equivalent_scf_per_bbl / 1e6 if gas_equivalent_scf_per_bbl else 0.0)

    slope, intercept = np.polyfit(G, pz, 1)
    yh = slope * G + intercept
    ss = float(np.sum((pz - yh) ** 2))
    st = float(np.sum((pz - pz.mean()) ** 2))
    r2 = (1.0 - ss / st) if st > 0 else 0.0

    ogip = -intercept / slope if slope < 0 else float("nan")
    z_ab = z_factor(p_abandon, t_degf, sg)
    g_ab = (p_abandon / z_ab - intercept) / slope if slope < 0 else float("nan")
    gp_now = float(G[-1])

    # --- Havlena-Odeh ---
    def Bg_cf(p):                        # ft3/scf
        return 0.02827 * z_factor(float(p), t_degf, sg) * TR / float(p)

    i_ref = int(np.argmax(P))
    p_i = float(P[i_ref])
    Bgi = Bg_cf(p_i)
    rows = []
    for k in range(len(P)):
        if P[k] >= p_i - 1:
            continue
        bg = Bg_cf(P[k])
        Eg = bg - Bgi
        Fg = Gs[k] * 1e6 * bg
        Fw = W[k] * 1e3 * 5.615
        Ftot = Fg + Fw
        if Eg <= 0 or Ftot <= 0:
            continue
        rows.append({"P": float(P[k]), "z": float(z[k]), "pz": float(pz[k]),
                     "Gp": float(G[k]), "Eg": Eg, "F": Ftot,
                     "F_over_Eg_Bcf": Ftot / Eg / 1e9})
    ho = pd.DataFrame(rows)

    ho_rise = float("nan")
    min_feg = float("nan")
    if len(ho) > 1:
        ho_rise = float(ho["F_over_Eg_Bcf"].iloc[-1] / ho["F_over_Eg_Bcf"].iloc[0])
        min_feg = float(ho["F_over_Eg_Bcf"].min())

    gp_bcf = gp_now / 1000.0
    impossible = math.isfinite(min_feg) and min_feg < gp_bcf
    volumetric = math.isfinite(ho_rise) and ho_rise < 1.25 and not impossible

    fetk = None
    if fit_aquifer and not impossible and len(ho) >= 3:
        fetk = fetkovich_fit(P, G, W, t_degf, sg, p_i, gp_bcf, min_feg)

    pts = pd.DataFrame({"P": P, "z": z, "pz": pz, "Gp": G, "Gp_gas_only": Gs,
                        "condensate_Mbbl": C, "water_Mbbl": W})

    return MBResult(
        points=pts, ogip_pz=float(ogip), pz_i=float(intercept), r2=r2,
        gp_now=gp_now, recoverable=float(g_ab),
        remaining=max(float(g_ab) - gp_now, 0.0),
        recovery_factor=100.0 * float(g_ab) / float(ogip) if math.isfinite(ogip) and ogip > 0 else float("nan"),
        ho_rise=ho_rise, min_f_over_eg=min_feg,
        volumetric=volumetric, impossible=impossible, fetkovich=fetk,
    )


def fetkovich_fit(P, G, W, t_degf: float, sg: float, p_i: float,
                  gp_bcf: float, min_feg: float, months: int = 12 * 18) -> dict:
    """Fetkovich pseudo-steady finite aquifer, fitted by grid search.

        Wei = ct * Wi * pi                              expansion capacity
        pa  = pi * (1 - We/Wei)                         average aquifer pressure
        dWe = (Wei/pi)(pa - pR)[1 - exp(-J*pi*dt/Wei)]  influx over dt

    NON-UNIQUENESS IS THE CENTRAL FACT. A large aquifer with a small J and a
    small aquifer with a large J give nearly the same influx history over a
    finite record; only late depletion of the aquifer separates them. So a
    per-G locus is returned alongside the best triplet, and quoting a single
    (Wei, J) pair as determined is a misuse of the result.

    Fetkovich rather than van Everdingen-Hurst because VEH needs superposition
    over every pressure change, which field surveys are too sparse to support.
    The cost is that Fetkovich understates early transient influx.
    """
    TR = t_degf + 459.67

    def Bg_rb(p):                        # rb/scf
        return 0.02827 * z_factor(float(p), t_degf, sg) * TR / float(p) / 5.615

    Bgi = Bg_rb(p_i)
    obs = []
    for k in range(len(P)):
        if P[k] >= p_i - 1:
            continue
        bg = Bg_rb(P[k])
        obs.append((float(G[k]), bg - Bgi, float(G[k]) * 1e6 * bg + float(W[k]) * 1e3))
    if len(obs) < 3:
        return None
    obs_G = np.array([o[0] for o in obs])
    obs_Eg = np.array([o[1] for o in obs])
    obs_F = np.array([o[2] for o in obs])

    # monthly grid with pressure interpolated linearly in Gp between surveys
    o = np.argsort(G)
    gG, gP = _arr(G)[o], _arr(P)[o]
    grid_Gp = np.linspace(gG[0], gG[-1], months + 1)
    grid_P = np.interp(grid_Gp, gG, gP)
    dt_days = DPM * months / months * (12 * 18 / months) * (months / (12 * 18)) * DPM / DPM
    dt_days = DPM * (12 * 18) / months

    def we_series(wei, j):
        we = 0.0
        out = np.empty(months + 1)
        out[0] = 0.0
        k = 1.0 - math.exp(-j * p_i * dt_days / wei)
        for i in range(1, months + 1):
            pa = p_i * (1.0 - we / wei)
            pr = 0.5 * (grid_P[i] + grid_P[i - 1])
            we += (wei / p_i) * (pa - pr) * k
            out[i] = max(we, 0.0)
        return out

    def sse(g_scf, wei, j):
        s = we_series(wei, j)
        we_at = np.interp(obs_G, grid_Gp, s)
        r = (obs_F - (g_scf * obs_Eg + we_at)) / obs_F
        return float(np.mean(r * r))

    g_lo = max(gp_bcf * 1.02, 1.0)
    g_hi = max(min_feg * 1.05, g_lo * 1.1) if math.isfinite(min_feg) else g_lo * 4
    weis = (20, 50, 100, 200, 400, 800, 1600, 3200, 6400)
    js = (0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500)

    best = None
    for gi in range(25):
        g = g_lo + (g_hi - g_lo) * gi / 24.0
        for wei in weis:
            for j in js:
                e = sse(g * 1e9, wei * 1e6, j)
                if best is None or e < best[3]:
                    best = (g, wei, j, e)

    locus = []
    for gi in range(9):
        g = g_lo + (g_hi - g_lo) * gi / 8.0
        b = None
        for wei in weis:
            for j in js:
                e = sse(g * 1e9, wei * 1e6, j)
                if b is None or e < b[2]:
                    b = (wei, j, e)
        locus.append({"G_Bcf": g, "Wei_MMbbl": b[0], "J_bbl_d_psi": b[1],
                      "rms_pct": 100.0 * math.sqrt(b[2])})

    g, wei, j, e = best
    s = we_series(wei * 1e6, j)
    return {
        "G_Bcf": g, "Wei_MMbbl": wei, "J_bbl_d_psi": j,
        "rms_pct": 100.0 * math.sqrt(e),
        "We_MMbbl": float(s[-1] / 1e6),
        "locus": pd.DataFrame(locus),
        "curve": pd.DataFrame({"Gp_MMscf": grid_Gp[::6], "We_MMbbl": s[::6] / 1e6}),
        "note": ("rms above 10% means the model cannot reproduce the history; "
                 "(Wei, J) is a locus, not a determined pair"),
    }


# ----------------------------------------------------------------------------
# data loading
# ----------------------------------------------------------------------------

@dataclass
class Series:
    """A production history on a month clock, ready to fit."""
    t: np.ndarray                       # months since the first record, mid-period
    q: np.ndarray                       # operated-day rate
    volume: np.ndarray
    days: np.ndarray
    dates: pd.Series
    cum: np.ndarray
    column: str
    is_gas: bool
    water: Optional[np.ndarray] = None
    frame: Optional[pd.DataFrame] = field(default=None, repr=False)

    @property
    def unit(self) -> str:
        return "Mcf" if self.is_gas else "bbl"

    def window(self, start: int = 0, end: Optional[int] = None) -> "Series":
        """Slice to a fit window, RE-REFERENCING time to the first kept period.

        Re-referencing is what makes qi a rate the well actually flowed rather
        than a back-extrapolation to a fictitious t=0.
        """
        sl = slice(start, end)
        t = self.t[sl]
        return Series(
            t=t - t[0] + 0.5, q=self.q[sl], volume=self.volume[sl], days=self.days[sl],
            dates=self.dates.iloc[sl], cum=self.cum[sl], column=self.column,
            is_gas=self.is_gas, water=None if self.water is None else self.water[sl],
            frame=None if self.frame is None else self.frame.iloc[sl],
        )

    def __repr__(self):
        return (f"<Series {self.column} n={len(self.t)} "
                f"{self.dates.iloc[0]:%Y-%m} to {self.dates.iloc[-1]:%Y-%m} "
                f"cum={self.cum[-1]:,.0f} {self.unit}>")


def load_csv(path: str, column: Optional[str] = None,
             days_column: str = "Days On", date_column: str = "Date") -> Series:
    """Read a monthly production CSV into a Series.

        Date,Days On,Oil (bbl),Gas (Mcf),Water (bbl)
        2018-06-01,30,4874,4058,5288

    Two things this does that a naive reader does not:

    1. TIME COMES FROM THE DATES, not the row order. Zero-production months are
       omitted from these files, so the date gaps ARE the shut-ins. Counting
       rows would compress them out of existence and fabricate decline.

    2. Rate is volume / producing days, i.e. an OPERATED-DAY rate. Feeding a
       reported rate straight in inherits whatever uptime convention the source
       used -- and calendar-day and operated-day rates differ by the uptime
       fraction, which is how downtime gets mistaken for decline.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if date_column not in df.columns:
        raise ValueError(f"no {date_column!r} column in {path}: {list(df.columns)}")

    if column is None:
        for c in df.columns:
            if c.lower().startswith(("oil", "gas")):
                column = c
                break
    if column is None or column not in df.columns:
        raise ValueError(f"no production column found in {path}: {list(df.columns)}")

    df[date_column] = pd.to_datetime(df[date_column])
    vol = pd.to_numeric(df[column], errors="coerce")
    days = (pd.to_numeric(df[days_column], errors="coerce")
            if days_column in df.columns else pd.Series(DPM, index=df.index))

    ok = vol.notna() & (vol > 0) & days.notna() & (days > 0)
    df, vol, days = df[ok].reset_index(drop=True), vol[ok].to_numpy(), days[ok].to_numpy()
    if len(df) < 5:
        raise ValueError(f"only {len(df)} usable periods in {path}")

    d = df[date_column]
    months = (d.dt.year * 12 + d.dt.month).to_numpy()
    t = (months - months[0]).astype(float) + 0.5

    water = None
    for c in df.columns:
        if c.lower().startswith("water"):
            water = pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy()
            break

    return Series(t=t, q=vol / days, volume=vol, days=days, dates=d,
                  cum=np.cumsum(vol), column=column,
                  is_gas="gas" in column.lower(), water=water, frame=df)


def load_pressure_csv(path: str) -> pd.DataFrame:
    """Read a pressure-survey CSV for material balance.

        Date,Pressure (psia),Cum gas (MMscf),Cum condensate (Mbbl),Cum water (Mbbl)
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if df.shape[1] < 3:
        raise ValueError(f"{path}: need at least date, pressure and cumulative gas")
    out = pd.DataFrame({
        "date": pd.to_datetime(df.iloc[:, 0]),
        "P": pd.to_numeric(df.iloc[:, 1], errors="coerce"),
        "Gp": pd.to_numeric(df.iloc[:, 2], errors="coerce"),
    })
    out["condensate"] = pd.to_numeric(df.iloc[:, 3], errors="coerce").fillna(0.0) if df.shape[1] > 3 else 0.0
    out["water"] = pd.to_numeric(df.iloc[:, 4], errors="coerce").fillna(0.0) if df.shape[1] > 4 else 0.0
    return out.dropna(subset=["P", "Gp"]).sort_values("Gp").reset_index(drop=True)


# ----------------------------------------------------------------------------
# plotting (optional -- matplotlib is imported lazily)
# ----------------------------------------------------------------------------

def plot_forecast(s: Series, fits: Sequence[FitResult], qab: float,
                  path: str, max_years: float = 50.0, band=None) -> str:
    """Semilog rate plot with every fitted model and, optionally, the P10-P90 band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(s.t, s.q, "o", ms=3.5, color="#14181a", label="observed", zorder=5)

    t_end = max(eur(f.key, f.p, qab, max_years).t_end for f in fits)
    # Start at the first observed period, not at zero: Duong's t^-m form dives to
    # nothing as t -> 0, and plotting that collapses the log axis.
    tt = np.linspace(float(s.t[0]), t_end, 400)
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#e34948"]
    for i, f in enumerate(fits):
        ax.semilogy(tt, f.rate(tt), lw=1.6, color=colors[i % len(colors)],
                    label=f"{f.key} (EUR {eur(f.key, f.p, qab, max_years).eur:,.0f})")
    if band is not None:
        lo, hi = band
        ax.fill_between(tt, lo, hi, color="#2a78d6", alpha=0.12, lw=0, label="P10-P90")

    # Frame on the data and the economic limit, never on a model's t -> 0 behaviour.
    ax.set_ylim(min(float(np.min(s.q)), qab) / 3.0, float(np.max(s.q)) * 3.0)
    ax.axhline(qab, ls="--", lw=1, color="#7e8286")
    ax.annotate(f"economic limit {qab:g} {s.unit}/d", (float(s.t[0]), qab), va="bottom",
                fontsize=8, color="#7e8286")
    ax.axvline(s.t[-1], ls=":", lw=1, color="#7e8286")
    ax.set_xlabel("months on production")
    ax.set_ylabel(f"rate ({s.unit}/d)")
    ax.set_title(f"{s.column} -- {len(s.t)} periods")
    ax.grid(True, which="both", lw=0.4, alpha=0.35)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def _cli_decline(a) -> int:
    s = load_csv(a.file, column=a.column, days_column=a.days)
    full = s
    end = a.to if a.to else len(s.t)
    start = max(a.__dict__["from"] - 1, 0)

    if a.plateau:
        pl = find_plateau(s.q[start:end])
        if pl:
            print(f"  plateau detected: {pl.end_index + 1} periods at "
                  f"~{pl.q_plateau:,.0f} {s.unit}/d -- fitting after it")
            start += pl.end_index

    w = s.window(start, end)
    qab = a.qab if a.qab is not None else (50.0 if s.is_gas else 10.0)
    # A horizon shorter than the history truncates the EUR before the last data point
    # and reports remaining = 0 -- which reads as a reservoir statement when it is only
    # an artefact of the cap.
    if a.horizon is None:
        a.horizon = max(50.0, math.ceil(float(s.t[-1]) / 12.0) + 20.0)
    dmin = nom_from_eff(a.dmin / 100.0) / 12.0
    cum_before = float(full.cum[start - 1]) if start > 0 else 0.0

    print(f"\n{a.file}")
    print(f"  {s.column} · {len(full.t)} producing periods · fitting {len(w.t)}"
          f" ({w.dates.iloc[0]:%Y-%m} to {w.dates.iloc[-1]:%Y-%m})")
    print(f"  abandonment {qab:g} {s.unit}/d · Dmin {a.dmin:g} %/yr effective")
    if cum_before > 0:
        print(f"  {cum_before:,.0f} {s.unit} produced before the fit window (excluded)")

    keys = a.models.split(",") if a.models else list(MODELS)
    rows = []
    for k in keys:
        try:
            f = fit(k, w.t, w.q, dmin=dmin if k == "modhyp" else None)
        except ValueError as e:
            print(f"  ! {k}: {e}")
            continue
        e = eur(k, f.p, qab, a.horizon)
        rem = max(e.eur - cumulative(k, f.p, float(w.t[-1])), 0.0)
        rows.append((f, e, rem))
    if not rows:
        print("  nothing could be fitted")
        return 1
    rows.sort(key=lambda r: r[0].aicc)
    best_aicc = rows[0][0].aicc

    print(f"\n  {'MODEL':<14}{'R2':>8}{'RMSE':>9}{'dAICc':>9}{'EUR':>15}{'REMAINING':>15}{'LIFE yr':>9}")
    for f, e, rem in rows:
        print(f"  {f.key:<14}{f.r2:>8.4f}{f.rmse_log:>9.4f}{f.aicc - best_aicc:>9.1f}"
              f"{e.eur:>15,.0f}{rem:>15,.0f}{e.years:>9.1f}")

    f, e, rem = rows[0]
    if w.q[-1] <= qab:
        print(f"\n  ! last rate {w.q[-1]:,.1f} {s.unit}/d is at or below the abandonment rate"
              f" {qab:g} — remaining reads zero. Lower --qab if the well is still economic.")

    ps = " · ".join(f"{k} {v:.4g}" for k, v in f.p.items())
    print(f"\n  best: {f.key}  {ps}")
    if "Di" in f.p:
        print(f"  initial effective decline {eff_from_nom(f.p['Di'] * 12) * 100:.1f} %/yr")

    # diagnostics
    lr = loss_ratio_b(w.t, w.q)
    slope = loglog_slope(w.t, w.q)
    if lr is not None:
        # Compare against the HYPERBOLIC b specifically -- that is the b the loss
        # ratio estimates -- even when another model wins on AICc.
        b_fit = f.p.get("b")
        via = ""
        if b_fit is None:
            hyp = next((r[0] for r in rows if r[0].key == "hyperbolic"), None)
            if hyp is not None:
                b_fit, via = hyp.p["b"], " (hyperbolic)"
        print(f"\n  loss-ratio b {lr.b:.2f} (+/-{lr.spread:.2f} across smoothing windows)"
              + (f" vs fitted b {b_fit:.2f}{via}" if b_fit is not None else ""))
        if not lr.reliable:
            print("    the loss ratio is not well determined on this well — no cross-check")
            print("    is available, which is itself a reason to distrust an extrapolation")
            print("    that leans on b")
        elif lr.agrees_with(b_fit) is False:
            print("    ** THEY DISAGREE ** — the fit is probably describing operations")
            print("    (interventions, offtake changes, downtime) rather than depletion, and")
            print("    the extrapolation will be wrong in a direction R2 cannot see")
        elif lr.agrees_with(b_fit):
            print("    they agree — the fit is consistent with reservoir depletion")
    if slope is not None:
        if slope > -0.15:
            regime = "flat or rising — this well is not in decline over this window"
        elif -0.75 < slope <= -0.3:
            regime = "linear flow — not boundary-dominated yet, prefer Duong or PLE"
        elif slope <= -0.75:
            regime = "boundary-dominated"
        else:
            regime = "shallow — check for a rate constraint"
        print(f"  log-log slope {slope:.2f} — {regime}")

    cum_to_date = float(full.cum[end - 1])
    mult, verdict = extrapolation_multiple(rem, cum_to_date)
    print(f"\n  cum to date {cum_to_date:,.0f} {s.unit} · remaining {rem:,.0f} {s.unit}")
    print(f"  extrapolation multiple {mult:.2f}x  ->  {verdict}")
    if a.in_place:
        cap = a.in_place * a.rf / 100.0
        print(f"  in-place cap: {a.in_place:,.0f} {s.unit} x RF {a.rf:g}% = {cap:,.0f}"
              f" recoverable; EUR {'EXCEEDS' if e.eur > cap else 'is within'} it")

    if a.boot > 0:
        dist = bootstrap_eur(f.key, w.t, w.q, f.p, qab, a.boot, a.horizon,
                             dmin if f.key == "modhyp" else None, seed=a.seed)
        p = pxx(dist)
        print(f"\n  {len(dist)}-replicate bootstrap "
              f"(exceedance convention — P90 is the LOW case)")
        print(f"  P90 {p['P90']:,.0f}   P50 {p['P50']:,.0f}   P10 {p['P10']:,.0f}  {s.unit}")
        print("  fit uncertainty only — not model choice, window choice, or geology")

    if s.water is not None and a.water:
        wa = wor_analysis(w.cum, w.volume, w.water)
        if wa:
            print(f"\n  water cut now {wa.fw_now:.1%} · WOR {wa.wor_now:.2f}")
            for nm, d in (("semilog", wa.semilog), ("X-plot", wa.xplot)):
                if d and d.get("r2", 0) >= 0.7 and math.isfinite(d.get("np_at_limit", float("nan"))):
                    print(f"  Np at {wa.ec_water_cut:.0%} water cut, {nm}: "
                          f"{d['np_at_limit']:,.0f} {s.unit} (R2 {d['r2']:.3f}, n={d['n']})")
                elif d:
                    print(f"  {nm}: R2 {d['r2']:.3f} — no usable trend, not reported")

    if a.plot:
        band = None
        if a.boot > 0:
            print(f"  (band on the plot is the best model's P10-P90)")
        plot_forecast(w, [r[0] for r in rows], qab, a.plot, a.horizon, band)
        print(f"\n  wrote {a.plot}")

    print()
    return 0


def _cli_mb(a) -> int:
    df = load_pressure_csv(a.file)
    ge = condensate_gas_equivalent(a.api) if a.api else 0.0
    r = material_balance(df["P"], df["Gp"], a.temp, a.sg,
                         condensate_mbbl=df["condensate"], water_mbbl=df["water"],
                         gas_equivalent_scf_per_bbl=ge, p_abandon=a.pab,
                         skip_early=a.skip)

    print(f"\n{a.file}")
    print(f"  {len(df)} surveys · T {a.temp:g} F · gas gravity {a.sg:g}"
          + (f" · condensate {a.api:g} API ({ge:,.0f} scf/bbl)" if ge else ""))
    print(f"\n  p/z straight line      R2 {r.r2:.4f}")
    print(f"  OGIP (p/z intercept)   {r.ogip_pz / 1000:,.1f} Bcf")
    print(f"  produced               {r.gp_now / 1000:,.1f} Bcf  ({100 * r.gp_now / r.ogip_pz:.1f}% of it)")
    print(f"  recoverable to {a.pab:g} psi  {r.recoverable / 1000:,.1f} Bcf  (RF {r.recovery_factor:.0f}%)")
    print(f"  remaining              {r.remaining / 1000:,.1f} Bcf")
    print(f"\n  Havlena-Odeh F/Eg rise {r.ho_rise:.2f}x  ->  {r.drive}")

    if r.impossible:
        print(f"\n  ** CONSISTENCY GUARD FIRED **")
        print(f"  Material balance requires G <= min(F/Eg) = {r.min_f_over_eg:,.1f} Bcf,")
        print(f"  but {r.gp_now / 1000:,.1f} Bcf has already been produced. The surveys and")
        print(f"  the volumes cannot both be right. The usual cause is a reference")
        print(f"  pressure taken AFTER first production — try --skip 1.")
        return 1

    if not r.volumetric:
        print(f"  The p/z intercept above is an ARTEFACT, not a volume: something outside")
        print(f"  the gas is supplying energy. Use the aquifer fit below, and a water-drive")
        print(f"  recovery factor of 50-70% rather than the 80-90% a volumetric tank earns.")

    if r.fetkovich:
        f = r.fetkovich
        print(f"\n  Fetkovich aquifer fit")
        print(f"    G    {f['G_Bcf']:,.0f} Bcf")
        print(f"    Wei  {f['Wei_MMbbl']:,.0f} MMbbl    J {f['J_bbl_d_psi']:g} bbl/d/psi")
        print(f"    We   {f['We_MMbbl']:,.0f} MMbbl     rms {f['rms_pct']:.1f}%")
        if f["rms_pct"] > 10:
            print(f"    ! rms above 10% — the model cannot reproduce the history; treat the")
            print(f"      parameters as indicative only")
        lo, hi = f["G_Bcf"] * 0.5, f["G_Bcf"] * 0.7
        print(f"    remaining at RF 50-70%: "
              f"{max(lo - r.gp_now / 1000, 0):,.0f} to {max(hi - r.gp_now / 1000, 0):,.0f} Bcf")
        print(f"\n  (Wei, J) locus — the solution is NOT unique:")
        print("    " + f["locus"].to_string(index=False, float_format=lambda v: f"{v:,.1f}")
              .replace("\n", "\n    "))
    print()
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="dca.py",
        description="Decline curve analysis and gas material balance.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples
  python dca.py data/ecmc-myers-19-11HA.csv --qab 3
  python dca.py data/well-A-26-oil.csv --to 72 --qab 50 --plot fit.png
  python dca.py data/synthetic-water-drive.csv --water --qab 100
  python dca.py data/reservoir-B-3-pressure.csv --mb --temp 180 --sg 0.735 --api 58
""")
    p.add_argument("file", help="production CSV, or a pressure CSV with --mb")
    p.add_argument("--mb", action="store_true", help="material balance instead of decline analysis")

    g = p.add_argument_group("decline analysis")
    g.add_argument("--column", help="production column (default: first Oil/Gas found)")
    g.add_argument("--days", default="Days On", help="producing-days column")
    g.add_argument("--models", help="comma-separated subset of models to fit")
    g.add_argument("--qab", type=float, help="abandonment rate per day (default 10 oil / 50 gas)")
    g.add_argument("--from", type=int, default=1, help="first period, 1-based")
    g.add_argument("--to", type=int, help="last period")
    g.add_argument("--plateau", action="store_true", help="detect a plateau and fit after it")
    g.add_argument("--dmin", type=float, default=6.0, help="terminal effective decline %%/yr")
    g.add_argument("--horizon", type=float, default=None,
                   help="maximum well life in years, counted from the start of the fit "
                        "window (default: 20 years past the end of the history)")
    g.add_argument("--boot", type=int, default=300, help="bootstrap replicates, 0 to skip")
    g.add_argument("--seed", type=int, default=None, help="bootstrap seed, for reproducibility")
    g.add_argument("--water", action="store_true", help="also run WOR / X-plot analysis")
    g.add_argument("--in-place", type=float, dest="in_place",
                   help="known in-place volume, to cap the forecast")
    g.add_argument("--rf", type=float, default=70.0, help="recovery factor %% on --in-place")
    g.add_argument("--plot", help="write a rate plot to this path (needs matplotlib)")

    m = p.add_argument_group("material balance")
    m.add_argument("--temp", type=float, default=230.0, help="reservoir temperature, degF")
    m.add_argument("--sg", type=float, default=0.72, help="gas gravity")
    m.add_argument("--api", type=float, help="condensate API, for the gas-equivalent correction")
    m.add_argument("--pab", type=float, default=1000.0, help="abandonment pressure, psia")
    m.add_argument("--skip", type=int, default=0, help="drop this many earliest surveys")

    a = p.parse_args(argv)
    try:
        return _cli_mb(a) if a.mb else _cli_decline(a)
    except (ValueError, KeyError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
