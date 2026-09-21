"""
================================================================================
 OIL DECLINE CURVE ANALYSIS  --  conventional reservoirs
================================================================================

A decline-curve workflow for conventional oil wells, built as a sibling to
`gas_condensate_dca` and importing the parts of it that are about curves and
data rather than about gas: the Arps family and its fitting, the date parser,
the QC discipline, the window-sensitivity check, the aquifer models and the
Monte Carlo conventions.

What is genuinely different about oil, and why this is a separate module:

  * **The material balance is three-termed.** A gas reservoir has one
    expansion term. An oil reservoir has three - oil and its dissolved gas,
    a gas cap if there is one, and the connate water and rock - plus water
    influx. Havlena and Odeh (1963) wrote it as a straight line:

        F = N (Eo + m Eg Boi/Bgi + Efw) + We

    and the whole craft is in choosing which grouping makes the line straight,
    because N and m are strongly correlated and a plot that looks linear can
    be linear for the wrong reason.

  * **Gas is a product of the oil, not the stream.** Below the bubble point
    solution gas comes out of the oil in the reservoir, the producing GOR
    climbs, and the reservoir loses its own drive energy doing it. The GOR
    history is therefore a diagnostic of pressure, not just a yield.

  * **Water is a displacement process, not a contaminant.** On a gas well
    water is a nuisance that ends the well. On an oil well it is often the
    drive, and the shape of the water-cut curve says whether it is arriving
    by coning, by channelling, or as ordinary displacement.

The reporting discipline is inherited deliberately: every fit reports its
pinned parameters, every inference that rests on an assumption says so, and a
calculation that cannot be trusted refuses rather than returning a plausible
number.

References
----------
Havlena, D. and Odeh, A.S. (1963) *The material balance as an equation of a
    straight line.* JPT 15(8) and 16(7).
Standing, M.B. (1947) *A pressure-volume-temperature correlation for mixtures
    of California oils and gases.* Drill. & Prod. Prac., API.
Vasquez, M. and Beggs, H.D. (1980) *Correlations for fluid physical property
    prediction.* JPT 32(6).
Beggs, H.D. and Robinson, J.R. (1975) *Estimating the viscosity of crude oil
    systems.* JPT 27(9).
Chan, K.S. (1995) *Water control diagnostic plots.* SPE 30775.
Arps, J.J. (1945) *Analysis of decline curves.* Trans. AIME 160.
"""

from __future__ import annotations

import math
import re
import textwrap
import warnings
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats

import gas_condensate_dca as gdca
from gas_condensate_dca import (
    DAYS_PER_YEAR, MSCF_PER_MMSCF,
    DateFormatError, parse_dates,
    FitResult, fit_decline, rank_models, window_sensitivity,
    detect_decline_start, detect_bdf_start,
    WaterTrend, fit_water_trend,
    percentiles_petroleum, cumulative_trapezoid,
    api_to_sg, sutton_pseudocriticals, gas_fvf_rb_per_scf,
)

__version__ = "1.0"

__all__ = [
    "__version__", "OilPVT", "OilProductionData", "OilQCReport",
    "MaterialBalanceOil", "havlena_odeh_oil",
    "GORDiagnostic", "gor_diagnostic",
    "WaterCutDiagnostic", "chan_diagnostic",
    "OilPIDiagnostic", "oil_pi_diagnostic",
    "GORModel", "fit_gor_model", "WaterCutModel", "fit_water_cut_model",
    "OilForecast", "forecast_oil", "monte_carlo_oil",
    "OilWellResult", "analyse_oil_well",
    "make_synthetic_oil_well", "run_self_tests",
]

# Standard conditions and unit bridges.
SCF_PER_MSCF = 1.0e3
STB_PER_MSTB = 1.0e3
BBL_PER_ACRE_FT = 7758.0


# ==============================================================================
# SECTION 1 -- OIL PVT
# ==============================================================================

def standing_rs(p: np.ndarray, api: float, gas_gravity: float,
                temperature_F: float) -> np.ndarray:
    """Solution gas-oil ratio, scf/STB, by Standing (1947).

        Rs = gg * [ (p/18.2 + 1.4) * 10^(0.0125 API - 0.00091 T) ]^1.2048

    Valid below the bubble point; above it Rs is constant at Rsi, which the
    caller enforces. Standing was fitted to California crudes and is the
    correlation most commonly paired with the Standing Bo below.
    """
    p = np.asarray(p, dtype=float)
    x = 0.0125 * float(api) - 0.00091 * float(temperature_F)
    return float(gas_gravity) * np.power(
        np.maximum(p, 1e-6) / 18.2 + 1.4, 1.2048) * (10.0 ** (1.2048 * x))


def standing_bubble_point(rsi: float, api: float, gas_gravity: float,
                          temperature_F: float) -> float:
    """Bubble point from the initial solution GOR, the inverse of Standing Rs.

    Inverted analytically rather than iterated: the correlation is monotonic
    in p, so there is exactly one root and no reason to search for it.
    """
    x = 0.0125 * float(api) - 0.00091 * float(temperature_F)
    inner = (float(rsi) / max(float(gas_gravity), 1e-9)) ** (1.0 / 1.2048)
    return float(18.2 * (inner / (10.0 ** x) - 1.4))


def standing_bo(p: np.ndarray, rs: np.ndarray, api: float,
                gas_gravity: float, temperature_F: float) -> np.ndarray:
    """Oil formation volume factor below the bubble point, Standing (1947).

        Bo = 0.9759 + 0.00012 * [ Rs (gg/go)^0.5 + 1.25 T ]^1.2
    """
    rs = np.asarray(rs, dtype=float)
    go = api_to_sg(float(api))
    f = rs * math.sqrt(float(gas_gravity) / go) + 1.25 * float(temperature_F)
    return 0.9759 + 0.00012 * np.power(np.maximum(f, 0.0), 1.2)


def vasquez_beggs_co(p: np.ndarray, rsi: float, api: float,
                     gas_gravity: float, temperature_F: float) -> np.ndarray:
    """Undersaturated oil compressibility, 1/psi, Vasquez-Beggs (1980).

        co = (-1433 + 5 Rs + 17.2 T - 1180 gg + 12.61 API) / (1e5 p)

    Above the bubble point the oil is single phase, so Bo shrinks with
    pressure through this compressibility rather than through Rs.
    """
    p = np.asarray(p, dtype=float)
    num = (-1433.0 + 5.0 * float(rsi) + 17.2 * float(temperature_F)
           - 1180.0 * float(gas_gravity) + 12.61 * float(api))
    return num / (1.0e5 * np.maximum(p, 1.0))


def beggs_robinson_muo(p: np.ndarray, rs: np.ndarray, api: float,
                       temperature_F: float) -> np.ndarray:
    """Saturated oil viscosity, cp, Beggs and Robinson (1975)."""
    T = float(temperature_F)
    z = 3.0324 - 0.02023 * float(api)
    y = 10.0 ** z
    x = y * (T ** -1.163)
    mu_od = (10.0 ** x) - 1.0                        # dead oil
    rs = np.asarray(rs, dtype=float)
    a = 10.715 * np.power(rs + 100.0, -0.515)
    b = 5.44 * np.power(rs + 150.0, -0.338)
    return a * np.power(max(mu_od, 1e-6), b)


def vasquez_beggs_muo_undersaturated(p: np.ndarray, p_bubble: float,
                                     mu_ob: np.ndarray) -> np.ndarray:
    """Undersaturated oil viscosity, cp, Vasquez and Beggs (1980).

        mu_o = mu_ob * (p / pb) ** m,
        m    = 2.6 * p**1.187 * exp(-11.513 - 8.98e-5 * p)

    Above the bubble point no more gas can dissolve, so the only thing left
    acting on the oil is compression, and compression makes it more viscous.
    Holding mu_o flat at mu_ob instead - which is what this module did until a
    self-test asked the question the wrong way round and made me look - is
    right to within a few per cent over a thousand psi, and wrong in a fixed
    direction, which is the kind of error that never shows up as scatter.
    """
    p = np.asarray(p, dtype=float)
    pb = max(float(p_bubble), 1e-9)
    m = 2.6 * np.power(np.maximum(p, 1e-9), 1.187) * np.exp(
        -11.513 - 8.98e-5 * np.maximum(p, 0.0))
    return np.asarray(mu_ob, dtype=float) * np.power(
        np.maximum(p, 1e-9) / pb, m)


# Published ranges of the correlations this module uses. Outside them the
# correlations are extrapolations, and they are extrapolations of a curve fit
# to somebody else's fluids - which is a different and much weaker thing than
# an extrapolation of your own data.
#
#   Standing (1947)        Rs, pb, Bo
#   Vasquez-Beggs (1980)   co above the bubble point
#   Beggs-Robinson (1975)  oil viscosity
#
# The ranges below are the data each author actually fitted, as published.
CORRELATION_RANGES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "Standing (Rs, pb, Bo)": {
        "API": (16.5, 63.8), "temperature_F": (100.0, 258.0),
        "Rsi": (20.0, 1425.0), "gas_gravity": (0.59, 0.95),
        "p_bubble": (130.0, 7000.0),
    },
    "Vasquez-Beggs (co)": {
        "API": (15.3, 59.5), "temperature_F": (75.0, 294.0),
        "Rsi": (9.3, 2199.0), "gas_gravity": (0.51, 1.35),
    },
    "Beggs-Robinson (viscosity)": {
        "API": (16.0, 58.0), "temperature_F": (70.0, 295.0),
        "Rsi": (20.0, 2070.0),
    },
}


def correlation_range_warnings(pvt: "OilPVT") -> List[str]:
    """Which correlations this fluid falls outside, and by how much.

    A volatile oil at 2,000 scf/STB is past the top of Standing's solution-gas
    range; a 14 API heavy oil is below the bottom of all three. The
    correlations still return numbers there - they are smooth functions - and
    those numbers carry no evidence at all. This does not refuse to run: a
    correlation outside its range is often the only thing available. It says
    so, once, next to the properties it produced.
    """
    vals = {
        "API": float(pvt.api),
        "temperature_F": float(pvt.temperature_F),
        "Rsi": float(pvt.rsi),
        "gas_gravity": float(pvt.gas_gravity),
        "p_bubble": float(pvt.p_bubble),
    }
    out: List[str] = []
    for name, rng in CORRELATION_RANGES.items():
        bad = []
        for key, (lo, hi) in rng.items():
            v = vals.get(key)
            if v is None or not np.isfinite(v):
                continue
            if v < lo:
                bad.append(f"{key} {v:,.4g} below {lo:,.4g}")
            elif v > hi:
                bad.append(f"{key} {v:,.4g} above {hi:,.4g}")
        if bad:
            out.append(f"{name}: " + "; ".join(bad))
    return out


def fluid_class(pvt: "OilPVT") -> str:
    """A rough label for the fluid, from Rsi and gravity.

    Boundaries follow the usual convention and are deliberately coarse; the
    point is to say which regime the correlations are being asked to work in,
    not to classify the fluid for its own sake.
    """
    rsi, api = float(pvt.rsi), float(pvt.api)
    if api < 20.0 or rsi < 200.0:
        return "heavy / low-GOR oil"
    if rsi > 1750.0 or (api > 40.0 and rsi > 1250.0):
        return "volatile oil"
    if rsi > 1000.0:
        return "light oil, high solution gas"
    return "black oil"


def pvt_physicality_warnings(pvt: "OilPVT", n: int = 400) -> List[str]:
    """Checks the fluid model against physics, not against a correlation.

    A correlation outside its published range is an extrapolation; a
    correlation that violates a conservation statement is wrong. These are
    the statements that hold for any black oil whatever the correlation:

      Bt never falls as pressure falls   - the total volume occupied by one
                                           stock-tank barrel and its gas can
                                           only expand on the way down
      Bo peaks at the bubble point       - above it the oil only compresses,
                                           below it gas leaves solution
      Bg falls as pressure rises         - gas compresses
      0 <= Rs <= Rsi                     - gas cannot come out of nowhere

    On a near-critical fluid at Rsi = 2,600 scf/STB - eighty per cent past
    the top of Standing's range - Bt came back non-monotone over a 580 psi
    window just below the bubble point. The dip was 0.0009 rb/STB, far too
    small to move a forecast, and it is reported anyway: it is the fluid
    model saying it has left the region where it means anything.
    """
    out: List[str] = []
    pb = float(pvt.p_bubble)
    lo = max(0.02 * pvt.p_init, 25.0)
    p = np.linspace(lo, float(pvt.p_init), int(n))
    try:
        bt, bo, bg, rs = pvt.bt(p), pvt.bo(p), pvt.bg(p), pvt.rs(p)
    except Exception as exc:
        return [f"the PVT could not be evaluated over the pressure range: "
                f"{exc}"]

    rise = np.diff(bt)
    bad = np.flatnonzero(rise > 1e-12)
    if bad.size:
        out.append(
            f"Bt is NOT monotone: it falls as pressure falls between "
            f"{p[bad[0]]:,.0f} and {p[bad[-1] + 1]:,.0f} psia "
            f"(bubble point {pb:,.0f}), by up to "
            f"{float(np.max(rise[bad])):.5f} rb/STB. A total formation volume "
            f"factor cannot do that - one stock-tank barrel and its gas can "
            f"only expand on the way down.")

    # The peak has to be located to within the GRID, not to the psi. On a
    # 400-point sweep of a 3,800 psia range the spacing is about 9 psi, and
    # comparing an exact bubble point against the nearest grid node reported
    # every fluid - including a textbook black oil - as peaking 6 psi off.
    step = float(p[1] - p[0]) if len(p) > 1 else 1.0
    i_pk = int(np.argmax(bo))
    # A reservoir SATURATED at discovery has its bubble point at or above the
    # initial pressure, so within the pressure range that exists Bo peaks at
    # the top of that range and nowhere else. Checking against a bubble point
    # outside the swept interval flagged a perfectly ordinary saturated fluid
    # as unphysical.
    if pb >= float(pvt.p_init) - 2.5 * step:
        pass
    elif abs(float(p[i_pk]) - pb) > 2.5 * step:
        out.append(
            f"Bo does not peak at the bubble point: its maximum "
            f"{float(np.max(bo)):.4f} is at {p[i_pk]:,.0f} psia, "
            f"not at {pb:,.0f}.")
    if np.any(np.diff(bg) >= 0):
        out.append("Bg does not fall monotonically with pressure.")
    # Below the bubble point gas comes out of solution - that is what a
    # bubble point IS. An Rs that stays flat for hundreds of psi below it
    # describes an undersaturated oil wearing a bubble point's label, and the
    # oil expansion term goes to zero across that range.
    below = p < pb - 2.5 * step
    if int(below.sum()) >= 3:
        flat = np.abs(float(pvt.rsi) - rs[below]) < 1e-6 * max(pvt.rsi, 1.0)
        if int(flat.sum()) >= 3:
            span = float(pb - np.min(p[below][flat]))
            if span > 5.0 * step:
                out.append(
                    f"Rs stays at Rsi for {span:,.0f} psi BELOW the bubble "
                    f"point of {pb:,.0f} psia. Gas does not come out of "
                    f"solution there, so the oil expansion term is zero over "
                    f"that range. The declared bubble point and Rsi describe "
                    f"different fluids.")
    if np.any(rs < -1e-9) or np.any(rs > float(pvt.rsi) + 1e-6):
        out.append(f"Rs leaves [0, Rsi]: range "
                   f"{float(np.min(rs)):,.0f} to {float(np.max(rs)):,.0f} "
                   f"against Rsi {pvt.rsi:,.0f} scf/STB.")
    if pvt.boi <= 1.0:
        out.append(f"Boi is {pvt.boi:.4f}, at or below 1.0 rb/STB.")
    return out


@dataclass
class OilPVT:
    """Black-oil properties for one reservoir fluid.

    Everything downstream - the material balance, the GOR ceiling, the
    productivity index - needs Bo, Rs and Bg as functions of pressure. They
    come either from correlations keyed on the four numbers every well has
    (API, gas gravity, temperature, initial solution GOR) or, better, from a
    measured PVT table supplied by the user.

    A measured table always wins where it covers the pressure; correlations
    are the fallback and the report says which was used, because a Standing
    Bo on a fluid Standing never saw is an assumption, not a measurement.
    """
    api: float
    gas_gravity: float
    temperature_F: float
    rsi: float                                    # scf/STB at p_i
    p_init: float
    p_bubble: Optional[float] = None              # psia; derived if absent
    salinity_ppm: float = 30_000.0
    sw_initial: float = 0.25
    cf_per_psi: float = 4.0e-6                    # formation compressibility
    cw_per_psi: float = 3.0e-6                    # water compressibility
    bw: float = 1.02                              # water FVF, rb/STB
    y_n2: float = 0.0
    y_co2: float = 0.0
    y_h2s: float = 0.0
    # Optional measured table: pressure (psia) with any of Rs, Bo, Bg, muo.
    pvt_table: Optional[pd.DataFrame] = None
    source_note: str = ""

    def __post_init__(self):
        if self.api <= 0:
            raise ValueError("API gravity must be positive.")
        if not 0.0 < self.gas_gravity < 2.0:
            raise ValueError("Gas gravity must be between 0 and 2.")
        if self.rsi < 0:
            raise ValueError("Initial solution GOR cannot be negative.")
        if not 0.0 <= self.sw_initial < 1.0:
            raise ValueError("Initial water saturation must be in [0, 1).")
        if self.p_bubble is None:
            self.p_bubble = standing_bubble_point(
                self.rsi, self.api, self.gas_gravity, self.temperature_F)
        if self.p_bubble > self.p_init * 1.001:
            # A bubble point above the initial pressure means the reservoir was
            # never undersaturated, which is possible - a saturated reservoir
            # with an initial gas cap - but it also happens when Rsi was
            # entered for the wrong well. Say so rather than deciding.
            warnings.warn(
                f"bubble point {self.p_bubble:,.0f} psia is above the initial "
                f"pressure {self.p_init:,.0f} psia: the reservoir is saturated "
                "at discovery and carries a gas cap, OR the initial solution "
                "GOR belongs to another fluid. Check before reading the "
                "material balance.")
        self.tpc_R, self.ppc_psia = sutton_pseudocriticals(
            self.gas_gravity, self.y_n2, self.y_co2, self.y_h2s)
        self._table_cols = (set(self.pvt_table.columns)
                            if self.pvt_table is not None else set())

    # ---- basics ---------------------------------------------------------
    @property
    def T_R(self) -> float:
        return self.temperature_F + 459.67

    @property
    def oil_sg(self) -> float:
        return api_to_sg(self.api)

    @property
    def saturated_at_discovery(self) -> bool:
        return bool(self.p_bubble >= self.p_init * 0.999)

    def _from_table(self, col: str, p: np.ndarray) -> Optional[np.ndarray]:
        """Interpolate a measured column, or None if it is not usable here."""
        if self.pvt_table is None or col not in self._table_cols:
            return None
        t = self.pvt_table.dropna(subset=["pressure", col])
        if len(t) < 2:
            return None
        t = t.sort_values("pressure")
        return np.interp(np.asarray(p, float),
                         t["pressure"].to_numpy(float), t[col].to_numpy(float))

    # ---- gas ------------------------------------------------------------
    def z(self, p: np.ndarray) -> np.ndarray:
        """Gas deviation factor, from the gas module's Dranchuk-Abou-Kassem."""
        return gdca.z_factor(np.asarray(p, float), self.T_R,
                             self.gas_gravity, self.y_n2, self.y_co2,
                             self.y_h2s)

    def bg(self, p: np.ndarray) -> np.ndarray:
        """Gas formation volume factor, rb/scf."""
        tab = self._from_table("bg", p)
        if tab is not None:
            return tab
        return gas_fvf_rb_per_scf(np.asarray(p, float), self.T_R, self.z(p))

    # ---- oil ------------------------------------------------------------
    def rs(self, p: np.ndarray) -> np.ndarray:
        """Solution gas-oil ratio, scf/STB. Constant above the bubble point."""
        p = np.asarray(p, dtype=float)
        tab = self._from_table("rs", p)
        if tab is not None:
            base = tab
        else:
            base = standing_rs(p, self.api, self.gas_gravity,
                               self.temperature_F)
            # The SHAPE comes from the correlation; the ANCHOR comes from the
            # bubble point. Where the bubble point is derived from Rsi the
            # two already agree and this factor is 1. Where it is declared
            # and disagrees, the correlation is rescaled so that Rs(pb) = Rsi.
            #
            # Without this, a declared bubble point of 3,493 psia on a fluid
            # whose Rsi of 391 scf/STB puts Standing's bubble point at 1,673
            # left Rs pinned at Rsi for 1,820 psi BELOW the bubble point the
            # user gave: no gas came out of solution, Bo and Bt stayed
            # constant, and the oil expansion term Eo was identically zero
            # over the whole range the surveys sat in. The balance then
            # divided the withdrawal by almost nothing and returned a ceiling
            # of a billion barrels. The opposite case - a declared bubble
            # point BELOW the correlation's - put a step in Rs at pb.
            anchor = float(standing_rs(np.array([float(self.p_bubble)]),
                                       self.api, self.gas_gravity,
                                       self.temperature_F)[0])
            if anchor > 0:
                base = base * (float(self.rsi) / anchor)
        # Above the bubble point nothing more can dissolve: Rs is pinned at
        # Rsi. Correlations do not know that and will keep climbing.
        return np.where(p >= self.p_bubble, float(self.rsi),
                        np.minimum(base, float(self.rsi)))

    @property
    def correlation_bubble_point(self) -> float:
        """The bubble point Standing gives for this Rsi, whatever was declared."""
        return float(standing_bubble_point(self.rsi, self.api,
                                           self.gas_gravity,
                                           self.temperature_F))

    def bo(self, p: np.ndarray) -> np.ndarray:
        """Oil formation volume factor, rb/STB, on both sides of the bubble point."""
        p = np.asarray(p, dtype=float)
        tab = self._from_table("bo", p)
        if tab is not None:
            return tab
        rs = self.rs(p)
        sat = standing_bo(p, rs, self.api, self.gas_gravity,
                          self.temperature_F)
        bob = float(standing_bo(np.array([self.p_bubble]),
                                np.array([float(self.rsi)]), self.api,
                                self.gas_gravity, self.temperature_F)[0])
        # Above the bubble point the oil is single phase and SHRINKS as
        # pressure rises, by its own compressibility. Using the saturated
        # correlation there would have Bo keep growing with Rs it cannot
        # dissolve, and the material balance would read expansion that is not
        # happening.
        co = vasquez_beggs_co(p, self.rsi, self.api, self.gas_gravity,
                              self.temperature_F)
        undersat = bob * np.exp(co * (self.p_bubble - p))
        return np.where(p >= self.p_bubble, undersat, sat)

    def bt(self, p: np.ndarray) -> np.ndarray:
        """Total (two-phase) formation volume factor, rb/STB.

            Bt = Bo + (Rsi - Rs) Bg

        The volume a stock-tank barrel of original oil occupies in the
        reservoir once the gas that came out of it is counted too. Below the
        bubble point Bo falls while Bt keeps rising, and it is Bt, not Bo,
        that the material balance expands.
        """
        p = np.asarray(p, dtype=float)
        return self.bo(p) + (float(self.rsi) - self.rs(p)) * self.bg(p)

    def muo(self, p: np.ndarray) -> np.ndarray:
        """Oil viscosity, cp."""
        p = np.asarray(p, dtype=float)
        tab = self._from_table("muo", p)
        if tab is not None:
            return tab
        mu = beggs_robinson_muo(p, self.rs(p), self.api, self.temperature_F)
        pb = float(self.p_bubble)
        above = p > pb
        if np.any(above):
            mu_ob = float(beggs_robinson_muo(
                np.array([pb]), np.array([self.rsi]), self.api,
                self.temperature_F)[0])
            mu = np.where(above,
                          vasquez_beggs_muo_undersaturated(p, pb, mu_ob), mu)
        return mu

    def mug(self, p: np.ndarray) -> np.ndarray:
        """Gas viscosity, cp, from the gas module's Lee-Gonzalez-Eakin."""
        return gdca.gas_viscosity(np.asarray(p, float), self.T_R,
                                  self.gas_gravity, z=self.z(p),
                                  y_n2=self.y_n2, y_co2=self.y_co2,
                                  y_h2s=self.y_h2s)

    # ---- derived quantities the material balance needs -------------------
    @property
    def boi(self) -> float:
        return float(self.bo(np.array([self.p_init]))[0])

    @property
    def bti(self) -> float:
        return float(self.bt(np.array([self.p_init]))[0])

    @property
    def bgi(self) -> float:
        return float(self.bg(np.array([self.p_init]))[0])

    def summary(self) -> str:
        src = ("measured table for " + ", ".join(sorted(
            self._table_cols - {"pressure"})) if self._table_cols
            else "Standing / Vasquez-Beggs correlations")
        lines = [
            f"  API                         : {self.api:.1f}",
            f"  gas_gravity                 : {self.gas_gravity:.4f}",
            f"  temperature_F               : {self.temperature_F:.1f}",
            f"  Rsi                         : {self.rsi:,.0f} scf/STB",
            f"  p_init                      : {self.p_init:,.0f} psia",
            f"  p_bubble                    : {self.p_bubble:,.0f} psia"
            + ("   <-- SATURATED at discovery" if self.saturated_at_discovery
               else f"   (undersaturated by "
                    f"{self.p_init - self.p_bubble:,.0f} psi)"),
            f"  Boi / Bti / Bgi             : {self.boi:.4f} / {self.bti:.4f}"
            f" / {self.bgi:.6f} rb per STB, scf",
            f"  Sw_initial                  : {self.sw_initial:.3f}",
            f"  cf / cw                     : {self.cf_per_psi:.2e} / "
            f"{self.cw_per_psi:.2e} 1/psi",
            f"  properties from             : {src}",
            f"  fluid                       : {fluid_class(self)}",
        ]
        if self.source_note:
            lines.append(f"  note                        : {self.source_note}")
        # Rsi and the bubble point are two measurements of one thing. When
        # they disagree by a factor, one of them belongs to another fluid,
        # and nothing downstream can tell which - so it is said here.
        if not self._table_cols:
            pb_c = self.correlation_bubble_point
            if (np.isfinite(pb_c) and pb_c > 0
                    and abs(self.p_bubble - pb_c) / pb_c > 0.25):
                lines.append(
                    f"  CHECK Rsi AND pb            : Standing puts the bubble "
                    f"point for Rsi {self.rsi:,.0f} at {pb_c:,.0f} psia;\n"
                    f"                                {self.p_bubble:,.0f} "
                    f"was used ({self.p_bubble / pb_c:.2f}x). Rs is anchored "
                    f"to the bubble point used,\n"
                    f"                                so the two agree, but "
                    f"one of the inputs probably belongs to another\n"
                    f"                                fluid - and the "
                    f"material balance is only as right as whichever it is.")
        # Where the correlations are being extrapolated, say so once, next to
        # the properties they produced. They still return numbers outside
        # their range - they are smooth functions - and those numbers carry
        # no evidence.
        if not self._table_cols:
            oor = correlation_range_warnings(self)
            if oor:
                lines.append(
                    "  OUTSIDE THE CORRELATIONS' PUBLISHED RANGE:")
                for w in oor:
                    lines.append(f"      {w}")
                lines.append(
                    "      These still return numbers, and the numbers are "
                    "extrapolations of a curve\n      fitted to other "
                    "people's fluids. Everything below - the balance, the "
                    "GOR\n      ceiling, the productivity index - inherits "
                    "them. Supply a measured PVT table\n      if you have "
                    "one.")
        # Physics, not provenance. A correlation outside its published range
        # is an extrapolation; a correlation that breaks a conservation
        # statement is wrong, and that is worth saying whatever the
        # properties came from - including a measured table, which can also
        # be inconsistent.
        phys = pvt_physicality_warnings(self)
        if phys:
            lines.append("  THE FLUID MODEL IS NOT PHYSICAL:")
            for w in phys:
                lines.append(f"      {w}")
        return "\n".join(lines)


# ==============================================================================
# SECTION 2 -- MATERIAL BALANCE (HAVLENA-ODEH, THREE TERMS)
# ==============================================================================
#
# Dake's form of the general material balance, written as a straight line:
#
#     F = N [ Eo + m Eg + Efw ] + We
#
#     F    = Np [ Bo + (Rp - Rs) Bg ] + Wp Bw          underground withdrawal
#     Eo   = Bt - Bti                                  oil + its solution gas
#     Eg   = Boi (Bg/Bgi - 1)                          gas-cap expansion
#     Efw  = (1+m) Boi [ (cw Swc + cf)/(1 - Swc) ] dp  water and rock
#     m    = gas-cap volume / oil volume, at reservoir conditions
#
# Two unknowns sit inside the bracket (N and m) and a third (We) sits outside,
# and all three trade against one another. The discipline is the same one the
# gas module applies to its aquifer fits: quote the bound that follows from
# We >= 0 alone, quote the fit, and show the drift that tells you whether the
# assumed drive is the right one - never a single number with a tolerance.

# For a closed reservoir on the right gas cap, apparent N = F/Et is the SAME
# NUMBER at every survey. Any systematic departure means the bracket is
# incomplete - and the departure is not a slope. Marched against a known
# 50 MMstb tank with 9 MMrb of influx, apparent N runs 51 -> 115 -> 62 MMstb:
# it climbs while Et is small and the influx dominates, then falls back as
# solution gas comes out and Et accelerates. A straight line through that has
# a p-value of 0.14 and finds nothing. The SPREAD finds it immediately, so
# the spread is what the verdict keys on and the slope is reported beside it
# as a description of the shape.
APPARENT_N_SPREAD = 0.10
# A gas cap, once allowed for, should leave the apparent-N sequence genuinely
# flat. Water influx does not: allowing a spurious gas cap on a water-drive
# well still leaves 10-24 % of spread, because the influx grows with time and
# a gas cap cannot imitate that shape. Across eighteen synthetic wells the two
# families separated cleanly at 5 % with nothing in between.
GAS_CAP_RESIDUAL_SPREAD = 0.05     # (max-min)/min above this: the drive is not as assumed
APPARENT_N_MIN_DP = 25.0     # psi; below this F/Et is two small numbers divided
N_MAX_REL_SE = 0.50          # above this the regression has not found N


@dataclass
class MaterialBalanceOil:
    """Havlena-Odeh on an oil reservoir, with its own non-uniqueness on show."""
    n_ooip_stb: float = float("nan")
    n_stderr_stb: float = float("nan")
    m_gas_cap: float = 0.0
    m_fitted: bool = False
    m_stderr: float = float("nan")
    # The gas-cap fit was asked for but the surveys could not carry it: fewer
    # than three usable points at every m, or a minimum so flat it spans
    # most of the scan. The m reported is then the one supplied (or 0), and
    # it is labelled as such rather than as FITTED.
    m_fit_requested: bool = False
    m_fit_why_not: str = ""
    n_uninformative_ceiling: bool = False
    depletion_note: str = ""
    r2: float = float("nan")
    n_surveys: int = 0
    n_skipped: int = 0
    p_initial: float = float("nan")
    p_initial_estimated: bool = False
    p_initial_source: str = "as entered"
    ho_table: Optional[pd.DataFrame] = None
    n_ceiling_stb: float = float("nan")     # min(F/Et): the We >= 0 bound
    apparent_n_drift_pct: float = float("nan")
    apparent_n_p_value: float = float("nan")
    # The range of N over the gas-cap values the data cannot tell apart, and
    # that range of m. Quoting either alone is quoting half a locus.
    n_locus_stb: Optional[Tuple[float, float]] = None
    m_locus: Optional[Tuple[float, float]] = None
    apparent_n_spread: float = float("nan")  # (max-min)/min over the surveys
    apparent_n_min_stb: float = float("nan")
    apparent_n_max_stb: float = float("nan")
    # Surveys dropped from the apparent-N sequence for having too little
    # depletion behind them. They still count in the fit; F/Et is simply not
    # readable there, and the ceiling is only as tight as the points allow.
    n_thin_dp: int = 0
    # A gas cap that was assumed away rather than tested. Fitting with m = 0
    # on a reservoir that has one makes N absorb the gas cap's expansion, and
    # it does so quietly: on a synthetic well with m = 0.60 the fitted N came
    # back at 943 MMstb against a truth of 83, an eleven-fold error, with an
    # R2 of 0.997.
    gas_cap_indicated: bool = False
    m_indicated: float = float("nan")
    n_at_indicated_m_stb: float = float("nan")
    spread_at_indicated_m: float = float("nan")
    drive: str = "unknown"
    drive_note: str = ""
    we_implied_rb: float = float("nan")
    aquifer: Optional[Dict] = None
    note: str = ""
    trend_ok: bool = True

    @property
    def n_ooip_mstb(self) -> float:
        return self.n_ooip_stb / STB_PER_MSTB

    @property
    def exceeds_ceiling(self) -> bool:
        return bool(np.isfinite(self.n_ooip_stb)
                    and np.isfinite(self.n_ceiling_stb)
                    and self.n_ooip_stb > 1.05 * self.n_ceiling_stb)

    @property
    def n_rel_se(self) -> float:
        if not (np.isfinite(self.n_stderr_stb) and self.n_ooip_stb > 0):
            return float("inf")
        return float(self.n_stderr_stb / self.n_ooip_stb)

    def summary(self) -> str:
        lines = [
            f"  surveys used      : {self.n_surveys}"
            + (f" ({self.n_skipped} earliest dropped)" if self.n_skipped else ""),
            f"  p_initial         : {self.p_initial:,.0f} psia "
            f"({self.p_initial_source})",
        ]
        if not self.trend_ok:
            lines.append(f"  NOTE              : {self.note}")
            return "\n".join(lines)
        lines += [
            (f"  OOIP (N)          : {self.n_ooip_mstb:,.0f} Mstb"
             + (f" +/- {self.n_stderr_stb / STB_PER_MSTB:,.0f}"
                if np.isfinite(self.n_stderr_stb) else ""))
            if np.isfinite(self.n_ooip_stb) else
            "  OOIP (N)          : NOT DETERMINED - see the drive note below",
            (f"  R2                : {self.r2:.4f}"
             if np.isfinite(self.r2) and self.r2 >= 0 else
             # A line forced through the origin can fit worse than a flat
             # line at the mean, and R2 then goes negative without limit. The
             # field report printed "R2 : -821.7984" bare, next to an N quoted
             # to +/- 23 %.
             f"  R2                : {self.r2:.1f}  - BELOW ZERO: a flat line "
             "fits F better than N x Et does.\n                      F and "
             "Et are not proportional on these surveys, so the N above is "
             "not a\n                      slope the data support - see the "
             "drive note."
             if np.isfinite(self.r2) else "  R2                : -"),
            (f"  gas cap m         : NOT DETERMINED - fit requested, but\n"
             f"                      {self.m_fit_why_not};\n"
             f"                      "
             f"m = {self.m_gas_cap:.4f} used in its place"
             if self.m_fit_requested and not self.m_fitted else
             f"  gas cap m         : {self.m_gas_cap:.4f}"
             + (f" +/- {self.m_stderr:.4f}  (FITTED)" if self.m_fitted
                else "  (as entered)")
             + ("   - no gas cap assumed" if self.m_gas_cap == 0.0
                and not self.m_fitted else "")),
        ]
        if self.gas_cap_indicated:
            lines += [
                f"  GAS CAP INDICATED : the surveys are far better explained "
                f"by m = {self.m_indicated:.2f} than by the "
                f"m = {self.m_gas_cap:.2f}\n                      used above. "
                f"At that m the apparent-N spread falls from "
                f"{100 * self.apparent_n_spread:.0f} % to "
                f"{100 * self.spread_at_indicated_m:.0f} % and N becomes "
                f"{self.n_at_indicated_m_stb / STB_PER_MSTB:,.0f} Mstb,\n"
                f"                      against the "
                f"{self.n_ooip_mstb:,.0f} Mstb above. Assuming a gas cap away "
                f"does not remove it: N\n                      simply absorbs "
                f"its expansion, and the error runs to an order of magnitude. "
                f"Set m\n                      from structure and logs, or "
                f"turn on the gas-cap fit, before using N."]
        n_ceil_pts = (int(self.ho_table["apparent_N_usable"].sum())
                      if self.ho_table is not None
                      and "apparent_N_usable" in self.ho_table else 0)
        lines += [
            f"  N ceiling, We>=0  : {self.n_ceiling_stb / STB_PER_MSTB:,.0f} "
            "Mstb = min(F/Et)"
            + (f"   ({self.n_thin_dp} early survey(s) excluded: too little "
               f"depletion to read F/Et)" if self.n_thin_dp else "")
            + (f"\n                      from {n_ceil_pts} survey(s) only - "
               "a bound, but a loose one" if 0 < n_ceil_pts < 3 else ""),
        ]
        if self.depletion_note:
            lines.append(f"  depletion seen    : {self.depletion_note}")
        if self.n_uninformative_ceiling:
            lines.append(
                "  NOTE              : the ceiling is over 50x the oil "
                "produced, so it constrains nothing.\n"
                "                      Little pressure drop for this much "
                "production means strong support (aquifer\n"
                "                      or gas cap) or a connected volume far "
                "larger than the well drains; the\n"
                "                      surveys cannot tell which. The ceiling "
                "is not an estimate of oil in place.")
        if np.isfinite(self.apparent_n_spread):
            lines.append(
                f"  apparent N spread : {100 * self.apparent_n_spread:.1f} % "
                f"({self.apparent_n_min_stb / STB_PER_MSTB:,.0f} to "
                f"{self.apparent_n_max_stb / STB_PER_MSTB:,.0f} Mstb)"
                f"  -> {self.drive}")
            if np.isfinite(self.apparent_n_drift_pct):
                lines.append(
                    f"                      shape: {self.apparent_n_drift_pct:+.0f} % "
                    f"net slope (p {self.apparent_n_p_value:.2g}) - read the "
                    "spread, not the slope,\n                      because "
                    "influx makes the sequence rise then fall rather than "
                    "trend")
        if self.exceeds_ceiling:
            lines.append(
                f"  WARNING           : the fitted N is "
                f"{self.n_ooip_stb / self.n_ceiling_stb:.2f}x the We >= 0 "
                "ceiling. Something outside the\n                      "
                "bracket is supplying volume - water influx, or a gas cap "
                "larger than assumed.")
        if not np.isfinite(self.n_ooip_stb):
            pass                        # the drive note already says why
        elif self.n_rel_se > N_MAX_REL_SE:
            lines.append(
                f"  WARNING           : N carries a standard error of "
                f"{100 * self.n_rel_se:.0f} %, which is not a number\n"
                "                      anything should be held to. The "
                "surveys do not determine it.")
        if self.m_fitted:
            loc = ""
            if self.n_locus_stb and self.m_locus:
                loc = (f"\n                      The data cannot separate "
                       f"m {self.m_locus[0]:.2f}-{self.m_locus[1]:.2f} from "
                       f"N {self.n_locus_stb[0] / STB_PER_MSTB:,.0f}-"
                       f"{self.n_locus_stb[1] / STB_PER_MSTB:,.0f} Mstb:\n"
                       "                      that pair IS the answer.")
            lines.append(
                "  NOTE              : N and m were fitted together and are "
                "almost perfectly anti-correlated\n                      "
                "(measured at -1.00). A larger gas cap with a smaller oil "
                "volume fits\n                      equally well, so prefer "
                "an m from structure and logs where you have one."
                + loc)
        if np.isfinite(self.we_implied_rb) and self.we_implied_rb > 0:
            lines.append(
                f"  We implied        : {self.we_implied_rb / 1.0e6:,.2f} MMrb "
                "to date at the fitted N")
        if self.drive_note:
            lines.append(f"  drive             : {self.drive_note}")
        return "\n".join(lines)


def havlena_odeh_oil(pressure: np.ndarray,
                     np_stb: np.ndarray,
                     gp_scf: np.ndarray,
                     wp_stb: np.ndarray,
                     pvt: OilPVT,
                     m_gas_cap: float = 0.0,
                     p_initial: Optional[float] = None) -> pd.DataFrame:
    """The per-survey withdrawal and expansion terms, as a table.

    Everything the straight-line methods need, computed once and exposed, so
    the reader can plot F against Et themselves rather than taking a fitted
    slope on faith. `apparent_N` is F/Et: the oil in place each survey implies
    on its own, if all the drive energy is inside the bracket. Its DRIFT is
    the diagnostic - a flat sequence means the assumed drive is complete, and
    a rising one means something outside it is adding volume.
    """
    p = np.asarray(pressure, dtype=float)
    npv = np.asarray(np_stb, dtype=float)
    gpv = np.asarray(gp_scf, dtype=float)
    wpv = (np.asarray(wp_stb, dtype=float) if wp_stb is not None
           else np.zeros_like(p))
    pi = float(p_initial) if p_initial is not None else float(p[0])

    bo, rs, bg, bt = pvt.bo(p), pvt.rs(p), pvt.bg(p), pvt.bt(p)
    boi, bgi, bti = pvt.boi, pvt.bgi, pvt.bti

    # Cumulative produced GOR. Undefined before any oil has been produced, so
    # the first survey of a record that starts at Np = 0 contributes nothing
    # and is carried as a zero withdrawal rather than a division by zero.
    with np.errstate(divide="ignore", invalid="ignore"):
        rp = np.where(npv > 0, gpv / np.maximum(npv, 1e-12), 0.0)

    F = npv * (bo + (rp - rs) * bg) + wpv * pvt.bw          # rb
    Eo = bt - bti                                            # rb/STB
    Eg = boi * (bg / bgi - 1.0)                              # rb/STB
    swc = pvt.sw_initial
    dp = pi - p
    Efw = ((1.0 + m_gas_cap) * boi
           * ((pvt.cw_per_psi * swc + pvt.cf_per_psi) / max(1.0 - swc, 1e-9))
           * dp)
    Et = Eo + m_gas_cap * Eg + Efw
    with np.errstate(divide="ignore", invalid="ignore"):
        app_n = np.where(Et > 0, F / Et, np.nan)

    return pd.DataFrame({
        "pressure": p, "dp": dp, "Np_stb": npv, "Gp_scf": gpv,
        "Wp_stb": wpv, "Rp_scf_stb": rp,
        "Bo": bo, "Rs": rs, "Bg": bg, "Bt": bt,
        "F_rb": F, "Eo": Eo, "Eg": Eg, "Efw": Efw, "Et": Et,
        "apparent_N_stb": app_n,
        "apparent_N_usable": np.isfinite(app_n) & (app_n > 0) & (Et > 0),
    })


def material_balance_oil(pressure: np.ndarray,
                         np_stb: np.ndarray,
                         gp_scf: np.ndarray,
                         wp_stb: Optional[np.ndarray],
                         pvt: OilPVT,
                         m_gas_cap: Optional[float] = None,
                         fit_gas_cap: bool = False,
                         p_initial: Optional[float] = None,
                         p_initial_estimated: bool = False,
                         skip_early: int = 0,
                         min_depletion_psi: float = 50.0
                         ) -> MaterialBalanceOil:
    """Fit N (and optionally m) to the Havlena-Odeh straight line.

    With `fit_gas_cap` off, m is whatever you supply - normally from structure
    and logs, which is where it should come from - and N is the slope of F
    against Et through the origin. With it on, N and m are fitted together and
    the report says so, because the two are strongly correlated: a bigger gas
    cap expanding harder and a smaller oil volume fit nearly the same history.

    `skip_early` drops the earliest surveys. Early in the life the expansion
    terms are tiny and F/Et is the ratio of two small numbers, so the first
    points carry enormous leverage on any line through them.
    """
    p = np.asarray(pressure, dtype=float)
    ok = np.isfinite(p) & (p > 0) & np.isfinite(np_stb)
    if int(ok.sum()) < 3:
        return MaterialBalanceOil(
            trend_ok=False, n_surveys=int(ok.sum()),
            note="fewer than three usable pressure surveys")
    p = p[ok]
    npv = np.asarray(np_stb, float)[ok]
    gpv = np.asarray(gp_scf, float)[ok]
    wpv = (np.asarray(wp_stb, float)[ok] if wp_stb is not None
           else np.zeros_like(p))

    order = np.argsort(npv)
    p, npv, gpv, wpv = p[order], npv[order], gpv[order], wpv[order]
    n_skip = int(np.clip(skip_early, 0, max(len(p) - 3, 0)))
    if n_skip:
        p, npv, gpv, wpv = p[n_skip:], npv[n_skip:], gpv[n_skip:], wpv[n_skip:]

    # Where p_i comes from is not a detail: Eo, Efw and Bti all hang off it,
    # so the whole balance does. Falling back to the highest survey pressure
    # and then printing "(as entered)" was a false provenance claim - on a
    # tank whose true p_i was 3,800 psia the first survey read 3,543 because
    # a month of production had already happened, and the report presented
    # that 7 % error as a number the user had supplied.
    if p_initial is not None:
        pi = float(p_initial)
        pi_src = ("ESTIMATED by this tool, not measured"
                  if p_initial_estimated else "as entered")
    else:
        pi = float(np.max(p))
        pi_src = ("highest survey pressure - NOT an entered p_i; if the "
                  "reservoir\n                      was already producing "
                  "when that survey was taken, p_i is higher and N is biased")

    # An expansion term needs a pressure drop to exist. A record whose
    # surveys all sit within a few psi of p_i has no material balance in it,
    # however many surveys there are, and a line through that noise produces
    # an N with no relationship to the reservoir.
    if float(np.max(pi - p)) < min_depletion_psi:
        return MaterialBalanceOil(
            trend_ok=False, n_surveys=len(p), n_skipped=n_skip, p_initial=pi,
            p_initial_estimated=p_initial_estimated, p_initial_source=pi_src,
            note=(f"the surveys span only {float(np.max(pi - p)):,.0f} psi of "
                  f"depletion, below the {min_depletion_psi:,.0f} psi this "
                  "needs. There is no expansion to measure yet."))

    def _fit_for_m(m: float) -> Tuple[float, float, float, pd.DataFrame]:
        """N, its standard error and R2 for a given gas-cap ratio."""
        tab = havlena_odeh_oil(p, npv, gpv, wpv, pvt, m_gas_cap=m,
                               p_initial=pi)
        use = tab["apparent_N_usable"].to_numpy()
        if int(use.sum()) < 3:
            return float("nan"), float("nan"), float("nan"), tab
        x = tab["Et"].to_numpy(float)[use]
        y = tab["F_rb"].to_numpy(float)[use]
        # Through the origin: F = N Et. An intercept has no physical meaning
        # here - at zero expansion there is zero withdrawal - and fitting one
        # lets the line absorb a systematic error in p_i instead of showing it.
        n_hat = float(np.sum(x * y) / max(np.sum(x * x), 1e-30))
        resid = y - n_hat * x
        dof = max(len(x) - 1, 1)
        se = float(math.sqrt(max(np.sum(resid ** 2) / dof, 0.0)
                             / max(np.sum(x * x), 1e-30)))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else np.nan
        return n_hat, se, r2, tab

    m_in = 0.0 if m_gas_cap is None else float(m_gas_cap)
    m_se = float("nan")
    m_locus = None
    n_locus_range = None
    m_fit_ok = False
    m_fit_why_not = ""
    if fit_gas_cap:
        # Scan m on a coarse grid then polish. A one-dimensional search is
        # enough because N is solved exactly for each m, and a grid makes the
        # flatness of the objective visible instead of hiding it in an
        # optimiser's convergence message.
        grid = np.concatenate([np.linspace(0.0, 1.0, 41),
                               np.linspace(1.1, 5.0, 40)])
        costs = []
        for mg in grid:
            n_g, _, r2_g, tab_g = _fit_for_m(float(mg))
            use = tab_g["apparent_N_usable"].to_numpy()
            if not np.isfinite(n_g) or int(use.sum()) < 3:
                costs.append(np.inf); continue
            x = tab_g["Et"].to_numpy(float)[use]
            y = tab_g["F_rb"].to_numpy(float)[use]
            costs.append(float(np.sum((y - n_g * x) ** 2)))
        costs = np.asarray(costs, float)
        m_fit_ok = bool(np.isfinite(costs).any())
        if m_fit_ok:
            best = int(np.argmin(costs))
            # The half-width of the region within 10 % of the minimum cost is
            # a far more honest error bar than a curvature-based one on an
            # objective this flat.
            near = grid[np.isfinite(costs) & (costs <= costs[best] * 1.10)]
            m_se = (float(0.5 * (near.max() - near.min()))
                    if near.size > 1 else 0.0)
            # A 'minimum' that covers most of the scan is not a fit: every m
            # from none to a gas cap several times the oil zone fits alike.
            if near.max() - near.min() >= 0.8 * (grid.max() - grid.min()):
                m_fit_ok = False
                m_fit_why_not = (
                    f"every m from {near.min():.1f} to {near.max():.1f} fits "
                    "the surveys equally well")
            else:
                m_in = float(grid[best])
        else:
            # Before this check, an all-infinite cost vector sent argmin to
            # the first grid point, and the report printed 'm 0.0000 +/-
            # 2.5000 (FITTED)' and an N-m correlation note on a field where
            # neither N nor m had been fitted.
            m_fit_why_not = ("fewer than three surveys carry enough "
                             "depletion at any m")
        # N and m are almost perfectly anti-correlated - measured at -1.00 on
        # a marched tank with 15 psi of survey scatter - so the regression
        # standard error on N, which conditions on the fitted m, is not the
        # uncertainty in N. It read +/-0.3 % where repeated noise realisations
        # moved N by +/-9 %. The honest bar is the spread of N across the m
        # values the data cannot distinguish.
        if m_fit_ok:
            n_locus = [_fit_for_m(float(mg))[0] for mg in near]
            n_locus = [v for v in n_locus if np.isfinite(v)]
            m_locus = (float(near.min()), float(near.max()))
            n_locus_range = ((float(np.min(n_locus)), float(np.max(n_locus)))
                             if len(n_locus) > 1 else None)
        else:
            m_se = float("nan")

    # Where m was ASSUMED rather than fitted, test the assumption.
    #
    # The balance is perfectly willing to report a gas-cap reservoir as a
    # volumetric one with an N ten times too big, because N and m trade off:
    # a larger oil volume expanding a little fits the same history as a
    # smaller one next to a gas cap expanding a lot. Nothing in the m = 0 fit
    # complains. So the free-m fit is run anyway, purely as a check, and if it
    # is materially better the report says a gas cap is indicated and gives
    # the number the data prefer. It does NOT silently switch: a user who
    # supplied m from structure and logs has asserted something the data
    # cannot see, and that assertion should win - but not in silence.
    gas_cap_indicated = False
    m_ind = n_at_ind = spread_ind = float("nan")
    if not fit_gas_cap:
        def _spread_for(mg: float) -> Tuple[float, float]:
            n_g, _, _, tab_g = _fit_for_m(float(mg))
            u = (tab_g["apparent_N_usable"].to_numpy()
                 & (tab_g["dp"].to_numpy(float) >= APPARENT_N_MIN_DP))
            a = tab_g["apparent_N_stb"].to_numpy(float)[u]
            if a.size < 3 or not np.isfinite(np.nanmin(a)) \
                    or np.nanmin(a) <= 0:
                return float("nan"), n_g
            return float((np.nanmax(a) - np.nanmin(a)) / np.nanmin(a)), n_g

        # The m is chosen by LEAST SQUARES, the same objective the gas-cap
        # fit uses, and only then scored on the apparent-N spread. Picking it
        # by the spread instead - which was the first attempt - optimises the
        # wrong thing: on a tank with m = 0.60 it returned m = 0.35 and an
        # N 62 % high, where the least-squares scan returned m = 0.53 and an
        # N 13 % high. The spread is the evidence that a gas cap is there;
        # it is not the estimator for how big it is.
        base_spread, _ = _spread_for(m_in)
        scan = np.concatenate([np.linspace(0.0, 1.0, 41),
                               np.linspace(1.1, 3.0, 20)])
        best_m, best_cost, best_n = m_in, np.inf, float("nan")
        for mg in scan:
            n_g, _, _, tab_g = _fit_for_m(float(mg))
            u = tab_g["apparent_N_usable"].to_numpy()
            if not np.isfinite(n_g) or int(u.sum()) < 3:
                continue
            x = tab_g["Et"].to_numpy(float)[u]
            y = tab_g["F_rb"].to_numpy(float)[u]
            cost = float(np.sum((y - n_g * x) ** 2))
            if cost < best_cost:
                best_m, best_cost, best_n = float(mg), cost, n_g
        best_spread, _ = _spread_for(best_m)
        # Indicated only if the gas cap is material, the improvement is large,
        # and the result actually looks volumetric afterwards. Any one of
        # those alone would fire on noise.
        # The test is NOT conditioned on the m = 0 fit already looking bad.
        # Requiring that was the first version and it excluded precisely the
        # dangerous case: a well with m = 0.60 whose m = 0 fit gave an N
        # eleven times the truth while its apparent-N spread sat at 9.1 %,
        # under the volumetric threshold - so the balance called it a closed
        # tank, reported the wrong N as an answer, and said nothing.
        if (abs(best_m - m_in) > 0.05 and np.isfinite(best_spread)
                and np.isfinite(base_spread)
                and best_spread < 0.5 * base_spread
                and best_spread <= GAS_CAP_RESIDUAL_SPREAD):
            gas_cap_indicated = True
            m_ind, n_at_ind, spread_ind = best_m, best_n, best_spread

    n_hat, n_se, r2, tab = _fit_for_m(m_in)
    if n_locus_range is not None:
        # Widen, never narrow: the conditional error is still a floor.
        n_se = max(n_se, 0.5 * (n_locus_range[1] - n_locus_range[0]))
    use = tab["apparent_N_usable"].to_numpy()
    app = tab["apparent_N_stb"].to_numpy(float)

    # Apparent N is only meaningful once there is a pressure drop to divide
    # by. At the first survey F and Et are both near zero and their ratio is
    # numerical noise dressed as an answer.
    usable = use & (tab["dp"].to_numpy(float) >= APPARENT_N_MIN_DP)
    n_thin = int(use.sum() - usable.sum())
    if int(usable.sum()) < 3:
        usable, n_thin = use, 0
    tab["apparent_N_usable"] = usable

    # The We >= 0 ceiling. Influx can only ADD to the withdrawal, so
    # N Et = F - We <= F, and no oil in place may exceed min(F/Et). This is a
    # bound, not an estimate - the same structure as the gas module's
    # min(F/Eg), and biased low by survey scatter for the same reason. On the
    # marched 50 MMstb tank with 9 MMrb of influx it returned 51.2 MMstb
    # against a fitted N of 64.6, which is the whole argument for quoting it.
    app_u = app[usable]
    ceiling = float(np.nanmin(app_u)) if app_u.size else float("nan")
    a_min = ceiling
    a_max = float(np.nanmax(app_u)) if app_u.size else float("nan")
    spread = ((a_max - a_min) / a_min
              if np.isfinite(a_min) and a_min > 0 else float("nan"))

    # How much depletion the surveys actually saw, for the reader to weigh the
    # ceiling against. A field that has given up 9 MMstb for a few hundred psi
    # returns F/Et in the billions of barrels - an arithmetically correct
    # bound that says nothing, because the expansion term is tiny. Printing
    # it bare invites it to be read as a number about the reservoir.
    dp_all = tab["dp"].to_numpy(float)
    np_all = tab["Np_stb"].to_numpy(float)
    depletion_note = ""
    uninformative_ceiling = False
    if dp_all.size and np.isfinite(dp_all).any():
        k = int(np.nanargmax(dp_all))
        np_max = float(np.nanmax(np_all)) if np.isfinite(np_all).any() else 0.0
        depletion_note = (
            f"deepest survey is {dp_all[k]:,.0f} psi below p_i "
            f"({100 * dp_all[k] / max(pi, 1e-9):.0f} %) after "
            f"{np_all[k] / STB_PER_MSTB:,.0f} Mstb produced")
        if np.isfinite(ceiling) and np_max > 0 and ceiling > 50.0 * np_max:
            uninformative_ceiling = True

    # The slope is kept as a description of the SHAPE, not as the verdict.
    drift, p_val = float("nan"), float("nan")
    if int(usable.sum()) >= 4:
        xi, yi = npv[usable], app[usable]
        lr = stats.linregress(xi, yi)
        if np.isfinite(lr.slope) and yi.mean() > 0:
            drift = float(100.0 * lr.slope * (xi.max() - xi.min()) / yi.mean())
            p_val = float(lr.pvalue)

    # The regression standard error is a statement about scatter around a
    # line, not about how well the surveys agree on N. On a clean record it
    # came back at +/- 17 Mstb on 50,240 - a claim to know oil in place to
    # three parts in ten thousand - while the apparent N read off those same
    # surveys ranged over 8 %. The surveys' own disagreement is the better
    # measure and it is used as a FLOOR: widen, never narrow.
    if (np.isfinite(spread) and np.isfinite(a_min) and np.isfinite(a_max)
            and np.isfinite(n_hat) and n_hat > 0):
        se_from_surveys = 0.5 * (a_max - a_min)
        if np.isfinite(n_se):
            n_se = max(n_se, se_from_surveys)
        else:
            n_se = se_from_surveys

    n_usable = int(usable.sum())
    if not np.isfinite(n_hat) or n_hat <= 0:
        # No N, so no verdict about it. The first version went on to compute
        # a spread from whatever surveys were left and printed "volumetric -
        # the fitted N can be read as oil in place" two lines under
        # "OOIP (N): nan", on a field where N could not be fitted at all.
        drive = "undetermined"
        note = (f"N could not be fitted: {n_usable} survey(s) carry enough "
                f"depletion to read F/Et, and a\n                      line "
                "through the origin needs three. Nothing here - the spread, "
                "the ceiling, the\n                      drive - is evidence "
                "about oil in place until there are more surveys below p_i.")
        spread = float("nan")
    elif n_usable < 3 or not np.isfinite(spread):
        drive = "undetermined"
        note = (f"only {n_usable} survey(s) carry enough depletion to read "
                "F/Et. A spread needs at\n                      least three "
                "points to mean anything; with fewer, 'the surveys agree' "
                "is\n                      true of any two numbers.")
        spread = float("nan")
    elif spread <= APPARENT_N_SPREAD and gas_cap_indicated:
        # The apparent-N sequence can look flat and the N still be wrong by
        # an order of magnitude, because a gas cap that is assumed away is
        # absorbed into N rather than showing up as drift. Reporting
        # "volumetric - the fitted N can be read as oil in place" next to a
        # gas-cap warning leaves the reader to resolve a contradiction the
        # tool created.
        drive = "volumetric ONLY ONCE THE GAS CAP IS ALLOWED FOR"
        note = (f"apparent N holds to {100 * spread:.1f} % at the m used "
                "above, but that m is not the one the\n                      "
                "data prefer - see the gas-cap line. A closed reservoir with "
                "the WRONG gas cap\n                      also gives a flat "
                "sequence, so flatness here is not evidence that N is right. "
                "Fix\n                      m before reading N as oil in "
                "place.")
    elif spread <= APPARENT_N_SPREAD:
        drive = "volumetric (depletion)"
        note = (f"apparent N holds to {100 * spread:.1f} % across every "
                "survey, which is what a closed\n                      "
                "reservoir on the assumed gas cap looks like. The fitted N "
                "can be read as oil\n                      in place.")
    else:
        drive = "NOT volumetric - influx or a larger gas cap"
        note = (f"apparent N moves {100 * spread:.0f} % across the surveys "
                f"({a_min / 1e6:,.1f} to {a_max / 1e6:,.1f} MMstb).\n"
                "                      A closed reservoir gives the same "
                "number every time, so volume is\n                      "
                "arriving from outside the bracket: water influx, or a gas "
                "cap larger than the m\n                      used. The "
                "fitted N absorbs it and will be too large - the We >= 0 "
                "ceiling\n                      is the number to hold to.")

    we = float("nan")
    if np.isfinite(n_hat) and use.any():
        we_series = tab["F_rb"].to_numpy(float) - n_hat * tab["Et"].to_numpy(float)
        we = float(we_series[use][-1])

    return MaterialBalanceOil(
        n_ooip_stb=n_hat, n_stderr_stb=n_se, m_gas_cap=m_in,
        m_fitted=bool(fit_gas_cap and m_fit_ok), m_stderr=m_se, r2=r2,
        m_fit_requested=bool(fit_gas_cap), m_fit_why_not=m_fit_why_not,
        n_uninformative_ceiling=bool(uninformative_ceiling),
        depletion_note=depletion_note,
        n_surveys=len(p), n_skipped=n_skip, p_initial=pi,
        p_initial_estimated=bool(p_initial_estimated),
        p_initial_source=pi_src,
        ho_table=tab, n_ceiling_stb=ceiling, n_thin_dp=n_thin,
        apparent_n_drift_pct=drift, apparent_n_p_value=p_val,
        n_locus_stb=n_locus_range, m_locus=m_locus,
        apparent_n_spread=spread, apparent_n_min_stb=a_min,
        apparent_n_max_stb=a_max,
        gas_cap_indicated=bool(gas_cap_indicated), m_indicated=m_ind,
        n_at_indicated_m_stb=n_at_ind, spread_at_indicated_m=spread_ind,
        drive=drive, drive_note=note, we_implied_rb=we)


# ==============================================================================
# SECTION 3 -- PRODUCTION DATA AND QC
# ==============================================================================

@dataclass
class OilQCReport:
    """What the QC step did, so it can be reported rather than hidden."""
    n_input: int = 0
    n_after_screen: int = 0
    n_zero_or_negative_oil: int = 0
    n_nonfinite: int = 0
    n_outliers_removed: int = 0
    n_low_uptime_removed: int = 0
    n_duplicate_dates: int = 0
    n_negative_water: int = 0
    n_pressure_surveys: int = 0
    excluded_np_mstb: float = 0.0
    excluded_np_frac: float = 0.0
    plateau_end_days: Optional[float] = None
    # The plateau end came from the 75 % guard rail, not from the data: the
    # smoothed rate had not fallen to 92 % of its peak by then (or at all).
    plateau_capped: bool = False
    plateau_uncapped_days: Optional[float] = None
    bdf_start_days: Optional[float] = None
    fit_start_days: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"  rows in                 : {self.n_input}",
            f"  rows retained           : {self.n_after_screen}",
            f"  zero/negative oil       : {self.n_zero_or_negative_oil}",
            f"  non-finite values       : {self.n_nonfinite}",
            f"  rate outliers removed   : {self.n_outliers_removed}",
            f"  low-uptime rows removed : {self.n_low_uptime_removed}",
        ]
        if self.n_duplicate_dates:
            lines.append(f"  duplicate dates dropped : {self.n_duplicate_dates}")
        if self.n_negative_water:
            lines.append(f"  negative water readings : {self.n_negative_water}"
                         " (clipped to zero)")
        if self.excluded_np_mstb > 0:
            lines.append(
                f"  volume in dropped rows  : {self.excluded_np_mstb:,.0f} "
                f"Mstb oil ({100 * self.excluded_np_frac:.1f} % of the "
                "cumulative)")
            if self.excluded_np_frac > 0.10:
                lines.append(
                    "  WARNING                 : those rows are excluded from "
                    "the decline fit but their oil\n"
                    "                            still counts in Np, so it "
                    "feeds the material balance and\n"
                    "                            the EUR. Check them before "
                    "trusting either.")
        if self.plateau_end_days is not None:
            lines.append(f"  plateau ends            : day "
                         f"{self.plateau_end_days:.0f} "
                         f"({self.plateau_end_days / DAYS_PER_YEAR:.2f} yr)"
                         + ("   <-- GUARD RAIL, NOT FOUND"
                            if self.plateau_capped else ""))
            if self.plateau_capped:
                lines.append(
                    "                            the smoothed rate "
                    + ("did not fall to 92 % of its peak until day "
                       f"{self.plateau_uncapped_days:,.0f}"
                       if self.plateau_uncapped_days is not None
                       else "never fell to 92 % of its peak")
                    + ",\n                            so the plateau end was "
                    "set by the rule that at most 75 % of the\n"
                    "                            record is discarded. The fit "
                    "window is that rule, not a detected\n"
                    "                            plateau - look at the rate "
                    "plot and set the fit window yourself.")
        if self.bdf_start_days is not None:
            lines.append(f"  BDF start estimate      : day "
                         f"{self.bdf_start_days:.0f} "
                         f"({self.bdf_start_days / DAYS_PER_YEAR:.2f} yr)")
        if self.fit_start_days is not None:
            lines.append(f"  decline fit starts      : day "
                         f"{self.fit_start_days:.0f} "
                         f"({self.fit_start_days / DAYS_PER_YEAR:.2f} yr)")
        if self.n_pressure_surveys:
            lines.append(f"  pressure surveys        : {self.n_pressure_surveys}")
        for n in self.notes:
            lines.append(f"  note: {n}")
        return "\n".join(lines)


OIL_COLUMN_ALIASES: Dict[str, Sequence[str]] = {
    "date": ("date", "dates", "prod_date", "month", "period", "time"),
    "well": ("well", "well_name", "wellname", "uwi", "api_no", "completion"),
    "days_on": ("days_on", "days_online", "uptime_days", "producing_days",
                "onstream_days", "prod_days", "uptime", "on_stream_days"),
    "q_oil": ("q_oil", "qoil", "oil", "oil_rate", "bopd", "stbd", "qo",
              "oil_bbl", "oil_stb", "liquid_oil", "pdoil"),
    "q_gas": ("q_gas", "qgas", "gas", "gas_rate", "mscfd", "mcfd", "qg",
              "gas_mscf", "pdgas"),
    "q_water": ("q_water", "qwater", "water", "water_rate", "bwpd", "qw",
                "water_bbl", "pdwat"),
    "p_wf": ("p_wf", "pwf", "bhp", "flowing_pressure", "fbhp",
             "bottomhole_pressure", "p_bh"),
    "p_res": ("p_res", "pres", "p_avg", "static_pressure", "sbhp", "p_r",
              "reservoir_pressure", "shut_in_pressure"),
    "p_wh": ("p_wh", "pwh", "thp", "wellhead_pressure", "ftp"),
    "q_water_inj": ("q_water_inj", "winj", "water_injected", "injection",
                    "q_inj", "wi"),
}


def map_oil_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename whatever the export called things to the names used here."""
    lower = {str(c).strip().lower().replace(" ", "_").replace("-", "_"): c
             for c in df.columns}
    out = {}
    for canon, aliases in OIL_COLUMN_ALIASES.items():
        for a in aliases:
            if a in lower:
                out[lower[a]] = canon
                break
    return df.rename(columns=out)


@dataclass
class OilProductionData:
    """One well's history, QC'd, with cumulatives that survive the QC.

    The same discipline as the gas module: cumulatives are accumulated over
    every producing month BEFORE any row is filtered, because a rate outlier
    is a statement about the rate, not about the volume that was produced.
    The rows the QC drops therefore still count in Np, Gp and Wp - and the QC
    report says how much volume that is, since a single bad cell can move the
    material balance without changing a row count.
    """
    df: pd.DataFrame
    pvt: OilPVT
    well: str
    qc: OilQCReport
    rate_basis: str = "stream-day"
    full_df: Optional[pd.DataFrame] = None

    # ---- array views ----------------------------------------------------
    @property
    def t(self) -> np.ndarray:
        return self.df["t"].to_numpy(float)

    @property
    def q_oil(self) -> np.ndarray:
        return self.df["q_oil"].to_numpy(float)

    @property
    def q_gas(self) -> np.ndarray:
        return self.df["q_gas"].to_numpy(float)

    @property
    def q_water(self) -> np.ndarray:
        return (self.df["q_water"].to_numpy(float)
                if "q_water" in self.df.columns
                else np.zeros(len(self.df)))

    @property
    def Np(self) -> np.ndarray:
        return self.df["Np_stb"].to_numpy(float)

    @property
    def Gp(self) -> np.ndarray:
        return self.df["Gp_scf"].to_numpy(float)

    @property
    def Wp(self) -> np.ndarray:
        return self.df["Wp_stb"].to_numpy(float)

    @property
    def gor(self) -> np.ndarray:
        return self.df["gor"].to_numpy(float)

    @property
    def water_cut(self) -> np.ndarray:
        return self.df["water_cut"].to_numpy(float)

    @property
    def p_wf(self) -> Optional[np.ndarray]:
        return (self.df["p_wf"].to_numpy(float)
                if "p_wf" in self.df.columns else None)

    @property
    def surveys(self) -> pd.DataFrame:
        """Static pressure surveys with their cumulatives, from the FULL record.

        A gauge reading is a measurement of the reservoir and has nothing to
        do with whether that month's rate passed the QC, so the rate filters
        have no business removing it.
        """
        src = self.full_df if self.full_df is not None else self.df
        cols = ["date", "t", "p_res", "p_wf", "Np_stb", "Gp_scf", "Wp_stb"]
        if "p_res" not in src.columns:
            return pd.DataFrame(columns=cols)
        pr = pd.to_numeric(src["p_res"], errors="coerce")
        keep = np.isfinite(pr) & (pr > 0)
        return src.loc[keep, [c for c in cols if c in src.columns]] \
                  .reset_index(drop=True)

    @property
    def flowing_tests(self) -> pd.DataFrame:
        """Producing months carrying a flowing pressure, from the FULL record."""
        src = self.full_df if self.full_df is not None else self.df
        cols = ["date", "t", "p_wf", "p_res", "q_oil", "q_gas", "q_water",
                "Np_stb"]
        if "p_wf" not in src.columns:
            return pd.DataFrame(columns=cols)
        pw = pd.to_numeric(src["p_wf"], errors="coerce")
        keep = (np.isfinite(pw) & (pw > 0)
                & (pd.to_numeric(src["q_oil"], errors="coerce") > 0))
        return src.loc[keep, [c for c in cols if c in src.columns]] \
                  .reset_index(drop=True)

    def window(self, t_min: Optional[float] = None,
               t_max: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        m = np.ones(len(self.df), dtype=bool)
        if t_min is not None:
            m &= self.t >= t_min
        if t_max is not None:
            m &= self.t <= t_max
        return self.t[m], self.q_oil[m]

    # ---- construction ---------------------------------------------------
    @classmethod
    def prepare(cls, df: pd.DataFrame, pvt: OilPVT, well: str = "WELL",
                rate_basis: str = "stream-day",
                min_uptime_frac: float = 0.35,
                outlier_sigma: Optional[float] = 3.5,
                outlier_window: int = 11,
                drop_leading_zeros: bool = True,
                detect_bdf: bool = True) -> "OilProductionData":
        qc = OilQCReport()
        d = map_oil_columns(df).copy().reset_index(drop=True)
        qc.n_input = len(d)
        if "q_oil" not in d.columns:
            raise ValueError("An oil rate column is required (e.g. 'q_oil' "
                             "in STB/d).")
        if "date" not in d.columns:
            raise ValueError("A date column is required.")

        d["date"] = parse_dates(d["date"])
        d = d.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        dup = d["date"].duplicated(keep="first")
        if bool(dup.any()):
            qc.n_duplicate_dates = int(dup.sum())
            dup_any = d["date"].duplicated(keep=False)
            differ = (d.loc[dup_any].groupby("date")["q_oil"].nunique() > 1)
            qc.notes.append(
                f"{int(dup.sum())} row(s) repeat a date already in the record "
                "and were dropped."
                + (f" {int(differ.sum())} carried DIFFERENT oil rates, which "
                   "usually means two wells have been merged into one series."
                   if bool(differ.any()) else ""))
            d = d.loc[~dup].reset_index(drop=True)

        for col in ("q_oil", "q_gas", "q_water", "q_water_inj",
                    "p_wf", "p_wh", "p_res", "days_on"):
            if col in d.columns:
                d[col] = pd.to_numeric(d[col], errors="coerce")
        if "q_gas" not in d.columns:
            d["q_gas"] = 0.0
            qc.notes.append("No gas column; GOR diagnostics are unavailable.")

        # Forward-differenced period, as in the gas module: a month's length is
        # the gap to the NEXT record, not the calendar length of its own month.
        dt = (d["date"].shift(-1) - d["date"]).dt.days.astype(float)
        if len(dt) > 1:
            dt.iloc[-1] = dt.iloc[-2] if np.isfinite(dt.iloc[-2]) else 30.4375
        else:
            dt.iloc[-1] = 30.4375
        d["period_days"] = dt.fillna(30.4375).clip(lower=1.0)
        if "days_on" in d.columns:
            d["days_on"] = d["days_on"].fillna(d["period_days"]).clip(
                lower=0.0, upper=d["period_days"] + 1.0)
        else:
            d["days_on"] = d["period_days"]
            qc.notes.append("No uptime column; assumed fully on-stream.")
        d["uptime_frac"] = d["days_on"] / d["period_days"]

        eff_days = d["days_on"].where(d["days_on"] > 0, d["period_days"])
        if rate_basis == "volume":
            for src, dst in (("q_oil", "vol_oil"), ("q_gas", "vol_gas"),
                             ("q_water", "vol_water")):
                if src in d.columns:
                    d[dst] = d[src].astype(float)
                    d[src] = d[dst] / eff_days
            qc.notes.append("Input read as PERIOD VOLUMES; rates are "
                            "volume / days-on.")
        else:
            d["vol_oil"] = d["q_oil"] * eff_days
            d["vol_gas"] = d["q_gas"].fillna(0.0) * eff_days
            if "q_water" in d.columns:
                d["vol_water"] = d["q_water"] * eff_days
            if rate_basis == "calendar-day":
                for r, v in (("q_oil", "vol_oil"), ("q_gas", "vol_gas"),
                             ("q_water", "vol_water")):
                    if v in d.columns:
                        d[r] = d[v] / d["period_days"]
            elif rate_basis != "stream-day":
                raise ValueError("rate_basis must be 'stream-day', "
                                 "'calendar-day' or 'volume'.")

        qc.n_nonfinite = int((~np.isfinite(d["q_oil"])).sum())
        d["q_oil"] = d["q_oil"].where(np.isfinite(d["q_oil"]), 0.0)
        qc.n_zero_or_negative_oil = int((d["q_oil"] <= 0).sum())

        d["t"] = (d["date"] - d["date"].iloc[0]).dt.days.astype(float)
        if drop_leading_zeros and (d["q_oil"] > 0).any():
            first = int(d["q_oil"].gt(0).idxmax())
            if first > 0:
                # A zero-rate row BEFORE first oil is usually a pre-production
                # pressure survey. Keep it only if it carries one.
                pre = d.iloc[:first]
                keep_pre = (pre["p_res"].notna() if "p_res" in pre.columns
                            else pd.Series(False, index=pre.index))
                d = pd.concat([d.iloc[:first][keep_pre], d.iloc[first:]])
                d = d.reset_index(drop=True)
                d["t"] = (d["date"] - d.loc[d["q_oil"].gt(0).idxmax(),
                                            "date"]).dt.days.astype(float)
                eff_days = d["days_on"].where(d["days_on"] > 0, d["period_days"])

        # Cumulatives from the period volumes, before any filtering.
        d["vol_gas"] = d.get("vol_gas", pd.Series(0.0, index=d.index)).fillna(0.0)
        d["Np_stb"] = np.cumsum(d["vol_oil"])
        d["Gp_scf"] = np.cumsum(d["vol_gas"]) * SCF_PER_MSCF   # Mscf -> scf
        if "vol_water" in d.columns:
            _vw = d["vol_water"].fillna(0.0)
            qc.n_negative_water = int((_vw < 0).sum())
            d["Wp_stb"] = np.cumsum(_vw.clip(lower=0.0))
        else:
            d["Wp_stb"] = 0.0

        with np.errstate(divide="ignore", invalid="ignore"):
            d["gor"] = np.where(d["q_oil"] > 0,
                                d["q_gas"] * SCF_PER_MSCF / d["q_oil"],
                                np.nan)                        # scf/STB
            liq = d["q_oil"] + d.get("q_water", 0.0)
            d["water_cut"] = np.where(liq > 0,
                                      d.get("q_water", 0.0) / liq, np.nan)

        # -- exclusions (cumulatives above are already correct) -------------
        keep = d["q_oil"].to_numpy() > 0
        low = d["uptime_frac"].to_numpy() < min_uptime_frac
        qc.n_low_uptime_removed = int((low & (d["q_oil"].to_numpy() > 0)).sum())
        keep &= ~low

        producing = d["q_oil"].to_numpy() > 0
        if outlier_sigma is not None and producing.sum() >= outlier_window:
            q = d["q_oil"].to_numpy(float)
            lq = np.where(producing, np.log(np.where(producing, q, 1.0)), np.nan)
            med = pd.Series(lq).rolling(outlier_window, center=True,
                                        min_periods=3).median().to_numpy()
            resid = lq - med
            mad = np.nanmedian(np.abs(resid - np.nanmedian(resid)))
            scale = 1.4826 * mad if mad > 0 else np.nanstd(resid)
            if scale and np.isfinite(scale) and scale > 0:
                out = np.isfinite(resid) & (np.abs(resid) > outlier_sigma * scale)
                qc.n_outliers_removed = int(out.sum())
                keep &= ~out

        last_prod = np.flatnonzero(producing)
        if last_prod.size:
            keep[last_prod[-1]] = True
        if keep.sum() < 4:
            raise ValueError(f"[{well}] Fewer than 4 usable points after QC.")

        vol_all = float(d["vol_oil"].sum())
        vol_drop = float(d["vol_oil"][~keep].sum())
        qc.excluded_np_mstb = max(vol_drop, 0.0) / STB_PER_MSTB
        qc.excluded_np_frac = (vol_drop / vol_all) if vol_all > 0 else 0.0

        full = d.copy()
        d = d[keep].reset_index(drop=True)
        qc.n_after_screen = len(d)
        if "p_res" in full.columns:
            qc.n_pressure_surveys = int(np.isfinite(
                pd.to_numeric(full["p_res"], errors="coerce")).sum())

        obj = cls(df=d, pvt=pvt, well=well, qc=qc, rate_basis=rate_basis,
                  full_df=full.reset_index(drop=True))

        if detect_bdf:
            p_idx, p_t = detect_decline_start(obj.t, obj.q_oil)
            qc.plateau_end_days = p_t
            # Re-run the plateau test without its cap, to say whether the cap
            # decided it. On a 54-year field record the fit window started at
            # exactly 75 % of the retained points and the report presented it
            # as a detected plateau end.
            try:
                nq = len(obj.t)
                if p_idx is not None and nq >= 6:
                    qs = gdca._smooth(np.asarray(obj.q_oil, float),
                                      min(5, (nq // 3) | 1))
                    ipk = int(np.argmax(qs))
                    blw = np.flatnonzero(qs[ipk:] < 0.92 * qs[ipk])
                    raw = int(ipk + blw[0]) if blw.size else None
                    cap_i = max(0, min(int(0.75 * nq), nq - 4))
                    if (raw is None or raw > cap_i) and p_idx == cap_i:
                        qc.plateau_capped = True
                        qc.plateau_uncapped_days = (float(obj.t[raw])
                                                    if raw is not None
                                                    else None)
            except (ValueError, IndexError, AttributeError):
                pass
            mask = obj.t >= (p_t if p_t is not None else obj.t[0])
            b_idx, b_t = (detect_bdf_start(obj.t[mask], obj.q_oil[mask])
                          if mask.sum() >= 12 else (None, None))
            qc.bdf_start_days = b_t
            starts = [v for v in (p_t, b_t) if v is not None]
            qc.fit_start_days = max(starts) if starts else None
            if qc.fit_start_days is not None:
                n_excl = int(np.sum(obj.t < float(qc.fit_start_days)))
                if n_excl > 0:
                    qc.notes.append(
                        f"{n_excl} of {len(obj.t)} retained points excluded "
                        "from the decline fit: plateau and pre-boundary-"
                        "dominated flow.")
        return obj


# ==============================================================================
# SECTION 3b -- AQUIFER HISTORY MATCH
# ==============================================================================
#
# Havlena-Odeh above fits N to a straight line and treats influx as something
# to detect and refuse. This section does what MBAL's regression does instead:
# it SIMULATES the tank forward through the production history with an
# aquifer attached, predicts the pressure at every survey, and adjusts N (and
# optionally m) and the aquifer until the predictions match.
#
# The difference from MBAL is in what is regressed. A radial Van Everdingen-
# Hurst aquifer is described by porosity, thickness, encroachment angle,
# reservoir radius, permeability, water viscosity, total compressibility and
# the outer/inner radius ratio - but the pressure history only ever sees three
# combinations of them:
#
#     U   = 1.119 phi ct h ro^2 (theta/360)       rb/psi     aquifer constant
#     t_c = phi mu_w ct ro^2 / (0.006328 k)       days       tD = t / t_c
#     reD = r_aquifer / r_o                                  aquifer size
#
# Regressing on phi, h and theta separately regresses on directions in which
# the answer does not change at all; a 170-degree aquifer 125 ft thick gives
# exactly the influx of an 85-degree one 250 ft thick. So the regression is on
# U, t_c and reD (or, for Fetkovich, on Wei and a time constant), and rock
# properties are only used afterwards to translate the answer - with the
# statement that only their products are determined.
#
# The other difference is that the fit refuses to run without enough surveys
# to leave degrees of freedom over. A regression with as many parameters as
# surveys always fits perfectly, and a perfect fit there says nothing about N.

AQUIFER_KINDS = ("none", "fetkovich", "radial")
VEH_U_CONST = 1.119          # rb/(psi ft^3) * ft^3 ... U = 1.119 phi ct h ro^2 f
VEH_TD_CONST = 0.006328      # tD = 0.006328 k t / (phi mu ct ro^2), t in days
AQ_MIN_DOF = 3               # surveys beyond the parameter count, at least
AQ_MAX_STEPS = 100           # simulation steps across the history
AQ_MAX_RANGE_RATIO = 3.0     # 95 % range wider than this: N not determined
_IND = "\n" + " " * 22


def _stehfest_weights(n: int = 12) -> np.ndarray:
    """Gaver-Stehfest weights for numerical Laplace inversion."""
    h = n // 2
    v = np.zeros(n)
    for i in range(1, n + 1):
        s = 0.0
        for k in range((i + 1) // 2, min(i, h) + 1):
            s += (k ** h * math.factorial(2 * k)
                  / (math.factorial(h - k) * math.factorial(k)
                     * math.factorial(k - 1) * math.factorial(i - k)
                     * math.factorial(2 * k - i)))
        v[i - 1] = (-1) ** (h + i) * s
    return v


_STEHFEST = _stehfest_weights(12)


def veh_wd(td: np.ndarray, red: float = float("inf")) -> np.ndarray:
    """Van Everdingen-Hurst dimensionless cumulative influx, radial aquifer.

    Constant-terminal-pressure solution with a no-flow outer boundary at
    `red` (infinite if not finite), inverted from Laplace space by Stehfest.
    The Bessel functions are the exponentially SCALED ones, arranged so that
    no term overflows at large arguments: the unscaled I1(reD*sqrt(s)) passes
    1e308 for reD*sqrt(s) above about 710, which a 20:1 aquifer at early
    dimensionless time reaches immediately.
    """
    from scipy import special as sp
    td = np.atleast_1d(np.asarray(td, dtype=float))
    out = np.zeros_like(td)
    pos = td > 0
    if not pos.any():
        return out
    t = td[pos][:, None]
    k = np.arange(1, len(_STEHFEST) + 1)[None, :]
    s = k * math.log(2.0) / t
    u = np.sqrt(s)
    if not np.isfinite(red):
        lap = sp.kve(1, u) / (s * u * sp.kve(0, u))
    else:
        a, b = u, float(red) * u
        e = np.exp(2.0 * (a - b))
        num = sp.ive(1, b) * sp.kve(1, a) - sp.kve(1, b) * sp.ive(1, a) * e
        den = sp.ive(0, a) * sp.kve(1, b) * e + sp.kve(0, a) * sp.ive(1, b)
        lap = num / (s * u * den)
    val = math.log(2.0) / t[:, 0] * (lap @ _STEHFEST)
    out[pos] = np.maximum(val, 0.0)
    if np.isfinite(red):
        out = np.minimum(out, 0.5 * (float(red) ** 2 - 1.0))
    return out


def _edwardson_wd(td: np.ndarray) -> np.ndarray:
    """Edwardson et al. (1962) fit to the INFINITE radial WD, 0.01 <= tD <= 200.

    Used only by the self-tests, as an independent check on the Stehfest
    inversion above.
    """
    td = np.asarray(td, dtype=float)
    r = np.sqrt(td)
    return ((1.12838 * r + 1.19328 * td + 0.269872 * td * r
             + 0.00855294 * td ** 2) / (1.0 + 0.616599 * r + 0.0413008 * td))


@dataclass
class _TankHistory:
    """A production history on a uniform time grid, with PVT tabulated."""
    t: np.ndarray            # (K+1,) days, t[0] = 0
    dt: float
    F: np.ndarray            # (K+1, G) withdrawal at each step, each pressure
    pg: np.ndarray           # (G,) pressure grid, ascending
    Eo: np.ndarray
    Eg: np.ndarray
    cefw: np.ndarray         # (G,) (cw Sw + cf)/(1 - Sw) * Boi * (pi - p)
    p_init: float
    np_end: float


def _tank_history(t_days: np.ndarray, np_stb: np.ndarray, gp_scf: np.ndarray,
                  wp_stb: np.ndarray, pvt: OilPVT, p_initial: float,
                  t_end: float, n_steps: int = AQ_MAX_STEPS,
                  n_grid: int = 500) -> _TankHistory:
    t_days = np.asarray(t_days, float)
    order = np.argsort(t_days)
    tt = t_days[order]
    cn = np.maximum.accumulate(np.nan_to_num(np.asarray(np_stb, float)[order]))
    cg = np.maximum.accumulate(np.nan_to_num(np.asarray(gp_scf, float)[order]))
    cw = (np.maximum.accumulate(np.nan_to_num(np.asarray(wp_stb, float)[order]))
          if wp_stb is not None else np.zeros_like(tt))
    # Cumulatives are zero at t = 0 (first production), which the record may
    # not contain as a row.
    tt = np.concatenate([[min(0.0, tt[0])], tt])
    cn, cg, cw = (np.concatenate([[0.0], cn]), np.concatenate([[0.0], cg]),
                  np.concatenate([[0.0], cw]))
    K = int(max(20, min(n_steps, math.ceil(t_end / 30.4375))))
    dt = float(t_end) / K
    grid = dt * np.arange(K + 1)
    npk = np.interp(grid, tt, cn)
    gpk = np.interp(grid, tt, cg)
    wpk = np.interp(grid, tt, cw)

    p_floor = max(50.0, 0.03 * p_initial)
    pg = np.linspace(p_floor, p_initial, int(n_grid))
    bo, rs, bg, bt = pvt.bo(pg), pvt.rs(pg), pvt.bg(pg), pvt.bt(pg)
    # The PVT is anchored on pvt.p_init; the balance on the p_initial the user
    # gave. They are normally the same number.
    boi = float(pvt.bo(np.array([p_initial]))[0])
    bgi = float(pvt.bg(np.array([p_initial]))[0])
    bti = float(pvt.bt(np.array([p_initial]))[0])
    swc = pvt.sw_initial
    with np.errstate(divide="ignore", invalid="ignore"):
        rp = np.where(npk > 0, gpk / np.maximum(npk, 1e-12), 0.0)
    F = (npk[:, None] * (bo[None, :] + (rp[:, None] - rs[None, :]) * bg[None, :])
         + wpk[:, None] * pvt.bw)
    Eo = bt - bti
    Eg = boi * (bg / bgi - 1.0)
    cefw = (boi * ((pvt.cw_per_psi * swc + pvt.cf_per_psi)
                   / max(1.0 - swc, 1e-9)) * (p_initial - pg))
    return _TankHistory(t=grid, dt=dt, F=F, pg=pg, Eo=Eo, Eg=Eg, cefw=cefw,
                        p_init=float(p_initial), np_end=float(npk[-1]))


def _simulate_tank(h: _TankHistory, n_stb: float, m: float, kind: str,
                   a1: float = 0.0, a2: float = 1.0, a3: float = 2.0
                   ) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Pressure and cumulative influx at every step of the history.

    kind 'fetkovich': a1 = Wei (rb), a2 = tau = Wei / (J pi) (days)
    kind 'radial'   : a1 = U (rb/psi), a2 = t_c (days), a3 = reD
    kind 'none'     : a closed tank

    Each step solves F(p) = N Et(p) + We(p) for the new pressure. We is linear
    in the new pressure for both aquifers (Fetkovich through the interval's
    mean pressure, Van Everdingen-Hurst through the last superposition
    term), so the residual is formed on the whole pressure grid at once and
    the root read off the sign change nearest the previous step's pressure.
    Returns (p, We, hit_floor).
    """
    K = len(h.t) - 1
    pi = h.p_init
    E = n_stb * (h.Eo + m * h.Eg + (1.0 + m) * h.cefw)
    p = np.full(K + 1, pi)
    we = np.zeros(K + 1)
    floor_hit = False
    pg = h.pg
    if kind == "fetkovich":
        wei, tau = float(a1), float(a2)
        f = 1.0 - math.exp(-h.dt / max(tau, 1e-9))
    elif kind == "radial":
        U, tc, red = float(a1), float(a2), float(a3)
        wd = veh_wd(h.dt * np.arange(1, K + 1) / max(tc, 1e-12), red)
        dps = np.zeros(K + 1)          # superposition pressure steps, known
    for k in range(1, K + 1):
        if kind == "fetkovich":
            pa = pi * (1.0 - we[k - 1] / wei)
            A = we[k - 1] + (wei / pi) * f * (pa - 0.5 * p[k - 1])
            B = -(wei / pi) * f * 0.5
        elif kind == "radial":
            # We_k = U [ sum_{j<=k-2} dp_j WD(k-j) + dp_{k-1}(p_k) WD(1) ]
            known = float(dps[:k - 1] @ wd[k - 1:0:-1]) if k >= 2 else 0.0
            ref = p[k - 2] if k >= 2 else pi
            A = U * (known + 0.5 * ref * wd[0])
            B = -U * 0.5 * wd[0]
        else:
            A, B = 0.0, 0.0
        R = h.F[k] - E - (A + B * pg)
        sg = np.signbit(R)
        cross = np.flatnonzero(sg[1:] != sg[:-1])
        if cross.size:
            r0, r1 = R[cross], R[cross + 1]
            w = np.where(r1 == r0, 0.0, r0 / (r0 - r1))
            roots = pg[cross] + w * (pg[cross + 1] - pg[cross])
            pk = float(roots[np.argmin(np.abs(roots - p[k - 1]))])
        elif R[0] > 0:
            pk, floor_hit = float(pg[0]), True
        else:
            pk = float(pg[-1])
        p[k] = pk
        we[k] = max(A + B * pk, 0.0) if kind != "none" else 0.0
        if kind == "radial":
            # dp_{k-1} is now fully known.
            dps[k - 1] = 0.5 * ((p[k - 2] if k >= 2 else pi) - pk)
    return p, we, floor_hit


@dataclass
class AquiferMatch:
    """A tank-plus-aquifer history match on the survey pressures."""
    kind: str = "none"
    ran: bool = False
    refused_reason: str = ""
    fit_m: bool = False
    param_names: Tuple[str, ...] = ()
    params: Dict[str, float] = field(default_factory=dict)
    # One-sigma MULTIPLICATIVE factors for the log-parametrised quantities
    # (x/÷), additive for m.
    param_sigma: Dict[str, float] = field(default_factory=dict)
    corr: Optional[pd.DataFrame] = None
    n_obs: int = 0
    n_par: int = 0
    dof: int = 0
    rms_psi: float = float("nan")
    max_resid_psi: float = float("nan")
    max_resid_t: float = float("nan")
    n_stb: float = float("nan")
    m: float = 0.0
    n_range_stb: Tuple[float, float] = (float("nan"), float("nan"))
    n_range_open: Tuple[bool, bool] = (False, False)
    n_range_at_np: bool = False
    p_initial: float = float("nan")
    p_initial_source: str = ""
    profile: Optional[pd.DataFrame] = None
    we_to_date_rb: float = float("nan")
    withdrawal_to_date_rb: float = float("nan")
    n_ceiling_stb: float = float("nan")
    floor_hit: bool = False
    t_sim: Optional[np.ndarray] = None
    p_sim: Optional[np.ndarray] = None
    we_sim: Optional[np.ndarray] = None
    p_closed: Optional[np.ndarray] = None
    t_obs: Optional[np.ndarray] = None
    p_obs: Optional[np.ndarray] = None
    rock_note: str = ""
    warnings_text: List[str] = field(default_factory=list)
    n_ho_stb: float = float("nan")

    @property
    def accepted(self) -> bool:
        return bool(self.ran and np.isfinite(self.n_stb) and self.n_stb > 0)

    @property
    def label(self) -> str:
        return {"fetkovich": "FETKOVICH", "radial":
                "RADIAL VAN EVERDINGEN-HURST", "none": "CLOSED TANK"}.get(
                    self.kind, self.kind.upper())

    def summary(self) -> str:
        ind = "\n" + " " * 22
        lines = [f"AQUIFER HISTORY MATCH ({self.label})"]
        if not self.ran:
            lines.append("  NOT RUN           : "
                         + textwrap.fill(self.refused_reason, width=78)
                         .replace("\n", ind))
            return "\n".join(lines)
        held = ("" if self.fit_m else f"   (m held at {self.m:.3f})")
        lines += [
            f"  regressed on      : {', '.join(self.param_names)}{held}",
            f"  p_initial         : {self.p_initial:,.0f} psia "
            f"({self.p_initial_source})",
            f"  surveys matched   : {self.n_obs}, for {self.n_par} "
            f"parameters - {self.dof} degrees of freedom left",
            f"  pressure mismatch : RMS {self.rms_psi:,.1f} psi; largest "
            f"{self.max_resid_psi:+,.1f} psi at day {self.max_resid_t:,.0f}"]
        lo, hi = self.n_range_stb
        rng = (("the oil already produced" if self.n_range_at_np
                else "below the scan" if self.n_range_open[0]
                else f"{lo / 1e6:,.1f}")
               + " to "
               + ("above the scan" if self.n_range_open[1]
                  else f"{hi / 1e6:,.1f}"))
        lines.append(
            f"  OOIP (N)          : {self.n_stb / 1e6:,.2f} MMstb; 95 % range "
            f"{rng} MMstb")
        if (np.isfinite(lo) and lo > 0 and np.isfinite(hi)
                and hi / lo > AQ_MAX_RANGE_RATIO):
            lines.append(
                f"  N NOT DETERMINED  : the 95 % range spans a factor of "
                f"{hi / lo:,.1f}. Every N in it matches the surveys\n"
                "                      as well as the point value does - "
                "the aquifer takes up whatever N leaves.\n"
                "                      Quote the range, not the number.")
        if self.n_range_at_np:
            lines.append(
                "                      the lower end is the oil already "
                "produced: the surveys cannot rule out\n"
                "                      that the aquifer did nearly all the "
                "work and N is barely more than Np.")
        if (self.n_range_open[0] and not self.n_range_at_np) \
                or self.n_range_open[1]:
            lines.append(
                "                      the range runs off the scan: these "
                "surveys do not bound N on that side,\n"
                "                      however good the match looks.")
        if np.isfinite(self.n_ho_stb):
            lines.append(
                f"                      (Havlena-Odeh without an aquifer: "
                f"{self.n_ho_stb / 1e6:,.2f} MMstb)")
        units = {"Wei": ("MMrb", 1e6), "tau": ("days", 1.0),
                 "U": ("rb/psi", 1.0), "t_c": ("days", 1.0),
                 "reD": ("", 1.0), "m": ("", 1.0)}
        for k in self.param_names:
            if k == "N":
                continue
            v = self.params[k]
            u, sc = units.get(k, ("", 1.0))
            sig = self.param_sigma.get(k, float("nan"))
            if k == "m":
                bar = f" +/- {sig:.3f}" if np.isfinite(sig) else ""
            elif np.isfinite(sig) and sig > 100.0:
                bar = "  NOT DETERMINED (1-sigma factor above 100)"
            else:
                bar = (f"  x/÷ {sig:.2f} (1 sigma)" if np.isfinite(sig)
                       else "")
            lines.append(f"  {k:<18}: {v / sc:,.4g} {u}{bar}")
        if self.corr is not None and len(self.corr) > 1:
            pairs = []
            names = list(self.corr.columns)
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    pairs.append((names[i], names[j],
                                  float(self.corr.iloc[i, j])))
            pairs.sort(key=lambda x: -abs(x[2]))
            txt = ", ".join(f"{a}-{b} {c:+.2f}" for a, b, c in pairs[:4])
            lines.append(f"  correlations      : {txt}")
            if any(abs(c) > 0.95 for _, _, c in pairs):
                lines.append(
                    "                      a correlation beyond +/-0.95 means "
                    "those two trade off almost freely:\n"
                    "                      the surveys fix a combination of "
                    "them, not each one.")
        if np.isfinite(self.we_to_date_rb):
            share = (100.0 * self.we_to_date_rb / self.withdrawal_to_date_rb
                     if self.withdrawal_to_date_rb > 0 else float("nan"))
            lines.append(
                f"  We to date        : {self.we_to_date_rb / 1e6:,.2f} MMrb"
                + (f" ({share:.0f} % of the reservoir withdrawal)"
                   if np.isfinite(share) else ""))
        if self.rock_note:
            lines.append("  in rock terms     : " + self.rock_note)
        if (np.isfinite(self.n_ceiling_stb)
                and self.n_stb > 1.02 * self.n_ceiling_stb):
            lines.append(
                f"  WARNING           : the matched N is "
                f"{self.n_stb / self.n_ceiling_stb:.2f}x the We >= 0 ceiling "
                "min(F/Et) at this m. No aquifer\n                      can "
                "make that true - influx only ADDS to the withdrawal - so the "
                "match is\n                      fitting survey scatter. Do "
                "not use this N.")
        if self.dof < self.n_par:
            lines.append(
                f"  THIN              : {self.dof} degrees of freedom for "
                f"{self.n_par} parameters. The match can bend to\n"
                "                      the scatter; read the 95 % range, not "
                "the point value.")
        if self.floor_hit:
            lines.append(
                "  NOTE              : at some trial values the simulated "
                "pressure hit the floor of the grid.")
        for w in self.warnings_text:
            lines.append("  NOTE              : " + w)
        return "\n".join(lines)


def aquifer_history_match(t_days: np.ndarray, np_stb: np.ndarray,
                          gp_scf: np.ndarray, wp_stb: Optional[np.ndarray],
                          t_survey: np.ndarray, p_survey: np.ndarray,
                          pvt: OilPVT, kind: str = "fetkovich",
                          m_gas_cap: float = 0.0, fit_m: bool = False,
                          p_initial: Optional[float] = None,
                          n_guess_stb: Optional[float] = None,
                          aq_porosity: Optional[float] = None,
                          aq_ro_ft: Optional[float] = None,
                          aq_mu_w_cp: float = 0.5,
                          min_dof: int = AQ_MIN_DOF,
                          profile_points: int = 15) -> AquiferMatch:
    """Regress N (and m) and an aquifer on the survey pressures.

    Parameters are fitted in log space (m linearly), by least squares on the
    pressure mismatch at the surveys, from several starting points. The
    uncertainty is reported two ways: the linearised covariance, as one-sigma
    factors and a correlation matrix, and a PROFILE of the mismatch against N
    - N held at each value in turn, everything else refitted - read at the
    95 % F-test level. On problems this badly conditioned the profile is the
    one to believe; the covariance says which parameters are trading off.
    """
    kind = str(kind).lower()
    if kind not in AQUIFER_KINDS:
        raise ValueError(f"aquifer kind must be one of {AQUIFER_KINDS}")
    pi = float(p_initial) if p_initial is not None else float(pvt.p_init)
    out = AquiferMatch(kind=kind, fit_m=bool(fit_m), m=float(m_gas_cap or 0.0),
                       p_initial=pi,
                       p_initial_source=("as entered" if p_initial is not None
                                         else "the PVT initial pressure"))
    ts = np.asarray(t_survey, float)
    ps = np.asarray(p_survey, float)
    npv = np.asarray(np_stb, float)
    tv = np.asarray(t_days, float)
    ok_s = np.isfinite(ts) & np.isfinite(ps) & (ps > 0)
    np_at_s = np.interp(ts, np.sort(tv),
                        np.maximum.accumulate(np.nan_to_num(npv[np.argsort(tv)])))
    obs = ok_s & (ts > 0) & (np_at_s > 0)
    names = ["N"] + (["m"] if fit_m else []) + (
        ["Wei", "tau"] if kind == "fetkovich" else
        ["U", "t_c", "reD"] if kind == "radial" else [])
    out.param_names = tuple(names)
    out.n_obs, out.n_par = int(obs.sum()), len(names)
    out.dof = out.n_obs - out.n_par
    if out.dof < min_dof:
        need = out.n_par + min_dof
        alt = []
        if fit_m:
            alt.append("hold m from structure and logs")
        if kind == "radial":
            alt.append("use Fetkovich (two aquifer parameters, not three)")
        alt.append("add surveys")
        out.refused_reason = (
            f"{out.n_par} parameters ({', '.join(names)}) need at least "
            f"{need} surveys after first production - that many parameters "
            f"plus {min_dof} to spare - and this record has {out.n_obs}. With "
            "fewer, the match can pass through every survey whatever N is. "
            "Options: " + "; ".join(alt) + ".")
        return out
    dp_max = float(np.max(pi - ps[obs]))
    if dp_max < 50.0:
        out.refused_reason = (
            f"the deepest survey is only {dp_max:,.0f} psi below p_i; "
            "there is no pressure history to match.")
        return out

    t_end = float(max(np.max(ts[obs]), 1.0))
    h = _tank_history(tv, npv, gp_scf, wp_stb, pvt, pi, t_end)
    t_o, p_o = ts[obs], ps[obs]
    np_last = h.np_end

    # -- parametrisation ------------------------------------------------
    i_m = 1 if fit_m else None
    j0 = 2 if fit_m else 1

    def unpack(x):
        n = math.exp(x[0])
        m = float(x[i_m]) if fit_m else out.m
        a = [math.exp(v) for v in x[j0:]]
        if kind == "radial":
            a[2] = 1.0 + a[2]          # reD - 1 is the log-fitted quantity
        return n, m, a

    def sim(x):
        n, m, a = unpack(x)
        if kind == "fetkovich":
            return _simulate_tank(h, n, m, kind, a[0], a[1])
        if kind == "radial":
            return _simulate_tank(h, n, m, kind, a[0], a[1], a[2])
        return _simulate_tank(h, n, m, "none")

    floor_any = [False]

    def resid(x):
        p, _, fl = sim(x)
        floor_any[0] |= fl
        return np.interp(t_o, h.t, p) - p_o

    n0 = float(n_guess_stb) if (n_guess_stb and np.isfinite(n_guess_stb)
                                and n_guess_stb > np_last) else 20.0 * np_last
    n_lo = max(np_last * 1.02, 1.0)
    lb = [math.log(n_lo)]
    ub = [math.log(max(n0, n_lo) * 1.0e3)]
    if fit_m:
        lb.append(0.0); ub.append(5.0)
    boi = float(pvt.bo(np.array([pi]))[0])
    T = t_end
    starts_aq: List[List[float]] = [[]]
    if kind == "fetkovich":
        lb += [math.log(1.0e3), math.log(1.0)]
        ub += [math.log(1.0e13), math.log(1.0e6)]
        starts_aq = [[math.log(n0 * boi * c), math.log(T * r)]
                     for c in (0.02, 0.2, 2.0) for r in (0.1, 1.0, 10.0)]
    elif kind == "radial":
        lb += [math.log(1.0e-3), math.log(1.0e-3), math.log(0.05)]
        ub += [math.log(1.0e9), math.log(1.0e6), math.log(300.0)]
        starts_aq = [[math.log(n0 * boi * c / pi), math.log(T * r),
                      math.log(4.0)]
                     for c in (1e-3, 1e-2, 1e-1) for r in (0.01, 0.1, 1.0)]
    lb_a, ub_a = np.array(lb), np.array(ub)

    # Starting points: every aquifer start is SCORED at three values of N,
    # and only the best few are optimised. Optimising all of them cost
    # 20-50 s a match on a radial aquifer, almost all of it spent polishing
    # starts that were never going to win.
    def x_start(nm, sa):
        x0 = [math.log(min(max(n0 * nm, n_lo * 1.01), math.exp(ub[0]) / 2))]
        if fit_m:
            x0.append(min(max(out.m, 0.05), 4.9))
        return np.clip(np.array(x0 + sa, float), lb_a + 1e-9, ub_a - 1e-9)

    cands = []
    for sa in starts_aq:
        for nm in (1.0, 0.3, 3.0):
            x0 = x_start(nm, sa)
            rv = resid(x0)
            cands.append((float(rv @ rv), x0))
    cands.sort(key=lambda c: c[0])
    best = None
    for _, x0 in cands[:4]:
        try:
            r = optimize.least_squares(resid, x0, bounds=(lb_a, ub_a),
                                       x_scale=1.0, diff_step=1e-3,
                                       max_nfev=250)
        except (ValueError, FloatingPointError):
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        out.refused_reason = "the regression failed from every start."
        return out

    x = best.x
    n_hat, m_hat, a_hat = unpack(x)
    res = best.fun
    ssr = float(res @ res)
    s2 = ssr / max(out.dof, 1)
    out.ran = True
    out.n_stb, out.m = n_hat, m_hat
    out.rms_psi = float(math.sqrt(ssr / len(res)))
    k_mx = int(np.argmax(np.abs(res)))
    out.max_resid_psi = float(-res[k_mx])      # observed minus simulated
    out.max_resid_t = float(t_o[k_mx])
    pnames = list(names)
    vals = [n_hat] + ([m_hat] if fit_m else []) + list(a_hat)
    out.params = dict(zip(pnames, vals))

    # Linearised covariance, in the fitted (log) coordinates.
    J = best.jac
    try:
        cov = s2 * np.linalg.pinv(J.T @ J)
        sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        with np.errstate(divide="ignore", invalid="ignore"):
            cr = cov / np.outer(sd, sd)
        out.corr = pd.DataFrame(np.clip(cr, -1, 1), index=pnames,
                                columns=pnames)
        for i, nm in enumerate(pnames):
            if nm == "m":
                out.param_sigma[nm] = float(sd[i])
            else:
                out.param_sigma[nm] = float(math.exp(min(sd[i], 50.0)))
    except np.linalg.LinAlgError:
        pass

    # Profile on N: hold N, refit everything else, read the 95 % F level.
    #
    # A coarse scan first, then each edge of the 95 % region refined by
    # bisection. Reading the edge off the coarse grid alone put it at the
    # nearest grid point; on a synthetic tank with N = 50 MMstb that made the
    # 95 % range 46.9 - 49.4 MMstb, which excludes the truth, because the
    # grid step there was 23 %.
    thresh = ssr * (1.0 + stats.f.ppf(0.95, 1, max(out.dof, 1))
                    / max(out.dof, 1))
    lo_b, hi_b = lb_a[0], ub_a[0]

    def prof_at(ln_n: float, start: Optional[np.ndarray]):
        if len(x) == 1:
            rv = resid(np.array([ln_n]))
            return float(rv @ rv), None

        def rr(y):
            return resid(np.concatenate([[ln_n], y]))
        y0 = np.clip(start, lb_a[1:] + 1e-9, ub_a[1:] - 1e-9)
        try:
            r = optimize.least_squares(rr, y0, bounds=(lb_a[1:], ub_a[1:]),
                                       diff_step=1e-3, max_nfev=100)
        except (ValueError, FloatingPointError):
            return float("nan"), start
        return float(2.0 * r.cost), r.x

    span = math.log(8.0)
    grid = np.linspace(max(x[0] - span, lo_b), min(x[0] + span, hi_b),
                       int(profile_points))
    i0 = int(np.argmin(np.abs(grid - x[0])))
    grid[i0] = x[0]
    pts: Dict[float, Tuple[float, Optional[np.ndarray]]] = {
        float(x[0]): (ssr, x[1:].copy() if len(x) > 1 else None)}
    for rng_idx in (range(i0 + 1, len(grid)), range(i0 - 1, -1, -1)):
        prev = x[1:].copy() if len(x) > 1 else None
        for idx in rng_idx:
            v, yx = prof_at(float(grid[idx]), prev)
            if np.isfinite(v):
                pts[float(grid[idx])] = (v, yx)
                prev = yx if yx is not None else prev

    # Extend the scan outwards while the 95 % region is still open on a side
    # and the bound has not been reached. With a fixed +/- 8x scan, a strong
    # aquifer whose best match sat at N = 1,340 MMstb reported "below the
    # scan" at 168 MMstb when the truth, 50 MMstb, matched inside the 95 %
    # level - the region did not end there, the scan did.
    for _ in range(3):
        keys_now = sorted(pts)
        v_now = [pts[k][0] for k in keys_now]
        grew = False
        if v_now[0] <= thresh and keys_now[0] > lo_b + 1e-9:
            prev = pts[keys_now[0]][1]
            for ln_n in np.unique(np.clip(np.linspace(
                    keys_now[0] - span / 3.0, keys_now[0] - span, 3),
                    lo_b, None))[::-1]:
                v, yx = prof_at(float(ln_n), prev)
                if np.isfinite(v):
                    pts[float(ln_n)] = (v, yx)
                    prev = yx if yx is not None else prev
                    if v > thresh:
                        break
            grew = True
        if v_now[-1] <= thresh and keys_now[-1] < hi_b - 1e-9:
            prev = pts[keys_now[-1]][1]
            for ln_n in np.unique(np.clip(np.linspace(
                    keys_now[-1] + span / 3.0, keys_now[-1] + span, 3),
                    None, hi_b)):
                v, yx = prof_at(float(ln_n), prev)
                if np.isfinite(v):
                    pts[float(ln_n)] = (v, yx)
                    prev = yx if yx is not None else prev
                    if v > thresh:
                        break
            grew = True
        if not grew:
            break

    refined: Dict[float, float] = {}

    def refine(inner: float, outer: float) -> float:
        yi = pts[inner][1]
        a, b = inner, outer
        for _ in range(7):
            mid = 0.5 * (a + b)
            v, yx = prof_at(mid, yi)
            if np.isfinite(v):
                refined[float(mid)] = v       # kept for the profile plot
            if np.isfinite(v) and v <= thresh:
                a, yi = mid, (yx if yx is not None else yi)
            else:
                b = mid
        return 0.5 * (a + b)

    keys = np.array(sorted(pts))
    vals_p = np.array([pts[k][0] for k in keys])
    inside = vals_p <= thresh
    j = int(np.argmin(np.abs(keys - x[0])))
    jl = j
    while jl - 1 >= 0 and inside[jl - 1]:
        jl -= 1
    jh = j
    while jh + 1 < len(keys) and inside[jh + 1]:
        jh += 1
    open_lo = jl == 0
    open_hi = jh == len(keys) - 1
    ln_lo = keys[jl] if open_lo else refine(float(keys[jl]),
                                            float(keys[jl - 1]))
    ln_hi = keys[jh] if open_hi else refine(float(keys[jh]),
                                            float(keys[jh + 1]))
    out.n_range_stb = (math.exp(ln_lo), math.exp(ln_hi))
    out.n_range_open = (bool(open_lo), bool(open_hi))
    out.n_range_at_np = bool(open_lo and keys[jl] <= lb_a[0] + 1e-6)
    allk = np.array(sorted(set(keys.tolist()) | set(refined)))
    allv = np.array([pts[k][0] if k in pts else refined[k] for k in allk])
    out.profile = pd.DataFrame({"N_stb": np.exp(allk), "ssr_psi2": allv,
                                "threshold": thresh})

    # The simulated history at the answer, and the closed tank for contrast.
    p_sim, we_sim, fl = sim(x)
    out.floor_hit = bool(fl)
    out.t_sim, out.p_sim, out.we_sim = h.t, p_sim, we_sim
    out.t_obs, out.p_obs = t_o, p_o
    out.p_closed = _simulate_tank(h, n_hat, m_hat, "none")[0]
    out.we_to_date_rb = float(we_sim[-1])
    k_last = len(h.t) - 1
    g_last = int(np.argmin(np.abs(h.pg - p_sim[-1])))
    out.withdrawal_to_date_rb = float(h.F[k_last, g_last])

    # The We >= 0 ceiling at the matched m, from the surveys themselves.
    try:
        gp_s = np.interp(t_o, np.sort(tv), np.maximum.accumulate(
            np.nan_to_num(np.asarray(gp_scf, float)[np.argsort(tv)])))
        wp_s = (np.interp(t_o, np.sort(tv), np.maximum.accumulate(
            np.nan_to_num(np.asarray(wp_stb, float)[np.argsort(tv)])))
            if wp_stb is not None else np.zeros_like(t_o))
        tab = havlena_odeh_oil(p_o, np.interp(t_o, np.sort(tv),
                               np.maximum.accumulate(np.nan_to_num(
                                   npv[np.argsort(tv)]))),
                               gp_s, wp_s, pvt, m_gas_cap=m_hat,
                               p_initial=pi)
        u = (tab["Et"].to_numpy(float) > 0) & (
            tab["dp"].to_numpy(float) >= APPARENT_N_MIN_DP)
        a = tab["apparent_N_stb"].to_numpy(float)[u]
        a = a[np.isfinite(a) & (a > 0)]
        if a.size:
            out.n_ceiling_stb = float(np.min(a))
    except (ValueError, KeyError):
        pass

    # Rock properties, translated - products only.
    ct_aq = pvt.cw_per_psi + pvt.cf_per_psi
    notes = []
    if kind == "fetkovich":
        wi = a_hat[0] / (ct_aq * pi)
        notes.append(f"aquifer water volume Wi = Wei/(ct pi) = "
                     f"{wi / 1e6:,.0f} MMrb at ct {ct_aq:.1e} 1/psi")
    elif kind == "radial" and aq_porosity and aq_ro_ft:
        phi, ro = float(aq_porosity), float(aq_ro_ft)
        h_theta = a_hat[0] / (VEH_U_CONST * phi * ct_aq * ro ** 2)
        k_md = phi * aq_mu_w_cp * ct_aq * ro ** 2 / (VEH_TD_CONST * a_hat[1])
        notes.append(
            f"h x (theta/360) = {h_theta:,.1f} ft (e.g. {h_theta:,.0f} ft at "
            f"360 deg or {4 * h_theta:,.0f} ft at 90 deg - only the product "
            f"is determined);{_IND}k = {k_md:,.0f} md; aquifer radius "
            f"{a_hat[2] * ro:,.0f} ft; at phi {phi:.2f}, ro {ro:,.0f} ft, "
            f"mu_w {aq_mu_w_cp:.2f} cp, ct {ct_aq:.1e} 1/psi")
    elif kind == "radial":
        notes.append("give aquifer porosity and reservoir radius to "
                     "translate U and t_c into h x theta and k")
    out.rock_note = "; ".join(notes)
    return out


def _mh_dmin_inactive(fit) -> bool:
    """A modified hyperbolic whose Dmin was fitted at or above Di."""
    p = getattr(fit, "params", {}) or {}
    return bool("Dmin" in p and "Di" in p
                and np.isfinite(p["Dmin"]) and np.isfinite(p["Di"])
                and p["Dmin"] >= p["Di"])


def _fit_summary_text(fit) -> str:
    """The shared fit summary, corrected where it misdescribes the curve.

    The modified hyperbolic clamps Dmin to just below Di when the optimiser
    returns it above, which switches the curve to exponential at t0 and makes
    b and Dmin inert. The shared summary printed the optimiser's Dmin anyway:
    a field report read "Di (eff) 0.5 %/yr, Dmin(eff) 8.0 %/yr" on a curve
    that declines at 0.47 %/yr for its whole length, with b and Dmin carrying
    error bars of 1.9e-09 and 0 - which is what a parameter with no effect on
    the fit looks like, not a precise estimate.
    """
    txt = fit.summary()
    if not _mh_dmin_inactive(fit):
        return txt
    di = float(fit.params["Di"])
    d_eff = 100.0 * (1.0 - math.exp(-di * DAYS_PER_YEAR))
    txt = re.sub(r"Dmin\(eff\) : [^\n]*",
                 "Dmin(eff) : as fitted, above Di - NOT USED by the curve",
                 txt)
    txt += (f"\n  NOTE      : Dmin was fitted above Di, so the curve switches "
            f"to exponential at t0 and\n              declines at "
            f"{d_eff:.2f} %/yr for its whole length. b and Dmin have no "
            "effect on the\n              rate, and their tiny error bars "
            "say only that. Read this fit as an\n              exponential "
            "at Di.")
    return txt


# ==============================================================================
# SECTION 4 -- DIAGNOSTICS
# ==============================================================================

@dataclass
class GORDiagnostic:
    """Producing GOR against cumulative oil, and what it says about pressure.

    Above the bubble point the produced gas is only what was dissolved, so the
    GOR sits flat at Rsi. Below it, gas comes out of solution in the reservoir,
    becomes mobile once it exceeds its critical saturation, and the producing
    GOR climbs - often to many times Rsi before falling away as the reservoir
    is depleted of gas. That break is the bubble point, seen from surface data
    alone, and it is a pressure measurement in a well that has no gauge.

    A GOR BELOW Rsi is the diagnostic that matters the other way: gas cannot
    be un-produced, so either the gas is being under-measured, some is going
    to fuel or flare unrecorded, or Rsi is wrong for this fluid.
    """
    ok: bool
    gor_initial: float = float("nan")       # median of the early plateau
    rsi: float = float("nan")
    gor_max: float = float("nan")
    gor_recent: float = float("nan")
    ratio_recent: float = float("nan")      # recent GOR / Rsi
    break_np_stb: Optional[float] = None    # cumulative at the GOR break
    break_frac_of_record: float = float("nan")
    gor_max_single: float = float("nan")    # highest single month
    peaked: bool = False                    # GOR has turned over
    peak_np_stb: Optional[float] = None
    scatter_frac: float = float("nan")      # robust noise / plateau level
    n_points: int = 0
    below_rsi_fraction: float = float("nan")
    reason: str = ""
    # Saturated at discovery: the reservoir starts AT its bubble point, so a
    # later GOR break cannot be the bubble-point crossing. It is the free gas
    # reaching critical saturation and starting to flow.
    saturated: bool = False
    # The sustained peak sits in the first few periods of the record: the
    # GOR falls from first production rather than rising then falling.
    peak_at_start: bool = False

    @property
    def implausible_for_black_oil(self) -> bool:
        # A sustained GOR of 73,445 scf/STB on a fluid with Rsi 391 was
        # analysed as a black oil without comment. Solution-gas drive takes
        # the producing GOR to a few times Rsi, sometimes ten; twenty times
        # Rsi and over 10,000 scf/STB is gas-condensate or wet-gas territory,
        # or gas entered in the wrong units.
        return bool(self.ok and np.isfinite(self.gor_max) and self.rsi > 0
                    and self.gor_max > 20.0 * self.rsi
                    and self.gor_max > 10000.0)

    @property
    def broke(self) -> bool:
        return self.ok and self.break_np_stb is not None

    def summary(self) -> str:
        if not self.ok:
            return f"  GOR diagnostic    : not run ({self.reason})"
        lines = [
            f"  GOR diagnostic    : early {self.gor_initial:,.0f}, recent "
            f"{self.gor_recent:,.0f}, peak {self.gor_max:,.0f} scf/STB "
            f"(Rsi {self.rsi:,.0f})"]
        # The peak quoted above is the SUSTAINED one. A single month far above
        # it is a metering spike or a test, and is named as such rather than
        # being quoted as the peak - 3,636 on a series running 350 to 500 was.
        if (np.isfinite(self.gor_max_single) and np.isfinite(self.gor_max)
                and self.gor_max_single > 1.5 * self.gor_max):
            lines.append(
                f"                      highest single month "
                f"{self.gor_max_single:,.0f} scf/STB is "
                f"{self.gor_max_single / self.gor_max:.1f}x the sustained "
                "peak - a spike, not\n                      reservoir "
                "behaviour, and not used as the GOR peak.")
        if self.broke and self.saturated:
            # The field report said "that is the reservoir crossing its
            # bubble point" twenty lines below "SATURATED at discovery".
            lines.append(
                f"                      GOR breaks upward at Np = "
                f"{self.break_np_stb / STB_PER_MSTB:,.0f} Mstb "
                f"({100 * self.break_frac_of_record:.0f} % of the record). "
                "The reservoir was\n                      saturated at "
                "discovery, so this is not the bubble point: it is the free "
                "gas\n                      reaching critical saturation "
                "and starting to flow.")
        elif self.broke:
            lines.append(
                f"                      GOR breaks upward at Np = "
                f"{self.break_np_stb / STB_PER_MSTB:,.0f} Mstb "
                f"({100 * self.break_frac_of_record:.0f} % of the record): "
                "that is the\n                      reservoir crossing its "
                "bubble point, read from surface data alone.")
        elif self.peak_at_start:
            pass                    # said below, with the peak
        elif self.peaked and self.saturated:
            lines.append(
                "                      the GOR has already peaked and is "
                "falling; the rising limb is not\n                      "
                "clean enough to date when free gas began to flow.")
        elif self.saturated:
            lines.append(
                "                      no upward break: the reservoir was "
                "saturated at discovery, but free\n                      "
                "gas has not yet reached critical saturation over this "
                "record, or the gas\n                      measurement is "
                "too noisy to show it.")
        elif self.peaked:
            lines.append(
                "                      the GOR has already peaked and is "
                "falling, so the bubble point is\n                      "
                "behind this record - but the rising limb is not clean "
                "enough to date it.")
        else:
            lines.append(
                "                      no upward break: the reservoir has not "
                "crossed its bubble point over\n                      this "
                "record, or the gas measurement is too noisy to show it.")
        # 1.0x Rsi is not evidence of anything. Gas and oil are metered
        # separately, so a producing GOR carries the noise of both, and a
        # ratio of 1.02 was being reported as "free gas is being produced"
        # on a well whose pressure never went near its bubble point. The
        # claim needs a margin wider than the measurement.
        if np.isfinite(self.ratio_recent) and self.ratio_recent > 1.15:
            lines.append(
                f"                      recent GOR is {self.ratio_recent:.1f}x "
                "Rsi - free gas is being produced, so the\n                    "
                "  reservoir is below the bubble point and losing its own "
                "drive energy.")
        if self.peak_at_start and self.peak_np_stb is not None:
            lines.append(
                f"                      the GOR is highest at the START of "
                f"the record (Np = {self.peak_np_stb / STB_PER_MSTB:,.0f} "
                "Mstb) and falls\n                      from there. That is "
                "not the rise and fall of solution-gas drive: any rising\n"
                "                      limb is before this history, and the "
                "first months of a well are often\n                      "
                "clean-up or test readings. No conclusion about drive energy "
                "is drawn from it.")
        elif self.peaked and self.peak_np_stb is not None:
            lines.append(
                f"                      GOR peaked at Np = "
                f"{self.peak_np_stb / STB_PER_MSTB:,.0f} Mstb and has since "
                f"fallen: the free gas is being\n                      "
                "depleted, so solution-gas drive is spent and the pressure "
                "support left\n                      is whatever the aquifer "
                "or gas cap provides.")
        # Past about a fifth of the plateau the month-to-month scatter is
        # comparable to the whole bubble-point step, and the break test fails
        # in BOTH directions: on synthetic records at 27 % gas scatter it
        # missed two wells that had crossed their bubble point and invented a
        # break on one that had not. The call is still made, because it is
        # the best available reading, but it is not worth acting on alone.
        if np.isfinite(self.scatter_frac) and self.scatter_frac > 0.20:
            lines.append(
                f"                      CAUTION: month-to-month GOR scatter "
                f"is {100 * self.scatter_frac:.0f} % of the plateau level. At "
                "that scatter the break\n                      and the "
                "peak are both unreliable, in both directions - treat every "
                "shape\n                      reading above as indicative "
                "and confirm it against a pressure survey.")
        if self.implausible_for_black_oil:
            lines.append(
                f"  WARNING           : the sustained GOR reaches "
                f"{self.gor_max:,.0f} scf/STB, {self.gor_max / self.rsi:,.0f}x "
                "Rsi. Solution-gas\n                      drive does not do "
                "that; above a few thousand scf/STB the fluid behaves as a\n"
                "                      gas condensate or wet gas. Check that "
                "q_gas is in Mscf/d (scf/d reads 1,000x\n                      "
                "high), and whether gas-cap or gas-zone gas is being produced. "
                "The black-oil\n                      PVT and material "
                "balance here assume neither.")
        if np.isfinite(self.below_rsi_fraction) and self.below_rsi_fraction > 0.5:
            lines.append(
                f"  WARNING           : {100 * self.below_rsi_fraction:.0f} % "
                "of periods report a GOR BELOW Rsi. Gas cannot be\n"
                "                      un-produced - either it is being "
                "under-measured, some is going to\n                      fuel "
                "or flare unrecorded, or Rsi is wrong for this fluid.")
        return "\n".join(lines)


def gor_diagnostic(gor: np.ndarray, np_stb: np.ndarray, rsi: float,
                   early_frac: float = 0.15,
                   recent_periods: int = 12,
                   saturated: bool = False) -> GORDiagnostic:
    """Find the bubble-point break in the producing GOR, if there is one."""
    g = np.asarray(gor, dtype=float)
    n = np.asarray(np_stb, dtype=float)
    ok = np.isfinite(g) & (g > 0) & np.isfinite(n)
    if int(ok.sum()) < 8:
        return GORDiagnostic(ok=False,
                             reason="fewer than eight periods with a GOR")
    g, n = g[ok], n[ok]
    n_early = max(3, int(early_frac * len(g)))
    g0 = float(np.median(g[:n_early]))
    recent = g[-min(recent_periods, len(g)):]
    g_recent = float(np.median(recent))
    below = float(np.mean(g < float(rsi) * 0.95)) if rsi > 0 else float("nan")

    # The scan runs over the RISING LIMB ONLY, which means finding the peak
    # first.
    #
    # The earlier version fitted flat-then-rising across the whole record and
    # required the right-hand segment to have a positive slope. That silently
    # fails on exactly the wells this diagnostic exists for. Solution gas
    # drive does not raise the GOR and leave it there: it rises after the
    # bubble point, peaks, and then falls away as the reservoir runs out of
    # gas to liberate. On a synthetic tank whose GOR went 600 -> 747 -> 569,
    # with no noise at all, no split point gave a rising right-hand segment
    # and the diagnostic reported "no upward break" on a well that had been
    # below its bubble point for two thirds of its life. The more mature the
    # well, the more certainly it was missed.
    sm = g.copy()
    if len(g) >= 5:
        sm[1:-1] = np.array([np.median(g[j - 1:j + 2])
                             for j in range(1, len(g) - 1)])

    # The PEAK is found on a much wider median than the break. A 3-point
    # median cannot remove a spike two months long, and one on a field
    # record - 3,636 scf/STB in a series running 350 to 500 - was read as
    # the physical GOR peak: the report declared solution-gas drive spent on
    # a GOR that was still rising, and the forecast was fitted to the 24
    # points after the spike, where its leverage made the slope anything at
    # all. A genuine solution-gas peak is a limb lasting years; a window of
    # about a twentieth of the record keeps it and removes the spike.
    w_pk = max(5, (len(g) // 20) | 1)
    h = w_pk // 2
    sm_pk = np.array([np.median(g[max(0, j - h):j + h + 1])
                      for j in range(len(g))])
    # The peak is placed at the END of the top plateau, not at its argmax.
    # A wide median flattens the broad top of a real solution-gas peak, and
    # argmax then picks the FIRST of several near-equal values: on one
    # synthetic well it returned index 19 where the GOR actually turned over
    # at 49, which cut the rising limb to twenty points and cost the break
    # test its power - two wells that had crossed their bubble point were
    # reported as not having done so. The end of the plateau is also where
    # the falling limb genuinely begins, which is what the GOR fit needs.
    top = float(np.max(sm_pk))
    on_top = np.flatnonzero(sm_pk >= 0.95 * top)
    pk = int(on_top[-1]) if on_top.size else int(np.argmax(sm_pk))
    # Sustained, not merely highest: the turn-over must be visible on the
    # wide median too, and there must be a falling limb long enough to fit.
    peaked = (pk < len(g) - max(8, w_pk)
              and top > float(np.median(g[-w_pk:])) * 1.10)

    # The limb the scan sees: up to the peak if the GOR has turned over,
    # otherwise the whole record.
    end = (pk + 1) if peaked else len(g)
    gl, nl = g[:end], n[:end]

    # The break is where the GOR leaves its early plateau - a departure test,
    # not a model-comparison one.
    #
    # Fitting flat-then-rising and asking whether it beats a single straight
    # line throws away the one thing that is actually known here: above the
    # bubble point the GOR is PINNED at Rsi, so the left segment has a level
    # and a scatter that can be measured, and the question is only when the
    # data leave it. Scored as a model comparison the same break came back at
    # a 4.7 % cost improvement - F = 0.94, p = 0.40, indistinguishable from
    # nothing - because a straight line through a flat-then-rising curve is
    # not a bad fit. Scored as a departure from a measured plateau it is
    # unambiguous, and that is the honest framing: the flat part is physics,
    # not a fitted parameter.
    n_plateau = max(4, len(gl) // 8)
    plateau = float(np.median(gl[:n_plateau]))
    # Above the bubble point the GOR is Rsi, not merely flat. Where the early
    # plateau agrees with Rsi, use Rsi: it is a measured fluid property with
    # no sampling error in it, and the median of four early points has plenty.
    if rsi > 0 and abs(plateau - rsi) < 0.20 * rsi:
        plateau = float(rsi)
    # The scatter is estimated from SUCCESSIVE DIFFERENCES over the whole
    # limb, not from the MAD of the handful of plateau points. Differencing
    # removes the trend and leaves the noise, so every point in the record
    # contributes to the estimate. Taking the MAD of four or five early
    # points instead gave a sigma that was itself mostly noise; a single high
    # early reading tripled the threshold and the break went undetected on 15
    # of 20 wells that had demonstrably crossed their bubble point.
    dif = np.abs(np.diff(g))
    sigma = float(np.median(dif)) * 1.4826 / math.sqrt(2.0) if len(dif) else 0.0
    sigma = max(sigma, 0.02 * plateau)

    # The test is on a WINDOW MEAN, against that mean's own standard error.
    #
    # Asking for individual readings to clear plateau + 3 sigma sets the bar
    # at three times the scatter of a single month, which on 7.5 % gas noise
    # is 150 scf/STB - most of the whole rise. The break then only registers
    # near the GOR peak, if at all, and at 5 % noise and above it never
    # registered. A run of w readings estimates the level w times better, so
    # the bar it has to clear is 3 sigma / sqrt(w), and the floor of 8 % of
    # the plateau keeps a low-noise record from breaking on nothing.
    run_needed = int(min(6, max(3, len(gl) // 6)))
    thresh = plateau + max(3.0 * sigma / math.sqrt(run_needed),
                           0.08 * plateau)

    best_i = None
    for i in range(n_plateau, len(gl) - run_needed + 1):
        if float(np.mean(gl[i:i + run_needed])) > thresh:
            best_i = i
            break

    # The confirmation is a difference of LEVELS, not a slope.
    #
    # Requiring the post-break limb to be significantly rising sounds right
    # and is wrong, because that is not the shape solution gas makes. The GOR
    # steps up to a new level once free gas becomes mobile and then sits
    # there, rolling over towards its peak - one well went 600 to 750 inside
    # four months and then held 750 for the next three years. Regressed over
    # the whole limb that is a flat line, p = 0.15, and the break was thrown
    # away on a record where it is visible by eye. What has to be shown is
    # that the GOR after the break sits above the plateau by more than the
    # scatter, which is a two-sample test, and the level test runs on the
    # MEAN rather than the maximum so that one high month cannot carry it.
    broke = False
    if best_i is not None and best_i >= 3:
        before, after = gl[:best_i], gl[best_i:]
        if len(after) >= 4 and float(np.mean(after)) > 1.15 * plateau:
            # The bar is the conventional 5 %, not a tighter one chosen to
            # make a particular well pass. At 1 % three wells whose level
            # shift was significant at 2-3 % were reported as having no
            # break; specificity was then checked the other way round, on 20
            # records from a tank that never crossed its bubble point, and
            # none of them broke at 5 % either.
            tt = stats.ttest_ind(after, before, equal_var=False)
            broke = bool(np.isfinite(tt.pvalue) and tt.pvalue < 0.05
                         and float(np.mean(after)) > float(np.mean(before)))

    return GORDiagnostic(
        ok=True, saturated=bool(saturated), gor_initial=g0, rsi=float(rsi),
        gor_max=float(np.max(sm_pk)), gor_max_single=float(np.max(g)),
        gor_recent=g_recent,
        ratio_recent=(g_recent / float(rsi)) if rsi > 0 else float("nan"),
        break_np_stb=(float(n[best_i]) if broke else None),
        break_frac_of_record=(float(best_i) / len(g) if broke else float("nan")),
        peaked=bool(peaked),
        peak_at_start=bool(peaked and pk < n_early),
        peak_np_stb=(float(n[pk]) if peaked else None),
        scatter_frac=(sigma / plateau if plateau > 0 else float("nan")),
        n_points=len(g), below_rsi_fraction=below)


@dataclass
class WaterCutDiagnostic:
    """Chan (1995) water-control diagnostic: WOR and its derivative on log-log.

    The shape of the water-oil ratio against time separates mechanisms that
    a water-cut plot alone cannot:

      normal displacement   WOR rises gently; WOR' small and flat
      channelling           WOR rises steeply and WOR' rises in parallel
      coning                WOR flattens after breakthrough; WOR' flat or
                            falling, because the cone stabilises

    The distinction is worth money: coning responds to rate reduction, and
    channelling does not.
    """
    ok: bool
    mechanism: str = "undetermined"
    wor_slope: float = float("nan")          # d log WOR / d log t, late
    wor_prime_slope: float = float("nan")
    wor_prime_departure: float = float("nan")   # s2 - (s1 - 1); see below
    curvature_z: float = float("nan")           # c2 / se(c2) on log-log
    falling_fraction: float = float("nan")
    breakthrough_t_days: Optional[float] = None
    wor_final: float = float("nan")
    water_cut_final: float = float("nan")
    n_points: int = 0
    n_fitted: int = 0
    r2: float = float("nan")
    reason: str = ""
    # Water in the very first period: breakthrough is before this history,
    # so 'time since breakthrough' - Chan's abscissa - is an assumption.
    water_from_start: bool = False

    def summary(self) -> str:
        if not self.ok:
            return f"  water diagnostic  : not run ({self.reason})"
        lines = [f"  water diagnostic  : final water cut "
                 f"{100 * self.water_cut_final:.1f} %, WOR "
                 f"{self.wor_final:,.2f}"]
        if self.breakthrough_t_days is not None:
            lines.append(
                f"                      breakthrough at day "
                f"{self.breakthrough_t_days:,.0f} "
                f"({self.breakthrough_t_days / DAYS_PER_YEAR:.1f} yr)")
        elif self.water_from_start:
            lines.append(
                "                      water from the first record: "
                "breakthrough is before this history, so\n"
                "                      Chan's time-since-breakthrough axis "
                "is assumed, and the slopes below with it.")
        if np.isfinite(self.wor_slope):
            lines.append(
                f"                      Chan WOR slope {self.wor_slope:+.2f} "
                f"over {self.n_fitted} periods after breakthrough "
                f"(R2 {self.r2:.2f})")
            # Below a WOR slope of about 0.3 the departure is 2*c2 divided by
            # a slope near zero, so it is arbitrarily large and means nothing.
            # A flat WOR is already the whole finding; printing a -7.9 next to
            # it would only invite someone to read a magnitude into noise.
            # On a FALLING WOR the departure 2*c2/a flips sign with a, and the
            # field report printed "+1.71 (-2.9 sigma)" - a curvature and its
            # significance with opposite signs. It only means something on
            # the rising WOR Chan's patterns are defined for.
            if self.wor_slope >= 0.3 and np.isfinite(
                    self.wor_prime_departure):
                lines.append(
                    f"                      WOR' slope "
                    f"{self.wor_prime_slope:+.2f}; departure from a straight "
                    f"log-log line {self.wor_prime_departure:+.2f} "
                    f"({self.curvature_z:+.1f} sigma)")
        # A mechanism read off a dozen wet months is a provisional reading,
        # and the number of periods behind it belongs next to it rather than
        # three lines up.
        if (self.n_fitted and self.n_fitted < 20
                and not self.mechanism.startswith(("no appreciable",
                                                   "INDETERMINATE"))):
            lines.append(
                f"                      PROVISIONAL: only {self.n_fitted} "
                "periods after breakthrough are fitted. Water\n"
                "                      mechanisms separate as the cone "
                "saturates or the channel accelerates,\n"
                "                      and neither has had time to happen "
                "yet.")
        lines.append(f"                      -> {self.mechanism}")
        return "\n".join(lines)


def chan_diagnostic(t_days: np.ndarray, q_oil: np.ndarray,
                    q_water: np.ndarray,
                    min_wor: float = 0.01) -> WaterCutDiagnostic:
    """Chan's WOR / WOR' log-log diagnostic."""
    t = np.asarray(t_days, dtype=float)
    qo = np.asarray(q_oil, dtype=float)
    qw = np.asarray(q_water, dtype=float)
    ok = (np.isfinite(t) & np.isfinite(qo) & np.isfinite(qw)
          & (qo > 0) & (qw >= 0) & (t > 0))
    if int(ok.sum()) < 10:
        return WaterCutDiagnostic(ok=False,
                                  reason="fewer than ten periods with oil, "
                                         "water and a time")
    t, qo, qw = t[ok], qo[ok], qw[ok]
    wor = qw / qo
    wc = qw / (qw + qo)

    prod = wor > min_wor
    if int(prod.sum()) < 8:
        return WaterCutDiagnostic(
            ok=True, mechanism="no appreciable water yet",
            wor_final=float(wor[-1]), water_cut_final=float(wc[-1]),
            n_points=int(len(wor)))
    bt_i = int(np.argmax(prod))
    tb = float(t[bt_i]) if bt_i > 0 else None
    from_start = bool(bt_i == 0)

    # Chan reads the trend AFTER breakthrough has settled, and the window has
    # to start clear of breakthrough itself. Anchoring it on the first wet
    # period puts log(WOR) at log(min_wor) on the left edge, so the fitted
    # slope is dominated by the climb out of that floor and comes back at +8
    # to +10 on data whose real late-time slope is nearer unity. The window
    # therefore starts at the later of: a quarter of the way through the wet
    # record, or the first period at five times the breakthrough threshold.
    wet_idx = np.nonzero(prod)[0]
    wet_idx = wet_idx[wet_idx >= bt_i]
    skip_frac = int(np.ceil(0.25 * len(wet_idx)))
    above = np.nonzero(wor[wet_idx] >= 5.0 * min_wor)[0]
    skip_lvl = int(above[0]) if len(above) else 0
    start = min(max(skip_frac, skip_lvl), max(len(wet_idx) - 8, 0))
    sel = wet_idx[start:]
    # The abscissa is time SINCE BREAKTHROUGH, not time since first oil.
    #
    # Chan plots against total producing time, which is fine in his field
    # cases because they break through early, so the two are nearly the same.
    # When breakthrough is late the total-time axis is actively misleading:
    # any WOR that rises from zero at t_bt carries a 1/(t - t_bt) factor in
    # its local slope, so it looks like it is flattening no matter what the
    # mechanism is. That is how an ordinary linear water cut came back
    # reading CONING with a WOR slope of +16 - the slope was measuring the
    # distance from breakthrough to the record start, not the water.
    t_bt = float(t[bt_i - 1]) if bt_i > 0 else 0.5 * float(t[0])
    tl = t[sel] - t_bt
    wl = wor[sel]
    if len(tl) < 8 or not np.all(tl > 0):
        return WaterCutDiagnostic(
            ok=True, mechanism="too little post-breakthrough history",
            breakthrough_t_days=tb, wor_final=float(wor[-1]),
            water_cut_final=float(wc[-1]), n_points=int(len(wor)))

    lt, lw = np.log(tl), np.log(np.maximum(wl, 1e-9))
    lr = stats.linregress(lt, lw)
    r2 = float(lr.rvalue ** 2)

    # The derivative comes from the FITTED curve, not from finite differences.
    #
    # The reason is arithmetic. If WOR follows a power law, WOR ~ t^a, then
    # WOR' ~ t^(a-1) exactly, so the derivative's log-log slope is a - 1 and
    # carries no information the WOR slope did not already carry. Everything
    # the derivative adds is the DEPARTURE from that identity - the curvature
    # of WOR on log-log - which is exactly the quantity a pointwise gradient
    # destroys. At 8 % noise on monthly rates a 3-point-smoothed gradient of a
    # perfectly straight displacement curve came back with a negative slope on
    # three runs in twelve, reading CONING off nothing but scatter.
    #
    # So: fit ln WOR against ln t with a quadratic, and differentiate it.
    #   ln W = c0 + c1 u + c2 u^2,  u = ln t
    #   a(u) = d ln W / d u = c1 + 2 c2 u          <- the WOR slope
    #   ln W' = ln W + ln a - u
    #   d ln W'/du = a + 2 c2 / a - 1              <- the WOR' slope
    # and the departure from the power-law identity is just 2 c2 / a: the
    # curvature, divided by the slope. Whether that departure is real is then
    # a question about c2 against its own standard error, which is a test the
    # data can actually answer.
    u0 = float(np.mean(lt))
    s1 = float(lr.slope)
    delta = float("nan")
    s2 = float("nan")
    curv_z = float("nan")
    if len(lt) >= 6 and np.ptp(lt) > 1e-6:
        X = np.column_stack([np.ones_like(lt), lt - u0, (lt - u0) ** 2])
        coef, *_ = np.linalg.lstsq(X, lw, rcond=None)
        resid = lw - X @ coef
        dof = len(lt) - 3
        if dof > 0:
            s2v = float(resid @ resid) / dof
            cov = s2v * np.linalg.pinv(X.T @ X)
            se_c2 = float(np.sqrt(max(cov[2, 2], 0.0)))
            c2 = float(coef[2])
            a0 = float(coef[1])          # slope at the window mid-point
            s1 = a0
            curv_z = c2 / se_c2 if se_c2 > 0 else float("nan")
            if abs(a0) > 1e-3:
                delta = 2.0 * c2 / a0
                s2 = a0 + delta - 1.0

    # A pointwise derivative is still computed, but only as a data-quality
    # flag and for the plot - never as the classifier.
    sm = wl.copy()
    if len(wl) >= 5:
        sm[1:-1] = np.array([np.median(wl[j - 1:j + 2])
                             for j in range(1, len(wl) - 1)])
    dwor = np.gradient(sm, tl)
    falling_frac = float((dwor <= 0).sum()) / max(len(dwor), 1)

    # Significance decides, effect size only guards against a curvature that
    # is real and trivial.
    #
    # The first version required |delta| > 0.5 as well as significance, and a
    # WOR slope above 1.5 for channelling. Both thresholds were calibrated on
    # full-life synthetic records, where coning has saturated hard and the
    # curvature is enormous. On eighteen wells truncated to a quarter or a
    # third of their lives they threw away exactly the cases the diagnostic
    # is for: three coning wells with curvature at -3.3 to -3.9 sigma, right
    # sign, were reported as normal displacement because |delta| came to 0.28
    # to 0.43; a channelling well at +2.1 sigma was lost because its WOR
    # slope was 1.25 rather than 1.5.
    #
    # Equally, a flat result is only evidence of a straight line when delta
    # itself is small. One well had delta = -2.09 at -0.19 sigma: no
    # significance at all, but nothing like a straight line either - the fit
    # simply could not see. That is INDETERMINATE, and saying "normal
    # displacement" there is a claim the data do not support.
    strong = np.isfinite(curv_z) and abs(curv_z) >= 2.0
    # A WOR that FALLS after breakthrough is none of Chan's patterns. The
    # field report classified a slope of -0.81 over 321 periods as "CONING
    # or a stabilised cone - WOR has gone flat", which it had not: it had
    # fallen, and a falling WOR points at the well or the data, not at the
    # reservoir's water mechanism.
    if s1 <= -0.3 and np.isfinite(lr.pvalue) and lr.pvalue < 0.05:
        mech = ("WOR FALLING after breakthrough - not one of Chan's patterns, "
                "and not coning.\n                      A falling WOR usually "
                "means an intervention (water shut-off, recompletion,\n"
                "                      a zone closed in), a change in how "
                "water is allocated or metered, or a\n                      "
                "water source that is itself depleting. Check the well "
                "history before reading\n                      a mechanism "
                "into it.")
    elif s1 < 0.3:
        mech = ("CONING or a stabilised cone - WOR has gone flat after "
                "breakthrough, which is\n                      the signature "
                "of a cone that reaches equilibrium. This one does "
                "respond\n                      to reducing the drawdown.")
    elif not np.isfinite(delta):
        mech = ("INDETERMINATE - the post-breakthrough WOR will not support "
                "a curvature fit")
    elif strong and delta > 0.7 and s1 > 0.5:
        mech = ("CHANNELLING - WOR is steepening and its derivative rises "
                "with it, which is\n                      flow behind pipe "
                "or a thief zone. Cutting the rate will not help; "
                "this\n                      needs a shut-off.")
    elif strong and delta < -0.2:
        mech = ("CONING - WOR climbs but is flattening, and its derivative "
                "falls away: the\n                      signature of a cone "
                "that stabilises. This one does respond to "
                "reducing\n                      the drawdown.")
    elif abs(delta) <= 0.7:
        # The two thresholds are deliberately asymmetric, because the two
        # mechanisms are not the same KIND of departure. Coning is
        # qualitative - the WOR stops rising - so any significant flattening
        # counts. Channelling is quantitative: the WOR rises faster than a
        # straight line, and so does an ordinary water cut climbing towards
        # 40 %. On synthetic wells the two separated at a departure of about
        # 0.7, with sweep running 0.3 to 0.55 and genuine channelling 1.3 to
        # 2.7. A symmetric threshold either loses the channels or calls
        # every rising water cut one.
        mech = ("normal displacement - WOR runs straight on log-log, "
                "consistent with ordinary\n                      sweep "
                "rather than a water problem.")
        if not strong:
            mech += ("\n                      The curvature is only "
                     f"{curv_z:+.1f} sigma, so this is the best reading "
                     "rather than a\n                      finding: a cone "
                     "that has not yet flattened, or a channel that has not "
                     "yet\n                      accelerated, looks the "
                     "same on this much record.")
    else:
        mech = ("INDETERMINATE - the WOR is not straight, but the curvature "
                f"is only {curv_z:+.1f} sigma,\n                      so it "
                "cannot be signed. A cone that has not yet flattened and an "
                "early\n                      channel look alike here; this "
                "needs more post-breakthrough history.")

    return WaterCutDiagnostic(
        water_from_start=from_start,
        ok=True, mechanism=mech, wor_slope=s1, wor_prime_slope=s2,
        wor_prime_departure=delta, curvature_z=curv_z,
        falling_fraction=falling_frac,
        breakthrough_t_days=tb, wor_final=float(wor[-1]),
        water_cut_final=float(wc[-1]), n_points=int(len(wor)),
        n_fitted=int(len(tl)), r2=r2)


MIN_PI_POINTS = 4
GOOD_PI_POINTS = 6


@dataclass
class OilPIDiagnostic:
    """Productivity index against time: damage and depletion, separated.

        PI = qo / (p_avg - p_wf)      STB/d/psi

    Dividing out the drawdown leaves mobility and contacted volume, so a
    falling index is a well problem rather than a reservoir one - skin growth,
    scale, a failing pump, liquid loading. Depletion moves p_avg, which the
    denominator already accounts for.

    Average reservoir pressure comes from a MEASURED survey on the same row
    wherever one exists; the material balance is the fallback, and the source
    is named, because a modelled p_avg carries the oil-in-place error with it.
    """
    ok: bool
    t_days: Optional[np.ndarray] = None
    pi: Optional[np.ndarray] = None
    p_avg: Optional[np.ndarray] = None
    pi_initial: float = float("nan")
    pi_final: float = float("nan")
    loss_frac: float = float("nan")
    trend_pct_per_year: float = float("nan")
    r2: float = float("nan")
    p_value: float = float("nan")
    n_points: int = 0
    p_avg_source: str = "measured p_res"
    below_bubble: bool = False
    reason: str = ""

    @property
    def indicative(self) -> bool:
        return bool(self.ok and 0 < self.n_points < GOOD_PI_POINTS)

    def summary(self) -> str:
        if not self.ok:
            return f"  PI diagnostic     : not run ({self.reason})"
        verb = "fell" if self.loss_frac >= 0 else "ROSE"
        head = (f"  PI diagnostic     : PI {verb} "
                f"{abs(100 * self.loss_frac):,.0f} % "
                f"({self.pi_initial:,.3g} -> {self.pi_final:,.3g} STB/d/psi), "
                f"trend {self.trend_pct_per_year:+,.1f} %/yr\n"
                f"                      (R2 {self.r2:.2f}, p "
                f"{self.p_value:.2g}, n={self.n_points}); p_avg from "
                f"{self.p_avg_source}")
        # The end-to-end change and the fitted trend are two readings of the
        # same series and must not be printed as if they agreed. A 19 % fall
        # next to a trend of +0 %/yr at p = 0.74 is not two findings, it is
        # one finding and one artefact, and the reader has no way to tell
        # which. Where the trend is not significant, the end-to-end number is
        # withdrawn rather than left standing next to its own refutation.
        # Significance is not materiality. With 120 clean monthly points a
        # trend of 0.4 %/yr comes back at p = 2e-7 - real, and worth nothing
        # to anyone deciding whether to work the well over.
        if (np.isfinite(self.trend_pct_per_year)
                and abs(self.trend_pct_per_year) < 1.0):
            head += ("\n                      Productivity is effectively "
                     "flat: under 1 %/yr either way, which is not\n"
                     "                      worth acting on however tight "
                     "the fit is.")
        elif np.isfinite(self.p_value) and self.p_value > 0.05:
            head += ("\n                      The trend is not significant "
                     "at 5 %, so the end-to-end change above is\n"
                     "                      within the scatter of this "
                     "series: read it as no measurable change in\n"
                     "                      productivity, not as a fall.")
        elif np.isfinite(self.p_value) and self.p_value <= 0.05 and (
                self.trend_pct_per_year * self.loss_frac > 0):
            head += ("\n                      The end-to-end change and the "
                     "fitted trend disagree in SIGN. Treat both\n"
                     "                      as unresolved and look at the "
                     "series itself before acting on either.")
        if self.below_bubble and self.loss_frac > 0:
            head += ("\n                      The reservoir is below its "
                     "bubble point over part of this window, so\n"
                     "                      some of the fall is free gas "
                     "taking relative permeability from the\n"
                     "                      oil - a reservoir effect, not "
                     "wellbore damage. Separating the two\n"
                     "                      needs a pressure-transient test.")
        if self.indicative:
            head += (f"\n                      INDICATIVE only - "
                     f"{self.n_points} flowing pressures is below the "
                     f"{GOOD_PI_POINTS} this wants.")
        return head


def oil_pi_diagnostic(t_days: np.ndarray, q_oil: np.ndarray,
                      p_wf: Optional[np.ndarray],
                      p_res: Optional[np.ndarray],
                      pvt: OilPVT,
                      np_stb: Optional[np.ndarray] = None,
                      n_ooip_stb: Optional[float] = None,
                      t_min: Optional[float] = None) -> OilPIDiagnostic:
    """Productivity index over time, from measured pressures where possible."""
    if p_wf is None:
        return OilPIDiagnostic(ok=False, reason="no flowing pressure column")
    t = np.asarray(t_days, float)
    q = np.asarray(q_oil, float)
    pw = np.asarray(p_wf, float)
    have = np.isfinite(pw) & (pw > 0)
    ok = np.isfinite(t) & np.isfinite(q) & (q > 0) & have
    n_have, n_ok = int(have.sum()), int(ok.sum())
    if n_ok < MIN_PI_POINTS:
        lost = n_have - n_ok
        detail = (f"{n_have} row(s) carry a flowing pressure but only {n_ok} "
                  "also have a rate")
        if lost > 0:
            detail += (f"; {lost} was/were recorded at zero rate (shut in), "
                       "where a productivity index cannot be formed")
        return OilPIDiagnostic(
            ok=False, reason=f"{detail} - {MIN_PI_POINTS} are needed")

    pr = (np.asarray(p_res, float) if p_res is not None
          else np.full_like(t, np.nan))
    t, q, pw, pr = t[ok], q[ok], pw[ok], pr[ok]
    if np_stb is not None:
        npv = np.asarray(np_stb, float)[ok]
    else:
        npv = np.full_like(t, np.nan)

    if t_min is not None and np.isfinite(t_min):
        keep = t >= float(t_min)
        if int(keep.sum()) >= MIN_PI_POINTS:
            t, q, pw, pr, npv = (t[keep], q[keep], pw[keep], pr[keep],
                                 npv[keep])

    # p_avg comes from ONE source for the whole series, never a mixture.
    #
    # Filling the gaps between surveys from a depletion model while keeping
    # the measured values where they exist sounds conservative and is not. On
    # a well with 19 surveys in 112 months the model ran 3,800 -> 2,200 psia
    # while the surveys ran 3,800 -> 1,060, so every survey month stepped the
    # computed PI one way and every modelled month stepped it back. The
    # resulting sawtooth swamped the real trend: the PI had fallen 19 % end
    # to end and the fitted trend came back at +0 %/yr, p = 0.74, on the same
    # numbers, in the same report.
    #
    # Where there are two or more surveys, they are interpolated against
    # cumulative oil - which is close to linear for a tank, and is the well's
    # own data rather than an assumed oil in place. The depletion model is
    # only for wells that have almost no surveys at all.
    src = "measured p_res"
    have_p = np.isfinite(pr) & (pr > 0)
    n_surv = int(have_p.sum())
    if n_surv == len(pr):
        p_avg = pr.copy()
    elif n_surv >= 2 and np.isfinite(npv).all() and np.ptp(npv[have_p]) > 0:
        order = np.argsort(npv[have_p])
        p_avg = np.interp(npv, npv[have_p][order], pr[have_p][order])
        src = (f"interpolated against cumulative oil between {n_surv} "
               f"measured survey(s)")
    elif n_surv >= 2 and np.ptp(t[have_p]) > 0:
        order = np.argsort(t[have_p])
        p_avg = np.interp(t, t[have_p][order], pr[have_p][order])
        src = f"interpolated against time between {n_surv} measured survey(s)"
    elif (n_ooip_stb is not None and np.isfinite(n_ooip_stb)
            and n_ooip_stb > 0 and np.isfinite(npv).any()):
        # A first-order depletion model: p falls in proportion to recovery.
        # Crude, and labelled as such.
        frac = np.clip(npv / float(n_ooip_stb), 0.0, 0.95)
        p_avg = pvt.p_init * (1.0 - frac)
        src = (f"a depletion model throughout ({n_surv} survey(s) is too few "
               "to interpolate)")
    else:
        return OilPIDiagnostic(
            ok=False,
            reason="fewer than two measured reservoir pressures on the "
                   "flowing rows, and no oil in place to model one from")

    dp = p_avg - pw
    good = np.isfinite(dp) & (dp > 0)
    if int(good.sum()) < MIN_PI_POINTS:
        return OilPIDiagnostic(
            ok=False,
            reason=f"only {int(good.sum())} point(s) have the flowing "
                   "pressure below the average reservoir pressure")
    t, q, p_avg, dp = t[good], q[good], p_avg[good], dp[good]
    pi = q / dp

    lr = stats.linregress(t / DAYS_PER_YEAR, np.log(np.maximum(pi, 1e-30)))
    n_edge = max(1, min(max(3, len(pi) // 10), len(pi) // 3))
    pi0 = float(np.median(pi[:n_edge]))
    pi1 = float(np.median(pi[-n_edge:]))
    return OilPIDiagnostic(
        ok=True, t_days=t, pi=pi, p_avg=p_avg,
        pi_initial=pi0, pi_final=pi1,
        loss_frac=float(1.0 - pi1 / pi0) if pi0 > 0 else float("nan"),
        trend_pct_per_year=float(100.0 * (math.exp(lr.slope) - 1.0)),
        r2=float(lr.rvalue ** 2), p_value=float(lr.pvalue),
        n_points=int(len(pi)), p_avg_source=src,
        below_bubble=bool(np.any(p_avg < pvt.p_bubble)))


# ==============================================================================
# SECTION 6 -- FORECAST: GOR MODEL, WATER MODEL, CONSTRAINTS
# ==============================================================================
#
# The oil rate declines on the fitted curve. Gas and water do NOT get their own
# independent declines - they are carried by the oil through a producing ratio,
# so the three streams cannot drift apart the way three separate Arps fits
# would. Gas comes from GOR(Np) and water from WOR(Np), both fitted to the
# recent record and both reported with the statistics that say whether they are
# worth believing.
#
# Cumulative oil is the argument rather than time because that is what the
# physics keys on. GOR rises because the reservoir has given up a certain
# fraction of its oil and the gas saturation has grown past its critical value;
# WOR rises because a certain fraction of the movable oil has been swept. Both
# are functions of how much has come out, not of how long the well has been
# open, and a well that is shut in for a year comes back with the same GOR and
# the same water cut it had when it stopped.


@dataclass
class RatioModel:
    """A producing ratio against cumulative oil, in one of several SHAPES.

    Used for both GOR and WOR. The shapes, all two-parameter except the last:

      log-linear    ln R = a + b Np        the X-plot; the usual choice
      linear        R    = a + b Np        a ratio rising at a steady rate
      power         ln R = a + b ln Np     scale-free growth
      logistic-cut  logit(R/(1+R)) = a + b Np   water cut saturating below 1
      constant      R    = a               the record supports no trend

    No record distinguishes these cleanly, and the choice between them moves
    the forecast more than the parameters within any one of them do. The
    log-linear form is the standard and is still the default reading, but
    extrapolated twenty years it is unbounded above, which is how a WOR trend
    once asked for 4.4e14 STB/d of water. `logistic-cut` is the shape that
    cannot do that: it works on the water CUT, which is bounded by one, so the
    WOR it implies saturates instead of exploding.

    Every shape's residuals are scored in LOG space whatever space it was
    fitted in, so their likelihoods are comparable and the Monte Carlo can
    draw among them by weight.
    """
    kind: str    # log-linear | linear | power | logistic-cut | constant | none
    label: str                      # "GOR" or "WOR"
    ln_r0: float                    # ln R at np_ref
    slope_per_stb: float            # d ln R / d Np
    np_ref_stb: float
    r_last: float = float("nan")
    r2: float = float("nan")
    p_value: float = float("nan")
    n_points: int = 0
    stderr_slope: float = float("nan")
    stderr_intercept: float = float("nan")
    floor: float = 0.0
    ceiling: float = float("inf")
    note: str = ""
    # Residual sum of squares in LOG space, and the fitted points, so every
    # shape can be scored against the others on one scale.
    log_rss: float = float("nan")
    fit_np_stb: Optional[np.ndarray] = None
    fit_ratio: Optional[np.ndarray] = None

    @property
    def log_residuals(self) -> Optional[np.ndarray]:
        if self.fit_np_stb is None or self.fit_ratio is None:
            return None
        pred = np.maximum(self(self.fit_np_stb), 1e-300)
        return (np.log(np.maximum(self.fit_ratio, 1e-300)) - np.log(pred))

    def __call__(self, np_stb) -> np.ndarray:
        x = np.asarray(np_stb, dtype=float)
        if self.kind == "none":
            return np.zeros_like(x)
        d = x - self.np_ref_stb
        if self.kind == "constant":
            r = np.full_like(x, math.exp(self.ln_r0))
        elif self.kind == "linear":
            # Fitted in linear space; the floor keeps a downward trend from
            # crossing zero and turning the ratio negative.
            r = math.exp(self.ln_r0) + self.slope_per_stb * d
        elif self.kind == "power":
            xr = np.maximum(x, 1.0)
            ref = max(self.np_ref_stb, 1.0)
            r = np.exp(self.ln_r0
                       + self.slope_per_stb * (np.log(xr) - math.log(ref)))
        elif self.kind == "logistic-cut":
            # a + b Np is the LOGIT of the water cut. wc -> 1 as Np grows, so
            # WOR = wc/(1-wc) still grows without bound - but it does so like
            # exp(b Np) only in the tail, and the cut itself can never exceed
            # one, which is the property the log-linear WOR lacks.
            z = np.clip(self.ln_r0 + self.slope_per_stb * d, -60.0, 60.0)
            wc = 1.0 / (1.0 + np.exp(-z))
            r = wc / np.maximum(1.0 - wc, 1e-9)
        else:                                            # log-linear
            r = np.exp(self.ln_r0 + self.slope_per_stb * d)
        return np.clip(np.asarray(r, dtype=float), self.floor, self.ceiling)

    def perturb(self, rng: np.random.Generator) -> "RatioModel":
        """One draw from the fit's own regression uncertainty.

        Without this the Monte Carlo treats a GOR trend fitted at R2 0.4 as if
        it were exact. On the gas tool the same omission collapsed the whole
        EUR distribution onto the deterministic answer and produced a P90
        above the P50; there is no reason to expect oil to be kinder.
        """
        if self.kind in ("none", "constant") or not (
                np.isfinite(self.stderr_slope)
                and np.isfinite(self.stderr_intercept)):
            return self
        return replace(
            self,
            ln_r0=float(rng.normal(self.ln_r0, self.stderr_intercept)),
            slope_per_stb=float(rng.normal(self.slope_per_stb,
                                           self.stderr_slope)))

    @property
    def significant(self) -> bool:
        return bool(self.kind not in ("none", "constant")
                    and self.n_points >= 8
                    and np.isfinite(self.p_value) and self.p_value < 0.05
                    and np.isfinite(self.r2) and self.r2 >= 0.15)

    @property
    def pct_per_mmstb(self) -> float:
        """Percent change in the ratio per million barrels of oil."""
        if self.kind in ("none", "constant"):
            return 0.0
        ref = max(self.np_ref_stb, 1.0)
        a = float(self(np.array([ref]))[0])
        b = float(self(np.array([ref + 1.0e6]))[0])
        if not (np.isfinite(a) and a > 0 and np.isfinite(b)):
            return 0.0
        return 100.0 * (b / a - 1.0)

    def summary(self) -> str:
        if self.kind == "none":
            return (f"  {self.label} model        : none"
                    + ("\n" + textwrap.fill(
                        self.note, width=100,
                        initial_indent=" " * 22,
                        subsequent_indent=" " * 22) if self.note else ""))
        if self.kind == "constant":
            return (f"  {self.label} model        : held constant at "
                    f"{math.exp(self.ln_r0):,.3g}"
                    + ("\n" + textwrap.fill(
                        self.note, width=100,
                        initial_indent=" " * 22,
                        subsequent_indent=" " * 22) if self.note else ""))
        return (f"  {self.label} model        : {self.kind}, "
                f"{self.pct_per_mmstb:+,.0f} % per MMstb of oil "
                f"(R2 {self.r2:.2f}, p {self.p_value:.1e}, n={self.n_points})"
                + ("\n" + textwrap.fill(self.note, width=100,
                                        initial_indent=" " * 22,
                                        subsequent_indent=" " * 22)
                   if self.note else ""))


def _fit_ratio_model(np_stb: np.ndarray, ratio: np.ndarray, label: str,
                     min_points: int = 8,
                     window: Optional[np.ndarray] = None,
                     floor: float = 0.0,
                     ceiling: float = float("inf"),
                     note: str = "") -> RatioModel:
    """Fit ln(ratio) against cumulative oil over a chosen window."""
    x = np.asarray(np_stb, dtype=float)
    r = np.asarray(ratio, dtype=float)
    ok = np.isfinite(x) & np.isfinite(r) & (r > 0)
    if window is not None:
        ok &= window
    n = int(ok.sum())
    r_last = float(r[np.isfinite(r)][-1]) if np.isfinite(r).any() else float("nan")
    if n == 0:
        return RatioModel(kind="none", label=label, ln_r0=0.0,
                          slope_per_stb=0.0, np_ref_stb=0.0, r_last=r_last,
                          note=note or "no positive readings")
    xs, rs_ = x[ok], r[ok]
    ref = float(xs[-1])
    if n < min_points or np.ptp(xs) <= 0:
        return RatioModel(
            kind="constant", label=label,
            ln_r0=float(np.log(max(np.median(rs_), 1e-12))),
            slope_per_stb=0.0, np_ref_stb=ref, r_last=r_last, n_points=n,
            floor=floor, ceiling=ceiling,
            note=note or f"only {n} usable point(s); held at the median")
    lr = stats.linregress(xs, np.log(rs_))
    # The intercept standard error is quoted at x = 0, which for a cumulative
    # of several million barrels is an extrapolation far outside the data and
    # makes the number meaningless. Re-express it at the reference point,
    # where it is the standard error of the fitted value.
    resid = np.log(rs_) - (lr.intercept + lr.slope * xs)
    dof = max(n - 2, 1)
    s2 = float(resid @ resid) / dof
    sxx = float(np.sum((xs - xs.mean()) ** 2))
    se_at_ref = math.sqrt(max(s2 * (1.0 / n + (ref - xs.mean()) ** 2
                                    / max(sxx, 1e-30)), 0.0))
    return RatioModel(
        kind="log-linear", label=label,
        ln_r0=float(lr.intercept + lr.slope * ref),
        slope_per_stb=float(lr.slope), np_ref_stb=ref, r_last=r_last,
        r2=float(lr.rvalue ** 2), p_value=float(lr.pvalue), n_points=n,
        stderr_slope=float(lr.stderr), stderr_intercept=se_at_ref,
        floor=floor, ceiling=ceiling, note=note)


def _fit_one_shape(x: np.ndarray, r: np.ndarray, kind: str, label: str,
                   floor: float, ceiling: float,
                   note: str) -> Optional["RatioModel"]:
    """Fit one shape. Each is fitted in ITS natural space."""
    n = len(x)
    ref = float(x[-1])
    lr_r = np.log(np.maximum(r, 1e-300))
    try:
        if kind == "constant":
            return RatioModel(
                kind="constant", label=label,
                ln_r0=float(np.log(max(np.median(r), 1e-12))),
                slope_per_stb=0.0, np_ref_stb=ref, n_points=n,
                floor=floor, ceiling=ceiling, note=note)
        if kind == "log-linear":
            lr = stats.linregress(x, lr_r)
            a, b = float(lr.intercept + lr.slope * ref), float(lr.slope)
            se_b = float(lr.stderr)
        elif kind == "linear":
            lr = stats.linregress(x, r)
            val = float(lr.intercept + lr.slope * ref)
            if not (val > 0):
                return None
            a, b = float(np.log(val)), float(lr.slope)
            se_b = float(lr.stderr)
        elif kind == "power":
            xs = np.maximum(x, 1.0)
            lr = stats.linregress(np.log(xs), lr_r)
            a = float(lr.intercept + lr.slope * math.log(max(ref, 1.0)))
            b, se_b = float(lr.slope), float(lr.stderr)
        elif kind == "logistic-cut":
            wc = r / (1.0 + r)
            wc = np.clip(wc, 1e-6, 1.0 - 1e-6)
            lr = stats.linregress(x, np.log(wc / (1.0 - wc)))
            a, b = float(lr.intercept + lr.slope * ref), float(lr.slope)
            se_b = float(lr.stderr)
        else:
            return None
    except Exception:
        return None
    if not (np.isfinite(a) and np.isfinite(b)):
        return None

    m = RatioModel(kind=kind, label=label, ln_r0=a, slope_per_stb=b,
                   np_ref_stb=ref, n_points=n, stderr_slope=se_b,
                   floor=floor, ceiling=ceiling, note=note)

    # EVERY shape is scored on the same scale - log of the ratio - whatever
    # space it was fitted in. Comparing a linear fit's residuals in barrels
    # against a log-linear fit's residuals in log-barrels is comparing two
    # different likelihoods and calling the smaller number better.
    pred = np.maximum(m(x), 1e-300)
    res = lr_r - np.log(pred)
    rss = float(res @ res)
    ss_tot = float(np.sum((lr_r - lr_r.mean()) ** 2))
    m = replace(
        m,
        r2=(1.0 - rss / ss_tot if ss_tot > 0 else float("nan")),
        p_value=float(getattr(lr, "pvalue", float("nan"))),
        stderr_intercept=float(math.sqrt(max(rss / max(n - 2, 1), 0.0)
                                         / max(n, 1))),
        log_rss=rss, fit_np_stb=x.copy(), fit_ratio=r.copy())
    return m


def fit_ratio_shapes(np_stb: np.ndarray, ratio: np.ndarray, label: str,
                     window: Optional[np.ndarray] = None,
                     shapes: Sequence[str] = ("log-linear", "linear", "power",
                                              "constant"),
                     min_points: int = 8, floor: float = 0.0,
                     ceiling: float = float("inf"),
                     note: str = "") -> Dict[str, "RatioModel"]:
    """Fit every candidate shape to the same window."""
    x = np.asarray(np_stb, dtype=float)
    r = np.asarray(ratio, dtype=float)
    ok = np.isfinite(x) & np.isfinite(r) & (r > 0)
    if window is not None:
        ok &= window
    n = int(ok.sum())
    if n == 0:
        return {}
    xs, rs_ = x[ok], r[ok]
    if n < min_points or np.ptp(xs) <= 0:
        m = _fit_one_shape(xs, rs_, "constant", label, floor, ceiling,
                           note or f"only {n} usable point(s); held at the "
                                   "median")
        return {"constant": m} if m is not None else {}
    out: Dict[str, RatioModel] = {}
    # Where the well IS now: the median of the last handful of readings. A
    # shape whose value at the last cumulative is far from this does not
    # describe the current state of the well, whatever its R2 - the forecast
    # would open with a step. On a field record the linear GOR shape began
    # its forecast at 70 scf/STB against a measured 480.
    n_recent = max(5, min(12, n // 6))
    recent = float(np.median(rs_[-n_recent:]))
    dropped: List[str] = []
    for kind in shapes:
        m = _fit_one_shape(xs, rs_, kind, label, floor, ceiling, note)
        if m is None or not np.isfinite(m.log_rss):
            continue
        v_now = float(m(xs[-1:])[0])
        if (kind != "constant" and recent > 0 and np.isfinite(v_now)
                and abs(math.log(max(v_now, 1e-12) / recent)) > math.log(1.35)):
            dropped.append(f"{kind} ({v_now:,.3g} vs {recent:,.3g})")
            continue
        out[kind] = m
    if not out:
        # Every trend opened with a step. Hold the ratio where it is and say
        # so, rather than forecast from a curve that does not pass through
        # the present.
        m = _fit_one_shape(xs, rs_, "constant", label, floor, ceiling,
                           (note + "; " if note else "")
                           + "every trend shape started far from the recent "
                             "readings, so the ratio is held at its recent "
                             "level")
        if m is not None:
            m = replace(m, ln_r0=float(math.log(max(recent, 1e-12))))
            out["constant"] = m
    elif dropped:
        for k in out:
            out[k] = replace(out[k], note=(out[k].note + "; " if out[k].note
                                           else "")
                             + "dropped for opening with a step: "
                             + ", ".join(dropped))
    return out


def ratio_shape_weights(shapes: Dict[str, "RatioModel"],
                        rho_cap: float = 0.95) -> Dict[str, float]:
    """Akaike weights over ratio shapes, on an effective sample size.

    The same correction the decline models get, and for the same reason: a
    producing ratio drifts away from its fitted shape for months at a time,
    so consecutive residuals are correlated and the record holds far less
    information than its length suggests. Rho is taken once, from the median
    across shapes, because AIC values are only comparable on a common sample
    size - computing it per shape rewards whichever happens to have whiter
    residuals regardless of fit.
    """
    if not shapes:
        return {}
    rhos, info = [], {}
    for k, m in shapes.items():
        if not (np.isfinite(m.log_rss) and m.log_rss > 0 and m.n_points >= 5):
            continue
        info[k] = m
        res = m.log_residuals
        if res is None or len(res) < 3:
            rhos.append(0.0)
            continue
        rhos.append(float(np.sum(res[1:] * res[:-1])
                          / max(float(res @ res), 1e-30)))
    if not info:
        return {}
    rho = float(np.clip(np.median(rhos) if rhos else 0.0, 0.0, rho_cap))
    rows = {}
    for k, m in info.items():
        n = int(m.n_points)
        kk = 1 if m.kind == "constant" else 2
        n_eff = float(np.clip(n * (1.0 - rho) / (1.0 + rho), kk + 2.0,
                              float(n)))
        aic = n_eff * math.log(m.log_rss / n) + 2.0 * kk
        denom = n_eff - kk - 1.0
        aicc = aic + (2.0 * kk * (kk + 1.0) / denom if denom > 0
                      else float("inf"))
        if np.isfinite(aicc):
            rows[k] = aicc
    if not rows:
        return {}
    best = min(rows.values())
    raw = {k: math.exp(-0.5 * (v - best)) for k, v in rows.items()}
    tot = sum(raw.values())
    return {k: v / tot for k, v in raw.items()} if tot > 0 else {}


def fit_gor_model(np_stb: np.ndarray, gor: np.ndarray, pvt: OilPVT,
                  diag: Optional[GORDiagnostic] = None,
                  min_points: int = 8
                  ) -> Tuple[RatioModel, Dict[str, RatioModel]]:
    """GOR against cumulative oil, fitted on the forward-relevant limb.

    Which part of the record to fit is decided by the diagnostic, not by a
    fixed fraction. Solution gas drive makes the GOR rise to a peak and then
    fall; fitting across the peak averages a rising limb against a falling one
    and forecasts a GOR that does neither. Where the diagnostic found a peak,
    only the falling limb after it is used, because that is the regime the
    well is in now and the one it will stay in.
    """
    x = np.asarray(np_stb, dtype=float)
    g = np.asarray(gor, dtype=float)
    note = ""
    window = None
    if diag is not None and diag.ok and diag.peaked and diag.peak_np_stb:
        window = x >= float(diag.peak_np_stb)
        if int((window & np.isfinite(g) & (g > 0)).sum()) >= min_points:
            note = ("fitted to the falling limb after the GOR peak - the "
                    "regime the well is in now")
        else:
            window = None
            note = ("the GOR has peaked but the falling limb is too short to "
                    "fit; the whole record is used and the forecast GOR will "
                    "be too high")
    if window is None and note == "":
        n_all = int((np.isfinite(g) & (g > 0)).sum())
        keep = max(min_points, n_all // 3)
        window = np.zeros_like(x, dtype=bool)
        idx = np.flatnonzero(np.isfinite(g) & (g > 0))
        if idx.size:
            window[idx[-keep:]] = True
        note = f"fitted to the last {min(keep, n_all)} periods"

    # A producing GOR below the solution GOR at reservoir conditions means gas
    # is being lost to measurement, fuel or flare, not that the reservoir has
    # stopped making it. The floor keeps the forecast from extrapolating a
    # metering problem into the future as though it were reservoir behaviour.
    floor = 0.10 * float(pvt.rsi)
    shapes = fit_ratio_shapes(x, g, "GOR", window=window,
                              min_points=min_points, floor=floor, note=note)
    # A shape fitted to the falling limb that then forecasts a RISE has not
    # described the falling limb - it has described whatever noise sat in the
    # window. On one field record a "falling limb" fit came back at +1,758 %
    # per MMstb and the report printed the two side by side without comment,
    # while the gas forecast ran to 3.7 times the recent GOR.
    if diag is not None and diag.ok and diag.peaked and shapes:
        rising = [k for k, m in shapes.items()
                  if k != "constant" and m.pct_per_mmstb > 0]
        if rising and len(rising) < len(shapes):
            shapes = {k: m for k, m in shapes.items() if k not in rising}
        elif rising:
            recent = float(np.median(g[np.isfinite(g) & (g > 0)][-8:]))
            shapes = {"constant": RatioModel(
                kind="constant", label="GOR",
                ln_r0=float(math.log(max(recent, 1e-12))),
                slope_per_stb=0.0, np_ref_stb=float(x[-1]),
                n_points=int(np.isfinite(g).sum()), floor=floor,
                note="every shape fitted to the falling limb forecast a "
                     "RISE, which contradicts the limb; held at the recent "
                     "level instead")}
    if not shapes:
        return _fit_ratio_model(x, g, "GOR", min_points=min_points,
                                window=window, floor=floor, note=note), {}
    w = ratio_shape_weights(shapes)
    # The deterministic reading is the best-supported SHAPE, not a fixed one.
    # Picking log-linear regardless would be picking a shape the data may
    # prefer three to one against, and then reporting its parameters to two
    # decimal places.
    best = max(w, key=w.get) if w else next(iter(shapes))
    return shapes[best], shapes


def fit_wor_model(np_stb: np.ndarray, q_oil: np.ndarray, q_water: np.ndarray,
                  min_points: int = 8, recent_fraction: float = 0.5
                  ) -> Tuple[RatioModel, Dict[str, RatioModel]]:
    """WOR against cumulative oil - the X-plot - fitted to the wet record.

    Only periods that are actually making water are fitted. Including the dry
    months before breakthrough puts a long flat run of zeros (or of a floor
    value) at the left of the fit and drags the slope down, which forecasts a
    well that never drowns.
    """
    x = np.asarray(np_stb, dtype=float)
    qo = np.asarray(q_oil, dtype=float)
    qw = np.asarray(q_water, dtype=float)
    ok = np.isfinite(qo) & (qo > 0) & np.isfinite(qw)
    wor = np.where(ok, qw / np.maximum(qo, 1e-12), np.nan)
    wet = ok & (wor > 1.0e-3)
    n_wet = int(wet.sum())
    if n_wet < min_points:
        return RatioModel(
            kind="none", label="WOR", ln_r0=0.0, slope_per_stb=0.0,
            np_ref_stb=0.0,
            r_last=(float(wor[ok][-1]) if int(ok.sum()) else float("nan")),
            n_points=n_wet,
            note=f"only {n_wet} period(s) with measurable water - no water "
                 "forecast is made, so the life below is a DRY-well life and "
                 "an upper bound"), {}
    idx = np.flatnonzero(wet)
    keep = max(min_points, int(recent_fraction * idx.size))
    window = np.zeros_like(x, dtype=bool)
    window[idx[-keep:]] = True
    note = f"fitted to the last {min(keep, idx.size)} wet periods"
    # The logistic shape is offered for water and not for gas, because it is
    # the water CUT that is bounded by one. There is no equivalent ceiling on
    # a producing GOR.
    shapes = fit_ratio_shapes(
        x, wor, "WOR", window=window, min_points=min_points, floor=0.0,
        note=note,
        shapes=("log-linear", "linear", "power", "logistic-cut", "constant"))
    if not shapes:
        return _fit_ratio_model(x, wor, "WOR", min_points=min_points,
                                window=window, floor=0.0, note=note), {}
    w = ratio_shape_weights(shapes)
    best = max(w, key=w.get) if w else next(iter(shapes))
    return shapes[best], shapes


@dataclass
class OilForecast:
    """Oil, gas and water forward, and what ends the well."""
    table: pd.DataFrame
    eur_oil_mstb: float
    eur_gas_mmscf: float
    eur_water_mstb: float
    remaining_oil_mstb: float
    remaining_gas_mmscf: float
    remaining_water_mstb: float
    economic_life_years: float
    forecast_years: float
    abandonment_reason: str
    constraint_years: Dict[str, float] = field(default_factory=dict)
    constraints_breached_at_start: List[str] = field(default_factory=list)
    # Limits the user set that could not be applied, with the reason. An
    # abandonment pressure needs a fitted N to turn production into pressure;
    # without one it was dropped without a word, and the report listed the
    # 1,000 psia in its settings as if it had shaped the forecast.
    constraints_not_applied: Dict[str, str] = field(default_factory=dict)
    recovery_factor: float = float("nan")
    gas_cap_fraction: float = float("nan")   # Gp at EUR / gas originally there
    p_implied_end_psia: float = float("nan")
    p_source: str = ""
    water_cut_end: float = float("nan")
    warnings_text: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"  EUR oil           : {self.eur_oil_mstb:,.0f} Mstb "
            f"({self.remaining_oil_mstb:,.0f} Mstb remaining)",
            f"  EUR gas           : {self.eur_gas_mmscf:,.0f} MMscf "
            f"({self.remaining_gas_mmscf:,.0f} MMscf remaining)",
            f"  EUR water         : {self.eur_water_mstb:,.0f} Mstb",
            f"  economic life     : {self.economic_life_years:,.1f} yr "
            f"({self.forecast_years:,.1f} yr of forecast), ends on "
            f"{self.abandonment_reason}",
        ]
        if self.abandonment_reason == "max forecast life":
            lines.append(
                "                      THE HORIZON SET THIS EUR, NOT THE "
                "RESERVOIR. The well never reached any\n                     "
                " limit inside the forecast window, so the volumes above say "
                "how much a\n                      curve this flat yields in "
                f"{self.forecast_years:,.0f} years and nothing about when "
                "the well dies.\n                      Shorten the horizon "
                "or look at why the fitted decline has no end.")
        if np.isfinite(self.recovery_factor):
            lines.append(
                f"  recovery factor   : {100 * self.recovery_factor:.1f} % of "
                "the fitted oil in place")
            # A recovery factor is the one number in an oil forecast that a
            # reader can sanity-check against the rest of the industry
            # without any of this tool's assumptions, so it is worth saying
            # when it lands outside the range that any drive mechanism
            # supports rather than leaving it to be noticed.
            if self.recovery_factor > 0.70:
                lines.append(
                    "                      ABOVE 70 % - no drive mechanism "
                    "recovers that much oil. Either the\n                     "
                    " decline is too flat or the oil in place is too small; "
                    "the forecast is\n                      not a forecast of "
                    "this reservoir.")
            elif self.recovery_factor > 0.55:
                lines.append(
                    "                      high - only a well-swept "
                    "waterflood reaches this. Worth checking\n                "
                    "      against analogues before it is used.")
        if np.isfinite(self.gas_cap_fraction):
            lines.append(
                f"  gas produced      : {100 * self.gas_cap_fraction:.0f} % of "
                "the gas the balance says the reservoir started with")
        if np.isfinite(self.water_cut_end):
            lines.append(
                f"  water cut at end  : {100 * self.water_cut_end:.1f} %")
        if np.isfinite(self.p_implied_end_psia):
            lines.append(
                f"  implied p at end  : {self.p_implied_end_psia:,.0f} psia "
                f"({self.p_source})")
        if self.constraint_years:
            # A limit the fitted decline reaches in 1,534 years has not been
            # reached. Printing the number invites someone to read it as a
            # long life rather than as what it is - a decline curve with no
            # meaningful terminal behaviour, usually b pinned at its bound.
            def _fmt(k: str, v: float) -> str:
                if not np.isfinite(v) or v > 200.0:
                    return f"{k} never"
                return f"{k} {v:,.1f} yr"
            lines.append("  constraints       : " + ", ".join(
                _fmt(k, v) for k, v in sorted(
                    self.constraint_years.items(), key=lambda kv: kv[1])))
            if any(np.isfinite(v) and v > 200.0
                   for v in self.constraint_years.values()):
                lines.append(
                    "                      A constraint shown as 'never' is "
                    "one the fitted curve does not reach\n                   "
                    "   within 200 years. That is a statement about the fit, "
                    "not about the well.")
        for k, why in self.constraints_not_applied.items():
            lines.append(
                textwrap.fill(f"NOT APPLIED       : {k} - {why}.",
                              width=100, initial_indent="  ",
                              subsequent_indent=" " * 22)
                + "\n                      The forecast above ran without "
                  "this limit.")
        if self.constraints_breached_at_start:
            lines.append(
                "                      ALREADY PAST: "
                + ", ".join(self.constraints_breached_at_start)
                + " - the well is past this limit on the first forecast "
                  "step, so the\n                      life above is the last "
                  "historical date, not a forecast of anything.")
        for w in self.warnings_text:
            lines.append(f"                      {w}")
        return "\n".join(lines)


def pressure_path_from_balance(pvt: OilPVT, n_ooip_stb: float,
                               m_gas_cap: float, we_rb: float,
                               np_stb: np.ndarray, gp_scf: np.ndarray,
                               wp_stb: np.ndarray,
                               p_floor: float = 120.0,
                               n_grid: int = 600) -> np.ndarray:
    """Invert the three-term balance for pressure at each withdrawal, at once.

    The obvious implementation bisects for each point, and each residual
    evaluation asks the PVT for Bo, Rs, Bg and Bt at a single pressure. That
    is four one-element array operations per iteration, sixty iterations per
    point, forty points per forecast - about nineteen thousand PVT calls for
    one realisation, which made a 250-draw Monte Carlo slower than everything
    else in the analysis put together.

    Instead the PVT is evaluated ONCE on a fixed pressure grid, the residual
    is formed for every point against that whole grid at the same time, and
    the root is found by interpolating across the sign change. Same equation,
    same answer to well inside a psi, about sixty times faster.
    """
    npv = np.atleast_1d(np.asarray(np_stb, dtype=float))
    gpv = np.atleast_1d(np.asarray(gp_scf, dtype=float))
    wpv = np.atleast_1d(np.asarray(wp_stb, dtype=float))
    out = np.full(npv.shape, np.nan)
    if not (np.isfinite(n_ooip_stb) and n_ooip_stb > 0):
        return out

    # The property grid depends only on the pressure range, which is the same
    # on every call, so it is built once and kept on the PVT object. Bg goes
    # through the z-factor, which is iterative; recomputing it on 600 points
    # per realisation cost 20 ms a draw and made an 800-draw Monte Carlo take
    # twenty seconds where the forecast itself takes none. The cached values
    # are the SAME numbers the function used before - the grid was already
    # fixed - so nothing about the answer changes, only how often it is
    # worked out.
    key = (round(float(p_floor), 6), int(n_grid), round(float(pvt.p_init), 6))
    cache = getattr(pvt, "_prop_grid_cache", None)
    if cache is None:
        cache = {}
        try:
            object.__setattr__(pvt, "_prop_grid_cache", cache)
        except Exception:
            cache = None
    hit = cache.get(key) if cache is not None else None
    if hit is None:
        pg = np.linspace(p_floor, pvt.p_init, int(n_grid))   # (G,)
        hit = (pg, pvt.bo(pg), pvt.rs(pg), pvt.bg(pg), pvt.bt(pg))
        if cache is not None:
            cache[key] = hit
    pg, bo, rs, bg, bt = hit
    swc = pvt.sw_initial
    Eo = bt - pvt.bti
    Eg = pvt.boi * (bg / pvt.bgi - 1.0)
    Efw = ((1.0 + m_gas_cap) * pvt.boi
           * ((pvt.cw_per_psi * swc + pvt.cf_per_psi) / (1.0 - swc))
           * (pvt.p_init - pg))
    rhs = n_ooip_stb * (Eo + m_gas_cap * Eg + Efw) + we_rb * (
        (pvt.p_init - pg) / max(pvt.p_init - p_floor, 1.0))

    live = np.isfinite(npv) & (npv > 0) & np.isfinite(gpv) & np.isfinite(wpv)
    if not live.any():
        return out
    ni, gi, wi = npv[live, None], gpv[live, None], wpv[live, None]
    rp = gi / np.maximum(ni, 1e-9)
    F = ni * (bo[None, :] + (rp - rs[None, :]) * bg[None, :]) + wi * pvt.bw
    resid = F - rhs[None, :]                                  # (k, G)

    # resid falls as the pressure falls, so it runs from positive at the
    # initial pressure down through zero. Find the first grid cell whose ends
    # straddle zero and interpolate inside it.
    sgn = np.signbit(resid)
    cross = np.diff(sgn.astype(np.int8), axis=1) != 0
    vals = np.full(ni.shape[0], np.nan)
    rows, cols = np.nonzero(cross)
    seen = set()
    for r, c in zip(rows, cols):
        if r in seen:
            continue
        seen.add(r)
        r0, r1 = resid[r, c], resid[r, c + 1]
        w = 0.0 if r1 == r0 else r0 / (r0 - r1)
        vals[r] = pg[c] + w * (pg[c + 1] - pg[c])
    out[live] = vals
    return out


def _pressure_from_balance(pvt: OilPVT, n_ooip_stb: float, m_gas_cap: float,
                           we_rb: float, np_stb: float, gp_scf: float,
                           wp_stb: float, p_floor: float = 120.0) -> float:
    """One point of `pressure_path_from_balance`."""
    return float(pressure_path_from_balance(
        pvt, n_ooip_stb, m_gas_cap, we_rb,
        np.array([np_stb]), np.array([gp_scf]), np.array([wp_stb]),
        p_floor=p_floor)[0])


def forecast_oil(fit: FitResult,
                 gor_model: RatioModel,
                 wor_model: RatioModel,
                 pvt: OilPVT,
                 q_econ_stbd: float,
                 np_to_date_stb: float = 0.0,
                 gp_to_date_scf: float = 0.0,
                 wp_to_date_stb: float = 0.0,
                 t_start_days: float = 0.0,
                 t_max_years: float = 30.0,
                 step_days: float = 30.4375,
                 n_ooip_stb: Optional[float] = None,
                 m_gas_cap: float = 0.0,
                 we_to_date_rb: float = 0.0,
                 water_cut_econ: Optional[float] = None,
                 q_water_econ_stbd: Optional[float] = None,
                 p_abandon_psia: Optional[float] = None,
                 q_liquid_cap_stbd: Optional[float] = None,
                 n_pressure_stb: Optional[float] = None) -> OilForecast:
    """Roll the oil decline forward and carry gas and water on it.

    `t_max_years` is the length of the FORECAST, measured from the last
    historical record, not the total life of the well. Reading it the other
    way is how a 14-year-old well asked for a 10-year forecast got an
    abandonment date earlier than its own last record, and remaining reserves
    that came out negative; the horizon is anchored on `t_start_days` so the
    window cannot close before it opens.
    """
    model = fit.model
    notes: List[str] = []

    horizon_days = max(float(t_max_years) * DAYS_PER_YEAR, step_days)
    t_end_max = float(t_start_days) + horizon_days

    # A zero or negative economic rate means no rate limit, not a division by
    # zero in the middle of the forecast.
    rate_limit_on = bool(np.isfinite(q_econ_stbd) and q_econ_stbd > 0)
    t_ab = (model.time_to_rate(q_econ_stbd,
                               t_max=max(t_end_max - model.t0, step_days))
            if rate_limit_on else float("inf"))
    if not np.isfinite(t_ab):
        t_ab = t_end_max
    t_ab = float(min(max(t_ab, t_start_days + step_days), t_end_max))

    n_steps = int(np.floor((t_ab - t_start_days) / step_days))
    t = t_start_days + step_days * np.arange(max(n_steps, 0) + 1)
    if t[-1] < t_ab - 1e-6:
        t = np.append(t, t_ab)
    if t.size < 2:
        t = np.array([t_start_days, t_ab])

    def _series(tt: np.ndarray) -> Dict[str, np.ndarray]:
        """Every forecast stream on an arbitrary time grid."""
        qo = model.rate(tt)                                 # STB/d
        cm = model.cum(tt)                                  # Mstb from t0
        c0 = float(model.cum(np.array([t_start_days]))[0])
        np_m = np_to_date_stb / STB_PER_MSTB + (cm - c0)
        np_s = np_m * STB_PER_MSTB
        g = gor_model(np_s)                                 # scf/STB
        qg = qo * g / SCF_PER_MSCF                          # Mscf/d
        w = wor_model(np_s)
        qw = qo * w                                         # STB/d
        # A WOR extrapolated log-linearly has no upper bound, and nothing was
        # bounding it. On a 60-month record with a steep early water trend the
        # forecast produced 7.6 x 10^12 Mstb of water - more water than the
        # planet has - because WOR = exp(a + b*Np) was evaluated over another
        # twenty years of cumulative oil and multiplied by the oil rate. This
        # is the same failure the gas module had when an unbounded CGR made a
        # wellstream 41 times more condensate than there was stream to carry
        # it. A well's inflow and lift do not improve with age, so the total
        # liquid it can move is bounded by what it has already demonstrated.
        if q_liquid_cap_stbd is not None and np.isfinite(q_liquid_cap_stbd) \
                and q_liquid_cap_stbd > 0:
            qw = np.minimum(qw, np.maximum(q_liquid_cap_stbd - qo, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            wc = np.where(qo + qw > 0, qw / (qo + qw), 0.0)
        gp = (gp_to_date_scf / SCF_PER_MSCF / MSCF_PER_MMSCF
              + np.concatenate([[0.0], cumulative_trapezoid(qg, tt)])
              / MSCF_PER_MMSCF)
        wp = (wp_to_date_stb / STB_PER_MSTB
              + np.concatenate([[0.0], cumulative_trapezoid(qw, tt)])
              / STB_PER_MSTB)
        return {"q_oil": qo, "np_mstb": np_m, "np_stb": np_s, "gor": g,
                "q_gas": qg, "wor": w, "q_water": qw, "wcut": wc,
                "gp_mmscf": gp, "wp_mstb": wp}

    S = _series(t)
    if (q_liquid_cap_stbd is not None and np.isfinite(q_liquid_cap_stbd)
            and q_liquid_cap_stbd > 0):
        raw_w = S["q_oil"] * wor_model(S["np_stb"])
        n_clip = int(np.sum(raw_w > S["q_water"] + 1e-9))
        if n_clip:
            notes.append(
                f"The forecast water rate was held at the liquid ceiling of "
                f"{q_liquid_cap_stbd:,.0f} STB/d on {n_clip} of "
                f"{len(raw_w)} steps\n                      (the WOR trend "
                f"asked for up to {float(np.max(raw_w)):,.0f} STB/d). An "
                f"extrapolated WOR has no upper bound of its own; this one "
                f"is\n                      the well's own demonstrated "
                f"liquid rate, and the water volumes are not a forecast "
                f"beyond it.")
    q_oil, np_mstb, np_stb = S["q_oil"], S["np_mstb"], S["np_stb"]
    gor, q_gas, wor, q_water, wcut = (S["gor"], S["q_gas"], S["wor"],
                                      S["q_water"], S["wcut"])
    gp_mmscf, wp_mstb = S["gp_mmscf"], S["wp_mstb"]

    # -- constraints -------------------------------------------------------
    # Every limit is evaluated on a PROBE grid that runs far past the
    # forecast window, so a constraint that does not bind inside the window
    # still reports the year it would. Recording only the binding constraint
    # was the first version, and it made a well that drowns at 90 % water in
    # year 44 and dies on rate in year 39 indistinguishable from one with no
    # water problem at all: the water-cut limit simply vanished from the
    # report because it was never reached. The probe grid is monthly out to
    # the horizon and yearly after it, to 200 years.
    PROBE_YEARS = 200.0
    # The yearly probe runs from the forecast START, not from the horizon.
    # Anchoring it on the horizon left a gap between the end of the monthly
    # window and the first yearly point whenever a constraint cut the
    # forecast short of the horizon - three and a half years on the first
    # well tried, during which any crossing was invisible.
    t_probe = np.unique(np.concatenate([
        t,
        np.arange(t_start_days,
                  t_start_days + PROBE_YEARS * DAYS_PER_YEAR,
                  DAYS_PER_YEAR)]))
    P = _series(t_probe)

    reason = "oil rate" if rate_limit_on else "max forecast life"
    constraint_years: Dict[str, float] = {}
    breached: List[str] = []
    cut_at = len(t)

    if rate_limit_on:
        # t_ab above has already been CLAMPED to the horizon, so quoting it as
        # the rate date would report the horizon under the wrong name.
        t_rate_true = model.time_to_rate(q_econ_stbd,
                                         t_max=PROBE_YEARS * DAYS_PER_YEAR)
        constraint_years["oil rate"] = (float(t_rate_true / DAYS_PER_YEAR)
                                        if np.isfinite(t_rate_true)
                                        else float("inf"))
        # The fitted curve can already be below the economic rate where the
        # forecast starts. The rate limit was then clamped to one step past
        # the start, so the well was given a month of production beyond its
        # own limit - 1 Mstb on a field record whose curve read 31 STB/d
        # against a 50 STB/d limit - and the report said "ends on oil rate"
        # with the rate constraint dated 24 years in the past. Every other
        # limit already stops dead when it is breached at the start; the rate
        # limit now does too.
        q_start = float(model.rate(np.array([t_start_days]))[0])
        if np.isfinite(q_start) and q_start < q_econ_stbd:
            breached.append("oil rate")
            cut_at, reason = 0, "oil rate"
            notes.append(
                f"The fitted curve reads {q_start:,.1f} STB/d at the last "
                f"record, below the {q_econ_stbd:,.0f} STB/d economic "
                "rate.\n                      Either the well is already "
                "sub-economic at that limit, or the fit window\n"
                "                      does not describe the current rate.")

    def _apply(label: str, key: str, limit: Optional[float],
               above: bool = True,
               probe_series: Optional[np.ndarray] = None,
               window_series: Optional[np.ndarray] = None) -> None:
        """Record when a limit binds, and cut the forecast if it binds early."""
        nonlocal cut_at, reason
        if limit is None or not np.isfinite(limit) or limit <= 0:
            return
        ps = P[key] if probe_series is None else probe_series
        hit_p = np.flatnonzero(ps > limit if above else ps < limit)
        constraint_years[label] = (
            float(t_probe[max(int(hit_p[0]) - 1, 0)] / DAYS_PER_YEAR)
            if hit_p.size else float("inf"))
        ws = (S[key] if window_series is None else window_series)
        hit = np.flatnonzero(ws > limit if above else ws < limit)
        if not hit.size:
            return
        i = int(hit[0])
        if i == 0:
            breached.append(label)
        if i < cut_at:
            cut_at, reason = i, label

    _apply("water cut", "wcut", water_cut_econ)
    _apply("water rate", "q_water", q_water_econ_stbd)

    if n_ooip_stb is not None and np.isfinite(n_ooip_stb) and n_ooip_stb > 0:
        if np_to_date_stb >= n_ooip_stb:
            notes.append(
                f"the well has already produced {np_to_date_stb / 1e6:,.1f} "
                f"MMstb against a fitted oil in place of "
                f"{n_ooip_stb / 1e6:,.1f} MMstb. That is a statement about "
                "the material balance, not about the well.")
        _apply("oil in place", "np_stb", float(n_ooip_stb))

        # Gas in place is a real cumulative ceiling too: the reservoir cannot
        # give up more gas than it started with, solution plus gas cap.
        g_total_scf = (n_ooip_stb * pvt.rsi
                       + (m_gas_cap * n_ooip_stb * pvt.boi / max(pvt.bgi, 1e-30)
                          if m_gas_cap > 0 else 0.0))
        g_total_mmscf = g_total_scf / SCF_PER_MSCF / MSCF_PER_MMSCF
        if g_total_mmscf > 0:
            _apply("gas in place", "gp_mmscf", g_total_mmscf)

    # Implied reservoir pressure, on a coarse grid - the balance is inverted
    # by bisection and doing it at every monthly step is not worth the time.
    #
    # The aquifer is FROZEN at the influx to date: no further water comes in.
    # That is wrong on a well with an active aquifer, and wrong in a known
    # direction - it understates the pressure support, so the implied pressure
    # is a floor and the abandonment date it produces is early. Modelling the
    # influx forward would need an aquifer fitted on its own, which this does
    # not have, and guessing one would put an invented number into the one
    # output whose job is to be a reality check.
    p_end = float("nan")
    p_source = ""
    not_applied: Dict[str, str] = {}
    # The N used to turn production into pressure. It defaults to the
    # oil-in-place CAP, but the two are different jobs: with the cap turned
    # off the first version had no N here at all, so a fitted N sat in the
    # material balance while the abandonment pressure was skipped unsaid.
    n_p = n_pressure_stb if n_pressure_stb is not None else n_ooip_stb
    want_p_limit = (p_abandon_psia is not None
                    and np.isfinite(p_abandon_psia) and p_abandon_psia > 0)
    if want_p_limit and not (n_p is not None
                             and np.isfinite(n_p) and n_p > 0):
        not_applied["abandonment pressure"] = (
            "needs an oil in place to turn production into pressure, and the "
            "material balance did not determine one")
    if (n_p is not None and np.isfinite(n_p)
            and n_p > 0 and len(t) >= 2):
        def _p_on(grid_np, grid_gp, grid_wp):
            vals = pressure_path_from_balance(
                pvt, float(n_p), float(m_gas_cap),
                float(we_to_date_rb), grid_np,
                grid_gp * MSCF_PER_MMSCF * SCF_PER_MSCF,
                grid_wp * STB_PER_MSTB)
            if not np.isfinite(vals).any():
                return None
            good = np.isfinite(vals)
            if not good.all():
                vals = np.interp(np.arange(len(vals)),
                                 np.flatnonzero(good), vals[good])
            return vals

        # The pressure path is only needed when an abandonment pressure is
        # set as a constraint; otherwise one inversion at the end is enough
        # for the reported figure. Each inversion is a 60-step bisection with
        # four PVT evaluations inside it, so building both full paths cost
        # about 19,000 PVT calls per realisation and made a 250-draw Monte
        # Carlo take longer than every other part of the analysis together.
        want_path = (p_abandon_psia is not None
                     and np.isfinite(p_abandon_psia))
        if want_path:
            p_win = _p_on(np_stb, gp_mmscf, wp_mstb)
            p_prb = _p_on(P["np_stb"], P["gp_mmscf"], P["wp_mstb"])
            if p_win is not None and p_prb is not None:
                p_source = ("balance inverted on the fitted N, with the "
                            "aquifer frozen at its influx to date")
                _apply("reservoir pressure", "", float(p_abandon_psia),
                       above=False, probe_series=p_prb, window_series=p_win)
            else:
                not_applied["abandonment pressure"] = (
                    "the balance could not be inverted for pressure on this "
                    "forecast")
        else:
            p_source = ("balance inverted on the fitted N, with the aquifer "
                        "frozen at its influx to date")

    if cut_at < len(t):
        if cut_at == 0:
            # The very first forecast step already violates a limit, so there
            # is no forecast: the well is past that limit today. The earlier
            # version kept a two-row minimum to avoid a degenerate table,
            # which manufactured one month of production PAST the breached
            # constraint - so a report that said "ALREADY PAST: reservoir
            # pressure" went on to promise 90 Mstb more oil at an implied
            # pressure of 858 psia against a 900 psia limit, in the same
            # eleven lines. The table now holds the starting point twice:
            # same shape, zero remaining, nothing invented.
            sl = np.array([0, 0])
        else:
            sl = np.arange(cut_at)
        t, q_oil, q_gas, q_water = (t[sl], q_oil[sl], q_gas[sl], q_water[sl])
        gor, wor, wcut = gor[sl], wor[sl], wcut[sl]
        np_mstb, gp_mmscf, wp_mstb = (np_mstb[sl], gp_mmscf[sl], wp_mstb[sl])
        np_stb = np_stb[sl]
    if t[-1] >= t_end_max - 1e-9 and reason in ("oil rate",
                                                "max forecast life"):
        reason = "max forecast life"

    if p_source:
        p_end = _pressure_from_balance(
            pvt, float(n_p), float(m_gas_cap), float(we_to_date_rb),
            float(np_stb[-1]),
            float(gp_mmscf[-1] * MSCF_PER_MMSCF * SCF_PER_MSCF),
            float(wp_mstb[-1] * STB_PER_MSTB))

    # The decline fit's own health belongs in the forecast, not only in the
    # fit report. A b pinned at its upper bound produces a curve that never
    # declines, and every volume below inherits that - so it is said here,
    # next to the numbers it governs.
    weak = list(getattr(fit, "weak_params", []) or [])
    if weak:
        notes.append(
            f"The decline fit has {', '.join(weak)} at or near a bound. A "
            "parameter at its bound is not\n                      a fitted "
            "value, and the volumes above inherit whatever the bound was set "
            "to.")

    # A ratio model that the data do not support is still a model, and the
    # forecast will use it without complaint. On one record a WOR trend
    # fitted at R2 0.08 quietly stopped the water-cut limit from binding at
    # all and the life went from 21 years to 39, with nothing in the report
    # to say the water forecast had become an extrapolation of scatter.
    for rm in (wor_model, gor_model):
        if rm.kind == "log-linear" and not rm.significant:
            notes.append(
                f"The {rm.label} trend is NOT statistically supported "
                f"(R2 {rm.r2:.2f}, p {rm.p_value:.2g}). It is still used "
                f"above,\n                      because there is nothing "
                f"better, but the "
                + ("life and the water volumes"
                   if rm.label == "WOR" else "gas volumes")
                + " below rest on it.")

    if wor_model.kind == "none":
        notes.append(
            "No water forecast: the record has too few wet periods to fit "
            "one. The life above\n                      is therefore a "
            "dry-well life and an UPPER BOUND - water is what usually ends "
            "an oil well.")

    table = pd.DataFrame({
        "t_days": t,
        "t_years": t / DAYS_PER_YEAR,
        "q_oil_stbd": q_oil,
        "q_gas_mscfd": q_gas,
        "q_water_stbd": q_water,
        "gor_scf_per_stb": gor,
        "wor": wor,
        "water_cut": wcut,
        "Np_mstb": np_mstb,
        "Gp_mmscf": gp_mmscf,
        "Wp_mstb": wp_mstb,
    })

    g_total_mmscf = float("nan")
    if n_ooip_stb is not None and np.isfinite(n_ooip_stb) and n_ooip_stb > 0:
        g_total_mmscf = (n_ooip_stb * pvt.rsi
                         + (m_gas_cap * n_ooip_stb * pvt.boi
                            / max(pvt.bgi, 1e-30) if m_gas_cap > 0 else 0.0)
                         ) / SCF_PER_MSCF / MSCF_PER_MMSCF

    return OilForecast(
        table=table,
        eur_oil_mstb=float(np_mstb[-1]),
        eur_gas_mmscf=float(gp_mmscf[-1]),
        eur_water_mstb=float(wp_mstb[-1]),
        remaining_oil_mstb=float(np_mstb[-1] - np_to_date_stb / STB_PER_MSTB),
        remaining_gas_mmscf=float(
            gp_mmscf[-1] - gp_to_date_scf / SCF_PER_MSCF / MSCF_PER_MMSCF),
        remaining_water_mstb=float(
            wp_mstb[-1] - wp_to_date_stb / STB_PER_MSTB),
        economic_life_years=float(t[-1] / DAYS_PER_YEAR),
        forecast_years=float((t[-1] - t_start_days) / DAYS_PER_YEAR),
        abandonment_reason=reason,
        constraint_years=constraint_years,
        constraints_breached_at_start=breached,
        constraints_not_applied=not_applied,
        recovery_factor=(float(np_mstb[-1] * STB_PER_MSTB / n_ooip_stb)
                         if n_ooip_stb else float("nan")),
        gas_cap_fraction=(float(gp_mmscf[-1] / g_total_mmscf)
                          if np.isfinite(g_total_mmscf) and g_total_mmscf > 0
                          else float("nan")),
        p_implied_end_psia=p_end,
        p_source=p_source,
        water_cut_end=float(wcut[-1]),
        warnings_text=notes,
    )


def model_weights(fits: Dict[str, FitResult],
                  rho_cap: float = 0.95) -> Dict[str, float]:
    """Akaike weights over the fitted declines, on an EFFECTIVE sample size.

    Plain AICc treats every monthly point as an independent observation. On
    production data they are nothing of the sort: a decline curve that mis-
    tracks the rate does so for months at a time, so the residuals are
    strongly autocorrelated and the information in the record is a fraction
    of its length. On one synthetic well the lag-1 residual correlation was
    0.86 - eighty-five monthly points carrying about six points' worth of
    independent information - and plain AICc duly separated two models by
    22 units and handed one of them a weight of 1.000. It is not possible to
    be that sure which curve a well is following after seven years.

    The standard correction is used:

        n_eff = n (1 - rho) / (1 + rho)

    clamped to at least k + 2, so the AICc small-sample term stays finite, and
    to at most n, so negative autocorrelation cannot manufacture confidence
    the record does not have. Weights are then the usual
    exp(-dAICc/2), normalised.

    These are weights for AVERAGING OVER models, not for picking one. The
    point is that a forecast should carry the disagreement between defensible
    curves rather than hide it behind whichever won on a statistic.
    """
    # ONE effective sample size, shared by every model.
    #
    # Computing n_eff per model looks more careful and is wrong: AIC values
    # are only comparable when they describe the same data, and n_eff
    # multiplies the log-likelihood term. A model with whiter residuals then
    # gets a larger n_eff and a much larger |AIC| whatever its fit, which on
    # one well handed 84 % of the weight to the third-best curve purely
    # because its residuals happened to be less correlated. The deflation has
    # to be a property of the RECORD, not of the candidate, so rho is taken
    # as the median across the fitted models and applied to all of them.
    resid: Dict[str, Tuple[np.ndarray, int, int]] = {}
    rhos: List[float] = []
    n_obs = 0
    for name, fr in fits.items():
        try:
            q = np.asarray(fr.q_fit, dtype=float)
            t = np.asarray(fr.t_fit, dtype=float)
            r = (np.log(np.maximum(q, 1e-12))
                 - np.log(np.maximum(fr.model.rate(t), 1e-12)))
            r = r - r.mean()
            n = int(len(r))
            k = int(len(fr.params))
            rss = float(r @ r)
            if n < k + 3 or rss <= 0:
                continue
            resid[name] = (r, n, k)
            n_obs = max(n_obs, n)
            rhos.append(float(np.sum(r[1:] * r[:-1]) / max(rss, 1e-30)))
        except Exception:
            continue
    if not resid:
        return {}
    rho = float(np.clip(np.median(rhos), 0.0, rho_cap))
    n_eff_base = n_obs * (1.0 - rho) / (1.0 + rho)

    rows: Dict[str, float] = {}
    for name, (r, n, k) in resid.items():
        rss = float(r @ r)
        n_eff = float(np.clip(n_eff_base, k + 2.0, float(n)))
        aic = n_eff * math.log(rss / n) + 2.0 * k
        denom = n_eff - k - 1.0
        aicc = aic + (2.0 * k * (k + 1.0) / denom if denom > 0
                      else float("inf"))
        if np.isfinite(aicc):
            rows[name] = aicc

    if not rows:
        return {}
    best = min(rows.values())
    raw = {k: math.exp(-0.5 * (v - best)) for k, v in rows.items()}
    tot = sum(raw.values())
    return {k: v / tot for k, v in raw.items()} if tot > 0 else {}


def monte_carlo_oil_eur(fit: FitResult,
                        gor_model: RatioModel,
                        wor_model: RatioModel,
                        pvt: OilPVT,
                        q_econ_stbd: float,
                        np_to_date_stb: float = 0.0,
                        gp_to_date_scf: float = 0.0,
                        wp_to_date_stb: float = 0.0,
                        t_start_days: float = 0.0,
                        n_samples: int = 1000,
                        t_max_years: float = 30.0,
                        b_prior: Optional[Tuple[float, float]] = None,
                        fits: Optional[Dict[str, FitResult]] = None,
                        gor_shapes: Optional[Dict[str, "RatioModel"]] = None,
                        wor_shapes: Optional[Dict[str, "RatioModel"]] = None,
                        n_ooip_stb: Optional[float] = None,
                        n_ooip_rel_sigma: float = 0.15,
                        n_ooip_hard_max: Optional[float] = None,
                        m_gas_cap: float = 0.0,
                        we_to_date_rb: float = 0.0,
                        water_cut_econ: Optional[float] = None,
                        q_water_econ_stbd: Optional[float] = None,
                        p_abandon_psia: Optional[float] = None,
                        q_liquid_cap_stbd: Optional[float] = None,
                        n_pressure_stb: Optional[float] = None,
                        max_rel_sd: float = 0.35,
                        seed: int = 11) -> pd.DataFrame:
    """Probabilistic oil EUR: fit covariance, ratio-model scatter, N scatter.

    Every constraint the deterministic forecast honours is passed to every
    realisation. A probabilistic EUR that ignores a limit the base case obeys
    is not a distribution around that case, it is a distribution around a
    different well - on the gas tool that omission put the P90 ABOVE the
    deterministic answer, which is nonsense on its face.

    The GOR and WOR models are perturbed through their own regression
    standard errors rather than held fixed, for the same reason: a water
    forecast fitted at R2 0.6 that is treated as exact collapses the spread
    of any well that ends on water.

    Results follow the petroleum convention - P90 is the low case.
    """
    rng = np.random.default_rng(seed)

    def _prep(fr: FitResult):
        """Mean, capped covariance and bounds for one fitted model."""
        c = type(fr.model)
        nm = list(c.param_names)
        mu = np.array([fr.params[q] for q in nm], dtype=float)
        cv = np.array(fr.cov, dtype=float)
        if cv.shape != (len(nm), len(nm)) or not np.all(np.isfinite(cv)):
            cv = np.diag((0.10 * np.abs(mu)) ** 2)
        cv = 0.5 * (cv + cv.T)
        # Cap each parameter's sampled sd without destroying the correlations,
        # which is where the parameter trade-offs live.
        sdv = np.sqrt(np.clip(np.diag(cv), 0.0, None))
        if max_rel_sd is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                cr = np.where(np.outer(sdv, sdv) > 0,
                              cv / np.outer(sdv, sdv), 0.0)
            np.fill_diagonal(cr, 1.0)
            sdv = np.minimum(sdv, max_rel_sd * np.abs(mu))
            cv = cr * np.outer(sdv, sdv)
        cv = 0.5 * (cv + cv.T)
        ev = np.linalg.eigvalsh(cv)
        if ev.min() < 0:
            cv = cv + np.eye(len(cv)) * (abs(ev.min()) + 1e-18)
        return c, nm, mu, cv, c.bounds(fr.t_fit - fr.t0, fr.q_fit)

    # MODEL FORM is sampled too, when the candidate fits are supplied.
    #
    # Sampling one model's covariance measures how well that curve's
    # parameters are pinned down. It does not measure whether it is the right
    # curve, and on blind tests against withheld production that was the term
    # that mattered: the band built from parameters alone contained the
    # eventual outturn in 2 cases of 18. Each realisation therefore draws a
    # model first, with probability equal to its autocorrelation-corrected
    # Akaike weight, and then draws that model's parameters.
    plan: List[Tuple[FitResult, float]] = []
    weights: Dict[str, float] = {}
    if fits:
        weights = model_weights(fits)
        for nm_, w_ in weights.items():
            if w_ > 0.005 and nm_ in fits:
                plan.append((fits[nm_], w_))
    if not plan:
        plan = [(fit, 1.0)]
    tot_w = sum(w for _, w in plan)
    plan = [(f_, w / tot_w) for f_, w in plan]

    # RATIO SHAPES are sampled alongside the decline. Extrapolated twenty
    # years, the candidate WOR shapes on one well gave water-oil ratios from
    # 2.6 to 70 at the same cumulative - a factor of 27 that a band built from
    # one shape's parameters cannot see at all.
    def _shape_plan(shapes, fallback):
        if not shapes:
            return [(fallback, 1.0)]
        wts = ratio_shape_weights(shapes)
        pl = [(shapes[k], v) for k, v in wts.items()
              if v > 0.005 and k in shapes]
        if not pl:
            return [(fallback, 1.0)]
        tw = sum(v for _, v in pl)
        return [(m, v / tw) for m, v in pl]

    g_plan = _shape_plan(gor_shapes, gor_model)
    w_plan = _shape_plan(wor_shapes, wor_model)
    g_models = [m for m, _ in g_plan]
    g_probs = np.array([v for _, v in g_plan], dtype=float)
    w_models = [m for m, _ in w_plan]
    w_probs = np.array([v for _, v in w_plan], dtype=float)

    rows: List[Dict[str, float]] = []
    n_failed = 0
    for fr_i, w_i in plan:
        want = int(round(w_i * n_samples))
        if fr_i is plan[-1][0]:
            want = max(n_samples - len(rows), 0)
        if want <= 0:
            continue
        cls, names, mean, cov, bnds = _prep(fr_i)
        got0 = len(rows)
        attempts, max_attempts = 0, 40 * max(want, 1)
        batch = max(256, want // 2)
        while len(rows) - got0 < want and attempts < max_attempts:
            draws = rng.multivariate_normal(mean, cov, size=batch)
            attempts += batch
            for x in draws:
                if len(rows) - got0 >= want:
                    break
                pars = dict(zip(names, x))
                if b_prior and "b" in pars:
                    pars["b"] = float(rng.normal(*b_prior))
                if any(not np.isfinite(v)
                       or v < bnds.get(k, (-np.inf, np.inf))[0]
                       or v > bnds.get(k, (-np.inf, np.inf))[1]
                       for k, v in pars.items()):
                    continue
                g_draw = g_models[int(rng.choice(len(g_models), p=g_probs))]
                w_draw = w_models[int(rng.choice(len(w_models), p=w_probs))]
                try:
                    mdl = cls(t0=fr_i.t0, **pars)
                    fr = replace(fr_i, model=mdl, params=pars, stderr={})
                    cap = None
                    if n_ooip_stb is not None and np.isfinite(n_ooip_stb):
                        cap = float(n_ooip_stb * rng.lognormal(
                            0.0, n_ooip_rel_sigma))
                        # min(F/Et) is not an estimate with a symmetric error. It
                        # is an UPPER bound on N that survey scatter biases LOW,
                        # never high, so a lognormal draw around the fitted N must
                        # still be held under it. Sampled freely, a tenth of the
                        # realisations sat above a number the report calls a bound
                        # that no oil in place may exceed.
                        if (n_ooip_hard_max is not None
                                and np.isfinite(n_ooip_hard_max)):
                            cap = min(cap, float(n_ooip_hard_max))
                        cap = max(cap, np_to_date_stb * 1.01)
                    n_pr = None
                    if n_pressure_stb is not None and np.isfinite(n_pressure_stb):
                        n_pr = max(float(n_pressure_stb * rng.lognormal(
                            0.0, n_ooip_rel_sigma)), np_to_date_stb * 1.01)
                    fc = forecast_oil(
                        fr, g_draw.perturb(rng), w_draw.perturb(rng), pvt,
                        q_econ_stbd,
                        np_to_date_stb=np_to_date_stb,
                        gp_to_date_scf=gp_to_date_scf,
                        wp_to_date_stb=wp_to_date_stb,
                        t_start_days=t_start_days, t_max_years=t_max_years,
                        n_ooip_stb=cap, m_gas_cap=m_gas_cap,
                        we_to_date_rb=we_to_date_rb,
                        water_cut_econ=water_cut_econ,
                        q_water_econ_stbd=q_water_econ_stbd,
                        p_abandon_psia=p_abandon_psia,
                        q_liquid_cap_stbd=q_liquid_cap_stbd,
                        n_pressure_stb=n_pr)
                except (ValueError, ArithmeticError, RuntimeError,
                        FloatingPointError):
                    # Numerical failures on an extreme draw are expected and
                    # skipped. Programming errors are NOT: a bare `except
                    # Exception` here swallowed a NameError on every single
                    # realisation and returned an empty frame, which the
                    # caller then indexed into and crashed on a missing
                    # column - three steps away from the actual fault.
                    n_failed += 1
                    continue
                if not (np.isfinite(fc.eur_oil_mstb)
                        and fc.eur_oil_mstb >= np_to_date_stb / STB_PER_MSTB):
                    continue
                rows.append({
                    "model": fr_i.model_name,
                    "gor_shape": g_draw.kind,
                    "wor_shape": w_draw.kind,
                    "eur_oil_mstb": fc.eur_oil_mstb,
                    "eur_gas_mmscf": fc.eur_gas_mmscf,
                    "eur_water_mstb": fc.eur_water_mstb,
                    "life_years": fc.economic_life_years,
                    "recovery_factor": fc.recovery_factor,
                    "water_cut_end": fc.water_cut_end,
                    "ends_on_water": float(fc.abandonment_reason.startswith(
                        "water")),
                    "ends_on_oil_in_place": float(
                        fc.abandonment_reason == "oil in place"),
                    "ends_on_pressure": float(
                        fc.abandonment_reason == "reservoir pressure"),
                })

    cols = ["model", "gor_shape", "wor_shape", "eur_oil_mstb",
            "eur_gas_mmscf", "eur_water_mstb", "life_years",
            "recovery_factor", "water_cut_end", "ends_on_water",
            "ends_on_oil_in_place", "ends_on_pressure"]
    # Always return the columns, even with no rows, so a caller that indexes
    # them gets an empty series rather than a KeyError from three frames away.
    out = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(
        {c: pd.Series(dtype="float64") for c in cols})
    if len(out) < max(50, n_samples // 20):
        warnings.warn(
            f"only {len(out)} of {n_samples} Monte Carlo realisations were "
            f"usable ({n_failed} failed outright) - the sampled parameters "
            "are mostly landing outside the model bounds, so the percentiles "
            "below are not a distribution. Check the fit before quoting "
            "them.")
    return out


def _ordinal(k: int) -> str:
    if 10 <= k % 100 <= 20:
        return f"{k}th"
    return f"{k}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(k % 10, 'th') }"


def summarise_oil_mc(mc: pd.DataFrame, deterministic: OilForecast,
                     n_requested: int = 0) -> str:
    """Percentiles, in the petroleum convention, with the caveats that matter."""
    if mc is None or len(mc) == 0:
        return "  probabilistic EUR : not run (no usable realisations)"
    e = mc["eur_oil_mstb"].to_numpy(float)
    p90, p50, p10 = np.percentile(e, [10, 50, 90])
    det = float(deterministic.eur_oil_mstb)
    lines = [
        f"  probabilistic EUR : P90 {p90:,.0f} / P50 {p50:,.0f} / "
        f"P10 {p10:,.0f} Mstb  (n={len(mc)})",
        f"                      deterministic {det:,.0f} Mstb"]
    # Where the base case sits in its own distribution. Once model form is
    # sampled the selected curve is one of several, and it can land near an
    # edge - which means the base case is not the central case, however it is
    # labelled. A reader comparing a single number against P90/P50/P10 has no
    # way to see that unless it is said.
    pctl = 100.0 * float(np.mean(e <= det))
    # A band too narrow to hold anything. When the forecast adds next to
    # nothing to the history, every realisation lands within a rounding
    # error of every other, and 'where the base case sits' is decided by
    # the last decimal. The field report said the base case was at the
    # "92th percentile" and "outside its own P90-P10" of a band printed as
    # 429 / 429 / 429 - and then blamed the choice of curve.
    degenerate = bool((p10 - p90) <= max(0.005 * abs(p50), 1.0))
    if degenerate:
        lines.append(
            f"                      Every realisation gives the same EUR to "
            f"within {p10 - p90:,.1f} Mstb: the forecast adds\n"
            "                      almost nothing to the history, so there is "
            "no spread to read. Where the\n                      base case "
            "sits in it, and which curves were sampled, do not matter here.")
        return "\n".join(lines)
    if len(e) >= 20 and not (25.0 <= pctl <= 75.0):
        lines.append(
            f"                      The base case sits at the "
            f"{_ordinal(int(round(pctl)))} percentile of its own "
            "distribution, not the "
            f"middle.\n                      It uses one decline curve; the "
            "realisations use several, weighted by fit. Read\n"
            "                      P50 as the central case and the "
            "deterministic figure as one member of\n"
            "                      the family.")
    if not (p90 <= det <= p10):
        # This check caught the gas tool twice, when the cause was a
        # constraint the realisations honoured and the base case did not. It
        # now has a second, entirely legitimate cause: the base case uses the
        # SELECTED decline curve while the realisations use the weighted
        # ensemble, and where those disagree the base case can sit outside
        # its own band without anything being wrong. Reporting the old
        # message there would send a reader hunting for a constraint mismatch
        # that does not exist.
        det_model = getattr(deterministic, "model_name", None) or ""
        mix_share = 0.0
        top_model = ""
        if "model" in mc and len(mc):
            vc = mc["model"].value_counts(normalize=True)
            top_model = str(vc.index[0])
            det_model = det_model or top_model
            mix_share = float(vc.iloc[0])
        if top_model and mix_share >= 0.5:
            lines.append(
                f"                      The base case lies outside its own "
                f"P90-P10. {100 * mix_share:.0f} % of realisations use\n"
                f"                      {top_model}, which the weights "
                f"prefer, while the base case uses the curve you\n"
                f"                      selected. That is a disagreement "
                f"about WHICH CURVE, not about the limits -\n"
                f"                      read P50 as the central case, or set "
                f"the model selection to 'auto'.")
        else:
            lines.append(
                "                      THE DETERMINISTIC CASE LIES OUTSIDE "
                "ITS OWN P90-P10, and the model mix\n                      "
                "does not explain it. The realisations are not being run "
                "under the same\n                      constraints as the "
                "base case; do not quote these percentiles.")
    band = 100.0 * (p10 - p90) / 2.0 / max(p50, 1e-9)
    # Where nearly every realisation ends on the same limit, the spread stops
    # being a spread on the EUR and becomes a spread on the parameters of
    # whichever curve that limit is read off. That is a much smaller thing,
    # and a +/-1 % band invites the reader to believe the opposite.
    for key, label, what in (
            ("ends_on_water", "water", "WOR trend"),
            ("ends_on_oil_in_place", "the oil-in-place cap", "cap"),
            ("ends_on_pressure", "reservoir pressure", "balance")):
        if key in mc and float(mc[key].mean()) >= 0.90:
            lines.append(
                f"                      {100 * float(mc[key].mean()):.0f} % "
                f"of realisations end on {label}, so this +/-{band:.1f} % "
                f"band is\n                      the uncertainty in the "
                f"fitted {what}, not in the EUR. The SHAPE of that curve "
                f"is\n                      assumed, and that assumption is "
                f"not in these numbers.")
            break
    else:
        mix = ""
        if "model" in mc:
            vc = mc["model"].value_counts(normalize=True)
            if len(vc) > 1:
                mix = ", ".join(f"{k} {100 * v:.0f} %"
                                for k, v in vc.items() if v >= 0.02)
        for col, what in (("gor_shape", "GOR"), ("wor_shape", "WOR")):
            if col in mc:
                vc2 = mc[col].value_counts(normalize=True)
                if len(vc2) > 1:
                    mix += (f"; {what} shape " + ", ".join(
                        f"{k} {100 * v:.0f} %"
                        for k, v in vc2.items() if v >= 0.02))
        if mix:
            lines.append(
                textwrap.fill(f"Sampled: {mix}.", width=100,
                              initial_indent=" " * 22,
                              subsequent_indent=" " * 22) + "\n"
                f"                      The +/-{band:.1f} % band covers both "
                "the parameters and the disagreement between\n"
                "                      curves. On synthetic wells given a "
                "quarter to a third of their life as\n"
                "                      history it contained the eventual "
                "outturn in 11 cases of 18 - and in all 6\n"
                "                      where nothing else in the report "
                "raised a warning. It is a range worth\n"
                "                      quoting, not a guarantee.")
        else:
            lines.append(
                f"                      The +/-{band:.1f} % band is "
                "parameter scatter within ONE assumed decline shape.\n"
                "                      Model choice is not sampled and is "
                "usually the larger term: sampled the same\n"
                "                      way on synthetic wells, this band "
                "contained the eventual outturn in 2 cases\n"
                "                      of 18, against 11 of 18 once the "
                "choice of curve and ratio shape were sampled.")
    if n_requested and len(mc) < 0.5 * n_requested:
        lines.append(
            f"                      only {len(mc)} of {n_requested} draws "
            "were usable; treat the percentiles as indicative.")
    return "\n".join(lines)


# ==============================================================================
# SECTION 5 -- SYNTHETIC WELL, FOR TESTING AGAINST A KNOWN ANSWER
# ==============================================================================

def make_synthetic_oil_well(pvt: OilPVT,
                            well: str = "OIL-1",
                            n_ooip_stb: float = 50.0e6,
                            m_gas_cap: float = 0.0,
                            we_total_rb: float = 0.0,
                            q_plateau_stbd: float = 6000.0,
                            plateau_frac: float = 0.35,
                            n_months: int = 120,
                            start: str = "2015-01-01",
                            water_mechanism: str = "none",
                            water_breakthrough_frac: float = 0.35,
                            p_wf_frac: float = 0.35,
                            pi_stbd_per_psi: Optional[float] = None,
                            pi_decline_per_year: float = 0.0,
                            p_wf_min: float = 200.0,
                            noise_frac: float = 0.05,
                            downtime_prob: float = 0.08,
                            survey_every: int = 6,
                            free_gas_mult: float = 3.0,
                            seed: int = 17) -> pd.DataFrame:
    """A monthly history from a tank whose answer is known.

    The pressure is solved from the material balance at each step, so the
    rates, the cumulatives and the reported pressures are mutually consistent
    - which is what makes recovering N a test rather than a rehearsal. The
    water mechanism shapes the WOR so the Chan diagnostic has a truth to be
    scored against:

        none          no water
        displacement  WOR climbs slowly and nearly linearly in time
        coning        WOR jumps at breakthrough then flattens
        channelling   WOR and its derivative climb together
    """
    rng = np.random.default_rng(seed)
    boi, bgi, bti, swc = pvt.boi, pvt.bgi, pvt.bti, pvt.sw_initial
    dates = pd.date_range(start, periods=n_months, freq="MS")
    period = np.array([(dates[i + 1] - dates[i]).days if i + 1 < n_months
                       else 30 for i in range(n_months)], dtype=float)

    uptime = np.ones(n_months)
    down = rng.random(n_months) < downtime_prob
    uptime[down] = rng.uniform(0.25, 0.85, size=int(down.sum()))

    Np = Gp = Wp = 0.0
    p = pvt.p_init
    rows = []
    cum_dp = 0.0
    bt_step: Optional[int] = None
    def _solve_pressure(Np_t: float, Gp_t: float, Wp_t: float) -> float:
        """Pressure from the balance for a given withdrawal."""
        def _resid(pp: float) -> float:
            pp = float(np.clip(pp, 120.0, pvt.p_init))
            bo = float(pvt.bo(np.array([pp]))[0])
            rsx = float(pvt.rs(np.array([pp]))[0])
            bg = float(pvt.bg(np.array([pp]))[0])
            bt = float(pvt.bt(np.array([pp]))[0])
            rp = Gp_t / max(Np_t, 1e-9)
            F = Np_t * (bo + (rp - rsx) * bg) + Wp_t * pvt.bw
            Eo = bt - bti
            Eg = boi * (bg / bgi - 1.0)
            Efw = ((1.0 + m_gas_cap) * boi
                   * ((pvt.cw_per_psi * swc + pvt.cf_per_psi) / (1.0 - swc))
                   * (pvt.p_init - pp))
            we = we_total_rb * (pvt.p_init - pp) / max(pvt.p_init - 120.0, 1.0)
            return F - (n_ooip_stb * (Eo + m_gas_cap * Eg + Efw) + we)
        # resid falls as the pressure falls (more expansion for the same
        # withdrawal), so resid > 0 means the trial pressure is too HIGH and
        # the bracket closes from above. Getting that the wrong way round
        # drove every well to the pressure floor on the first step, which then
        # pinned Rs near zero and made the GOR history nonsense.
        lo, hi = 120.0, pvt.p_init
        if not (_resid(lo) <= 0 <= _resid(hi)):
            return float("nan")
        for _ in range(60):
            if hi - lo < 1.0e-3:          # psi; far finer than any survey
                break
            mid = 0.5 * (lo + hi)
            if _resid(mid) > 0:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    for i in range(n_months):
        days = period[i] * uptime[i]
        # Rate: on plateau while the reservoir can support it, then declining
        # with the oil mobility as gas comes out of solution.
        frac = Np / n_ooip_stb
        if frac < plateau_frac:
            qo = q_plateau_stbd
        else:
            below = max(pvt.p_bubble - p, 0.0) / max(pvt.p_bubble, 1.0)
            qo = q_plateau_stbd * math.exp(-2.2 * (frac - plateau_frac)) \
                * (1.0 - 0.55 * below)
        qo = max(qo * float(rng.lognormal(0.0, noise_frac)), 1.0)

        # Deliverability caps the schedule. A well cannot produce whatever the
        # decline curve asks for: it can only produce what the drawdown it has
        # available will lift. Without this cap the generator kept asking for
        # 6,000 STB/d at a PI of 3.0 with 1,800 psia in the reservoir - 2,000
        # psi of drawdown out of 1,800 psi of pressure - so the flowing
        # pressure pinned to its floor from month 35 on, the drawdown stopped
        # tracking the PI, and a well built to lose 8 %/yr of its
        # productivity reported itself flat.
        if pi_stbd_per_psi is not None and pi_stbd_per_psi > 0:
            yrs = float(i) * 30.4 / DAYS_PER_YEAR
            pi_now = (pi_stbd_per_psi * math.exp(-pi_decline_per_year * yrs)
                      if pi_decline_per_year else pi_stbd_per_psi)
            qo = min(qo, pi_now * max(p - p_wf_min, 1.0))
            qo = max(qo, 1.0)

        dNp = qo * days

        rs = float(pvt.rs(np.array([p]))[0])
        d = max(pvt.p_bubble - p, 0.0) / max(pvt.p_bubble, 1.0)
        gor = rs * (1.0 + free_gas_mult * d)
        dGp = gor * dNp

        # Water, by mechanism.
        #
        # The ramp variable x runs on TIME SINCE BREAKTHROUGH over the rest of
        # the record, not on recovery factor. The first version normalised by
        # (1 - breakthrough_frac), i.e. it assumed the well would recover the
        # whole tank; a well that stops at 42 % recovery after breaking through
        # at 35 % only ever reached x = 0.11, so the accelerating shape never
        # got off the floor and the channelling case finished at 0.7 % water
        # cut with nothing for the diagnostic to read.
        wc_target = 0.0
        if water_mechanism != "none":
            if bt_step is None and frac > water_breakthrough_frac:
                bt_step = i
            if bt_step is not None:
                x = (i - bt_step) / max(n_months - 1 - bt_step, 1)
                # Chan's three signatures, written as water-cut shapes:
                #   displacement  gentle and nearly linear
                #   coning        a jump at breakthrough that then SATURATES,
                #                 so WOR flattens and its derivative falls
                #   channelling   ACCELERATING, so WOR and its derivative climb
                #                 together
                if water_mechanism == "displacement":
                    wc_target = 0.40 * x
                elif water_mechanism == "coning":
                    wc_target = 0.45 * (1.0 - math.exp(-9.0 * x))
                elif water_mechanism == "channelling":
                    wc_target = min(0.93, 0.90 * x ** 2.4)
        qw = (qo * wc_target / max(1.0 - wc_target, 1e-3)) if wc_target > 0 else 0.0
        qw *= float(rng.lognormal(0.0, noise_frac))
        dWp = qw * days

        Np += dNp; Gp += dGp; Wp += dWp
        p_new = _solve_pressure(Np, Gp, Wp)
        if not np.isfinite(p_new):
            p_new = p

        # Second pass on deliverability.
        #
        # The rate was priced against the pressure at the START of the step
        # but the row reports the pressure at its END, and when the well is
        # deliverability-limited those two must be the same number or the
        # drawdown stops equalling q/PI exactly. Uncorrected, PI_obs comes out
        # as PI*(p_start - p_wf_min)/(p_end - p_wf_min) on every limited
        # month. The step is re-priced once against the solved pressure, which
        # is enough: the correction is second order in the pressure drop over
        # one month, and it is worth about 0.1 % on the recovered PI rather
        # than the 1.2 pp that a units error in the first test attributed to
        # it. Kept because it is right, not because it was load-bearing.
        if pi_stbd_per_psi is not None and pi_stbd_per_psi > 0:
            qo_cap = max(pi_now * max(p_new - p_wf_min, 1.0), 1.0)
            if qo > qo_cap * 1.000001:
                Np -= dNp; Gp -= dGp; Wp -= dWp
                scale = qo_cap / qo
                qo, dNp, dGp = qo_cap, dNp * scale, dGp * scale
                qw *= scale; dWp *= scale
                Np += dNp; Gp += dGp; Wp += dWp
                p2 = _solve_pressure(Np, Gp, Wp)
                if np.isfinite(p2):
                    p_new = p2
        p = p_new
        cum_dp += max(pvt.p_init - p, 0.0)

        # Flowing pressure. Where a productivity index is supplied, p_wf is
        # DERIVED from it - p_wf = p_res - q/PI - so the PI diagnostic has a
        # known answer to recover, the same discipline the material balance
        # test uses. The alternative, p_wf as a fixed fraction of p_res, is
        # not a neutral default: it makes the drawdown shrink in proportion
        # to the pressure while the rate holds up on plateau, so the computed
        # PI rises 61 % on a well with nothing wrong with it, and the
        # diagnostic can neither pass nor fail against it.
        if pi_stbd_per_psi is not None and pi_stbd_per_psi > 0:
            p_wf_i = max(p - qo / max(pi_now, 1e-9), p_wf_min)
        else:
            p_wf_i = p * p_wf_frac

        rows.append({
            "date": dates[i], "well": well,
            "days_on": round(days, 2),
            "q_oil": round(qo, 2),
            # Gas carries its own measurement noise: in the field it is
            # metered separately from the oil, often worse, so a GOR built
            # from the two is noisier than either. Sharing the oil's noise
            # term would have cancelled out of the ratio and made the GOR
            # diagnostic look better than it can be.
            "q_gas": round(dGp * float(rng.lognormal(0.0, 1.5 * noise_frac))
                           / max(days, 1e-9) / SCF_PER_MSCF, 3),  # Mscf/d
            "q_water": round(qw, 2),
            "p_wf": round(p_wf_i, 1),
            "p_res": (round(p, 1) if i % survey_every == 0 else np.nan),
        })
    return pd.DataFrame(rows)


# ==============================================================================
# SECTION 7 -- THE WHOLE WORKFLOW, AND THE REPORT
# ==============================================================================


@dataclass
class OilWellResult:
    """Everything produced for one oil well, in one object."""
    well: str
    data: OilProductionData
    pvt: OilPVT
    model_table: pd.DataFrame
    fits: Dict[str, FitResult]
    best_fit: FitResult
    gor_model: RatioModel
    wor_model: RatioModel
    forecast: OilForecast
    gor_diag: Optional[GORDiagnostic] = None
    water_diag: Optional[WaterCutDiagnostic] = None
    pi_diag: Optional[OilPIDiagnostic] = None
    matbal: Optional[MaterialBalanceOil] = None
    aquifer_match: Optional["AquiferMatch"] = None
    aquifer_used: bool = False
    aquifer_note: str = ""
    mc: Optional[pd.DataFrame] = None
    n_mc_requested: int = 0
    settings: Dict = field(default_factory=dict)
    # The same limits as `settings`, unformatted. The settings block is for a
    # reader and holds strings like "150 STB/d"; a chart needs the number, and
    # parsing it back out of the prose would be absurd.
    limits: Dict[str, float] = field(default_factory=dict)

    def _window_block(self) -> str:
        """How much the rate trend depends on where the window starts."""
        try:
            ws = window_sensitivity(self.data.t, self.data.q_oil,
                                    self.data.qc.fit_start_days)
        except Exception:
            return ""
        if ws is None or len(ws) < 2:
            return ""
        lines = ["", "  window check: the same oil-rate trend over other "
                     "windows"]
        for _, r in ws.iterrows():
            lines.append(f"    {str(r['window']):<34} {r['trend_pct_yr']:+7.1f}"
                         f" %/yr  (n={int(r['n']):>3}, R2 {r['r2']:.3f}, "
                         f"p {r['p_value']:.2g})")
        sig = ws[ws["p_value"] < 0.10]
        if len(sig) and (sig["trend_pct_yr"].max() > 0
                         > sig["trend_pct_yr"].min()):
            lines.append("    WARNING: the trend changes SIGN between windows "
                         "that are each significant.")
            lines.append("             The forecast is a consequence of where "
                         "the window starts, not of the well.")

        # A sign change was the only thing checked for. On one field the last
        # quarter of the record declined at -18.3 %/yr, significant at
        # p = 1e-10, against -3.5 %/yr over the fitted window - five times
        # steeper - and nothing was said. The fit had b pinned at 2 and a
        # terminal decline of 2.5 %/yr, reached the economic limit in 113
        # years, and ran the whole forecast to the horizon. The recent trend
        # is the best available evidence of the regime the well is in now;
        # when it disagrees with the curve being extrapolated by a factor,
        # that is the headline, not a line in a table.
        try:
            fw = ws[ws["window"].astype(str).str.startswith("fitted")]
            late = ws[ws["window"].astype(str).str.startswith(
                ("last quarter", "last half"))]
            if len(fw):
                f_tr = float(fw["trend_pct_yr"].iloc[0])
                for _, r in late.iterrows():
                    l_tr = float(r["trend_pct_yr"])
                    if (float(r["p_value"]) < 0.01 and f_tr < 0 and l_tr < 0
                            and abs(l_tr) > 2.0 * abs(f_tr)):
                        lines.append(
                            f"    WARNING: over the {r['window']} the rate "
                            f"falls {abs(l_tr):.1f} %/yr - "
                            f"{abs(l_tr) / abs(f_tr):.1f}x the "
                            f"{abs(f_tr):.1f} %/yr of the fitted window.")
                        lines.append(
                            "             The well is declining faster now "
                            "than the curve being forecast. Refit from later "
                            "in the\n             record, or treat the "
                            "EUR below as an upper bound.")
                        break
                    if (float(r["p_value"]) < 0.01 and f_tr < 0 and l_tr < 0
                            and abs(l_tr) < 0.5 * abs(f_tr)):
                        lines.append(
                            f"    NOTE: over the {r['window']} the decline has "
                            f"slowed to {abs(l_tr):.1f} %/yr, under half the "
                            f"fitted {abs(f_tr):.1f} %/yr.")
                        break
        except Exception:
            pass
        return "\n".join(lines)

    def report(self) -> str:
        """The whole analysis as text, in the order a reader needs it."""
        w = self.well
        rule = "=" * 78
        out = [rule, f"  OIL DECLINE-CURVE ANALYSIS -- {w}", rule, ""]

        out.append("DATA AND QC")
        out.append(self.data.qc.summary())
        out.append("")

        out.append("PVT")
        out.append(self.pvt.summary())
        out.append("")

        out.append("DIAGNOSTICS")
        for d in (self.gor_diag, self.water_diag, self.pi_diag):
            if d is not None:
                out.append(d.summary())
        out.append("")

        if self.matbal is not None:
            out.append("MATERIAL BALANCE (HAVLENA-ODEH)")
            out.append(self.matbal.summary())
            out.append("")
        if self.aquifer_match is not None:
            out.append(self.aquifer_match.summary())
            if self.aquifer_note:
                out.append(("  USED FOR FORECAST : " if self.aquifer_used
                            else "  NOT USED          : ") + self.aquifer_note)
            out.append("")

        out.append("DECLINE FIT")
        out.append(self.model_table.to_string(index=False))
        out.append("")
        out.append(f"  selected          : {self.best_fit.model_name}")
        out.append("  " + _fit_summary_text(self.best_fit)
                   .replace("\n", "\n  ").strip())
        wk = list(getattr(self.best_fit, "weak_params", []) or [])
        if wk:
            out.append(
                f"  AT A BOUND        : {', '.join(wk)}. A parameter at its "
                "bound is not a fitted value;\n                      the "
                "forecast below inherits whatever the bound was set to.")
        out.append(self._window_block())
        out.append("")

        out.append("PRODUCING RATIOS")
        out.append(self.gor_model.summary())
        out.append(self.wor_model.summary())
        out.append("")

        out.append("FORECAST")
        out.append(self.forecast.summary())
        out.append("")

        if self.mc is not None and len(self.mc):
            out.append("UNCERTAINTY")
            out.append(summarise_oil_mc(self.mc, self.forecast,
                                        self.n_mc_requested))
            out.append("")

        # The settings block goes at the END and is complete, because every
        # number above is conditional on it. A report that cannot be
        # reproduced from its own footer is not a record of anything.
        out.append("SETTINGS USED")
        for k in sorted(self.settings):
            out.append(f"  {k:<24}: {self.settings[k]}")
        out.append(rule)
        return "\n".join(out)


def analyse_oil_well(df: "pd.DataFrame | OilProductionData",
                     pvt: OilPVT,
                     well: str = "WELL",
                     q_econ_stbd: float = 20.0,
                     models: Sequence[str] = ("arps", "modified_hyperbolic",
                                              "ple", "sepd"),
                     select: str = "modified_hyperbolic",
                     fit_from_bdf: bool = True,
                     fit_window_days: Optional[Tuple[float, float]] = None,
                     fixed_params: Optional[Dict[str, float]] = None,
                     t_max_years: float = 30.0,
                     use_material_balance: bool = True,
                     mb_p_initial: Optional[float] = None,
                     mb_fit_gas_cap: bool = False,
                     mb_m_gas_cap: Optional[float] = None,
                     mb_skip_early: int = 0,
                     mb_aquifer: str = "none",
                     aq_porosity: Optional[float] = None,
                     aq_ro_ft: Optional[float] = None,
                     aq_mu_w_cp: float = 0.5,
                     apply_ooip_cap: bool = True,
                     water_cut_econ: Optional[float] = None,
                     q_water_econ_stbd: Optional[float] = None,
                     p_abandon_psia: Optional[float] = None,
                     run_monte_carlo: bool = True,
                     n_mc: int = 800,
                     b_prior: Optional[Tuple[float, float]] = None,
                     sample_model_form: bool = True,
                     prepare_kwargs: Optional[Dict] = None,
                     seed: int = 11) -> OilWellResult:
    """Run the whole oil workflow on one well.

    In the order the steps must happen:
      1. QC, and cumulatives that survive it
      2. Diagnostics - GOR, Chan water, productivity index
      3. Material balance for an independent N, used as a cap
      4. Fit and rank decline models on the OIL rate
      5. GOR and WOR models against cumulative oil
      6. Deterministic forecast under every constraint
      7. Monte Carlo

    The diagnostics run BEFORE the forecast because the GOR model needs to
    know whether the GOR has peaked, and fitting across a peak forecasts a
    GOR that does neither what the rising limb nor the falling limb does.
    """
    data = (df if isinstance(df, OilProductionData)
            else OilProductionData.prepare(df, pvt, well=well,
                                           **(prepare_kwargs or {})))
    well = data.well

    # -- diagnostics ------------------------------------------------------
    gor_diag = water_diag = pi_diag = None
    try:
        gor_diag = gor_diagnostic(data.gor, data.Np, pvt.rsi,
                                  saturated=bool(pvt.saturated_at_discovery))
    except Exception as exc:
        warnings.warn(f"[{well}] GOR diagnostic failed: {exc}")
    try:
        water_diag = chan_diagnostic(data.t, data.q_oil, data.q_water)
    except Exception as exc:
        warnings.warn(f"[{well}] water diagnostic failed: {exc}")

    # -- material balance -------------------------------------------------
    matbal = None
    n_cap = n_hard_max = None
    m_used = float(mb_m_gas_cap or 0.0)
    we_to_date = 0.0
    if use_material_balance:
        sv = data.surveys
        if len(sv) >= 3:
            try:
                matbal = material_balance_oil(
                    sv["p_res"].to_numpy(float), sv["Np_stb"].to_numpy(float),
                    sv["Gp_scf"].to_numpy(float),
                    (sv["Wp_stb"].to_numpy(float) if "Wp_stb" in sv else None),
                    pvt, m_gas_cap=mb_m_gas_cap, fit_gas_cap=mb_fit_gas_cap,
                    p_initial=mb_p_initial, skip_early=mb_skip_early)
            except Exception as exc:
                warnings.warn(f"[{well}] material balance failed: {exc}")
    n_press = None
    if matbal is not None and matbal.trend_ok and np.isfinite(matbal.n_ooip_stb):
        m_used = float(matbal.m_gas_cap)
        # With the oil-in-place cap off, the fitted N (at the m it was fitted
        # with) still turns production into pressure for the abandonment
        # limit and the implied end pressure.
        if not apply_ooip_cap and matbal.n_ooip_stb > 0:
            n_press = float(matbal.n_ooip_stb)
        we_to_date = (float(matbal.we_implied_rb)
                      if np.isfinite(matbal.we_implied_rb) else 0.0)
        if apply_ooip_cap:
            n_cap = float(matbal.n_ooip_stb)
            # Where the balance itself says a gas cap is indicated, the N it
            # just reported is known to be wrong - it has absorbed the gas
            # cap's expansion - so capping the forecast with it caps nothing.
            # The N at the indicated m is used instead, and the report says
            # which number the cap came from. Substituting silently would be
            # no better than the error it is fixing.
            if (matbal.gas_cap_indicated
                    and np.isfinite(matbal.n_at_indicated_m_stb)
                    and matbal.n_at_indicated_m_stb > 0):
                n_cap = float(matbal.n_at_indicated_m_stb)
                m_used = float(matbal.m_indicated)
                warnings.warn(
                    f"[{well}] a gas cap is indicated (m ~ "
                    f"{matbal.m_indicated:.2f}); the forecast cap uses the N "
                    f"at that m, {n_cap / 1e6:,.1f} MMstb, not the "
                    f"{matbal.n_ooip_stb / 1e6:,.1f} MMstb fitted with m = 0. "
                    "Set m from structure and logs to remove the guess.")
            # min(F/Et) is an upper BOUND on N, not an estimate with symmetric
            # error. The Monte Carlo held every realisation under it while the
            # deterministic case was allowed to use a fitted N that sat above
            # it - only 0.6 % above on the well that showed this up, but
            # enough that the base case fell outside its own P90-P10 and the
            # report had to refuse its own percentiles. A bound the report
            # calls a bound applies to the base case too.
            if np.isfinite(matbal.n_ceiling_stb) and not (
                    matbal.gas_cap_indicated
                    and np.isfinite(matbal.n_at_indicated_m_stb)):
                n_hard_max = float(matbal.n_ceiling_stb)
                if n_cap > n_hard_max:
                    warnings.warn(
                        f"[{well}] the fitted N of {n_cap / 1e6:,.2f} MMstb "
                        f"exceeds the We>=0 ceiling min(F/Et) = "
                        f"{n_hard_max / 1e6:,.2f} MMstb; the ceiling is used "
                        "as the cap, because influx can only add to the "
                        "withdrawal.")
                    n_cap = n_hard_max

    # -- aquifer history match -------------------------------------------
    # Run only when asked for. Its N replaces the Havlena-Odeh one for the
    # forecast only when the surveys bound it from above and it respects the
    # We >= 0 ceiling; otherwise it is reported and left alone, with the
    # reason.
    aq_match = None
    aq_used = False
    aq_note = ""
    aq_sigma = 0.15
    kind_aq = str(mb_aquifer or "none").lower()
    if use_material_balance and kind_aq != "none":
        sv = data.surveys
        src = data.full_df if data.full_df is not None else data.df
        n_guess = None
        if matbal is not None:
            for v in (matbal.n_ooip_stb, matbal.n_ceiling_stb):
                if np.isfinite(v) and v > 0:
                    n_guess = float(v)
                    break
        try:
            aq_match = aquifer_history_match(
                src["t"].to_numpy(float), src["Np_stb"].to_numpy(float),
                src["Gp_scf"].to_numpy(float),
                (src["Wp_stb"].to_numpy(float) if "Wp_stb" in src else None),
                sv["t"].to_numpy(float) if len(sv) else np.array([]),
                sv["p_res"].to_numpy(float) if len(sv) else np.array([]),
                pvt, kind=kind_aq, m_gas_cap=float(mb_m_gas_cap or 0.0),
                fit_m=bool(mb_fit_gas_cap), p_initial=mb_p_initial,
                n_guess_stb=n_guess, aq_porosity=aq_porosity,
                aq_ro_ft=aq_ro_ft, aq_mu_w_cp=aq_mu_w_cp)
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
            warnings.warn(f"[{well}] aquifer history match failed: {exc}")
        if aq_match is not None and matbal is not None:
            aq_match.n_ho_stb = float(matbal.n_ooip_stb)
        if aq_match is not None and not aq_match.ran:
            aq_note = "the match did not run - see above."
        elif aq_match is not None and aq_match.accepted:
            why = []
            lo_r, hi_r = aq_match.n_range_stb
            if aq_match.n_range_open[1]:
                why.append("the surveys do not bound N from above")
            elif lo_r > 0 and hi_r / lo_r > AQ_MAX_RANGE_RATIO:
                why.append(f"the 95 % range on N spans a factor of "
                           f"{hi_r / lo_r:,.1f}")
            if (np.isfinite(aq_match.n_ceiling_stb)
                    and aq_match.n_stb > 1.02 * aq_match.n_ceiling_stb):
                why.append("the matched N breaks the We >= 0 ceiling")
            if why:
                aq_note = ("; ".join(why) + ", so the forecast keeps the "
                           "Havlena-Odeh values.")
            else:
                aq_used = True
                lo, hi = aq_match.n_range_stb
                if lo > 0 and hi > lo:
                    aq_sigma = float(min(max(
                        (math.log(hi) - math.log(lo)) / (2.0 * 1.96), 0.02),
                        1.0))
                same_m = abs(aq_match.m - m_used) < 1e-9
                m_used = float(aq_match.m)
                we_to_date = float(aq_match.we_to_date_rb)
                if not same_m:
                    n_hard_max = None      # the ceiling was taken at another m
                if apply_ooip_cap:
                    n_cap = float(aq_match.n_stb)
                    if n_hard_max is not None and n_cap > n_hard_max:
                        n_cap = float(n_hard_max)
                    n_press = None
                else:
                    n_cap = None
                    n_press = float(aq_match.n_stb)
                aq_note = (f"N {aq_match.n_stb / 1e6:,.2f} MMstb, m "
                           f"{aq_match.m:.3f} and We to date "
                           f"{aq_match.we_to_date_rb / 1e6:,.2f} MMrb\n"
                           "                      replace the Havlena-Odeh "
                           "values; the Monte Carlo spreads N over the 95 %\n"
                           "                      range. The forecast "
                           "pressure still freezes the aquifer at its influx "
                           "to date.")

    # PI needs oil in place for its fallback p_avg, so it runs after the
    # balance; it prefers measured surveys and says which it used.
    try:
        ft = data.flowing_tests
        if len(ft):
            pi_diag = oil_pi_diagnostic(
                ft["t"].to_numpy(float), ft["q_oil"].to_numpy(float),
                ft["p_wf"].to_numpy(float),
                (ft["p_res"].to_numpy(float) if "p_res" in ft else None),
                pvt, np_stb=ft["Np_stb"].to_numpy(float),
                n_ooip_stb=n_cap)
        else:
            pi_diag = OilPIDiagnostic(
                ok=False, reason="no rows carry a flowing pressure")
    except Exception as exc:
        warnings.warn(f"[{well}] PI diagnostic failed: {exc}")

    # -- fitting window ---------------------------------------------------
    if fit_window_days is not None:
        t_lo, t_hi = fit_window_days
    elif fit_from_bdf and data.qc.fit_start_days is not None:
        t_lo, t_hi = data.qc.fit_start_days, None
    else:
        t_lo, t_hi = None, None
    t_w, q_w = data.window(t_lo, t_hi)
    if len(t_w) < 8:
        warnings.warn(f"[{well}] only {len(t_w)} points in the fitting "
                      "window; fitting the full history instead.")
        t_w, q_w, t_lo = data.t, data.q_oil, None

    table, fits = rank_models(t_w, q_w, models=models,
                              fixed=(fixed_params or None))
    if not fits:
        raise RuntimeError(f"[{well}] no decline model could be fitted.")
    if select == "auto":
        best_name = str(table.iloc[0]["model"])
    elif select in fits:
        best_name = select
    else:
        best_name = str(table.iloc[0]["model"])
        warnings.warn(f"[{well}] '{select}' did not fit; using {best_name}.")
    best = fits[best_name]

    # -- producing ratios --------------------------------------------------
    gor_model, gor_shapes = fit_gor_model(data.Np, data.gor, pvt,
                                          diag=gor_diag)
    wor_model, wor_shapes = fit_wor_model(data.Np, data.q_oil, data.q_water)

    # -- forecast ----------------------------------------------------------
    # The well's own demonstrated liquid rate, with headroom for a workover.
    liq = data.q_oil + data.q_water
    liq_cap = (float(np.nanmax(liq)) * 1.5
               if np.isfinite(liq).any() and float(np.nanmax(liq)) > 0
               else None)

    fkw = dict(q_liquid_cap_stbd=liq_cap,
               np_to_date_stb=float(data.Np[-1]),
               gp_to_date_scf=float(data.Gp[-1]),
               wp_to_date_stb=float(data.Wp[-1]),
               t_start_days=float(data.t[-1]), t_max_years=t_max_years,
               n_ooip_stb=n_cap, m_gas_cap=m_used, we_to_date_rb=we_to_date,
               n_pressure_stb=n_press,
               water_cut_econ=water_cut_econ,
               q_water_econ_stbd=q_water_econ_stbd,
               p_abandon_psia=p_abandon_psia)
    forecast = forecast_oil(best, gor_model, wor_model, pvt, q_econ_stbd,
                            **fkw)

    mc = None
    if run_monte_carlo:
        try:
            mc = monte_carlo_oil_eur(
                best, gor_model, wor_model, pvt, q_econ_stbd,
                n_samples=n_mc, b_prior=b_prior, seed=seed,
                fits=(fits if sample_model_form else None),
                gor_shapes=(gor_shapes if sample_model_form else None),
                wor_shapes=(wor_shapes if sample_model_form else None),
                n_ooip_hard_max=n_hard_max, n_ooip_rel_sigma=aq_sigma,
                **fkw)
        except Exception as exc:
            warnings.warn(f"[{well}] Monte Carlo failed: {exc}")

    settings = {
        "rate basis": data.rate_basis,
        "fit window": ("full history" if t_lo is None
                       else f"from day {t_lo:,.0f}"),
        "models tried": ", ".join(models),
        "model selected": f"{best_name} (select='{select}')",
        "fixed parameters": (fixed_params or "none"),
        "economic oil rate": f"{q_econ_stbd:,.0f} STB/d",
        "water cut limit": (f"{100 * water_cut_econ:.0f} %"
                            if water_cut_econ else "none"),
        "water rate limit": (f"{q_water_econ_stbd:,.0f} STB/d"
                             if q_water_econ_stbd else "none"),
        "abandonment pressure": (f"{p_abandon_psia:,.0f} psia"
                                 + (" (NOT APPLIED - see FORECAST)"
                                    if "abandonment pressure"
                                    in forecast.constraints_not_applied
                                    else "")
                                 if p_abandon_psia else "none"),
        "forecast horizon": f"{t_max_years:,.0f} yr from the last record",
        "material balance": ("on" if use_material_balance else "off"),
        "aquifer model": (kind_aq if kind_aq == "none" else
                          kind_aq + (" - matched N used for the forecast"
                                     if aq_used else " - match NOT used")),
        "gas cap m": (f"{m_used:.3f} "
                      + (("(fitted)" if (matbal is not None
                                         and matbal.m_fitted)
                          else "(fit requested, not determined - "
                               "value used in its place)")
                         if mb_fit_gas_cap else "(supplied)")),
        "oil-in-place cap": (f"{n_cap / 1e6:,.1f} MMstb"
                             if n_cap else "not applied"),
        "N hard ceiling": (f"{n_hard_max / 1e6:,.1f} MMstb = min(F/Et)"
                           if n_hard_max else "none"),
        "Monte Carlo": (f"{n_mc:,} draws, seed {seed}"
                        + (", model form sampled" if sample_model_form
                           else ", one model only")
                        if run_monte_carlo else "off"),
    }

    limits = {
        "q_econ_stbd": float(q_econ_stbd),
        "water_cut_econ": (float(water_cut_econ) if water_cut_econ else
                           float("nan")),
        "q_water_econ_stbd": (float(q_water_econ_stbd) if q_water_econ_stbd
                              else float("nan")),
        "p_abandon_psia": (float(p_abandon_psia) if p_abandon_psia
                           else float("nan")),
        "n_ooip_stb": (float(n_cap) if n_cap else float("nan")),
        "n_ceiling_stb": (float(n_hard_max) if n_hard_max else float("nan")),
        "fit_start_days": (float(t_lo) if t_lo else float("nan")),
        "t_max_years": float(t_max_years),
    }

    return OilWellResult(
        limits=limits,
        well=well, data=data, pvt=pvt, model_table=table, fits=fits,
        best_fit=best, gor_model=gor_model, wor_model=wor_model,
        forecast=forecast, gor_diag=gor_diag, water_diag=water_diag,
        pi_diag=pi_diag, matbal=matbal, mc=mc,
        aquifer_match=aq_match, aquifer_used=aq_used, aquifer_note=aq_note,
        n_mc_requested=(n_mc if run_monte_carlo else 0), settings=settings)


# ==============================================================================
# SECTION 8 -- SELF-TESTS
# ==============================================================================


def _demo_pvt() -> "OilPVT":
    return OilPVT(api=34.0, gas_gravity=0.75, temperature_F=190.0,
                  rsi=600.0, p_init=3800.0, sw_initial=0.22)


def _march_tank(n_stb: float, m: float, we_total_rb: float = 0.0,
                p_final: float = 1200.0, n: int = 250,
                free_gas_mult: float = 3.0) -> Dict[str, np.ndarray]:
    """Step the PRESSURE down and solve the production the balance demands.

    This inverts exactly what `material_balance_oil` has to undo, so a failure
    to recover N is the fit's and not the generator's - the same discipline
    the gas module uses, and the reason the recovery tests below are worth
    anything.
    """
    pvt = _demo_pvt()
    boi, bgi, bti, swc = pvt.boi, pvt.bgi, pvt.bti, pvt.sw_initial
    ps = np.linspace(pvt.p_init, p_final, n + 1)[1:]
    cum_dp = np.cumsum(pvt.p_init - ps)
    we = we_total_rb * cum_dp / cum_dp[-1]
    Np = Gp = 0.0
    out_p, out_np, out_gp = [], [], []
    for i, pp in enumerate(ps):
        bo = float(pvt.bo(np.array([pp]))[0])
        rs = float(pvt.rs(np.array([pp]))[0])
        bg = float(pvt.bg(np.array([pp]))[0])
        bt = float(pvt.bt(np.array([pp]))[0])
        Eo = bt - bti
        Eg = boi * (bg / bgi - 1.0)
        Efw = ((1 + m) * boi * ((pvt.cw_per_psi * swc + pvt.cf_per_psi)
                                / (1 - swc)) * (pvt.p_init - pp))
        F_req = n_stb * (Eo + m * Eg + Efw) + we[i]
        d = max(pvt.p_bubble - pp, 0.0) / max(pvt.p_bubble, 1.0)
        gor = rs * (1.0 + free_gas_mult * d)
        a = bo + (gor - rs) * bg
        b = Np * bo + (Gp - Np * rs) * bg
        dNp = (F_req - b) / max(a, 1e-12)
        if dNp <= 0:
            continue
        Np += dNp
        Gp += gor * dNp
        out_p.append(pp); out_np.append(Np); out_gp.append(Gp)
    return {"p": np.array(out_p), "Np": np.array(out_np),
            "Gp": np.array(out_gp), "pvt": pvt}


def _march_fetkovich(pvt: OilPVT, n_stb: float, m: float, wei_rb: float,
                     tau_days: float, q_stbd: float = 4000.0,
                     months: int = 120, gor_mult: float = 2.0
                     ) -> Dict[str, np.ndarray]:
    """A tank with a Fetkovich aquifer, marched month by month.

    Deliberately written the slow, obvious way - scalar PVT calls and a
    bisection per month - and sharing no code with `_simulate_tank`, so that
    agreement between the two is evidence rather than a tautology.
    """
    pi = pvt.p_init
    boi, bgi, bti, swc = pvt.boi, pvt.bgi, pvt.bti, pvt.sw_initial
    ce = (pvt.cw_per_psi * swc + pvt.cf_per_psi) / (1.0 - swc)
    dt = 30.4375
    Np = Gp = We = 0.0
    p = pi
    T, P, NP, GP, WE = [0.0], [pi], [0.0], [0.0], [0.0]
    for k in range(1, months + 1):
        rs = float(pvt.rs(np.array([p]))[0])
        d = max(pvt.p_bubble - p, 0.0) / pvt.p_bubble
        Np += q_stbd * dt
        Gp += q_stbd * dt * rs * (1.0 + gor_mult * d)
        pa = pi * (1.0 - We / wei_rb)
        fac = 1.0 - math.exp(-dt / tau_days)

        def we_of(pp, We=We, pa=pa, fac=fac, p=p):
            return We + (wei_rb / pi) * fac * (pa - 0.5 * (p + pp))

        def resid(pp, Np=Np, Gp=Gp, we_of=we_of):
            a = np.array([pp])
            bo, rsx = float(pvt.bo(a)[0]), float(pvt.rs(a)[0])
            bg, bt = float(pvt.bg(a)[0]), float(pvt.bt(a)[0])
            F = Np * (bo + (Gp / Np - rsx) * bg)
            Et = ((bt - bti) + m * boi * (bg / bgi - 1.0)
                  + (1.0 + m) * boi * ce * (pi - pp))
            return F - n_stb * Et - we_of(pp)
        lo, hi = 60.0, pi
        if resid(hi) < 0:
            pn = pi
        else:
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                if resid(mid) > 0:
                    hi = mid
                else:
                    lo = mid
            pn = 0.5 * (lo + hi)
        We = we_of(pn)
        p = pn
        T.append(k * dt); P.append(p); NP.append(Np); GP.append(Gp)
        WE.append(We)
    return {"t": np.array(T), "p": np.array(P), "Np": np.array(NP),
            "Gp": np.array(GP), "Wp": np.zeros(len(T)), "We": np.array(WE)}


def run_self_tests(verbose: bool = True) -> bool:
    """Internal consistency and parameter-recovery checks for the oil module.

    These do not replace benchmarking against your own field data. They catch
    the failures that silently corrupt an oil DCA: a PVT that jumps at the
    bubble point, a balance that cannot recover a known N, a diagnostic that
    reads a mechanism off scatter, a forecast whose cumulative does not
    integrate its own rate, and a Monte Carlo that ignores a constraint the
    deterministic case obeys.
    """
    results: List[Tuple[str, bool, str]] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        results.append((name, bool(cond), detail))

    pvt = _demo_pvt()

    # -- 1. PVT ------------------------------------------------------------
    pb = pvt.p_bubble
    check("bubble point is inside the pressure range",
          500.0 < pb < pvt.p_init, f"pb={pb:,.0f} psia")

    p_hi = np.linspace(pb + 50.0, pvt.p_init, 40)
    rs_hi = pvt.rs(p_hi)
    check("Rs is pinned at Rsi above the bubble point",
          float(np.max(np.abs(rs_hi - pvt.rsi))) < 1e-6,
          f"max dev {float(np.max(np.abs(rs_hi - pvt.rsi))):.2e}")

    p_lo = np.linspace(200.0, pb - 1.0, 60)
    check("Rs increases with pressure below the bubble point",
          bool(np.all(np.diff(pvt.rs(p_lo)) > 0)))

    eps = 0.5
    bo_lo = float(pvt.bo(np.array([pb - eps]))[0])
    bo_hi = float(pvt.bo(np.array([pb + eps]))[0])
    check("Bo is continuous at the bubble point",
          abs(bo_hi - bo_lo) / bo_lo < 1e-3,
          f"{bo_lo:.5f} vs {bo_hi:.5f}")
    check("Bo peaks at the bubble point",
          bo_lo > float(pvt.bo(np.array([0.5 * pb]))[0])
          and bo_hi > float(pvt.bo(np.array([pvt.p_init]))[0]))

    p_all = np.linspace(200.0, pvt.p_init, 120)
    check("Bt never falls as pressure falls",
          bool(np.all(np.diff(pvt.bt(p_all)) <= 1e-9)))
    check("Bg falls as pressure rises",
          bool(np.all(np.diff(pvt.bg(p_all)) < 0)))
    check("Bt equals Bo above the bubble point",
          float(np.max(np.abs(pvt.bt(p_hi) - pvt.bo(p_hi)))) < 1e-9)
    # Below the bubble point, falling pressure means gas leaving solution and
    # heavier oil, so viscosity RISES as pressure falls. Above it, no more gas
    # can dissolve and compression takes over, so viscosity rises with
    # pressure instead. The minimum sits at the bubble point. Asserting one
    # direction over the whole range - which the first version of this test
    # did - is asserting that half the curve is wrong.
    check("oil viscosity rises as pressure FALLS below the bubble point",
          bool(np.all(np.diff(pvt.muo(p_lo)) < 0)))
    check("oil viscosity rises as pressure RISES above the bubble point",
          bool(np.all(np.diff(pvt.muo(p_hi)) > 0)))
    check("oil viscosity is continuous at the bubble point",
          abs(float(pvt.muo(np.array([pb + eps]))[0])
              - float(pvt.muo(np.array([pb - eps]))[0]))
          / float(pvt.muo(np.array([pb]))[0]) < 1e-3)
    check("oil viscosity is at its minimum at the bubble point",
          float(pvt.muo(np.array([pb]))[0])
          <= float(np.min(np.concatenate([pvt.muo(p_lo), pvt.muo(p_hi)])))
          + 1e-9)

    # -- 1b. other fluids --------------------------------------------------
    #
    # Two different questions, which must not be run together. Whether the
    # INVERSION works on a volatile or a heavy oil is testable here, because
    # the tank is marched with the same PVT the balance inverts. Whether the
    # CORRELATIONS are trustworthy for those fluids is not testable here at
    # all, and no synthetic can make it so - the synthetic inherits them.
    # That question is answered by the published-range check below.
    OTHER_FLUIDS = [
        ("light, high GOR", dict(api=40.0, gas_gravity=0.80,
                                 temperature_F=220.0, rsi=1200.0,
                                 p_init=5200.0), False),
        ("volatile", dict(api=45.0, gas_gravity=0.85, temperature_F=250.0,
                          rsi=2000.0, p_init=6500.0), True),
        ("near-critical", dict(api=48.0, gas_gravity=0.90,
                               temperature_F=265.0, rsi=2600.0,
                               p_init=7200.0), True),
        ("heavy", dict(api=14.0, gas_gravity=0.65, temperature_F=120.0,
                       rsi=120.0, p_init=1500.0), True),
        ("extra-heavy", dict(api=10.0, gas_gravity=0.62, temperature_F=105.0,
                             rsi=60.0, p_init=900.0), True),
    ]
    check("the base fluid is inside every correlation's published range",
          not correlation_range_warnings(pvt),
          f"{len(correlation_range_warnings(pvt))} warning(s)")
    check("the base fluid model is physical",
          not pvt_physicality_warnings(pvt),
          "; ".join(pvt_physicality_warnings(pvt))[:80])

    for tag_f, kw_f, expect_oor in OTHER_FLUIDS:
        pvt_f = OilPVT(sw_initial=0.25, **kw_f)
        pb_f = float(pvt_f.p_bubble)
        check(f"a bubble point is found for a {tag_f} fluid",
              50.0 < pb_f < pvt_f.p_init, f"pb {pb_f:,.0f} psia")
        check(f"Rs is pinned at Rsi above the bubble point, {tag_f}",
              float(np.max(np.abs(pvt_f.rs(np.linspace(pb_f + 25.0,
                                                       pvt_f.p_init, 30))
                                  - pvt_f.rsi))) < 1e-6)
        p_lo_f = np.linspace(max(0.05 * pb_f, 40.0), pb_f - 1.0, 40)
        check(f"viscosity rises as pressure falls below pb, {tag_f}",
              bool(np.all(np.diff(pvt_f.muo(p_lo_f)) < 0)),
              f"muo at pb {float(pvt_f.muo(np.array([pb_f]))[0]):,.2f} cp")
        check(f"being outside the published range is reported, {tag_f}",
              bool(correlation_range_warnings(pvt_f)) == expect_oor,
              "; ".join(correlation_range_warnings(pvt_f))[:70] or "in range")

        # The balance must still invert exactly, whatever the fluid.
        N_f = 50.0e6
        ps_f = np.linspace(pvt_f.p_init, max(0.30 * pvt_f.p_init, 120.0),
                           201)[1:]
        Np_f = Gp_f = 0.0
        P_f, NP_f, GP_f = [], [], []
        for pp_f in ps_f:
            bo_f = float(pvt_f.bo(np.array([pp_f]))[0])
            rs_f = float(pvt_f.rs(np.array([pp_f]))[0])
            bg_f = float(pvt_f.bg(np.array([pp_f]))[0])
            bt_f = float(pvt_f.bt(np.array([pp_f]))[0])
            efw_f = (pvt_f.boi * ((pvt_f.cw_per_psi * pvt_f.sw_initial
                                   + pvt_f.cf_per_psi)
                                  / (1 - pvt_f.sw_initial))
                     * (pvt_f.p_init - pp_f))
            f_req = N_f * ((bt_f - pvt_f.bti) + efw_f)
            d_f = max(pvt_f.p_bubble - pp_f, 0.0) / max(pvt_f.p_bubble, 1.0)
            gor_f = rs_f * (1.0 + 3.0 * d_f)
            a_f = bo_f + (gor_f - rs_f) * bg_f
            b_f = Np_f * bo_f + (Gp_f - Np_f * rs_f) * bg_f
            dnp_f = (f_req - b_f) / max(a_f, 1e-12)
            if dnp_f <= 0:
                continue
            Np_f += dnp_f
            Gp_f += gor_f * dnp_f
            P_f.append(pp_f); NP_f.append(Np_f); GP_f.append(Gp_f)
        if len(P_f) >= 5:
            mb_f = material_balance_oil(np.array(P_f), np.array(NP_f),
                                        np.array(GP_f), None, pvt_f,
                                        p_initial=pvt_f.p_init)
            err_f = abs(mb_f.n_ooip_stb - N_f) / N_f
            check(f"N is recovered on a {tag_f} fluid",
                  err_f < 0.01,
                  f"{mb_f.n_ooip_stb / 1e6:,.2f} MMstb ({100 * err_f:+.2f} %)")

    # A near-critical fluid is where Standing breaks: Bt came back
    # non-monotone over a 580 psi window below the bubble point. The dip is
    # far too small to move a forecast and is reported anyway, because it is
    # the fluid model saying it has left the region where it means anything.
    pvt_nc = OilPVT(api=48.0, gas_gravity=0.90, temperature_F=265.0,
                    rsi=2600.0, p_init=7200.0, sw_initial=0.25)
    check("an unphysical Bt is detected rather than computed with",
          any("Bt is NOT monotone" in w
              for w in pvt_physicality_warnings(pvt_nc)),
          "; ".join(pvt_physicality_warnings(pvt_nc))[:80] or "none found")
    check("the unphysical region is reported in the PVT summary",
          "NOT PHYSICAL" in pvt_nc.summary())
    check("a heavy oil is classified as one", fluid_class(OilPVT(
        api=14.0, gas_gravity=0.65, temperature_F=120.0, rsi=120.0,
        p_init=1500.0)) == "heavy / low-GOR oil")
    check("a volatile oil is classified as one", fluid_class(pvt_nc)
          == "volatile oil")
    check("the base fluid is classified as a black oil",
          fluid_class(pvt) == "black oil")

    # -- 1c. a declared bubble point that disagrees with Rsi ---------------
    #
    # A field report declared pb = 3,493 psia on a fluid whose Rsi of 391
    # scf/STB puts Standing's bubble point at 1,673. Rs was then pinned at Rsi
    # for 1,820 psi below the declared bubble point, the oil expansion term was
    # identically zero across that range, and the material balance returned a
    # ceiling of a billion barrels.
    pvt_d = OilPVT(api=35.0, gas_gravity=0.70, temperature_F=140.0,
                   rsi=391.0, p_init=3493.0, p_bubble=3493.0, sw_initial=0.22)
    rs_below = pvt_d.rs(np.array([3200.0, 2400.0]))
    check("gas comes out of solution below a DECLARED bubble point",
          bool(rs_below[0] < pvt_d.rsi - 1.0 and rs_below[1] < rs_below[0]),
          f"Rs {rs_below[0]:,.0f} and {rs_below[1]:,.0f} scf/STB below pb")
    check("Rs is continuous at a declared bubble point",
          abs(float(pvt_d.rs(np.array([3493.0 - 0.5]))[0]) - pvt_d.rsi) < 1.0)
    eo_below = float(pvt_d.bt(np.array([3000.0]))[0]) - pvt_d.bti
    check("the oil expansion term is not zero below a declared pb",
          eo_below > 1e-3, f"Eo at 3,000 psia = {eo_below:.5f} rb/STB")
    check("a bubble point far from the correlation's is flagged",
          "CHECK Rsi AND pb" in pvt_d.summary())
    check("the base fluid's Rs is unchanged by the anchoring",
          float(np.max(np.abs(pvt.rs(np.linspace(300.0, pb - 5.0, 40))
                              - np.minimum(standing_rs(
                                  np.linspace(300.0, pb - 5.0, 40), pvt.api,
                                  pvt.gas_gravity, pvt.temperature_F),
                                  pvt.rsi)))) < 1e-6)
    # And the physicality check must catch a flat Rs below pb from ANY source,
    # e.g. a user table - the correlation path is now anchored, so a table is
    # the only way left to produce it.
    flat_tab = pd.DataFrame({"pressure": [500.0, 1500.0, 2500.0, 3493.0],
                             "rs": [391.0, 391.0, 391.0, 391.0]})
    pvt_flat = OilPVT(api=35.0, gas_gravity=0.70, temperature_F=140.0,
                      rsi=391.0, p_init=3493.0, p_bubble=3493.0,
                      sw_initial=0.22, pvt_table=flat_tab)
    check("an Rs that stays at Rsi below the bubble point is reported",
          any("BELOW the bubble" in w
              for w in pvt_physicality_warnings(pvt_flat)))

    # -- 2. material balance ----------------------------------------------
    for m_true in (0.0, 0.40):
        tk = _march_tank(50.0e6, m_true)
        mb = material_balance_oil(tk["p"], tk["Np"], tk["Gp"], None, pvt,
                                  m_gas_cap=m_true, p_initial=pvt.p_init)
        err = abs(mb.n_ooip_stb - 50.0e6) / 50.0e6
        check(f"N recovered on a marched tank, m={m_true:.2f}",
              err < 0.01, f"{mb.n_ooip_stb / 1e6:,.2f} MMstb, err {100 * err:.2f} %")
        check(f"drive read as volumetric, m={m_true:.2f}",
              mb.drive.startswith("volumetric"), mb.drive)

    tk = _march_tank(50.0e6, 0.0, we_total_rb=9.0e6)
    mb_w = material_balance_oil(tk["p"], tk["Np"], tk["Gp"], None, pvt,
                                m_gas_cap=0.0, p_initial=pvt.p_init)
    check("water drive is refused, not fitted away",
          not mb_w.drive.startswith("volumetric"), mb_w.drive)
    check("the We>=0 ceiling sits below the fitted N under influx",
          mb_w.n_ceiling_stb < mb_w.n_ooip_stb,
          f"ceiling {mb_w.n_ceiling_stb / 1e6:,.1f} vs fitted "
          f"{mb_w.n_ooip_stb / 1e6:,.1f} MMstb")
    check("the ceiling is nearer the truth than the fitted N is",
          abs(mb_w.n_ceiling_stb - 50e6) < abs(mb_w.n_ooip_stb - 50e6))

    for m_true in (0.0, 0.25, 0.60, 1.20):
        tk = _march_tank(50.0e6, m_true)
        mb_m = material_balance_oil(tk["p"], tk["Np"], tk["Gp"], None, pvt,
                                    fit_gas_cap=True, p_initial=pvt.p_init)
        check(f"gas cap m recovered blind, truth {m_true:.2f}",
              abs(mb_m.m_gas_cap - m_true) < 0.03,
              f"fitted {mb_m.m_gas_cap:.3f}")

    # A gas cap that is assumed away is absorbed into N, quietly: the fit
    # stays tight and the apparent-N sequence can stay flat while N comes out
    # an order of magnitude too big. So the balance tests its own assumption.
    for m_true, want in ((0.60, True), (0.25, True), (0.0, False)):
        tk_g = _march_tank(50.0e6, m_true)
        mb_g = material_balance_oil(tk_g["p"], tk_g["Np"], tk_g["Gp"], None,
                                    pvt, p_initial=pvt.p_init)
        check(f"a gas cap of m={m_true:.2f} is "
              + ("detected when m was assumed to be zero"
                 if want else "not invented when there is none"),
              mb_g.gas_cap_indicated == want,
              (f"indicated m {mb_g.m_indicated:.2f}, N "
               f"{mb_g.n_at_indicated_m_stb / 1e6:,.1f} vs "
               f"{mb_g.n_ooip_stb / 1e6:,.1f} MMstb at m=0" if want
               else f"flagged={mb_g.gas_cap_indicated}"))
        if want:
            check(f"the N indicated at m={m_true:.2f} is far nearer the truth "
                  "than the m=0 one",
                  abs(mb_g.n_at_indicated_m_stb - 50e6)
                  < 0.25 * abs(mb_g.n_ooip_stb - 50e6),
                  f"{mb_g.n_at_indicated_m_stb / 1e6:,.1f} vs "
                  f"{mb_g.n_ooip_stb / 1e6:,.1f} MMstb (truth 50.0)")

    tk = _march_tank(50.0e6, 0.0)
    mb_p = material_balance_oil(tk["p"], tk["Np"], tk["Gp"], None, pvt)
    check("an inferred p_i is not labelled as entered",
          "NOT an entered" in mb_p.p_initial_source, mb_p.p_initial_source)
    check("N's stated error is not tighter than the surveys disagree",
          not np.isfinite(mb_p.apparent_n_spread)
          or mb_p.n_stderr_stb >= 0.5 * (mb_p.apparent_n_max_stb
                                         - mb_p.apparent_n_min_stb) - 1.0,
          f"+/-{mb_p.n_stderr_stb / 1e3:,.0f} Mstb on a "
          f"{100 * mb_p.apparent_n_spread:.1f} % spread")

    # An N that could not be fitted must not be followed by a verdict about
    # it. The field report printed "the fitted N can be read as oil in place"
    # two lines under "OOIP (N): nan".
    mb_nan = material_balance_oil(np.array([3493.0, 3350.0, 3200.0]),
                                  np.array([0.0, 2.0e6, 4.5e6]),
                                  np.array([0.0, 9.0e8, 2.0e9]), None, pvt_d,
                                  m_gas_cap=0.358, p_initial=3493.0)
    check("an undetermined N gets no drive verdict",
          not np.isfinite(mb_nan.n_ooip_stb)
          and mb_nan.drive == "undetermined"
          and "can be read as oil" not in mb_nan.summary(),
          mb_nan.drive)
    check("an undetermined N is said in words, not printed as nan",
          "NOT DETERMINED" in mb_nan.summary()
          and " nan " not in mb_nan.summary())

    # The same record with the gas-cap fit ON. The field report printed
    # "m 0.0000 +/- 2.5000 (FITTED)" and an N-m anti-correlation note: with
    # under three usable surveys every m cost infinity, argmin fell on the
    # first grid point, and the whole grid counted as 'near' the minimum.
    mb_nf = material_balance_oil(np.array([3493.0, 3350.0, 3200.0]),
                                 np.array([0.0, 2.0e6, 4.5e6]),
                                 np.array([0.0, 9.0e8, 2.0e9]), None, pvt_d,
                                 fit_gas_cap=True, p_initial=3493.0)
    sm_nf = mb_nf.summary()
    check("an m that could not be fitted is not reported as FITTED",
          not mb_nf.m_fitted and "(FITTED)" not in sm_nf
          and "NOT DETERMINED - fit requested" in sm_nf, sm_nf.splitlines()[5]
          if len(sm_nf.splitlines()) > 5 else sm_nf)
    check("no N-m correlation note when neither was fitted",
          "anti-correlated" not in sm_nf and not np.isfinite(mb_nf.m_stderr))
    check("the depletion the surveys saw is stated",
          "depletion seen" in sm_nf and "psi below p_i" in sm_nf)
    mb_big = material_balance_oil(np.array([3493.0, 3470.0, 3450.0, 3430.0]),
                                  np.array([0.0, 3.0e6, 6.0e6, 9.0e6]),
                                  np.array([0.0, 1.2e9, 2.4e9, 3.6e9]), None,
                                  pvt_d, p_initial=3493.0)
    check("a ceiling far above the oil produced is called uninformative",
          mb_big.n_uninformative_ceiling
          and "constrains nothing" in mb_big.summary(),
          f"ceiling {mb_big.n_ceiling_stb / 1e6:,.0f} MMstb")
    check("a normal tank's ceiling is not called uninformative",
          not mb_w.n_uninformative_ceiling)
    rm_c = RatioModel(kind="constant", ln_r0=math.log(502.0), label="GOR",
                      slope_per_stb=0.0, np_ref_stb=0.0,
                      note="x " * 90)
    check("a held-constant ratio note is wrapped, not one long line",
          max(len(ln) for ln in rm_c.summary().splitlines()) <= 110)

    # -- 2b. the fourth field report ---------------------------------------
    from types import SimpleNamespace
    mb_neg = MaterialBalanceOil(n_ooip_stb=4.8e8, n_stderr_stb=1.1e8,
                                r2=-821.8, n_surveys=6, p_initial=3493.0)
    check("a negative material-balance R2 is explained, not printed bare",
          "BELOW ZERO" in mb_neg.summary())
    check("ordinals are English", [_ordinal(k) for k in (1, 2, 3, 11, 92)]
          == ["1st", "2nd", "3rd", "11th", "92nd"])
    rng_t = np.random.default_rng(3)
    mc_flat = pd.DataFrame({
        "eur_oil_mstb": 429.0 + rng_t.normal(0.0, 0.05, 400),
        "model": rng_t.choice(["arps", "sepd"], 400, p=[0.1, 0.9])})
    sm_flat = summarise_oil_mc(mc_flat, SimpleNamespace(
        eur_oil_mstb=429.3, model_name="modified_hyperbolic"))
    check("a band with nothing in it makes no claims about percentiles",
          "no spread to read" in sm_flat and "percentile" not in sm_flat
          and "OUTSIDE" not in sm_flat.upper().replace("OUTSIDE THE", ""))
    fake_fit = SimpleNamespace(
        params={"qi": 31.0, "Di": 1.29e-5, "b": 1.99, "Dmin": 2.28e-4},
        summary=lambda: "  Di (eff)  : 0.5 %/yr at t0\n  Dmin(eff) : 8.0 %/yr")
    txt_mh = _fit_summary_text(fake_fit)
    check("a Dmin fitted above Di is not reported as the terminal decline",
          "8.0 %/yr" not in txt_mh and "NOT USED" in txt_mh
          and "exponential" in txt_mh)
    # GOR highest at first production and far beyond any black oil.
    npg = np.linspace(1.0e3, 4.3e5, 300)
    g_hi = 60000.0 * np.exp(-npg / 1.5e5) + 4000.0
    g_hi = g_hi * (1.0 + 0.05 * rng_t.standard_normal(300))
    gd_hi = gor_diagnostic(g_hi, npg, 391.0, saturated=True)
    sm_hi = gd_hi.summary()
    check("a GOR peak at the start of the record is not read as spent "
          "solution-gas drive", gd_hi.peak_at_start
          and "solution-gas drive is spent" not in sm_hi
          and "START of the record" in sm_hi)
    check("a GOR no black oil can make is flagged, units included",
          gd_hi.implausible_for_black_oil and "Mscf/d" in sm_hi)
    # WOR falling from the first record.
    tw = 30.4375 * np.arange(1, 301)
    qo_w = np.full(300, 100.0)
    wor_w = 10.0 * (tw / tw[0]) ** -0.3 * np.exp(
        0.1 * rng_t.standard_normal(300))
    wd_f = chan_diagnostic(tw, qo_w, qo_w * wor_w)
    sm_wf = wd_f.summary()
    check("a falling WOR is not called coning",
          wd_f.mechanism.startswith("WOR FALLING")
          and "CONING" not in sm_wf, wd_f.mechanism.split("\n")[0])
    check("no departure/sigma pair is printed for a falling WOR",
          "departure from a straight" not in sm_wf)
    check("water from the first record is said",
          wd_f.water_from_start and "water from the first record" in sm_wf)
    # A plateau end set by the 75 % guard rail.
    dts = pd.date_range("1970-01-01", periods=400, freq="MS")
    qpl = np.where(np.arange(400) < 360, 100.0,
                   100.0 - 30.0 * (np.arange(400) - 360) / 39.0)
    pd_cap = OilProductionData.prepare(
        pd.DataFrame({"date": dts, "q_oil": qpl, "q_gas": qpl * 0.4}),
        pvt, well="CAP")
    check("a plateau end set by the guard rail says so",
          pd_cap.qc.plateau_capped
          and "GUARD RAIL" in pd_cap.qc.summary())
    qpl2 = np.where(np.arange(400) < 100, 100.0,
                    100.0 * np.exp(-0.01 * (np.arange(400) - 100)))
    pd_ok = OilProductionData.prepare(
        pd.DataFrame({"date": dts, "q_oil": qpl2, "q_gas": qpl2 * 0.4}),
        pvt, well="OK")
    check("a real plateau end is not called a guard rail",
          not pd_ok.qc.plateau_capped
          and "GUARD RAIL" not in pd_ok.qc.summary(),
          f"plateau end day {pd_ok.qc.plateau_end_days}")

    # -- 2c. aquifer history match ----------------------------------------
    td_chk = np.array([0.01, 0.1, 1.0, 10.0, 100.0])
    rel = np.abs(veh_wd(td_chk) / _edwardson_wd(td_chk) - 1.0)
    check("the radial WD matches Edwardson's published fit (infinite aquifer)",
          float(rel.max()) < 1e-3, f"worst {100 * float(rel.max()):.3f} %")
    check("a finite aquifer's WD tends to (reD^2 - 1)/2",
          abs(float(veh_wd(np.array([1e5]), 5.0)[0]) - 12.0) < 1e-3)
    check("a large aquifer behaves as an infinite one at early time",
          abs(float(veh_wd(np.array([1.0]), 20.0)[0])
              - float(veh_wd(np.array([1.0]))[0])) < 1e-6)

    fk = _march_fetkovich(pvt, 50.0e6, 0.0, 40.0e6, 600.0)
    # Same monthly steps as the marcher, so the comparison is of the solver
    # and not of two discretisations of the Fetkovich recursion (which moves
    # the answer by a few psi between monthly and 1.2-monthly steps).
    hk = _tank_history(fk["t"], fk["Np"], fk["Gp"], fk["Wp"], pvt,
                       pvt.p_init, float(fk["t"][-1]), n_steps=120)
    p_k, we_k, _ = _simulate_tank(hk, 50.0e6, 0.0, "fetkovich", 40.0e6, 600.0)
    dpk = float(np.max(np.abs(np.interp(fk["t"], hk.t, p_k) - fk["p"])))
    check("the grid solver reproduces an independently marched aquifer tank",
          dpk < 3.0, f"max |dp| {dpk:.2f} psi over "
          f"{pvt.p_init - fk['p'][-1]:,.0f} psi of depletion")
    rng_a = np.random.default_rng(1)
    ia = np.arange(6, 121, 6)
    ts_a = fk["t"][ia]
    ps_a = fk["p"][ia] + rng_a.normal(0.0, 10.0, len(ia))
    am = aquifer_history_match(fk["t"], fk["Np"], fk["Gp"], fk["Wp"], ts_a,
                               ps_a, pvt, kind="fetkovich",
                               n_guess_stb=80.0e6)
    check("a Fetkovich match recovers N inside its own 95 % range",
          am.ran and am.n_range_stb[0] <= 50.0e6 <= am.n_range_stb[1],
          f"N {am.n_stb / 1e6:.1f} MMstb, range "
          f"{am.n_range_stb[0] / 1e6:.1f}-{am.n_range_stb[1] / 1e6:.1f}")
    check("a Fetkovich match recovers the aquifer volume",
          abs(am.params.get("Wei", 0.0) / 40.0e6 - 1.0) < 0.10,
          f"Wei {am.params.get('Wei', float('nan')) / 1e6:.1f} vs 40.0 MMrb")
    check("Havlena-Odeh on the same record would not have: influx is not "
          "absorbed into the matched N", abs(am.n_stb / 50.0e6 - 1.0) < 0.10)
    am_r = aquifer_history_match(fk["t"], fk["Np"], fk["Gp"], fk["Wp"],
                                 ts_a[:6], ps_a[:6], pvt, kind="radial",
                                 fit_m=True)
    check("a match with too few surveys for its parameters refuses to run",
          not am_r.ran and "need at least 8 surveys" in am_r.summary(),
          am_r.refused_reason[:60])
    am_f6 = aquifer_history_match(fk["t"], fk["Np"], fk["Gp"], fk["Wp"],
                                  ts_a[:6], ps_a[:6], pvt, kind="fetkovich")
    check("N plus a Fetkovich aquifer runs on six surveys",
          am_f6.ran and am_f6.dof == 3, f"dof {am_f6.dof}")
    # A strong aquifer holding the pressure within ~500 psi: N and the
    # aquifer trade off almost completely, and the match must say so rather
    # than print its point value. Best-fit N on six noise draws of this tank
    # ran from 17 to 1,344 MMstb against a truth of 50.
    fs = _march_fetkovich(pvt, 50.0e6, 0.0, 150.0e6, 200.0)
    rng_s = np.random.default_rng(102)
    am_s = aquifer_history_match(fs["t"], fs["Np"], fs["Gp"], fs["Wp"],
                                 fs["t"][ia], fs["p"][ia]
                                 + rng_s.normal(0.0, 10.0, len(ia)), pvt,
                                 kind="fetkovich", n_guess_stb=80.0e6)
    lo_s, hi_s = am_s.n_range_stb
    check("a strong aquifer's N is reported as not determined",
          am_s.ran and "N NOT DETERMINED" in am_s.summary()
          and lo_s <= 50.0e6 <= hi_s,
          f"best {am_s.n_stb / 1e6:,.0f}, range {lo_s / 1e6:,.0f}-"
          f"{hi_s / 1e6:,.0f} MMstb")
    # The pot aquifer of the tool's own synthetic generator is a Fetkovich
    # aquifer with a time constant near zero; a different generator again.
    d_pot = make_synthetic_oil_well(pvt, n_ooip_stb=50.0e6, we_total_rb=12.0e6,
                                    noise_frac=0.03, survey_every=6, seed=5)
    r_pot = analyse_oil_well(d_pot, pvt, well="POT", q_econ_stbd=50.0,
                             mb_aquifer="fetkovich", run_monte_carlo=False)
    amp = r_pot.aquifer_match
    check("the matched N is used, and says so, when the surveys bound it",
          r_pot.aquifer_used and "matched N used" in r_pot.report()
          and "AQUIFER HISTORY MATCH" in r_pot.report())
    check("a pot aquifer is recovered by the Fetkovich match",
          amp is not None and abs(amp.n_stb / 50.0e6 - 1.0) < 0.03
          and amp.params.get("tau", 1e9) < 30.0,
          f"N {amp.n_stb / 1e6:.2f} MMstb, tau "
          f"{amp.params.get('tau', float('nan')):.1f} d" if amp else "")
    check("the Havlena-Odeh N on that record is biased by the influx the "
          "match accounts for", r_pot.matbal is not None
          and r_pot.matbal.n_ooip_stb > 1.2 * 50.0e6,
          f"HO {r_pot.matbal.n_ooip_stb / 1e6:.1f} MMstb")

    # -- 3. the balance inverted ------------------------------------------
    tk = _march_tank(50.0e6, 0.0)
    j = len(tk["p"]) // 2
    p_back = pressure_path_from_balance(
        pvt, 50.0e6, 0.0, 0.0, tk["Np"][j:j + 5], tk["Gp"][j:j + 5],
        np.zeros(5), p_floor=120.0)
    check("pressure inverted from the balance returns the pressure that "
          "produced it",
          float(np.max(np.abs(p_back - tk["p"][j:j + 5]))) < 5.0,
          f"max |dp| = {float(np.max(np.abs(p_back - tk['p'][j:j + 5]))):.2f} psi")

    # -- 4. diagnostics ----------------------------------------------------
    for mech, want in (("displacement", "normal displacement"),
                       ("coning", "CONING"), ("channelling", "CHANNELLING"),
                       ("none", "no appreciable water")):
        d = make_synthetic_oil_well(pvt, water_mechanism=mech, seed=11,
                                    noise_frac=0.05)
        tt = (d["date"] - d["date"].iloc[0]).dt.days.to_numpy(float) + 30.0
        wd = chan_diagnostic(tt, d["q_oil"].to_numpy(float),
                             d["q_water"].to_numpy(float))
        check(f"Chan reads {mech} correctly",
              wd.mechanism.startswith(want), wd.mechanism.split(" -")[0])

    d = make_synthetic_oil_well(pvt, water_mechanism="none", seed=11,
                                noise_frac=0.05)
    g = d["q_gas"].to_numpy(float) * SCF_PER_MSCF / d["q_oil"].to_numpy(float)
    npc = np.cumsum(d["q_oil"].to_numpy(float) * d["days_on"].to_numpy(float))
    gd = gor_diagnostic(g, npc, pvt.rsi)
    check("the bubble-point break is found on a well that crossed it",
          gd.broke, f"break at Np={0.0 if not gd.broke else gd.break_np_stb / 1e6:,.1f} MMstb")
    check("the GOR peak is reported once the GOR turns over", gd.peaked)
    # A reservoir saturated at discovery has no bubble point to cross: the
    # field report said "the reservoir crossing its bubble point" under a PVT
    # section that read "SATURATED at discovery".
    gd_sat = gor_diagnostic(g, npc, pvt.rsi, saturated=True)
    check("a GOR break on a saturated reservoir is not called the bubble "
          "point", gd_sat.broke
          and "crossing its bubble point" not in gd_sat.summary()
          and "critical saturation" in gd_sat.summary())
    check("an undersaturated reservoir still reports the bubble point",
          "crossing its bubble point" in gd.summary())

    d2 = make_synthetic_oil_well(pvt, water_mechanism="none", seed=11,
                                 noise_frac=0.05, we_total_rb=42.0e6,
                                 q_plateau_stbd=3000.0)
    g2 = d2["q_gas"].to_numpy(float) * SCF_PER_MSCF / d2["q_oil"].to_numpy(float)
    npc2 = np.cumsum(d2["q_oil"].to_numpy(float)
                     * d2["days_on"].to_numpy(float))
    gd2 = gor_diagnostic(g2, npc2, pvt.rsi)
    check("no break is invented on a well that stayed above its bubble point",
          not gd2.broke)

    for truth in (0.0, 0.05, 0.12):
        d3 = make_synthetic_oil_well(pvt, water_mechanism="none", seed=11,
                                     noise_frac=0.04, pi_stbd_per_psi=4.0,
                                     pi_decline_per_year=truth)
        tt = (d3["date"] - d3["date"].iloc[0]).dt.days.to_numpy(float) + 30.0
        q3 = d3["q_oil"].to_numpy(float)
        pid = oil_pi_diagnostic(
            tt, q3, d3["p_wf"].to_numpy(float), d3["p_res"].to_numpy(float),
            pvt, np_stb=np.cumsum(q3 * d3["days_on"].to_numpy(float)),
            n_ooip_stb=50e6)
        # The tool reports percent per year, exp(slope)-1; the generator is
        # given an exponential rate constant. Comparing the two directly is
        # what made a correct diagnostic look 1.2 pp low the first time.
        want = 100.0 * (1.0 - math.exp(-truth))
        got = -pid.trend_pct_per_year
        check(f"PI trend recovered, truth {100 * truth:.0f} %/yr nominal",
              abs(got - want) < 1.0, f"{got:+.2f} vs {want:+.2f} %/yr")

    # A two-month GOR spike is not a solution-gas peak. On a field record
    # one was read as the peak, "solution-gas drive is spent" was printed on
    # a GOR still rising, and the forecast was fitted to the 24 points after
    # the spike.
    rng_s = np.random.default_rng(4)
    np_s = np.linspace(0.05e6, 9.47e6, 380)
    g_s = (350.0 + 150.0 * (np_s / np_s[-1]) ** 2) * rng_s.lognormal(
        0, 0.12, 380)
    k_s = int(np.searchsorted(np_s, 8.89e6))
    g_s[k_s:k_s + 2] = [3636.0, 2900.0]
    gd_s = gor_diagnostic(g_s, np_s, 391.0)
    check("a two-month GOR spike is not read as the GOR peak",
          not gd_s.peaked, f"peaked={gd_s.peaked}")
    check("the spike is named as a spike",
          "a spike, not" in gd_s.summary())
    gm_s, gsh_s = fit_gor_model(np_s, g_s, pvt_d, diag=gd_s)
    recent_s = float(np.median(g_s[-10:]))
    check("every GOR shape opens its forecast near the recent readings",
          all(abs(math.log(float(m(np_s[-1:])[0]) / recent_s))
              <= math.log(1.35) + 1e-9 for m in gsh_s.values()),
          ", ".join(f"{k} {float(m(np_s[-1:])[0]):,.0f}"
                    for k, m in gsh_s.items()) + f" vs {recent_s:,.0f}")

    # -- 5. ratio models ---------------------------------------------------
    npx = np.linspace(1.0e6, 20.0e6, 60)
    wor_true = 0.05 * np.exp(2.0e-7 * (npx - npx[0]))
    rm = _fit_ratio_model(npx, wor_true, "WOR")
    check("a log-linear ratio is recovered exactly",
          abs(rm.slope_per_stb - 2.0e-7) / 2.0e-7 < 1e-6
          and abs(float(rm(npx[-1:])[0]) - wor_true[-1]) / wor_true[-1] < 1e-6)
    check("a clean ratio fit is reported as significant", rm.significant)
    rng = np.random.default_rng(0)
    draws = np.array([rm.perturb(rng).slope_per_stb for _ in range(200)])
    check("perturbing a ratio model actually moves it",
          float(np.std(draws)) > 0.0)
    rm_floor = _fit_ratio_model(npx, wor_true, "GOR", floor=1.0)
    check("a ratio model honours its floor",
          float(np.min(rm_floor(npx))) >= 1.0 - 1e-12)
    rm_few = _fit_ratio_model(npx[:4], wor_true[:4], "WOR")
    check("too few points falls back to a constant, not a trend",
          rm_few.kind == "constant" and not rm_few.significant)
    noise = wor_true * rng.lognormal(0.0, 1.2, size=len(npx))
    rm_bad = _fit_ratio_model(npx, noise, "WOR")
    check("a ratio fit through noise is not called significant",
          not rm_bad.significant or rm_bad.r2 >= 0.15,
          f"R2 {rm_bad.r2:.2f}, p {rm_bad.p_value:.2g}")

    # -- 6. forecast -------------------------------------------------------
    wl = make_synthetic_oil_well(pvt, n_ooip_stb=50.0e6,
                                 water_mechanism="displacement",
                                 water_breakthrough_frac=0.18,
                                 seed=11, noise_frac=0.05, n_months=150)
    tt = (wl["date"] - wl["date"].iloc[0]).dt.days.to_numpy(float) + 30.0
    dd = wl["days_on"].to_numpy(float)
    qo_h, qw_h = wl["q_oil"].to_numpy(float), wl["q_water"].to_numpy(float)
    qg_h = wl["q_gas"].to_numpy(float)
    np_h = np.cumsum(qo_h * dd)
    gp_h = np.cumsum(qg_h * dd * SCF_PER_MSCF)
    wp_h = np.cumsum(qw_h * dd)
    gor_h = qg_h * SCF_PER_MSCF / qo_h

    sel = tt >= tt[60]
    _, fits = rank_models(tt[sel], qo_h[sel])
    bf = fits["modified_hyperbolic"]
    gm, gm_shapes = fit_gor_model(np_h, gor_h, pvt,
                                  diag=gor_diagnostic(gor_h, np_h, pvt.rsi))
    wm, wm_shapes = fit_wor_model(np_h, qo_h, qw_h)
    base = dict(np_to_date_stb=float(np_h[-1]), gp_to_date_scf=float(gp_h[-1]),
                wp_to_date_stb=float(wp_h[-1]), t_start_days=float(tt[-1]),
                t_max_years=30.0)

    fc = forecast_oil(bf, gm, wm, pvt, 150.0, **base)
    tb = fc.table
    check("the forecast starts at the last historical record, not before",
          abs(float(tb["t_days"].iloc[0]) - float(tt[-1])) < 1e-6)
    check("remaining reserves are never negative",
          fc.remaining_oil_mstb >= 0.0 and fc.remaining_gas_mmscf >= 0.0,
          f"{fc.remaining_oil_mstb:,.1f} Mstb")
    check("EUR includes history",
          fc.eur_oil_mstb >= np_h[-1] / STB_PER_MSTB - 1e-6)
    for col in ("Np_mstb", "Gp_mmscf", "Wp_mstb"):
        check(f"{col} never goes backwards",
              bool(np.all(np.diff(tb[col].to_numpy(float)) >= -1e-9)))
    num = float(np.trapezoid(tb["q_oil_stbd"].to_numpy(float),
                             tb["t_days"].to_numpy(float))) / STB_PER_MSTB
    ana = float(tb["Np_mstb"].iloc[-1] - tb["Np_mstb"].iloc[0])
    check("forecast Np integrates the forecast oil rate",
          abs(num - ana) / max(ana, 1e-9) < 1e-3, f"{num:,.1f} vs {ana:,.1f}")
    check("the forecast does not overshoot its horizon",
          fc.forecast_years <= 30.0 + 1e-6, f"{fc.forecast_years:.3f} yr")

    short = forecast_oil(bf, gm, wm, pvt, 150.0,
                         **{**base, "t_max_years": 1.0 / 12.0})
    check("a horizon shorter than the history still runs forwards",
          short.forecast_years >= 0.0
          and short.economic_life_years >= tt[-1] / DAYS_PER_YEAR - 1e-6,
          f"{short.forecast_years:.3f} yr of forecast")

    no_limit = forecast_oil(bf, gm, wm, pvt, 0.0, **base)
    check("a zero economic rate means no rate limit, not a division by zero",
          no_limit.abandonment_reason == "max forecast life")
    check("an EUR set by the horizon rather than the reservoir says so",
          "THE HORIZON SET THIS EUR" in no_limit.summary())

    q_now = float(bf.model.rate(np.array([float(tt[-1])]))[0])
    sub = forecast_oil(bf, gm, wm, pvt, q_now * 1.6, **base)
    check("a curve already below the economic rate gives no forecast",
          "oil rate" in sub.constraints_breached_at_start
          and abs(sub.remaining_oil_mstb) < 1e-6
          and "ALREADY PAST" in sub.summary(),
          f"{sub.remaining_oil_mstb:,.3f} Mstb remaining")

    # An abandonment pressure with no N to turn production into pressure
    # cannot be applied. The field report dropped it without a word and
    # listed "abandonment pressure: 1,000 psia" in its settings.
    pa_none = forecast_oil(bf, gm, wm, pvt, 150.0,
                           **{**base, "p_abandon_psia": 1000.0})
    check("an abandonment pressure that cannot be applied says so",
          "abandonment pressure" in pa_none.constraints_not_applied
          and "NOT APPLIED" in pa_none.summary())
    # With the oil-in-place cap off, a fitted N must still drive the
    # pressure limit - the cap and the pressure are different jobs.
    pa_n = forecast_oil(bf, gm, wm, pvt, 150.0,
                        **{**base, "p_abandon_psia": 1000.0,
                           "n_pressure_stb": 50.0e6})
    check("a fitted N applies the pressure limit with the cap off",
          "reservoir pressure" in pa_n.constraint_years
          and not pa_n.constraints_not_applied
          and np.isfinite(pa_n.p_implied_end_psia),
          f"{pa_n.constraint_years}")
    check("the pressure N does not become an oil-in-place cap",
          "oil in place" not in pa_n.constraint_years)

    tight = forecast_oil(bf, gm, wm, pvt, 150.0,
                         **{**base, "water_cut_econ": 0.01})
    check("a limit already breached gives no forecast at all",
          "water cut" in tight.constraints_breached_at_start
          and abs(tight.remaining_oil_mstb) < 1e-6,
          f"{tight.remaining_oil_mstb:,.3f} Mstb remaining")

    wc = forecast_oil(bf, gm, wm, pvt, 150.0,
                      **{**base, "water_cut_econ": 0.70})
    check("a water-cut limit ends the well and is named",
          wc.abandonment_reason == "water cut"
          and wc.eur_oil_mstb < fc.eur_oil_mstb)
    check("a limit that does not bind is still reported",
          "water cut" in forecast_oil(
              bf, gm, wm, pvt, 150.0,
              **{**base, "water_cut_econ": 0.99}).constraint_years)

    capped = forecast_oil(bf, gm, wm, pvt, 1.0,
                          **{**base, "n_ooip_stb": 26.0e6})
    check("the oil-in-place cap truncates the forecast",
          capped.eur_oil_mstb <= 26.0e3 * 1.001,
          f"{capped.eur_oil_mstb:,.0f} Mstb against a 26,000 Mstb cap")

    # A WOR extrapolated log-linearly is unbounded above. Without a ceiling a
    # short record with a steep early water trend produced 7.6e12 Mstb of
    # water, the oil analogue of the gas module's runaway CGR.
    liq_hist = float(np.nanmax(qo_h + qw_h))
    free = forecast_oil(bf, gm, wm, pvt, 1.0, **base)
    capped_l = forecast_oil(bf, gm, wm, pvt, 1.0,
                            **{**base, "q_liquid_cap_stbd": 1.5 * liq_hist})
    check("the forecast liquid rate is bounded by what the well has shown",
          bool(np.all(capped_l.table["q_oil_stbd"]
                      + capped_l.table["q_water_stbd"]
                      <= 1.5 * liq_hist + 1e-6)),
          f"peak liquid {float(np.max(capped_l.table['q_oil_stbd'] + capped_l.table['q_water_stbd'])):,.0f} "
          f"against a {1.5 * liq_hist:,.0f} STB/d ceiling")
    check("water volumes never grow when the liquid ceiling is applied",
          capped_l.eur_water_mstb <= free.eur_water_mstb + 1e-6)

    # A short, plateau-heavy record is where the fitted N most often drifts
    # above the We>=0 ceiling, which is where the base case used to escape a
    # bound every realisation was held to.
    short_w = make_synthetic_oil_well(pvt, n_ooip_stb=40.0e6, seed=3,
                                      n_months=60,
                                      water_mechanism="displacement",
                                      water_breakthrough_frac=0.2)
    res_s = analyse_oil_well(short_w, pvt, well="SHORT", q_econ_stbd=20.0,
                             n_mc=150, t_max_years=30.0)
    if res_s.matbal is not None and np.isfinite(res_s.matbal.n_ceiling_stb):
        check("the base case is held under the same We>=0 ceiling as the "
              "realisations",
              res_s.forecast.eur_oil_mstb
              <= res_s.matbal.n_ceiling_stb / STB_PER_MSTB * 1.001,
              f"EUR {res_s.forecast.eur_oil_mstb:,.0f} vs ceiling "
              f"{res_s.matbal.n_ceiling_stb / STB_PER_MSTB:,.0f} Mstb")
    if res_s.mc is not None and len(res_s.mc):
        lo, hi = np.percentile(res_s.mc["eur_oil_mstb"], [10, 90])
        check("the base case lies inside its own P90-P10 on a short record",
              lo <= res_s.forecast.eur_oil_mstb <= hi,
              f"{res_s.forecast.eur_oil_mstb:,.0f} in [{lo:,.0f}, {hi:,.0f}]")
    check("a runaway water forecast is reported, not silently produced",
          res_s.forecast.eur_water_mstb < 1.0e7,
          f"{res_s.forecast.eur_water_mstb:,.0f} Mstb of water")

    # -- 7. Monte Carlo ----------------------------------------------------
    for tag, kw in (("unconstrained", {}),
                    ("water-limited", dict(water_cut_econ=0.70)),
                    ("N-capped", dict(n_ooip_stb=50.0e6))):
        det = forecast_oil(bf, gm, wm, pvt, 150.0, **{**base, **kw})
        mc = monte_carlo_oil_eur(bf, gm, wm, pvt, 150.0, n_samples=200,
                                 seed=5, n_ooip_hard_max=52.0e6,
                                 **{**base, **kw})
        if len(mc) == 0:
            check(f"Monte Carlo produced realisations ({tag})", False)
            continue
        p90, p10 = np.percentile(mc["eur_oil_mstb"], [10, 90])
        check(f"the deterministic case lies inside its own P90-P10 ({tag})",
              p90 <= det.eur_oil_mstb <= p10,
              f"{det.eur_oil_mstb:,.0f} in [{p90:,.0f}, {p10:,.0f}]")
        check(f"no realisation produces less than the well already has ({tag})",
              bool((mc["eur_oil_mstb"] >= np_h[-1] / STB_PER_MSTB
                    - 1e-6).all()))
        if "n_ooip_stb" in kw:
            check("no realisation exceeds the hard min(F/Et) ceiling",
                  float(mc["eur_oil_mstb"].max()) <= 52.0e3 * 1.001,
                  f"max {float(mc['eur_oil_mstb'].max()):,.0f} Mstb")

    # -- 7b. model-form uncertainty ----------------------------------------
    #
    # Sampling one model's covariance measures how well that curve's
    # parameters are pinned down, not whether it is the right curve. On blind
    # tests against withheld production that was the term that mattered.
    w_all = model_weights(fits)
    check("model weights are a normalised distribution",
          bool(w_all) and abs(sum(w_all.values()) - 1.0) < 1e-9,
          f"{len(w_all)} models")
    check("no single decline is given all the weight on a real record",
          max(w_all.values()) < 0.999 if w_all else False,
          f"top weight {max(w_all.values()):.3f}" if w_all else "none")

    # The correction has to be a property of the record, not of the candidate:
    # a per-model n_eff makes the AIC values incomparable and can hand the
    # weight to a worse curve with whiter residuals.
    rho_probe = []
    for fr_w in fits.values():
        rr = (np.log(np.maximum(fr_w.q_fit, 1e-12))
              - np.log(np.maximum(fr_w.model.rate(fr_w.t_fit), 1e-12)))
        rr = rr - rr.mean()
        rho_probe.append(float(np.sum(rr[1:] * rr[:-1])
                               / max(float(rr @ rr), 1e-30)))
    check("residual autocorrelation is measured, not assumed away",
          np.isfinite(np.median(rho_probe)),
          f"median rho {np.median(rho_probe):+.2f} over "
          f"{len(rho_probe)} models")

    mc_one = monte_carlo_oil_eur(bf, gm, wm, pvt, 150.0, n_samples=200,
                                 seed=5, **base)
    mc_mf = monte_carlo_oil_eur(bf, gm, wm, pvt, 150.0, n_samples=200,
                                seed=5, fits=fits, **base)
    if len(mc_one) and len(mc_mf):
        b1 = np.percentile(mc_one["eur_oil_mstb"], 90) - np.percentile(
            mc_one["eur_oil_mstb"], 10)
        b2 = np.percentile(mc_mf["eur_oil_mstb"], 90) - np.percentile(
            mc_mf["eur_oil_mstb"], 10)
        check("sampling model form widens the band rather than moving it",
              b2 > b1,
              f"P90-P10 {b1:,.0f} -> {b2:,.0f} Mstb")
        check("more than one decline actually appears in the realisations",
              "model" in mc_mf and mc_mf["model"].nunique() > 1,
              f"{mc_mf['model'].nunique()} models"
              if "model" in mc_mf else "no model column")
        det_mf = forecast_oil(bf, gm, wm, pvt, 150.0, **base)
        lo, hi = np.percentile(mc_mf["eur_oil_mstb"], [10, 90])
        check("the deterministic case still lies inside the wider band",
              lo <= det_mf.eur_oil_mstb <= hi,
              f"{det_mf.eur_oil_mstb:,.0f} in [{lo:,.0f}, {hi:,.0f}]")

    # A well declining much faster now than the curve being forecast. The
    # only check was for a SIGN change, and a last quarter five times steeper
    # than the fitted window went unmentioned on a field record.
    rng_w = np.random.default_rng(2)
    t_w = np.arange(240) * 30.44
    q_w = np.where(t_w < 1.5 * 365, 3000.0,
                   3000.0 * np.exp(-0.035 * (t_w - 1.5 * 365) / 365))
    i75 = int(0.75 * 240)
    q_w = np.where(t_w > t_w[i75],
                   q_w[i75] * np.exp(-0.20 * (t_w - t_w[i75]) / 365), q_w)
    q_w = q_w * rng_w.lognormal(0, 0.08, 240)
    df_w = pd.DataFrame({"date": pd.date_range("2000-01-01", periods=240,
                                               freq="MS"),
                         "q_oil": q_w, "q_gas": q_w * 0.6,
                         "q_water": q_w * 0.3, "days_on": 30.0})
    r_w = analyse_oil_well(df_w, pvt, well="LATE", q_econ_stbd=50.0,
                           run_monte_carlo=False)
    check("a late decline much steeper than the fit is warned about",
          "declining faster now" in r_w._window_block())

    # -- 8. QC -------------------------------------------------------------
    raw = wl.copy()
    dup = pd.concat([raw, raw.iloc[[10, 11]]], ignore_index=True)
    pdta = OilProductionData.prepare(dup, pvt, well="T")
    check("repeated dates are dropped and counted",
          pdta.qc.n_duplicate_dates == 2, f"{pdta.qc.n_duplicate_dates}")
    base_data = OilProductionData.prepare(raw, pvt, well="T")
    check("cumulatives survive the QC filters",
          abs(base_data.Np[-1] - np_h[-1]) / np_h[-1] < 0.02,
          f"{base_data.Np[-1] / 1e6:,.2f} vs {np_h[-1] / 1e6:,.2f} MMstb")
    check("surveys are taken from the full record, not the filtered one",
          len(base_data.surveys) >= int(np.isfinite(
              pd.to_numeric(raw["p_res"], errors="coerce")).sum()) - 1)

    # -- 9. end to end -----------------------------------------------------
    res = analyse_oil_well(raw, pvt, well="SELFTEST", q_econ_stbd=150.0,
                           water_cut_econ=0.90, n_mc=150, t_max_years=30.0)
    rep = res.report()
    check("the report runs end to end and is not empty", len(rep) > 2000)
    for section in ("DATA AND QC", "PVT", "DIAGNOSTICS", "MATERIAL BALANCE",
                    "DECLINE FIT", "PRODUCING RATIOS", "FORECAST",
                    "SETTINGS USED"):
        check(f"the report contains the {section} section", section in rep)
    check("N is recovered end to end on a well built from a known tank",
          res.matbal is not None and res.matbal.trend_ok
          and abs(res.matbal.n_ooip_stb - 50.0e6) / 50.0e6 < 0.05,
          f"{res.matbal.n_ooip_stb / 1e6:,.2f} MMstb"
          if res.matbal is not None else "no balance")
    check("the settings block records every choice that moved a number",
          all(k in res.settings for k in
              ("rate basis", "fit window", "model selected",
               "economic oil rate", "forecast horizon", "oil-in-place cap",
               "Monte Carlo")))

    n_pass = sum(1 for _, ok, _ in results if ok)
    if verbose:
        print(f"{'-' * 78}\n  OIL MODULE SELF-TESTS\n{'-' * 78}")
        for name, ok, detail in results:
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] {name}" + (f"   ({detail})" if detail else ""))
        print(f"{'-' * 78}\n  {n_pass}/{len(results)} passed\n{'-' * 78}")
    return n_pass == len(results)


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_self_tests() else 1)
