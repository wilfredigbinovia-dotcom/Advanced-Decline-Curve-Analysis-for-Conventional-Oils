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
        base = (tab if tab is not None
                else standing_rs(p, self.api, self.gas_gravity,
                                 self.temperature_F))
        # Above the bubble point nothing more can dissolve: Rs is pinned at
        # Rsi. Correlations do not know that and will keep climbing.
        return np.where(p >= self.p_bubble, float(self.rsi),
                        np.minimum(base, float(self.rsi)))

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
            f"  OOIP (N)          : {self.n_ooip_mstb:,.0f} Mstb"
            + (f" +/- {self.n_stderr_stb / STB_PER_MSTB:,.0f}"
               if np.isfinite(self.n_stderr_stb) else ""),
            f"  R2                : {self.r2:.4f}",
            f"  gas cap m         : {self.m_gas_cap:.4f}"
            + (f" +/- {self.m_stderr:.4f}  (FITTED)" if self.m_fitted
               else "  (as entered)")
            + ("   - no gas cap assumed" if self.m_gas_cap == 0.0
               and not self.m_fitted else ""),
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
        lines += [
            f"  N ceiling, We>=0  : {self.n_ceiling_stb / STB_PER_MSTB:,.0f} "
            "Mstb = min(F/Et)"
            + (f"   ({self.n_thin_dp} early survey(s) excluded: too little "
               f"depletion to read F/Et)" if self.n_thin_dp else ""),
        ]
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
        if self.n_rel_se > N_MAX_REL_SE:
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
                       f"{self.n_locus_stb[1] / STB_PER_MSTB:,.0f} Mstb: that "
                       "pair IS the answer.")
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
        best = int(np.argmin(costs))
        m_in = float(grid[best])
        # The half-width of the region within 10 % of the minimum cost is a
        # far more honest error bar than a curvature-based one on an objective
        # this flat.
        near = grid[costs <= costs[best] * 1.10]
        m_se = float(0.5 * (near.max() - near.min())) if near.size > 1 else 0.0
        # N and m are almost perfectly anti-correlated - measured at -1.00 on
        # a marched tank with 15 psi of survey scatter - so the regression
        # standard error on N, which conditions on the fitted m, is not the
        # uncertainty in N. It read +/-0.3 % where repeated noise realisations
        # moved N by +/-9 %. The honest bar is the spread of N across the m
        # values the data cannot distinguish.
        n_locus = [_fit_for_m(float(mg))[0] for mg in near]
        n_locus = [v for v in n_locus if np.isfinite(v)]
        m_locus = (float(near.min()), float(near.max()))
        n_locus_range = ((float(np.min(n_locus)), float(np.max(n_locus)))
                         if len(n_locus) > 1 else None)

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

    if not np.isfinite(spread):
        drive, note = "unknown", ("too few usable surveys to read apparent "
                                  "oil in place.")
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
        m_fitted=bool(fit_gas_cap), m_stderr=m_se, r2=r2,
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
                         f"({self.plateau_end_days / DAYS_PER_YEAR:.2f} yr)")
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
    peaked: bool = False                    # GOR has turned over
    peak_np_stb: Optional[float] = None
    scatter_frac: float = float("nan")      # robust noise / plateau level
    n_points: int = 0
    below_rsi_fraction: float = float("nan")
    reason: str = ""

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
        if self.broke:
            lines.append(
                f"                      GOR breaks upward at Np = "
                f"{self.break_np_stb / STB_PER_MSTB:,.0f} Mstb "
                f"({100 * self.break_frac_of_record:.0f} % of the record): "
                "that is the\n                      reservoir crossing its "
                "bubble point, read from surface data alone.")
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
        if self.peaked and self.peak_np_stb is not None:
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
                   recent_periods: int = 12) -> GORDiagnostic:
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
    pk = int(np.argmax(sm))
    peaked = pk < len(g) - 3 and float(sm[pk]) > float(sm[-1]) * 1.10

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
        ok=True, gor_initial=g0, rsi=float(rsi), gor_max=float(np.max(g)),
        gor_recent=g_recent,
        ratio_recent=(g_recent / float(rsi)) if rsi > 0 else float("nan"),
        break_np_stb=(float(n[best_i]) if broke else None),
        break_frac_of_record=(float(best_i) / len(g) if broke else float("nan")),
        peaked=bool(peaked),
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
        if np.isfinite(self.wor_slope):
            lines.append(
                f"                      Chan WOR slope {self.wor_slope:+.2f} "
                f"over {self.n_fitted} periods after breakthrough "
                f"(R2 {self.r2:.2f})")
            # Below a WOR slope of about 0.3 the departure is 2*c2 divided by
            # a slope near zero, so it is arbitrarily large and means nothing.
            # A flat WOR is already the whole finding; printing a -7.9 next to
            # it would only invite someone to read a magnitude into noise.
            if abs(self.wor_slope) >= 0.3 and np.isfinite(
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
    if s1 < 0.3:
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
            return f"  {self.label} model        : none ({self.note})"
        if self.kind == "constant":
            return (f"  {self.label} model        : held constant at "
                    f"{math.exp(self.ln_r0):,.3g} ({self.note})")
        return (f"  {self.label} model        : {self.kind}, "
                f"{self.pct_per_mmstb:+,.0f} % per MMstb of oil "
                f"(R2 {self.r2:.2f}, p {self.p_value:.1e}, n={self.n_points})"
                + (f"\n                      {self.note}" if self.note else ""))


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
    for kind in shapes:
        m = _fit_one_shape(xs, rs_, kind, label, floor, ceiling, note)
        if m is not None and np.isfinite(m.log_rss):
            out[kind] = m
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
        if self.constraints_breached_at_start:
            lines.append(
                "                      ALREADY PAST: "
                + ", ".join(self.constraints_breached_at_start)
                + " - the well is over this limit on the first forecast "
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
                 q_liquid_cap_stbd: Optional[float] = None) -> OilForecast:
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
    if (n_ooip_stb is not None and np.isfinite(n_ooip_stb)
            and n_ooip_stb > 0 and len(t) >= 2):
        def _p_on(grid_np, grid_gp, grid_wp):
            vals = pressure_path_from_balance(
                pvt, float(n_ooip_stb), float(m_gas_cap),
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
            pvt, float(n_ooip_stb), float(m_gas_cap), float(we_to_date_rb),
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
                        q_liquid_cap_stbd=q_liquid_cap_stbd)
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
    if len(e) >= 20 and not (25.0 <= pctl <= 75.0):
        lines.append(
            f"                      The base case sits at the "
            f"{pctl:.0f}th percentile of its own distribution, not the "
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
                f"                      Sampled: {mix}.\n"
                f"                      The +/-{band:.1f} % band covers both "
                "the parameters and the disagreement between\n"
                "                      curves. On synthetic wells given a "
                "quarter to a third of their life as\n"
                "                      history it contained the eventual "
                "outturn in 9 cases of 18 - and in 4 of the\n"
                "                      6 where nothing else in the report "
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
                "                      of 18, against 9 of 18 once the "
                "choice of curve was sampled as well.")
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

        out.append("DECLINE FIT")
        out.append(self.model_table.to_string(index=False))
        out.append("")
        out.append(f"  selected          : {self.best_fit.model_name}")
        out.append("  " + self.best_fit.summary().replace("\n", "\n  ").strip())
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
        gor_diag = gor_diagnostic(data.gor, data.Np, pvt.rsi)
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
    if matbal is not None and matbal.trend_ok and np.isfinite(matbal.n_ooip_stb):
        m_used = float(matbal.m_gas_cap)
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
                n_ooip_hard_max=n_hard_max, **fkw)
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
                                 if p_abandon_psia else "none"),
        "forecast horizon": f"{t_max_years:,.0f} yr from the last record",
        "material balance": ("on" if use_material_balance else "off"),
        "gas cap m": (f"{m_used:.3f} "
                      + ("(fitted)" if mb_fit_gas_cap else "(supplied)")),
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
