"""
================================================================================
 oil_app.py -- Streamlit front end for oil decline curve analysis
================================================================================

Run locally:
    streamlit run oil_app.py

Layout
------
  sidebar : fluid definition (black-oil PVT), run settings, limits
  tabs    : Data & QC -> Decline fit -> Diagnostics -> Material balance ->
            Forecast -> Uncertainty -> Field -> Export -> About

The analysis all lives in `oil_dca.py` and the charts in `oil_charts.py`; this
file is intake, state and presentation. Expensive steps are cached on their
inputs so moving a slider does not re-run the whole field.

This is a SEPARATE app from the gas condensate one. The two share the decline
machinery and the chart contract through imports and nothing else, so a change
here cannot reach a gas well.
================================================================================
"""

from __future__ import annotations

import io
import math
import os
import sys
import traceback
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# -- local module import ------------------------------------------------------
# Streamlit Cloud redacts import errors, so a missing sibling file arrives as an
# unexplained ModuleNotFoundError. Look in the obvious places, then fail with a
# message that says what is missing and what is actually present.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _candidate in (_HERE, os.path.join(_HERE, "streamlit_app"),
                   os.path.join(_HERE, "src"), os.path.dirname(_HERE)):
    if os.path.isdir(_candidate) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

_import_error: Optional[BaseException] = None
try:
    import oil_dca as od
    import oil_charts as oc
except BaseException as _exc:                      # noqa: BLE001
    _import_error = _exc
    od = oc = None                                 # type: ignore[assignment]

st.set_page_config(page_title="Oil DCA", page_icon="\U0001F6E2",
                   layout="wide", initial_sidebar_state="expanded")

if _import_error is not None:
    st.error("The analysis modules could not be imported.")
    st.code(f"{type(_import_error).__name__}: {_import_error}")
    st.caption("Files next to this app:")
    st.code("\n".join(sorted(os.listdir(_HERE))))
    st.stop()


# ==============================================================================
# Presentation helpers
# ==============================================================================

def current_theme() -> str:
    try:
        base = st.get_option("theme.base")
    except Exception:
        base = None
    return "dark" if str(base).lower() == "dark" else "light"


def show_fig(fig, key: Optional[str] = None) -> None:
    # `width="stretch"` rather than `use_container_width=True`: the old
    # argument still works on Streamlit 1.63 but is past its stated removal
    # date and emits a deprecation line for every chart on every rerun, which
    # buries anything the app actually wants to say in the log.
    st.plotly_chart(fig, width="stretch", key=key,
                    config={"displaylogo": False, "scrollZoom": True})


def show_df(df: pd.DataFrame, **kwargs) -> None:
    st.dataframe(df, width="stretch", hide_index=True, **kwargs)


def note(text: str) -> None:
    st.caption(text)


def mono(text: str) -> None:
    """Report text, in a monospaced block so its column alignment survives."""
    st.code(text, language=None)


# Every chart that shares a row with another is given the SAME height.
# Streamlit columns do not equalise their contents, so two figures at 440 and
# 400 render as one tall chart beside one short one, which reads as though
# the taller one matters more.
PANEL_H = 420

PASTE_ROWS = 60
PASTE_COLS = ["date", "q_oil", "q_gas", "q_water", "days_on", "p_res", "p_wf"]


def blank_paste_frame(n: int = PASTE_ROWS) -> pd.DataFrame:
    return pd.DataFrame({c: [None] * n for c in PASTE_COLS})


@st.cache_data(show_spinner=False)
def read_upload(data: bytes, name: str) -> pd.DataFrame:
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data))
    return pd.read_csv(io.BytesIO(data))


@st.cache_data(show_spinner=False)
def make_demo(n_wells: int, seed: int, pvt_sig: tuple,
              mechanism: str) -> pd.DataFrame:
    """A synthetic field whose oil in place is known, for trying the tool out.

    Every well is marched from a tank with a known N, so the material balance
    tab has a right answer to be judged against rather than merely a plausible
    one. The wells differ in size, breakthrough timing and productivity decline
    so the diagnostics have something to separate.
    """
    pvt = build_pvt(*pvt_sig)
    rng = np.random.default_rng(seed)
    frames = []
    mechs = ([mechanism] * n_wells if mechanism != "mixed"
             else ["displacement", "coning", "channelling", "none"] * 4)
    for i in range(n_wells):
        n_stb = float(rng.uniform(25.0, 80.0)) * 1.0e6
        frames.append(od.make_synthetic_oil_well(
            pvt, well=f"OIL-{i + 1:02d}", n_ooip_stb=n_stb,
            m_gas_cap=float(rng.choice([0.0, 0.0, 0.25])),
            we_total_rb=float(rng.choice([0.0, 0.0, 6.0e6])),
            q_plateau_stbd=float(rng.uniform(2500.0, 7000.0)),
            water_mechanism=mechs[i % len(mechs)],
            water_breakthrough_frac=float(rng.uniform(0.15, 0.35)),
            pi_stbd_per_psi=float(rng.uniform(2.5, 6.0)),
            pi_decline_per_year=float(rng.uniform(0.0, 0.10)),
            n_months=int(rng.integers(90, 160)), seed=int(seed + i)))
    return pd.concat(frames, ignore_index=True)


def build_pvt(api: float, gas_gravity: float, temperature_F: float,
              rsi: float, p_init: float, p_bubble: Optional[float],
              sw: float, cf: float, cw: float, bw: float,
              y_n2: float, y_co2: float, y_h2s: float) -> "od.OilPVT":
    return od.OilPVT(api=api, gas_gravity=gas_gravity,
                     temperature_F=temperature_F, rsi=rsi, p_init=p_init,
                     p_bubble=p_bubble, sw_initial=sw, cf_per_psi=cf,
                     cw_per_psi=cw, bw=bw, y_n2=y_n2, y_co2=y_co2, y_h2s=y_h2s)


@st.cache_data(show_spinner=False)
def run_analysis(df: pd.DataFrame, pvt_sig: tuple, settings: tuple,
                 well_col: Optional[str]):
    """Analyse every well. Keyed on the frame, the PVT and the settings."""
    (q_econ, model, t_max, run_mc, n_mc, use_mb, fit_m, m_in, mb_pi, mb_skip,
     cap, wcut_lim, qw_lim, p_ab, rate_basis, min_uptime, outlier_sigma,
     fit_from_bdf, window, sample_mf, aq_kind, aq_phi, aq_ro, aq_mu) = settings
    pvt = build_pvt(*pvt_sig)

    raw = od.map_oil_columns(df)
    groups = ({str(k): v for k, v in raw.groupby(well_col)}
              if well_col and well_col in raw.columns else {"FIELD": raw})

    results, errors = {}, {}
    for name, grp in groups.items():
        try:
            data = od.OilProductionData.prepare(
                grp, pvt, well=name, rate_basis=rate_basis,
                min_uptime_frac=min_uptime,
                outlier_sigma=(None if outlier_sigma <= 0 else outlier_sigma))
            results[name] = od.analyse_oil_well(
                data, pvt, well=name, q_econ_stbd=q_econ, select=model,
                t_max_years=t_max, run_monte_carlo=run_mc, n_mc=n_mc,
                use_material_balance=use_mb, mb_fit_gas_cap=fit_m,
                mb_m_gas_cap=(None if fit_m else m_in),
                mb_p_initial=(mb_pi if mb_pi and mb_pi > 0 else None),
                mb_skip_early=int(mb_skip), apply_ooip_cap=cap,
                mb_aquifer=aq_kind,
                aq_porosity=(aq_phi if aq_phi and aq_phi > 0 else None),
                aq_ro_ft=(aq_ro if aq_ro and aq_ro > 0 else None),
                aq_mu_w_cp=float(aq_mu or 0.5),
                water_cut_econ=(wcut_lim / 100.0 if wcut_lim > 0 else None),
                q_water_econ_stbd=(qw_lim if qw_lim > 0 else None),
                p_abandon_psia=(p_ab if p_ab and p_ab > 0 else None),
                fit_from_bdf=fit_from_bdf, fit_window_days=window,
                sample_model_form=bool(sample_mf))
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"

    rows = []
    for name, r in results.items():
        mb = r.matbal
        row = {
            "well": name,
            "model": r.best_fit.model_name,
            "points_fitted": r.best_fit.n_points,
            "R2_log": r.best_fit.r2,
            "b": r.best_fit.params.get("b", np.nan),
            "Np_to_date_Mstb": float(r.data.Np[-1]) / od.STB_PER_MSTB,
            "EUR_oil_Mstb": r.forecast.eur_oil_mstb,
            "remaining_oil_Mstb": r.forecast.remaining_oil_mstb,
            "EUR_gas_MMscf": r.forecast.eur_gas_mmscf,
            "EUR_water_Mstb": r.forecast.eur_water_mstb,
            "life_yr": r.forecast.economic_life_years,
            "ended_by": r.forecast.abandonment_reason,
            "recovery_factor": r.forecast.recovery_factor,
            "water_cut_end": r.forecast.water_cut_end,
            "N_Mstb": (mb.n_ooip_mstb if mb and mb.n_determined else np.nan),
            "N_ceiling_Mstb": (mb.n_ceiling_stb / od.STB_PER_MSTB
                               if mb and mb.trend_ok else np.nan),
            "drive": (mb.drive if mb and mb.trend_ok else "not run"),
            "water_mechanism": (r.water_diag.mechanism.split(" -")[0]
                                if r.water_diag and r.water_diag.ok
                                else "not run"),
            "PI_trend_pct_yr": (r.pi_diag.trend_pct_per_year
                                if r.pi_diag and r.pi_diag.ok else np.nan),
        }
        if r.mc is not None and len(r.mc):
            v = r.mc["eur_oil_mstb"].to_numpy(float)
            row["EUR_oil_P90"], row["EUR_oil_P50"], row["EUR_oil_P10"] = (
                np.percentile(v, [10, 50, 90]))
        rows.append(row)
    summary = (pd.DataFrame(rows).sort_values("EUR_oil_Mstb", ascending=False)
               .reset_index(drop=True) if rows else pd.DataFrame())
    return results, summary, errors


# ==============================================================================
# Sidebar
# ==============================================================================

with st.sidebar:
    st.markdown("### Fluid definition")
    note("Rs, Bo, Bt and Bg are built from these. The bubble point is derived "
         "from Rsi unless you set it, and everything downstream - the "
         "balance, the GOR ceiling, the PI - hangs off them.")

    c1, c2 = st.columns(2)
    api = c1.number_input("Oil gravity, API", 8.0, 60.0, 34.0, 0.5)
    gas_gravity = c2.number_input("Gas gravity (air=1)", 0.55, 1.40, 0.75,
                                  0.01, format="%.3f")
    temperature_F = c1.number_input("Reservoir temperature, F", 80.0, 400.0,
                                    190.0, 1.0)
    rsi = c2.number_input("Rsi, scf/STB", 10.0, 3000.0, 600.0, 10.0)
    p_init = c1.number_input("Initial pressure, psia", 200.0, 15000.0, 3800.0,
                             50.0)
    pb_override = c2.number_input("Bubble point, psia (0 = derive)", 0.0,
                                  15000.0, 0.0, 50.0)
    sw = c1.number_input("Connate water saturation", 0.0, 0.7, 0.22, 0.01)
    bw = c2.number_input("Bw, rb/STB", 0.95, 1.20, 1.02, 0.01)
    cf = c1.number_input("cf, 1/psi (x1e-6)", 1.0, 30.0, 4.0, 0.5) * 1e-6
    cw = c2.number_input("cw, 1/psi (x1e-6)", 1.0, 10.0, 3.0, 0.1) * 1e-6
    with st.expander("Gas impurities"):
        y_n2 = st.number_input("N2 fraction", 0.0, 0.5, 0.0, 0.01)
        y_co2 = st.number_input("CO2 fraction", 0.0, 0.5, 0.0, 0.01)
        y_h2s = st.number_input("H2S fraction", 0.0, 0.5, 0.0, 0.01)

    pvt_sig = (api, gas_gravity, temperature_F, rsi, p_init,
               (pb_override if pb_override > 0 else None), sw, cf, cw, bw,
               y_n2, y_co2, y_h2s)
    try:
        _pvt_preview = build_pvt(*pvt_sig)
        note(f"**{od.fluid_class(_pvt_preview)}** -- bubble point "
             f"{_pvt_preview.p_bubble:,.0f} psia, "
             + ("undersaturated by "
                f"{p_init - _pvt_preview.p_bubble:,.0f} psi"
                if _pvt_preview.p_bubble < p_init
                else "SATURATED at discovery"))
        # Standing, Vasquez-Beggs and Beggs-Robinson were each fitted to a
        # range of fluids. Outside it they still return numbers, and the
        # numbers are extrapolations of somebody else's curve - which is a
        # different and much weaker thing than an extrapolation of this
        # well's own data.
        _oor = od.correlation_range_warnings(_pvt_preview)
        if _oor:
            st.warning("**Outside the correlations' published range**\n\n"
                       + "\n\n".join(f"- {w}" for w in _oor)
                       + "\n\nEverything downstream inherits these. Supply "
                         "a measured PVT table if you have one.")
        _phys = od.pvt_physicality_warnings(_pvt_preview)
        if _phys:
            st.error("**This fluid model is not physical**\n\n"
                     + "\n\n".join(f"- {w}" for w in _phys))
    except Exception as exc:
        st.error(f"PVT could not be built: {exc}")

    st.markdown("---")
    st.markdown("### Decline fit")
    model = st.selectbox(
        "Model carried into the forecast",
        ["modified_hyperbolic", "arps", "ple", "sepd", "auto"], index=0,
        help="The modified hyperbolic is the default because it is the one "
             "with a defensible late-time limit. 'auto' takes the lowest "
             "AICc, which is a statistical choice and not a physical one.")
    fit_from_bdf = st.checkbox("Start the fit after the plateau", True)
    manual_window = st.checkbox("Set the fit window by hand", False)
    window = None
    if manual_window:
        w1, w2 = st.columns(2)
        lo = w1.number_input("from day", 0.0, 40000.0, 0.0, 30.0)
        hi = w2.number_input("to day (0 = end)", 0.0, 40000.0, 0.0, 30.0)
        window = (lo if lo > 0 else None, hi if hi > 0 else None)

    st.markdown("---")
    st.markdown("### Limits that end the well")
    note("The EARLIEST of these ends the forecast. Every one is reported "
         "whether it binds or not, because a well that drowns in three years "
         "and dies on rate in thirteen is a different asset from one where "
         "those numbers are the other way round.")
    q_econ = st.number_input("Economic oil rate, STB/d", 0.0, 5000.0, 20.0,
                             5.0, help="0 means no rate limit.")
    wcut_lim = st.number_input("Water cut limit, %", 0.0, 100.0, 95.0, 1.0,
                               help="0 means no water-cut limit.")
    qw_lim = st.number_input("Water handling limit, STB/d", 0.0, 500000.0,
                             0.0, 100.0, help="0 means no water-rate limit.")
    p_ab = st.number_input("Abandonment pressure, psia", 0.0, 10000.0, 0.0,
                           50.0,
                           help="0 means no pressure limit. Needs the "
                                "material balance; the aquifer is held at its "
                                "influx to date, so the date is early.")
    t_max = st.slider("Forecast horizon, years from the last record", 1, 60,
                      30, key="horizon_yr")

    st.markdown("---")
    st.markdown("### Material balance")
    use_mb = st.checkbox("Run the Havlena-Odeh balance", True)
    fit_m = st.checkbox("Fit the gas cap m as well as N", False,
                        help="N and m are strongly anti-correlated: a bigger "
                             "gas cap with a smaller oil volume fits nearly "
                             "the same history. Prefer an m from structure "
                             "and logs where you have one.")
    m_in = st.number_input("Gas cap m (if not fitted)", 0.0, 5.0, 0.0, 0.05)
    mb_pi = st.number_input("p_initial for the balance, psia (0 = infer)",
                            0.0, 15000.0, 0.0, 50.0,
                            help="Left at 0 the highest survey pressure is "
                                 "used, which is NOT p_i if the reservoir was "
                                 "already producing when it was taken.")
    mb_skip = st.number_input("Drop this many earliest surveys", 0, 20, 0, 1)
    cap = st.checkbox("Cap the forecast at the fitted oil in place", True)
    aq_kind = st.selectbox(
        "Aquifer history match", ["none", "fetkovich", "radial"],
        format_func=lambda k: {"none": "off",
                               "fetkovich": "Fetkovich (2 aquifer parameters)",
                               "radial": "Radial Van Everdingen-Hurst (3)"}[k],
        key="aq_kind",
        help="Simulates the tank through the production history with an "
             "aquifer attached and regresses N (and m, if the gas-cap fit is "
             "on) and the aquifer on the survey pressures, as MBAL does. The "
             "radial aquifer is regressed on the three combinations the "
             "pressures can see - U, the time constant and reD - not on "
             "porosity, thickness and angle separately, which trade off "
             "exactly. It refuses to run without at least three more "
             "surveys than parameters.")
    aq_phi = aq_ro = 0.0
    aq_mu = 0.5
    if aq_kind == "radial":
        with st.expander("Aquifer rock, to translate the answer (optional)"):
            aq_phi = st.number_input("Aquifer porosity", 0.0, 0.5, 0.0, 0.01,
                                     key="aq_phi")
            aq_ro = st.number_input("Reservoir radius ro, ft", 0.0, 1.0e5,
                                    0.0, 100.0, key="aq_ro")
            aq_mu = st.number_input("Water viscosity, cp", 0.1, 5.0, 0.5,
                                    0.05, key="aq_mu")

    st.markdown("---")
    st.markdown("### Data handling")
    rate_basis = st.radio("The rate columns are",
                          ["stream-day", "calendar-day", "volume"], index=0,
                          help="'volume' means the columns hold the volume "
                               "produced in the period, not a rate. "
                               "Cumulatives are then summed from the volumes "
                               "and never from rate times days.")
    min_uptime = st.slider("Drop months with uptime below", 0.0, 0.9, 0.35,
                           0.05)
    outlier_sigma = st.slider("Rate outlier rejection, sigma (0 = off)", 0.0,
                              8.0, 3.5, 0.5)

    st.markdown("---")
    st.markdown("### Uncertainty")
    run_mc = st.checkbox("Run the Monte Carlo", True, key="run_mc")
    n_mc = st.select_slider("Draws", [200, 400, 800, 1500, 3000], value=800,
                            key="mc_draws")
    sample_mf = st.checkbox(
        "Sample the choice of decline curve too", True, key="mc_model_form",
        help="Sampling one curve's covariance measures how well ITS "
             "parameters are pinned down, not whether it is the right curve. "
             "With this on, each realisation draws a decline model first - "
             "weighted by an Akaike weight corrected for the autocorrelation "
             "in monthly production, which is what makes a plain AICc hand "
             "one curve a weight of 1.000 after seven years of data. On "
             "synthetic wells given a quarter to a third of their life as "
             "history, the band went from containing the eventual outturn in "
             "2 cases of 18 to 11 of 18 once the curve and the GOR and WOR "
             "shapes were all sampled.")

settings = (q_econ, model, float(t_max), run_mc, int(n_mc), use_mb, fit_m,
            m_in, mb_pi, mb_skip, cap, wcut_lim, qw_lim, p_ab, rate_basis,
            min_uptime, outlier_sigma, fit_from_bdf, window, bool(sample_mf),
            aq_kind, float(aq_phi), float(aq_ro), float(aq_mu))

# ==============================================================================
# Intake
# ==============================================================================

st.title("Oil decline curve analysis")
st.caption(f"oil_dca v{od.__version__} -- conventional oil, solution gas, "
           "gas cap and water drive")

src = st.radio("Data source", ["Demo field", "Upload a file", "Paste a table"],
               horizontal=True)

df: Optional[pd.DataFrame] = None
if src == "Demo field":
    c1, c2, c3 = st.columns(3)
    n_wells = c1.slider("Wells", 1, 12, 4, key="demo_wells")
    seed = c2.number_input("Seed", 0, 9999, 17, 1, key="demo_seed")
    mech = c3.selectbox("Water mechanism",
                        ["mixed", "displacement", "coning", "channelling",
                         "none"], index=0)
    df = make_demo(int(n_wells), int(seed), pvt_sig, mech)
    note("Each demo well is marched from a tank with a KNOWN oil in place, so "
         "the material balance tab can be judged against a right answer "
         "rather than a plausible one.")
elif src == "Upload a file":
    up = st.file_uploader("CSV or Excel", type=["csv", "xlsx", "xls"])
    if up is not None:
        try:
            df = read_upload(up.getvalue(), up.name)
        except Exception as exc:
            st.error(f"That file could not be read: {exc}")
else:
    st.caption("Columns: date, q_oil, q_gas, q_water, days_on, p_res, p_wf. "
               "Only date and q_oil are required.")
    edited = st.data_editor(blank_paste_frame(), num_rows="dynamic",
                            width="stretch", key="paste_grid")
    cleaned = edited.dropna(how="all")
    if len(cleaned):
        df = cleaned.copy()

if df is None or not len(df):
    st.info("Load some production data to begin.")
    st.stop()

well_col = None
for cand in ("well", "Well", "WELL", "well_name", "uwi", "UWI", "api"):
    if cand in df.columns:
        well_col = cand
        break
if well_col is None:
    opts = ["(single well)"] + list(df.columns)
    pick = st.selectbox("Well identifier column", opts, index=0)
    well_col = None if pick == "(single well)" else pick

try:
    with st.spinner("Fitting declines, diagnostics and the material "
                    "balance..."):
        results, summary, errors = run_analysis(df, pvt_sig, settings,
                                                well_col)
except od.DateFormatError as exc:
    st.error(str(exc))
    st.stop()
except Exception:
    st.error("The analysis failed.")
    st.code(traceback.format_exc())
    st.stop()

if errors:
    with st.expander(f"{len(errors)} well(s) could not be analysed",
                     expanded=not results):
        for name, msg in errors.items():
            st.write(f"**{name}** -- {msg}")
if not results:
    st.stop()

names = list(results)
well = st.selectbox("Well", names, index=0)
res = results[well]
theme = current_theme()

tabs = st.tabs(["Data & QC", "Decline fit", "Diagnostics",
                "Material balance", "Forecast", "Uncertainty", "Field",
                "Export", "About"])

# ------------------------------------------------------------------- Data & QC
with tabs[0]:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### QC")
        mono(res.data.qc.summary())
    with c2:
        st.markdown("#### PVT")
        mono(res.pvt.summary())
    st.markdown("#### Produced streams")
    # Three charts rather than three stacked panels: equal width, equal
    # height, and each one's axis titles its own.
    s1, s2, s3 = st.columns(3)
    for col, which in ((s1, "oil"), (s2, "gas"), (s3, "water")):
        with col:
            show_fig(oc.chart_stream(res, which, theme, height=PANEL_H),
                     key=f"stream_{which}")
    show_df(res.data.df.head(500))

# ---------------------------------------------------------------- Decline fit
with tabs[1]:
    c1, c2 = st.columns(2)
    with c1:
        show_fig(oc.chart_rate_time(res, theme, height=PANEL_H), key="rate_t")
    with c2:
        show_fig(oc.chart_rate_cum(res, theme, height=PANEL_H), key="rate_cum")
    st.markdown("#### Model ranking")
    note("Lower AICc is better, but statistics is not physics: prefer the "
         "model whose late-time behaviour you can defend. A parameter shown "
         "under 'at_bounds' is not a fitted value.")
    show_df(res.model_table)
    mono(od._fit_summary_text(res.best_fit))
    wk = list(getattr(res.best_fit, "weak_params", []) or [])
    if wk:
        st.warning(f"At a bound: {', '.join(wk)}. The forecast inherits "
                   "whatever the bound was set to, not something the data "
                   "said.")
    blk = res._window_block()
    if blk.strip():
        mono(blk)

# ----------------------------------------------------------------- Diagnostics
with tabs[2]:
    c1, c2 = st.columns(2)
    with c1:
        show_fig(oc.chart_gor(res, theme, height=PANEL_H), key="gor")
        mono(res.gor_diag.summary() if res.gor_diag else "not run")
        show_fig(oc.chart_pi(res, theme, height=PANEL_H), key="pi")
        mono(res.pi_diag.summary() if res.pi_diag else "not run")
    with c2:
        show_fig(oc.chart_chan(res, theme, height=PANEL_H), key="chan")
        mono(res.water_diag.summary() if res.water_diag else "not run")
        show_fig(oc.chart_water(res, theme, height=PANEL_H), key="wcut")
    st.markdown("#### Producing ratios carried into the forecast")
    mono(res.gor_model.summary() + "\n" + res.wor_model.summary())

# ------------------------------------------------------------ Material balance
with tabs[3]:
    if res.matbal is None:
        st.info("The material balance was not run, or there were fewer than "
                "three pressure surveys.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            show_fig(oc.chart_havlena_odeh(res, theme, height=PANEL_H), key="ho")
        with c2:
            show_fig(oc.chart_apparent_n(res, theme, height=PANEL_H), key="appn")
        mono(res.matbal.summary())
        if res.matbal.ho_table is not None:
            with st.expander("Survey-by-survey table"):
                show_df(res.matbal.ho_table)
    am = getattr(res, "aquifer_match", None)
    if am is not None:
        st.markdown("#### Aquifer history match")
        if am.ran:
            c1, c2 = st.columns(2)
            with c1:
                show_fig(oc.chart_aquifer_match(res, theme, height=PANEL_H),
                         key="aqm")
            with c2:
                show_fig(oc.chart_aquifer_profile(res, theme, height=PANEL_H),
                         key="aqp")
        txt = am.summary()
        if res.aquifer_note:
            txt += ("\n" + ("  USED FOR FORECAST : " if res.aquifer_used
                            else "  NOT USED          : ") + res.aquifer_note)
        mono(txt)

# -------------------------------------------------------------------- Forecast
with tabs[4]:
    fc = res.forecast
    m = st.columns(4)
    m[0].metric("EUR oil, Mstb", f"{fc.eur_oil_mstb:,.0f}")
    m[1].metric("Remaining oil, Mstb", f"{fc.remaining_oil_mstb:,.0f}")
    m[2].metric("Life, yr", f"{fc.economic_life_years:,.1f}")
    m[3].metric("Ends on", fc.abandonment_reason)
    mono(fc.summary())
    show_fig(oc.chart_rate_time(res, theme, height=480), key="fcst_rate")
    with st.expander("Forecast table"):
        show_df(fc.table)

# ----------------------------------------------------------------- Uncertainty
with tabs[5]:
    if res.mc is None or not len(res.mc):
        st.info("The Monte Carlo was not run.")
    else:
        mono(od.summarise_oil_mc(res.mc, res.forecast, res.n_mc_requested))
        c1, c2 = st.columns(2)
        with c1:
            show_fig(oc.chart_eur_cdf(res, "eur_oil_mstb", "EUR oil (Mstb)",
                                      theme, height=PANEL_H), key="cdf_oil")
        with c2:
            show_fig(oc.chart_mc_scatter(res, "eur_oil_mstb", "life_years",
                                         "EUR oil (Mstb)", "Life (years)",
                                         theme, height=PANEL_H), key="sc1")
        if "model" in res.mc:
            mix = res.mc["model"].value_counts(normalize=True)
            if len(mix) > 1:
                st.caption(
                    "Decline models sampled: "
                    + ", ".join(f"{k} {100 * v:.0f} %"
                                for k, v in mix.items() if v >= 0.02)
                    + ". Each realisation draws a curve first, then that "
                      "curve's parameters, so the band carries the "
                      "disagreement between defensible declines rather than "
                      "hiding it behind whichever won on a statistic.")
        show_df(res.mc.describe(include="all").T.reset_index()
                .rename(columns={"index": "quantity"}))

# ----------------------------------------------------------------------- Field
with tabs[6]:
    if summary is None or not len(summary):
        st.info("Nothing to summarise.")
    else:
        show_df(summary)
        tot = summary[["Np_to_date_Mstb", "EUR_oil_Mstb",
                       "remaining_oil_Mstb"]].sum()
        m = st.columns(3)
        m[0].metric("Field Np to date, Mstb",
                    f"{tot['Np_to_date_Mstb']:,.0f}")
        m[1].metric("Field EUR oil, Mstb", f"{tot['EUR_oil_Mstb']:,.0f}")
        m[2].metric("Field remaining, Mstb",
                    f"{tot['remaining_oil_Mstb']:,.0f}")
        note("These are sums of independent single-well analyses. They are "
             "not a field forecast: the wells share a tank, and summing "
             "per-well oil in place double-counts wherever two wells drain "
             "the same rock.")

# ---------------------------------------------------------------------- Export
with tabs[7]:
    st.download_button("Report (.txt)", res.report(),
                       file_name=f"{well}_oil_report.txt", mime="text/plain")
    st.download_button("Forecast table (.csv)",
                       res.forecast.table.to_csv(index=False),
                       file_name=f"{well}_oil_forecast.csv", mime="text/csv")
    if summary is not None and len(summary):
        st.download_button("Field summary (.csv)",
                           summary.to_csv(index=False),
                           file_name="oil_field_summary.csv", mime="text/csv")
    if res.mc is not None and len(res.mc):
        st.download_button("Monte Carlo realisations (.csv)",
                           res.mc.to_csv(index=False),
                           file_name=f"{well}_oil_mc.csv", mime="text/csv")
    st.markdown("#### Report")
    mono(res.report())

# ----------------------------------------------------------------------- About
with tabs[8]:
    st.markdown(
        """
#### What this tool does

Conventional oil decline analysis with the reservoir kept in the loop. The oil
rate declines on a fitted Arps, modified hyperbolic, power-law exponential or
stretched exponential curve; **gas and water are not given declines of their
own** but are carried on the oil through a producing GOR and a WOR, both
fitted against cumulative oil. Three separate curves would drift apart; these
cannot.

Cumulative oil is the argument rather than time because that is what the
physics keys on. The GOR rises because the reservoir has given up a certain
fraction of its oil and the gas saturation has grown past its critical value;
the WOR rises because a certain fraction of the movable oil has been swept. A
well shut in for a year comes back with the same GOR and the same water cut it
had when it stopped.

#### Material balance

The full three-term Havlena-Odeh:

    F = N (Eo + m Eg + Efw) + We

fitted through the origin for N, with m either supplied or fitted alongside.
`Eo = Bt - Bti`, `Eg = Boi (Bg/Bgi - 1)`, `Efw = (1+m) Boi [(cw Swc + cf)/(1 - Swc)] dp`.

Two things are reported that a single number would hide. **Apparent N**,
`F/Et` survey by survey, is flat for a closed tank and climbs when volume is
arriving from outside the bracket - so the drive verdict is read off its
spread, not off a regression slope, because under influx the sequence rises
and then falls and a straight line through it says nothing. And the **We >= 0
ceiling**, `min(F/Et)`: influx can only add to the withdrawal, so no oil in
place may exceed it. On a marched tank with 9 MMrb of influx the fitted N came
back 29 % high and the ceiling was within 4 % of the truth.

When N and m are fitted together they are almost perfectly anti-correlated -
measured at -1.00 - so the report quotes the LOCUS of pairs the data cannot
separate rather than a standard error that describes only half of it.

#### Aquifer history match

Optional, and off by default. Where Havlena-Odeh treats influx as something to
detect, this does what MBAL's regression does: it simulates the tank forward
through the production history with an aquifer attached, predicts the pressure
at every survey, and regresses N (and m, if the gas-cap fit is on) and the
aquifer on the mismatch. Two aquifers: **Fetkovich** (pseudo-steady state,
two parameters - the encroachable water Wei and a time constant) and **radial
Van Everdingen-Hurst** (three - see below). The dimensionless influx WD is
inverted from Laplace space and agrees with Edwardson's published fit to
0.02 %.

It differs from MBAL in three deliberate ways.

- **It regresses on what the pressures can see.** A radial aquifer's porosity,
  thickness, angle, reservoir radius and permeability only reach the pressure
  through three combinations: `U = 1.119 phi ct h ro^2 (theta/360)`, the time
  constant `phi mu ct ro^2 / (0.006328 k)`, and `reD`. A 170-degree aquifer
  125 ft thick is identical to an 85-degree one 250 ft thick, so regressing on
  both at once regresses on nothing. Rock properties are used afterwards, to
  translate the answer - as products.
- **It refuses to run without degrees of freedom to spare** - at least three
  more surveys than parameters. With as many parameters as surveys the match
  passes through every point whatever N is, and a standard deviation of
  1e-8 psi is the sign of that, not of a good answer.
- **It reports a range, not a point.** N is held at each value in turn, the
  rest refitted, and the 95 % range read off at the F-test level. Where the
  range spans more than a factor of three the report says N is not
  determined, and the forecast keeps the Havlena-Odeh values.

On marched synthetic tanks with 10 psi of survey noise: a moderate Fetkovich
aquifer (1,600 psi of depletion) gave N within 6 % and a range of about
+/-9 %; with a gas cap and m fitted too, the range widened to a factor of 2.5;
a radial aquifer with 1,100 psi of depletion gave a range spanning a factor
of ten; and a strong aquifer holding the pressure within 500 psi gave
best-fit N anywhere from 17 to 1,340 MMstb on a 50 MMstb tank - every one of
them matching the surveys to within the noise. That last case is what a
single regressed N hides.

#### Diagnostics

**GOR.** The bubble point read from surface data alone: the break where the
producing GOR leaves its plateau at Rsi. The scan runs on the rising limb only,
because solution gas drive makes the GOR rise to a peak and then fall away, and
a scan that requires a rising tail misses the bubble point on exactly the
mature wells this exists for.

**Water, after Chan (1995).** WOR and its derivative on log-log. For a power
law the derivative's slope is the WOR slope minus one identically, so the only
thing the derivative adds is the departure from that - the curvature. The
classification keys on it: steepening is channelling, flattening is a
stabilising cone, straight is ordinary displacement. The distinction is worth
money, because coning responds to cutting the rate and channelling does not.
The abscissa is time since breakthrough rather than Chan's total producing
time; on a well that breaks through late the two are not the same, and the
total-time axis makes every mechanism look like it is flattening.

**Productivity index.** `PI = qo/(p_avg - p_wf)`, with p_avg from ONE source
for the whole series - the surveys, interpolated against cumulative oil -
never a measured value here and a modelled one there. Mixing them produced a
sawtooth that reported a 19 % fall and a +0 %/yr trend on the same numbers.

#### What ends the well

The earliest of: economic oil rate, water cut, water handling limit, the oil
in place from the balance, the gas originally in place, and an abandonment
pressure obtained by inverting the balance forward. All of them are reported
whether they bind or not. Where the abandonment pressure is used, the aquifer
is frozen at its influx to date - wrong in a known direction, understating the
support, so the date it gives is early.

#### What the uncertainty band is, and is not

The Monte Carlo samples the decline covariance, the GOR and WOR regressions
through their own standard errors, and the oil in place - and every
realisation is run under every constraint the base case obeys. A probabilistic
EUR that ignores a limit the deterministic case honours is not a distribution
around that case but around a different well.

It also samples **which decline curve**. Each realisation draws a model first
and then that model's parameters, so the band carries the disagreement between
defensible curves instead of hiding it behind whichever won on a statistic. The
weights are Akaike weights corrected for autocorrelation: monthly production
residuals are strongly correlated - 0.86 at lag 1 on one synthetic well, so
eighty-five points carried about six points' worth of independent information -
and an uncorrected AICc will hand a single curve a weight of 1.000 after seven
years of data, which is not a defensible level of confidence about which shape
a well is following.

This matters more than the parameters do. On synthetic wells marched to
abandonment and then truncated to a quarter or a third of their life, the
P90-P10 band contained the eventual outturn in **2 cases of 18** with
parameters alone, **9 of 18** once the choice of decline curve was sampled,
and **11 of 18** once the shapes of the GOR and WOR curves were sampled too -
including **all 6** of the wells where nothing else in the report raised a
warning. The point estimates barely moved; only the honesty of the range did.

The GOR and WOR are each fitted in five shapes - log-linear, linear, power,
constant, and for water a logistic on the cut, which cannot run away because a
water cut cannot exceed one - and a shape that would open the forecast with a
step away from the recent readings is dropped. Where nearly every realisation
ends on the same limit the band stops being a spread
on EUR at all and becomes a spread on the parameters of whichever curve that
limit is read off - so the report says so rather than letting a +/-1 % band
speak for itself.

#### Verification

181 self-tests, run with `python oil_dca.py`. They cover PVT shape and
continuity, N and m recovery on tanks marched from a known answer (exact at
m = 0.00, 0.25, 0.60 and 1.20), the water-drive refusal, the balance inverted
against the pressure that produced it, all four Chan mechanisms, the GOR break
in both directions, PI trend recovery, thirteen forecast invariants including
that forecast cumulative integrates forecast rate, the model-weighting and
model-form sampling, and the Monte Carlo consistency checks.

Beyond the self-tests there is a **blind forecast test**: eighteen wells
marched to abandonment, the history truncated to 19-42 % of life, the tool
given only the truncated part and its forecast scored against the future that
was withheld. Oil in place came back within 5 % on 9 of the 9 wells with a
simple drive; a gas cap was detected on 6 of 6 where one was present with no
false alarms on the other 12; influx was refused on 3 of 3. EUR error was a
**median 10.6 %, worst 20.2 %, on the six wells the tool raised no warning
about** - and a median 24 % with a worst case of 182 % on the twelve where it
did. That separation is the result worth quoting: the warnings are where the
information is.

#### What it is not for

Unconventional wells. The decline models are here, but the material balance,
the GOR ceiling and the PI diagnostic all assume a tank with boundary-dominated
flow, and a shale well in transient linear flow satisfies none of that.
        """
    )
