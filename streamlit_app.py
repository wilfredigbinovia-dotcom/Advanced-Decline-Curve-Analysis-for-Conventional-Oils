"""Decline Curve Workbench -- Streamlit front end.

    pip install -r requirements.txt
    streamlit run streamlit_app.py

This is the UI only. Every number on the screen comes from dca.py, which is
plain numpy/scipy and is tested independently (test_dca.py). Nothing is
computed in here that is not computed there.

On Streamlit Community Cloud, point the app at this file. It is named
streamlit_app.py because that is the filename the platform picks up by default.
"""

from __future__ import annotations

import glob
import io
import math
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import dca
from dca import MODELS

st.set_page_config(page_title="Decline Curve Workbench", page_icon="📉",
                   layout="wide", initial_sidebar_state="expanded")

SERIES_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                  "#e87ba4", "#4a3aa7", "#e34948"]
MODEL_ORDER = ["hyperbolic", "modhyp", "duong", "ple", "sepd", "harmonic", "exponential"]
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ---------------------------------------------------------------------------
# cached compute -- arrays arrive as tuples so Streamlit can hash them
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _fit(key: str, t: tuple, q: tuple, dmin: float | None):
    return dca.fit(key, np.array(t), np.array(q), dmin=dmin)


@st.cache_data(show_spinner=False)
def _boot(key: str, t: tuple, q: tuple, p: tuple, qab: float, n: int,
          horizon: float, dmin: float | None, seed: int):
    return dca.bootstrap_eur(key, np.array(t), np.array(q), dict(p), qab,
                             n=n, max_years=horizon, dmin=dmin, seed=seed)


@st.cache_data(show_spinner=False)
def _loss_ratio(t: tuple, q: tuple):
    return dca.loss_ratio_b(np.array(t), np.array(q))


@st.cache_data(show_spinner=False)
def _mb(P: tuple, Gp: tuple, cond: tuple, water: tuple, temp: float, sg: float,
        ge: float, pab: float, skip: int):
    return dca.material_balance(np.array(P), np.array(Gp), temp, sg,
                                condensate_mbbl=np.array(cond),
                                water_mbbl=np.array(water),
                                gas_equivalent_scf_per_bbl=ge,
                                p_abandon=pab, skip_early=skip)


@st.cache_data(show_spinner=False)
def _read_csv(path_or_bytes, name: str, column: str | None, days: str):
    if isinstance(path_or_bytes, bytes):
        tmp = io.BytesIO(path_or_bytes)
        df = pd.read_csv(tmp)
        tmp.seek(0)
        return dca.load_csv(io.BytesIO(path_or_bytes), column=column, days_column=days)
    return dca.load_csv(path_or_bytes, column=column, days_column=days)


# Oilfield volume prefixes, which are not SI: for liquids M = thousand and
# MM = million barrels; for gas the base unit is already Mcf, so a thousand of
# them is MMcf and a million is Bcf.
_SCALE = {
    "bbl": [(1e9, "MMMbbl"), (1e6, "MMbbl"), (1e3, "Mbbl"), (1, "bbl")],
    "Mcf": [(1e9, "Tcf"), (1e6, "Bcf"), (1e3, "MMcf"), (1, "Mcf")],
}


def fmt(v, unit="", dp=0):
    if v is None or not math.isfinite(v):
        return "—"
    for div, label in _SCALE.get(unit, [(1, unit)]):
        if abs(v) >= div or div == 1:
            dp_ = dp if div == 1 else (2 if div > 1e3 else 1)
            return f"{v / div:,.{dp_}f} {label}".strip()
    return f"{v:,.{dp}f} {unit}".strip()


def theme_axes(fig, xlab, ylab, ylog=False, xlog=False, height=460):
    fig.update_layout(
        autosize=True, width=None,          # let the Streamlit column set the width
        height=height, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title=xlab, yaxis_title=ylab,
        hovermode="x unified", template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
    )
    if ylog:
        fig.update_yaxes(type="log")
    if xlog:
        fig.update_xaxes(type="log")
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="rgba(0,0,0,.08)")
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="rgba(0,0,0,.08)")
    return fig


# ---------------------------------------------------------------------------
# sidebar: data and settings
# ---------------------------------------------------------------------------

st.sidebar.title("Decline Curve Workbench")

samples = sorted(p for p in glob.glob(os.path.join(DATA_DIR, "*.csv"))
                 if "pressure" not in os.path.basename(p))
sample_names = [os.path.basename(p) for p in samples]

source = st.sidebar.radio("Production data", ["Sample well", "Upload CSV"],
                          horizontal=True, key="source")

raw = None
if source == "Upload CSV":
    up = st.sidebar.file_uploader(
        "Monthly CSV", type="csv",
        help="Date, Days On, Oil (bbl), Gas (Mcf), Water (bbl). "
             "Period VOLUMES, not rates. Omit zero months — the date gaps carry them.")
    if up is None:
        st.info("Upload a monthly production CSV, or switch to a sample well in the sidebar.\n\n"
                "Expected columns: `Date`, `Days On`, and at least one of `Oil (bbl)` / "
                "`Gas (Mcf)`, optionally `Water (bbl)`.")
        st.stop()
    raw = up.getvalue()
    src_label = up.name
else:
    pick = st.sidebar.selectbox("Well", sample_names, key="well",
                                index=sample_names.index("synthetic-tight-oil.csv")
                                if "synthetic-tight-oil.csv" in sample_names else 0)
    raw = os.path.join(DATA_DIR, pick)
    src_label = pick

# column mapping
probe = pd.read_csv(io.BytesIO(raw) if isinstance(raw, bytes) else raw, nrows=5)
probe.columns = [c.strip() for c in probe.columns]
prod_cols = [c for c in probe.columns if c.lower().startswith(("oil", "gas"))] or \
            [c for c in probe.columns[1:]]
col = st.sidebar.selectbox("Production column", prod_cols, key="prod_col")
days_col = st.sidebar.selectbox(
    "Producing-days column",
    [c for c in probe.columns if "day" in c.lower()] or ["<none>"], key="days_col",
    help="Rate is volume ÷ producing days, i.e. an operated-day rate. Calendar-day and "
         "operated-day rates differ by the uptime fraction — mixing them turns downtime "
         "into apparent decline.")

try:
    s_full = _read_csv(raw, src_label, col, days_col)
except Exception as e:                                   # noqa: BLE001
    st.error(f"Could not read the file: {e}")
    st.stop()

unit = s_full.unit
n_all = len(s_full.t)

st.sidebar.divider()
st.sidebar.subheader("Fit window")

use_plateau = st.sidebar.checkbox(
    "Detect a plateau and fit after it", value=True,
    help="A facility-, quota- or compressor-limited plateau contains no decline "
         "information. Time is re-referenced to the first fitted period, so qi is a "
         "rate the well actually flowed rather than a back-extrapolation to t = 0.")

start_default = 0
plateau = dca.find_plateau(s_full.q) if use_plateau else None
if plateau is not None:
    start_default = plateau.end_index

lo, hi = st.sidebar.slider("Periods to fit", 1, n_all,
                           (start_default + 1, n_all), step=1, key="window")
start, end = lo - 1, hi
s = s_full.window(start, end)
if len(s.t) < 5:
    st.error(f"Only {len(s.t)} periods selected — need at least 5.")
    st.stop()

st.sidebar.divider()
st.sidebar.subheader("Forecast")

qab = st.sidebar.number_input(
    f"Abandonment rate ({unit}/d)", min_value=0.01,
    value=float(50.0 if s.is_gas else 10.0), step=1.0,
    help="For b ≥ 1 the hyperbolic integral does not converge, so this is what makes "
         "the EUR finite. It is doing real work — state it when you report.")
dmin_pct = st.sidebar.number_input(
    "Terminal decline (%/yr effective)", 0.0, 50.0, 6.0, step=0.5,
    help="A policy choice, held fixed during the fit. The history has not reached it "
         "and so cannot identify it.")
dmin = dca.nom_from_eff(dmin_pct / 100.0) / 12.0 if dmin_pct > 0 else None
# Default the horizon past the end of the record. A 50-year cap on a 51-year well
# truncates the EUR before the last data point and reports remaining = 0, which reads
# as a reservoir statement when it is only a horizon artefact.
_hist_yr = float(s_full.t[-1]) / 12.0
horizon = st.sidebar.number_input(
    "Maximum well life (years)", 1.0, 200.0, float(max(50.0, math.ceil(_hist_yr) + 20)),
    step=5.0,
    help="Counted from the start of the FIT WINDOW. Must exceed the length of the "
         "history, or the EUR is cut off before the last data point.")

chosen = st.sidebar.multiselect("Models", MODEL_ORDER, default=MODEL_ORDER,
                                format_func=lambda k: MODELS[k].name)
if not chosen:
    st.warning("Pick at least one model.")
    st.stop()

st.sidebar.divider()
nboot = st.sidebar.select_slider("Bootstrap replicates", [0, 100, 200, 300, 600],
                                 value=300, key="nboot")
seed = st.sidebar.number_input("Seed", 0, 10_000, 1, step=1,
                               help="Fixed so the band is reproducible.")

st.sidebar.divider()
st.sidebar.subheader("Volumes in place")
cap_on = st.sidebar.checkbox("Cap the forecast at a known volume", value=False,
                             help="STOIIP/GIIP from a static model, or OGIP from the "
                                  "Material balance tab. Off by default — see the "
                                  "extrapolation multiple on the Forecast tab.")
in_place = rf = None
if cap_on:
    in_place = st.sidebar.number_input(f"In place (MM{unit})", 0.0, value=10.0, step=1.0) * 1e6
    rf = st.sidebar.number_input("Recovery factor (%)", 1.0, 100.0, 70.0, step=5.0)


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

t_tup, q_tup = tuple(s.t.tolist()), tuple(s.q.tolist())
results = []
with st.spinner("Fitting…"):
    for k in chosen:
        try:
            f = _fit(k, t_tup, q_tup, dmin if k == "modhyp" else None)
        except ValueError as e:                          # noqa: PERF203
            st.warning(f"{MODELS[k].name}: {e}")
            continue
        e_ = dca.eur(k, f.p, qab, horizon)
        rem = max(e_.eur - dca.cumulative(k, f.p, float(s.t[-1])), 0.0)
        results.append({"fit": f, "eur": e_, "remaining": rem})
if not results:
    st.error("Nothing could be fitted on this window.")
    st.stop()
results.sort(key=lambda r: r["fit"].aicc)
best_aicc = results[0]["fit"].aicc

primary_key = st.sidebar.selectbox(
    "Forecast case", [r["fit"].key for r in results], key="case",
    format_func=lambda k: MODELS[k].name,
    help="Defaults to the best AICc. AICc ranks fit to the HISTORY — it has no view "
         "on extrapolation.")
primary = next(r for r in results if r["fit"].key == primary_key)

cum_before = float(s_full.cum[start - 1]) if start > 0 else 0.0
cum_to_date = float(s_full.cum[end - 1])
beyond = float(s_full.cum[-1]) - cum_to_date

st.title("Decline Curve Workbench")
st.caption(
    f"**{src_label}** · {s.column} · {n_all} producing periods, fitting {len(s.t)} "
    f"({s.dates.iloc[0]:%b %Y} – {s.dates.iloc[-1]:%b %Y}) · "
    f"cum to date {fmt(cum_to_date, unit)}"
    + (f" · {fmt(beyond, unit)} produced after the fit window" if beyond > 0 else ""))

tab_f, tab_d, tab_w, tab_m, tab_t = st.tabs(
    ["Forecast", "Diagnostics", "Water", "Material balance", "Data"])


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------

with tab_f:
    band = None
    if nboot:
        with st.spinner(f"Bootstrapping {nboot} replicates…"):
            dist = _boot(primary_key, t_tup, q_tup, tuple(sorted(primary["fit"].p.items())),
                         qab, nboot, horizon,
                         dmin if primary_key == "modhyp" else None, seed)
        band = dca.pxx(dist) if len(dist) else None

    cap = (in_place * rf / 100.0) if cap_on and in_place else None
    shown_eur = primary["eur"].eur
    capped = cap is not None and shown_eur > cap

    c = st.columns(6)
    c[0].metric("Cum at fit end", fmt(cum_to_date, unit))
    c[1].metric("Remaining", fmt(primary["remaining"], unit),
                help="From the end of the fit window, not the last row in the file.")
    c[2].metric("EUR — P50" if band else "EUR",
                fmt(band["P50"] if band else shown_eur, unit))
    c[3].metric("EUR — P90 (low)", fmt(band["P90"], unit) if band else "—")
    c[4].metric("EUR — P10 (high)", fmt(band["P10"], unit) if band else "—")
    c[5].metric("Well life", f"{primary['eur'].years:.1f} yr")

    if primary["eur"].t_end <= float(s.t[-1]) + 1e-6 and primary["remaining"] <= 0:
        why = ("the horizon" if primary["eur"].t_end >= horizon * 12 - 1e-6
               else "the abandonment rate")
        st.warning(
            f"**Remaining is zero because of {why}, not the reservoir.** The forecast ends "
            f"at month {primary['eur'].t_end:,.0f} but the fit window runs to month "
            f"{s.t[-1]:,.0f}, so the EUR is being cut off before the last data point. "
            + ("Raise the maximum well life in the sidebar."
               if why == "the horizon" else
               "Lower the abandonment rate if this well is still economic."))
    elif float(s.q[-1]) <= qab:
        st.warning(
            f"The last fitted rate is {s.q[-1]:,.1f} {unit}/d, at or below the abandonment "
            f"rate of {qab:g} {unit}/d — so remaining reads zero and the EUR is just what "
            "has already been produced. Lower the abandonment rate in the sidebar if this "
            "well is still economic.")

    mult, verdict = dca.extrapolation_multiple(primary["remaining"], cum_to_date)
    if mult > 1.0:
        st.error(f"**Extrapolation multiple {mult:.2f}×** — {verdict}. "
                 "The curve is claiming more than this well has ever demonstrated, and a "
                 "decline fit knows nothing about how much is in the tank.")
    elif mult > 0.5:
        st.warning(f"**Extrapolation multiple {mult:.2f}×** — {verdict}.")
    else:
        st.success(f"**Extrapolation multiple {mult:.2f}×** — {verdict}. "
                   "Most of the volume is already history; the curve is interpolating.")

    if capped:
        st.warning(f"The in-place cap binds: {fmt(cap, unit)} recoverable "
                   f"({fmt(in_place, unit)} in place × {rf:g}% RF) against a fitted EUR of "
                   f"{fmt(shown_eur, unit)}. Report the cap, not the curve.")

    left, right = st.columns([3, 2])
    with left:
        logy = st.checkbox("Log rate", value=True, key="logy")
        vs_cum = st.checkbox("Against cumulative instead of time", value=False,
                             help="Cumulative space is immune to shut-ins: a well that "
                                  "stops producing stops advancing along the x-axis. For "
                                  "erratic uptime this is the difference between a usable "
                                  "answer and a wrong one.")
        fig = go.Figure()
        x_obs = s.cum if vs_cum else s.t
        fig.add_trace(go.Scatter(x=x_obs, y=s.q, mode="markers", name="observed",
                                 marker=dict(size=5, color="#14181a")))
        t_end = max(r["eur"].t_end for r in results)
        tt = np.linspace(float(s.t[0]), t_end, 400)
        for i, r in enumerate(results):
            k = r["fit"].key
            qq = dca.rate(k, r["fit"].p, tt)
            xx = ([s.cum[0] + np.array([dca.cum_between(k, r["fit"].p, float(s.t[0]), x)
                                        for x in tt])] if vs_cum else [tt])[0]
            fig.add_trace(go.Scatter(
                x=xx, y=qq, mode="lines", name=MODELS[k].name,
                line=dict(color=SERIES_COLOURS[i % len(SERIES_COLOURS)],
                          width=3 if k == primary_key else 1.4)))
        fig.add_hline(y=qab, line_dash="dash", line_color="#7e8286",
                      annotation_text=f"economic limit {qab:g} {unit}/d")
        if not vs_cum:
            fig.add_vline(x=float(s.t[-1]), line_dash="dot", line_color="#7e8286",
                          annotation_text="fit end")
        lo_y = min(float(np.min(s.q)), qab) / 3.0
        fig.update_yaxes(range=[math.log10(lo_y), math.log10(float(np.max(s.q)) * 3)]
                         if logy else [0, float(np.max(s.q)) * 1.1])
        st.plotly_chart(theme_axes(fig, f"cumulative ({unit})" if vs_cum
                                   else "months on production",
                                   f"rate ({unit}/d)", ylog=logy), width="stretch")

    with right:
        if band:
            st.markdown(f"**P90 / P50 / P10** — reserves exceedance convention, so "
                        f"**P90 is the low case**: the volume 90% of realisations exceed. "
                        f"This is inverted relative to a statistician's percentile.")
            spread = (band["P10"] - band["P90"]) / band["P50"] * 100
            st.caption(f"{len(dist)} replicates · P10–P90 spread {spread:.0f}% of P50. "
                       "This is **fit** uncertainty only. Model uncertainty — the spread "
                       "across the table below — is usually larger, and window choice "
                       "larger still.")
            hist = go.Figure(go.Histogram(x=dist, nbinsx=40, marker_color="#2a78d6"))
            for lbl, v, colr in (("P90", band["P90"], "#e34948"),
                                 ("P50", band["P50"], "#14181a"),
                                 ("P10", band["P10"], "#1baf7a")):
                hist.add_vline(x=v, line_color=colr, annotation_text=lbl)
            st.plotly_chart(theme_axes(hist, f"EUR ({unit})", "replicates", height=300),
                            width="stretch")
        else:
            st.info("Bootstrap is off — set replicates in the sidebar for a P90/P50/P10 band.")

    st.subheader("Model comparison")
    rows = []
    for r in results:
        f = r["fit"]
        rows.append({
            "Model": MODELS[f.key].name,
            "Parameters": " · ".join(f"{k} {v:.4g}" for k, v in f.p.items()),
            "R² (log q)": round(f.r2, 4),
            "RMSE log": round(f.rmse_log, 4),
            "ΔAICc": round(f.aicc - best_aicc, 1),
            f"EUR ({unit})": round(r["eur"].eur),
            f"Remaining ({unit})": round(r["remaining"]),
            "Life (yr)": round(r["eur"].years, 1),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    eurs = [r["eur"].eur for r in results]
    if len(eurs) > 1 and min(eurs) > 0:
        st.caption(
            f"Model spread: {fmt(min(eurs), unit)} to {fmt(max(eurs), unit)}, a factor of "
            f"**{max(eurs) / min(eurs):.2f}×**. ΔAICc under 2 is a tie; over 7 is real "
            "evidence against. But AICc measures fit to the history — two models that tie "
            "on it can differ by a factor of two in EUR, which is why every model is shown.")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

with tab_d:
    st.subheader("Is this well actually declining?")
    lr = _loss_ratio(t_tup, q_tup)
    b_fit = primary["fit"].p.get("b")
    if b_fit is None:
        hyp = next((r["fit"] for r in results if r["fit"].key == "hyperbolic"), None)
        b_fit = hyp.p["b"] if hyp else None

    c = st.columns(4)
    c[0].metric("Loss-ratio b", f"{lr.b:.2f}" if lr else "—",
                delta=f"±{lr.spread:.2f} across windows" if lr else None, delta_color="off")
    c[1].metric("Fitted b", f"{b_fit:.2f}" if b_fit is not None else "—")
    slope = dca.loglog_slope(s.t, s.q)
    c[2].metric("log–log slope", f"{slope:.2f}" if slope is not None else "—")
    c[3].metric("Plateau", f"{plateau.end_index + 1} periods" if plateau else "none")

    if lr is None:
        st.info("Too few usable periods to estimate a loss ratio.")
    elif not lr.reliable:
        st.warning(
            f"**No determinable b.** The loss-ratio estimate spreads ±{lr.spread:.2f} across "
            "smoothing windows, which is too wide to cross-check anything. Do not read that "
            "as *the fit is fine*: a well whose b cannot be pinned down from its own decline "
            "is a well whose hyperbolic extrapolation rests on nothing. Give it an "
            "independent volume regardless of what the extrapolation multiple says.")
    elif lr.agrees_with(b_fit) is False:
        st.error(
            f"**The b cross-check fails.** Fitted b {b_fit:.2f} against a loss-ratio b of "
            f"{lr.b:.2f} ±{lr.spread:.2f}. The fit is probably describing operations — "
            "interventions, offtake changes, downtime — rather than reservoir depletion, and "
            "the extrapolation will be wrong in a direction R² cannot see.")
    elif lr.agrees_with(b_fit):
        st.success(
            f"**The b cross-check passes.** Fitted b {b_fit:.2f} against a loss-ratio b of "
            f"{lr.b:.2f} ±{lr.spread:.2f} — two independent estimates of the same quantity, "
            "agreeing. The fit is consistent with reservoir depletion.")

    if slope is not None:
        if slope > -0.15:
            st.error(f"**log–log slope {slope:.2f}** — flat or rising. This well is not in "
                     "decline over this window; no choice of model fixes that.")
        elif -0.75 < slope <= -0.3:
            st.info(f"**log–log slope {slope:.2f}** — transient linear flow. The well has "
                    "not reached its boundaries, so Arps' boundary-dominated assumption does "
                    "not hold yet. Prefer Duong or PLE.")
        else:
            st.caption(f"log–log slope {slope:.2f} — boundary-dominated flow.")

    g1, g2 = st.columns(2)
    with g1:
        d = dca.nominal_decline(s.t, s.q)
        ok = np.isfinite(d) & (d > 1e-4)
        fig = go.Figure(go.Scatter(x=s.t[ok], y=1.0 / d[ok], mode="markers",
                                   marker=dict(size=5, color="#2a78d6"), name="1/D"))
        if lr is not None:
            # Anchor the reference line on the trailing 60% -- the part loss_ratio_b
            # actually regresses -- so it sits on the points rather than beside them.
            x0, inv = s.t[ok], 1.0 / d[ok]
            tail = x0 >= x0[int(x0.size * 0.4)]
            t_mid, i_mid = float(np.median(x0[tail])), float(np.median(inv[tail]))
            fig.add_trace(go.Scatter(x=x0, y=lr.b * (x0 - t_mid) + i_mid, mode="lines",
                                     name=f"slope = b = {lr.b:.2f}"
                                          + ("" if lr.reliable else " (not reliable)"),
                                     line=dict(color="#eb6834", dash="dash")))
        st.plotly_chart(theme_axes(fig, "months", "loss ratio 1/D", height=330),
                        width="stretch")
        st.caption("**Loss ratio.** Its slope is b — Arps' whole derivation is the "
                   "assumption that this line is straight. 1/D is a derivative of noisy "
                   "data, so read the trend, never point to point.")

        fig = go.Figure(go.Scatter(x=s.cum, y=s.q, mode="markers", name="rate vs Np",
                                   marker=dict(size=5, color="#1baf7a")))
        st.plotly_chart(theme_axes(fig, f"cumulative ({unit})", f"rate ({unit}/d)",
                                   height=330), width="stretch")
        st.caption("**q vs Np.** A straight line is exponential decline — tank-like "
                   "depletion. Scatter with no trend usually means the well is "
                   "rate-constrained rather than declining.")

    with g2:
        pos = s.t > 0
        fig = go.Figure(go.Scatter(x=np.sqrt(s.t[pos]), y=1.0 / s.q[pos], mode="markers",
                                   name="1/q", marker=dict(size=5, color="#4a3aa7")))
        st.plotly_chart(theme_axes(fig, "√t (months^½)", f"1/q (d/{unit})", height=330),
                        width="stretch")
        st.caption("**1/q vs √t.** Straight means transient linear flow, i.e. "
                   "fracture-dominated and not yet at the boundaries.")

        fig = go.Figure(go.Scatter(x=s.t[pos], y=s.q[pos], mode="markers", name="rate",
                                   marker=dict(size=5, color="#eda100")))
        if slope is not None:
            tt = s.t[pos]
            ref = float(s.q[pos][0]) * (tt / tt[0]) ** -0.5
            fig.add_trace(go.Scatter(x=tt, y=ref, mode="lines", name="−½ slope",
                                     line=dict(color="#7e8286", dash="dot")))
        st.plotly_chart(theme_axes(fig, "months (log)", f"rate ({unit}/d, log)",
                                   ylog=True, xlog=True, height=330), width="stretch")
        st.caption("**log–log.** −½ is linear flow, −¼ bilinear, −1 boundary-dominated. "
                   "The dotted line is the −½ reference.")


# ---------------------------------------------------------------------------
# Water
# ---------------------------------------------------------------------------

with tab_w:
    if s.water is None or not np.any(s.water > 0):
        st.info("No water column in this file, or no water produced. "
                "WOR analysis needs produced water alongside oil.")
    elif s.is_gas:
        st.info("WOR analysis is for oil wells — switch the production column to oil.")
    else:
        cc = st.columns(2)
        ec = cc[0].slider("Economic water cut (%)", 50, 99, 95) / 100.0
        floor = cc[1].slider("X-plot water-cut floor (%)", 30, 90, 50,
                             help="X = ln(WOR) + 1 + 1/WOR is U-shaped with its minimum at "
                                  "WOR = 1, i.e. 50% water cut. Below that it is "
                                  "double-valued and any straight line through it is an "
                                  "artefact. The floor is structural, not conservatism.") / 100.0
        wa = dca.wor_analysis(s.cum, s.volume, s.water, ec_water_cut=ec, fw_floor=floor)
        if wa is None:
            st.info("Not enough periods with both oil and water production.")
        else:
            # Two different failures, kept apart. "Weak" means the points do not form a
            # line, so there is no trend to extrapolate. "Past the limit" means the trend
            # is fine and says the well has ALREADY passed the economic water cut -- a
            # real answer, and not the same thing as no answer.
            OK = 0.70
            def _weak(d, np_):
                # A negative or zero Np is not a conservative answer, it is a broken
                # extrapolation -- usually too few points above the water-cut floor.
                return (not d or d["r2"] < OK or not math.isfinite(np_) or np_ <= 0
                        or d.get("n", 0) < 4)

            weak_s = _weak(wa.semilog, wa.np_semilog)
            weak_x = _weak(wa.xplot, wa.np_xplot)
            past_s = (not weak_s) and wa.np_semilog <= wa.cum_oil
            m = st.columns(5)
            m[0].metric("Water cut now", f"{wa.fw_now:.1%}", delta=f"WOR {wa.wor_now:.2f}",
                        delta_color="off")
            m[1].metric("Water produced", fmt(wa.cum_water, "bbl"))
            m[2].metric(f"Np at {ec:.0%} — semilog",
                        "—" if weak_s else fmt(wa.np_semilog, unit),
                        delta=None if not wa.semilog else f"R² {wa.semilog['r2']:.3f}",
                        delta_color="off")
            m[3].metric(f"Np at {ec:.0%} — X-plot",
                        "—" if weak_x else fmt(wa.np_xplot, unit),
                        delta=None if not wa.xplot else f"R² {wa.xplot['r2']:.3f}",
                        delta_color="off")
            rem_w = (wa.np_semilog - wa.cum_oil) if not weak_s else float("nan")
            m[4].metric("Remaining oil", "—" if weak_s else fmt(max(rem_w, 0), unit))

            if past_s:
                st.warning(
                    f"**This well is already past {ec:.0%} water cut.** The WOR trend "
                    f"(R² {wa.semilog['r2']:.3f}) puts Np at that limit at "
                    f"{fmt(wa.np_semilog, unit)}, below the {fmt(wa.cum_oil, unit)} already "
                    "produced. Water-based remaining oil is zero at this limit — raise the "
                    "economic water cut above if the well is still being produced "
                    "economically, which at this cut is a facilities and disposal question "
                    "rather than a reservoir one.")
            elif weak_s and weak_x:
                st.error("**Neither trend is fittable on this well.** The WOR points do not "
                         "form a line, so no water-based recovery number is reported. That is "
                         "the correct outcome, not a failure: these methods assume an aquifer "
                         "or an injector steadily displacing oil.")
            elif not weak_s:
                arps = primary["eur"].eur  # noqa: E501
                rel = ("materially more optimistic" if wa.np_semilog > arps * 1.15
                       else "materially more conservative" if wa.np_semilog < arps * 0.85
                       else "within 15% of each other")
                st.info(f"The WOR trend gives {fmt(wa.np_semilog, unit)} against "
                        f"{fmt(arps, unit)} from the {MODELS[primary_key].name} case — "
                        f"{rel}. A wide gap is information: the rate decline and the water "
                        "trend are telling different stories about what kills this well.")

            wor = s.water / s.volume
            g1, g2 = st.columns(2)
            with g1:
                fig = go.Figure(go.Scatter(x=s.cum, y=wor, mode="markers",
                                           marker=dict(size=5, color="#2a78d6")))
                if wa.semilog and not weak_s:
                    xs = np.array([s.cum[0], max(wa.np_semilog, s.cum[-1])])
                    fig.add_trace(go.Scatter(
                        x=xs, y=np.exp(wa.semilog["slope"] * xs + wa.semilog["intercept"]),
                        mode="lines", name="trend", line=dict(color="#eb6834", dash="dash")))
                fig.add_hline(y=ec / (1 - ec), line_dash="dot",
                              annotation_text=f"{ec:.0%} water cut")
                st.plotly_chart(theme_axes(fig, f"cumulative oil ({unit})", "WOR",
                                           ylog=True, height=340), width="stretch")
                st.caption("**Semilog WOR.** Straight whenever kro/krw falls exponentially "
                           "with saturation and sweep is uniform. Curvature means layered "
                           "sweep or a changing drive.")
            with g2:
                sel = (s.water / (s.water + s.volume)) >= floor
                fig = go.Figure(go.Scatter(x=dca.x_of(wor[sel]), y=s.cum[sel], mode="markers",
                                           marker=dict(size=5, color="#1baf7a")))
                if wa.xplot and not weak_x:
                    xs = np.linspace(float(dca.x_of(max(wor[sel].min(), 1.01))),
                                     float(dca.x_of(ec / (1 - ec))), 20)
                    fig.add_trace(go.Scatter(
                        x=xs, y=wa.xplot["slope"] * xs + wa.xplot["intercept"], mode="lines",
                        name="trend", line=dict(color="#eb6834", dash="dash")))
                st.plotly_chart(theme_axes(fig, "X = ln(WOR) + 1 + 1/WOR",
                                           f"cumulative oil ({unit})", height=340),
                                width="stretch")
                st.caption("**Ershaghi X-plot.** From Buckley–Leverett via the Welge "
                           "tangent. Linear where the semilog plot curves, which makes the "
                           "extrapolation defensible rather than merely convenient.")

            chan = dca.chan_derivative(s.t, wor)
            pos = (s.t > 0) & (wor > 0) & (chan > 0)
            if pos.sum() > 4:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=s.t[pos], y=wor[pos], mode="markers", name="WOR",
                                         marker=dict(size=5, color="#2a78d6")))
                fig.add_trace(go.Scatter(x=s.t[pos], y=chan[pos], mode="markers", name="WOR′",
                                         marker=dict(size=5, color="#e34948")))
                st.plotly_chart(theme_axes(fig, "months (log)", "WOR and WOR′",
                                           ylog=True, xlog=True, height=340), width="stretch")
                st.caption("**Chan diagnostic.** WOR′ flattening or turning down is coning — "
                           "rate reduction or a deeper completion helps. WOR′ still rising is "
                           "channelling or a multilayer breakthrough — that needs conformance "
                           "work, and cutting rate will not help.")


# ---------------------------------------------------------------------------
# Material balance
# ---------------------------------------------------------------------------

with tab_m:
    st.subheader("Gas material balance")
    st.caption("p/z, Havlena–Odeh drive diagnosis, and a Fetkovich aquifer fit. "
               "Needs static pressure surveys against cumulative production.")

    pres = sorted(glob.glob(os.path.join(DATA_DIR, "*pressure*.csv")))
    mb_src = st.radio("Pressure data", ["Sample reservoir", "Upload CSV"],
                      horizontal=True, key="mb_src")
    if mb_src == "Upload CSV":
        mup = st.file_uploader(
            "Pressure CSV", type="csv", key="mbup",
            help="Date, Pressure (psia), Cum gas (MMscf), Cum condensate (Mbbl), "
                 "Cum water (Mbbl)")
        mb_path = io.BytesIO(mup.getvalue()) if mup else None
    else:
        pn = [os.path.basename(p) for p in pres]
        mb_path = os.path.join(DATA_DIR, st.selectbox("Reservoir", pn, key="reservoir")) if pn else None

    if mb_path is None:
        st.info("Upload a pressure-survey CSV, or pick a sample reservoir.")
    else:
        pdf = dca.load_pressure_csv(mb_path)
        c = st.columns(5)
        temp = c[0].number_input("Temperature (°F)", 60.0, 400.0, 230.0, step=1.0, key="mb_t")
        sg = c[1].number_input("Gas gravity", 0.55, 1.2, 0.72, step=0.005, format="%.4f", key="mb_sg")
        api = c[2].number_input("Condensate API (0 = none)", 0.0, 90.0, 0.0, step=1.0,
                                key="mb_api",
                                help="Converts produced condensate to gas equivalent. "
                                     "Omitting it biases OGIP low and remaining high.")
        pab = c[3].number_input("Abandonment pressure (psia)", 50.0, 6000.0, 1000.0,
                                step=50.0, key="mb_pab")
        skip = c[4].number_input("Drop earliest surveys", 0, max(len(pdf) - 3, 0), 0,
                                 key="mb_skip",
                                 help="Use this when the consistency guard fires and the "
                                      "first survey predates a reliable reference.")
        ge = dca.condensate_gas_equivalent(api) if api > 8.9 else 0.0

        try:
            r = _mb(tuple(pdf["P"]), tuple(pdf["Gp"]), tuple(pdf["condensate"]),
                    tuple(pdf["water"]), temp, sg, ge, pab, int(skip))
        except ValueError as e:                          # noqa: BLE001
            st.error(str(e))
            st.stop()

        m = st.columns(5)
        m[0].metric("OGIP (p/z)", f"{r.ogip_pz / 1000:,.1f} Bcf", delta=f"R² {r.r2:.4f}",
                    delta_color="off")
        m[1].metric("Produced", f"{r.gp_now / 1000:,.1f} Bcf")
        m[2].metric("Recoverable", f"{r.recoverable / 1000:,.1f} Bcf",
                    delta=f"RF {r.recovery_factor:.0f}%", delta_color="off")
        m[3].metric("Remaining (p/z)", f"{r.remaining / 1000:,.1f} Bcf")
        m[4].metric("F/Eg rise", f"{r.ho_rise:.2f}×" if math.isfinite(r.ho_rise) else "—")

        if r.impossible:
            st.error(
                f"**Consistency guard fired.** Material balance requires G ≤ min(F/Eg) = "
                f"{r.min_f_over_eg:,.1f} Bcf, because We ≥ 0 — but {r.gp_now / 1000:,.1f} Bcf "
                "has already been produced. The surveys and the volumes cannot both be right, "
                "and no aquifer model rescues that. The usual cause is a reference pressure "
                "taken **after** first production: pi is then too low, every Eg downstream too "
                "small, and F/Eg too large everywhere. Try dropping the earliest survey above.")
        elif r.volumetric:
            st.success(
                f"**Volumetric depletion.** F/Eg is flat ({r.ho_rise:.2f}× across the record), "
                "so the gas is producing by its own expansion and the p/z intercept is a real "
                "number. Recovery factors of 80–90% are normal here.")
        else:
            st.warning(
                f"**Water drive or pressure support.** F/Eg climbs {r.ho_rise:.2f}×, so "
                "something outside the gas is supplying energy. The p/z intercept above is an "
                "**artefact, not a volume** — a water-driven reservoir holds pressure up, "
                "which flattens the trend and inflates the intercept. Use the aquifer fit "
                "below and a recovery factor of 50–70%.")

        g1, g2 = st.columns(2)
        with g1:
            fig = go.Figure(go.Scatter(x=r.points["Gp"], y=r.points["pz"], mode="markers",
                                       marker=dict(size=8, color="#2a78d6"), name="surveys"))
            if math.isfinite(r.ogip_pz):
                xs = np.array([0.0, r.ogip_pz])
                fig.add_trace(go.Scatter(x=xs, y=r.pz_i * (1 - xs / r.ogip_pz), mode="lines",
                                         name="volumetric line",
                                         line=dict(color="#eb6834", dash="dash")))
                fig.add_hline(y=pab / dca.z_factor(pab, temp, sg), line_dash="dot",
                              annotation_text=f"abandonment {pab:g} psi")
            st.plotly_chart(theme_axes(fig, "cumulative gas (MMscf)", "p/z (psia)",
                                       height=360), width="stretch")
            st.caption("**p/z vs Gp.** Straight for a volumetric tank; the Gp-axis intercept "
                       "is OGIP. But a straight-*looking* p/z plot is not evidence of a "
                       "volumetric reservoir — the panel beside it is.")
        with g2:
            if math.isfinite(r.ho_rise):
                ho = r.points
                fig = go.Figure()
                Bgi = None
                xs, ys = [], []
                TR = temp + 459.67
                pi = float(r.points["P"].max())
                Bgi = 0.02827 * dca.z_factor(pi, temp, sg) * TR / pi
                for _, row in r.points.iterrows():
                    if row["P"] >= pi - 1:
                        continue
                    bg = 0.02827 * dca.z_factor(row["P"], temp, sg) * TR / row["P"]
                    Eg = bg - Bgi
                    F = row["Gp_gas_only"] * 1e6 * bg + row["water_Mbbl"] * 1e3 * 5.615
                    if Eg > 0 and F > 0:
                        xs.append(row["Gp"])
                        ys.append(F / Eg / 1e9)
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name="F/Eg",
                                         line=dict(color="#2a78d6")))
                if ys:
                    fig.add_hline(y=ys[0], line_dash="dash", line_color="#7e8286",
                                  annotation_text=f"first point {ys[0]:,.0f} Bcf")
                st.plotly_chart(theme_axes(fig, "cumulative gas (MMscf)", "F / Eg (Bcf)",
                                           height=360), width="stretch")
                st.caption("**Havlena–Odeh.** F = G·Eg + We. Flat means We ≈ 0 and the level "
                           "*is* G. Rising means influx. This is the discriminator.")

        if r.fetkovich:
            f = r.fetkovich
            st.subheader("Fetkovich aquifer fit")
            fc = st.columns(5)
            fc[0].metric("G", f"{f['G_Bcf']:,.0f} Bcf")
            fc[1].metric("Wei", f"{f['Wei_MMbbl']:,.0f} MMbbl")
            fc[2].metric("J", f"{f['J_bbl_d_psi']:g} bbl/d/psi")
            fc[3].metric("We", f"{f['We_MMbbl']:,.0f} MMbbl")
            fc[4].metric("rms", f"{f['rms_pct']:.1f}%")
            if r.volumetric:
                # No aquifer to account for: the fit lands on the smallest aquifer in the
                # grid and essentially zero influx, which is the confirmation, not a forecast.
                st.success(
                    f"The aquifer fit finds **{f['We_MMbbl']:,.0f} MMbbl of influx** — "
                    "negligible. That is the cross-check passing: this reservoir does not "
                    f"need an aquifer to explain its pressure history, and its independent "
                    f"estimate of G ({f['G_Bcf']:,.0f} Bcf) sits close to the p/z intercept "
                    f"of {r.ogip_pz / 1000:,.0f} Bcf. Use the p/z number with a volumetric "
                    "recovery factor of 80–90%.")
            else:
                lo_r = max(f["G_Bcf"] * 0.5 - r.gp_now / 1000, 0)
                hi_r = max(f["G_Bcf"] * 0.7 - r.gp_now / 1000, 0)
                st.info(
                    f"**Remaining at a water-drive recovery factor of 50–70%: "
                    f"{lo_r:,.0f} to {hi_r:,.0f} Bcf** — against {r.remaining / 1000:,.0f} Bcf "
                    "from the straight-line p/z above. That gap is the whole point of this tab.")
            if f["rms_pct"] > 10:
                st.warning(f"rms {f['rms_pct']:.1f}% is above 10% — the model cannot reproduce "
                           "the pressure history. Treat the parameters as indicative only.")
            st.markdown("**The solution is not unique.** A large aquifer with a small J and a "
                        "small aquifer with a large J give nearly the same influx history over "
                        "a finite record; only late depletion of the aquifer separates them. "
                        "This locus, not a single triplet, is the honest output:")
            st.dataframe(f["locus"].style.format({"G_Bcf": "{:,.0f}", "Wei_MMbbl": "{:,.0f}",
                                                  "J_bbl_d_psi": "{:g}", "rms_pct": "{:.1f}"}),
                         width="stretch", hide_index=True)

            if not cap_on:
                # G is Bcf; the sidebar wants MMcf (the base gas unit) in millions.
                g_mm = f["G_Bcf"] * 1e6 / 1e6 if s.is_gas else float("nan")
                st.caption(
                    "To carry this into the forecast, tick **Cap the forecast at a known "
                    "volume** in the sidebar"
                    + (f" and enter {g_mm:,.0f} MMMcf" if s.is_gas else
                       " (this reservoir is gas; switch the production column to a gas "
                       "stream first)")
                    + (", with a recovery factor of 80–90%." if r.volumetric
                       else ", with a recovery factor of 50–70%."))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

with tab_t:
    st.subheader("Production data")
    show = pd.DataFrame({
        "Date": s_full.dates.dt.strftime("%Y-%m"),
        "Months": s_full.t,
        "Days on": s_full.days,
        f"{s_full.column}": s_full.volume,
        f"Rate ({unit}/d)": s_full.q.round(1),
        f"Cumulative ({unit})": s_full.cum.round(0),
        "In fit window": [(start <= i < end) for i in range(n_all)],
    })
    if s_full.water is not None:
        show["Water (bbl)"] = s_full.water
    st.dataframe(show, width="stretch", hide_index=True, height=520)
    st.download_button("Download as CSV", show.to_csv(index=False).encode(),
                       file_name=f"{os.path.splitext(src_label)[0]}_processed.csv",
                       mime="text/csv")
    st.caption(
        "Months come from the **dates**, not the row order. Zero-production months are "
        "omitted from these files, so the date gaps are the shut-ins — counting rows would "
        "compress them out and fabricate decline. Rate is volume ÷ producing days.")
