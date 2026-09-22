"""
================================================================================
 oil_charts.py -- interactive Plotly charts for the oil DCA app
================================================================================

The chart contract is the gas module's, unchanged, because the two apps should
read as one system:

  * Categorical colours come from a fixed slot order, never cycled. Slot 1 is
    always measured oil, slot 2 always the model, slot 3 gas, slot 4 water. A
    series keeps its colour when others are toggled off.
  * No chart has two y-axes. Oil, gas and water are different measures on
    different scales; overlaying them on twin axes lets you manufacture any
    correlation you like by choosing the scales.
  * Light and dark are both selected, not an automatic inversion.

Four things every chart here does, which a plot of the same numbers need not:

  EXCLUDED DATA IS SHADED, NOT MARKED. A thin line saying "fit starts" tells
  you a boundary exists; a shaded band tells you how much of the record is on
  the far side of it. On a well with three years of plateau that is the
  difference between a detail and the main fact about the fit.

  LIMITS ARE DRAWN AND LABELLED WITH THEIR VALUE. The economic rate, the
  water-cut limit, the oil in place: these are what end the well, and a
  forecast read against them is a different picture from a forecast read
  against nothing.

  HOVER CARRIES THE DATE AND THE COMPANION STREAMS. A point on an oil-rate
  chart is a month, and what matters about that month is usually the GOR or
  the water cut next to it. Reading the rate alone and then hunting for the
  date in a table is how a reader stops checking.

  MARKERS ARE OUTLINED IN THE SURFACE COLOUR. Overlapping points on a dark
  background merge into a single blob without it; the outline is what keeps a
  cluster readable as a cluster.

Every public function takes an `OilWellResult` (or plain arrays) plus `theme`,
and returns a `plotly.graph_objects.Figure`.
================================================================================
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dca_charts import THEMES, FONT, palette, rgba, _layout, _empty  # noqa: F401
from oil_dca import DAYS_PER_YEAR, SCF_PER_MSCF, STB_PER_MSTB


# ------------------------------------------------------------------------------
# Shared furniture
# ------------------------------------------------------------------------------

def _dates(res) -> np.ndarray:
    """Month labels for hover, aligned to the QC'd record."""
    d = res.data.df
    if "date" in d.columns:
        return d["date"].dt.strftime("%b %Y").to_numpy()
    return np.array([""] * len(d))


def _marker(colour: str, theme: str, size: float = 7.0,
            opacity: float = 1.0) -> dict:
    c = palette(theme)
    return dict(size=size, color=colour, opacity=opacity,
                line=dict(width=1.2, color=c["surface"]))


def _shade_excluded(fig, res, theme: str, x_is_years: bool = True,
                    row: Optional[int] = None) -> None:
    """Shade the part of the record the decline fit was not shown."""
    c = palette(theme)
    t0 = res.limits.get("fit_start_days", float("nan"))
    if not (np.isfinite(t0) and t0 > 0):
        return
    t_start = float(res.data.t[0])
    if t0 <= t_start:
        return
    kw = dict(fillcolor=c["band"], line_width=0, layer="below",
              annotation_text="excluded from the fit",
              annotation_position="top left",
              annotation_font=dict(size=10, color=c["muted"]))
    if row is not None:
        kw["row"] = row
        kw["col"] = 1
        kw.pop("annotation_text", None)
        kw.pop("annotation_position", None)
        kw.pop("annotation_font", None)
    if x_is_years:
        fig.add_vrect(x0=t_start / DAYS_PER_YEAR, x1=t0 / DAYS_PER_YEAR, **kw)
    else:
        np_at = float(np.interp(t0, res.data.t, res.data.Np))
        fig.add_vrect(x0=float(res.data.Np[0]) / 1.0e6, x1=np_at / 1.0e6, **kw)


def _set_log_range(fig, series: Sequence[np.ndarray],
                   limits: Sequence[float] = (), pad_decades: float = 0.12,
                   max_decades: float = 6.0) -> None:
    """Fix the log y range from the data, rather than leaving it to autorange.

    Two of these charts autoranged to 10^20 on data spanning 248 to 5,440
    STB/d - eighteen empty decades with the well squashed into the bottom
    line of the plot. The trigger was an economic-limit line sitting two
    decades below the data; the chart whose limit line was scoped to a
    subplot row was unaffected, which is what makes it plotly's autorange
    rather than the numbers.

    Chasing that is not worth it when the range is better set explicitly in
    any case. A rate chart should be framed by the rates, with the limit line
    included only far enough to be visible - a limit six decades below the
    data is a fact about the limit, not a reason to draw six empty decades.
    """
    vals: List[float] = []
    for arr in series:
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a) & (a > 0)]
        if a.size:
            vals.extend([float(a.min()), float(a.max())])
    if not vals:
        return
    lo, hi = min(vals), max(vals)
    for L in limits:
        if np.isfinite(L) and L > 0:
            lo, hi = min(lo, float(L)), max(hi, float(L))
    lo_d = math.log10(lo) - pad_decades
    hi_d = math.log10(hi) + pad_decades
    # Never draw more decades than the eye can use. Clamped from the TOP so
    # the recent, low rates - the part a forecast is read from - keep their
    # resolution.
    if hi_d - lo_d > max_decades:
        lo_d = hi_d - max_decades
    fig.update_yaxes(range=[lo_d, hi_d])
    # The integer tick format inherited from the shared layout is right for
    # rates in the thousands and wrong for anything below 1: on the Chan plot,
    # where WOR' runs from 1e-6 to 1, every decade label rounded to "0" and
    # the axis shipped with six identical ticks.
    if lo < 10.0:
        fig.update_yaxes(tickformat="~g", exponentformat="power")


def _set_linear_range(fig, x_vals: Sequence[np.ndarray] = (),
                     y_vals: Sequence[np.ndarray] = (),
                     pad: float = 0.06) -> None:
    """Fix a linear axis range from the data, for the same reason as the log.

    On the Monte Carlo scatter, 727 of 800 realisations shared one life
    (42.75 years, the horizon) and plotly autoranged the y axis to 41 - so
    ninety-one per cent of the points were clipped out of a chart whose whole
    purpose is to show where the mass of the realisations sits. Setting the
    range from the numbers costs three lines and cannot do that.
    """
    def _rng(vals):
        allv: List[float] = []
        for a in vals:
            arr = np.asarray(a, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                allv.extend([float(arr.min()), float(arr.max())])
        if not allv:
            return None
        lo, hi = min(allv), max(allv)
        span = hi - lo
        if span <= 0:                      # every point identical
            span = max(abs(hi), 1.0) * 0.05
        return [lo - pad * span, hi + pad * span]

    rx, ry = _rng(x_vals), _rng(y_vals)
    if rx:
        fig.update_xaxes(range=rx)
    if ry:
        fig.update_yaxes(range=ry)


def _limit_line(fig, y: float, text: str, theme: str,
                position: str = "top right") -> None:
    """A limit line, labelled with its value.

    The label sits ABOVE the line by default. Below it, on a limit near the
    bottom of the axis, the text was clipped by the plot edge and the chart
    shipped with an unexplained red dotted line across it.
    """
    c = palette(theme)
    if not (np.isfinite(y) and y > 0):
        return
    fig.add_hline(y=float(y), line=dict(color=c["critical"], width=1.4,
                                        dash="dot"),
                  annotation_text=text, annotation_position=position,
                  annotation_font=dict(size=10, color=c["critical"]))


# ------------------------------------------------------------------------------
# Rates
# ------------------------------------------------------------------------------

def chart_rate_time(res, theme: str = "light", log_y: bool = True,
                    height: int = 440) -> go.Figure:
    """Oil rate: history, the fitted model, the forecast, and what ends it."""
    c = palette(theme)
    d, fc = res.data, res.forecast.table
    fig = go.Figure()
    _shade_excluded(fig, res, theme)

    dates = _dates(res)
    fig.add_trace(go.Scatter(
        x=d.t / DAYS_PER_YEAR, y=d.q_oil, mode="markers", name="Oil",
        marker=_marker(c["series"][0], theme),
        customdata=np.stack([dates, d.gor, 100.0 * d.water_cut], axis=-1),
        hovertemplate=("<b>%{customdata[0]}</b><br>"
                       "Oil %{y:,.0f} STB/d<br>"
                       "GOR %{customdata[1]:,.0f} scf/STB<br>"
                       "Water cut %{customdata[2]:.1f} %<extra></extra>")))

    # The fitted curve is drawn only over the window it was fitted to. Drawn
    # across the plateau as well it would look like a model missing data it
    # was never asked to match, and invite a reader to distrust a fit that is
    # doing exactly what it was told.
    tf = res.best_fit.t_fit
    if tf is not None and len(tf):
        grid = np.linspace(float(np.min(tf)), float(np.max(tf)), 400)
        fig.add_trace(go.Scatter(
            x=grid / DAYS_PER_YEAR, y=res.best_fit.model.rate(grid),
            mode="lines", name=f"{res.best_fit.model_name} fit",
            line=dict(width=2.4, color=c["series"][1]),
            hovertemplate="Fit %{y:,.0f} STB/d<extra></extra>"))

    if len(fc) > 1:
        fig.add_trace(go.Scatter(
            x=fc["t_years"], y=fc["q_oil_stbd"], mode="lines", name="Forecast",
            line=dict(width=2.4, color=c["series"][1], dash="dash"),
            customdata=np.stack([fc["gor_scf_per_stb"],
                                 100.0 * fc["water_cut"]], axis=-1),
            hovertemplate=("Year %{x:.1f}<br>Oil %{y:,.0f} STB/d<br>"
                           "GOR %{customdata[0]:,.0f} scf/STB<br>"
                           "Water cut %{customdata[1]:.1f} %<extra></extra>")))

    q_econ = res.limits.get("q_econ_stbd", float("nan"))
    _limit_line(fig, q_econ, f"economic limit {q_econ:,.0f} STB/d", theme)

    _layout(fig, theme, f"Oil rate: history, fit and forecast -- {res.well}",
            "Time on production (years)", "Oil rate (STB/d)",
            log_y=log_y, height=height)
    if log_y:
        _set_log_range(fig, [d.q_oil, fc["q_oil_stbd"].to_numpy(float)],
                       limits=[q_econ])
    fig.update_layout(hovermode="x unified")
    return fig


STREAMS = {
    "oil":   dict(hist="q_oil", fcst="q_oil_stbd",   slot=0, unit="STB/d",
                  label="Oil rate"),
    "gas":   dict(hist="q_gas", fcst="q_gas_mscfd",  slot=2, unit="Mscf/d",
                  label="Gas rate"),
    "water": dict(hist="q_water", fcst="q_water_stbd", slot=3, unit="STB/d",
                  label="Water rate"),
}


def chart_stream(res, stream: str = "oil", theme: str = "light",
                 height: int = 340) -> go.Figure:
    """One produced stream: history and forecast, on its own axes.

    These were one figure with three stacked panels. Plotly puts a subplot's
    title above its plotting area and the shared x-axis title below it, so at
    three rows each panel's title collided with the axis label of the panel
    above - "Time on production (years)" printed three times, twice on top of
    something else. Three separate figures also let each stream keep its own
    height and be placed where it is wanted, instead of all three being
    hostage to one layout.
    """
    c = palette(theme)
    spec = STREAMS.get(stream, STREAMS["oil"])
    d, tb = res.data, res.forecast.table
    colour = c["series"][spec["slot"]]
    hist = getattr(d, spec["hist"])
    dates = _dates(res)

    fig = go.Figure()
    # Only the oil panel carries the decline-fit band: the fit is fitted to
    # the oil rate, and marking the gas or water chart "excluded from the
    # fit" would say those months were dropped from a fit they were never
    # part of.
    if stream == "oil":
        _shade_excluded(fig, res, theme)

    fig.add_trace(go.Scatter(
        x=d.t / DAYS_PER_YEAR, y=hist, mode="markers", name="Measured",
        marker=_marker(colour, theme, size=6),
        customdata=dates,
        hovertemplate=("<b>%{customdata}</b><br>%{y:,.0f} " + spec["unit"]
                       + "<extra></extra>")))
    if len(tb) > 1:
        fig.add_trace(go.Scatter(
            x=tb["t_years"], y=tb[spec["fcst"]], mode="lines", name="Forecast",
            line=dict(width=2.4, color=colour, dash="dash"),
            hovertemplate=("Year %{x:.1f}<br>%{y:,.0f} " + spec["unit"]
                           + "<extra></extra>")))

    lim = float("nan")
    if stream == "oil":
        lim = res.limits.get("q_econ_stbd", float("nan"))
        _limit_line(fig, lim, f"economic limit {lim:,.0f} STB/d", theme)
    elif stream == "water":
        lim = res.limits.get("q_water_econ_stbd", float("nan"))
        if np.isfinite(lim) and lim > 0:
            _limit_line(fig, lim, f"handling limit {lim:,.0f} STB/d", theme)

    fig = _layout(fig, theme, f"{spec['label']} -- {res.well}",
                  "Time on production (years)",
                  f"{spec['label']} ({spec['unit']})",
                  log_y=True, height=height)
    vals = [hist] + ([tb[spec["fcst"]].to_numpy(float)] if len(tb) > 1 else [])
    _set_log_range(fig, vals, limits=[lim])
    fig.update_layout(hovermode="x unified")
    return fig


def chart_rate_cum(res, theme: str = "light", height: int = 400) -> go.Figure:
    """Oil rate against cumulative oil - the plot that shows the EUR."""
    c = palette(theme)
    d, tb = res.data, res.forecast.table
    fig = go.Figure()
    _shade_excluded(fig, res, theme, x_is_years=False)
    dates = _dates(res)

    fig.add_trace(go.Scatter(
        x=d.Np / 1.0e6, y=d.q_oil, mode="markers", name="Oil",
        marker=_marker(c["series"][0], theme),
        customdata=np.stack([dates, d.t / DAYS_PER_YEAR], axis=-1),
        hovertemplate=("<b>%{customdata[0]}</b> (%{customdata[1]:.1f} yr)<br>"
                       "Np %{x:,.2f} MMstb<br>Oil %{y:,.0f} STB/d"
                       "<extra></extra>")))
    if len(tb) > 1:
        fig.add_trace(go.Scatter(
            x=tb["Np_mstb"] / 1.0e3, y=tb["q_oil_stbd"], mode="lines",
            name="Forecast",
            line=dict(width=2.4, color=c["series"][1], dash="dash"),
            hovertemplate=("Np %{x:,.2f} MMstb<br>Oil %{y:,.0f} STB/d"
                           "<extra></extra>")))

    # The oil-in-place line is drawn only when it is near enough to the data
    # to be worth the width. On one well the cap sat at 71 MMstb against a
    # forecast reaching 28, so three quarters of the plot was empty space
    # holding one dotted line; the number is more useful as a caption.
    n_cap = res.limits.get("n_ooip_stb", float("nan"))
    x_max = max(float(np.nanmax(d.Np)),
                float(tb["Np_mstb"].max()) * 1.0e3 if len(tb) else 0.0)
    if np.isfinite(n_cap) and n_cap > 0:
        if n_cap <= 1.35 * x_max:
            fig.add_vline(x=n_cap / 1.0e6,
                          line=dict(color=c["critical"], width=1.4,
                                    dash="dot"),
                          annotation_text=f"oil in place "
                                          f"{n_cap / 1e6:,.1f} MMstb",
                          annotation_font=dict(size=10, color=c["critical"]))
        else:
            fig.add_annotation(
                text=f"oil in place {n_cap / 1e6:,.1f} MMstb - off scale to "
                     f"the right",
                showarrow=False, xref="paper", yref="paper", x=0.98, y=0.06,
                xanchor="right",
                font=dict(size=10, color=c["critical"], family=FONT))
            fig.update_xaxes(range=[-0.03 * x_max / 1.0e6,
                                    1.05 * x_max / 1.0e6])
    q_econ = res.limits.get("q_econ_stbd", float("nan"))
    _limit_line(fig, q_econ, f"economic limit {q_econ:,.0f} STB/d", theme)

    fig = _layout(fig, theme,
                  f"Oil rate against cumulative -- {res.well}",
                  "Cumulative oil (MMstb)", "Oil rate (STB/d)",
                  log_y=True, height=height)
    _set_log_range(fig, [d.q_oil, tb["q_oil_stbd"].to_numpy(float)],
                   limits=[q_econ])
    return fig


# ------------------------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------------------------

def chart_gor(res, theme: str = "light", height: int = 420) -> go.Figure:
    """Producing GOR against cumulative oil, with Rsi, the break and the peak."""
    c = palette(theme)
    d, tb = res.data, res.forecast.table
    dates = _dates(res)
    fig = go.Figure()

    g = res.gor_diag
    if g is not None and g.ok and g.broke and g.break_np_stb:
        # Everything left of the break is above the bubble point: the GOR
        # there is Rsi and carries no information about free gas. Shading it
        # says which part of the record the rising limb actually is.
        fig.add_vrect(x0=float(d.Np[0]) / 1.0e6,
                      x1=float(g.break_np_stb) / 1.0e6,
                      fillcolor=c["band"], line_width=0, layer="below",
                      annotation_text="above the bubble point",
                      annotation_position="bottom left",
                      annotation_font=dict(size=10, color=c["muted"]))

    fig.add_trace(go.Scatter(
        x=d.Np / 1.0e6, y=d.gor, mode="markers", name="Producing GOR",
        marker=_marker(c["series"][2], theme),
        customdata=np.stack([dates, d.q_oil], axis=-1),
        hovertemplate=("<b>%{customdata[0]}</b><br>Np %{x:,.2f} MMstb<br>"
                       "GOR %{y:,.0f} scf/STB<br>"
                       "Oil %{customdata[1]:,.0f} STB/d<extra></extra>")))

    if len(tb) > 1:
        fig.add_trace(go.Scatter(
            x=tb["Np_mstb"] / 1.0e3, y=tb["gor_scf_per_stb"], mode="lines",
            name=f"GOR model ({res.gor_model.kind})",
            line=dict(width=2.4, color=c["series"][1], dash="dash"),
            hovertemplate=("Np %{x:,.2f} MMstb<br>GOR %{y:,.0f} scf/STB"
                           "<extra></extra>")))

    fig.add_hline(y=float(res.pvt.rsi),
                  line=dict(color=c["muted"], width=1.4, dash="dash"),
                  annotation_text=f"Rsi {res.pvt.rsi:,.0f} scf/STB",
                  annotation_position="bottom right",
                  annotation_font=dict(size=10, color=c["muted"]))
    if g is not None and g.ok:
        if g.broke and g.break_np_stb:
            fig.add_vline(x=float(g.break_np_stb) / 1.0e6,
                          line=dict(color=c["critical"], width=1.4,
                                    dash="dot"),
                          annotation_text="bubble point",
                          annotation_font=dict(size=10, color=c["critical"]))
        if g.peaked and g.peak_np_stb:
            fig.add_vline(x=float(g.peak_np_stb) / 1.0e6,
                          line=dict(color=c["series"][4], width=1.4,
                                    dash="dot"),
                          annotation_text="GOR peak",
                          annotation_position="top right",
                          annotation_font=dict(size=10, color=c["series"][4]))
    return _layout(fig, theme, f"Producing GOR -- {res.well}",
                   "Cumulative oil (MMstb)", "GOR (scf/STB)", height=height)


def chart_chan(res, theme: str = "light", height: int = 420) -> go.Figure:
    """Chan's WOR and WOR' against time since breakthrough, log-log.

    The abscissa is time SINCE BREAKTHROUGH rather than Chan's total producing
    time. On a well that breaks through late, every WOR rising from zero at
    breakthrough carries a 1/(t - t_bt) factor in its local slope, so on a
    total-time axis it looks like it is flattening whatever the mechanism -
    which is how an ordinary linear water cut once read as coning with a WOR
    slope of +16. The two axes coincide when breakthrough is early, which is
    the case Chan's own field examples come from.
    """
    c = palette(theme)
    d, wd = res.data, res.water_diag
    qo, qw, t = d.q_oil, d.q_water, d.t
    ok = (qo > 0) & (qw > 0) & (t > 0)
    if int(ok.sum()) < 8 or wd is None or not wd.ok:
        return _empty(theme, "not enough water production for a Chan plot",
                      height)
    wor = qw[ok] / qo[ok]
    tt = t[ok]
    lab = _dates(res)[ok]
    t_bt = (float(wd.breakthrough_t_days) if wd.breakthrough_t_days
            else float(tt[0]))
    x = np.maximum(tt - t_bt, 1e-6)
    keep = x > 1e-3
    x, wor, lab = x[keep], wor[keep], lab[keep]
    if len(x) < 6:
        return _empty(theme, "not enough post-breakthrough history", height)

    sm = wor.copy()
    if len(wor) >= 5:
        sm[1:-1] = np.array([np.median(wor[j - 1:j + 2])
                             for j in range(1, len(wor) - 1)])
    dwor = np.gradient(sm, x)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=wor, mode="markers", name="WOR",
        marker=_marker(c["series"][3], theme),
        customdata=lab,
        hovertemplate=("<b>%{customdata}</b><br>%{x:,.0f} d after "
                       "breakthrough<br>WOR %{y:,.3f}<extra></extra>")))
    pos = dwor > 0
    fig.add_trace(go.Scatter(
        x=x[pos], y=dwor[pos], mode="markers", name="WOR'",
        marker=dict(size=6, color=c["series"][4], symbol="diamond",
                    line=dict(width=1.0, color=c["surface"])),
        hovertemplate="%{x:,.0f} d<br>WOR' %{y:.3g} /d<extra></extra>"))

    # The window the slopes were actually fitted over, so a reader can see
    # that the verdict rests on the late part and not on breakthrough itself.
    if wd.n_fitted and wd.n_fitted < len(x):
        fig.add_vrect(x0=float(x[0]), x1=float(x[len(x) - wd.n_fitted]),
                      fillcolor=c["band"], line_width=0, layer="below",
                      annotation_text="transition, not fitted",
                      annotation_position="top left",
                      annotation_font=dict(size=10, color=c["muted"]))

    fig = _layout(fig, theme, f"Chan water diagnostic -- {res.well}",
                  "Days since breakthrough", "WOR and WOR'",
                  log_y=True, height=height)
    _set_log_range(fig, [wor, dwor[pos]])
    # The X axis needs the same treatment as the Y. Left to autorange on a
    # log scale it came back spanning 1 to 10^35 days - the whole record
    # compressed into the first pixel - because WOR' values near zero drag
    # the companion axis with them.
    fig.update_xaxes(type="log", dtick=1, tickformat="~g",
                     exponentformat="none",
                     range=[math.log10(max(float(np.min(x)), 1.0)) - 0.1,
                            math.log10(float(np.max(x))) + 0.1],
                     minor=dict(showgrid=True, gridcolor=c["grid"],
                                gridwidth=1))
    fig.add_annotation(
        text=wd.mechanism.split(" -")[0], showarrow=False, xref="paper",
        yref="paper", x=0.02, y=0.06, xanchor="left",
        font=dict(size=13, color=c["ink"], family=FONT))
    return fig


def chart_pi(res, theme: str = "light", height: int = 400) -> go.Figure:
    """Productivity index over time, with its source labelled."""
    c = palette(theme)
    pi = res.pi_diag
    if pi is None or not pi.ok or pi.t_days is None:
        return _empty(theme, (pi.reason if pi is not None and pi.reason
                              else "no productivity index available"), height)
    t_yr = np.asarray(pi.t_days) / DAYS_PER_YEAR
    vals = np.asarray(pi.pi)
    fig = go.Figure()
    # No "excluded from the fit" band here. That band marks where the DECLINE
    # fit starts, and the productivity index has nothing to do with the
    # decline fit - it is computed on every flowing test in the record.
    # Shading it implied a whole third of these points had been discarded
    # when none of them had.
    fig.add_trace(go.Scatter(
        x=t_yr, y=vals, mode="markers", name="PI",
        marker=_marker(c["series"][0], theme, size=6.5),
        customdata=np.asarray(pi.p_avg),
        hovertemplate=("%{x:.2f} yr<br>PI %{y:,.2f} STB/d/psi<br>"
                       "p_avg %{customdata:,.0f} psia<extra></extra>")))
    if np.isfinite(pi.trend_pct_per_year) and len(t_yr) > 1:
        k = np.log1p(pi.trend_pct_per_year / 100.0)
        anchor = float(np.median(vals))
        t_mid = float(np.median(t_yr))
        fig.add_trace(go.Scatter(
            x=t_yr, y=anchor * np.exp(k * (t_yr - t_mid)), mode="lines",
            name=f"{pi.trend_pct_per_year:+.1f} %/yr",
            line=dict(width=2.4, color=c["series"][1])))
    fig = _layout(fig, theme, f"Productivity index -- {res.well}",
                  "Time on production (years)", "PI (STB/d/psi)",
                  height=height)
    fig.add_annotation(
        text=f"p_avg from {pi.p_avg_source}", showarrow=False, xref="paper",
        yref="paper", x=0.02, y=0.06, xanchor="left",
        font=dict(size=10, color=c["muted"], family=FONT))
    fig.update_layout(hovermode="x unified")
    return fig


def chart_water(res, theme: str = "light", height: int = 400) -> go.Figure:
    """Water cut history and forecast, against the limit that ends the well."""
    c = palette(theme)
    d, tb = res.data, res.forecast.table
    dates = _dates(res)
    fig = go.Figure()
    # The band that matters on a water chart is the DRY period, not the
    # decline-fit window: the WOR model is fitted to wet months only, so the
    # months before breakthrough are the ones excluded here.
    wd = res.water_diag
    if wd is not None and wd.ok and wd.breakthrough_t_days:
        fig.add_vrect(x0=float(d.t[0]) / DAYS_PER_YEAR,
                      x1=float(wd.breakthrough_t_days) / DAYS_PER_YEAR,
                      fillcolor=c["band"], line_width=0, layer="below",
                      annotation_text="before breakthrough",
                      annotation_position="top left",
                      annotation_font=dict(size=10, color=c["muted"]))
    fig.add_trace(go.Scatter(
        x=d.t / DAYS_PER_YEAR, y=100.0 * d.water_cut, mode="markers",
        name="Water cut", marker=_marker(c["series"][3], theme),
        customdata=np.stack([dates, d.q_water, d.q_oil], axis=-1),
        hovertemplate=("<b>%{customdata[0]}</b><br>Water cut %{y:.1f} %<br>"
                       "Water %{customdata[1]:,.0f} STB/d<br>"
                       "Oil %{customdata[2]:,.0f} STB/d<extra></extra>")))
    if len(tb) > 1:
        fig.add_trace(go.Scatter(
            x=tb["t_years"], y=100.0 * tb["water_cut"], mode="lines",
            name=f"Forecast ({res.wor_model.kind})",
            line=dict(width=2.4, color=c["series"][1], dash="dash"),
            hovertemplate="Year %{x:.1f}<br>Water cut %{y:.1f} %"
                          "<extra></extra>"))
    lim = res.limits.get("water_cut_econ", float("nan"))
    if np.isfinite(lim) and lim > 0:
        _limit_line(fig, 100.0 * lim, f"limit {100 * lim:.0f} %", theme,
                    position="top right")
    fig = _layout(fig, theme, f"Water cut -- {res.well}",
                  "Time on production (years)", "Water cut (%)",
                  height=height)
    fig.update_layout(hovermode="x unified")
    return fig


# ------------------------------------------------------------------------------
# Material balance
# ------------------------------------------------------------------------------

def chart_havlena_odeh(res, theme: str = "light",
                       height: int = 420) -> go.Figure:
    """F against Et. A straight line through the origin is a closed tank."""
    c = palette(theme)
    mb = res.matbal
    if mb is None or mb.ho_table is None or not mb.trend_ok:
        return _empty(theme, "no material balance available", height)
    tab = mb.ho_table
    et = tab["Et"].to_numpy(float)
    f = tab["F_rb"].to_numpy(float)
    pr = (tab["p"].to_numpy(float) if "p" in tab
          else np.full(len(et), np.nan))
    npv = (tab["Np_stb"].to_numpy(float) if "Np_stb" in tab
           else np.full(len(et), np.nan))
    fig = go.Figure()

    if np.isfinite(mb.n_ooip_stb) and len(et):
        xs = np.linspace(0.0, float(np.nanmax(et)) * 1.05, 50)
        fig.add_trace(go.Scatter(
            x=xs, y=mb.n_ooip_stb * xs / 1.0e6, mode="lines",
            name=(f"N = {mb.n_ooip_mstb / 1e3:,.1f} MMstb"
                  if mb.n_determined else
                  "line through origin - N NOT determined"),
            line=(dict(width=2.4, color=c["series"][1]) if mb.n_determined
                  else dict(width=1.4, color=c["muted"], dash="dash")),
            hoverinfo="skip"))
    if np.isfinite(mb.n_ceiling_stb) and len(et):
        xs = np.linspace(0.0, float(np.nanmax(et)) * 1.05, 50)
        fig.add_trace(go.Scatter(
            x=xs, y=mb.n_ceiling_stb * xs / 1.0e6, mode="lines",
            name=f"ceiling min(F/Et) = {mb.n_ceiling_stb / 1e6:,.1f} MMstb",
            line=dict(width=1.6, color=c["critical"], dash="dot"),
            hoverinfo="skip"))

    fig.add_trace(go.Scatter(
        x=et, y=f / 1.0e6, mode="markers", name="Surveys",
        marker=_marker(c["series"][0], theme, size=9),
        customdata=np.stack([pr, npv / 1.0e6], axis=-1),
        hovertemplate=("p %{customdata[0]:,.0f} psia<br>"
                       "Np %{customdata[1]:,.2f} MMstb<br>"
                       "Et %{x:.5f} rb/STB<br>F %{y:,.2f} MMrb"
                       "<extra></extra>")))
    return _layout(fig, theme, f"Havlena-Odeh -- {res.well}",
                   "Et = Eo + m Eg + Efw (rb/STB)", "F (MMrb)", height=height)


def chart_apparent_n(res, theme: str = "light", height: int = 380) -> go.Figure:
    """Apparent N survey by survey. Flat is a closed tank; rising is influx."""
    c = palette(theme)
    mb = res.matbal
    if mb is None or mb.ho_table is None or not mb.trend_ok:
        return _empty(theme, "no material balance available", height)
    tab = mb.ho_table
    use = tab["apparent_N_usable"].to_numpy(bool)
    x = tab["Np_stb"].to_numpy(float)[use] / 1.0e6
    y = tab["apparent_N_stb"].to_numpy(float)[use] / 1.0e6
    pr = (tab["p"].to_numpy(float)[use] if "p" in tab
          else np.full(len(x), np.nan))
    if not len(x):
        return _empty(theme, "no survey has enough depletion to read F/Et",
                      height)
    fig = go.Figure()

    # The band the drive verdict is read off. A closed tank keeps apparent N
    # inside it; influx walks it out of the top.
    if np.isfinite(mb.apparent_n_min_stb):
        lo = mb.apparent_n_min_stb / 1.0e6
        fig.add_hrect(y0=lo, y1=lo * 1.10, fillcolor=c["band"], line_width=0,
                      layer="below", annotation_text="within 10 % of min",
                      annotation_position="top left",
                      annotation_font=dict(size=10, color=c["muted"]))

    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines+markers", name="apparent N = F/Et",
        marker=_marker(c["series"][0], theme, size=8),
        line=dict(width=1.6, color=rgba(c["series"][0], 0.5)),
        customdata=pr,
        hovertemplate=("Np %{x:,.2f} MMstb<br>p %{customdata:,.0f} psia<br>"
                       "F/Et %{y:,.1f} MMstb<extra></extra>")))
    if np.isfinite(mb.n_ooip_stb):
        det = mb.n_determined
        col = c["series"][1] if det else c["muted"]
        fig.add_hline(y=mb.n_ooip_stb / 1.0e6,
                      line=dict(color=col, width=2.0 if det else 1.2,
                                dash=None if det else "dash"),
                      annotation_text=(f"fitted N "
                                       f"{mb.n_ooip_stb / 1e6:,.1f} MMstb"
                                       if det else "line fit - N NOT "
                                       "determined"),
                      annotation_font=dict(size=10, color=col))
    if np.isfinite(mb.n_ceiling_stb):
        fig.add_hline(y=mb.n_ceiling_stb / 1.0e6,
                      line=dict(color=c["critical"], width=1.4, dash="dot"),
                      annotation_text="ceiling, We >= 0",
                      annotation_position="bottom right",
                      annotation_font=dict(size=10, color=c["critical"]))
    fig = _layout(fig, theme, f"Apparent oil in place -- {res.well}",
                  "Cumulative oil (MMstb)", "F / Et (MMstb)", height=height)
    if mb.drive:
        fig.add_annotation(
            text=mb.drive.split(" -")[0], showarrow=False, xref="paper",
            yref="paper", x=0.02, y=0.06, xanchor="left",
            font=dict(size=12, color=c["ink"], family=FONT))
    return fig


def chart_aquifer_match(res, theme: str = "light",
                        height: int = 420) -> go.Figure:
    """Survey pressures against the simulated tank, with and without aquifer."""
    c = palette(theme)
    am = getattr(res, "aquifer_match", None)
    if am is None or not am.ran or am.t_sim is None:
        return _empty(theme, "aquifer history match not run", height)
    yrs = am.t_sim / 365.25
    fig = go.Figure()
    # The same N with the aquifer taken away: the gap between the two lines
    # is the pressure support the match credits to the aquifer.
    if am.p_closed is not None:
        fig.add_trace(go.Scatter(
            x=yrs, y=am.p_closed, mode="lines", name="same N, no aquifer",
            line=dict(width=1.4, color=c["muted"], dash="dash"),
            hovertemplate="%{x:.1f} yr<br>%{y:,.0f} psia<extra>closed tank"
                          "</extra>"))
    fig.add_trace(go.Scatter(
        x=yrs, y=am.p_sim, mode="lines", name="matched tank + aquifer",
        line=dict(width=2.2, color=c["series"][0]),
        hovertemplate="%{x:.1f} yr<br>%{y:,.0f} psia<extra>matched</extra>"))
    fig.add_trace(go.Scatter(
        x=am.t_obs / 365.25, y=am.p_obs, mode="markers", name="surveys",
        marker=_marker(c["series"][1], theme, size=9),
        hovertemplate="%{x:.1f} yr<br>%{y:,.0f} psia<extra>survey</extra>"))
    ys = [am.p_sim, am.p_obs] + ([am.p_closed] if am.p_closed is not None
                                 else [])
    _set_linear_range(fig, [yrs, am.t_obs / 365.25], ys)
    fig = _layout(fig, theme, f"Pressure history match -- {res.well}",
                  "Years on production", "Reservoir pressure (psia)",
                  height=height)
    fig.add_annotation(
        text=f"RMS {am.rms_psi:,.1f} psi, {am.dof} dof", showarrow=False,
        xref="paper", yref="paper", x=0.98, y=0.96, xanchor="right",
        font=dict(size=11, color=c["muted"], family=FONT))
    return fig


def chart_aquifer_profile(res, theme: str = "light",
                          height: int = 420) -> go.Figure:
    """Best achievable mismatch at each N - the range the surveys allow."""
    c = palette(theme)
    am = getattr(res, "aquifer_match", None)
    if am is None or not am.ran or am.profile is None or not len(am.profile):
        return _empty(theme, "no N profile available", height)
    pr = am.profile
    x = pr["N_stb"].to_numpy(float) / 1e6
    rms = np.sqrt(pr["ssr_psi2"].to_numpy(float) / max(am.n_obs, 1))
    thr = float(np.sqrt(pr["threshold"].iloc[0] / max(am.n_obs, 1)))
    fig = go.Figure()
    lo, hi = am.n_range_stb
    if np.isfinite(lo) and np.isfinite(hi):
        fig.add_vrect(x0=lo / 1e6, x1=hi / 1e6, fillcolor=c["band"],
                      line_width=0, layer="below",
                      annotation_text="95 % range", annotation_position="top left",
                      annotation_font=dict(size=10, color=c["muted"]))
    fig.add_trace(go.Scatter(
        x=x, y=rms, mode="lines+markers", name="best match at this N",
        marker=_marker(c["series"][0], theme, size=7),
        line=dict(width=1.8, color=c["series"][0]),
        hovertemplate="N %{x:,.1f} MMstb<br>RMS %{y:,.1f} psi<extra></extra>"))
    fig.add_hline(y=thr, line=dict(color=c["critical"], width=1.4, dash="dot"),
                  annotation_text="95 % level", annotation_position="top left",
                  annotation_font=dict(size=10, color=c["critical"]))
    if am.n_ho_stb and np.isfinite(am.n_ho_stb):
        fig.add_vline(x=am.n_ho_stb / 1e6,
                      line=dict(color=c["muted"], width=1.2, dash="dash"),
                      annotation_text="Havlena-Odeh N",
                      annotation_position="bottom right",
                      annotation_font=dict(size=10, color=c["muted"]))
    fig = _layout(fig, theme, f"How well each N can be matched -- {res.well}",
                  "Oil in place N (MMstb)", "Pressure mismatch, RMS (psi)",
                  height=height)
    # An explicit log range. Left to autorange, the vrect and vline put the
    # axis out to 1e90 on the first screenshot and the curve collapsed onto
    # a single vertical line.
    # Zoom on the part of the profile that is on the chart vertically: the
    # scan runs to 8x either side, and on a well-determined N the whole
    # interesting region is a narrow V in the middle of it.
    top_y = min(float(np.nanmax(rms)), thr * 6.0)
    on = np.flatnonzero(np.isfinite(rms) & (rms <= top_y))
    if on.size:
        i_a, i_b = max(on[0] - 1, 0), min(on[-1] + 1, len(x) - 1)
        x_view = list(x[i_a:i_b + 1])
    else:
        x_view = list(x)
    xs = [v for v in x_view + [lo / 1e6, hi / 1e6]
          if np.isfinite(v) and v > 0]
    x_lo, x_hi = math.log10(min(xs)), math.log10(max(xs))
    pad = 0.05 * max(x_hi - x_lo, 0.1)
    fig.update_xaxes(type="log", range=[x_lo - pad, x_hi + pad])
    top = float(np.nanmax(rms[np.isfinite(rms)])) if np.isfinite(rms).any() \
        else thr * 2
    fig.update_yaxes(range=[0.0, min(top, thr * 6.0) * 1.05])
    return fig


# ------------------------------------------------------------------------------
# Uncertainty
# ------------------------------------------------------------------------------

def chart_eur_cdf(res, column: str = "eur_oil_mstb",
                  label: str = "EUR oil (Mstb)", theme: str = "light",
                  height: int = 400) -> go.Figure:
    """The sampled EUR distribution, with P90/P50/P10 and the base case."""
    c = palette(theme)
    if res.mc is None or not len(res.mc) or column not in res.mc:
        return _empty(theme, "Monte Carlo was not run", height)
    v = np.sort(res.mc[column].to_numpy(float))
    v = v[np.isfinite(v)]
    if not len(v):
        return _empty(theme, "no finite realisations", height)
    cdf = np.arange(1, len(v) + 1) / len(v)
    p90, p50, p10 = np.percentile(v, [10, 50, 90])

    fig = go.Figure()
    # The P90-P10 band, shaded. The percentile markers alone leave the reader
    # to hold three numbers in mind; the band is the range itself.
    fig.add_vrect(x0=p90, x1=p10, fillcolor=c["band"], line_width=0,
                  layer="below", annotation_text="P90 - P10",
                  annotation_position="top left",
                  annotation_font=dict(size=10, color=c["muted"]))
    fig.add_trace(go.Scatter(
        x=v, y=100.0 * (1.0 - cdf), mode="lines", name="Exceedance",
        line=dict(width=2.4, color=c["series"][0]),
        hovertemplate="%{x:,.0f}<br>%{y:.0f} % chance of exceeding"
                      "<extra></extra>"))
    # P90, P50 and P10 can sit within a percent of each other on a tight
    # distribution, so their labels are staggered vertically rather than all
    # placed at the top, where they overprinted into an unreadable smear.
    for (val, name), ypos in zip(((p90, "P90"), (p50, "P50"), (p10, "P10")),
                                 (0.98, 0.88, 0.78)):
        fig.add_vline(x=float(val),
                      line=dict(color=c["muted"], width=1.2, dash="dot"))
        fig.add_annotation(
            x=float(val), y=ypos, xref="x", yref="paper", text=name,
            showarrow=False, xanchor="left", xshift=3,
            font=dict(size=10, color=c["muted"], family=FONT))
    det = getattr(res.forecast,
                  {"eur_oil_mstb": "eur_oil_mstb",
                   "eur_gas_mmscf": "eur_gas_mmscf",
                   "eur_water_mstb": "eur_water_mstb"}.get(column,
                                                           "eur_oil_mstb"),
                  float("nan"))
    if np.isfinite(det):
        pctl = 100.0 * float(np.mean(v <= det))
        fig.add_vline(x=float(det),
                      line=dict(color=c["critical"], width=1.8),
                      annotation_text=f"base case ({pctl:.0f}th)",
                      annotation_position="bottom right",
                      annotation_font=dict(size=10, color=c["critical"]))
    return _layout(fig, theme, f"EUR exceedance -- {res.well}", label,
                   "Chance of exceeding (%)", height=height)


def chart_mc_scatter(res, x: str, y: str, xlabel: str, ylabel: str,
                     theme: str = "light", colour_by: Optional[str] = "model",
                     height: int = 380) -> go.Figure:
    """Two sampled quantities against each other, to show what drives what."""
    c = palette(theme)
    if res.mc is None or not len(res.mc) or x not in res.mc or y not in res.mc:
        return _empty(theme, "Monte Carlo was not run", height)
    fig = go.Figure()
    # Colouring by the sampled decline is the whole point: it shows whether
    # the spread is the parameters moving or the choice of curve.
    if colour_by and colour_by in res.mc and res.mc[colour_by].nunique() > 1:
        for i, (name, grp) in enumerate(res.mc.groupby(colour_by)):
            fig.add_trace(go.Scatter(
                x=grp[x], y=grp[y], mode="markers", name=str(name),
                marker=dict(size=5, color=c["series"][i % len(c["series"])],
                            opacity=0.55,
                            line=dict(width=0.6, color=c["surface"])),
                hovertemplate="%{x:,.0f}<br>%{y:,.1f}<extra></extra>"))
    else:
        fig.add_trace(go.Scatter(
            x=res.mc[x], y=res.mc[y], mode="markers", name="Realisations",
            marker=dict(size=5, color=c["series"][0], opacity=0.5,
                        line=dict(width=0.6, color=c["surface"])),
            hovertemplate="%{x:,.0f}<br>%{y:,.1f}<extra></extra>"))
    fig = _layout(fig, theme, "", xlabel, ylabel, height=height,
                  legend=bool(colour_by))
    _set_linear_range(fig, [res.mc[x].to_numpy(float)],
                      [res.mc[y].to_numpy(float)])
    return fig
