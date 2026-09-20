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
  * Hover is on. A production chart that cannot tell you the rate and date
    under the cursor is a picture, not a tool.
  * Light and dark are both selected, not an automatic inversion.

Every public function takes an `OilWellResult` (or plain arrays) plus `theme`,
and returns a `plotly.graph_objects.Figure`.
================================================================================
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dca_charts import THEMES, FONT, palette, rgba, _layout, _empty  # noqa: F401
from oil_dca import DAYS_PER_YEAR, SCF_PER_MSCF, STB_PER_MSTB


# ------------------------------------------------------------------------------
# Rates
# ------------------------------------------------------------------------------

def chart_rate_time(res, theme: str = "light", log_y: bool = True,
                    height: int = 400) -> go.Figure:
    """Oil rate history, the fitted model, and the forecast."""
    c = palette(theme)
    d, fc = res.data, res.forecast
    t_hist = d.t / DAYS_PER_YEAR
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_hist, y=d.q_oil, mode="markers", name="oil, measured",
        marker=dict(size=4, color=c["series"][0], opacity=0.75),
        hovertemplate="%{x:.2f} yr<br>%{y:,.0f} STB/d<extra></extra>"))

    # The fitted curve is drawn only over the window it was fitted to. Drawing
    # it across the plateau as well would show a model apparently missing data
    # it was never asked to match, and invite the reader to distrust a fit
    # that is doing exactly what it was told.
    t_fit = res.best_fit.t_fit
    if t_fit is not None and len(t_fit):
        grid = np.linspace(float(np.min(t_fit)), float(np.max(t_fit)), 400)
        fig.add_trace(go.Scatter(
            x=grid / DAYS_PER_YEAR, y=res.best_fit.model.rate(grid),
            mode="lines", name=f"{res.best_fit.model_name} fit",
            line=dict(width=2, color=c["series"][1]),
            hovertemplate="%{x:.2f} yr<br>%{y:,.0f} STB/d<extra></extra>"))

    tb = fc.table
    if len(tb) > 1:
        fig.add_trace(go.Scatter(
            x=tb["t_years"], y=tb["q_oil_stbd"], mode="lines",
            name="forecast", line=dict(width=2, color=c["series"][1],
                                       dash="dash"),
            hovertemplate="%{x:.2f} yr<br>%{y:,.0f} STB/d<extra></extra>"))
    if d.qc.fit_start_days:
        fig.add_vline(x=float(d.qc.fit_start_days) / DAYS_PER_YEAR,
                      line=dict(color=c["muted"], width=1, dash="dot"),
                      annotation_text="fit starts",
                      annotation_font=dict(size=10, color=c["muted"]))
    return _layout(fig, theme, f"Oil rate -- {res.well}", "years on production",
                   "oil rate, STB/d", log_y=log_y, height=height)


def chart_three_streams(res, theme: str = "light",
                        height: int = 420) -> go.Figure:
    """Oil, gas and water on three stacked panels, never on twin axes."""
    from plotly.subplots import make_subplots
    c = palette(theme)
    d, tb = res.data, res.forecast.table
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.06,
                        subplot_titles=("oil, STB/d", "gas, Mscf/d",
                                        "water, STB/d"))
    series = [(d.q_oil, "q_oil_stbd", c["series"][0], "oil"),
              (d.q_gas, "q_gas_mscfd", c["series"][2], "gas"),
              (d.q_water, "q_water_stbd", c["series"][3], "water")]
    for i, (hist, col, colour, name) in enumerate(series, start=1):
        fig.add_trace(go.Scatter(
            x=d.t / DAYS_PER_YEAR, y=hist, mode="markers", name=name,
            marker=dict(size=3.5, color=colour, opacity=0.75),
            showlegend=False,
            hovertemplate="%{x:.2f} yr<br>%{y:,.0f}<extra></extra>"),
            row=i, col=1)
        if len(tb) > 1:
            fig.add_trace(go.Scatter(
                x=tb["t_years"], y=tb[col], mode="lines", name=f"{name} fcst",
                line=dict(width=1.8, color=colour, dash="dash"),
                showlegend=False,
                hovertemplate="%{x:.2f} yr<br>%{y:,.0f}<extra></extra>"),
                row=i, col=1)
    fig = _layout(fig, theme, f"Three streams -- {res.well}",
                  "years on production", "", height=height, legend=False)
    fig.update_yaxes(type="log", dtick=1, tickformat=",d",
                     exponentformat="none")
    for ann in fig.layout.annotations:
        ann.font.update(size=11, color=c["ink2"], family=FONT)
    return fig


def chart_rate_cum(res, theme: str = "light", height: int = 380) -> go.Figure:
    """Oil rate against cumulative oil - the plot that shows the EUR."""
    c = palette(theme)
    d, tb = res.data, res.forecast.table
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d.Np / 1.0e6, y=d.q_oil, mode="markers", name="measured",
        marker=dict(size=4, color=c["series"][0], opacity=0.75),
        hovertemplate="%{x:,.2f} MMstb<br>%{y:,.0f} STB/d<extra></extra>"))
    if len(tb) > 1:
        fig.add_trace(go.Scatter(
            x=tb["Np_mstb"] / 1.0e3, y=tb["q_oil_stbd"], mode="lines",
            name="forecast", line=dict(width=2, color=c["series"][1],
                                       dash="dash"),
            hovertemplate="%{x:,.2f} MMstb<br>%{y:,.0f} STB/d<extra></extra>"))
    mb = res.matbal
    if mb is not None and mb.trend_ok and np.isfinite(mb.n_ooip_stb):
        fig.add_vline(x=float(mb.n_ooip_stb) / 1.0e6,
                      line=dict(color=c["critical"], width=1.2, dash="dot"),
                      annotation_text="N from the balance",
                      annotation_font=dict(size=10, color=c["critical"]))
    return _layout(fig, theme, f"Oil rate against cumulative -- {res.well}",
                   "cumulative oil, MMstb", "oil rate, STB/d",
                   log_y=True, height=height)


# ------------------------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------------------------

def chart_gor(res, theme: str = "light", height: int = 400) -> go.Figure:
    """Producing GOR against cumulative oil, with Rsi, the break and the peak."""
    c = palette(theme)
    d, tb = res.data, res.forecast.table
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d.Np / 1.0e6, y=d.gor, mode="markers", name="producing GOR",
        marker=dict(size=4, color=c["series"][2], opacity=0.75),
        hovertemplate="%{x:,.2f} MMstb<br>%{y:,.0f} scf/STB<extra></extra>"))
    fig.add_hline(y=float(res.pvt.rsi),
                  line=dict(color=c["muted"], width=1.2, dash="dash"),
                  annotation_text="Rsi",
                  annotation_font=dict(size=10, color=c["muted"]))
    if len(tb) > 1:
        fig.add_trace(go.Scatter(
            x=tb["Np_mstb"] / 1.0e3, y=tb["gor_scf_per_stb"], mode="lines",
            name="GOR model", line=dict(width=2, color=c["series"][1],
                                        dash="dash"),
            hovertemplate="%{x:,.2f} MMstb<br>%{y:,.0f} scf/STB<extra></extra>"))
    g = res.gor_diag
    if g is not None and g.ok:
        if g.broke and g.break_np_stb:
            fig.add_vline(x=float(g.break_np_stb) / 1.0e6,
                          line=dict(color=c["critical"], width=1.2,
                                    dash="dot"),
                          annotation_text="bubble point",
                          annotation_font=dict(size=10, color=c["critical"]))
        if g.peaked and g.peak_np_stb:
            fig.add_vline(x=float(g.peak_np_stb) / 1.0e6,
                          line=dict(color=c["series"][4], width=1.2,
                                    dash="dot"),
                          annotation_text="GOR peak",
                          annotation_font=dict(size=10, color=c["series"][4]))
    return _layout(fig, theme, f"Producing GOR -- {res.well}",
                   "cumulative oil, MMstb", "GOR, scf/STB", height=height)


def chart_chan(res, theme: str = "light", height: int = 400) -> go.Figure:
    """Chan's WOR and WOR' against time since breakthrough, log-log.

    The abscissa is time SINCE BREAKTHROUGH rather than Chan's total producing
    time. On a well that breaks through late, every WOR rising from zero at
    breakthrough carries a 1/(t - t_bt) factor in its local slope, so on a
    total-time axis it looks like it is flattening whatever the mechanism -
    which is how an ordinary linear water cut read as coning with a WOR slope
    of +16. The two axes coincide when breakthrough is early, which is the
    case Chan's own field examples come from.
    """
    c = palette(theme)
    d = res.data
    wd = res.water_diag
    qo, qw, t = d.q_oil, d.q_water, d.t
    ok = (qo > 0) & (qw > 0) & (t > 0)
    if int(ok.sum()) < 8 or wd is None or not wd.ok:
        return _empty(theme, "not enough water production for a Chan plot",
                      height)
    wor = qw[ok] / qo[ok]
    tt = t[ok]
    t_bt = (float(wd.breakthrough_t_days) if wd.breakthrough_t_days
            else float(tt[0]))
    x = np.maximum(tt - t_bt, 1e-6)
    keep = x > 1e-3
    x, wor = x[keep], wor[keep]
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
        marker=dict(size=4.5, color=c["series"][3], opacity=0.8),
        hovertemplate="%{x:,.0f} d after breakthrough<br>WOR "
                      "%{y:,.3f}<extra></extra>"))
    pos = dwor > 0
    fig.add_trace(go.Scatter(
        x=x[pos], y=dwor[pos], mode="markers", name="WOR'",
        marker=dict(size=4.5, color=c["series"][4], opacity=0.8,
                    symbol="diamond"),
        hovertemplate="%{x:,.0f} d<br>WOR' %{y:.3g}/d<extra></extra>"))
    fig = _layout(fig, theme, f"Chan water diagnostic -- {res.well}",
                  "days since breakthrough", "WOR and WOR'",
                  log_y=True, height=height)
    fig.update_xaxes(type="log")
    fig.add_annotation(
        text=wd.mechanism.split(" -")[0], showarrow=False, xref="paper",
        yref="paper", x=0.02, y=0.06, xanchor="left",
        font=dict(size=12, color=c["ink"], family=FONT))
    return fig


def chart_pi(res, theme: str = "light", height: int = 380) -> go.Figure:
    """Productivity index over time, with its source labelled."""
    c = palette(theme)
    pi = res.pi_diag
    if pi is None or not pi.ok or pi.t_days is None:
        reason = (pi.reason if pi is not None and pi.reason
                  else "no productivity index available")
        return _empty(theme, reason, height)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.asarray(pi.t_days) / DAYS_PER_YEAR, y=np.asarray(pi.pi),
        mode="markers", name="PI",
        marker=dict(size=4.5, color=c["series"][0], opacity=0.8),
        hovertemplate="%{x:.2f} yr<br>%{y:,.2f} STB/d/psi<extra></extra>"))
    t_yr = np.asarray(pi.t_days) / DAYS_PER_YEAR
    if np.isfinite(pi.trend_pct_per_year) and len(t_yr) > 1:
        k = np.log1p(pi.trend_pct_per_year / 100.0)
        anchor = float(np.median(np.asarray(pi.pi)))
        t_mid = float(np.median(t_yr))
        fig.add_trace(go.Scatter(
            x=t_yr, y=anchor * np.exp(k * (t_yr - t_mid)), mode="lines",
            name=f"{pi.trend_pct_per_year:+.1f} %/yr",
            line=dict(width=2, color=c["series"][1])))
    fig = _layout(fig, theme, f"Productivity index -- {res.well}",
                  "years on production", "PI, STB/d/psi", height=height)
    fig.add_annotation(
        text=f"p_avg from {pi.p_avg_source}", showarrow=False, xref="paper",
        yref="paper", x=0.02, y=0.06, xanchor="left",
        font=dict(size=10, color=c["muted"], family=FONT))
    return fig


def chart_water(res, theme: str = "light", water_cut_limit: float = 0.0,
                height: int = 380) -> go.Figure:
    """Water cut history and forecast, against the limit that ends the well."""
    c = palette(theme)
    d, tb = res.data, res.forecast.table
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d.t / DAYS_PER_YEAR, y=100.0 * d.water_cut, mode="markers",
        name="measured", marker=dict(size=4, color=c["series"][3],
                                     opacity=0.75),
        hovertemplate="%{x:.2f} yr<br>%{y:.1f} %<extra></extra>"))
    if len(tb) > 1:
        fig.add_trace(go.Scatter(
            x=tb["t_years"], y=100.0 * tb["water_cut"], mode="lines",
            name="forecast", line=dict(width=2, color=c["series"][1],
                                       dash="dash"),
            hovertemplate="%{x:.2f} yr<br>%{y:.1f} %<extra></extra>"))
    if water_cut_limit and water_cut_limit > 0:
        fig.add_hline(y=100.0 * water_cut_limit,
                      line=dict(color=c["critical"], width=1.2, dash="dash"),
                      annotation_text="limit",
                      annotation_font=dict(size=10, color=c["critical"]))
    return _layout(fig, theme, f"Water cut -- {res.well}",
                   "years on production", "water cut, %", height=height)


# ------------------------------------------------------------------------------
# Material balance
# ------------------------------------------------------------------------------

def chart_havlena_odeh(res, theme: str = "light",
                       height: int = 400) -> go.Figure:
    """F against Et. A straight line through the origin is a closed tank."""
    c = palette(theme)
    mb = res.matbal
    if mb is None or mb.ho_table is None or not mb.trend_ok:
        return _empty(theme, "no material balance available", height)
    tab = mb.ho_table
    et = tab["Et"].to_numpy(float)
    f = tab["F_rb"].to_numpy(float)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=et, y=f / 1.0e6, mode="markers", name="surveys",
        marker=dict(size=7, color=c["series"][0], opacity=0.85),
        hovertemplate="Et %{x:.5f} rb/STB<br>F %{y:,.2f} MMrb<extra></extra>"))
    if np.isfinite(mb.n_ooip_stb) and len(et):
        xs = np.linspace(0.0, float(np.nanmax(et)) * 1.05, 50)
        fig.add_trace(go.Scatter(
            x=xs, y=mb.n_ooip_stb * xs / 1.0e6, mode="lines",
            name=f"N = {mb.n_ooip_mstb / 1e3:,.1f} MMstb",
            line=dict(width=2, color=c["series"][1])))
    if np.isfinite(mb.n_ceiling_stb) and len(et):
        xs = np.linspace(0.0, float(np.nanmax(et)) * 1.05, 50)
        fig.add_trace(go.Scatter(
            x=xs, y=mb.n_ceiling_stb * xs / 1.0e6, mode="lines",
            name=f"ceiling min(F/Et) = {mb.n_ceiling_stb / 1e6:,.1f} MMstb",
            line=dict(width=1.5, color=c["critical"], dash="dot")))
    return _layout(fig, theme, f"Havlena-Odeh -- {res.well}",
                   "Et = Eo + m Eg + Efw, rb/STB", "F, MMrb", height=height)


def chart_apparent_n(res, theme: str = "light", height: int = 360) -> go.Figure:
    """Apparent N survey by survey. Flat is a closed tank; rising is influx."""
    c = palette(theme)
    mb = res.matbal
    if mb is None or mb.ho_table is None or not mb.trend_ok:
        return _empty(theme, "no material balance available", height)
    tab = mb.ho_table
    use = tab["apparent_N_usable"].to_numpy(bool)
    x = tab["Np_stb"].to_numpy(float)[use] / 1.0e6
    y = tab["apparent_N_stb"].to_numpy(float)[use] / 1.0e6
    if not len(x):
        return _empty(theme, "no survey has enough depletion to read F/Et",
                      height)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines+markers", name="apparent N = F/Et",
        marker=dict(size=7, color=c["series"][0]),
        line=dict(width=1.5, color=rgba(c["series"][0], 0.5)),
        hovertemplate="Np %{x:,.2f} MMstb<br>F/Et %{y:,.1f} "
                      "MMstb<extra></extra>"))
    if np.isfinite(mb.n_ooip_stb):
        fig.add_hline(y=mb.n_ooip_stb / 1.0e6,
                      line=dict(color=c["series"][1], width=1.5),
                      annotation_text="fitted N",
                      annotation_font=dict(size=10, color=c["series"][1]))
    if np.isfinite(mb.n_ceiling_stb):
        fig.add_hline(y=mb.n_ceiling_stb / 1.0e6,
                      line=dict(color=c["critical"], width=1.2, dash="dot"),
                      annotation_text="ceiling, We >= 0",
                      annotation_font=dict(size=10, color=c["critical"]))
    return _layout(fig, theme, f"Apparent oil in place -- {res.well}",
                   "cumulative oil, MMstb", "F / Et, MMstb", height=height)


# ------------------------------------------------------------------------------
# Uncertainty
# ------------------------------------------------------------------------------

def chart_eur_cdf(res, column: str = "eur_oil_mstb",
                  label: str = "EUR oil, Mstb", theme: str = "light",
                  height: int = 380) -> go.Figure:
    """The sampled EUR distribution, with P90/P50/P10 and the base case."""
    c = palette(theme)
    if res.mc is None or not len(res.mc) or column not in res.mc:
        return _empty(theme, "Monte Carlo was not run", height)
    v = np.sort(res.mc[column].to_numpy(float))
    cdf = np.arange(1, len(v) + 1) / len(v)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=v, y=100.0 * (1.0 - cdf), mode="lines", name="exceedance",
        line=dict(width=2, color=c["series"][0]),
        hovertemplate="%{x:,.0f}<br>%{y:.0f} % chance of "
                      "exceeding<extra></extra>"))
    for pct, name in ((10, "P90"), (50, "P50"), (90, "P10")):
        fig.add_vline(x=float(np.percentile(v, pct)),
                      line=dict(color=c["muted"], width=1, dash="dot"),
                      annotation_text=name,
                      annotation_font=dict(size=10, color=c["muted"]))
    det = getattr(res.forecast, {"eur_oil_mstb": "eur_oil_mstb",
                                 "eur_gas_mmscf": "eur_gas_mmscf",
                                 "eur_water_mstb": "eur_water_mstb"}
                  .get(column, "eur_oil_mstb"), float("nan"))
    if np.isfinite(det):
        fig.add_vline(x=float(det),
                      line=dict(color=c["critical"], width=1.5),
                      annotation_text="base case",
                      annotation_font=dict(size=10, color=c["critical"]))
    return _layout(fig, theme, f"EUR exceedance -- {res.well}", label,
                   "chance of exceeding, %", height=height)


def chart_mc_scatter(res, x: str, y: str, xlabel: str, ylabel: str,
                     theme: str = "light", height: int = 360) -> go.Figure:
    """Two sampled quantities against each other, to show what drives what."""
    c = palette(theme)
    if res.mc is None or not len(res.mc) or x not in res.mc or y not in res.mc:
        return _empty(theme, "Monte Carlo was not run", height)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=res.mc[x], y=res.mc[y], mode="markers", name="realisations",
        marker=dict(size=4, color=c["series"][0], opacity=0.45),
        hovertemplate="%{x:,.1f}<br>%{y:,.1f}<extra></extra>"))
    return _layout(fig, theme, "", xlabel, ylabel, height=height,
                   legend=False)
