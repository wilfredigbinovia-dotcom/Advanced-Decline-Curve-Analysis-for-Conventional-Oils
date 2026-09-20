"""
================================================================================
 dca_charts.py -- interactive Plotly charts for the gas condensate DCA app
================================================================================

Every chart here follows the same rules, so the app reads as one system:

  * Categorical colours are assigned from a fixed slot order, never cycled.
    Slot 1 is always the measured wellstream, slot 2 always the model. A series
    keeps its colour when other series are toggled off.
  * No chart ever has two y-axes. Gas and liquid are different measures on
    different scales, so they get different panels - overlaying them on twin
    axes lets you manufacture any correlation you like by choosing the scales.
  * Marks are thin, the grid is recessive, and text wears text colours rather
    than the series colour.
  * Hover is on by default. A production chart that cannot tell you the rate
    and date under the cursor is a picture, not a tool.
  * Light and dark are both selected, from the same hues stepped for each
    surface - not an automatic inversion.

Public functions all take a `WellResult` (or plain arrays) plus `theme`, which
is "light" or "dark", and return a `plotly.graph_objects.Figure`.
================================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from gas_condensate_dca import DAYS_PER_YEAR, diagnose_b

# ------------------------------------------------------------------------------
# Palette -- two selected modes, same hues stepped per surface
# ------------------------------------------------------------------------------

THEMES = {
    "light": {
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                   "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "critical": "#d03b3b",
        "band": "rgba(137,135,129,0.14)",
    },
    "dark": {
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500",
                   "#d55181", "#008300", "#9085e9", "#e66767"],
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "critical": "#d03b3b",
        "band": "rgba(137,135,129,0.20)",
    },
}

FONT = ('system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, '
        'sans-serif')


def palette(theme: str = "light") -> dict:
    return THEMES.get(theme, THEMES["light"])


def rgba(hex_color: str, alpha: float) -> str:
    """Hex to rgba(), for area fills that must not hide the gridlines."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _layout(fig: go.Figure, theme: str, title: str, xlabel: str, ylabel: str,
            log_y: bool = False, height: int = 380,
            legend: bool = True) -> go.Figure:
    """Apply the shared chart chrome. Called by every chart in this module."""
    c = palette(theme)
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", y=0.985, yanchor="top",
                   font=dict(size=15, color=c["ink"], family=FONT)),
        paper_bgcolor=c["surface"],
        plot_bgcolor=c["surface"],
        font=dict(family=FONT, size=12, color=c["ink2"]),
        # Room for the title AND a legend row beneath it: at two-column width
        # the legend wraps back over the title if the margin is any tighter.
        margin=dict(l=8, r=8, t=74, b=8),
        height=height,
        hovermode="closest",
        hoverlabel=dict(font=dict(family=FONT, size=12),
                        bgcolor=c["surface"], bordercolor=c["axis"]),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.015, xanchor="right",
                    x=1.0, font=dict(size=11, color=c["ink2"]),
                    bgcolor="rgba(0,0,0,0)"),
        dragmode="pan",
    )
    axis_common = dict(
        showgrid=True, gridcolor=c["grid"], gridwidth=1,
        zeroline=False, linecolor=c["axis"], linewidth=1,
        ticks="outside", tickcolor=c["axis"], ticklen=4,
        tickfont=dict(size=11, color=c["ink2"]),
        title_font=dict(size=12, color=c["ink2"]),
        # Without this the tight outer margins clip the tick labels away
        # entirely and the chart ships with an unlabelled scale.
        automargin=True,
    )
    fig.update_xaxes(title_text=xlabel, **axis_common)
    fig.update_yaxes(title_text=ylabel, **axis_common)
    if log_y:
        # Decade majors with labelled values, plus minor gridlines. Plotly's
        # SI ("~s") format renders blank labels on a log axis over these
        # ranges, which silently leaves the reader with an unlabelled scale.
        fig.update_yaxes(
            type="log", dtick=1, tickformat=",d", exponentformat="none",
            minor=dict(showgrid=True, gridcolor=c["grid"], gridwidth=1,
                       ticks="outside", ticklen=2, tickcolor=c["axis"]))
    return fig


def _empty(theme: str, message: str, height: int = 380) -> go.Figure:
    c = palette(theme)
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False,
                       font=dict(family=FONT, size=13, color=c["muted"]),
                       xref="paper", yref="paper", x=0.5, y=0.5)
    fig.update_layout(paper_bgcolor=c["surface"], plot_bgcolor=c["surface"],
                      height=height, margin=dict(l=8, r=8, t=8, b=8),
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


# ------------------------------------------------------------------------------
# Rate history and forecast
# ------------------------------------------------------------------------------

def chart_rate_time(res, theme: str = "light", show_separator: bool = True,
                    height: int = 440) -> go.Figure:
    """Semilog wellstream rate against time, with the fit and the forecast."""
    c = palette(theme)
    d, fc = res.data, res.forecast.table
    dates = d.df["date"].dt.strftime("%b %Y")
    fig = go.Figure()

    t_fit0 = res.settings["fit_window_days"][0]
    if t_fit0:
        fig.add_vrect(x0=float(d.t[0]) / DAYS_PER_YEAR,
                      x1=float(t_fit0) / DAYS_PER_YEAR,
                      fillcolor=c["band"], line_width=0, layer="below",
                      annotation_text="excluded", annotation_position="top left",
                      annotation_font=dict(size=10, color=c["muted"]))

    fig.add_trace(go.Scatter(
        x=d.t / DAYS_PER_YEAR, y=d.q_ws, mode="markers", name="Wellstream",
        marker=dict(size=7, color=c["series"][0],
                    line=dict(width=1.2, color=c["surface"])),
        customdata=np.stack([dates, d.q_cond, d.cgr], axis=-1),
        hovertemplate=("<b>%{customdata[0]}</b><br>"
                       "Wellstream %{y:,.0f} Mscf/d<br>"
                       "Condensate %{customdata[1]:,.0f} STB/d<br>"
                       "CGR %{customdata[2]:,.1f} STB/MMscf<extra></extra>")))

    if show_separator:
        fig.add_trace(go.Scatter(
            x=d.t / DAYS_PER_YEAR, y=d.q_gas, mode="markers",
            name="Separator gas",
            marker=dict(size=5, color=c["series"][2], opacity=0.6),
            hovertemplate="Separator gas %{y:,.0f} Mscf/d<extra></extra>"))

    tf = res.best_fit.t_fit
    fig.add_trace(go.Scatter(
        x=tf / DAYS_PER_YEAR, y=res.best_fit.predict(tf), mode="lines",
        name=f"{res.best_fit.model_name} fit",
        line=dict(width=2.4, color=c["series"][1]),
        hovertemplate="Fit %{y:,.0f} Mscf/d<extra></extra>"))

    fig.add_trace(go.Scatter(
        x=fc["t_years"], y=fc["q_wellstream_mscfd"], mode="lines",
        name="Forecast",
        line=dict(width=2.4, color=c["series"][1], dash="dash"),
        hovertemplate=("Year %{x:.1f}<br>Forecast %{y:,.0f} Mscf/d"
                       "<extra></extra>")))

    q_econ = res.settings["q_econ_mscfd"]
    fig.add_hline(y=q_econ, line=dict(color=c["critical"], width=1.4, dash="dot"),
                  annotation_text=f"economic limit {q_econ:,.0f} Mscf/d",
                  annotation_position="bottom right",
                  annotation_font=dict(size=10, color=c["critical"]))

    _layout(fig, theme, "Wellstream rate: history, fit and forecast",
            "Time on production (years)", "Gas rate (Mscf/d)",
            log_y=True, height=height)
    fig.update_layout(hovermode="x unified")
    return fig


def chart_rate_cum(res, theme: str = "light", height: int = 380) -> go.Figure:
    """Rate against cumulative - where an OGIP inconsistency shows up fastest."""
    c = palette(theme)
    d, fc = res.data, res.forecast.table
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d.Gp_ws, y=d.q_ws, mode="markers", name="Measured",
        marker=dict(size=7, color=c["series"][0],
                    line=dict(width=1.2, color=c["surface"])),
        hovertemplate=("Gp %{x:,.0f} MMscf<br>Rate %{y:,.0f} Mscf/d"
                       "<extra></extra>")))
    fig.add_trace(go.Scatter(
        x=fc["Gp_wellstream_mmscf"], y=fc["q_wellstream_mscfd"], mode="lines",
        name="Forecast", line=dict(width=2.4, color=c["series"][1]),
        hovertemplate=("Gp %{x:,.0f} MMscf<br>Rate %{y:,.0f} Mscf/d"
                       "<extra></extra>")))
    if res.matbal is not None:
        fig.add_vline(
            x=res.matbal.ogip_mmscf,
            line=dict(color=c["series"][3], width=1.6, dash="dash"),
            annotation_text=f"OGIP {res.matbal.ogip_mmscf:,.0f} MMscf",
            annotation_position="top left",
            annotation_font=dict(size=10, color=c["ink2"]))
    return _layout(fig, theme, "Rate vs cumulative wellstream gas",
                   "Cumulative wellstream gas (MMscf)", "Gas rate (Mscf/d)",
                   log_y=True, height=height)


def chart_condensate(res, theme: str = "light", height: int = 380) -> go.Figure:
    """Condensate on its own axis. Never share a panel with the gas rate."""
    c = palette(theme)
    d, fc = res.data, res.forecast.table
    dates = d.df["date"].dt.strftime("%b %Y")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d.t / DAYS_PER_YEAR, y=d.q_cond, mode="markers", name="Measured",
        marker=dict(size=7, color=c["series"][2],
                    line=dict(width=1.2, color=c["surface"])),
        customdata=dates,
        hovertemplate=("<b>%{customdata}</b><br>Condensate %{y:,.0f} STB/d"
                       "<extra></extra>")))
    fig.add_trace(go.Scatter(
        x=fc["t_years"], y=fc["q_condensate_stbd"], mode="lines",
        name="Forecast (gas x CGR model)",
        line=dict(width=2.4, color=c["series"][1]),
        hovertemplate=("Year %{x:.1f}<br>Condensate %{y:,.0f} STB/d"
                       "<extra></extra>")))
    return _layout(fig, theme, "Condensate rate: history and forecast",
                   "Time on production (years)", "Condensate rate (STB/d)",
                   height=height)


# ------------------------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------------------------

def chart_b_diagnostic(res, theme: str = "light", height: int = 360) -> go.Figure:
    """b(t) from the data against the fitted b. Flat means Arps is defensible."""
    c = palette(theme)
    d = res.data
    t0 = res.settings["fit_window_days"][0] or d.t[0]
    m = d.t >= t0
    diag = diagnose_b(d.t[m], d.q_ws[m]) if m.sum() >= 6 else diagnose_b(d.t, d.q_ws)
    if diag.empty:
        return _empty(theme, "Not enough data for a loss-ratio diagnostic", height)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=diag["t"] / DAYS_PER_YEAR, y=diag["b"], mode="lines",
        name="b from data", line=dict(width=2, color=c["series"][0]),
        hovertemplate="Year %{x:.2f}<br>b = %{y:.2f}<extra></extra>"))
    b_fit = res.best_fit.params.get("b")
    if b_fit is not None:
        fig.add_hline(y=b_fit, line=dict(color=c["series"][1], width=2.2),
                      annotation_text=f"fitted b = {b_fit:.2f}",
                      annotation_position="bottom left",
                      annotation_font=dict(size=10, color=c["ink2"]))
    fig.add_hline(y=0.5, line=dict(color=c["muted"], width=1, dash="dot"),
                  annotation_text="b = 0.5, BDF gas theory",
                  annotation_position="top right",
                  annotation_font=dict(size=10, color=c["muted"]))
    _layout(fig, theme, "Loss-ratio diagnostic: is b really constant?",
            "Time on production (years)", "b  =  d(1/D)/dt", height=height)
    fig.update_yaxes(range=[-0.5, 2.5])
    return fig


def chart_decline_rate(res, theme: str = "light", height: int = 360) -> go.Figure:
    """Effective decline from the data and from the model, on one scale."""
    c = palette(theme)
    d, fc = res.data, res.forecast.table
    t0 = res.settings["fit_window_days"][0] or d.t[0]
    m = d.t >= t0
    diag = diagnose_b(d.t[m], d.q_ws[m]) if m.sum() >= 6 else diagnose_b(d.t, d.q_ws)

    fig = go.Figure()
    if not diag.empty:
        fig.add_trace(go.Scatter(
            x=diag["t"] / DAYS_PER_YEAR,
            y=100 * (1 - np.exp(-diag["D"] * DAYS_PER_YEAR)),
            mode="lines", name="From data",
            line=dict(width=2, color=c["series"][0]),
            hovertemplate="Year %{x:.2f}<br>D = %{y:.1f} %/yr<extra></extra>"))
    t_all = np.concatenate([d.t[m], fc["t_days"].to_numpy()])
    t_all = t_all[t_all >= res.best_fit.t0]
    if t_all.size:
        fig.add_trace(go.Scatter(
            x=t_all / DAYS_PER_YEAR,
            y=100 * (1 - np.exp(-res.best_fit.model.D(t_all) * DAYS_PER_YEAR)),
            mode="lines", name="From model",
            line=dict(width=2.4, color=c["series"][1]),
            hovertemplate="Year %{x:.2f}<br>D = %{y:.1f} %/yr<extra></extra>"))
    _layout(fig, theme, "Nominal decline rate",
            "Time on production (years)", "D (effective %/yr)", height=height)
    fig.update_yaxes(range=[0, 100])
    return fig


# ------------------------------------------------------------------------------
# Yield and material balance
# ------------------------------------------------------------------------------

def chart_cgr(res, theme: str = "light", height: int = 380) -> go.Figure:
    """Condensate yield against cumulative gas, with the fitted decay."""
    c = palette(theme)
    d, fc = res.data, res.forecast.table
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d.Gp_ws, y=d.cgr, mode="markers", name="Measured CGR",
        marker=dict(size=7, color=c["series"][0],
                    line=dict(width=1.2, color=c["surface"])),
        hovertemplate=("Gp %{x:,.0f} MMscf<br>CGR %{y:,.1f} STB/MMscf"
                       "<extra></extra>")))
    g_end = max(float(fc["Gp_wellstream_mmscf"].max()), float(d.Gp_ws.max()))
    gg = np.linspace(0, g_end, 300)
    fig.add_trace(go.Scatter(
        x=gg, y=res.yield_model(gg), mode="lines", name="Yield model",
        line=dict(width=2.4, color=c["series"][1]),
        hovertemplate=("Gp %{x:,.0f} MMscf<br>CGR %{y:,.1f} STB/MMscf"
                       "<extra></extra>")))
    if res.yield_model.Gp_dew > 0:
        fig.add_vline(
            x=res.yield_model.Gp_dew,
            line=dict(color=c["series"][3], width=1.5, dash="dash"),
            annotation_text="dew-point break", annotation_position="top left",
            annotation_font=dict(size=10, color=c["ink2"]))
    return _layout(fig, theme, "Condensate yield vs cumulative gas",
                   "Cumulative wellstream gas (MMscf)", "CGR (STB/MMscf)",
                   height=height)


def chart_pz(res, theme: str = "light", height: int = 380) -> go.Figure:
    """p/z material balance, with the single-phase points for comparison."""
    c = palette(theme)
    if res.matbal is None:
        return _empty(theme, "No reservoir pressure data - material balance "
                             "needs a p_res column", height)
    mb = res.matbal
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mb.gp, y=mb.pz, mode="markers", name="Two-phase z",
        marker=dict(size=8, color=c["series"][0],
                    line=dict(width=1.2, color=c["surface"])),
        customdata=mb.pressure,
        hovertemplate=("Gp %{x:,.0f} MMscf<br>p/z %{y:,.0f} psia<br>"
                       "p %{customdata:,.0f} psia<extra></extra>")))
    if np.isfinite(mb.ogip_mmscf) and mb.ogip_mmscf > 0:
        xs = np.array([0.0, mb.ogip_mmscf])
        fig.add_trace(go.Scatter(
            x=xs, y=mb.pz_i * (1 - xs / mb.ogip_mmscf), mode="lines",
            name=f"OGIP = {mb.ogip_mmscf:,.0f} MMscf",
            line=dict(width=2.4, color=c["series"][1]),
            hovertemplate="%{y:,.0f} psia<extra></extra>"))
    else:
        # No intercept to draw. The points still carry the story, and an
        # annotation is more use than a missing trace nobody can explain.
        fig.add_annotation(
            xref="paper", yref="paper", x=0.02, y=0.06, showarrow=False,
            align="left", font=dict(size=12, color=c["critical"]),
            text="p/z does not decline — no straight-line OGIP")
    if mb.ogip_single_phase:
        pz_sp = mb.pressure / res.pvt.z(mb.pressure)
        fig.add_trace(go.Scatter(
            x=mb.gp, y=pz_sp, mode="markers",
            name=f"Single-phase z ({mb.ogip_single_phase:,.0f} MMscf)",
            marker=dict(size=7, color=c["series"][4], symbol="square",
                        opacity=0.8),
            hovertemplate="p/z single-phase %{y:,.0f} psia<extra></extra>"))
    _layout(fig, theme, "Material balance: p/z vs cumulative gas",
            "Cumulative wellstream gas (MMscf)", "p/z (psia)", height=height)
    fig.update_yaxes(rangemode="tozero")
    return fig


# ------------------------------------------------------------------------------
# Uncertainty
# ------------------------------------------------------------------------------

def chart_eur_cdf(res, column: str = "eur_wellstream_mmscf",
                  label: str = "EUR wellstream gas (MMscf)",
                  stats_key: str = "EUR wellstream gas (MMscf)",
                  theme: str = "light", height: int = 400) -> go.Figure:
    """Exceedance curve. P90 is the low case, in the petroleum convention."""
    c = palette(theme)
    if res.mc is None or not len(res.mc):
        return _empty(theme, "Monte Carlo was not run", height)
    v = np.sort(res.mc[column].to_numpy())
    prob = 1.0 - np.arange(1, len(v) + 1) / len(v)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=v, y=prob, mode="lines", name="Monte Carlo",
        line=dict(width=2.4, color=c["series"][0]),
        hovertemplate=("%{x:,.0f}<br>%{y:.0%} chance of exceeding"
                       "<extra></extra>")))
    st = (res.mc_stats or {}).get(stats_key)
    if st:
        for name, col, val in (("P90", c["series"][3], st["P90"]),
                               ("P50", c["series"][1], st["P50"]),
                               ("P10", c["series"][6], st["P10"])):
            fig.add_vline(x=val, line=dict(color=col, width=1.6, dash="dash"),
                          annotation_text=f"{name} {val:,.0f}",
                          annotation_position="top",
                          annotation_font=dict(size=10, color=c["ink2"]))
    _layout(fig, theme, "Probabilistic EUR", label,
            "Probability of exceeding", height=height)
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    return fig


def chart_mc_scatter(res, x: str, y: str, xlabel: str, ylabel: str,
                     theme: str = "light", height: int = 380) -> go.Figure:
    """Which parameter is actually driving the EUR spread."""
    c = palette(theme)
    if res.mc is None or x not in res.mc or y not in res.mc:
        return _empty(theme, "Monte Carlo was not run", height)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=res.mc[x], y=res.mc[y], mode="markers", name="Realisations",
        marker=dict(size=5, color=c["series"][0], opacity=0.45,
                    line=dict(width=0)),
        hovertemplate="%{x:,.4g}<br>%{y:,.0f}<extra></extra>"))
    return _layout(fig, theme, "What drives the spread", xlabel, ylabel,
                   height=height, legend=False)


# ------------------------------------------------------------------------------
# Field level
# ------------------------------------------------------------------------------

def chart_field_profile(profile: pd.DataFrame, column: str, title: str,
                        ylabel: str, theme: str = "light",
                        height: int = 380) -> go.Figure:
    """One measure per panel - gas and condensate never share an axis."""
    c = palette(theme)
    if profile is None or profile.empty:
        return _empty(theme, "No field profile available", height)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=profile["t_years"], y=profile[column], mode="lines", name=ylabel,
        line=dict(width=2.4, color=c["series"][0]),
        fill="tozeroy", fillcolor=rgba(c["series"][0], 0.16),
        hovertemplate="Year %{x:.1f}<br>%{y:,.0f}<extra></extra>"))
    return _layout(fig, theme, title, "Years from first field production",
                   ylabel, height=height, legend=False)


def chart_well_bars(summary: pd.DataFrame, column: str, title: str,
                    xlabel: str, theme: str = "light",
                    height: int = 340) -> go.Figure:
    """Ranked wells. One hue - the bars are one series, not eight."""
    c = palette(theme)
    if summary is None or summary.empty or column not in summary:
        return _empty(theme, "No field summary available", height)
    s = summary.sort_values(column)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=s[column], y=s["well"], orientation="h",
        marker=dict(color=c["series"][0],
                    line=dict(width=2, color=c["surface"])),
        hovertemplate="%{y}: %{x:,.0f}<extra></extra>",
        text=[f"{v:,.0f}" for v in s[column]], textposition="auto",
        textfont=dict(color=c["ink2"], size=11)))
    _layout(fig, theme, title, xlabel, "", height=height, legend=False)
    fig.update_yaxes(showgrid=False)
    return fig


def chart_havlena_odeh(res, theme: str = "light", height: int = 380) -> go.Figure:
    """F/Eg against cumulative. Flat means no influx and the level IS G.

    This is the discriminator the p/z plot cannot be: a supported reservoir
    produces a perfectly straight p/z line with an inflated intercept, and only
    this panel tells you it happened.
    """
    c = palette(theme)
    mb = res.matbal
    if mb is None or mb.ho_table.empty:
        return _empty(theme, "No material balance available", height)
    d = mb.ho_table
    ok = d["F_over_Eg_reliable"].to_numpy()
    fig = go.Figure()
    if ok.any():
        fig.add_trace(go.Scatter(
            x=d.loc[ok, "Gp_mmscf"], y=d.loc[ok, "F_over_Eg_mmscf"],
            mode="lines+markers", name="F / Eg",
            line=dict(width=2.4, color=c["series"][0]),
            marker=dict(size=8, line=dict(width=1.2, color=c["surface"])),
            hovertemplate=("Gp %{x:,.0f} MMscf<br>F/Eg %{y:,.0f} MMscf"
                           "<extra></extra>")))
    weak = d["F_over_Eg_mmscf"].notna().to_numpy() & ~ok
    if weak.any():
        fig.add_trace(go.Scatter(
            x=d.loc[weak, "Gp_mmscf"], y=d.loc[weak, "F_over_Eg_mmscf"],
            mode="markers", name="too near the reference",
            marker=dict(size=9, color=c["muted"], symbol="circle-open"),
            hovertemplate="Eg is too small here to trust<extra></extra>"))
    if np.isfinite(mb.g_ceiling_mmscf):
        fig.add_hline(y=mb.g_ceiling_mmscf,
                      line=dict(color=c["series"][1], width=1.8, dash="dash"),
                      annotation_text=f"ceiling {mb.g_ceiling_mmscf:,.0f} MMscf",
                      annotation_position="bottom right",
                      annotation_font=dict(size=10, color=c["ink2"]))
    if np.isfinite(mb.ogip_mmscf):
        fig.add_hline(y=mb.ogip_mmscf,
                      line=dict(color=c["critical"], width=1.4, dash="dot"),
                      annotation_text=f"p/z intercept {mb.ogip_mmscf:,.0f}",
                      annotation_position="top right",
                      annotation_font=dict(size=10, color=c["critical"]))
    return _layout(fig, theme, "Havlena-Odeh:  F = G·Eg + We",
                   "Cumulative wellstream gas (MMscf)", "F / Eg  (MMscf)",
                   height=height)


def chart_apparent_g(res, theme: str = "light", height: int = 340) -> go.Figure:
    """Apparent G survey by survey. A closed tank returns one number."""
    c = palette(theme)
    mb = res.matbal
    if mb is None or mb.ho_table.empty:
        return _empty(theme, "No material balance available", height)
    d = mb.ho_table
    ok = d["apparent_G_usable"].to_numpy()
    if not ok.any():
        return _empty(theme, "No survey is deep enough into depletion "
                             "to give an apparent G", height)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d.loc[ok, "Gp_mmscf"], y=d.loc[ok, "apparent_G_mmscf"],
        mode="lines+markers", name="apparent G",
        line=dict(width=2.4, color=c["series"][2]),
        marker=dict(size=9, line=dict(width=1.2, color=c["surface"])),
        customdata=d.loc[ok, "p"],
        hovertemplate=("Gp %{x:,.0f} MMscf<br>p %{customdata:,.0f} psia<br>"
                       "apparent G %{y:,.0f} MMscf<extra></extra>")))
    if np.isfinite(mb.g_bound_mmscf):
        fig.add_hline(y=mb.g_bound_mmscf,
                      line=dict(color=c["series"][1], width=1.8, dash="dash"),
                      annotation_text=f"tightest bound {mb.g_bound_mmscf:,.0f}",
                      annotation_position="bottom right",
                      annotation_font=dict(size=10, color=c["ink2"]))
    return _layout(fig, theme,
                   "Apparent G:  Gp / (1 − (p/z)/(p/z)ᵢ)",
                   "Cumulative wellstream gas (MMscf)", "Apparent G (MMscf)",
                   height=height, legend=False)


def chart_aquifer_match(res, theme: str = "light", height: int = 360) -> go.Figure:
    """Observed against Fetkovich-predicted pressure. The fit's own evidence."""
    c = palette(theme)
    f = res.matbal.fetkovich if res.matbal else None
    if not f:
        return _empty(theme, "No aquifer fit", height)
    yrs = np.asarray(f["t_days"], float) / DAYS_PER_YEAR
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yrs, y=f["p_observed"], mode="markers", name="Observed",
        marker=dict(size=9, color=c["series"][0],
                    line=dict(width=1.2, color=c["surface"])),
        hovertemplate="Year %{x:.1f}<br>%{y:,.0f} psia<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=yrs, y=f["p_predicted"], mode="lines+markers", name="Fetkovich model",
        line=dict(width=2.4, color=c["series"][1]),
        marker=dict(size=6),
        hovertemplate="Year %{x:.1f}<br>%{y:,.0f} psia<extra></extra>"))
    return _layout(fig, theme,
                   f"Aquifer pressure match  (rms {f['rms_pct']:.2f} %)",
                   "Years on production", "Reservoir pressure (psia)",
                   height=height)


def chart_aquifer_influx(res, theme: str = "light", height: int = 360) -> go.Figure:
    """Cumulative influx against cumulative water actually produced."""
    c = palette(theme)
    f = res.matbal.fetkovich if res.matbal else None
    if not f:
        return _empty(theme, "No aquifer fit", height)
    yrs = np.asarray(f["t_days"], float) / DAYS_PER_YEAR
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yrs, y=np.asarray(f["we_bbl"], float) / 1e6, mode="lines",
        name="Influx We", line=dict(width=2.4, color=c["series"][0]),
        fill="tozeroy", fillcolor=rgba(c["series"][0], 0.16),
        hovertemplate="Year %{x:.1f}<br>We %{y:,.2f} MMbbl<extra></extra>"))
    return _layout(fig, theme, "Cumulative water influx",
                   "Years on production", "We (MMbbl)", height=height,
                   legend=False)


def chart_aquifer_locus(res, theme: str = "light", height: int = 360) -> go.Figure:
    """The valley in the objective: which G values the data actually allow."""
    c = palette(theme)
    f = res.matbal.fetkovich if res.matbal else None
    if not f or f["locus"].empty:
        return _empty(theme, "No aquifer locus", height)
    d = f["locus"]
    # The locus carries whichever two parameters the fitted model uses, so the
    # hover is built from the frame rather than from hard-coded Fetkovich
    # column names - which is what broke when a second model arrived.
    # Wei and J are carried by every model, so prefer them: the hover then
    # reads the same whichever aquifer law was fitted, and two loci can be
    # compared without translating parameter names in your head.
    stable = [c0 for c0 in ("Wei_mmbbl", "J_bbl_d_psi") if c0 in d.columns]
    par = stable or [col for col in d.columns
                     if col not in ("G_mmscf", "rms_pct")][:2]
    fig = go.Figure()
    hover = "G %{x:,.0f} MMscf<br>rms %{y:.2f} %"
    if par:
        fig_custom = np.stack([d[col].to_numpy(float) for col in par], axis=-1)
        for i, col in enumerate(par):
            hover += f"<br>{col} %{{customdata[{i}]:,.4g}}"
    else:
        fig_custom = None
    fig.add_trace(go.Scatter(
        x=d["G_mmscf"], y=d["rms_pct"], mode="lines+markers",
        name="best achievable fit",
        line=dict(width=2.4, color=c["series"][0]),
        marker=dict(size=7, line=dict(width=1.2, color=c["surface"])),
        customdata=fig_custom,
        hovertemplate=hover + "<extra></extra>"))
    fig.add_hline(y=f["rms_threshold"],
                  line=dict(color=c["series"][1], width=1.6, dash="dash"),
                  annotation_text="acceptable fit",
                  annotation_position="top left",
                  annotation_font=dict(size=10, color=c["ink2"]))
    lo, hi = f["g_range_mmscf"]
    fig.add_vrect(x0=lo, x1=hi, fillcolor=c["band"], line_width=0, layer="below",
                  annotation_text=f"{lo:,.0f} – {hi:,.0f} MMscf",
                  annotation_position="bottom right",
                  annotation_font=dict(size=10, color=c["ink2"]))
    return _layout(fig, theme, "Which G the data allow",
                   "Gas in place (MMscf)", "Best achievable rms (%)",
                   height=height, legend=False)


def chart_water_forecast(res, theme: str = "light", q_limit: float = 0.0,
                         height: int = 380) -> go.Figure:
    """Produced water, history and trend, against the handling limit.

    Water gets its own panel rather than a second axis on the gas chart. They
    are different measures on different scales, and overlaying them on twin
    axes lets you manufacture any correlation you like by choosing the scales.
    """
    c = palette(theme)
    d = res.data
    wt = res.forecast.water_trend
    if "q_water" not in d.df.columns:
        return _empty(theme, "No water rate column", height)
    qw = pd.to_numeric(d.df["q_water"], errors="coerce").to_numpy(float)
    if not np.any(np.isfinite(qw) & (qw > 0)):
        return _empty(theme, "No produced water recorded", height)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d.t / DAYS_PER_YEAR, y=qw, mode="markers", name="Water (measured)",
        marker=dict(size=7, color=c["series"][0],
                    line=dict(width=1.2, color=c["surface"])),
        customdata=d.df["date"].dt.strftime("%b %Y"),
        hovertemplate=("<b>%{customdata}</b><br>Water %{y:,.0f} STB/d"
                       "<extra></extra>")))

    if wt is not None:
        tf = res.forecast.table["t_days"].to_numpy(float)
        span = np.linspace(float(d.t[0]), float(tf[-1]), 200)
        fig.add_trace(go.Scatter(
            x=span / DAYS_PER_YEAR, y=wt.rate(span), mode="lines",
            name=f"Trend {wt.growth_pct_per_year:+,.0f} %/yr",
            line=dict(width=2.4, color=c["series"][1]),
            hovertemplate="%{y:,.0f} STB/d<extra></extra>"))

    if q_limit and q_limit > 0:
        fig.add_hline(y=q_limit, line=dict(color=c["critical"], width=1.6,
                                           dash="dash"),
                      annotation_text=f"limit {q_limit:,.0f} STB/d",
                      annotation_position="top left",
                      annotation_font=dict(size=10, color=c["ink2"]))
    t_end = float(res.data.t[-1]) / DAYS_PER_YEAR
    fig.add_vline(x=t_end, line=dict(color=c["axis"], width=1.2, dash="dot"),
                  annotation_text="today", annotation_position="bottom right",
                  annotation_font=dict(size=10, color=c["muted"]))
    return _layout(fig, theme, "Produced water and the handling limit",
                   "Years on production", "Water rate (STB/d)",
                   log_y=True, height=height)


def chart_bank_pi(res, theme: str = "light", height: int = 380) -> go.Figure:
    """Productivity index against time: the condensate bank, measured.

    q / [m(p_avg) - m(p_wf)] divides out the drawdown and the gas properties,
    so what is left is mobility and contacted volume. A fall that starts where
    the reservoir crosses the dew point is the bank.
    """
    c = palette(theme)
    b = getattr(res, "bank", None)
    if b is None or not b.ok:
        return _empty(theme,
                      f"No bank diagnostic — {b.reason if b else 'not run'}",
                      height)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=b.t_days / DAYS_PER_YEAR, y=b.pi, mode="markers",
        name="Productivity index",
        marker=dict(size=7, color=c["series"][0],
                    line=dict(width=1.2, color=c["surface"])),
        customdata=b.p_avg,
        hovertemplate=("%{x:,.1f} yr<br>PI %{y:,.4g}<br>"
                       "p_avg %{customdata:,.0f} psia<extra></extra>")))
    if b.p_dew_crossed_days is not None:
        fig.add_vline(x=b.p_dew_crossed_days / DAYS_PER_YEAR,
                      line=dict(color=c["critical"], width=1.6, dash="dash"),
                      annotation_text="dew point crossed",
                      annotation_position="top right",
                      annotation_font=dict(size=10, color=c["ink2"]))
    return _layout(fig, theme,
                   f"Productivity index — {100 * b.loss_frac:,.0f} % lost",
                   "Years on production",
                   "q / [m(p_avg) − m(p_wf)]", log_y=True, height=height,
                   legend=False)
