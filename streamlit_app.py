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
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:                    # the app's own folder, for `import dca`
    sys.path.insert(0, HERE)

st.set_page_config(page_title="Decline Curve Workbench", page_icon="📉",
                   layout="wide", initial_sidebar_state="expanded")

try:
    import dca
    from dca import MODELS
except ModuleNotFoundError as exc:
    # A deployment problem, not a code problem: dca.py has to sit next to this file.
    # Streamlit Cloud redacts the real traceback, so spell it out here instead.
    if exc.name != "dca":
        raise
    listing = "\n".join(sorted(os.listdir(HERE))[:40]) or "(empty)"
    st.error("**`dca.py` is missing.**")
    st.markdown(
        "`streamlit_app.py` is only the user interface — every calculation lives in "
        "`dca.py`, which has to sit **in the same folder**.\n\n"
        f"This app is running from `{HERE}`, which contains:\n\n"
        f"```\n{listing}\n```\n\n"
        "Push `dca.py` to the repository root, next to `streamlit_app.py`. The "
        "repository needs at minimum:\n\n"
        "```\n"
        "streamlit_app.py      this file\n"
        "dca.py                all the mathematics\n"
        "requirements.txt      numpy, scipy, pandas, streamlit, plotly\n"
        "data/                 the sample CSVs (optional — you can upload instead)\n"
        "```\n\n"
        "Streamlit Cloud redeploys automatically once the file is pushed.")
    st.stop()

SERIES_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                  "#e87ba4", "#4a3aa7", "#e34948"]
MODEL_ORDER = ["hyperbolic", "modhyp", "duong", "ple", "sepd", "harmonic", "exponential"]
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Why the input format is what it is. Shown in the app, because a format request
# without its reasoning reads as bureaucracy.
COLUMN_NOTES = """
**Date** — months are counted from these dates, never from the row order. Shut-in months
are usually just missing from a production report, so if you count rows a well that sat
idle for two years looks like it declined through them. The gap has to stay a gap.
Any recognisable date format works.

**Producing days** — the difference between a *calendar-day* rate (volume ÷ days in the
month) and an *operated-day* rate (volume ÷ days actually flowing) is the uptime
fraction. A well that made 9,000 bbl over 10 days is not a 300 bbl/d well that is
dying; it is a 900 bbl/d well that was down. Without this column every downtime month
reads as decline, and the fit obliges by finding one.

**Oil or gas volume** — period **volumes**, not rates, because a rate has an uptime
convention already baked into it and you generally cannot tell which one the source
used. A volume is unambiguous, and dividing it by producing days gives a rate whose
meaning is known.

**Water** *(optional)* — unlocks the Water tab: WOR extrapolation, the Ershaghi X-plot,
and the Chan coning-versus-channelling diagnostic. For a water-drive reservoir the
water trend is usually the more reliable forecast, because the well dies of water
rather than of pressure.

**Zero months** — leave them out rather than writing zeros. The fit minimises the error
in **ln q**, which is undefined at zero; and an omitted month with its date gap intact
carries exactly the right information, which is that nothing was produced and no time
should be credited against the decline.

Column *names* do not matter — the app guesses, and anything it gets wrong you can remap
in the sidebar.
"""


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
        ge: float, pab: float, skip: int, p_init=None):
    return dca.material_balance(np.array(P), np.array(Gp), temp, sg,
                                condensate_mbbl=np.array(cond),
                                water_mbbl=np.array(water),
                                gas_equivalent_scf_per_bbl=ge,
                                p_abandon=pab, skip_early=skip, p_initial=p_init)


@st.cache_data(show_spinner=False)
def _read_csv(src, name: str, column: str | None, days: str | None, date: str | None,
              fluid: str, vfactor: float, cumulative: bool | None):
    # src is a path, upload bytes (csv or a workbook), or pasted text -- read_table
    # works out which. vfactor is the ONE place metric input becomes field units.
    src = io.BytesIO(src) if isinstance(src, bytes) else src
    return dca.load_frame(dca.read_table(src), column=column, days_column=days,
                          date_column=date, label=name, fluid=fluid,
                          volume_factor=vfactor, water_factor=vfactor,
                          cumulative=cumulative)


# Oilfield volume prefixes, which are not SI: for liquids M = thousand and
# MM = million barrels; for gas the base unit is already Mcf, so a thousand of
# them is MMcf and a million is Bcf.
_SCALE = {
    # field
    "bbl": [(1e9, "MMMbbl"), (1e6, "MMbbl"), (1e3, "Mbbl"), (1, "bbl")],
    "Mcf": [(1e9, "Tcf"), (1e6, "Bcf"), (1e3, "MMcf"), (1, "Mcf")],
    # metric -- SI prefixes would read as 10^3 m3 = "dam3", which nobody writes
    "m³": [(1e6, "10⁶m³"), (1e3, "10³m³"), (1, "m³")],
    "10³m³": [(1e6, "10⁹m³"), (1e3, "10⁶m³"), (1, "10³m³")],
}


def fmt(v, unit="", dp=0):
    """Scale a number into a readable oilfield prefix and label it."""
    if v is None or not math.isfinite(v):
        return "—"
    for div, label in _SCALE.get(unit, [(1, unit)]):
        if abs(v) >= div or div == 1:
            dp_ = dp if div == 1 else (2 if div > 1e3 else 1)
            return f"{v / div:,.{dp_}f} {label}".strip()
    return f"{v:,.{dp}f} {unit}".strip()


def vol(v, dp=0):
    """Format a FIELD-unit volume in the display system.

    Every number that reaches the screen goes through here or through dsp().
    Keeping the conversion in one pair of functions is what stops a metric
    session from showing three numbers in field units and one in m3.
    """
    if v is None or not math.isfinite(v):
        return "—"
    return fmt(v / VFACTOR, VUNIT, dp)


def liq(v, dp=0):
    """Format a FIELD-unit LIQUID volume (bbl) in the display system.

    Water and condensate are liquids even on a gas well, so they do not follow
    the well's own volume unit.
    """
    if v is None or not math.isfinite(v):
        return "—"
    if U.name == "field":
        return fmt(v, "bbl", dp)
    return fmt(v / dca.M3_TO_BBL, "m³", dp)


def dsp(x):
    """Field-unit value or array -> display units."""
    return np.asarray(x, dtype=float) / VFACTOR if np.ndim(x) else float(x) / VFACTOR


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

# The sidebar holds the paste box and the column mapping, which are both wider than
# Streamlit's 244px default comfortably shows.
st.markdown(
    "<style>section[data-testid='stSidebar']{width:390px!important;min-width:390px!important}"
    "section[data-testid='stSidebar'] textarea{font-family:ui-monospace,SFMono-Regular,"
    "Menlo,monospace;font-size:12px;white-space:pre}</style>",
    unsafe_allow_html=True)

st.sidebar.title("Decline Curve Workbench")

# --- what this is, before anything about how to read it --------------------
st.sidebar.subheader("Well")
wc1, wc2 = st.sidebar.columns(2)
well_name = wc1.text_input("Well", value="", placeholder="e.g. A-26 LS",
                           key="well_name")
reservoir_name = wc2.text_input("Reservoir", value="", placeholder="e.g. D2300X",
                                key="reservoir_name")

fc1, fc2 = st.sidebar.columns(2)
fluid = fc1.radio("Fluid", ["Oil", "Gas"], key="fluid", horizontal=True,
                  help="Declared, not guessed from the column name. It sets the "
                       "abandonment-rate default, which volume tab you get, and whether "
                       "the water analysis is offered.")
in_name = fc2.radio("Data is in", ["Field", "Metric"], key="unit_system",
                    horizontal=True,
                    help="What YOUR NUMBERS mean. Field: bbl, Mcf, psia, °F. "
                         "Metric: m³, 10³m³, kPa, °C. Get this wrong and every volume "
                         "is out by a factor of 6.3.")
basis = st.sidebar.radio(
    "Volumes are", ["Detect", "Per period", "Cumulative"], key="vol_basis",
    horizontal=True,
    help="Per period: each row is that month's own volume. Cumulative: each row is "
         "everything produced up to and including that month, so the app differences "
         "them. Summing a cumulative column is meaningless — 16.5 Bcf of running total "
         "adds up to 582 Bcf.")
CUM_IN = {"Detect": None, "Per period": False, "Cumulative": True}[basis]

out_name = st.sidebar.radio("Show results in", ["Same as data", "Field", "Metric"],
                            key="unit_out", horizontal=True,
                            help="Reading and reporting are separate choices — field-unit "
                                 "data reported in metric is a normal thing to want, and "
                                 "one control doing both jobs would just relabel the "
                                 "numbers without converting them.")
unit_name = in_name if out_name == "Same as data" else out_name

UIN = dca.units(in_name)                          # what the data means
U = dca.units(unit_name)                          # what the screen shows
IS_GAS = fluid == "Gas"
VUNIT = U.volume_label(IS_GAS)                    # display label
VIN = UIN.gas_to_mcf if IS_GAS else UIN.oil_to_bbl     # data -> field, on the way in
VFACTOR = U.gas_to_mcf if IS_GAS else U.oil_to_bbl     # field -> display, on the way out

label_bits = [b for b in (well_name.strip(), reservoir_name.strip()) if b]
WELL_LABEL = " · ".join(label_bits) if label_bits else ""

# --- keep entered values meaning what they meant when they were typed -------
#
# Switching units changes what a box MEANS, and Streamlit keeps whatever number
# is in it. "10" typed as bbl/d silently becomes 10 m3/d, which is 63 bbl/d, and
# the forecast moves without anything on screen saying why. So on a unit or
# fluid change, every unit-bearing widget is converted through field units into
# the new system.
_UNIT_WIDGETS = {
    "qab": "rate", "ip_direct": "volume",
    "v_area": "area", "v_pay": "length", "v_top": "length", "v_base": "length",
    "v_pi": "pressure", "v_t": "temperature",
    "mb_t": "temperature", "mb_pab": "pressure", "mb_pi": "pressure",
    "omb_t": "temperature",
    "omb_pb": "pressure", "mb_depth": "length", "mb_twh": "temperature",
}


def _reunit(value: float, kind: str, old: "dca.UnitSystem", new: "dca.UnitSystem",
            was_gas: bool, is_gas: bool) -> float:
    if kind in ("rate", "volume"):
        f_old = old.gas_to_mcf if was_gas else old.oil_to_bbl
        f_new = new.gas_to_mcf if is_gas else new.oil_to_bbl
        return value * f_old / f_new
    if kind == "area":
        return value * old.area_to_acre / new.area_to_acre
    if kind == "length":
        return value * old.length_to_ft / new.length_to_ft
    if kind == "pressure":
        return new.pressure_from_psia(old.pressure_to_psia(value))
    if kind == "temperature":
        return new.temperature_from_degf(old.temperature_to_degf(value))
    return value


def unum(container, label, key, default, **kw):
    """A number_input whose default is seeded into session state once.

    Passing both `value=` and writing the key from session state makes Streamlit
    warn, and these boxes must be writable from session state -- that is how a
    unit change rewrites them.
    """
    if key not in st.session_state:
        st.session_state[key] = float(default)
    return container.number_input(label, key=key, **kw)


_sig = (unit_name, IS_GAS)
_prev = st.session_state.get("_unit_sig")
if _prev is not None and _prev != _sig:
    _old_u, _old_gas = dca.units(_prev[0]), _prev[1]
    for _k, _kind in _UNIT_WIDGETS.items():
        if _k in st.session_state and isinstance(st.session_state[_k], (int, float)):
            st.session_state[_k] = float(
                _reunit(float(st.session_state[_k]), _kind, _old_u, U, _old_gas, IS_GAS))
st.session_state["_unit_sig"] = _sig

st.sidebar.divider()


samples = sorted(p for pat in ("*.csv", "*.tsv", "*.txt")
                 for p in glob.glob(os.path.join(DATA_DIR, pat))
                 if "pressure" not in os.path.basename(p))
sample_names = [os.path.basename(p) for p in samples]

# The sample CSVs are optional. Without a data/ folder the app still works -- it just
# has nothing to show until something is loaded, so don't offer a dead choice.
MODES = (["Sample well"] if sample_names else []) + ["Paste", "Upload"]
source = st.sidebar.radio("Production data", MODES, horizontal=True, key="source",
                          index=0)
if not sample_names:
    st.sidebar.caption("No `data/` folder found, so there are no sample wells. "
                       "Paste or upload data to get started.")

PASTE_HELP = (
    "**Paste it straight out of a spreadsheet** — select the range including the header "
    "row and paste. Excel and Sheets copy as tab-separated text, which is read directly; "
    "comma- and semicolon-separated both work too, as does a block with no header row.\n\n"
    "One row per month. A date column, a production column, and ideally a producing-days "
    "column. Column names do not matter — anything unrecognised can be mapped below."
)

raw = None
if source == "Paste":
    # A grid, not a text box. The data starts in a spreadsheet, and a spreadsheet
    # range pasted into a grid lands in cells; pasted into a text box it lands as
    # tab-separated text that then has to be re-parsed. Both work here -- the text
    # route is kept below for browsers whose clipboard does not cooperate with the
    # grid -- but the grid is what the data already looks like.
    # Column names stay constant whatever the fluid or units are, so switching
    # either does not rebuild the grid and lose what has been pasted into it. The
    # units live in the caption and the column help instead.
    VCOL, WCOL = "Production", "Water"
    BLANK_ROWS = 14
    # Every column is text. Two reasons: this Streamlit renders an empty NUMERIC
    # cell as a greyed literal "None", which looks like data; and a text column
    # accepts whatever the clipboard holds -- "1,234", "1 234", a date in any
    # format -- which dca's parser then coerces with the same code that handles
    # pasted text and uploaded files. The grid is the entry surface; the parser
    # is the validator.
    template = pd.DataFrame({c: pd.Series([""] * BLANK_ROWS, dtype="string")
                             for c in ("Date", "Days on", VCOL, WCOL)})

    st.markdown("### Paste your production history")
    st.caption(
        "Click the first cell and paste — a whole range at once is fine. Rows are added "
        "as you need them, and anything can be corrected in place afterwards. Only the "
        "date and the production volume are required; leave a column blank if you do "
        "not have it.")
    st.markdown(
        f"**{fluid.lower()} in {VUNIT} · water in {UIN.oil} · monthly volumes, not rates**"
        + ("" if UIN.name == "field" else "  — reading as metric, per the sidebar"))

    # The key carries a nonce. Deleting a data_editor's key does not clear it --
    # the edits live in the frontend and are replayed onto whatever frame is passed
    # in. Bumping the key builds a genuinely new widget, which is the only reliable
    # way to empty it.
    gnonce = st.session_state.get("grid_nonce", 0)
    grid = st.data_editor(
        template, key=f"paste_grid_{gnonce}", num_rows="dynamic", width="stretch",
        height=460,
        column_config={
            "Date": st.column_config.TextColumn(
                "Date", help="Any recognisable format: 2018-06-01, 06/01/2018, "
                             "1-Jun-2018, Jun 2018.", width="medium"),
            "Days on": st.column_config.TextColumn(
                "Days on", help="Days the well actually flowed that month. Without it "
                                "rates are calendar-day and downtime reads as decline."),
            VCOL: st.column_config.TextColumn(
                VCOL, help=f"Period volume of {fluid.lower()} in {VUNIT} — a volume, "
                           "not a rate."),
            WCOL: st.column_config.TextColumn(
                WCOL, help=f"Produced water in {UIN.oil}. Optional; it unlocks the "
                           "Water tab."),
        })

    def _has(col):
        return grid[col].astype("string").fillna("").str.strip().ne("")

    filled = grid[_has("Date") & _has(VCOL)]

    tc1, tc2 = st.columns([3, 1])
    tc1.caption(f"**{len(filled)}** row{'' if len(filled) == 1 else 's'} with a date and a "
                "volume. Five is the minimum to fit.")
    if tc2.button("Clear the table", use_container_width=True):
        st.session_state["grid_nonce"] = gnonce + 1
        st.session_state.pop(f"paste_grid_{gnonce}", None)
        st.rerun()

    with st.expander("Paste as text instead"):
        st.caption(
            "If the grid will not take your clipboard, paste the raw block here — tab, "
            "comma or semicolon separated, header row optional.")
        txt = st.text_area(
            "Raw paste", height=180, key="paste", label_visibility="collapsed",
            placeholder="Date\tDays On\tOil (bbl)\tWater (bbl)\n"
                        "2018-06-01\t30\t4874\t5288\n"
                        "2018-07-01\t31\t9896\t3991\n"
                        "2018-08-01\t31\t9143\t2536")
    with st.expander("What the columns mean, and why"):
        st.markdown(COLUMN_NOTES)

    # The grid wins when it has data. Someone who pasted text once, then switched to
    # the grid, means the grid -- and text left behind in a collapsed expander should
    # not quietly override what is visibly on screen.
    if len(filled) >= 5:
        raw, src_label = filled.reset_index(drop=True), "pasted table"
        if (txt or "").strip():
            st.info("Using the table. There is also text in **Paste as text instead** "
                    "below — clear the table if you meant to use that.")
    elif (txt or "").strip():
        raw, src_label = txt, "pasted text"
    else:
        st.stop()
elif source == "Upload":

    up = st.sidebar.file_uploader(
        "Production file", type=["csv", "txt", "tsv", "xlsx", "xlsm", "xls"],
        help="CSV, tab-separated text, or an Excel workbook. In a workbook the first "
             "sheet that holds a table with a date column is used, so a cover sheet in "
             "front of the data is fine.")
    if up is None:
        st.markdown("### Upload a production history")
        st.markdown(PASTE_HELP.replace("**Paste it straight out of a spreadsheet**",
                                       "**A CSV, TSV or tab-separated text file**")
                    .replace(" — select the range including the header row and paste", ""))
        with st.expander("What the columns mean, and why"):
            st.markdown(COLUMN_NOTES)
        st.stop()
    raw = up.getvalue()
    src_label = up.name
else:
    pick = st.sidebar.selectbox("Well", sample_names, key="well",
                                index=sample_names.index("synthetic-tight-oil.csv")
                                if "synthetic-tight-oil.csv" in sample_names else 0)
    raw = os.path.join(DATA_DIR, pick)
    src_label = pick

# --- column mapping, guessed then overridable ---
try:
    probe = dca.read_table(io.BytesIO(raw) if isinstance(raw, bytes) else raw)
except (ValueError, FileNotFoundError) as e:
    st.error(f"**Could not read that.** {e}")
    st.markdown(PASTE_HELP)
    st.stop()

cols = list(probe.columns)
guess_date = dca.find_date_column(probe)
guess_days = dca.find_days_column(probe)
guess_prod = dca.find_production_columns(probe)

def _pick(label, options, guess, key, help_=None):
    opts = list(options)
    idx = opts.index(guess) if guess in opts else 0
    return st.sidebar.selectbox(label, opts, index=idx, key=key, help=help_)

col = _pick("Production column", guess_prod or cols, (guess_prod or cols)[0], "prod_col",
            "Oil or gas volume per month. Period volumes, not rates.")
date_col = _pick("Date column", cols, guess_date or cols[0], "date_col",
                 "Months are counted from these dates, not from the row order — so an "
                 "omitted shut-in month stays a gap instead of being compressed away.")
days_col = _pick("Producing-days column", ["(none — use calendar days)"] + cols,
                 guess_days or "(none — use calendar days)", "days_col",
                 "Rate is volume ÷ producing days, i.e. an operated-day rate. Without "
                 "this, downtime reads as decline.")
days_col = None if days_col.startswith("(none") else days_col

if guess_days is None and days_col is None:
    st.sidebar.caption("⚠️ No producing-days column found — rates are calendar-day, so "
                       "downtime will look like decline.")

try:
    s_full = _read_csv(raw, src_label, col, days_col, date_col,
                       fluid.lower(), VIN, CUM_IN)
except (ValueError, FileNotFoundError) as e:
    st.error(f"**Could not use that data.** {e}")
    st.markdown(PASTE_HELP)
    st.stop()

unit = VUNIT          # display label; the data itself is field units
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

qab_disp = unum(
    st.sidebar, f"Abandonment rate ({unit}/d)", "qab",
    (50.0 if IS_GAS else 10.0) / VFACTOR, min_value=1e-6, step=1.0, format="%.4g",
    help="For b ≥ 1 the hyperbolic integral does not converge, so this is what makes "
         "the EUR finite. It is doing real work — state it when you report.")
qab = qab_disp * VFACTOR          # the fit and EUR work in field units
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
st.sidebar.subheader("Volume in place — this well")
st.sidebar.caption(
    "The volume **this well** is connected to, not the reservoir's total. A single-well "
    "forecast can only be checked against a single-well volume.")

# A value sent over from the Material balance tab OVERRIDES the route below, and is
# not written back into the radio. Streamlit forbids setting a widget's key after the
# widget has been created in that run -- the buttons on the MB tab run long after this
# radio does -- so the hand-off carries its own state and wins on its own.
MB_VALUE = st.session_state.get("ip_from_mb")

IP_ROUTES = ["Off", "Enter it", "Volumetric", "From the decline"]
ip_route = st.sidebar.radio(
    "How to get it", IP_ROUTES, key="ip_route", disabled=MB_VALUE is not None,
    help="Material balance is on its own tab — it has a button to send its answer here.")
if MB_VALUE is not None:
    ip_route = "From material balance"

in_place = None
ip_source = ""
FLUID = fluid.lower()

if ip_route == "Enter it":
    # Gas volumes are already in Mcf, so a Bcf entry is x1e6; oil is bbl, so MMbbl is x1e6.
    # Entered in the display system, converted straight to field like everything else.
    big = ("Bcf" if IS_GAS else "MMbbl") if U.name == "field" else \
          ("10⁹m³" if IS_GAS else "10⁶m³")
    scale_disp = 1e6                       # both Bcf/Mcf and 10⁹m³/10³m³ are x1e6
    val = unum(st.sidebar, f"{'GIIP' if IS_GAS else 'STOIIP'} ({big})", "ip_direct", 10.0,
               min_value=0.0, step=1.0, format="%.6g",
               help="For THIS well's drainage volume, not the reservoir.")
    in_place = val * scale_disp * VFACTOR if val > 0 else None
    ip_source = f"entered directly: {val:g} {big}"

elif ip_route == "Volumetric":
    st.sidebar.caption("Drainage area for **this well**, not the field.")
    area_d = unum(st.sidebar, f"Drainage area ({U.area})", "v_area", 160.0 / U.area_to_acre,
                  min_value=0.01, max_value=1e6, step=10.0, format="%.4g")
    area = area_d * U.area_to_acre

    # Net pay two ways, and only one of them live at a time. Streamlit reruns the
    # whole script on every keystroke, so reading the other group out of session
    # state before building this one is enough to decide what to grey out. Zero
    # is the "not entered" value in both directions -- a real net pay is never 0,
    # and neither is a real gross interval.
    _pay_set = float(st.session_state.get("v_pay", 0.0) or 0.0) > 0
    _top = float(st.session_state.get("v_top", 0.0) or 0.0)
    _base = float(st.session_state.get("v_base", 0.0) or 0.0)
    _interval_set = _base > _top > 0

    c2a, c2b = st.sidebar.columns([1, 1])
    # Asymmetric on purpose. If both somehow hold values -- restored session
    # state, or a default changing under an old key -- disabling both leaves no
    # box the user can edit to escape. Net pay wins and stays editable, so
    # clearing it always re-opens the interval.
    pay_d = unum(c2a, f"Net pay ({U.length})", "v_pay", 0.0,
                 min_value=0.0, max_value=2e4, step=5.0, format="%.4g",
                 disabled=_interval_set and not _pay_set,
                 help="Leave at 0 to build it from the interval instead.")
    c2b.markdown("<div style='padding-top:1.9rem;color:#7e8286;font-size:.8rem'>"
                 "or ↓</div>", unsafe_allow_html=True)

    c7, c8, c9 = st.sidebar.columns(3)
    top_d = unum(c7, f"Top ({U.length})", "v_top", 0.0, min_value=0.0, max_value=6e4,
                 step=10.0, format="%.6g", disabled=_pay_set,
                 help="True vertical depth to the top of the reservoir.")
    base_d = unum(c8, f"Base ({U.length})", "v_base", 0.0, min_value=0.0, max_value=6e4,
                  step=10.0, format="%.6g", disabled=_pay_set,
                  help="Same reference as the top — both subsea, or both along hole.")
    ntg = c9.number_input("NTG", 0.01, 1.0, 1.0, step=0.05, key="v_ntg",
                          disabled=_pay_set,
                          help="Net-to-gross. Multiplies the gross interval to give net pay.")

    pay_err = None
    if _pay_set:
        pay = pay_d * U.length_to_ft
        st.sidebar.caption(f"Net pay **{pay_d:,.4g} {U.length}**, entered directly. "
                           "Clear it to build net pay from the interval instead.")
    elif base_d > 0 or top_d > 0:
        try:
            pay = dca.net_pay_from_interval(top_d * U.length_to_ft,
                                            base_d * U.length_to_ft, ntg)
            st.sidebar.caption(
                f"Gross **{base_d - top_d:,.6g} {U.length}** × NTG {ntg:.0%} = net pay "
                f"**{pay / U.length_to_ft:,.4g} {U.length}**. Enter a net pay above to "
                "override this.")
        except ValueError as e:
            pay, pay_err = 0.0, str(e)
    else:
        pay, pay_err = 0.0, ("Enter a **net pay**, or a **top, base and NTG** to build one "
                             "from the interval.")
    if pay_err:
        st.sidebar.info(pay_err) if pay <= 0 and "Enter a" in pay_err else st.sidebar.error(pay_err)
    c3, c4 = st.sidebar.columns(2)
    poro = c3.number_input("Porosity (frac)", 0.01, 0.60, 0.20, step=0.01, key="v_poro")
    swi = c4.number_input("Water sat. (frac)", 0.0, 0.95, 0.25, step=0.05, key="v_sw")
    try:
        if pay <= 0:
            raise ValueError("no net pay yet")
        if IS_GAS:
            c5, c6 = st.sidebar.columns(2)
            pi_d = unum(c5, f"Initial pressure ({U.pressure})", "v_pi",
                        U.pressure_from_psia(4000.0), min_value=1.0, max_value=2e5,
                        step=100.0, format="%.6g")
            t_d = unum(c6, f"Temperature ({U.temperature})", "v_t",
                       U.temperature_from_degf(200.0), min_value=-50.0, max_value=500.0,
                       step=5.0, format="%.4g")
            sg_v = st.sidebar.number_input("Gas gravity", 0.55, 1.2, 0.65, step=0.01,
                                           key="v_sg")
            pi_v, t_v = U.pressure_to_psia(pi_d), U.temperature_to_degf(t_d)
            # scf -> Mcf, which is the field unit the rest of the app works in
            in_place = dca.volumetric_gas_in_place(area, pay, poro, swi, pi_v, t_v,
                                                   sg_v) / 1000.0
            ip_source = (f"volumetric · {area_d:,.4g} {U.area} × {pay_d:,.4g} {U.length} × "
                         f"{poro:.0%} φ × {1-swi:.0%} Sg at {pi_d:,.6g} {U.pressure}")
        else:
            boi = st.sidebar.number_input("Bo initial (rb/STB)", 1.0, 3.0, 1.25, step=0.05,
                                          key="v_boi",
                                          help="Reservoir barrels per stock-tank barrel — "
                                               "dimensionless in effect, so the same number "
                                               "in either unit system.")
            in_place = dca.volumetric_oil_in_place(area, pay, poro, swi, boi)
            ip_source = (f"volumetric · {area_d:,.4g} {U.area} × {pay_d:,.4g} {U.length} × "
                         f"{poro:.0%} φ × {1-swi:.0%} So ÷ Boi {boi:.2f}")
    except ValueError as e:
        if str(e) != "no net pay yet":
            st.sidebar.error(str(e))

elif ip_route == "From the decline":
    mv_side = dca.movable_volume_from_decline(s.cum, s.q)
    if mv_side is None:
        st.sidebar.warning("Rate is not falling with cumulative on this window, so there "
                           "is no line to extrapolate. Use another route.")
    else:
        rf_mov = st.sidebar.number_input(
            "Recovery factor on the movable volume (%)", 1.0, 100.0,
            60.0 if IS_GAS else 30.0, step=5.0, key="mv_rf",
            help="The intercept is what the well can MOVE, not what is in place. "
                 "Divide by a recovery factor to get in place.")
        in_place = mv_side.movable / (rf_mov / 100.0)
        ip_source = (f"decline · movable {vol(mv_side.movable)} at q→0 "
                     f"(R² {mv_side.r2:.3f}) ÷ {rf_mov:g}% RF")
        if mv_side.r2 < 0.7:
            st.sidebar.warning(f"R² {mv_side.r2:.2f} — the q-vs-cumulative points do not "
                               "form a line, so this intercept is not meaningful.")

elif ip_route == "From material balance":
    in_place = MB_VALUE
    ip_source = st.session_state.get("ip_from_mb_src", "material balance")
    st.sidebar.success(f"Using **{vol(in_place)}** from the Material balance tab. "
                       "The routes above are switched off while it is in use.")
    if st.sidebar.button("Clear it and choose another route"):
        st.session_state.pop("ip_from_mb", None)
        st.session_state.pop("ip_from_mb_src", None)
        st.rerun()

DEFAULT_RF = {"gas_vol": 85.0, "gas_wd": 60.0, "oil": 30.0}
max_rf = st.sidebar.number_input(
    "Maximum recovery factor (%)", 1.0, 100.0,
    float(DEFAULT_RF["gas_vol"] if IS_GAS else DEFAULT_RF["oil"]),
    step=5.0, key="max_rf",
    help=("Gas: 80–90% volumetric, 50–70% under water drive. "
          "Oil: 5–15% depletion drive, 20–40% water drive or waterflood, "
          "30–60% with good sweep. This is the ceiling the forecast is checked against."))
cap_on = in_place is not None
rf = max_rf


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

st.title(WELL_LABEL or "Decline Curve Workbench")
st.caption(
    (f"{fluid.lower()} · {unit_name.lower()} units · " if WELL_LABEL else "")
    + f"**{src_label}** · {s.column} · {n_all} producing periods, fitting {len(s.t)} "
    f"({s.dates.iloc[0]:%b %Y} – {s.dates.iloc[-1]:%b %Y}) · "
    f"cum to date {vol(cum_to_date)}"
    + (f" · {vol(beyond)} produced after the fit window" if beyond > 0 else ""))

# The four numbers that describe this well, reconciled against each other. Built
# before the tabs so the Forecast tab and the In-place tab show the same thing.
wv = dca.WellVolumes(in_place=in_place, cum=cum_to_date, eur=primary["eur"].eur,
                     unit=unit, max_rf_pct=max_rf, source=ip_source)

# Two readings the loader had to make on its own. Both are silent killers when
# wrong -- a cumulative column summed, or DD/MM read as MM/DD -- so neither is
# allowed to stay implicit.
if s_full.cumulative_input:
    st.info(
        f"**Read as a running total.** The {s.column!r} column never falls, so each row "
        "is being treated as everything produced up to that month and differenced into "
        f"monthly volumes. Cumulative to date is **{vol(cum_to_date)}** — the last figure "
        "in the column, not the sum of it. Set **Volumes are** in the sidebar to "
        "*Per period* if that is wrong.")
if s_full.dayfirst:
    st.caption(
        f"Dates read as **day-first** (DD/MM/YYYY), giving {s_full.dates.iloc[0]:%b %Y} to "
        f"{s_full.dates.iloc[-1]:%b %Y} across {len(s_full.t)} months. Read the other way "
        "these month-starts would collapse into a handful of Januaries.")

# Computed BEFORE the tabs, because the Summary tab needs it and Summary has to be
# written before any block that can call st.stop() -- which halts the whole script,
# not just its tab. The oil material balance hands off and stops; everything
# appended after that never ran.
band = None
if nboot:
    with st.spinner(f"Bootstrapping {nboot} replicates…"):
        dist = _boot(primary_key, t_tup, q_tup, tuple(sorted(primary["fit"].p.items())),
                     qab, nboot, horizon,
                     dmin if primary_key == "modhyp" else None, seed)
    band = dca.pxx(dist) if len(dist) else None
shown_eur = primary["eur"].eur

tab_s, tab_f, tab_d, tab_w, tab_v, tab_m, tab_t = st.tabs(
    ["Summary", "Forecast", "Diagnostics", "Water", "In place", "Material balance", "Data"])
# ---------------------------------------------------------------------------
# Summary
#
# Written last, rendered first. Streamlit tab containers are positional, not
# sequential, so this fills the tab declared at the top of the list using values
# every other tab has already finished computing -- including the in-place volume
# handed over from Material balance, which does not exist until that block runs.
# ---------------------------------------------------------------------------

with tab_s:
    P = primary["fit"].p
    # At the START OF THE FIT WINDOW, never at t = 0. For several models t = 0 is
    # outside the fitted domain, and Duong has a pole there -- it reported 802
    # Mcf/d against a well doing 10,300, and a decline of -1.5e187 %/yr.
    fit_t0 = float(s.t[0])
    q_initial = float(np.asarray(dca.rate(primary_key, P, fit_t0)).reshape(-1)[0])
    d_initial = dca.effective_decline(primary_key, P, fit_t0)
    d_now = dca.effective_decline(primary_key, P, float(s.t[-1]))
    b_val = P.get("b")
    # b is a property of the well, not of whichever model won AICc -- and Duong,
    # PLE and SEPD carry none. The loss ratio gives one from the data either way,
    # so it fills in when the model cannot and cross-checks when it can.
    lr = _loss_ratio(t_tup, q_tup)
    b_data = lr.b if (lr is not None and lr.reliable) else None
    eur_val = band["P50"] if band else shown_eur
    ip_label = "GIIP" if IS_GAS else "STOIIP"

    title = " · ".join(label_bits) if label_bits else "This well"
    st.subheader(title)
    st.caption(
        f"{fluid} · {len(s.t)} months in the fit window of {n_all} on file · "
        f"{s_full.dates.iloc[0]:%b %Y} to {s_full.dates.iloc[-1]:%b %Y} · "
        f"model **{MODELS[primary_key].name}**, R² {primary['fit'].r2:.4f}")

    k = st.columns(4)
    k[0].metric("Initial forecast rate", f"{dsp(q_initial):,.4g} {unit}/d",
                help="The fitted curve at the start of the fit window, not the first "
                     "row in the file and not extrapolated back to first production.")
    if b_val is not None:
        b_show, b_sub = f"{b_val:.2f}", (f"loss ratio {b_data:.2f}" if b_data is not None
                                         else "loss ratio not determinable")
    elif b_data is not None:
        b_show, b_sub = f"{b_data:.2f}", "from the loss ratio; this model has no b"
    else:
        b_show, b_sub = "—", f"{MODELS[primary_key].name} has no b, and the loss ratio "
        b_sub += "is not determinable"
    k[1].metric("b", b_show, delta=b_sub, delta_color="off",
                help="Arps shape parameter. Duong, PLE and SEPD carry none, so the "
                     "loss-ratio estimate off the data stands in — and cross-checks the "
                     "fitted value when there is one.")
    def _decl(d):
        # A negative effective decline is a rate that RISES over the year ahead.
        # Printing it as "-394.5 %/yr" is arithmetically true and unreadable; the
        # multiple is the same fact in a form someone can act on.
        if not math.isfinite(d):
            return "—"
        return f"{100 * d:,.1f} %/yr" if d >= 0 else f"rising ×{1 - d:,.2f}"

    k[2].metric("Initial decline", _decl(d_initial),
                delta=f"now {_decl(d_now)}", delta_color="off",
                help="Effective decline over the year ahead — 1 − q(t+12)/q(t), the "
                     "reserves-report definition, taken at the start of the fit window.")
    k[3].metric("Well life", f"{primary['eur'].years:.1f} yr")

    k2 = st.columns(4)
    k2[0].metric("Cumulative production", vol(cum_to_date))
    k2[1].metric("Reserves (remaining)", vol(primary["remaining"]),
                 help="From the end of the fit window to the abandonment rate.")
    k2[2].metric("EUR" + (" — P50" if band else ""), vol(eur_val),
                 delta=(f"P90 {vol(band['P90'])} · P10 {vol(band['P10'])}" if band else None),
                 delta_color="off")
    k2[3].metric(ip_label, vol(in_place) if in_place else "—",
                 help="Set it in the sidebar, or hand one over from Material balance."
                      if not in_place else None)

    if math.isfinite(d_initial) and d_initial < 0:
        st.warning(
            f"**{MODELS[primary_key].name} forecasts a RISING rate** at the start of the "
            f"fit window — ×{1 - d_initial:,.2f} over the first year. A decline "
            "model that does not decline is fitting something other than depletion: "
            "check the Diagnostics tab before any number on this page is used.")

    # The reconciliation, not just the numbers. Four quantities that have to agree
    # with each other, and the one line that says whether they do.
    if in_place:
        rf_now = 100.0 * cum_to_date / in_place
        rf_eur = 100.0 * eur_val / in_place
        st.markdown(
            f"**Recovery.** {rf_now:,.1f}% of {ip_label} produced to date, "
            f"{rf_eur:,.1f}% by end of life against a {max_rf:.0f}% ceiling.")
        if wv.verdict:
            (st.error if rf_eur > max_rf else st.success)(wv.verdict)
    else:
        mult, _v = dca.extrapolation_multiple(primary["remaining"], cum_to_date)
        st.info(
            f"**No {ip_label} set**, so nothing checks this forecast against the tank. "
            f"The extrapolation multiple is {mult:.2f}× — "
            + ("under 0.5×, so decline alone is defensible here."
               if mult < 0.5 else
               "at that level an independent volume is worth having."))

    rows = [
        ("Well", " · ".join(label_bits) if label_bits else ""),
        ("Fluid", fluid),
        ("Model", MODELS[primary_key].name),
        ("Fit R2 (log q)", f"{primary['fit'].r2:.4f}"),
        (f"Initial forecast rate ({unit}/d)", f"{dsp(q_initial):.6g}"),
        ("b (fitted)", f"{b_val:.4f}" if b_val is not None else ""),
        ("b (loss ratio)", f"{b_data:.4f}" if b_data is not None else ""),
        ("Initial effective decline (%/yr)",
         f"{100 * d_initial:.3f}" if math.isfinite(d_initial) else ""),
        ("Current effective decline (%/yr)",
         f"{100 * d_now:.3f}" if math.isfinite(d_now) else ""),
        (f"Cumulative production ({unit})", f"{dsp(cum_to_date):.6g}"),
        (f"Reserves, remaining ({unit})", f"{dsp(primary['remaining']):.6g}"),
        (f"EUR ({unit})", f"{dsp(eur_val):.6g}"),
        (f"EUR P90 ({unit})", f"{dsp(band['P90']):.6g}" if band else ""),
        (f"EUR P10 ({unit})", f"{dsp(band['P10']):.6g}" if band else ""),
        (f"{ip_label} ({unit})", f"{dsp(in_place):.6g}" if in_place else ""),
        (f"{ip_label} basis", ip_source if in_place else ""),
        ("Well life (yr)", f"{primary['eur'].years:.2f}"),
        (f"Abandonment rate ({unit}/d)", f"{qab_disp:.6g}"),
    ]
    summary = pd.DataFrame(rows, columns=["Quantity", "Value"])
    st.dataframe(summary, width="stretch", hide_index=True, height=420)
    st.download_button(
        "Download the summary as CSV", summary.to_csv(index=False).encode(),
        file_name=(("_".join(b.replace(" ", "-") for b in label_bits) or "well")
                   + "_summary.csv"),
        mime="text/csv")
    st.caption(
        f"Volumes in {unit}, rates in {unit}/d. Every figure is the one shown on its own "
        "tab — this page restates, it does not recompute.")



# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------

with tab_f:

    # Four columns, not six: at six the values truncate to "321.4 M..." on a laptop.
    c = st.columns(4)
    c[0].metric("Cumulative to date", vol(cum_to_date))
    c[1].metric("Remaining (decline)", vol(primary["remaining"]),
                help="From the end of the fit window, not the last row in the file.")
    c[2].metric("EUR — P50" if band else "EUR",
                vol(band["P50"] if band else shown_eur),
                delta=(f"P90 {vol(band['P90'])}  ·  P10 {vol(band['P10'])}"
                       if band else None), delta_color="off",
                help="P90 and P10 under the reserves exceedance convention, so P90 is the "
                     "LOW case." if band else None)
    c[3].metric("Well life", f"{primary['eur'].years:.1f} yr")

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
            f"The last fitted rate is {dsp(s.q[-1]):,.4g} {unit}/d, at or below the "
            f"abandonment rate of {qab_disp:g} {unit}/d — so remaining reads zero and the "
            f"EUR is just what "
            "has already been produced. Lower the abandonment rate in the sidebar if this "
            "well is still economic.")

    # A fit that explains nothing must not be reported as a forecast. R2 near zero
    # in log space means the model is no better than a horizontal line -- and for a
    # flat well that is exactly what it has fitted, so the "EUR" is just the last
    # rate times the horizon, and moving the horizon moves the answer.
    flat_rate = float(s.q[-1]) / max(float(s.q[0]), 1e-9)
    if dca.not_declining(s.t, s.q, primary["fit"].r2):
        horizon_vol = float(s.q[-1]) * 365.25 * horizon
        st.error(
            f"**This well is not declining, so there is no decline to extrapolate.** "
            f"The rate is {dsp(s.q[0]):,.4g} {unit}/d at the start of the fit window and "
            f"{dsp(s.q[-1]):,.4g} {unit}/d at the end, {s.t[-1] / 12:.1f} years later — a "
            f"ratio of {flat_rate:.2f}. The best fit scores R² {primary['fit'].r2:.3f} in "
            "log space, i.e. no better than a horizontal line.\n\n"
            f"The EUR below is therefore **not a forecast**: it is the last rate held flat "
            f"for the {horizon:.0f}-year horizon ({vol(horizon_vol)}), and it moves with "
            "the horizon rather than with the reservoir. Change the maximum well life and "
            "watch it change.\n\n"
            "A flat well is usually offtake-constrained — facility, compressor, contract or "
            "quota — and its forecast is a facilities question, not a decline one. What it "
            "needs is pressure data: a flat rate with **falling** pressure is depleting "
            "behind a constraint, a flat rate with **steady** pressure has support or a "
            "large connected volume. Neither is knowable from the rate alone.")

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

    if wv.in_place:
        v = st.columns(4)
        v[0].metric(f"{'GIIP' if s.is_gas else 'STOIIP'} — this well", vol(wv.in_place),
                    help=wv.source or None)
        v[1].metric("Recovered to date", f"{wv.rf_to_date_pct:.1f}%",
                    delta=vol(wv.cum), delta_color="off")
        v[2].metric("Ultimate recovery (EUR ÷ in place)", f"{wv.rf_ultimate_pct:.1f}%",
                    delta=f"ceiling {wv.max_rf_pct:.0f}%", delta_color="off")
        v[3].metric(f"Remaining at {wv.max_rf_pct:.0f}% RF", vol(wv.capped_remaining),
                    delta=f"decline says {vol(wv.remaining)}", delta_color="off")

        if wv.rf_to_date_pct and wv.rf_to_date_pct > wv.max_rf_pct:
            st.error(f"**{wv.verdict.capitalize()}.** This well has already produced more "
                     "than the in-place volume allows. What has been produced is measured; "
                     "the in-place number is the estimate, so that is the one to revisit — "
                     "most often the drainage area is too small, or the well is draining "
                     "beyond the rock mapped to it.")
        elif wv.rf_ultimate_pct > wv.max_rf_pct:
            st.error(f"**{wv.verdict.capitalize()}.** The decline curve does not know how "
                     f"much fluid is in the ground; this says the forecast is asking for "
                     f"more than the rock holds. Either the in-place volume is too small, "
                     f"the recovery ceiling is too low, or — most often — the extrapolation "
                     f"is too optimistic. The capped remaining of "
                     f"{vol(wv.capped_remaining)} is the defensible number.")
        elif wv.rf_ultimate_pct > 0.9 * wv.max_rf_pct:
            st.warning(f"**{wv.verdict.capitalize()}.** Worth a second look at both the "
                       "extrapolation and the in-place estimate.")
        else:
            st.success(f"**{wv.verdict.capitalize()}.** The forecast and the volume in "
                       "place are consistent.")
    else:
        st.caption("No in-place volume set — the forecast is unconstrained. Set one in the "
                   "sidebar, or work one out on the **In place** tab.")

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
        fig.add_trace(go.Scatter(x=dsp(x_obs), y=dsp(s.q), mode="markers", name="observed",
                                 marker=dict(size=5, color="#14181a")))
        t_end = max(r["eur"].t_end for r in results)
        tt = np.linspace(float(s.t[0]), t_end, 400)
        for i, r in enumerate(results):
            k = r["fit"].key
            qq = dca.rate(k, r["fit"].p, tt)
            xx = ([s.cum[0] + np.array([dca.cum_between(k, r["fit"].p, float(s.t[0]), x)
                                        for x in tt])] if vs_cum else [tt])[0]
            fig.add_trace(go.Scatter(
                x=dsp(xx) if vs_cum else xx, y=dsp(qq), mode="lines", name=MODELS[k].name,
                line=dict(color=SERIES_COLOURS[i % len(SERIES_COLOURS)],
                          width=3 if k == primary_key else 1.4)))
        fig.add_hline(y=dsp(qab), line_dash="dash", line_color="#7e8286",
                      annotation_text=f"economic limit {qab_disp:.4g} {unit}/d")
        if not vs_cum:
            fig.add_vline(x=float(s.t[-1]), line_dash="dot", line_color="#7e8286",
                          annotation_text="fit end")
        lo_y = dsp(min(float(np.min(s.q)), qab)) / 3.0
        hi_y = dsp(float(np.max(s.q)))
        fig.update_yaxes(range=[math.log10(lo_y), math.log10(hi_y * 3)]
                         if logy else [0, hi_y * 1.1])
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
            hist = go.Figure(go.Histogram(x=dsp(dist), nbinsx=40, marker_color="#2a78d6"))
            for lbl, v, colr in (("P90", band["P90"], "#e34948"),
                                 ("P50", band["P50"], "#14181a"),
                                 ("P10", band["P10"], "#1baf7a")):
                hist.add_vline(x=dsp(v), line_color=colr, annotation_text=lbl)
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
            f"Model spread: {vol(min(eurs))} to {vol(max(eurs))}, a factor of "
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

        fig = go.Figure(go.Scatter(x=dsp(s.cum), y=dsp(s.q), mode="markers",
                                   name="rate vs Np",
                                   marker=dict(size=5, color="#1baf7a")))
        st.plotly_chart(theme_axes(fig, f"cumulative ({unit})", f"rate ({unit}/d)",
                                   height=330), width="stretch")
        st.caption("**q vs Np.** A straight line is exponential decline — tank-like "
                   "depletion. Scatter with no trend usually means the well is "
                   "rate-constrained rather than declining.")

    with g2:
        pos = s.t > 0
        fig = go.Figure(go.Scatter(x=np.sqrt(s.t[pos]), y=1.0 / dsp(s.q[pos]),
                                   mode="markers", name="1/q",
                                   marker=dict(size=5, color="#4a3aa7")))
        st.plotly_chart(theme_axes(fig, "√t (months^½)", f"1/q (d/{unit})", height=330),
                        width="stretch")
        st.caption("**1/q vs √t.** Straight means transient linear flow, i.e. "
                   "fracture-dominated and not yet at the boundaries.")

        fig = go.Figure(go.Scatter(x=s.t[pos], y=dsp(s.q[pos]), mode="markers",
                                   name="rate", marker=dict(size=5, color="#eda100")))
        if slope is not None:
            tt = s.t[pos]
            ref = dsp(float(s.q[pos][0])) * (tt / tt[0]) ** -0.5
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
            m[1].metric("Water produced", liq(wa.cum_water))
            m[2].metric(f"Np at {ec:.0%} — semilog",
                        # vol(), not fmt(): its neighbour is the same quantity by a
                        # different route, and 14,002,387 beside 14.11 MMbbl reads as
                        # a disagreement when the two actually agree to 0.8%.
                        "—" if weak_s else vol(wa.np_semilog),
                        delta=None if not wa.semilog else f"R² {wa.semilog['r2']:.3f}",
                        delta_color="off")
            m[3].metric(f"Np at {ec:.0%} — X-plot",
                        "—" if weak_x else vol(wa.np_xplot),
                        delta=None if not wa.xplot else f"R² {wa.xplot['r2']:.3f}",
                        delta_color="off")
            rem_w = (wa.np_semilog - wa.cum_oil) if not weak_s else float("nan")
            m[4].metric("Remaining oil", "—" if weak_s else vol(max(rem_w, 0)))

            if past_s:
                st.warning(
                    f"**This well is already past {ec:.0%} water cut.** The WOR trend "
                    f"(R² {wa.semilog['r2']:.3f}) puts Np at that limit at "
                    f"{vol(wa.np_semilog)}, below the {vol(wa.cum_oil)} already "
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
                st.info(f"The WOR trend gives {vol(wa.np_semilog)} against "
                        f"{vol(arps)} from the {MODELS[primary_key].name} case — "
                        f"{rel}. A wide gap is information: the rate decline and the water "
                        "trend are telling different stories about what kills this well.")

            wor = s.water / s.volume
            g1, g2 = st.columns(2)
            with g1:
                fig = go.Figure(go.Scatter(x=dsp(s.cum), y=wor, mode="markers",
                                           marker=dict(size=5, color="#2a78d6")))
                if wa.semilog and not weak_s:
                    xs = np.array([s.cum[0], max(wa.np_semilog, s.cum[-1])])
                    fig.add_trace(go.Scatter(
                        x=dsp(xs), y=np.exp(wa.semilog["slope"] * xs + wa.semilog["intercept"]),
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
                fig = go.Figure(go.Scatter(x=dca.x_of(wor[sel]), y=dsp(s.cum[sel]),
                                           mode="markers",
                                           marker=dict(size=5, color="#1baf7a")))
                if wa.xplot and not weak_x:
                    xs = np.linspace(float(dca.x_of(max(wor[sel].min(), 1.01))),
                                     float(dca.x_of(ec / (1 - ec))), 20)
                    fig.add_trace(go.Scatter(
                        x=xs, y=dsp(wa.xplot["slope"] * xs + wa.xplot["intercept"]),
                        mode="lines",
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
# In place -- the well's own volume, and the four numbers reconciled
# ---------------------------------------------------------------------------

with tab_v:
    st.subheader(f"{'Gas' if s.is_gas else 'Oil'} in place — this well")
    st.markdown(
        "This is the volume **this well** is connected to and can drain, not the "
        "reservoir's total. The distinction is the whole point: a single-well forecast "
        "can only be checked against a single-well volume. Summing well forecasts against "
        "a reservoir number is valid under volumetric depletion and fails badly under a "
        "shared aquifer — one string in the sample water-drive reservoir forecasts "
        "145.7 Bcf remaining on its own history where the entire reservoir allows "
        "roughly 8–52 Bcf.")

    st.markdown("#### The four numbers")
    if wv.in_place:
        frame = wv.as_frame()
        frame[f"Volume ({unit})"] = [vol(v) for v in frame[f"Volume ({unit})"]]
        st.dataframe(frame, width="stretch", hide_index=True)
    else:
        st.info("Set an in-place volume in the sidebar and the table fills in: in place, "
                "cumulative to date, EUR, remaining reserves, and the recovery factor each "
                "one implies.")
        st.dataframe(pd.DataFrame([
            ("In place", "— set it in the sidebar"),
            ("Cumulative to date", vol(wv.cum)),
            ("EUR (decline forecast)", vol(wv.eur)),
            ("Remaining reserves", vol(wv.remaining)),
        ], columns=["", f"Volume ({unit})"]), width="stretch", hide_index=True)

    st.markdown("#### The three routes, and what each one needs")
    r1, r2, r3 = st.columns(3)
    r1.markdown(
        "**Volumetric**\n\nDrainage area, net pay, porosity, water saturation, and an FVF.\n\n"
        "The only route that works before the well has produced anything. Its weakness is "
        "the drainage area: for a vertical well that is roughly the spacing unit, but for a "
        "horizontal it is the stimulated volume, and getting it wrong scales the answer "
        "linearly.")
    r2.markdown(
        "**Material balance**\n\nWell-level static pressures against this well's own "
        "cumulative.\n\n"
        "Gives the **connected** volume — what the well has actually shown it can reach, "
        "which is often less than the rock mapped to it. Needs buildups long enough to have "
        "stabilised. On the Material balance tab.")
    r3.markdown(
        "**From the decline**\n\nNothing but the production history.\n\n"
        "Under boundary-dominated flow at roughly constant bottomhole pressure, rate falls "
        "linearly with cumulative and the intercept is the movable volume. Most wells have "
        "a production history and no surveys, which is what makes this worth having.")

    st.markdown("#### Rate against cumulative — the decline route, worked")
    mv = dca.movable_volume_from_decline(s.cum, s.q)
    if mv is None:
        st.warning("Rate is not falling with cumulative on this fit window, so there is no "
                   "line to extrapolate. That is usually a well still in transient flow, one "
                   "held on a plateau, or one whose drawdown keeps changing.")
    else:
        m = st.columns(4)
        m[0].metric("Movable volume at q → 0", vol(mv.movable))
        m[1].metric("Already produced", vol(mv.already))
        m[2].metric("Movable remaining", vol(mv.remaining_movable))
        m[3].metric("R² of the line", f"{mv.r2:.3f}",
                    delta="usable" if mv.r2 >= 0.7 else "not a line",
                    delta_color="normal" if mv.r2 >= 0.7 else "inverse")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dsp(s.cum), y=dsp(s.q), mode="markers", name="observed",
                                 marker=dict(size=5, color="#14181a")))
        # The same trailing-window line dca fitted, drawn out to its intercept.
        late = s.cum >= s.cum[int(len(s.cum) * 0.4)]
        sl, ic = np.polyfit(s.cum[late], s.q[late], 1)
        xs = np.array([float(s.cum[0]), mv.movable])
        fig.add_trace(go.Scatter(x=dsp(xs), y=dsp(sl * xs + ic), mode="lines",
                                 name="boundary-dominated trend",
                                 line=dict(color="#eb6834", dash="dash")))
        fig.add_vline(x=dsp(mv.movable), line_dash="dot", line_color="#1baf7a",
                      annotation_text=f"movable {vol(mv.movable)}")
        fig.add_vline(x=dsp(float(s.cum[-1])), line_dash="dot", line_color="#7e8286",
                      annotation_text="produced")
        st.plotly_chart(theme_axes(fig, f"cumulative ({unit})", f"rate ({unit}/d)",
                                   height=400), width="stretch")
        if mv.r2 < 0.7:
            st.warning(
                f"**R² {mv.r2:.2f} — do not use this intercept.** The points do not form a "
                "line, so extrapolating one is arithmetic without meaning. A well in "
                "transient flow has not felt its boundaries yet and has no drainage volume "
                "to read; a well whose drawdown keeps changing has a different line every "
                "few months.")
        else:
            st.caption(
                f"The intercept is what the well can **move** — recoverable to zero rate, "
                f"not what is in place, and 'zero rate' is not 'zero pressure'. Divide by a "
                f"recovery factor to get in place: at {wv.max_rf_pct:.0f}% that is "
                f"{vol(mv.movable / (wv.max_rf_pct / 100.0))}.")

    st.markdown("#### Recovery factors worth arguing with")
    st.dataframe(pd.DataFrame([
        ("Gas — volumetric depletion", "80–90%",
         "abandonment pressure is a small fraction of initial, so most of the gas comes out"),
        ("Gas — water drive", "50–70%",
         "water reaches the wells and the reservoir is abandoned with gas still in it"),
        ("Oil — solution gas drive", "5–15%",
         "the only energy is dissolved gas coming out of solution"),
        ("Oil — gas cap expansion", "20–40%", "an expanding gas cap sweeps the oil down"),
        ("Oil — water drive", "35–60%", "strong aquifer, good sweep"),
        ("Oil — waterflood", "30–50%", "depends almost entirely on sweep efficiency"),
    ], columns=["Drive", "Typical RF", "Why"]), width="stretch", hide_index=True)
    st.caption("These are ranges, not answers. The recovery factor is where most of the "
               "uncertainty in a reserves number actually lives, and it is worth more "
               "argument than the decline exponent.")


# ---------------------------------------------------------------------------
# Material balance
# ---------------------------------------------------------------------------

with tab_m:
    mb_fluid = st.radio("Fluid", ["Gas", "Oil"], horizontal=True, key="mb_fluid",
                        index=0 if s.is_gas else 1)
    if mb_fluid == "Gas":
        st.subheader("Gas material balance")
        st.caption("p/z, Havlena–Odeh drive diagnosis, and a Fetkovich aquifer fit. "
                   "Needs static pressure surveys against cumulative gas.")
    else:
        st.subheader("Oil material balance")
        st.caption("Havlena–Odeh as a straight line: F = N·Et. Needs static pressure "
                   "surveys against cumulative oil.")
    st.caption("Feed it **this well's** pressures and **this well's** cumulative and the "
               "answer is that well's connected volume. Feed it field averages and it is "
               "the reservoir's — useful, but not something a single-well forecast can be "
               "checked against.")

    pres = sorted(p for pat in ("*pressure*.csv", "*pressure*.tsv")
                  for p in glob.glob(os.path.join(DATA_DIR, pat)))
    MB_MODES = (["Sample reservoir"] if pres else []) + ["Paste", "Upload"]
    mb_src = st.radio("Pressure data", MB_MODES, horizontal=True, key="mb_src")

    MB_HELP = (
        "One row per pressure survey, in column order: **date, pressure (psia), "
        "cumulative gas (MMscf)**, then optionally **cumulative condensate (Mbbl)** and "
        "**cumulative water (Mbbl)**.\n\n"
        "Columns are taken by position rather than by name, because survey tables get "
        "titled a dozen different ways and the order is the one thing that stays put. "
        "Three surveys is the minimum; the more of the depletion history they span, the "
        "better the drive diagnosis.\n\n"
        "Static, datum-corrected pressures from builds long enough to have stabilised are "
        "what the balance wants. If all you have is **wellhead** pressure — which is the "
        "usual case — enter it anyway and set *Pressures were measured at* to Wellhead; "
        "the gas column is added for you. Flowing readings are accepted too, and the "
        "answer they give is labelled for what it is: a lower bound."
    )

    mb_path = None
    if mb_src == "Paste":
        # Same grid treatment as the production history: a survey table is a
        # table, and a text box makes the reader guess whether their clipboard
        # arrived intact. It matters more here than there, because the loader
        # takes these columns BY POSITION -- headers on the grid show the order
        # the tool expects instead of leaving it to be read in the help text.
        MB_BLANK_ROWS = 10
        GAS_U = "MMscf" if U.name == "field" else "10⁶m³"
        LIQ_U = "Mbbl" if U.name == "field" else "10³m³"
        mb_cols = ("Date", f"Pressure ({U.pressure})", f"Cum gas ({GAS_U})",
                   f"Cum condensate ({LIQ_U})", f"Cum water ({LIQ_U})")
        mb_template = pd.DataFrame({c: pd.Series([""] * MB_BLANK_ROWS, dtype="string")
                                    for c in mb_cols})

        st.markdown("#### Paste your pressure surveys")
        st.caption(
            "Click the first cell and paste — a whole range at once is fine. Rows are "
            "added as you need them. Only the first three columns are required; leave "
            "condensate and water blank if you do not have them.")

        # Nonce in the key, for the same reason as the production grid: deleting a
        # data_editor's key does not clear it, because the edits live in the
        # frontend and are replayed onto whatever frame is passed in.
        mnonce = st.session_state.get("mb_grid_nonce", 0)
        mb_grid = st.data_editor(
            mb_template, key=f"mb_paste_grid_{mnonce}", num_rows="dynamic",
            width="stretch", height=360,
            column_config={
                "Date": st.column_config.TextColumn(
                    "Date", help="Any recognisable format: 2018-06-01, 06/01/2018, "
                                 "Jun 2018.", width="medium"),
                mb_cols[1]: st.column_config.TextColumn(
                    mb_cols[1], help="Static, datum-corrected if you have it. Wellhead "
                                     "readings are fine — set *Pressures were measured "
                                     "at* below."),
                mb_cols[2]: st.column_config.TextColumn(
                    mb_cols[2], help="Cumulative GAS produced at the survey date, not "
                                     "the volume that month."),
                mb_cols[3]: st.column_config.TextColumn(
                    mb_cols[3], help="Cumulative condensate. Optional, but omitting it "
                                     "biases OGIP low."),
                mb_cols[4]: st.column_config.TextColumn(
                    mb_cols[4], help="Cumulative water. Optional."),
            })

        def _mb_has(col):
            return mb_grid[col].astype("string").fillna("").str.strip().ne("")

        mb_filled = mb_grid[_mb_has(mb_cols[0]) & _mb_has(mb_cols[1]) & _mb_has(mb_cols[2])]

        mc1, mc2 = st.columns([3, 1])
        mc1.caption(f"**{len(mb_filled)}** survey{'' if len(mb_filled) == 1 else 's'} with "
                    "a date, a pressure and a cumulative. Three is the minimum.")
        if mc2.button("Clear the table", use_container_width=True, key="mb_clear"):
            st.session_state["mb_grid_nonce"] = mnonce + 1
            st.session_state.pop(f"mb_paste_grid_{mnonce}", None)
            st.rerun()

        with st.expander("Paste as text instead"):
            st.caption("If the grid will not take your clipboard, paste the raw block "
                       "here — tab, comma or semicolon separated, header row optional.")
            mtxt = st.text_area(
                "Raw paste", height=150, key="mb_paste", label_visibility="collapsed",
                placeholder="Date\tP (psia)\tCum gas (MMscf)\tCum cond (Mbbl)\tCum water (Mbbl)\n"
                            "2015-09-01\t5496.2\t253.6\t19.6\t0\n"
                            "2016-11-01\t5477.0\t1030.8\t64.9\t0.006\n"
                            "2019-11-01\t5141.4\t14949.5\t1305.7\t17.034")

        # The grid wins when it has data, matching the production side: text left
        # behind in a collapsed expander must not override what is on screen.
        if len(mb_filled) >= 3:
            mb_path = mb_filled.reset_index(drop=True)
            if (mtxt or "").strip():
                st.info("Using the table. There is also text in **Paste as text "
                        "instead** — clear the table if you meant to use that.")
        else:
            mb_path = mtxt if (mtxt or "").strip() else None
    elif mb_src == "Upload":
        mup = st.file_uploader(
            "Pressure survey file", type=["csv", "txt", "tsv", "xlsx", "xlsm", "xls"],
            key="mbup", help="CSV, tab-separated text, or an Excel workbook.")
        mb_path = io.BytesIO(mup.getvalue()) if mup else None
    else:
        pn = [os.path.basename(p) for p in pres]
        mb_path = os.path.join(DATA_DIR, st.selectbox("Reservoir", pn, key="reservoir")) if pn else None

    if mb_path is None:
        st.markdown(MB_HELP)
    else:
        try:
            pdf = dca.load_pressure_csv(mb_path)
        except (ValueError, FileNotFoundError) as e:
            st.error(f"**Could not read those surveys.** {e}")
            st.markdown(MB_HELP)
            st.stop()

        # One conversion point for the survey table, matching the production side.
        # Column order is date, pressure, cumulative, condensate, water -- so the
        # cumulative column is 10^6 m3 in metric against MMscf in field, and the
        # liquid columns are 10^3 m3 against Mbbl.
        if U.name != "field":
            # 1e6 m3 = 1000 x (1e3 m3) = 1000 x 35.3147 Mcf = 35,314.7 Mcf = 35.3147 MMscf
            MMSCF_PER_E6M3 = dca.E3M3_TO_MCF
            pdf = pdf.copy()
            pdf["P"] = U.pressure_to_psia(pdf["P"])
            pdf["Gp"] = pdf["Gp"] * MMSCF_PER_E6M3
            pdf["condensate"] = pdf["condensate"] * dca.M3_TO_BBL
            pdf["water"] = pdf["water"] * dca.M3_TO_BBL
            st.caption(f"Surveys read as {U.pressure}, 10⁶m³ cumulative gas and 10³m³ "
                       "liquids, converted to field units for the balance.")
        if mb_fluid == "Oil":
            oc = st.columns(4)
            o_t_d = unum(oc[0], f"Temperature ({U.temperature})", "omb_t",
                         U.temperature_from_degf(200.0), min_value=-50.0, max_value=500.0,
                         step=5.0, format="%.4g")
            o_t = U.temperature_to_degf(o_t_d)
            o_api = oc[1].number_input("Oil API", 5.0, 60.0, 35.0, step=1.0, key="omb_api")
            o_sg = oc[2].number_input("Gas gravity", 0.55, 1.2, 0.75, step=0.01, key="omb_sg")
            o_pb_d = unum(oc[3], f"Bubble point ({U.pressure})", "omb_pb",
                          U.pressure_from_psia(2500.0), min_value=1.0, max_value=1e5,
                          step=100.0, format="%.6g",
                                      help="Above it the oil is undersaturated and produces "
                                           "by rock and fluid expansion alone.")
            o_pb = U.pressure_to_psia(o_pb_d)
            oc2 = st.columns(4)
            o_sw = oc2[0].number_input("Connate Sw (frac)", 0.0, 0.9, 0.25, step=0.05,
                                       key="omb_sw")
            o_cf = oc2[1].number_input("Rock compressibility (1/psi ×1e-6)", 0.5, 50.0, 4.0,
                                       step=0.5, key="omb_cf",
                                       help="An undersaturated balance is very sensitive to "
                                            "this. 3–6e-6 is typical for consolidated "
                                            "sandstone.") * 1e-6
            o_cw = oc2[2].number_input("Water compressibility (1/psi ×1e-6)", 1.0, 10.0, 3.0,
                                       step=0.5, key="omb_cw") * 1e-6
            o_m = oc2[3].number_input("Gas cap ratio m", 0.0, 10.0, 0.0, step=0.1,
                                      key="omb_m",
                                      help="Gas-cap pore volume ÷ oil-zone pore volume. "
                                           "Zero for no initial gas cap.")

            # The second column is pressure and the third the cumulative -- for oil that
            # third column is Np in STB, so the loader's Gp is reused as-is.
            try:
                omb = dca.oil_material_balance(
                    pdf["P"], pdf["Gp"], o_t, o_api, o_sg, o_pb,
                    wp_stb=pdf["water"] * 1000.0 if pdf["water"].abs().sum() else None,
                    cw=o_cw, cf=o_cf, sw=o_sw, gas_cap_m=o_m)
            except ValueError as e:
                st.error(f"**Could not run the balance.** {e}")
                st.stop()

            om = st.columns(4)
            om[0].metric("STOIIP", liq(omb.stoiip), delta=f"R² {omb.r2:.4f}",
                         delta_color="off")
            om[1].metric("Produced", liq(omb.np_now),
                         delta=f"{100 * omb.np_now / omb.stoiip:.1f}% recovered"
                         if omb.stoiip > 0 else None, delta_color="off")
            om[2].metric("Drive", omb.drive.split(" or ")[0].title())
            om[3].metric("F/Et rise", f"{omb.rising_f_over_et:.2f}×")

            if omb.water_drive_pct > 25:
                st.warning(
                    f"**F/Et climbs {omb.rising_f_over_et:.2f}×**, so roughly "
                    f"{omb.water_drive_pct:.0f}% of the withdrawal is being met by something "
                    "outside the oil and its rock — an aquifer, an injector, or a connected "
                    "compartment. The STOIIP above is then an **upper bound**, not a volume: "
                    "influx and a larger tank look identical to a straight line. In a "
                    "synthetic test, adding influx inflated a known 8 MMSTB to 301 MMSTB.")
            elif not omb.saturated:
                st.info(
                    f"**Undersaturated depletion** — every survey is above the "
                    f"{o_pb_d:,.6g} {U.pressure} bubble point, so the drive is rock and fluid "
                    "expansion alone. Two consequences worth stating in any report: recovery "
                    "factors here are small (5–10% is normal), and the STOIIP is very "
                    "sensitive to rock compressibility, because it is the denominator. Move "
                    "cf from 4 to 6e-6 and watch the answer move with it.")
            else:
                st.success(
                    f"**Solution-gas drive** — pressure has fallen below the {o_pb_d:,.6g} {U.pressure} "
                    f"bubble point, so gas is coming out of solution and doing the work. "
                    f"{omb.depletion_drive_pct:.0f}% of the withdrawal is accounted for by "
                    "oil, dissolved gas and rock expansion.")

            g1, g2 = st.columns(2)
            with g1:
                fig = go.Figure(go.Scatter(x=omb.points["Et"], y=omb.points["F"],
                                           mode="markers", name="surveys",
                                           marker=dict(size=9, color="#2a78d6")))
                xs = np.array([0.0, float(omb.points["Et"].max())])
                fig.add_trace(go.Scatter(x=xs, y=omb.stoiip * xs, mode="lines",
                                         name=f"N = {liq(omb.stoiip)}",
                                         line=dict(color="#eb6834", dash="dash")))
                st.plotly_chart(theme_axes(fig, "Et — total expansion (rb/STB)",
                                           "F — net withdrawal (rb)", height=360),
                                width="stretch")
                st.caption("**Havlena–Odeh.** F against Et is a straight line through the "
                           "origin whose slope *is* N. Curvature upward means influx; the "
                           "line is then fitting energy that did not come from the oil.")
            with g2:
                fig = go.Figure(go.Scatter(x=omb.points["Np"], y=omb.points["F_over_Et"],
                                           mode="lines+markers", name="F/Et",
                                           line=dict(color="#2a78d6")))
                fig.add_hline(y=float(omb.points["F_over_Et"].iloc[0]), line_dash="dash",
                              line_color="#7e8286", annotation_text="first point")
                st.plotly_chart(theme_axes(fig, "cumulative oil (STB)", "F / Et (STB)",
                                           height=360), width="stretch")
                st.caption("Flat means the oil and its rock are supplying all the energy, "
                           "and the level *is* N. Rising means they are not.")

            st.dataframe(
                omb.points[["P", "Np", "Bo", "Rs", "Et", "F", "F_over_Et"]]
                .rename(columns={"P": "Pressure (psia)", "Np": "Cum oil (STB)",
                                 "Bo": "Bo (rb/STB)", "Rs": "Rs (scf/STB)",
                                 "Et": "Et (rb/STB)", "F": "F (rb)",
                                 "F_over_Et": "F/Et (STB)"})
                .style.format("{:,.4g}"), width="stretch", hide_index=True)

            oil_mismatch = IS_GAS
            if oil_mismatch:
                st.caption("⚠️ The well is set to **gas** in the sidebar but this is an "
                           "**oil** balance, so the answer cannot be sent across — it "
                           "would land in the wrong units.")
            if st.button(f"Use {liq(omb.stoiip)} as this well's STOIIP",
                         type="primary", key="send_oil", disabled=oil_mismatch):
                st.session_state["ip_from_mb"] = omb.stoiip
                st.session_state["ip_from_mb_src"] = (
                    f"oil material balance · R² {omb.r2:.3f} · {omb.drive}")
                st.rerun()
            st.caption("Sends it to the sidebar, where it becomes the in-place volume the "
                       "forecast is checked against. Only meaningful if these pressures and "
                       "this cumulative are **this well's**.")
            st.stop()

        c = st.columns(6)
        temp_d = unum(c[0], f"Temperature ({U.temperature})", "mb_t",
                      U.temperature_from_degf(230.0), min_value=-50.0, max_value=500.0,
                      step=1.0, format="%.4g")
        temp = U.temperature_to_degf(temp_d)
        sg = c[1].number_input("Gas gravity", 0.55, 1.2, 0.72, step=0.005, format="%.4f", key="mb_sg")
        api = c[2].number_input("Condensate API (0 = none)", 0.0, 90.0, 0.0, step=1.0,
                                key="mb_api",
                                help="Converts produced condensate to gas equivalent. "
                                     "Omitting it biases OGIP low and remaining high.")
        pab_d = unum(c[3], f"Abandonment pressure ({U.pressure})", "mb_pab",
                     U.pressure_from_psia(1000.0), min_value=1.0, max_value=1e5,
                     step=50.0, format="%.6g")
        pab = U.pressure_to_psia(pab_d)
        skip = c[4].number_input("Drop earliest surveys", 0, max(len(pdf) - 3, 0), 0,
                                 key="mb_skip",
                                 help="Use this when the consistency guard fires and the "
                                      "first survey predates a reliable reference.")
        # Usually nobody has this. Production starts in May and the first build is
        # run in October, so the earliest survey is already down the depletion
        # line. Left at 0 the balance extrapolates p/z back to zero cumulative and
        # uses that, which is what makes the answer ORIGINAL gas in place rather
        # than gas in place on the day of the first survey.
        pi_known_d = unum(c[5], f"Initial pressure ({U.pressure})", "mb_pi",
                          0.0, min_value=0.0, max_value=2e5, step=50.0, format="%.6g",
                          help="Leave at 0 and it is estimated from the surveys. Set it "
                               "only if a real initial reservoir pressure is known.")
        pi_known = U.pressure_to_psia(pi_known_d) if pi_known_d > 0 else None
        ge = dca.condensate_gas_equivalent(api) if api > 8.9 else 0.0

        # --- wellhead pressures -------------------------------------------
        # Most operators have tubing-head pressure and no datum-corrected
        # survey at all, so refusing anything but a datum pressure means
        # refusing the data that actually exists. The gas column is cheap to
        # add and the assumptions are statable, so state them.
        where = st.radio(
            "Pressures were measured at", ["Datum (bottomhole)", "Wellhead"],
            horizontal=True, key="mb_where",
            help="A wellhead reading is lighter than the datum pressure by the weight "
                 "of the gas column above it — roughly 0.02–0.13 psi/ft. Uncorrected, "
                 "it reads as a reservoir that was never as full as it was.")
        if where == "Wellhead":
            wc = st.columns(4)
            depth_d = unum(wc[0], f"Datum depth ({U.length})", "mb_depth",
                           10000.0 / U.length_to_ft, min_value=1.0, max_value=1e5,
                           step=100.0, format="%.6g",
                           help="True vertical depth of the datum the balance is "
                                "referenced to — mid-perforations or the gas–water "
                                "contact, not measured depth along a deviated hole.")
            depth_ft = depth_d * U.length_to_ft
            twh_d = unum(wc[1], f"Wellhead temperature ({U.temperature})", "mb_twh",
                         U.temperature_from_degf(90.0), min_value=-50.0, max_value=400.0,
                         step=5.0, format="%.4g")
            t_wh = U.temperature_to_degf(twh_d)
            kind = wc[2].radio("Readings are", ["Shut-in", "Flowing"], key="mb_pkind",
                               help="Shut-in readings become datum pressures the balance "
                                    "can use directly. Flowing readings carry drawdown "
                                    "with them and bias the answer low.")
            tub = wc[3].number_input("Tubing ID (in)", 0.5, 9.0, 2.441, step=0.005,
                                     format="%.3f", key="mb_tub",
                                     disabled=(kind == "Shut-in"),
                                     help="Only the flowing case needs it — friction "
                                          "scales as 1/d⁵, so this is not a detail.")

            raw = pdf["P"].to_numpy(dtype=float)
            if kind == "Shut-in":
                pdf = pdf.copy()
                pdf["P"] = [dca.static_gas_column(p, depth_ft, t_wh, temp, sg)
                            for p in raw]
            else:
                # Flowing needs the rate at each survey. Take it from this
                # well's own history by date rather than asking twice.
                qs = np.full(len(raw), float(np.median(s.q)))
                try:
                    sd = pd.to_datetime(s.dates).to_numpy().astype("datetime64[D]")
                    pd_ = pd.to_datetime(pdf["date"]).to_numpy().astype("datetime64[D]")
                    for i, d in enumerate(pd_):
                        qs[i] = float(s.q[int(np.argmin(np.abs(sd - d)))])
                except (KeyError, ValueError, TypeError):
                    st.caption("Survey dates could not be matched to the production "
                               f"history, so the median rate ({dsp(qs[0]):,.4g} "
                               f"{unit}/d) was used for every point.")
                pdf = pdf.copy()
                pdf["P"] = [dca.flowing_bhp(p, q, depth_ft, tub, t_wh, temp, sg)
                            for p, q in zip(raw, qs)]
                st.warning(
                    "**Flowing pressures bias this answer low.** A flowing bottomhole "
                    "pressure is the reservoir pressure minus the drawdown, so every "
                    "point on the p/z plot sits below where the reservoir actually is. "
                    "If the rate has been steady the offset is roughly steady too, the "
                    "line stays straight, and it shifts down — which moves the intercept "
                    "left. **Read the OGIP below as a lower bound**, not an estimate. It "
                    "also assumes the tubing is carrying gas alone: once a well starts "
                    "loading up with water, the column is heavier than this model and the "
                    "computed pressure falls for a reason that is not depletion.")

            lift = float(np.mean(pdf["P"].to_numpy(dtype=float) - raw))
            st.caption(
                f"Corrected from wellhead to datum through a {sg:.3f}-gravity gas column "
                f"over {depth_d:,.6g} {U.length}: **+{U.pressure_from_psia(lift):,.4g} "
                f"{U.pressure} on average** ({lift / depth_ft:.4f} psi/ft), "
                f"{U.pressure_from_psia(float(raw[0])):,.6g} → "
                f"{U.pressure_from_psia(float(pdf['P'].iloc[0])):,.6g} {U.pressure} at the "
                f"first survey. Average temperature and z, iterated — within about 0.5% of "
                "a stepwise integration.")

        try:
            r = _mb(tuple(pdf["P"]), tuple(pdf["Gp"]), tuple(pdf["condensate"]),
                    tuple(pdf["water"]), temp, sg, ge, pab, int(skip), pi_known)
        except ValueError as e:                          # noqa: BLE001
            st.error(str(e))
            st.stop()

        if math.isfinite(r.p_initial):
            gap = float(pdf["Gp"].min())
            st.caption(
                (f"Initial pressure **{U.pressure_from_psia(r.p_initial):,.6g} "
                 f"{U.pressure}**, as entered."
                 if r.p_initial_known else
                 f"Initial pressure taken as **{U.pressure_from_psia(r.p_initial):,.6g} "
                 f"{U.pressure}** at zero cumulative"
                 + (f", estimated because the earliest survey is already "
                    f"{gap / 1000:,.2f} Bcf into the history."
                    if gap > 0.01 else "."))
                + " Volumes below are **original** gas in place, not gas in place at the "
                  "first survey.")

        m = st.columns(5)
        m[0].metric("OGIP (p/z)", f"{r.ogip_pz / 1000:,.1f} Bcf", delta=f"R² {r.r2:.4f}",
                    delta_color="off")
        m[1].metric("Produced", f"{r.gp_now / 1000:,.1f} Bcf")
        m[2].metric("Recoverable", f"{r.recoverable / 1000:,.1f} Bcf",
                    delta=f"RF {r.recovery_factor:.0f}%", delta_color="off")
        m[3].metric("Remaining (p/z)", f"{r.remaining / 1000:,.1f} Bcf")
        m[4].metric("F/Eg rise", f"{r.ho_rise:.2f}×" if math.isfinite(r.ho_rise) else "—")

        # Remaining is measured from the LAST SURVEY, because that is the last
        # point the pressure line actually knows about. If the well has kept
        # producing since, that volume is already gone and remaining overstates
        # what is left by exactly the gap.
        if IS_GAS:
            since = float(s.cum[-1]) / 1000.0 - r.gp_now        # MMscf
            if since > 0.01 * max(r.gp_now, 1.0):
                st.caption(
                    f"**Remaining is measured from the last survey**, at "
                    f"{r.gp_now / 1000:,.2f} Bcf. This well has produced "
                    f"{since / 1000:,.2f} Bcf since then, so counted from today's "
                    f"{float(s.cum[-1]) / 1e6:,.2f} Bcf the remaining volume is "
                    f"**{(r.remaining - since) / 1000:,.1f} Bcf**. The pressure line "
                    "knows nothing past its last point; only the production does.")

        if r.impossible:
            st.error(
                f"**Consistency guard fired.** Material balance requires G ≤ min(F/Eg) = "
                f"{r.min_f_over_eg:,.1f} Bcf, because We ≥ 0 — but {r.gp_now / 1000:,.1f} Bcf "
                "has already been produced. The surveys and the volumes cannot both be right, "
                "and no aquifer model rescues that. The usual cause is a reference pressure "
                "taken **after** first production: pi is then too low, every Eg downstream too "
                "small, and F/Eg too large everywhere. Try dropping the earliest survey above.")
        elif r.volumetric and r.ho_rise <= 1.10:
            st.success(
                f"**Volumetric depletion.** F/Eg is flat ({r.ho_rise:.2f}× across the record), "
                "so the gas is producing by its own expansion and the p/z intercept is a real "
                "number. Recovery factors of 80–90% are normal here.")
        elif r.volumetric:
            # 1.10-1.25 is not flat. Calling it flat because it cleared a
            # threshold hides the fact that the threshold nearly caught it.
            st.warning(
                f"**Borderline — F/Eg climbs {r.ho_rise:.2f}×.** That is under the 1.25× at "
                "which this tool calls water drive, but it is not flat either, and a rise this "
                "size is what early or weak pressure support looks like before it becomes "
                "obvious. Treat the p/z intercept as an upper bound, check whether produced "
                "water is accelerating, and prefer a recovery factor nearer 70% than 90% until "
                "another survey settles which way it is going.")
        else:
            st.warning(
                f"**Water drive or pressure support.** F/Eg climbs {r.ho_rise:.2f}×, so "
                "something outside the gas is supplying energy. The p/z intercept above is an "
                "**artefact, not a volume** — a water-driven reservoir holds pressure up, "
                "which flattens the trend and inflates the intercept. Use the aquifer fit "
                "below and a recovery factor of 50–70%.")

        # We >= 0 makes min(F/Eg) a ceiling on G, and it binds whether or not
        # the drive verdict came out volumetric. A p/z intercept above it is the
        # straight line reading through curvature it should have caught.
        if (math.isfinite(r.min_f_over_eg) and math.isfinite(r.ogip_pz)
                and r.ogip_pz / 1000.0 > r.min_f_over_eg * 1.10 and not r.impossible):
            over = r.ogip_pz / 1000.0 / r.min_f_over_eg
            st.error(
                f"**The p/z intercept is above what material balance allows.** Since We ≥ 0, "
                f"G ≤ min(F/Eg) = **{r.min_f_over_eg:,.1f} Bcf**, but the straight line reads "
                f"{r.ogip_pz / 1000:,.1f} Bcf — {over:.2f}× the ceiling. Take "
                f"{r.min_f_over_eg:,.1f} Bcf as the upper bound on this tank and treat the "
                "intercept as what it is: a line fitted through a trend that is curving. "
                "Recoverable volumes computed from the intercept are overstated by at least "
                "the same factor.")

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
                d = r.points
                ok = d["F_over_Eg_reliable"]
                fig = go.Figure()
                if ok.any():
                    fig.add_trace(go.Scatter(x=d.loc[ok, "Gp"], y=d.loc[ok, "F_over_Eg_Bcf"],
                                             mode="lines+markers", name="F/Eg",
                                             line=dict(color="#2a78d6")))
                    fig.add_hline(y=float(d.loc[ok, "F_over_Eg_Bcf"].iloc[0]),
                                  line_dash="dash", line_color="#7e8286",
                                  annotation_text=f"first reliable "
                                                  f"{d.loc[ok, 'F_over_Eg_Bcf'].iloc[0]:,.0f} Bcf")
                weak = d["F_over_Eg_Bcf"].notna() & ~ok
                if weak.any():
                    fig.add_trace(go.Scatter(x=d.loc[weak, "Gp"], y=d.loc[weak, "F_over_Eg_Bcf"],
                                             mode="markers", name="too near the reference",
                                             marker=dict(color="#c2c7cc", size=9,
                                                         symbol="circle-open")))
                st.plotly_chart(theme_axes(fig, "cumulative gas (MMscf)", "F / Eg (Bcf)",
                                           height=360), width="stretch")
                st.caption("**Havlena–Odeh.** F = G·Eg + We. Flat means We ≈ 0 and the level "
                           "*is* G. Rising means influx. This is the discriminator.")

        # Apparent G, survey by survey. Same physics as the panel above, none of
        # the PVT algebra: a closed tank returns one number six times.
        d = r.points
        ok = d["apparent_G_usable"]
        if ok.sum() >= 2:
            show = d.loc[ok, ["P", "Gp", "depleted", "apparent_G_Bcf"]].copy()
            show.columns = [f"SBHP ({U.pressure})", "Cum (MMscf)", "Depleted", "Apparent G (Bcf)"]
            show[f"SBHP ({U.pressure})"] = U.pressure_from_psia(show[f"SBHP ({U.pressure})"])
            show["Depleted"] = (100 * show["Depleted"]).map("{:.1f}%".format)
            st.markdown("**Apparent G, survey by survey** — "
                        "`G = Gp / (1 − (p/z)/(pi/zi))`")
            st.dataframe(show.style.format({f"SBHP ({U.pressure})": "{:,.6g}",
                                            "Cum (MMscf)": "{:,.1f}",
                                            "Apparent G (Bcf)": "{:,.1f}"}),
                         hide_index=True, width="stretch")
            spread = float(d.loc[ok, "apparent_G_Bcf"].iloc[-1]
                           / d.loc[ok, "apparent_G_Bcf"].iloc[0])
            if spread > 1.10:
                st.warning(
                    f"**Apparent G climbs {spread:.2f}× across the surveys**, from "
                    f"{r.g_bound:,.1f} to {d.loc[ok, 'apparent_G_Bcf'].iloc[-1]:,.1f} Bcf. A "
                    "closed tank returns the same number every time; support holds p/z up, "
                    "which inflates every estimate and inflates the later ones more. So the "
                    f"**smallest** value is the tightest bound: **G ≤ {r.g_bound:,.1f} Bcf**, "
                    "and the straight-line intercept is the least reliable reading of the set "
                    "because it is dominated by the latest, most inflated points.")
            elif spread < 0.91:
                st.warning(
                    f"**Apparent G falls {spread:.2f}× across the surveys.** Influx only "
                    "accumulates, so it cannot produce a falling trend. Suspect the reference "
                    "pressure, the datum correction, or production allocated to this well.")
            else:
                st.success(
                    f"**Apparent G is level ({spread:.2f}× across the surveys)** at about "
                    f"{d.loc[ok, 'apparent_G_Bcf'].median():,.1f} Bcf. That is what a closed "
                    "tank looks like, and it is independent confirmation of the p/z intercept.")

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

            # The hand-off must not cross fluids: a gas volume dropped into an oil
            # in-place slot is formatted in bbl and reads as a plausible number.
            mismatch = (mb_fluid == "Gas") != IS_GAS
            if mismatch:
                st.caption(f"⚠️ The well is set to **{fluid.lower()}** in the sidebar but "
                           f"this balance is **{mb_fluid.lower()}**, so the answer cannot "
                           "be sent across — it would land in the wrong units. Change the "
                           "fluid in the sidebar if this is a gas well.")
            gcol = st.columns(2)
            if gcol[0].button(f"Use the Fetkovich G ({f['G_Bcf']:,.0f} Bcf) as this well's "
                              "GIIP", key="send_fetk", disabled=mismatch):
                st.session_state["ip_from_mb"] = f["G_Bcf"] * 1e6      # Bcf -> Mcf
                st.session_state["ip_from_mb_src"] = (
                    f"Fetkovich aquifer fit · G {f['G_Bcf']:,.0f} Bcf · rms {f['rms_pct']:.1f}%")
                st.rerun()
            if gcol[1].button(f"Use the p/z intercept ({r.ogip_pz / 1000:,.0f} Bcf) instead",
                              key="send_pz",
                              disabled=mismatch or not r.volumetric,
                              help="Disabled: the sidebar fluid is oil." if mismatch
                              else (None if r.volumetric else
                                    "Disabled: F/Eg is rising, so the p/z intercept is an "
                                    "artefact rather than a volume.")):
                st.session_state["ip_from_mb"] = r.ogip_pz * 1000.0    # MMscf -> Mcf
                st.session_state["ip_from_mb_src"] = (
                    f"p/z intercept · {r.ogip_pz / 1000:,.0f} Bcf · R² {r.r2:.4f}")
                st.rerun()
            st.caption("Either number becomes the in-place volume the forecast is checked "
                       "against. It is only a **well** volume if these pressures and this "
                       "cumulative came from this well.")

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
