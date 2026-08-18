import datetime as dt
import math
from zoneinfo import ZoneInfo
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, text

# ============================================================
# DESIGN TOKENS
# ============================================================

# Deep navy base. Every chrome colour is drawn from this blue ramp so that
# green and red are never used for anything except stat-vs-baseline scales.
PAGE_BG = "#060A12"
SURFACE_1 = "#0E1626"
SURFACE_2 = "#172236"
SURFACE_3 = "#1F2C44"
BORDER = "#1F2B40"

TEXT_1 = "#E8ECF4"
TEXT_2 = "#93A0B8"
TEXT_3 = "#5F6E88"

# Team identity: brand blue for home, dark gold for away, neutral grey for
# the draw. Deliberately desaturated so the stat scales stay the loudest
# thing on the page.
HOME = "#4C8DFF"
AWAY = "#C9A227"
DRAW = "#3D434D"

# For "against" series, which represent an opponent in general rather than
# the away side of this fixture.
NEUTRAL = "#7A8598"

# RGB forms, used when tinting cells toward a side rather than good/bad.
HOME_RGB = np.array([76, 141, 255])
AWAY_RGB = np.array([201, 162, 39])

# Reserved exclusively for stat scales (goals, shots on target, ATT/DEF,
# SOD differential, model edge). Nothing structural uses these.
GOOD = np.array([16, 185, 129])
BAD = np.array([239, 68, 68])
# String forms for CSS. Do not build these with f"rgb{tuple(GOOD)}" — on
# numpy 2.x that renders as "rgb(np.int64(16), ...)", which browsers discard.
GOOD_HEX = "#10B981"
BAD_HEX = "#EF4444"
CELL_BASE = np.array([23, 34, 54])  # matches SURFACE_2 so neutral cells recede

TZ = ZoneInfo("Europe/London")
CACHE_TTL = 900

st.set_page_config(page_title="Underlying Stats", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown(
    f"""
    <style>
    header[data-testid="stHeader"] {{ display: none; }}
    html, body, .stApp {{ background-color: {PAGE_BG}; color: {TEXT_1}; }}
    .block-container {{
        padding: 1.1rem 1.1rem 3rem 1.1rem !important;
        max-width: 1400px;
    }}
    h1, h2, h3, h4 {{ color: {TEXT_1} !important; font-weight: 500 !important; }}

    .stTextInput input,
    .stSelectbox div[data-baseweb="select"] > div {{
        background-color: {SURFACE_2} !important;
        color: {TEXT_1} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 8px !important;
    }}
    div[role="listbox"] {{
        background-color: {SURFACE_2} !important;
        border: 1px solid {BORDER} !important;
    }}
    div.stButton > button {{
        width: 100%; border-radius: 8px;
        background-color: {SURFACE_2}; color: {TEXT_1};
        border: 1px solid {BORDER};
    }}
    div.stButton > button:hover {{
        background-color: {SURFACE_3};
        border-color: {HOME}; color: {HOME};
    }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 2px; border-bottom: 1px solid {BORDER}; }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent; color: {TEXT_3};
        padding: 8px 14px; font-size: 14px;
    }}
    .stTabs [aria-selected="true"] {{ color: {TEXT_1} !important; }}

    .stat-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    .stat-table thead th {{
        color: {TEXT_3}; font-weight: 400; font-size: 12px;
        padding: 7px 8px; text-align: center;
        border-bottom: 1px solid {BORDER}; white-space: nowrap;
    }}
    .stat-table tbody td {{
        padding: 6px 8px; text-align: center; border: 2px solid {PAGE_BG};
        border-radius: 3px;
    }}
    .stat-table tbody td:first-child,
    .stat-table thead th:first-child {{ text-align: left; color: {TEXT_1}; }}
    .table-wrap {{ overflow-x: auto; }}

    .league-table td.stick0, .league-table th.stick0 {{
        position: sticky; left: 0; z-index: 2;
        background-color: {PAGE_BG};
    }}
    .league-table td.stick1, .league-table th.stick1 {{
        position: sticky; left: 42px; z-index: 2;
        background-color: {PAGE_BG};
        box-shadow: 1px 0 0 {BORDER};
    }}
    .league-table th.stick0, .league-table th.stick1 {{ z-index: 3; }}
    .league-table tbody tr:hover td {{ filter: brightness(1.15); }}

    @media (max-width: 640px) {{
        .block-container {{ padding: 0.6rem !important; }}
        .stat-table {{ font-size: 12px; }}
        .stat-table tbody td, .stat-table thead th {{ padding: 5px 6px; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

_PLOT_LAYOUT = dict(
    paper_bgcolor=SURFACE_1,
    plot_bgcolor=SURFACE_1,
    font=dict(color=TEXT_1, size=13),
    margin=dict(l=6, r=6, t=10, b=6),
    showlegend=False,
    # No chart here benefits from zoom or pan, and on a phone the drag
    # handlers intercept scrolling. dragmode=False plus fixedrange on both
    # axes disables pinch-zoom, drag-pan and box-select while leaving hover
    # tooltips working — staticPlot would remove those too.
    dragmode=False,
    hoverlabel=dict(bgcolor=SURFACE_2, bordercolor=BORDER,
                    font=dict(color=TEXT_1, size=12)),
)
_PLOT_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
    "scrollZoom": False,
    "doubleClick": False,
    "showAxisDragHandles": False,
    "showAxisRangeEntryBoxes": False,
}


# ============================================================
# CONNECTION
# ============================================================

def get_required_secret(name: str) -> str:
    value = str(st.secrets.get(name, "")).strip()
    if not value:
        st.error(f"Missing Streamlit secret: {name}")
        st.stop()
    return value


@st.cache_resource
def get_engine():
    host = get_required_secret("SUPABASE_HOST")
    port = get_required_secret("SUPABASE_PORT")
    db = get_required_secret("SUPABASE_DB")
    user = get_required_secret("SUPABASE_USER")
    password = get_required_secret("SUPABASE_PASS")

    return create_engine(
        f"postgresql+psycopg2://{quote_plus(user)}:{quote_plus(password)}"
        f"@{host}:{port}/{db}?sslmode=require",
        pool_pre_ping=True,
        pool_recycle=300,
        # The engine is cached once per container and shared by every visitor.
        # Queries are cached for 15 minutes, so most page views touch no
        # connection at all; this headroom covers simultaneous cold loads.
        pool_size=3,
        max_overflow=4,
        # Fail fast rather than leaving a visitor on a spinner if the pool
        # is saturated.
        pool_timeout=10,
        connect_args={"connect_timeout": 10},
        future=True,
    )


ENGINE = get_engine()


def read_sql_df(sql: str, params: dict | None = None,
                critical: bool = False) -> pd.DataFrame:
    """
    Non-critical queries degrade to an empty frame rather than halting the app.
    The original called st.stop() on any failure, so one transient error
    blanked the entire page.
    """
    try:
        with ENGINE.connect() as conn:
            return pd.read_sql(text(sql), conn, params=params or {})
    except Exception as exc:
        if critical:
            st.error(
                "Couldn't load match data right now. This is usually "
                "temporary — try refreshing in a moment."
            )
            st.caption(f"Details: {type(exc).__name__}")
            st.stop()
        st.warning("Some data couldn't be loaded for this match.")
        return pd.DataFrame()


def read_sql_one(sql: str, params: dict | None = None,
                 critical: bool = False) -> dict | None:
    df = read_sql_df(sql, params=params, critical=critical)
    return None if df.empty else df.iloc[0].to_dict()


# ============================================================
# DATA
# ============================================================

@st.cache_data(ttl=CACHE_TTL)
def load_fixtures() -> pd.DataFrame:
    df = read_sql_df(
        """
        SELECT eventid, hometeam, awayteam, league, date, kickoff,
               home AS homeodds, draw AS drawodds, away AS awayodds
        FROM fixtures
        WHERE home IS NOT NULL AND away IS NOT NULL AND home > 0 AND away > 0
        ORDER BY date DESC, league, hometeam
        """,
        critical=True,
    )
    if df.empty:
        return df

    for c in ["homeodds", "drawodds", "awayodds"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[(df["homeodds"] > 0) & (df["awayodds"] > 0)].copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["datestr"] = df["date"].dt.strftime("%Y-%m-%d")
    df["fixturename"] = (df["hometeam"].fillna("N/A") + " vs "
                         + df["awayteam"].fillna("N/A"))
    return df


@st.cache_data(ttl=CACHE_TTL)
def load_teams() -> pd.DataFrame:
    """
    The whole team table, loaded once. Replaces get_team_stats,
    get_team_att_def and get_opponent_att_def, which previously ran one
    query per team per lookup. Also carries the standings columns, so the
    league table tab needs no query of its own.
    """
    df = read_sql_df(
        """
        SELECT "TeamName", "League", "Games", "AGF", "AGA",
               "ASOF", "ASOA", "ATT", "DEF", "Form",
               "StandingPosition", "StandingGames", "StandingPPG",
               "StandingPoints", "StandingWins", "StandingDraws",
               "StandingLosses"
        FROM list_of_teams
        """
    )
    if df.empty:
        return df
    for c in ["Games", "AGF", "AGA", "ASOF", "ASOA", "ATT", "DEF", "Form",
              "StandingPosition", "StandingGames", "StandingPPG",
              "StandingPoints", "StandingWins", "StandingDraws",
              "StandingLosses"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.set_index("TeamName")


def team_row(teams: pd.DataFrame, name: str) -> dict:
    empty = {"Games": 0, "AGF": 0, "AGA": 0, "ASOF": 0, "ASOA": 0,
             "ATT": 0, "DEF": 0, "Form": None, "League": None}
    if teams.empty or name not in teams.index:
        return empty
    return teams.loc[name].to_dict()


@st.cache_data(ttl=CACHE_TTL)
def load_matches(teams: tuple[str, ...], days: int) -> pd.DataFrame:
    """One query covering both teams, replacing the per-opponent N+1 loop."""
    since = (dt.datetime.now(TZ) - dt.timedelta(days=days)).date()
    df = read_sql_df(
        """
        SELECT "Date" AS date, "League" AS league,
               "HomeTeam" AS hometeam, "AwayTeam" AS awayteam,
               "HomeGoals" AS homegoals, "AwayGoals" AS awaygoals,
               "HomeShots" AS homeshots, "AwayShots" AS awayshots,
               "HomeShotsOn" AS homeshotson, "AwayShotsOn" AS awayshotson,
               "HomeAttacks" AS homeattacks, "AwayAttacks" AS awayattacks,
               "HomeDangerousAttacks" AS homedangerousattacks,
               "AwayDangerousAttacks" AS awaydangerousattacks
        FROM matchstats
        WHERE "Date" >= :since
          AND ("HomeTeam" = ANY(:teams) OR "AwayTeam" = ANY(:teams))
        ORDER BY "Date" DESC
        """,
        params={"since": since, "teams": list(teams)},
    )
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@st.cache_data(ttl=CACHE_TTL)
def league_baselines(league_name: str) -> dict:
    """
    Single baseline function. The original had get_league_wide_baselines and
    get_league_recent_baselines returning overlapping values from the same
    270-day window — this replaces both.
    """
    since = (dt.datetime.now(TZ) - dt.timedelta(days=270)).date()
    df = read_sql_df(
        """
        SELECT "HomeGoals" AS hg, "AwayGoals" AS ag,
               "HomeShotsOn" AS hso, "AwayShotsOn" AS aso,
               "HomeShots" AS hs, "AwayShots" AS as_
        FROM matchstats
        WHERE "League" = :league AND "Date" >= :since
        """,
        params={"league": league_name, "since": since},
    )

    def pooled(a, b):
        if df.empty:
            return 0.0
        s = pd.to_numeric(pd.concat([df[a], df[b]], ignore_index=True),
                          errors="coerce").dropna()
        return float(s.mean()) if not s.empty else 0.0

    teams = load_teams()
    if not teams.empty and league_name:
        sub = teams[teams["League"] == league_name]
        att = sub["ATT"].replace(0, np.nan).dropna()
        deff = sub["DEF"].replace(0, np.nan).dropna()
    else:
        att = deff = pd.Series(dtype=float)

    return {
        "GF": pooled("hg", "ag"), "GA": pooled("ag", "hg"),
        "SOF": pooled("hso", "aso"), "SOA": pooled("aso", "hso"),
        "SF": pooled("hs", "as_"), "SA": pooled("as_", "hs"),
        "Opp ATT": float(att.mean()) if not att.empty else 0.0,
        "Opp DEF": float(deff.mean()) if not deff.empty else 0.0,
    }


# ============================================================
# DERIVED VIEWS
# ============================================================

def perspective(df: pd.DataFrame, team: str) -> pd.DataFrame:
    """Flip each match into the given team's point of view."""
    if df.empty:
        return df
    m = df[(df["hometeam"] == team) | (df["awayteam"] == team)].copy()
    if m.empty:
        return m
    is_home = m["hometeam"] == team

    out = pd.DataFrame(index=m.index)
    out["date"] = m["date"]
    out["Opponent"] = np.where(is_home, m["awayteam"], m["hometeam"])
    out["GF"] = np.where(is_home, m["homegoals"], m["awaygoals"])
    out["GA"] = np.where(is_home, m["awaygoals"], m["homegoals"])
    out["SOF"] = np.where(is_home, m["homeshotson"], m["awayshotson"])
    out["SOA"] = np.where(is_home, m["awayshotson"], m["homeshotson"])
    out["SF"] = np.where(is_home, m["homeshots"], m["awayshots"])
    out["SA"] = np.where(is_home, m["awayshots"], m["homeshots"])
    out["Venue"] = np.where(is_home, "H", "A")
    return out.sort_values("date", ascending=False)


def recent_form(matches: pd.DataFrame, team: str,
                teams: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    df = perspective(matches, team).head(limit).copy()
    if df.empty:
        return df
    df["Opp ATT"] = df["Opponent"].map(
        teams["ATT"] if not teams.empty else {})
    df["Opp DEF"] = df["Opponent"].map(
        teams["DEF"] if not teams.empty else {})
    df["When"] = df["date"].map(relative_day)
    return df[["Opponent", "Venue", "Opp ATT", "Opp DEF", "When",
               "GF", "SOF", "SF", "GA", "SOA", "SA"]]


MUTUAL_WINDOWS = {
    "Last 2 weeks": 14,
    "Last month": 30,
    "Last 2 months": 60,
    "Last 3 months": 90,
}


def mutual_opponents(matches: pd.DataFrame, home: str, away: str,
                     teams: pd.DataFrame, days: int = 90) -> pd.DataFrame:
    """
    Shared opponents within the last `days`. The caller already holds a
    270-day window, so narrowing happens in memory — changing the timeframe
    costs no extra query.
    """
    if matches.empty:
        return pd.DataFrame()

    if days:
        cutoff = pd.Timestamp(dt.datetime.now(TZ).date()
                              - dt.timedelta(days=days))
        matches = matches[matches["date"] >= cutoff]
        if matches.empty:
            return pd.DataFrame()

    h = perspective(matches, home)
    a = perspective(matches, away)
    if h.empty or a.empty:
        return pd.DataFrame()

    common = set(h["Opponent"]) & set(a["Opponent"])
    common.discard(home)
    common.discard(away)
    if not common:
        return pd.DataFrame()

    h_latest = h[h["Opponent"].isin(common)].drop_duplicates("Opponent")
    a_latest = a[a["Opponent"].isin(common)].drop_duplicates("Opponent")

    merged = h_latest.merge(a_latest, on="Opponent", suffixes=("_h", "_a"))
    if merged.empty:
        return merged

    out = pd.DataFrame({
        "Opponent": merged["Opponent"],
        "ATT": merged["Opponent"].map(teams["ATT"] if not teams.empty else {}),
        "DEF": merged["Opponent"].map(teams["DEF"] if not teams.empty else {}),
        "H when": merged["date_h"].map(relative_day),
        "H SOF": merged["SOF_h"], "H SOA": merged["SOA_h"],
        "A when": merged["date_a"].map(relative_day),
        "A SOF": merged["SOF_a"], "A SOA": merged["SOA_a"],
        "_sort": merged["date_h"],
    })

    # Each side's own shots-on-target differential against this shared
    # opponent, then the difference between those two differentials:
    #   (A SOFor - A SOAgainst) - (B SOFor - B SOAgainst)
    # A comparison of raw shots-on-target for would ignore what each side
    # conceded to the same opponent, which is half the signal.
    h_sof = pd.to_numeric(out["H SOF"], errors="coerce")
    h_soa = pd.to_numeric(out["H SOA"], errors="coerce")
    a_sof = pd.to_numeric(out["A SOF"], errors="coerce")
    a_soa = pd.to_numeric(out["A SOA"], errors="coerce")

    out["H Diff"] = h_sof - h_soa
    out["A Diff"] = a_sof - a_soa
    out["Diff"] = out["H Diff"] - out["A Diff"]

    out = out[["Opponent", "ATT", "DEF",
               "H when", "H SOF", "H SOA", "H Diff",
               "A when", "A SOF", "A SOA", "A Diff",
               "Diff", "_sort"]]

    return (out.sort_values("_sort", ascending=False)
               .drop(columns="_sort")
               .reset_index(drop=True))


def head_to_head(matches_all: pd.DataFrame, home: str, away: str,
                 limit: int = 5) -> pd.DataFrame:
    if matches_all.empty:
        return matches_all
    mask = (((matches_all["hometeam"] == home) & (matches_all["awayteam"] == away))
            | ((matches_all["hometeam"] == away) & (matches_all["awayteam"] == home)))
    return matches_all[mask].sort_values("date", ascending=False).head(limit)


def relative_day(value) -> str:
    d = pd.to_datetime(value, errors="coerce")
    if pd.isna(d):
        return "—"
    days = (dt.date.today() - d.date()).days
    if days <= 0:
        return "today"
    if days < 30:
        return f"{days}d"
    return d.strftime("%d %b")


# ============================================================
# STYLING
# ============================================================

def cell_style(value, avg, higher_is_better=True) -> str:
    """
    Interpolates from the table surface rather than white, so mid-range values
    recede into the page instead of glaring. tanh keeps extremes separable
    where the original clipped everything above +100% to one colour.
    """
    try:
        value = float(value)
        if avg in (None, 0) or pd.isna(value) or pd.isna(avg):
            return f"background-color:{SURFACE_2};color:{TEXT_1};"
        return _tint((value - avg) / abs(avg), higher_is_better)
    except Exception:
        return f"background-color:{SURFACE_2};color:{TEXT_1};"


def strength_style(value, scale) -> str:
    """
    For directional differentials like SODD and the mutual-opponent Diff,
    where sign indicates which side is favoured rather than good or bad.
    Green/red would misread these: a large negative is a strong away signal,
    not a bad one. Intensity therefore tracks magnitude, and hue tracks the
    side — home blue or away gold — so weak values sit near neutral and
    strong ones in either direction read as strong.
    """
    try:
        value = float(value)
        if pd.isna(value) or not scale:
            return f"background-color:{SURFACE_2};color:{TEXT_1};"

        intensity = float(np.tanh(abs(value) / scale * 1.2))
        target = HOME_RGB if value > 0 else AWAY_RGB
        rgb = np.clip(CELL_BASE + (target - CELL_BASE) * intensity,
                      0, 255).astype(int)
        lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        fg = PAGE_BG if lum > 140 else TEXT_1
        return f"background-color:rgb({rgb[0]},{rgb[1]},{rgb[2]});color:{fg};"
    except (TypeError, ValueError):
        return f"background-color:{SURFACE_2};color:{TEXT_1};"


def signed_style(value, scale) -> str:
    """
    For differentials, where zero is neutral rather than a baseline ratio.
    Large positive shades green, large negative red. Measuring these against
    a baseline would paint a neutral zero a strong colour.
    """
    try:
        value = float(value)
        if pd.isna(value) or not scale:
            return f"background-color:{SURFACE_2};color:{TEXT_1};"
        return _tint(value / scale, True)
    except (TypeError, ValueError):
        return f"background-color:{SURFACE_2};color:{TEXT_1};"


def _tint(norm: float, higher_is_better: bool) -> str:
    intensity = float(np.tanh(abs(norm) * 1.4))
    good = norm > 0 if higher_is_better else norm < 0
    rgb = np.clip(CELL_BASE + ((GOOD if good else BAD) - CELL_BASE)
                  * intensity, 0, 255).astype(int)
    lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    fg = PAGE_BG if lum > 140 else TEXT_1
    return f"background-color:rgb({rgb[0]},{rgb[1]},{rgb[2]});color:{fg};"


def signed_scale(series: pd.Series, floor: float = 4.0) -> float:
    """
    Intensity reference for a differential column: the 90th percentile of its
    absolute values, with a floor so a quiet fixture doesn't make a one-shot
    difference look dramatic.
    """
    values = pd.to_numeric(series, errors="coerce").abs().dropna()
    values = values[values > 0]
    if len(values) < 3:
        return floor
    return max(floor, float(values.quantile(0.90)))


def deviation_marker(value, avg, higher_is_better=True) -> str:
    """
    Redundant encoding for the green/red scale. Roughly 8% of men have a
    red-green colour deficiency, so a public app cannot rely on hue alone.
    Only marks meaningful deviations, keeping ordinary rows uncluttered.
    """
    try:
        value, avg = float(value), float(avg)
        if avg == 0 or pd.isna(value) or pd.isna(avg):
            return ""
        norm = (value - avg) / avg
        if abs(norm) < 0.25:
            return ""
        good = norm > 0 if higher_is_better else norm < 0
        return " ▲" if good else " ▼"
    except (TypeError, ValueError):
        return ""


def render_table(df: pd.DataFrame, baselines: dict,
                 positive: list[str], negative: list[str],
                 formats: dict | None = None,
                 signed: dict | None = None,
                 strength: dict | None = None):
    """
    Hand-rendered HTML so the gradient survives, wrapped for horizontal scroll
    on narrow screens rather than crushing columns.

    `signed`   maps column -> scale for zero-neutral good/bad columns.
    `strength` maps column -> scale for directional columns, tinted toward
               the side they favour rather than green/red.
    """
    if df.empty:
        st.caption("No data available.")
        return

    formats = formats or {}
    signed = signed or {}
    strength = strength or {}
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    body = []

    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            val = row[col]
            fmt = formats.get(col)
            if pd.isna(val):
                text_val = "—"
            elif fmt:
                try:
                    text_val = fmt.format(float(val))
                except (TypeError, ValueError):
                    text_val = str(val)
            else:
                text_val = str(val)

            if col in strength:
                style = strength_style(val, strength[col])
                cells.append(
                    f'<td style="{style}" title="{col}: {text_val}">'
                    f"{text_val}</td>")
            elif col in signed:
                style = signed_style(val, signed[col])
                mark = ""
                try:
                    if not pd.isna(val) and abs(float(val)) >= signed[col] / 2:
                        mark = " ▲" if float(val) > 0 else " ▼"
                except (TypeError, ValueError):
                    pass
                cells.append(
                    f'<td style="{style}" title="{col}: {text_val}">{text_val}'
                    f'<span style="font-size:10px;opacity:.75;">{mark}</span></td>'
                )
            elif col in positive or col in negative:
                avg = baselines.get(col)
                higher_better = col in positive
                style = cell_style(val, avg, higher_is_better=higher_better)
                mark = deviation_marker(val, avg, higher_is_better=higher_better)
                if avg:
                    title = (f"{col}: {text_val} · league average {avg:.2f}")
                else:
                    title = col
                cells.append(
                    f'<td style="{style}" title="{title}">{text_val}'
                    f'<span style="font-size:10px;opacity:.75;">{mark}</span></td>'
                )
            else:
                cells.append(f'<td style="color:{TEXT_2}">{text_val}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")

    st.markdown(
        f'<div class="table-wrap"><table class="stat-table">'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody>"
        f"</table></div>",
        unsafe_allow_html=True,
    )


# ============================================================
# CHARTS
# ============================================================

def comparison_chart(home_stats, away_stats, home_team, away_team, base):
    """
    Each stat scaled to its own range with a league baseline marker. The
    original used a single hardcoded ±15 axis, so ATT and DEF (~1.0) rendered
    as slivers next to ASOF (~5.0).
    """
    stats = [("AGF", "GF", True), ("AGA", "GA", False),
             ("ASOF", "SOF", True), ("ASOA", "SOA", False),
             ("ATT", "Opp ATT", True), ("DEF", "Opp DEF", False)]

    rows = []
    for label, base_key, _ in stats:
        h = float(home_stats.get(label) or 0)
        a = float(away_stats.get(label) or 0)
        b = base.get(base_key) or 0
        ref = max(h, a, b) * 1.3 or 1.0
        rows.append((label, h, a, b, ref))

    labels = [r[0] for r in rows]
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=labels, x=[-(r[1] / r[4]) for r in rows], orientation="h",
        marker_color=HOME, width=0.5,
        customdata=[[r[1], r[3]] for r in rows],
        hovertemplate=(f"<b>{home_team}</b><br>%{{y}}: %{{customdata[0]:.2f}}"
                       "<br>league %{customdata[1]:.2f}<extra></extra>"),
    ))
    fig.add_trace(go.Bar(
        y=labels, x=[r[2] / r[4] for r in rows], orientation="h",
        marker_color=AWAY, width=0.5,
        customdata=[[r[2], r[3]] for r in rows],
        hovertemplate=(f"<b>{away_team}</b><br>%{{y}}: %{{customdata[0]:.2f}}"
                       "<br>league %{customdata[1]:.2f}<extra></extra>"),
    ))

    for i, (label, h, a, b, ref) in enumerate(rows):
        fig.add_annotation(x=-(h / ref) - 0.05, y=i, text=f"{h:.2f}",
                           showarrow=False, xanchor="right",
                           font=dict(color=TEXT_1, size=12))
        fig.add_annotation(x=(a / ref) + 0.05, y=i, text=f"{a:.2f}",
                           showarrow=False, xanchor="left",
                           font=dict(color=TEXT_1, size=12))
        if b:
            for sign in (-1, 1):
                fig.add_shape(type="line",
                              x0=sign * b / ref, x1=sign * b / ref,
                              y0=i - 0.32, y1=i + 0.32,
                              line=dict(color=TEXT_3, width=1, dash="dot"))

    fig.add_vline(x=0, line_color=BORDER, line_width=1)
    fig.update_layout(**_PLOT_LAYOUT, barmode="overlay",
                      height=44 * len(rows) + 30)
    fig.update_xaxes(range=[-1.4, 1.4], showgrid=False, zeroline=False,
                     showticklabels=False, fixedrange=True)
    fig.update_yaxes(showgrid=False, autorange="reversed", fixedrange=True,
                     tickfont=dict(color=TEXT_2, size=12))
    return fig


def h2h_chart(row: dict, fixture_home: str, fixture_away: str):
    """
    Per-stat scaling, with two independent encodings:

        position — left is whoever was at home in that past meeting,
                   right is whoever was away
        colour   — the club's identity colour in the upcoming fixture, blue
                   for its home side and gold for its away side

    Keeping those separate matters when the last meeting was the reverse
    fixture. Colouring by the past match's venue would have shown a club blue
    here and gold everywhere else on the dashboard.
    """
    stats = [("Goals", "homegoals", "awaygoals"),
             ("Shots", "homeshots", "awayshots"),
             ("On target", "homeshotson", "awayshotson"),
             ("Attacks", "homeattacks", "awayattacks"),
             ("Dangerous", "homedangerousattacks", "awaydangerousattacks")]

    past_home = row.get("hometeam") or fixture_home
    past_away = row.get("awayteam") or fixture_away

    # Was the upcoming fixture's home side also at home last time?
    same_orientation = (past_home == fixture_home)
    left_colour = HOME if same_orientation else AWAY
    right_colour = AWAY if same_orientation else HOME

    rows = []
    for label, hk, ak in stats:
        h = float(row.get(hk) or 0)
        a = float(row.get(ak) or 0)
        rows.append((label, h, a, max(h, a, 1) * 1.3))

    labels = [r[0] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=[-(r[1] / r[3]) for r in rows], orientation="h",
        marker_color=left_colour, width=0.45,
        customdata=[[r[1]] for r in rows],
        hovertemplate=f"<b>{past_home}</b> (home)<br>%{{y}}: "
                      "%{customdata[0]:.0f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=labels, x=[r[2] / r[3] for r in rows], orientation="h",
        marker_color=right_colour, width=0.45,
        customdata=[[r[2]] for r in rows],
        hovertemplate=f"<b>{past_away}</b> (away)<br>%{{y}}: "
                      "%{customdata[0]:.0f}<extra></extra>",
    ))

    # Name each side above its own half, in its own colour. Positioned in
    # paper coordinates rather than data coordinates so the y-axis is left
    # exactly as the other charts have it — an explicit numeric range on a
    # categorical axis can drop bars entirely.
    fig.add_annotation(x=0, y=1.0, xref="paper", yref="paper",
                       text=f"{past_home} (H)", showarrow=False,
                       xanchor="left", yanchor="bottom",
                       font=dict(color=left_colour, size=12))
    fig.add_annotation(x=1, y=1.0, xref="paper", yref="paper",
                       text=f"{past_away} (A)", showarrow=False,
                       xanchor="right", yanchor="bottom",
                       font=dict(color=right_colour, size=12))

    for i, (label, h, a, ref) in enumerate(rows):
        fig.add_annotation(x=-(h / ref) - 0.05, y=i, text=f"{h:.0f}",
                           showarrow=False, xanchor="right",
                           font=dict(color=TEXT_1, size=12))
        fig.add_annotation(x=(a / ref) + 0.05, y=i, text=f"{a:.0f}",
                           showarrow=False, xanchor="left",
                           font=dict(color=TEXT_1, size=12))

    fig.add_vline(x=0, line_color=BORDER, line_width=1)
    # Top margin widened for the two labels; every other layout value and the
    # axes themselves are identical to the other charts.
    fig.update_layout(**{**_PLOT_LAYOUT, "margin": dict(l=6, r=6, t=28, b=6)},
                      barmode="overlay", height=44 * len(rows) + 48)
    fig.update_xaxes(range=[-1.4, 1.4], showgrid=False, zeroline=False,
                     showticklabels=False, fixedrange=True)
    fig.update_yaxes(showgrid=False, autorange="reversed", fixedrange=True,
                     tickfont=dict(color=TEXT_2, size=12))
    return fig


def form_trend(df: pd.DataFrame, team: str, base: dict,
               colour: str = HOME):
    """
    Shots on target for and against across recent matches. `colour` is the
    team's own identity colour, so the home chart reads blue and the away
    chart gold.
    """
    if df.empty:
        return None
    d = df.iloc[::-1]
    x = list(range(len(d)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=pd.to_numeric(d["SOF"], errors="coerce"),
        mode="lines+markers", line=dict(color=colour, width=2),
        marker=dict(size=6), name="SOF",
        customdata=d["Opponent"],
        hovertemplate="vs %{customdata}<br>SOF %{y:.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=x, y=pd.to_numeric(d["SOA"], errors="coerce"),
        # Neutral, not AWAY — inside a single team's trend this is shots
        # conceded, nothing to do with the away side of the fixture.
        mode="lines+markers", line=dict(color=NEUTRAL, width=2, dash="dot"),
        marker=dict(size=6), name="SOA",
        customdata=d["Opponent"],
        hovertemplate="vs %{customdata}<br>SOA %{y:.0f}<extra></extra>"))

    if base.get("SOF"):
        fig.add_hline(y=base["SOF"], line=dict(color=TEXT_3, width=1,
                                               dash="dash"))

    fig.update_layout(**_PLOT_LAYOUT, height=180)
    fig.update_xaxes(showgrid=False, showticklabels=False, fixedrange=True)
    fig.update_yaxes(showgrid=True, gridcolor=BORDER, zeroline=False,
                     fixedrange=True, tickfont=dict(color=TEXT_3, size=11))
    return fig


# ============================================================
# HEADER COMPONENTS
# ============================================================

def metric_tile(label: str, value: str, color: str = TEXT_1) -> str:
    return (
        f'<div style="background:{SURFACE_1};border-radius:8px;padding:9px 11px;">'
        f'<div style="font-size:11px;color:{TEXT_3};letter-spacing:.04em;">{label}</div>'
        f'<div style="font-size:19px;font-weight:500;color:{color};">{value}</div>'
        f"</div>"
    )


def metric_pair(label: str, home_value: str, away_value: str) -> str:
    """
    Paired stat with each side in its own colour, so the reader never has to
    work out which of the two numbers belongs to which team.
    """
    return (
        f'<div style="background:{SURFACE_1};border-radius:8px;padding:9px 11px;">'
        f'<div style="font-size:11px;color:{TEXT_3};letter-spacing:.04em;">{label}</div>'
        f'<div style="font-size:19px;font-weight:500;">'
        f'<span style="color:{HOME};">{home_value}</span>'
        f'<span style="color:{TEXT_3};font-size:13px;"> · </span>'
        f'<span style="color:{AWAY};">{away_value}</span>'
        f"</div></div>"
    )


GLOSSARY = [
    ("GF / GA", "Goals for / against."),
    ("SOF / SOA", "Shots on target for / against."),
    ("SF / SA", "Total shots for / against."),
    ("AGF / AGA", "Average goals for / against, per game this season."),
    ("ASOF / ASOA", "Average shots on target for / against, per game."),
    ("ATT", "Attack rating. Above 1.00 means the team creates more than the "
            "league average."),
    ("DEF", "Defence rating. Below 1.00 means the team concedes less than the "
            "league average."),
    ("xG", "Expected goals — the goals a team would typically score from the "
           "chances the model gives them."),
    ("Expected SoT", "Expected shots on target, from the same model."),
    ("H Diff / A Diff", "Each side's shots on target for minus against, "
                        "against that same shared opponent."),
    ("Diff", "H Diff minus A Diff — how much better the home side performed "
             "against a common opponent than the away side did. Positive "
             "favours the home team."),
    ("SOD diff", "Shots-on-target differential against shared opponents. "
                 "Positive favours the home side."),
    ("Form", "Recent performance score. Higher is better."),
    ("Edge", "Model probability minus the probability implied by the odds, in "
             "percentage points. Positive means the model rates the outcome "
             "more likely than the market does."),
    ("Venue", "H = played at home, A = played away."),
]


def render_help():
    with st.expander("What do these abbreviations mean?"):
        st.markdown(
            f'<div style="font-size:13px;color:{TEXT_2};line-height:1.7;">'
            + "".join(
                f'<div><span style="color:{TEXT_1};">{term}</span> — {desc}</div>'
                for term, desc in GLOSSARY
            )
            + f'<div style="margin-top:10px;color:{TEXT_3};">'
              f'<span style="color:{GOOD_HEX};">Green ▲</span> is above the '
              f'league average, <span style="color:{BAD_HEX};">red ▼</span> '
              f"below. Directional columns like Diff and SOD instead shade "
              f'toward <span style="color:{HOME};">home</span> or '
              f'<span style="color:{AWAY};">away</span>, depth showing '
              f"strength. Hover any cell for its league average.</div></div>",
            unsafe_allow_html=True,
        )


def build_league_table(teams: pd.DataFrame, league_name: str) -> pd.DataFrame:
    """
    Built entirely from the cached team table — no query of its own.

    Mirrors the public league dashboard: only teams with games played, ordered
    by stored position, then PPG, points and name. AGF/AGA/ASOF/ASOA are
    per-game averages already held in list_of_teams, so no goal totals need
    aggregating from matchstats.
    """
    cols = ["StandingPosition", "StandingGames", "StandingPPG",
            "StandingPoints", "StandingWins", "StandingDraws",
            "StandingLosses", "AGF", "AGA", "ASOF", "ASOA"]

    if teams.empty or not league_name or "League" not in teams.columns:
        return pd.DataFrame()

    table = teams[teams["League"] == league_name].copy()
    if table.empty:
        return pd.DataFrame()

    for c in cols:
        if c not in table.columns:
            table[c] = pd.NA
        table[c] = pd.to_numeric(table[c], errors="coerce")

    table = table[table["StandingGames"] > 0]
    if table.empty:
        return pd.DataFrame()

    table = table.reset_index().rename(columns={
        "TeamName": "Team",
        "StandingPosition": "Pos",
        "StandingGames": "G",
        "StandingPPG": "PPG",
        "StandingPoints": "Pts",
        "StandingWins": "W",
        "StandingDraws": "D",
        "StandingLosses": "L",
    })

    table = table.sort_values(
        ["Pos", "PPG", "Pts", "Team"],
        ascending=[True, False, False, True],
    )

    return table[["Pos", "Team", "G", "PPG", "Pts", "W", "D", "L",
                  "AGF", "AGA", "ASOF", "ASOA"]].reset_index(drop=True)


def render_league_table(table: pd.DataFrame, baselines: dict,
                        home_team: str, away_team: str):
    """
    Rendered through st.dataframe so column headers stay click-sortable.
    Styler only changes how values display, so sorting still runs on the
    underlying numbers rather than the formatted strings.
    """
    if table.empty:
        st.caption("No standings available for this league.")
        return

    shaded = {
        "AGF": (baselines.get("GF"), True),
        "AGA": (baselines.get("GA"), False),
        "ASOF": (baselines.get("SOF"), True),
        "ASOA": (baselines.get("SOA"), False),
    }

    def style_row(row):
        team = row["Team"]
        fixture_side = (HOME if team == home_team
                        else AWAY if team == away_team else None)
        out = []
        for col in table.columns:
            if col in shaded:
                avg, higher = shaded[col]
                out.append(cell_style(row[col], avg, higher_is_better=higher))
            elif fixture_side:
                colour = fixture_side if col == "Team" else TEXT_1
                out.append(f"background-color:{SURFACE_2};color:{colour};"
                           "font-weight:500;")
            else:
                out.append("")
        return out

    def rate(col):
        avg, higher = shaded[col]

        def fmt(v):
            if pd.isna(v):
                return "—"
            return f"{float(v):.2f}{deviation_marker(v, avg, higher)}"
        return fmt

    formats = {
        "Pos": lambda v: "—" if pd.isna(v) else f"{int(v)}",
        "G": lambda v: "—" if pd.isna(v) else f"{int(v)}",
        "Pts": lambda v: "—" if pd.isna(v) else f"{int(v)}",
        "W": lambda v: "—" if pd.isna(v) else f"{int(v)}",
        "D": lambda v: "—" if pd.isna(v) else f"{int(v)}",
        "L": lambda v: "—" if pd.isna(v) else f"{int(v)}",
        "PPG": lambda v: "—" if pd.isna(v) else f"{float(v):.2f}",
        **{c: rate(c) for c in shaded},
    }

    styler = table.style.apply(style_row, axis=1).format(formats)

    st.dataframe(
        styler,
        hide_index=True,
        use_container_width=True,
        height=min(700, 36 * len(table) + 44),
    )



def render_header(row, home_team, away_team, home_stats, away_stats,
                  league, match_date, kickoff):
    def num(key, fmt="{:.2f}"):
        v = row.get(key)
        return fmt.format(float(v)) if v is not None and pd.notna(v) else "—"

    sodd_raw = row.get("sodd")
    sodd_color = TEXT_1
    if sodd_raw is not None and pd.notna(sodd_raw):
        # Same reasoning as the mutual-opponent Diff: sign indicates which
        # side is favoured, not good or bad.
        sodd_color = HOME if float(sodd_raw) > 0 else AWAY

    games = (f'<span style="color:{HOME};">'
             f'{int(home_stats.get("Games") or 0)}</span> vs '
             f'<span style="color:{AWAY};">'
             f'{int(away_stats.get("Games") or 0)}</span> games')

    st.markdown(
        f"""
        <div style="display:flex;justify-content:space-between;align-items:baseline;">
          <span style="font-size:19px;font-weight:500;">
            <span style="color:{HOME};">{home_team}</span>
            <span style="color:{TEXT_3};font-size:15px;"> vs </span>
            <span style="color:{AWAY};">{away_team}</span>
          </span>
          <span style="font-size:13px;color:{TEXT_2};">{kickoff}</span>
        </div>
        <div style="font-size:12px;color:{TEXT_3};margin-bottom:12px;">
          {league} · {match_date} · {games}
        </div>
        """,
        unsafe_allow_html=True,
    )

    tiles = [
        metric_pair("xG", num("xgh"), num("xga")),
        metric_pair("Expected SoT",
                    num("esoth", "{:.1f}"), num("esota", "{:.1f}")),
        metric_tile("SOD diff", num("sodd", "{:+.1f}"), sodd_color),
        metric_pair("Form",
                    f'{float(home_stats.get("Form") or 0):.2f}',
                    f'{float(away_stats.get("Form") or 0):.2f}'),
    ]
    st.markdown(
        '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(112px,1fr));'
        f'gap:8px;margin-bottom:12px;">{"".join(tiles)}</div>',
        unsafe_allow_html=True,
    )


def render_edge(row, home_team, away_team):
    """
    Model probability against market-implied probability. Both numbers were
    already displayed but never compared — this is the derived signal.
    """
    outcomes = [
        (home_team, row.get("homewin"), row.get("homeodds"), HOME),
        ("Draw", row.get("drawwin"), row.get("drawodds"), DRAW),
        (away_team, row.get("awaywin"), row.get("awayodds"), AWAY),
    ]

    parts, foot = [], []
    total = sum(float(o[1]) for o in outcomes
                if o[1] is not None and pd.notna(o[1])) or 100.0

    for label, model, odds, color in outcomes:
        if model is None or pd.isna(model):
            continue
        model = float(model)
        width = model / total * 100
        fg = PAGE_BG if color != DRAW else "#C9D1DE"
        parts.append(
            f'<div style="width:{width:.1f}%;background:{color};display:flex;'
            f"align-items:center;justify-content:center;font-size:12px;"
            f'color:{fg};">{model:.0f}%</div>'
        )

        if odds and pd.notna(odds) and float(odds) > 0:
            implied = 100.0 / float(odds)
            edge = model - implied
            ec = GOOD_HEX if edge > 0 else BAD_HEX
            foot.append(
                f'<div style="width:{width:.1f}%;text-align:center;font-size:12px;'
                f'color:{TEXT_2};">{float(odds):.2f} · '
                f'<span style="color:{ec};">{edge:+.1f}</span></div>'
            )
        else:
            foot.append(f'<div style="width:{width:.1f}%;"></div>')

    st.markdown(
        f'<div style="background:{SURFACE_1};border-radius:8px;padding:11px 12px;'
        f'margin-bottom:14px;">'
        f'<div style="font-size:11px;color:{TEXT_3};margin-bottom:7px;">'
        f"Model vs market · edge in percentage points</div>"
        f'<div style="display:flex;height:26px;border-radius:6px;overflow:hidden;'
        f'margin-bottom:5px;">{"".join(parts)}</div>'
        f'<div style="display:flex;">{"".join(foot)}</div></div>',
        unsafe_allow_html=True,
    )


# ============================================================
# APP
# ============================================================

fixtures_df = load_fixtures()
teams_df = load_teams()

if fixtures_df.empty:
    st.warning("No fixtures with valid odds found.")
    st.stop()

st.markdown(
    f'<h1 style="font-size:22px;margin:0 0 10px 0;color:{TEXT_1};">'
    f"Underlying stats</h1>",
    unsafe_allow_html=True,
)

for k in ("sel_date", "sel_league", "sel_fixture", "sel_eventid"):
    st.session_state.setdefault(k, None)

PLACEHOLDER_DATE = "Select date"
PLACEHOLDER_LEAGUE = "Select league"
PLACEHOLDER_FIXTURE = "Select fixture"


def _reset_below(level: str):
    """
    Clearing the cascade must also clear the EventID box, otherwise a stale
    ID stays visible while the app displays a different fixture. Assigning to
    widget keys is only legal inside a callback, which is where this runs.
    """
    if level == "date":
        st.session_state.sel_league = None
        st.session_state.sel_fixture = None
    if level in ("date", "league"):
        st.session_state.sel_fixture = None
    st.session_state.sel_eventid = None
    st.session_state.eventid_input = ""


def _on_date():
    sel = st.session_state.get("date_select")
    st.session_state.sel_date = None if sel == PLACEHOLDER_DATE else sel
    _reset_below("date")


def _on_league():
    sel = st.session_state.get("league_select")
    st.session_state.sel_league = None if sel == PLACEHOLDER_LEAGUE else sel
    _reset_below("league")


def _on_fixture():
    sel = st.session_state.get("fixture_select")
    st.session_state.sel_fixture = None if sel == PLACEHOLDER_FIXTURE else sel
    if all([st.session_state.sel_date, st.session_state.sel_league,
            st.session_state.sel_fixture]):
        sub = fixtures_df[
            (fixtures_df["datestr"] == st.session_state.sel_date)
            & (fixtures_df["league"] == st.session_state.sel_league)
            & (fixtures_df["fixturename"] == st.session_state.sel_fixture)
        ]
        if not sub.empty:
            st.session_state.sel_eventid = str(int(sub.iloc[0]["eventid"]))


def _on_eventid():
    """
    EventID is a direct lookup that bypasses the cascade entirely. Resetting
    the three selectbox widgets (not just the tracking keys) keeps the
    dropdowns from displaying a fixture other than the one on screen.
    """
    ev = st.session_state.get("eventid_input", "").strip()
    if not ev:
        st.session_state.sel_eventid = None
        return

    st.session_state.sel_eventid = ev
    st.session_state.sel_date = None
    st.session_state.sel_league = None
    st.session_state.sel_fixture = None
    st.session_state.date_select = PLACEHOLDER_DATE
    st.session_state.league_select = PLACEHOLDER_LEAGUE
    st.session_state.fixture_select = PLACEHOLDER_FIXTURE


# Newest first — the original sorted ascending, opening on the oldest fixture
# in the database.
date_options = sorted(fixtures_df["datestr"].unique(), reverse=True)

# A first-time visitor arriving from a link should land on something. Default
# to the next date that has fixtures, falling back to the most recent past one.
if "date_select" not in st.session_state:
    _today = dt.datetime.now(TZ).strftime("%Y-%m-%d")
    _upcoming = [d for d in date_options if d >= _today]
    _default = min(_upcoming) if _upcoming else date_options[0]
    st.session_state.date_select = _default
    st.session_state.sel_date = _default

c_event, c_date, c_league, c_fixture = st.columns([1, 1, 2, 3])

c_event.text_input("EventID", key="eventid_input", max_chars=12,
                   placeholder="EventID", on_change=_on_eventid,
                   label_visibility="collapsed")

c_date.selectbox("Date", [PLACEHOLDER_DATE] + date_options,
                 key="date_select", on_change=_on_date,
                 label_visibility="collapsed")

if st.session_state.sel_date:
    df_date = fixtures_df[fixtures_df["datestr"] == st.session_state.sel_date]
    leagues = sorted(df_date["league"].dropna().unique())
else:
    df_date, leagues = fixtures_df.iloc[0:0], []

c_league.selectbox("League", [PLACEHOLDER_LEAGUE] + leagues,
                   key="league_select", on_change=_on_league,
                   label_visibility="collapsed")

if st.session_state.sel_league and not df_date.empty:
    fixtures_list = df_date[
        df_date["league"] == st.session_state.sel_league
    ]["fixturename"].tolist()
else:
    fixtures_list = []

c_fixture.selectbox("Fixture", [PLACEHOLDER_FIXTURE] + fixtures_list,
                    key="fixture_select", on_change=_on_fixture,
                    label_visibility="collapsed")

st.caption("EventID, or date → league → fixture.")

if not st.session_state.sel_eventid:
    st.info("Select a fixture above.")
    render_help()
    st.stop()

try:
    ev = int(str(st.session_state.sel_eventid).strip())
except ValueError:
    st.warning("EventID must be a number.")
    st.stop()

row = read_sql_one(
    """
    SELECT eventid, hometeam, awayteam, league, date, kickoff,
           xgh, xga, esoth, esota, homewin, drawwin, awaywin,
           hcosod, acosod, sodd,
           home AS homeodds, draw AS drawodds, away AS awayodds
    FROM fixtures WHERE eventid = :eventid LIMIT 1
    """,
    params={"eventid": ev},
)

if not row:
    st.warning("EventID not found.")
    st.stop()

home_team = row.get("hometeam") or "Home"
away_team = row.get("awayteam") or "Away"
league = row.get("league") or "—"

_d = pd.to_datetime(row.get("date"), errors="coerce")
match_date = _d.strftime("%a %d %b") if pd.notna(_d) else "—"
_ko = row.get("kickoff")
kickoff = _ko.strftime("%H:%M") if hasattr(_ko, "strftime") else (str(_ko or "—"))

home_stats = team_row(teams_df, home_team)
away_stats = team_row(teams_df, away_team)
league_key = home_stats.get("League") or league
base = league_baselines(league_key) if league_key else {}

with st.spinner("Loading match data…"):
    matches = load_matches((home_team, away_team), days=270)

render_header(row, home_team, away_team, home_stats, away_stats,
              league, match_date, kickoff)
render_edge(row, home_team, away_team)
render_help()

tab_cmp, tab_mutual, tab_form, tab_h2h, tab_league = st.tabs(
    ["Comparison", "Mutual opponents", "Recent form", "Head to head",
     "League table"]
)

POSITIVE = ["GF", "SOF", "SF", "Opp ATT", "ATT", "H SOF", "A SOF"]
NEGATIVE = ["GA", "SOA", "SA", "Opp DEF", "DEF", "H SOA", "A SOA"]

with tab_cmp:
    st.plotly_chart(
        comparison_chart(home_stats, away_stats, home_team, away_team, base),
        use_container_width=True, config=_PLOT_CONFIG,
    )
    st.caption("Dotted line: league average.")

with tab_mutual:
    col_window, col_note = st.columns([1, 3])
    window_label = col_window.selectbox(
        "Timeframe", list(MUTUAL_WINDOWS.keys()),
        index=len(MUTUAL_WINDOWS) - 1,
        key="mutual_window", label_visibility="collapsed",
    )
    window_days = MUTUAL_WINDOWS[window_label]

    mutual = mutual_opponents(matches, home_team, away_team, teams_df,
                              days=window_days)

    if mutual.empty:
        # Point at a window that would actually return something, rather than
        # leaving the reader to try each one.
        wider = [
            label for label, days in MUTUAL_WINDOWS.items()
            if days > window_days
            and not mutual_opponents(matches, home_team, away_team,
                                     teams_df, days=days).empty
        ]
        if wider:
            st.caption(f"None in this window. Try {wider[0].lower()}.")
        else:
            st.caption("No shared opponents.")
    else:
        diff_scale = signed_scale(mutual["Diff"])
        component_scale = signed_scale(
            pd.concat([mutual["H Diff"], mutual["A Diff"]], ignore_index=True)
        )
        col_note.markdown(
            f'<div style="font-size:13px;color:{TEXT_3};padding-top:8px;">'
            f"{len(mutual)} shared</div>",
            unsafe_allow_html=True,
        )
        st.caption("Diff: shade depth = strength, colour = side favoured.")
        render_table(
            mutual,
            {**base, "ATT": base.get("Opp ATT"), "DEF": base.get("Opp DEF"),
             "H SOF": base.get("SOF"), "A SOF": base.get("SOF"),
             "H SOA": base.get("SOA"), "A SOA": base.get("SOA")},
            POSITIVE, NEGATIVE,
            {"ATT": "{:.2f}", "DEF": "{:.2f}", "H SOF": "{:.0f}",
             "H SOA": "{:.0f}", "A SOF": "{:.0f}", "A SOA": "{:.0f}",
             "H Diff": "{:+.0f}", "A Diff": "{:+.0f}", "Diff": "{:+.0f}"},
            signed={"H Diff": component_scale, "A Diff": component_scale},
            strength={"Diff": diff_scale},
        )

with tab_form:
    formats = {"Opp ATT": "{:.2f}", "Opp DEF": "{:.2f}", "GF": "{:.0f}",
               "GA": "{:.0f}", "SOF": "{:.0f}", "SOA": "{:.0f}",
               "SF": "{:.0f}", "SA": "{:.0f}"}

    for team, color in ((home_team, HOME), (away_team, AWAY)):
        st.markdown(
            f'<div style="font-size:14px;color:{color};margin:6px 0 8px;">'
            f"{team}</div>",
            unsafe_allow_html=True,
        )
        form = recent_form(matches, team, teams_df)
        if form.empty:
            st.caption("No recent matches found.")
            continue
        render_table(form, base, POSITIVE, NEGATIVE, formats)
        trend = form_trend(form, team, base, colour=color)
        if trend:
            st.plotly_chart(trend, use_container_width=True,
                            config=_PLOT_CONFIG)

with tab_h2h:
    h2h = head_to_head(matches, home_team, away_team)
    if h2h.empty:
        st.caption("No previous meetings on record.")
    else:
        latest = h2h.iloc[0].to_dict()
        st.caption(
            f"Last met {relative_day(latest['date'])} · "
            f"{latest.get('hometeam', home_team)} at home · "
            f"{len(h2h)} meetings"
        )
        st.plotly_chart(h2h_chart(latest, home_team, away_team),
                        use_container_width=True, config=_PLOT_CONFIG)

        if len(h2h) > 1:
            def score(a, b):
                out = []
                for x, y in zip(h2h[a], h2h[b]):
                    if pd.isna(x) or pd.isna(y):
                        out.append("—")
                    else:
                        out.append(f"{int(x)}–{int(y)}")
                return out

            hist = pd.DataFrame({
                "When": h2h["date"].map(relative_day),
                "Home": h2h["hometeam"],
                "Score": score("homegoals", "awaygoals"),
                "Away": h2h["awayteam"],
                "SoT": score("homeshotson", "awayshotson"),
            })
            render_table(hist, {}, [], [])

with tab_league:
    league_table = build_league_table(teams_df, league_key)
    if league_table.empty:
        st.caption("No standings for this league.")
    else:
        st.caption("Ranked by points per game.")
        render_league_table(league_table, base, home_team, away_team)

st.divider()
if st.button("Refresh data"):
    st.cache_data.clear()
    st.rerun()
