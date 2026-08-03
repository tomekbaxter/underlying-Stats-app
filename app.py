import datetime as dt
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

# Team identity: brand blue vs neutral steel. The away side is deliberately
# not red — red belongs to the stat scale only.
HOME = "#4C8DFF"
AWAY = "#8FA3BE"
DRAW = "#33415C"

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
    hoverlabel=dict(bgcolor=SURFACE_2, bordercolor=BORDER,
                    font=dict(color=TEXT_1, size=12)),
)
_PLOT_CONFIG = {"displayModeBar": False, "responsive": True}


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
        pool_size=1,
        max_overflow=2,
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
            st.error(f"Database unavailable: {exc}")
            st.stop()
        st.warning("Some data could not be loaded.")
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
    query per team per lookup.
    """
    df = read_sql_df(
        """
        SELECT "TeamName", "League", "Games", "AGF", "AGA",
               "ASOF", "ASOA", "ATT", "DEF", "Form"
        FROM list_of_teams
        """
    )
    if df.empty:
        return df
    for c in ["Games", "AGF", "AGA", "ASOF", "ASOA", "ATT", "DEF", "Form"]:
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


def mutual_opponents(matches: pd.DataFrame, home: str, away: str,
                     teams: pd.DataFrame) -> pd.DataFrame:
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
        "A SOF": merged["SOF_a"], "A SOA": merged["SOA_a"],
        "A when": merged["date_a"].map(relative_day),
        "_sort": merged["date_h"],
    })
    out["Diff"] = (pd.to_numeric(out["H SOF"], errors="coerce")
                   - pd.to_numeric(out["A SOF"], errors="coerce"))
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

        norm = (value - avg) / avg
        intensity = float(np.tanh(abs(norm) * 1.4))
        good = norm > 0 if higher_is_better else norm < 0
        target = GOOD if good else BAD

        rgb = np.clip(CELL_BASE + (target - CELL_BASE) * intensity,
                      0, 255).astype(int)
        lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        fg = PAGE_BG if lum > 140 else TEXT_1
        return f"background-color:rgb({rgb[0]},{rgb[1]},{rgb[2]});color:{fg};"
    except Exception:
        return f"background-color:{SURFACE_2};color:{TEXT_1};"


def render_table(df: pd.DataFrame, baselines: dict,
                 positive: list[str], negative: list[str],
                 formats: dict | None = None):
    """
    Hand-rendered HTML so the gradient survives, wrapped for horizontal scroll
    on narrow screens rather than crushing columns.
    """
    if df.empty:
        st.caption("No data available.")
        return

    formats = formats or {}
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

            if col in positive or col in negative:
                avg = baselines.get(col)
                style = cell_style(val, avg, higher_is_better=col in positive)
                title = f"{col} {text_val} · league {avg:.2f}" if avg else col
                cells.append(f'<td style="{style}" title="{title}">{text_val}</td>')
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
                     showticklabels=False)
    fig.update_yaxes(showgrid=False, autorange="reversed",
                     tickfont=dict(color=TEXT_2, size=12))
    return fig


def h2h_chart(row: dict, home_team: str, away_team: str):
    """
    Per-stat scaling. The original called ax.set_xlim() inside the stat loop,
    so every stat inherited the last one's range and team labels were drawn
    outside the visible axis.
    """
    stats = [("Goals", "homegoals", "awaygoals"),
             ("Shots", "homeshots", "awayshots"),
             ("On target", "homeshotson", "awayshotson"),
             ("Attacks", "homeattacks", "awayattacks"),
             ("Dangerous", "homedangerousattacks", "awaydangerousattacks")]

    rows = []
    for label, hk, ak in stats:
        h = float(row.get(hk) or 0)
        a = float(row.get(ak) or 0)
        rows.append((label, h, a, max(h, a, 1) * 1.3))

    labels = [r[0] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=[-(r[1] / r[3]) for r in rows], orientation="h",
        marker_color=HOME, width=0.45,
        customdata=[[r[1]] for r in rows],
        hovertemplate=f"<b>{home_team}</b><br>%{{y}}: "
                      "%{customdata[0]:.0f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=labels, x=[r[2] / r[3] for r in rows], orientation="h",
        marker_color=AWAY, width=0.45,
        customdata=[[r[2]] for r in rows],
        hovertemplate=f"<b>{away_team}</b><br>%{{y}}: "
                      "%{customdata[0]:.0f}<extra></extra>",
    ))

    for i, (label, h, a, ref) in enumerate(rows):
        fig.add_annotation(x=-(h / ref) - 0.05, y=i, text=f"{h:.0f}",
                           showarrow=False, xanchor="right",
                           font=dict(color=TEXT_1, size=12))
        fig.add_annotation(x=(a / ref) + 0.05, y=i, text=f"{a:.0f}",
                           showarrow=False, xanchor="left",
                           font=dict(color=TEXT_1, size=12))

    fig.add_vline(x=0, line_color=BORDER, line_width=1)
    fig.update_layout(**_PLOT_LAYOUT, barmode="overlay",
                      height=44 * len(rows) + 30)
    fig.update_xaxes(range=[-1.4, 1.4], showgrid=False, zeroline=False,
                     showticklabels=False)
    fig.update_yaxes(showgrid=False, autorange="reversed",
                     tickfont=dict(color=TEXT_2, size=12))
    return fig


def form_trend(df: pd.DataFrame, team: str, base: dict):
    """Shots-on-target for and against across recent matches."""
    if df.empty:
        return None
    d = df.iloc[::-1]
    x = list(range(len(d)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=pd.to_numeric(d["SOF"], errors="coerce"),
        mode="lines+markers", line=dict(color=HOME, width=2),
        marker=dict(size=6), name="SOF",
        customdata=d["Opponent"],
        hovertemplate="vs %{customdata}<br>SOF %{y:.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=x, y=pd.to_numeric(d["SOA"], errors="coerce"),
        mode="lines+markers", line=dict(color=AWAY, width=2, dash="dot"),
        marker=dict(size=6), name="SOA",
        customdata=d["Opponent"],
        hovertemplate="vs %{customdata}<br>SOA %{y:.0f}<extra></extra>"))

    if base.get("SOF"):
        fig.add_hline(y=base["SOF"], line=dict(color=TEXT_3, width=1,
                                               dash="dash"))

    fig.update_layout(**_PLOT_LAYOUT, height=180)
    fig.update_xaxes(showgrid=False, showticklabels=False)
    fig.update_yaxes(showgrid=True, gridcolor=BORDER, zeroline=False,
                     tickfont=dict(color=TEXT_3, size=11))
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


def render_header(row, home_team, away_team, home_stats, away_stats,
                  league, match_date, kickoff):
    def num(key, fmt="{:.2f}"):
        v = row.get(key)
        return fmt.format(float(v)) if v is not None and pd.notna(v) else "—"

    sodd_raw = row.get("sodd")
    sodd_color = TEXT_1
    if sodd_raw is not None and pd.notna(sodd_raw):
        sodd_color = GOOD_HEX if float(sodd_raw) > 0 else BAD_HEX

    games = (f'{int(home_stats.get("Games") or 0)} vs '
             f'{int(away_stats.get("Games") or 0)} games')

    st.markdown(
        f"""
        <div style="display:flex;justify-content:space-between;align-items:baseline;">
          <span style="font-size:19px;font-weight:500;color:{TEXT_1};">
            {home_team} <span style="color:{TEXT_3};font-size:15px;">vs</span> {away_team}
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
        metric_tile("xG", f'{num("xgh")} · {num("xga")}'),
        metric_tile("Expected SoT", f'{num("esoth", "{:.1f}")} · {num("esota", "{:.1f}")}'),
        metric_tile("SOD diff", num("sodd", "{:+.1f}"), sodd_color),
        metric_tile("Form", f'{float(home_stats.get("Form") or 0):.2f} · '
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

st.caption("Enter an EventID directly, or filter by date → league → fixture.")

if not st.session_state.sel_eventid:
    st.info("Select a fixture, or enter an EventID.")
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

tab_cmp, tab_mutual, tab_form, tab_h2h = st.tabs(
    ["Comparison", "Mutual opponents", "Recent form", "Head to head"]
)

POSITIVE = ["GF", "SOF", "SF", "Opp ATT", "ATT", "H SOF", "A SOF", "Diff"]
NEGATIVE = ["GA", "SOA", "SA", "Opp DEF", "DEF", "H SOA", "A SOA"]

with tab_cmp:
    st.plotly_chart(
        comparison_chart(home_stats, away_stats, home_team, away_team, base),
        use_container_width=True, config=_PLOT_CONFIG,
    )
    st.caption("Dotted line marks the league average for each stat.")

with tab_mutual:
    mutual = mutual_opponents(matches, home_team, away_team, teams_df)
    if mutual.empty:
        st.caption("No shared opponents in the last 270 days.")
    else:
        st.caption(
            f"{len(mutual)} shared opponents · Diff is "
            f"{home_team} shots on target minus {away_team}, same opponent."
        )
        render_table(
            mutual,
            {**base, "ATT": base.get("Opp ATT"), "DEF": base.get("Opp DEF"),
             "H SOF": base.get("SOF"), "A SOF": base.get("SOF"),
             "H SOA": base.get("SOA"), "A SOA": base.get("SOA"),
             "Diff": 0.001},
            POSITIVE, NEGATIVE,
            {"ATT": "{:.2f}", "DEF": "{:.2f}", "H SOF": "{:.0f}",
             "H SOA": "{:.0f}", "A SOF": "{:.0f}", "A SOA": "{:.0f}",
             "Diff": "{:+.0f}"},
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
        trend = form_trend(form, team, base)
        if trend:
            st.plotly_chart(trend, use_container_width=True,
                            config=_PLOT_CONFIG)

with tab_h2h:
    h2h = head_to_head(matches, home_team, away_team)
    if h2h.empty:
        st.caption("No previous meetings on record.")
    else:
        latest = h2h.iloc[0].to_dict()
        st.caption(f"Last met {relative_day(latest['date'])} · "
                   f"{len(h2h)} meetings on record")
        st.plotly_chart(h2h_chart(latest, latest.get("hometeam", home_team),
                                  latest.get("awayteam", away_team)),
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

st.divider()
if st.button("Refresh data"):
    st.cache_data.clear()
    st.rerun()
