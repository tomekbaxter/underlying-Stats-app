import os
import datetime as dt
from zoneinfo import ZoneInfo
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sqlalchemy import create_engine, text

# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(page_title="Underlying Stats", layout="wide")

st.markdown(
    """
    <style>
    header[data-testid="stHeader"] { display: none; }
    .block-container { padding-top: 1.2rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    html, body, .stApp { background-color: #0e1117; color: #e6e6e6; }

    .block-container {
        padding-top: 4rem;
        padding-bottom: 0rem;
        padding-left: 1.4rem;
        padding-right: 1.4rem;
    }

    div.stButton > button {
        width: 100%;
        height: 3.1em;
        font-size: 1.05rem;
        font-weight: 600;
        border-radius: 8px;
        background-color: #111827;
        color: #e6e6e6;
        border: 1px solid #2a2f3a;
    }
    div.stButton > button:hover {
        background-color: #1a2233;
        border-color: #3b4252;
    }

    .stTextInput input, .stSelectbox div[data-baseweb="select"] > div {
        background-color: #111827 !important;
        color: #e6e6e6 !important;
        border: 1px solid #2a2f3a !important;
        border-radius: 8px !important;
    }

    div[role="listbox"] {
        background-color: #111827 !important;
        color: #e6e6e6 !important;
        border: 1px solid #2a2f3a !important;
    }

    h1, h2, h3, h4 { color: #e6e6e6 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

TZ = ZoneInfo("Europe/London")

# ============================================================
# SUPABASE / POSTGRES CONNECTION
# ============================================================

CACHE_TTL_SECONDS = 900

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

    db_url = (
        f"postgresql+psycopg2://{quote_plus(user)}:"
        f"{quote_plus(password)}@{host}:{port}/{db}"
        f"?sslmode=require"
    )

    return create_engine(
        db_url,
        pool_pre_ping=True,
        pool_recycle=300,
        future=True,
    )

ENGINE = get_engine()

def read_sql_df(sql: str, params: dict | None = None) -> pd.DataFrame:
    try:
        with ENGINE.connect() as conn:
            return pd.read_sql(text(sql), conn, params=params or {})
    except Exception as e:
        st.error(f"Database query failed: {e}")
        st.stop()

def read_sql_one(sql: str, params: dict | None = None) -> dict | None:
    df = read_sql_df(sql, params=params)
    if df.empty:
        return None
    return df.iloc[0].to_dict()
# ============================================================
# DATA HELPERS (SUPABASE)
# ============================================================

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_fixtures_with_odds() -> pd.DataFrame:
    sql = """
        SELECT
            eventid,
            hometeam,
            awayteam,
            league,
            date,
            kickoff,
            home  AS homeodds,
            draw  AS drawodds,
            away  AS awayodds
        FROM fixtures
        WHERE home IS NOT NULL AND away IS NOT NULL
          AND home > 0 AND away > 0
        ORDER BY date DESC, league, hometeam
    """
    df = read_sql_df(sql)

    for c in ["homeodds", "drawodds", "awayodds"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[(df["homeodds"] > 0) & (df["awayodds"] > 0)].copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["datestr"] = df["date"].dt.strftime("%Y-%m-%d")
    df["fixturename"] = df["hometeam"].fillna("N/A") + " vs " + df["awayteam"].fillna("N/A")
    return df

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def fetch_fixture_row(event_id: str) -> dict | None:
    sql = """
        SELECT
            eventid, hometeam, awayteam, league, date, kickoff,
            xgh, xga, esoth, esota,
            homewin, drawwin, awaywin,
            hcosod, acosod, sodd,
            home AS homeodds, draw AS drawodds, away AS awayodds
        FROM fixtures
        WHERE eventid = :eventid
        LIMIT 1
    """
    try:
        ev = int(str(event_id).strip())
    except Exception:
        return None
    return read_sql_one(sql, params={"eventid": ev})

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_team_stats(team_name: str) -> dict:
    sql = """
        SELECT
            "Games"  AS games,
            "AGF"    AS agf,
            "AGA"    AS aga,
            "ASOF"   AS asof,
            "ASOA"   AS asoa,
            "ATT"    AS att,
            "DEF"    AS def,
            "Form"   AS form,
            "League" AS league
        FROM list_of_teams
        WHERE "TeamName" = :teamname
        LIMIT 1
    """
    row = read_sql_one(sql, params={"teamname": team_name})
    if not row:
        return {"Games": 0, "AGF": 0, "AGA": 0, "ASOF": 0, "ASOA": 0, "ATT": 0, "DEF": 0, "Form": None, "League": None}

    return {
        "Games": row.get("games") or 0,
        "AGF": row.get("agf") or 0,
        "AGA": row.get("aga") or 0,
        "ASOF": row.get("asof") or 0,
        "ASOA": row.get("asoa") or 0,
        "ATT": row.get("att") or 0,
        "DEF": row.get("def") or 0,
        "Form": row.get("form"),
        "League": row.get("league"),
    }

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_team_att_def(team_name: str) -> dict:
    sql = """
        SELECT
            "ATT" AS att,
            "DEF" AS def
        FROM list_of_teams
        WHERE "TeamName" = :teamname
        LIMIT 1
    """
    row = read_sql_one(sql, params={"teamname": team_name})
    if not row:
        return {"ATT": "N/A", "DEF": "N/A"}
    return {"ATT": row.get("att", "N/A"), "DEF": row.get("def", "N/A")}

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_latest_head_to_head_row(home_team: str, away_team: str) -> dict | None:
    sql = """
        SELECT
            "Date" AS date,
            "HomeTeam" AS hometeam,
            "AwayTeam" AS awayteam,
            "HomeGoals" AS homegoals,
            "AwayGoals" AS awaygoals,
            "HomeShots" AS homeshots,
            "AwayShots" AS awayshots,
            "HomeShotsOn" AS homeshotson,
            "AwayShotsOn" AS awayshotson,
            "HomeAttacks" AS homeattacks,
            "AwayAttacks" AS awayattacks,
            "HomeDangerousAttacks" AS homedangerousattacks,
            "AwayDangerousAttacks" AS awaydangerousattacks
        FROM matchstats
        WHERE ("HomeTeam" = :h1 AND "AwayTeam" = :a1)
           OR ("HomeTeam" = :h2 AND "AwayTeam" = :a2)
        ORDER BY "Date" DESC
        LIMIT 1
    """
    return read_sql_one(sql, params={"h1": home_team, "a1": away_team, "h2": away_team, "a2": home_team})

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_opponents_past_3_months(team_name: str) -> set:
    since = (dt.datetime.now(TZ) - dt.timedelta(days=100)).date()
    sql = """
        SELECT
            "HomeTeam" AS hometeam,
            "AwayTeam" AS awayteam
        FROM matchstats
        WHERE "Date" >= :since
          AND ("HomeTeam" = :t OR "AwayTeam" = :t)
    """
    df = read_sql_df(sql, params={"since": since, "t": team_name})
    opponents = set()
    for _, r in df.iterrows():
        opponent = r["awayteam"] if r["hometeam"] == team_name else r["hometeam"]
        if pd.notna(opponent):
            opponents.add(opponent)
    return opponents

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_match_data_extended(team_name: str, opponent: str) -> dict:
    sql = """
        SELECT
            "Date" AS date,
            "HomeShotsOn" AS homeshotson,
            "AwayShotsOn" AS awayshotson,
            "HomeTeam" AS hometeam,
            "AwayTeam" AS awayteam
        FROM matchstats
        WHERE ("HomeTeam" = :t AND "AwayTeam" = :o)
           OR ("HomeTeam" = :o AND "AwayTeam" = :t)
        ORDER BY "Date" DESC
        LIMIT 1
    """
    row = read_sql_one(sql, params={"t": team_name, "o": opponent})
    if not row:
        return {"Date": "N/A", "SOF": "N/A", "SOA": "N/A"}

    d = row.get("date")
    date_str = pd.to_datetime(d, errors="coerce").strftime("%Y-%m-%d") if d is not None and pd.notna(pd.to_datetime(d, errors="coerce")) else "N/A"

    home_sof = row.get("homeshotson")
    away_sof = row.get("awayshotson")

    if row.get("hometeam") == team_name:
        sof, soa = home_sof, away_sof
    else:
        sof, soa = away_sof, home_sof

    return {"Date": date_str, "SOF": sof, "SOA": soa}

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_league_wide_baselines(league_name: str) -> dict:
    since = (dt.datetime.now(TZ) - dt.timedelta(days=270)).date()

    sql = """
        SELECT
            "HomeShotsOn" AS homeshotson,
            "AwayShotsOn" AS awayshotson
        FROM matchstats
        WHERE "League" = :league AND "Date" >= :since
    """

    df = read_sql_df(sql, params={"league": league_name, "since": since})

    sof_values = pd.to_numeric(
        pd.concat([df["homeshotson"], df["awayshotson"]], ignore_index=True),
        errors="coerce"
    ).dropna()

    soa_values = pd.to_numeric(
        pd.concat([df["awayshotson"], df["homeshotson"]], ignore_index=True),
        errors="coerce"
    ).dropna()

    tdf = read_sql_df(
        """
        SELECT
            "ATT" AS att,
            "DEF" AS def_rating
        FROM list_of_teams
        WHERE "League" = :league
        """,
        params={"league": league_name},
    )

    att_values = pd.to_numeric(tdf["att"], errors="coerce").replace(0, np.nan).dropna()
    def_values = pd.to_numeric(tdf["def_rating"], errors="coerce").replace(0, np.nan).dropna()

    return {
        "SOF": float(sof_values.mean()) if not sof_values.empty else 0.0,
        "SOA": float(soa_values.mean()) if not soa_values.empty else 0.0,
        "ATT": float(att_values.mean()) if not att_values.empty else 0.0,
        "DEF": float(def_values.mean()) if not def_values.empty else 0.0,
    }

def gradient_background(value, avg, positive=True):
    try:
        value = float(value)
        diff = value - avg
        norm_diff = diff / avg if avg != 0 else 0

        deep_green = np.array([46, 125, 50])
        deep_red = np.array([200, 50, 40])
        white = np.array([255, 255, 255])

        base_color = deep_green if positive else deep_red
        reverse_color = deep_red if positive else deep_green

        if norm_diff > 0:
            color = white + (base_color - white) * min(norm_diff, 1)
        else:
            color = white + (reverse_color - white) * min(abs(norm_diff), 1)

        color = np.clip(color, 0, 255).astype(int)
        return f"background-color: rgb({color[0]}, {color[1]}, {color[2]}); color: black;"
    except Exception:
        return "background-color: #1E1E1E; color: black;"

def get_common_opponents_data_extended(home_team: str, away_team: str) -> list[dict]:
    home_opponents = get_opponents_past_3_months(home_team)
    away_opponents = get_opponents_past_3_months(away_team)
    common = home_opponents.intersection(away_opponents)

    data = []
    for opponent in common:
        home_data = get_match_data_extended(home_team, opponent)
        away_data = get_match_data_extended(away_team, opponent)
        opp_stats = get_team_att_def(opponent)

        data.append(
            {
                "Opponent": opponent,
                "ATT": opp_stats["ATT"],
                "DEF": opp_stats["DEF"],
                "Home Date": home_data["Date"],
                "Home SOF": home_data["SOF"],
                "Home SOA": home_data["SOA"],
                "Away SOF": away_data["SOF"],
                "Away SOA": away_data["SOA"],
                "Away Date": away_data["Date"],
            }
        )

    data.sort(
        key=lambda x: dt.datetime.strptime(x["Home Date"], "%Y-%m-%d") if x["Home Date"] != "N/A" else dt.datetime.min,
        reverse=True,
    )
    return data

def display_mutual_opponents_section(home_team: str, away_team: str, league_name: str | None):
    st.subheader("Mutual Opponent Games")

    league_data = get_league_wide_baselines(league_name) if league_name else None
    data = get_common_opponents_data_extended(home_team, away_team)

    if not data:
        st.write("No common opponents found in the past 3 months.")
        return

    df = pd.DataFrame(data)

    sof_cols = ["Home SOF", "Away SOF"]
    soa_cols = ["Home SOA", "Away SOA"]

    # Ensure numeric where needed (keeps gradients correct)
    for c in ["ATT", "DEF"] + sof_cols + soa_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    sof_avg = league_data["SOF"] if league_data else 1
    soa_avg = league_data["SOA"] if league_data else 1
    att_avg = league_data["ATT"] if league_data else 1
    def_avg = league_data["DEF"] if league_data else 1

    styled = (
        df.style
        # Display: NO decimal places for all numeric columns
        .format(
            {
                "ATT": "{:.2f}",
                "DEF": "{:.2f}",
                "Home SOF": "{:.0f}",
                "Home SOA": "{:.0f}",
                "Away SOF": "{:.0f}",
                "Away SOA": "{:.0f}",
            },
            na_rep="N/A",
        )
        .applymap(lambda x: gradient_background(x, att_avg, positive=True), subset=["ATT"])
        .applymap(lambda x: gradient_background(x, def_avg, positive=False), subset=["DEF"])
        .applymap(lambda x: gradient_background(x, sof_avg, positive=True), subset=sof_cols)
        .applymap(lambda x: gradient_background(x, soa_avg, positive=False), subset=soa_cols)
        .set_table_styles(
            [
                {
                    "selector": "thead th",
                    "props": [
                        ("background-color", "#2B2B2B"),
                        ("color", "#E0E0E0"),
                        ("font-weight", "bold"),
                        ("text-align", "center"),
                    ],
                },
                {
                    "selector": "tbody td",
                    "props": [
                        ("text-align", "center"),
                        ("color", "black"),
                        ("border", "1px solid #444444"),
                    ],
                },
                {
                    "selector": "table",
                    "props": [
                        ("width", "100%"),
                        ("table-layout", "fixed"),
                        ("border-collapse", "collapse"),
                    ],
                },
            ]
        )
        .hide(axis="index")
    )

    st.write(styled)

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_league_recent_baselines(league_name: str) -> dict:
    since = (dt.datetime.now(TZ) - dt.timedelta(days=270)).date()

    sql = """
        SELECT
            "HomeGoals" AS homegoals,
            "AwayGoals" AS awaygoals,
            "HomeShotsOn" AS homeshotson,
            "AwayShotsOn" AS awayshotson,
            "HomeShots" AS homeshots,
            "AwayShots" AS awayshots
        FROM matchstats
        WHERE "League" = :league AND "Date" >= :since
    """

    df = read_sql_df(sql, params={"league": league_name, "since": since})

    gf = pd.to_numeric(pd.concat([df["homegoals"], df["awaygoals"]], ignore_index=True), errors="coerce").dropna()
    ga = pd.to_numeric(pd.concat([df["awaygoals"], df["homegoals"]], ignore_index=True), errors="coerce").dropna()

    sof = pd.to_numeric(pd.concat([df["homeshotson"], df["awayshotson"]], ignore_index=True), errors="coerce").dropna()
    soa = pd.to_numeric(pd.concat([df["awayshotson"], df["homeshotson"]], ignore_index=True), errors="coerce").dropna()

    sf = pd.to_numeric(pd.concat([df["homeshots"], df["awayshots"]], ignore_index=True), errors="coerce").dropna()
    sa = pd.to_numeric(pd.concat([df["awayshots"], df["homeshots"]], ignore_index=True), errors="coerce").dropna()

    tdf = read_sql_df(
        """
        SELECT
            "ATT" AS att,
            "DEF" AS def_rating
        FROM list_of_teams
        WHERE "League" = :league
        """,
        params={"league": league_name},
    )

    att = pd.to_numeric(tdf["att"], errors="coerce").replace(0, np.nan).dropna()
    deff = pd.to_numeric(tdf["def_rating"], errors="coerce").replace(0, np.nan).dropna()

    return {
        "GF": float(gf.mean()) if not gf.empty else 0.0,
        "GA": float(ga.mean()) if not ga.empty else 0.0,
        "SOF": float(sof.mean()) if not sof.empty else 0.0,
        "SOA": float(soa.mean()) if not soa.empty else 0.0,
        "SF": float(sf.mean()) if not sf.empty else 0.0,
        "SA": float(sa.mean()) if not sa.empty else 0.0,
        "Opp ATT": float(att.mean()) if not att.empty else 0.0,
        "Opp DEF": float(deff.mean()) if not deff.empty else 0.0,
    }

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_opponent_att_def(opponent_name: str) -> tuple[float | None, float | None]:
    sql = """
        SELECT
            "ATT" AS att,
            "DEF" AS def
        FROM list_of_teams
        WHERE "TeamName" = :t
        LIMIT 1
    """
    row = read_sql_one(sql, params={"t": opponent_name})
    if not row:
        return None, None
    return row.get("att"), row.get("def")

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_recent_form(team_name: str) -> list[dict]:
    since = (dt.datetime.now(TZ) - dt.timedelta(days=270)).date()
    sql = """
        SELECT
            "Date" AS date,
            "HomeTeam" AS hometeam,
            "AwayTeam" AS awayteam,
            "HomeGoals" AS homegoals,
            "AwayGoals" AS awaygoals,
            "HomeShotsOn" AS homeshotson,
            "AwayShotsOn" AS awayshotson,
            "HomeShots" AS homeshots,
            "AwayShots" AS awayshots
        FROM matchstats
        WHERE ("HomeTeam" = :t OR "AwayTeam" = :t)
          AND "Date" >= :since
        ORDER BY "Date" DESC
        LIMIT 8
    """
    df = read_sql_df(sql, params={"t": team_name, "since": since})

    data = []
    for _, r in df.iterrows():
        is_home = r["hometeam"] == team_name
        opponent = r["awayteam"] if is_home else r["hometeam"]
        opp_att, opp_def = get_opponent_att_def(opponent)

        goals_for = r["homegoals"] if is_home else r["awaygoals"]
        goals_against = r["awaygoals"] if is_home else r["homegoals"]
        shots_on_for = r["homeshotson"] if is_home else r["awayshotson"]
        shots_on_against = r["awayshotson"] if is_home else r["homeshotson"]
        shots_for = r["homeshots"] if is_home else r["awayshots"]
        shots_against = r["awayshots"] if is_home else r["homeshots"]

        d = pd.to_datetime(r["date"], errors="coerce")
        date_str = d.strftime("%d/%m/%Y") if pd.notna(d) else "N/A"

        data.append(
            {
                "Opponent": opponent,
                "Opp ATT": opp_att,
                "Opp DEF": opp_def,
                "Date": date_str,
                "GF": goals_for,
                "SOF": shots_on_for,
                "SF": shots_for,
                "GA": goals_against,
                "SOA": shots_on_against,
                "SA": shots_against,
            }
        )

    return data

def display_recent_form_section(home_team: str, away_team: str, league_name: str | None):
    st.subheader("Recent Form")

    league_baselines = get_league_recent_baselines(league_name) if league_name else None

    home_form = get_recent_form(home_team)
    away_form = get_recent_form(away_team)

    home_df = pd.DataFrame(home_form)
    away_df = pd.DataFrame(away_form)

    positive_cols = ["GF", "SOF", "SF", "Opp ATT"]
    negative_cols = ["GA", "SOA", "SA", "Opp DEF"]

    def apply_form_styling(df: pd.DataFrame):
        if df.empty:
            return None

        stat_cols = ["GF", "GA", "SOF", "SOA", "SF", "SA", "Opp ATT", "Opp DEF"]
        df[stat_cols] = df[stat_cols].apply(pd.to_numeric, errors="coerce")

        avg_values = (
            league_baselines
            if league_baselines
            else {col: float(np.nanmean(df[col])) for col in stat_cols}
        )

        styled = (
            df.style
            # Display: NO decimal places for all numeric columns
            .format(
                {
                    "GF": "{:.0f}",
                    "GA": "{:.0f}",
                    "SOF": "{:.0f}",
                    "SOA": "{:.0f}",
                    "SF": "{:.0f}",
                    "SA": "{:.0f}",
                    "Opp ATT": "{:.2f}",
                    "Opp DEF": "{:.2f}",
                },
                na_rep="N/A",
            )
            .applymap(lambda x: gradient_background(x, avg_values["Opp ATT"], positive=True), subset=["Opp ATT"])
            .applymap(lambda x: gradient_background(x, avg_values["Opp DEF"], positive=False), subset=["Opp DEF"])
            .applymap(lambda x: gradient_background(x, avg_values["GF"], positive=True), subset=["GF"])
            .applymap(lambda x: gradient_background(x, avg_values["GA"], positive=False), subset=["GA"])
            .applymap(lambda x: gradient_background(x, avg_values["SOF"], positive=True), subset=["SOF"])
            .applymap(lambda x: gradient_background(x, avg_values["SOA"], positive=False), subset=["SOA"])
            .applymap(lambda x: gradient_background(x, avg_values["SF"], positive=True), subset=["SF"])
            .applymap(lambda x: gradient_background(x, avg_values["SA"], positive=False), subset=["SA"])
            .set_table_styles(
                [
                    {
                        "selector": "thead th",
                        "props": [
                            ("background-color", "#2B2B2B"),
                            ("color", "#E0E0E0"),
                            ("font-weight", "bold"),
                            ("text-align", "center"),
                        ],
                    },
                    {
                        "selector": "tbody td",
                        "props": [
                            ("text-align", "center"),
                            ("color", "black"),
                            ("border", "1px solid #444444"),
                        ],
                    },
                    {
                        "selector": "table",
                        "props": [
                            ("width", "100%"),
                            ("table-layout", "fixed"),
                            ("border-collapse", "collapse"),
                        ],
                    },
                ]
            )
            .hide(axis="index")
        )
        return styled


    st.markdown(f"### {home_team} Recent Form")
    if not home_df.empty:
        st.write(apply_form_styling(home_df))
    else:
        st.write(f"No recent matches found for {home_team}.")

    st.markdown(f"### {away_team} Recent Form")
    if not away_df.empty:
        st.write(apply_form_styling(away_df))
    else:
        st.write(f"No recent matches found for {away_team}.")

# ============================================================
# CHARTS
# ============================================================

def generate_team_stats_chart(home_stats, away_stats):
    stats = ["AGF", "AGA", "ASOF", "ASOA", "ATT", "DEF"]
    home_values = [home_stats.get(stat, 0) for stat in stats]
    away_values = [away_stats.get(stat, 0) for stat in stats]

    home_color = "#3498db"
    away_color = "#e74c3c"
    text_color = "#E0E0E0"
    bg_color = "#1E1E1E"

    fig, ax = plt.subplots(figsize=(6, 3))
    bar_height = 0.15
    y_positions = range(len(stats))

    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    max_value = 15
    indent_offset = 0.5

    ax.barh(y_positions, [-x for x in home_values], height=bar_height, color=home_color, align="center")
    for i, value in enumerate(home_values):
        ax.text(-value - indent_offset, i, f"{value:.2f}", va="center", ha="right", fontsize=10, color=text_color, fontweight="bold")

    ax.barh(y_positions, away_values, height=bar_height, color=away_color, align="center")
    for i, value in enumerate(away_values):
        ax.text(value + indent_offset, i, f"{value:.2f}", va="center", ha="left", fontsize=10, color=text_color, fontweight="bold")

    for i, stat in enumerate(stats):
        ax.text(0, i, stat, va="center", ha="center", fontsize=12, color=text_color, fontweight="bold")

    ax.set_yticks([])
    ax.set_xticks([])
    ax.axvline(0, color="#444444", lw=1)
    ax.set_xlim(-max_value, max_value)
    ax.invert_yaxis()
    ax.set_frame_on(False)

    return fig

def display_team_stats_section(home_stats, away_stats):
    st.subheader("Team Stats")

    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; align-items: center; padding-bottom: 5px;">
            <span style="flex: 1; text-align: right; font-size: 20px; font-weight: bold; color: #3498db;">{home_stats.get('Games', 0)}</span>
            <span style="flex: 1; text-align: center; font-size: 16px; color: #E0E0E0;">Games</span>
            <span style="flex: 1; text-align: left; font-size: 20px; font-weight: bold; color: #e74c3c;">{away_stats.get('Games', 0)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.pyplot(generate_team_stats_chart(home_stats, away_stats))

def generate_head_to_head_bar_chart(match_row: dict):
    home_team = match_row.get("hometeam", "")
    away_team = match_row.get("awayteam", "")

    stats = [
        ("Goals", match_row.get("homegoals"), match_row.get("awaygoals"), 10),
        ("Shots", match_row.get("homeshots"), match_row.get("awayshots"), 30),
        ("Shots on Target", match_row.get("homeshotson"), match_row.get("awayshotson"), 20),
        ("Attacks", match_row.get("homeattacks"), match_row.get("awayattacks"), 100),
        ("Dangerous Attacks", match_row.get("homedangerousattacks"), match_row.get("awaydangerousattacks"), 80),
    ]

    home_color = "#3498db"
    away_color = "#e74c3c"
    text_color = "#E0E0E0"
    bg_color = "#1E1E1E"

    fig, ax = plt.subplots(figsize=(6, 3))
    bar_height = 0.13
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    max_value = max([s[3] for s in stats])

    indent_offset = 0.8
    label_offset = 0.2
    team_font_size = 12

    ax.text(-max_value * 1.15, -1.5, home_team, fontsize=team_font_size, color=home_color, fontweight="bold", ha="left")
    ax.text(max_value * 1.15, -1.5, away_team, fontsize=team_font_size, color=away_color, fontweight="bold", ha="right")

    for i, (label, home_val, away_val, max_val) in enumerate(stats):
        home_val = 0 if home_val is None else home_val
        away_val = 0 if away_val is None else away_val

        # bars can stay numeric
        ax.barh(i, -home_val, height=bar_height, color=home_color, align="center")
        ax.barh(i, away_val, height=bar_height, color=away_color, align="center")

        # text: force integer display (no decimals)
        ax.text(
            -home_val - indent_offset,
            i,
            f"{int(round(home_val))}",
            va="center",
            ha="right",
            fontsize=12,
            color=text_color,
            fontweight="bold",
        )

        ax.text(
            away_val + indent_offset,
            i,
            f"{int(round(away_val))}",
            va="center",
            ha="left",
            fontsize=12,
            color=text_color,
            fontweight="bold",
        )

        ax.text(
            0,
            i - label_offset,
            label,
            va="bottom",
            ha="center",
            fontsize=11,
            color=text_color,
            fontweight="bold",
        )

        ax.set_xlim(-max_val, max_val)


    ax.axvline(0, color="#444444", lw=1)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.invert_yaxis()
    ax.set_frame_on(False)

    return fig

# ============================================================
# UI: FIXTURE SEARCH (SUPABASE)
# ============================================================

if st.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()

fixtures_df = load_fixtures_with_odds()

if fixtures_df.empty:
    st.warning("No fixtures with valid odds found in Supabase.")
    st.stop()

st.markdown(
    """
    <div style="display:flex; justify-content:space-between; align-items:center; padding-bottom:10px;">
        <h1 style="margin:0; padding:0; color:#E0E0E0;">Underlying Stats</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("Search Fixture")
st.markdown("<small style='color:#AAAAAA;'>Select Date -> League -> Fixture or enter EventID</small>", unsafe_allow_html=True)

for k in ("sel_date", "sel_league", "sel_fixture", "sel_eventid"):
    if k not in st.session_state:
        st.session_state[k] = None

def _on_eventid_change():
    ev = st.session_state.get("eventid_input", "").strip()
    st.session_state.sel_eventid = ev if ev else None
    if st.session_state.sel_eventid:
        st.session_state.sel_date = None
        st.session_state.sel_league = None
        st.session_state.sel_fixture = None

def _on_date_change():
    sel = st.session_state.get("date_select")
    st.session_state.sel_date = None if not sel or sel.startswith("—") else sel
    st.session_state.sel_league = None
    st.session_state.sel_fixture = None
    st.session_state.sel_eventid = None
    st.session_state.eventid_input = ""

def _on_league_change():
    sel = st.session_state.get("league_select")
    st.session_state.sel_league = None if not sel or sel.startswith("—") else sel
    st.session_state.sel_fixture = None
    st.session_state.sel_eventid = None
    st.session_state.eventid_input = ""

def _on_fixture_change():
    sel = st.session_state.get("fixture_select")
    st.session_state.sel_fixture = None if not sel or sel.startswith("—") else sel
    st.session_state.sel_eventid = None
    st.session_state.eventid_input = ""

    if st.session_state.sel_date and st.session_state.sel_league and st.session_state.sel_fixture:
        d = st.session_state.sel_date
        lg = st.session_state.sel_league
        fx = st.session_state.sel_fixture

        sub = fixtures_df[
            (fixtures_df["datestr"] == d)
            & (fixtures_df["league"] == lg)
            & (fixtures_df["fixturename"] == fx)
        ]
        if not sub.empty:
            st.session_state.sel_eventid = str(int(sub.iloc[0]["eventid"]))

col_eventid, col_date, col_league, col_fixture = st.columns([1, 1, 1, 3])

col_eventid.text_input(
    "EventID",
    key="eventid_input",
    value=st.session_state.get("eventid_input", "") if st.session_state.get("eventid_input") else "",
    max_chars=12,
    placeholder="Enter EventID",
    on_change=_on_eventid_change,
)

date_options = sorted(fixtures_df["datestr"].unique())
default_date_label = "— Select Date —"
date_index = 0
if st.session_state.sel_date and st.session_state.sel_date in date_options:
    date_index = date_options.index(st.session_state.sel_date) + 1

col_date.selectbox(
    "Date",
    options=[default_date_label] + date_options,
    index=date_index,
    key="date_select",
    on_change=_on_date_change,
)

if st.session_state.sel_date:
    df_date = fixtures_df[fixtures_df["datestr"] == st.session_state.sel_date]
    leagues_for_date = sorted(df_date["league"].dropna().unique())
else:
    df_date = pd.DataFrame(columns=fixtures_df.columns)
    leagues_for_date = []

default_league_label = "— Select League —"
league_index = 0
if st.session_state.sel_league and st.session_state.sel_league in leagues_for_date:
    league_index = leagues_for_date.index(st.session_state.sel_league) + 1

col_league.selectbox(
    "League",
    options=[default_league_label] + leagues_for_date,
    index=league_index,
    key="league_select",
    on_change=_on_league_change,
)

if st.session_state.sel_league and not df_date.empty:
    df_league = df_date[df_date["league"] == st.session_state.sel_league]
    fixtures_for_league = df_league["fixturename"].tolist()
else:
    df_league = pd.DataFrame(columns=fixtures_df.columns)
    fixtures_for_league = []

default_fixture_label = "— Select Fixture —"
fixture_index = 0
if st.session_state.sel_fixture and st.session_state.sel_fixture in fixtures_for_league:
    fixture_index = fixtures_for_league.index(st.session_state.sel_fixture) + 1

col_fixture.selectbox(
    "Fixture",
    options=[default_fixture_label] + fixtures_for_league,
    index=fixture_index,
    key="fixture_select",
    on_change=_on_fixture_change,
)

# ============================================================
# DISPLAY FIXTURE DETAILS (NO EXTERNAL API CALLS)
# ============================================================

event_id_to_use = st.session_state.sel_eventid

if event_id_to_use:
    row = fetch_fixture_row(event_id_to_use)
    if row:
        home_team = row.get("hometeam", "N/A")
        away_team = row.get("awayteam", "N/A")
        league = row.get("league", "N/A")

        d = row.get("date")
        match_date = pd.to_datetime(d, errors="coerce").strftime("%d %b %Y") if d is not None and pd.notna(pd.to_datetime(d, errors="coerce")) else "N/A"

        ko = row.get("kickoff")
        kickoff = ko.strftime("%H:%M") if hasattr(ko, "strftime") else (str(ko) if ko else "N/A")

        xgh = f"{row.get('xgh'):.2f}" if row.get("xgh") is not None else "N/A"
        xga = f"{row.get('xga'):.2f}" if row.get("xga") is not None else "N/A"
        esoth = f"{row.get('esoth'):.1f}" if row.get("esoth") is not None else "N/A"
        esota = f"{row.get('esota'):.1f}" if row.get("esota") is not None else "N/A"

        win_h = f"{row.get('homewin'):.1f}%" if row.get("homewin") is not None else "N/A"
        drawp = f"{row.get('drawwin'):.1f}%" if row.get("drawwin") is not None else "N/A"
        win_a = f"{row.get('awaywin'):.1f}%" if row.get("awaywin") is not None else "N/A"

        hcosod = f"{row.get('hcosod'):.1f}" if row.get("hcosod") is not None else "0"
        acosod = f"{row.get('acosod'):.1f}" if row.get("acosod") is not None else "0"
        sodd = f"{row.get('sodd'):.1f}" if row.get("sodd") is not None else "0"

        home_odds = row.get("homeodds")
        draw_odds = row.get("drawodds")
        away_odds = row.get("awayodds")

        home_odds_s = f"{float(home_odds):.2f}" if home_odds is not None else "N/A"
        draw_odds_s = f"{float(draw_odds):.2f}" if draw_odds is not None else "N/A"
        away_odds_s = f"{float(away_odds):.2f}" if away_odds is not None else "N/A"

        home_team_stats = get_team_stats(home_team)
        away_team_stats = get_team_stats(away_team)

        home_form_val = home_team_stats.get("Form")
        away_form_val = away_team_stats.get("Form")
        home_form = f"{home_form_val:.2f}" if isinstance(home_form_val, (int, float)) and home_form_val is not None else "N/A"
        away_form = f"{away_form_val:.2f}" if isinstance(away_form_val, (int, float)) and away_form_val is not None else "N/A"

        league_name_for_baselines = home_team_stats.get("League") or league

        st.markdown(
            f"""
            <div style='background-color:#2B2B2B;padding:15px;border-radius:8px;margin-bottom:10px;color:#E0E0E0;font-size:15px;'>
                <b>{home_team}</b> vs <b>{away_team}</b><br>
                <span style='font-size:13px;'>League: {league} | Date: {match_date} | Kickoff: {kickoff}</span><br><br>
                <b>xG:</b> {xgh} - {xga} &nbsp;&nbsp;&nbsp;
                <b>Expected Shots on Target:</b> {esoth} - {esota}<br>
                <b>Win Probabilities:</b> {win_h} (Home), {drawp} (Draw), {win_a} (Away)<br>
                <b>Form Score:</b> {home_team} = {home_form} &nbsp;&nbsp;&nbsp; {away_team} = {away_form}<br>
                <b>Common Opponent Shots On Diff:</b><br>
                {home_team}: {hcosod} &nbsp;&nbsp;&nbsp; {away_team}: {acosod}<br>
                <b>SOD Difference:</b> {sodd}<br><br>
                <b>Odds:</b> Home = {home_odds_s}, Draw = {draw_odds_s}, Away = {away_odds_s}
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns([1, 2])
        with col1:
            display_team_stats_section(home_team_stats, away_team_stats)

        with col2:
            display_mutual_opponents_section(home_team, away_team, league_name_for_baselines)

        col3, col4 = st.columns([1, 1])
        with col3:
            display_recent_form_section(home_team, away_team, league_name_for_baselines)

        with col4:
            st.subheader("Head to Head")
            match_row = get_latest_head_to_head_row(home_team, away_team)
            if match_row:
                latest_date = pd.to_datetime(match_row.get("date"), errors="coerce").strftime("%d %b %Y") if match_row.get("date") is not None and pd.notna(pd.to_datetime(match_row.get("date"), errors="coerce")) else "N/A"
                st.markdown(
                    f"<div style='text-align:center;font-size:14px;color:#E0E0E0;'>Last played on <b>{latest_date}</b></div>",
                    unsafe_allow_html=True,
                )
                st.pyplot(generate_head_to_head_bar_chart(match_row))
            else:
                st.markdown(
                    "<div style='text-align:center;font-size:12px;color:#888;'>No recent head-to-head match found.</div>",
                    unsafe_allow_html=True,
                )
    else:
        st.warning("EventID not found in Supabase fixtures (or invalid EventID).")
else:
    st.info("Please select a fixture or enter an EventID.")
