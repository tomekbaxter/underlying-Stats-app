import datetime as dt
from zoneinfo import ZoneInfo
from urllib.parse import quote_plus

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from streamlit.errors import StreamlitSecretNotFoundError

# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(page_title="Underlying Stats", layout="wide")

st.markdown(
    """
    <style>
    header[data-testid="stHeader"] { display: none; }

    html, body, .stApp {
        background-color: #0e1117;
        color: #e6e6e6;
    }

    .block-container {
        padding-top: 1.2rem !important;
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

    .stTextInput input,
    .stSelectbox div[data-baseweb="select"] > div {
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

    h1, h2, h3, h4 {
        color: #e6e6e6 !important;
    }

    table {
        width: 100%;
        border-collapse: collapse;
        table-layout: fixed;
    }

    th {
        background-color: #2B2B2B !important;
        color: #E0E0E0 !important;
        font-weight: bold !important;
        text-align: center !important;
        border: 1px solid #444444 !important;
        padding: 6px !important;
        font-size: 12px !important;
    }

    td {
        text-align: center !important;
        border: 1px solid #444444 !important;
        padding: 6px !important;
        font-size: 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

TZ = ZoneInfo("Europe/London")

# ============================================================
# SUPABASE / POSTGRES CONNECTION
# ============================================================

SECRETS_TXT_PATH = r"C:\Users\TomekBaxter\Dropbox\football_app\Secrets.txt"


def _read_kv_file(path: str) -> dict:
    out: dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    except FileNotFoundError:
        return {}
    return out


def _build_db_url_from_parts(host: str, port: str, db: str, user: str, pw: str) -> str:
    return f"postgresql+psycopg2://{quote_plus(user)}:{quote_plus(pw)}@{host}:{port}/{db}"


def _build_db_url_from_txt(path: str) -> str:
    kv = _read_kv_file(path)

    host = kv.get("SUPABASE_HOST", "").strip()
    port = kv.get("SUPABASE_PORT", "").strip()
    db = kv.get("SUPABASE_DB", "").strip()
    user = kv.get("SUPABASE_USER", "").strip()
    pw = kv.get("SUPABASE_PASS", "").strip()

    missing = [
        k for k, val in {
            "SUPABASE_HOST": host,
            "SUPABASE_PORT": port,
            "SUPABASE_DB": db,
            "SUPABASE_USER": user,
            "SUPABASE_PASS": pw,
        }.items()
        if not val
    ]

    if missing:
        st.error(
            "Database secrets are missing.\n\n"
            f"Missing values: {', '.join(missing)}\n\n"
            "On Streamlit Cloud, set SUPABASE_DB_URL in app Secrets."
        )
        st.stop()

    return _build_db_url_from_parts(host, port, db, user, pw)


def _get_db_url() -> str:
    try:
        db_url = str(st.secrets.get("SUPABASE_DB_URL", "")).strip()
        if db_url:
            return db_url

        needed = ["SUPABASE_HOST", "SUPABASE_PORT", "SUPABASE_DB", "SUPABASE_USER", "SUPABASE_PASS"]
        if all(k in st.secrets and str(st.secrets[k]).strip() for k in needed):
            return _build_db_url_from_parts(
                str(st.secrets["SUPABASE_HOST"]).strip(),
                str(st.secrets["SUPABASE_PORT"]).strip(),
                str(st.secrets["SUPABASE_DB"]).strip(),
                str(st.secrets["SUPABASE_USER"]).strip(),
                str(st.secrets["SUPABASE_PASS"]).strip(),
            )
    except StreamlitSecretNotFoundError:
        pass
    except Exception:
        pass

    return _build_db_url_from_txt(SECRETS_TXT_PATH)


@st.cache_resource
def get_engine():
    return create_engine(
        _get_db_url(),
        pool_pre_ping=True,
        pool_size=1,
        max_overflow=0,
        pool_timeout=15,
        pool_recycle=300,
        connect_args={"sslmode": "require"},
        future=True,
    )


ENGINE = get_engine()


def db_healthcheck() -> None:
    try:
        with ENGINE.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        st.stop()


db_healthcheck()


def read_sql_df(sql: str, params: dict | None = None) -> pd.DataFrame:
    with ENGINE.begin() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def read_sql_one(sql: str, params: dict | None = None) -> dict | None:
    df = read_sql_df(sql, params=params)
    if df.empty:
        return None
    return df.iloc[0].to_dict()


# ============================================================
# SAFE HELPERS
# ============================================================

def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_int_display(value) -> str:
    try:
        if value is None or pd.isna(value):
            return "0"
        return str(int(round(float(value))))
    except Exception:
        return "0"


def safe_float_display(value, decimals: int = 2, default: str = "N/A") -> str:
    try:
        if value is None or pd.isna(value):
            return default
        return f"{float(value):.{decimals}f}"
    except Exception:
        return default


def safe_percent_display(value, decimals: int = 1) -> str:
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{float(value):.{decimals}f}%"
    except Exception:
        return "N/A"


def render_styled_table(styler) -> None:
    html = styler.to_html()
    st.markdown(html, unsafe_allow_html=True)


def gradient_background(value, avg, positive=True):
    try:
        value = float(value)
        avg = float(avg)

        if avg == 0:
            return "background-color: #ffffff; color: black;"

        diff = value - avg
        norm_diff = diff / avg

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
        return "background-color: #ffffff; color: black;"


# ============================================================
# DATA HELPERS
# ============================================================

@st.cache_data(ttl=300)
def load_fixtures_with_odds() -> pd.DataFrame:
    sql = """
        SELECT
            eventid,
            hometeam,
            awayteam,
            league,
            date,
            kickoff,
            home AS homeodds,
            draw AS drawodds,
            away AS awayodds
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
    df["fixturename"] = df["hometeam"].fillna("N/A").astype(str) + " vs " + df["awayteam"].fillna("N/A").astype(str)

    return df


@st.cache_data(ttl=120)
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


@st.cache_data(ttl=300)
def get_team_stats(team_name: str) -> dict:
    sql = """
        SELECT
            "Games" AS games,
            "AGF" AS agf,
            "AGA" AS aga,
            "ASOF" AS asof,
            "ASOA" AS asoa,
            "ATT" AS att,
            "DEF" AS def,
            "Form" AS form,
            "League" AS league
        FROM list_of_teams
        WHERE "TeamName" = :teamname
        LIMIT 1
    """
    row = read_sql_one(sql, params={"teamname": team_name})

    if not row:
        return {
            "Games": 0,
            "AGF": 0,
            "AGA": 0,
            "ASOF": 0,
            "ASOA": 0,
            "ATT": 0,
            "DEF": 0,
            "Form": None,
            "League": None,
        }

    return {
        "Games": safe_float(row.get("games"), 0),
        "AGF": safe_float(row.get("agf"), 0),
        "AGA": safe_float(row.get("aga"), 0),
        "ASOF": safe_float(row.get("asof"), 0),
        "ASOA": safe_float(row.get("asoa"), 0),
        "ATT": safe_float(row.get("att"), 0),
        "DEF": safe_float(row.get("def"), 0),
        "Form": row.get("form"),
        "League": row.get("league"),
    }


@st.cache_data(ttl=300)
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
        return {"ATT": np.nan, "DEF": np.nan}

    return {
        "ATT": safe_float(row.get("att"), np.nan),
        "DEF": safe_float(row.get("def"), np.nan),
    }


@st.cache_data(ttl=300)
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
    return read_sql_one(
        sql,
        params={
            "h1": home_team,
            "a1": away_team,
            "h2": away_team,
            "a2": home_team,
        },
    )


@st.cache_data(ttl=300)
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


@st.cache_data(ttl=300)
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
        return {"Date": "N/A", "SOF": np.nan, "SOA": np.nan}

    d = pd.to_datetime(row.get("date"), errors="coerce")
    date_str = d.strftime("%Y-%m-%d") if pd.notna(d) else "N/A"

    home_sof = row.get("homeshotson")
    away_sof = row.get("awayshotson")

    if row.get("hometeam") == team_name:
        sof, soa = home_sof, away_sof
    else:
        sof, soa = away_sof, home_sof

    return {
        "Date": date_str,
        "SOF": safe_float(sof, np.nan),
        "SOA": safe_float(soa, np.nan),
    }


@st.cache_data(ttl=300)
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
        errors="coerce",
    ).dropna()

    tdf = read_sql_df(
        """
        SELECT
            "ATT" AS att,
            "DEF" AS def
        FROM list_of_teams
        WHERE "League" = :league
        """,
        params={"league": league_name},
    )

    att_values = pd.to_numeric(tdf["att"], errors="coerce").replace(0, np.nan).dropna()
    def_values = pd.to_numeric(tdf["def"], errors="coerce").replace(0, np.nan).dropna()

    return {
        "SOF": float(sof_values.mean()) if not sof_values.empty else 1.0,
        "SOA": float(sof_values.mean()) if not sof_values.empty else 1.0,
        "ATT": float(att_values.mean()) if not att_values.empty else 1.0,
        "DEF": float(def_values.mean()) if not def_values.empty else 1.0,
    }


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
        key=lambda x: dt.datetime.strptime(x["Home Date"], "%Y-%m-%d")
        if x["Home Date"] != "N/A"
        else dt.datetime.min,
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

    for c in ["ATT", "DEF", "Home SOF", "Home SOA", "Away SOF", "Away SOA"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    sof_avg = league_data["SOF"] if league_data else 1.0
    soa_avg = league_data["SOA"] if league_data else 1.0
    att_avg = league_data["ATT"] if league_data else 1.0
    def_avg = league_data["DEF"] if league_data else 1.0

    styled = (
        df.style
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
        .map(lambda x: gradient_background(x, att_avg, positive=True), subset=["ATT"])
        .map(lambda x: gradient_background(x, def_avg, positive=False), subset=["DEF"])
        .map(lambda x: gradient_background(x, sof_avg, positive=True), subset=["Home SOF", "Away SOF"])
        .map(lambda x: gradient_background(x, soa_avg, positive=False), subset=["Home SOA", "Away SOA"])
        .hide(axis="index")
    )

    render_styled_table(styled)


@st.cache_data(ttl=300)
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
    sof = pd.to_numeric(pd.concat([df["homeshotson"], df["awayshotson"]], ignore_index=True), errors="coerce").dropna()
    sf = pd.to_numeric(pd.concat([df["homeshots"], df["awayshots"]], ignore_index=True), errors="coerce").dropna()

    tdf = read_sql_df(
        """
        SELECT
            "ATT" AS att,
            "DEF" AS def
        FROM list_of_teams
        WHERE "League" = :league
        """,
        params={"league": league_name},
    )

    att = pd.to_numeric(tdf["att"], errors="coerce").replace(0, np.nan).dropna()
    deff = pd.to_numeric(tdf["def"], errors="coerce").replace(0, np.nan).dropna()

    return {
        "GF": float(gf.mean()) if not gf.empty else 1.0,
        "GA": float(gf.mean()) if not gf.empty else 1.0,
        "SOF": float(sof.mean()) if not sof.empty else 1.0,
        "SOA": float(sof.mean()) if not sof.empty else 1.0,
        "SF": float(sf.mean()) if not sf.empty else 1.0,
        "SA": float(sf.mean()) if not sf.empty else 1.0,
        "Opp ATT": float(att.mean()) if not att.empty else 1.0,
        "Opp DEF": float(deff.mean()) if not deff.empty else 1.0,
    }


@st.cache_data(ttl=300)
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

    return safe_float(row.get("att"), np.nan), safe_float(row.get("def"), np.nan)


@st.cache_data(ttl=300)
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

        d = pd.to_datetime(r["date"], errors="coerce")
        date_str = d.strftime("%d/%m/%Y") if pd.notna(d) else "N/A"

        data.append(
            {
                "Opponent": opponent,
                "Opp ATT": opp_att,
                "Opp DEF": opp_def,
                "Date": date_str,
                "GF": r["homegoals"] if is_home else r["awaygoals"],
                "SOF": r["homeshotson"] if is_home else r["awayshotson"],
                "SF": r["homeshots"] if is_home else r["awayshots"],
                "GA": r["awaygoals"] if is_home else r["homegoals"],
                "SOA": r["awayshotson"] if is_home else r["homeshotson"],
                "SA": r["awayshots"] if is_home else r["homeshots"],
            }
        )

    return data


def display_recent_form_section(home_team: str, away_team: str, league_name: str | None):
    st.subheader("Recent Form")

    league_baselines = get_league_recent_baselines(league_name) if league_name else None

    home_df = pd.DataFrame(get_recent_form(home_team))
    away_df = pd.DataFrame(get_recent_form(away_team))

    def apply_form_styling(df: pd.DataFrame):
        if df.empty:
            return None

        stat_cols = ["GF", "GA", "SOF", "SOA", "SF", "SA", "Opp ATT", "Opp DEF"]

        for c in stat_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        avg_values = league_baselines if league_baselines else {
            col: safe_float(df[col].mean(), 1.0) for col in stat_cols
        }

        styled = (
            df.style
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
            .map(lambda x: gradient_background(x, avg_values["Opp ATT"], positive=True), subset=["Opp ATT"])
            .map(lambda x: gradient_background(x, avg_values["Opp DEF"], positive=False), subset=["Opp DEF"])
            .map(lambda x: gradient_background(x, avg_values["GF"], positive=True), subset=["GF"])
            .map(lambda x: gradient_background(x, avg_values["GA"], positive=False), subset=["GA"])
            .map(lambda x: gradient_background(x, avg_values["SOF"], positive=True), subset=["SOF"])
            .map(lambda x: gradient_background(x, avg_values["SOA"], positive=False), subset=["SOA"])
            .map(lambda x: gradient_background(x, avg_values["SF"], positive=True), subset=["SF"])
            .map(lambda x: gradient_background(x, avg_values["SA"], positive=False), subset=["SA"])
            .hide(axis="index")
        )

        return styled

    st.markdown(f"### {home_team} Recent Form")
    if not home_df.empty:
        render_styled_table(apply_form_styling(home_df))
    else:
        st.write(f"No recent matches found for {home_team}.")

    st.markdown(f"### {away_team} Recent Form")
    if not away_df.empty:
        render_styled_table(apply_form_styling(away_df))
    else:
        st.write(f"No recent matches found for {away_team}.")


# ============================================================
# CHARTS
# ============================================================

def generate_team_stats_chart(home_stats, away_stats):
    stats = ["AGF", "AGA", "ASOF", "ASOA", "ATT", "DEF"]
    home_values = [safe_float(home_stats.get(stat), 0) for stat in stats]
    away_values = [safe_float(away_stats.get(stat), 0) for stat in stats]

    home_color = "#3498db"
    away_color = "#e74c3c"
    text_color = "#E0E0E0"
    bg_color = "#1E1E1E"

    fig, ax = plt.subplots(figsize=(7, 3.4), dpi=130)
    bar_height = 0.22
    y_positions = np.arange(len(stats))

    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    max_seen = max(home_values + away_values + [1])
    max_value = max(15, max_seen * 1.4)
    indent_offset = max_value * 0.035

    ax.barh(y_positions, [-x for x in home_values], height=bar_height, color=home_color, align="center")
    ax.barh(y_positions, away_values, height=bar_height, color=away_color, align="center")

    for i, value in enumerate(home_values):
        ax.text(
            -value - indent_offset,
            i,
            f"{value:.2f}",
            va="center",
            ha="right",
            fontsize=9,
            color=text_color,
            fontweight="bold",
        )

    for i, value in enumerate(away_values):
        ax.text(
            value + indent_offset,
            i,
            f"{value:.2f}",
            va="center",
            ha="left",
            fontsize=9,
            color=text_color,
            fontweight="bold",
        )

    for i, stat in enumerate(stats):
        ax.text(
            0,
            i,
            stat,
            va="center",
            ha="center",
            fontsize=10,
            color=text_color,
            fontweight="bold",
        )

    ax.set_yticks([])
    ax.set_xticks([])
    ax.axvline(0, color="#444444", lw=1)
    ax.set_xlim(-max_value, max_value)
    ax.invert_yaxis()
    ax.set_frame_on(False)

    plt.tight_layout()
    return fig


def display_team_stats_section(home_stats, away_stats):
    st.subheader("Team Stats")

    st.markdown(
        f"""
        <div style="display:flex;justify-content:center;align-items:center;padding-bottom:5px;">
            <span style="flex:1;text-align:right;font-size:20px;font-weight:bold;color:#3498db;">
                {safe_int_display(home_stats.get('Games'))}
            </span>
            <span style="flex:1;text-align:center;font-size:16px;color:#E0E0E0;">Games</span>
            <span style="flex:1;text-align:left;font-size:20px;font-weight:bold;color:#e74c3c;">
                {safe_int_display(away_stats.get('Games'))}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig = generate_team_stats_chart(home_stats, away_stats)
    st.pyplot(fig, clear_figure=True, use_container_width=True)
    plt.close(fig)


def generate_head_to_head_bar_chart(match_row: dict):
    home_team = match_row.get("hometeam", "")
    away_team = match_row.get("awayteam", "")

    stats = [
        ("Goals", safe_float(match_row.get("homegoals"), 0), safe_float(match_row.get("awaygoals"), 0), 10),
        ("Shots", safe_float(match_row.get("homeshots"), 0), safe_float(match_row.get("awayshots"), 0), 30),
        ("Shots on Target", safe_float(match_row.get("homeshotson"), 0), safe_float(match_row.get("awayshotson"), 0), 20),
        ("Attacks", safe_float(match_row.get("homeattacks"), 0), safe_float(match_row.get("awayattacks"), 0), 100),
        ("Dangerous Attacks", safe_float(match_row.get("homedangerousattacks"), 0), safe_float(match_row.get("awaydangerousattacks"), 0), 80),
    ]

    home_color = "#3498db"
    away_color = "#e74c3c"
    text_color = "#E0E0E0"
    bg_color = "#1E1E1E"

    fig, ax = plt.subplots(figsize=(7, 3.4), dpi=130)
    bar_height = 0.18

    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    max_value = max([s[3] for s in stats] + [1])
    max_seen = max([max(s[1], s[2]) for s in stats] + [1])
    max_value = max(max_value, max_seen * 1.25)

    indent_offset = max_value * 0.025
    label_offset = 0.18

    ax.text(
        -max_value,
        -1.25,
        str(home_team),
        fontsize=10,
        color=home_color,
        fontweight="bold",
        ha="left",
    )
    ax.text(
        max_value,
        -1.25,
        str(away_team),
        fontsize=10,
        color=away_color,
        fontweight="bold",
        ha="right",
    )

    for i, (label, home_val, away_val, _) in enumerate(stats):
        ax.barh(i, -home_val, height=bar_height, color=home_color, align="center")
        ax.barh(i, away_val, height=bar_height, color=away_color, align="center")

        ax.text(
            -home_val - indent_offset,
            i,
            f"{int(round(home_val))}",
            va="center",
            ha="right",
            fontsize=10,
            color=text_color,
            fontweight="bold",
        )

        ax.text(
            away_val + indent_offset,
            i,
            f"{int(round(away_val))}",
            va="center",
            ha="left",
            fontsize=10,
            color=text_color,
            fontweight="bold",
        )

        ax.text(
            0,
            i - label_offset,
            label,
            va="bottom",
            ha="center",
            fontsize=9,
            color=text_color,
            fontweight="bold",
        )

    ax.axvline(0, color="#444444", lw=1)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(-max_value, max_value)
    ax.invert_yaxis()
    ax.set_frame_on(False)

    plt.tight_layout()
    return fig


# ============================================================
# UI
# ============================================================

fixtures_df = load_fixtures_with_odds()

st.markdown(
    """
    <div style="display:flex;justify-content:space-between;align-items:center;padding-bottom:10px;">
        <h1 style="margin:0;padding:0;color:#E0E0E0;">Underlying Stats</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("Search Fixture")
st.markdown(
    "<small style='color:#AAAAAA;'>Select Date -> League -> Fixture or enter EventID</small>",
    unsafe_allow_html=True,
)

for k in ("sel_date", "sel_league", "sel_fixture", "sel_eventid"):
    if k not in st.session_state:
        st.session_state[k] = None

if "eventid_input" not in st.session_state:
    st.session_state.eventid_input = ""


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
        sub = fixtures_df[
            (fixtures_df["datestr"] == st.session_state.sel_date)
            & (fixtures_df["league"] == st.session_state.sel_league)
            & (fixtures_df["fixturename"] == st.session_state.sel_fixture)
        ]

        if not sub.empty:
            st.session_state.sel_eventid = str(int(sub.iloc[0]["eventid"]))


col_eventid, col_date, col_league, col_fixture = st.columns([1, 1, 1, 3])

col_eventid.text_input(
    "EventID",
    key="eventid_input",
    max_chars=12,
    placeholder="Enter EventID",
    on_change=_on_eventid_change,
)

date_options = sorted(fixtures_df["datestr"].dropna().unique())
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
# DISPLAY FIXTURE DETAILS
# ============================================================

event_id_to_use = st.session_state.sel_eventid

if event_id_to_use:
    row = fetch_fixture_row(event_id_to_use)

    if row:
        home_team = row.get("hometeam", "N/A")
        away_team = row.get("awayteam", "N/A")
        league = row.get("league", "N/A")

        d = pd.to_datetime(row.get("date"), errors="coerce")
        match_date = d.strftime("%d %b %Y") if pd.notna(d) else "N/A"

        ko = row.get("kickoff")
        kickoff = ko.strftime("%H:%M") if hasattr(ko, "strftime") else (str(ko) if ko else "N/A")

        xgh = safe_float_display(row.get("xgh"), 2)
        xga = safe_float_display(row.get("xga"), 2)
        esoth = safe_float_display(row.get("esoth"), 1)
        esota = safe_float_display(row.get("esota"), 1)

        win_h = safe_percent_display(row.get("homewin"), 1)
        drawp = safe_percent_display(row.get("drawwin"), 1)
        win_a = safe_percent_display(row.get("awaywin"), 1)

        hcosod = safe_float_display(row.get("hcosod"), 1, default="0.0")
        acosod = safe_float_display(row.get("acosod"), 1, default="0.0")
        sodd = safe_float_display(row.get("sodd"), 1, default="0.0")

        home_odds_s = safe_float_display(row.get("homeodds"), 2)
        draw_odds_s = safe_float_display(row.get("drawodds"), 2)
        away_odds_s = safe_float_display(row.get("awayodds"), 2)

        home_team_stats = get_team_stats(home_team)
        away_team_stats = get_team_stats(away_team)

        home_form = safe_float_display(home_team_stats.get("Form"), 2)
        away_form = safe_float_display(away_team_stats.get("Form"), 2)

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
                latest_date_dt = pd.to_datetime(match_row.get("date"), errors="coerce")
                latest_date = latest_date_dt.strftime("%d %b %Y") if pd.notna(latest_date_dt) else "N/A"

                st.markdown(
                    f"<div style='text-align:center;font-size:14px;color:#E0E0E0;'>Last played on <b>{latest_date}</b></div>",
                    unsafe_allow_html=True,
                )

                fig = generate_head_to_head_bar_chart(match_row)
                st.pyplot(fig, clear_figure=True, use_container_width=True)
                plt.close(fig)
            else:
                st.markdown(
                    "<div style='text-align:center;font-size:12px;color:#888;'>No recent head-to-head match found.</div>",
                    unsafe_allow_html=True,
                )
    else:
        st.warning("EventID not found in Supabase fixtures, or invalid EventID.")
else:
    st.info("Please select a fixture or enter an EventID.")
