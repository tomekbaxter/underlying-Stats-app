import datetime as dt
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from streamlit.errors import StreamlitSecretNotFoundError

# ============================================================
# DESIGN TOKENS  (shared with the Underlying Stats dashboard)
# ============================================================

PAGE_BG = "#060A12"
SURFACE_1 = "#0E1626"
SURFACE_2 = "#172236"
SURFACE_3 = "#1F2C44"
BORDER = "#1F2B40"

TEXT_1 = "#E8ECF4"
TEXT_2 = "#93A0B8"
TEXT_3 = "#5F6E88"

HOME = "#4C8DFF"
AWAY = "#B8894A"

# Reserved exclusively for value-vs-baseline scales. Nothing structural
# uses these, so a red cell always means "worse", never "away team".
GOOD_HEX = "#10B981"
BAD_HEX = "#EF4444"
GOOD = np.array([16, 185, 129])
BAD = np.array([239, 68, 68])
CELL_BASE = np.array([23, 34, 54])  # matches SURFACE_2

TZ = ZoneInfo("Europe/London")
FIXTURES_TTL = 180
STANDINGS_TTL = 900
MATCHSTATS_TTL = 900

st.set_page_config(
    page_title="Pre-Game Finder",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    f"""
    <style>
    header[data-testid="stHeader"] {{ display: none; }}
    html, body, .stApp {{ background-color: {PAGE_BG}; color: {TEXT_1}; }}

    .block-container {{
        padding: 1rem 0.9rem 3rem 0.9rem !important;
        max-width: 1600px;
    }}
    h1, h2, h3 {{ color: {TEXT_1} !important; font-weight: 500 !important; }}

    div[data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 10px;
        overflow: hidden;
    }}

    div[role="radiogroup"] {{ gap: 4px; flex-wrap: wrap; }}
    div[role="radiogroup"] label {{
        background: {SURFACE_1};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 6px 12px;
        margin: 0 !important;
    }}
    div[role="radiogroup"] label:hover {{ border-color: {HOME}; }}

    .stTextInput input,
    .stSelectbox div[data-baseweb="select"] > div {{
        background-color: {SURFACE_2} !important;
        color: {TEXT_1} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 8px !important;
    }}

    div.stButton > button, div[data-testid="stDownloadButton"] > button {{
        border-radius: 8px;
        background-color: {SURFACE_2};
        color: {TEXT_1};
        border: 1px solid {BORDER};
    }}
    div.stButton > button:hover,
    div[data-testid="stDownloadButton"] > button:hover {{
        background-color: {SURFACE_3};
        border-color: {HOME};
        color: {HOME};
    }}

    @media (max-width: 768px) {{
        .block-container {{ padding: 0.6rem 0.4rem 2rem 0.4rem !important; }}
        div[role="radiogroup"] label {{ padding: 5px 9px; font-size: 0.8rem; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONNECTION
# ============================================================

def _get_db_url() -> str:
    try:
        db_url = st.secrets.get("SUPABASE_DB_URL", "")
    except StreamlitSecretNotFoundError:
        st.error("Missing Streamlit secrets. Set SUPABASE_DB_URL.")
        st.stop()

    if not isinstance(db_url, str) or not db_url.strip():
        st.error("SUPABASE_DB_URL is missing or empty.")
        st.stop()
    return db_url.strip()


@st.cache_resource
def get_engine():
    return create_engine(
        _get_db_url(),
        pool_pre_ping=True,
        # The engine is cached once per container and shared by every visitor.
        # The original allowed a single connection with no overflow, so two
        # simultaneous cold loads queued behind each other.
        pool_size=3,
        max_overflow=4,
        pool_timeout=10,
        pool_recycle=300,
        connect_args={"sslmode": "require", "connect_timeout": 10},
        future=True,
    )


ENGINE = get_engine()


@st.cache_data(ttl=300)
def _healthcheck() -> bool:
    """
    Cached so it runs at most once every five minutes rather than on every
    single interaction. The original ran SELECT 1 on every rerun, against a
    pool that only had one connection.
    """
    try:
        with ENGINE.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


if not _healthcheck():
    st.error(
        "Couldn't reach the fixtures database. This is usually temporary — "
        "try refreshing in a moment."
    )
    st.stop()


# ============================================================
# DATA
# ============================================================

@st.cache_data(ttl=60)
def get_last_refresh() -> str:
    try:
        df = pd.read_sql(
            text('SELECT MAX(last_synced_at) AS last_synced_at FROM fixtures'),
            ENGINE,
        )
    except Exception:
        return "Unavailable"

    if df.empty or pd.isna(df.loc[0, "last_synced_at"]):
        return "Unknown"

    ts = pd.to_datetime(df.loc[0, "last_synced_at"], errors="coerce")
    if pd.isna(ts):
        return "Unknown"
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(TZ).strftime("%d/%m/%Y %H:%M")


@st.cache_data(ttl=FIXTURES_TTL)
def load_fixtures() -> pd.DataFrame:
    df = pd.read_sql(
        text(
            """
            SELECT eventid, hometeam, awayteam, league, date, kickoff,
                   home, draw, away, comopp, sodd,
                   xgh, xga, esoth, esota, hcosod, acosod,
                   homewin, drawwin, awaywin, score, value,
                   "xconvh", "xconva"
            FROM fixtures
            WHERE date >= CURRENT_DATE
            """
        ),
        ENGINE,
    )

    df = df.rename(columns={
        "eventid": "EventID", "hometeam": "HomeTeam", "awayteam": "AwayTeam",
        "league": "League", "date": "Date", "kickoff": "Kickoff",
        "home": "Home", "draw": "Draw", "away": "Away",
        "comopp": "ComOpp", "sodd": "SODD",
        "xgh": "XGH", "xga": "XGA", "esoth": "ESOTH", "esota": "ESOTA",
        "hcosod": "HCOSOD", "acosod": "ACOSOD",
        "homewin": "HomeWin%", "drawwin": "Draw%", "awaywin": "AwayWin%",
        "score": "Score", "value": "Value",
        "xconvh": "XConvH", "xconva": "XConvA",
    })

    df["KickoffDT"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Kickoff"].astype(str),
        errors="coerce",
    )
    return df


@st.cache_data(ttl=STANDINGS_TTL)
def load_standings() -> pd.DataFrame:
    try:
        teams = pd.read_sql(
            text(
                """
                SELECT "League", "TeamName", "StandingPosition",
                       "StandingPPG", "StandingGames"
                FROM list_of_teams
                WHERE "StandingGames" > 0
                  AND "StandingPosition" > 0
                  AND "StandingPPG" IS NOT NULL
                """
            ),
            ENGINE,
        )
    except Exception:
        return pd.DataFrame()

    if teams.empty:
        return teams

    for col in ("League", "TeamName"):
        teams[col] = teams[col].fillna("").astype(str).str.strip()
    for col in ("StandingPosition", "StandingPPG", "StandingGames"):
        teams[col] = pd.to_numeric(teams[col], errors="coerce")

    return teams.dropna(subset=["League", "TeamName", "StandingPosition",
                                "StandingPPG", "StandingGames"])


@st.cache_data(ttl=MATCHSTATS_TTL)
def load_recent_matchstats(days: int = 90) -> pd.DataFrame:
    """
    Cached. The original queried 90 days of matchstats inside the H2H filter
    on every rerun, so each click re-ran the whole thing.
    """
    cutoff = (datetime.now(TZ).date() - dt.timedelta(days=days))
    try:
        h2h = pd.read_sql(
            text(
                """
                SELECT "HomeTeam", "AwayTeam", "Date",
                       "HomeGoals", "AwayGoals",
                       "HomeShots", "AwayShots",
                       "HomeShotsOn", "AwayShotsOn"
                FROM matchstats
                WHERE "Date" >= :cutoff
                """
            ),
            ENGINE,
            params={"cutoff": cutoff},
        )
    except Exception:
        return pd.DataFrame()

    if h2h.empty:
        return h2h

    h2h["Date"] = pd.to_datetime(h2h["Date"], errors="coerce")
    stat_cols = ["HomeGoals", "AwayGoals", "HomeShots", "AwayShots",
                 "HomeShotsOn", "AwayShotsOn"]
    for c in stat_cols:
        h2h[c] = pd.to_numeric(h2h[c], errors="coerce")

    h2h = h2h.dropna(subset=["Date"] + stat_cols)
    if h2h.empty:
        return h2h

    # Sanity filters: shots must exceed goals, and enough shots to be real.
    total_goals = h2h["HomeGoals"] + h2h["AwayGoals"]
    total_shots = h2h["HomeShots"] + h2h["AwayShots"]
    total_sot = h2h["HomeShotsOn"] + h2h["AwayShotsOn"]
    h2h = h2h[(total_shots > total_goals)
              & (total_sot >= total_goals)
              & (total_shots >= 6)].copy()
    if h2h.empty:
        return h2h

    # Vectorised pair key — the original used df.apply row-wise, twice.
    a = h2h["HomeTeam"].fillna("").astype(str).str.strip()
    b = h2h["AwayTeam"].fillna("").astype(str).str.strip()
    h2h["PairKey"] = np.where(a <= b, a + "||" + b, b + "||" + a)

    return (h2h.sort_values("Date", ascending=False)
               .drop_duplicates("PairKey", keep="first")
               .copy())


def add_standings(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Home St.Pos", "Away St.Pos", "Home St.PPG", "Away St.PPG",
            "Home St.Games", "Away St.Games"]
    if df.empty:
        for c in cols:
            df[c] = pd.NA
        return df

    df = df.copy()
    teams = load_standings()
    if teams.empty:
        for c in cols:
            df[c] = pd.NA
        return df

    for col in ("League", "HomeTeam", "AwayTeam"):
        df[col] = df[col].fillna("").astype(str).str.strip()

    for side in ("Home", "Away"):
        lookup = teams.rename(columns={
            "TeamName": f"{side}Team",
            "StandingPosition": f"{side} St.Pos",
            "StandingPPG": f"{side} St.PPG",
            "StandingGames": f"{side} St.Games",
        })
        df = df.drop(columns=[f"{side} St.Pos", f"{side} St.PPG",
                              f"{side} St.Games"], errors="ignore")
        df = df.merge(
            lookup[["League", f"{side}Team", f"{side} St.Pos",
                    f"{side} St.PPG", f"{side} St.Games"]],
            on=["League", f"{side}Team"], how="left",
        )

    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Home St.PPG"] = df["Home St.PPG"].round(2)
    df["Away St.PPG"] = df["Away St.PPG"].round(2)
    return df


def apply_global_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    now = datetime.now(TZ).replace(tzinfo=None)
    df = df.copy()

    for c in ("Home", "Draw", "Away"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Home", "Draw", "Away"])
    df = df[(df["Home"] > 0) & (df["Draw"] > 0) & (df["Away"] > 0)]
    df = df[df["KickoffDT"].notna()]
    df = df[df["KickoffDT"] > now]
    return df.sort_values("KickoffDT")


# ============================================================
# SHARED FILTER HELPERS
# ============================================================

def tag_side(df: pd.DataFrame, is_home: pd.Series) -> pd.DataFrame:
    """
    Every filter identifies an advantaged side internally, but the original
    never surfaced it — a visitor saw a qualifying fixture without knowing
    which team the signal favoured. This writes it into shared columns.
    """
    df = df.copy()
    df["Side"] = np.where(is_home, "Home", "Away")
    df["Pick"] = np.where(is_home, df["HomeTeam"], df["AwayTeam"])
    df["AdvOdds"] = np.where(is_home, df["Home"], df["Away"]).astype(float)
    df["ImpliedProb"] = (1.0 / df["AdvOdds"] * 100).round(1)
    return df


def sliding_required_odds(magnitude: pd.Series, m0: float, m1: float,
                          odds0: float, odds1: float) -> pd.Series:
    """Interpolate a minimum acceptable price from the signal strength."""
    req = odds0 + (odds1 - odds0) * (magnitude - m0) / (m1 - m0)
    return req.clip(lower=odds1)


# ============================================================
# FILTERS
# ============================================================

def filter_all(df: pd.DataFrame) -> pd.DataFrame:
    return df


def filter_sodd(df: pd.DataFrame) -> pd.DataFrame:
    S0, S1, ODDS0, ODDS1, PMAX = 7.0, 10.0, 1.60, 1.40, 0.80
    if df.empty:
        return df

    df = df.copy()
    for c in ("SODD", "Home", "Away"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["SODD", "Home", "Away"])
    df = df[df["SODD"].abs() >= S0]
    df = df[df["SODD"] != 0]
    if df.empty:
        return df

    df = tag_side(df, df["SODD"] > 0)
    df["Signal"] = df["SODD"].abs().round(1)
    df["MinOdds"] = sliding_required_odds(df["Signal"], S0, S1,
                                          ODDS0, ODDS1).round(2)

    df = df[(df["AdvOdds"] >= df["MinOdds"])
            & (1.0 / df["AdvOdds"] <= PMAX)]
    return df


def filter_sodd_cosod(df: pd.DataFrame) -> pd.DataFrame:
    S0, S1, ODDS0, ODDS1, PMAX = 3.0, 7.0, 2.20, 1.40, 0.80
    if df.empty:
        return df

    df = df.copy()
    required = ["SODD", "Home", "Away", "HCOSOD", "ACOSOD"]
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required)
    df = df[(df["SODD"].abs() >= S0) & (df["SODD"] != 0)]
    if df.empty:
        return df

    is_home = df["SODD"] > 0
    cosod_adv = np.where(is_home, df["HCOSOD"], df["ACOSOD"])
    cosod_weak = np.where(is_home, df["ACOSOD"], df["HCOSOD"])

    df = df[(cosod_adv > 1) & (cosod_weak < -1)]
    if df.empty:
        return df

    df = tag_side(df, df["SODD"] > 0)
    df["Signal"] = df["SODD"].abs().round(1)
    df["MinOdds"] = sliding_required_odds(df["Signal"], S0, S1,
                                          ODDS0, ODDS1).round(2)
    df["COSOD Adv"] = np.where(df["Side"] == "Home",
                               df["HCOSOD"], df["ACOSOD"]).round(1)
    df["COSOD Opp"] = np.where(df["Side"] == "Home",
                               df["ACOSOD"], df["HCOSOD"]).round(1)

    df = df[(df["AdvOdds"] >= df["MinOdds"])
            & (1.0 / df["AdvOdds"] <= PMAX)]
    return df


def filter_xg_xsot(df: pd.DataFrame) -> pd.DataFrame:
    W_ESOT, W_XG = 1.0, 0.8
    D0, D1, ODDS0, ODDS1, PMAX = 3.0, 5.0, 2.40, 1.40, 0.60
    if df.empty:
        return df

    df = df.copy()
    required = ["XGH", "XGA", "ESOTH", "ESOTA", "Home", "Away"]
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required)
    if df.empty:
        return df

    esot_gap = df["ESOTH"] - df["ESOTA"]
    xg_gap = df["XGH"] - df["XGA"]
    dom = (W_ESOT * esot_gap) + (W_XG * xg_gap)

    # Both signals must agree with the dominance direction.
    agree = (((dom > 0) & (esot_gap > 0) & (xg_gap > 0))
             | ((dom < 0) & (esot_gap < 0) & (xg_gap < 0)))
    keep = (dom.abs() >= D0) & agree

    df = df[keep].copy()
    if df.empty:
        return df

    esot_gap, xg_gap = esot_gap[keep], xg_gap[keep]
    dom = dom[keep]

    df = tag_side(df, dom > 0)
    df["ESOT Gap"] = esot_gap.round(2)
    df["xG Gap"] = xg_gap.round(2)
    df["Signal"] = dom.abs().round(2)
    df["MinOdds"] = sliding_required_odds(df["Signal"], D0, D1,
                                          ODDS0, ODDS1).round(2)

    df = df[(df["AdvOdds"] >= df["MinOdds"])
            & (1.0 / df["AdvOdds"] <= PMAX)]
    return df


def filter_xwin_percent(df: pd.DataFrame) -> pd.DataFrame:
    MIN_ODDS, MIN_ABS_EDGE, MIN_REL_EDGE = 1.60, 0.07, 0.75
    if df.empty:
        return df

    df = df.copy()
    required = ["Home", "Draw", "Away", "HomeWin%", "AwayWin%", "Draw%"]
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required)
    df = df[(df["HomeWin%"] > 0) & (df["AwayWin%"] > 0)]
    if df.empty:
        return df

    overround = 1 / df["Home"] + 1 / df["Draw"] + 1 / df["Away"]
    p_home_mkt = (1 / df["Home"]) / overround
    p_away_mkt = (1 / df["Away"]) / overround

    home_abs = df["HomeWin%"] / 100 - p_home_mkt
    away_abs = df["AwayWin%"] / 100 - p_away_mkt

    home_ok = ((df["Home"] >= MIN_ODDS) & (home_abs >= MIN_ABS_EDGE)
               & (home_abs / p_home_mkt >= MIN_REL_EDGE))
    away_ok = ((df["Away"] >= MIN_ODDS) & (away_abs >= MIN_ABS_EDGE)
               & (away_abs / p_away_mkt >= MIN_REL_EDGE))

    keep = home_ok | away_ok
    df = df[keep].copy()
    if df.empty:
        return df

    # Where both sides qualify, take the larger edge.
    prefer_home = (home_ok[keep]
                   & (~away_ok[keep] | (home_abs[keep] >= away_abs[keep])))
    df = tag_side(df, prefer_home)

    df["Model %"] = np.where(prefer_home, df["HomeWin%"],
                             df["AwayWin%"]).round(1)
    df["Market %"] = np.where(prefer_home, p_home_mkt[keep] * 100,
                              p_away_mkt[keep] * 100).round(1)
    df["Edge"] = (df["Model %"] - df["Market %"]).round(1)
    return df


def filter_head_to_head(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    h2h = load_recent_matchstats(90)
    if h2h.empty:
        return df.iloc[0:0]

    df = df.copy()
    a = df["HomeTeam"].fillna("").astype(str).str.strip()
    b = df["AwayTeam"].fillna("").astype(str).str.strip()
    df["PairKey"] = np.where(a <= b, a + "||" + b, b + "||" + a)

    df = df.merge(
        h2h[["PairKey", "HomeTeam", "AwayTeam", "HomeShotsOn",
             "AwayShotsOn", "Date"]].rename(columns={
                 "HomeTeam": "H2H_Home", "AwayTeam": "H2H_Away",
                 "HomeShotsOn": "H2H_HomeSoT", "AwayShotsOn": "H2H_AwaySoT",
                 "Date": "H2H_Date"}),
        on="PairKey", how="inner",
    )
    if df.empty:
        return df

    # The underdog by price is the side we test.
    df = df[df["Home"] != df["Away"]].copy()
    if df.empty:
        return df

    higher_is_home = df["Home"] > df["Away"]
    high_team = np.where(higher_is_home, df["HomeTeam"], df["AwayTeam"])
    low_team = np.where(higher_is_home, df["AwayTeam"], df["HomeTeam"])

    # Match the previous meeting's orientation to the current one.
    high_was_home = (df["H2H_Home"].values == high_team)
    orientation_ok = high_was_home | (df["H2H_Away"].values == high_team)
    low_matches = np.where(high_was_home,
                           df["H2H_Away"].values == low_team,
                           df["H2H_Home"].values == low_team)

    high_sot = np.where(high_was_home, df["H2H_HomeSoT"], df["H2H_AwaySoT"])
    low_sot = np.where(high_was_home, df["H2H_AwaySoT"], df["H2H_HomeSoT"])

    keep = orientation_ok & low_matches & (high_sot > 2 * low_sot)
    df = df[keep].copy()
    if df.empty:
        return df

    df = tag_side(df, higher_is_home[keep])
    df["H2H SoT"] = [f"{int(h)}–{int(l)}" for h, l
                     in zip(high_sot[keep], low_sot[keep])]
    df["H2H Date"] = pd.to_datetime(
        df["H2H_Date"], errors="coerce").dt.strftime("%d/%m/%Y")
    return df


def filter_league_table(df: pd.DataFrame) -> pd.DataFrame:
    MIN_GAMES, MIN_POS_GAP, MIN_PPG_RATIO = 5, 2, 1.10
    if df.empty:
        return df

    df = df.copy()
    required = ["Home", "Away", "SODD", "Home St.Pos", "Away St.Pos",
                "Home St.PPG", "Away St.PPG",
                "Home St.Games", "Away St.Games"]
    for col in required:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required)
    df = df[(df["Home St.Games"] >= MIN_GAMES)
            & (df["Away St.Games"] >= MIN_GAMES)
            & (df["Home St.PPG"] > 0) & (df["Away St.PPG"] > 0)]
    if df.empty:
        return df

    home_edge = (((df["Away St.Pos"] - df["Home St.Pos"]) >= MIN_POS_GAP)
                 & (df["Home St.PPG"] >= df["Away St.PPG"] * MIN_PPG_RATIO)
                 & (df["Home"] > df["Away"]) & (df["SODD"] > 0))
    away_edge = (((df["Home St.Pos"] - df["Away St.Pos"]) >= MIN_POS_GAP)
                 & (df["Away St.PPG"] >= df["Home St.PPG"] * MIN_PPG_RATIO)
                 & (df["Away"] > df["Home"]) & (df["SODD"] < 0))

    keep = home_edge | away_edge
    df = df[keep].copy()
    if df.empty:
        return df

    # Recomputed against the filtered index rather than reusing the stale mask.
    df = tag_side(df, home_edge[keep])
    df["Pos Gap"] = (df[["Home St.Pos", "Away St.Pos"]].max(axis=1)
                     - df[["Home St.Pos", "Away St.Pos"]].min(axis=1))
    df["PPG Ratio"] = (df[["Home St.PPG", "Away St.PPG"]].max(axis=1)
                       / df[["Home St.PPG", "Away St.PPG"]].min(axis=1)).round(2)
    return df


FILTERS = {
    "All": (filter_all,
            "Every upcoming fixture with valid odds. No signal applied."),
    "SODD": (filter_sodd,
             "Shots-on-target differential against shared opponents of at "
             "least 7, with the price on the favoured side above a sliding "
             "minimum that tightens as the signal strengthens."),
    "SODD + COSOD": (filter_sodd_cosod,
                     "As SODD but from a threshold of 3, and only where the "
                     "favoured side also outperformed shared opponents while "
                     "the other side underperformed them."),
    "xG / xSoT": (filter_xg_xsot,
                  "Expected goals and expected shots on target must both "
                  "favour the same side, with a combined dominance score of "
                  "at least 3."),
    "XWin %": (filter_xwin_percent,
               "Model win probability exceeds the odds-implied probability by "
               "at least 7 percentage points and 75% in relative terms."),
    "H2H": (filter_head_to_head,
            "The higher-priced side had more than twice the shots on target "
            "of its opponent when these two last met, within 90 days."),
    "League": (filter_league_table,
               "The favoured side is at least 2 league places above its "
               "opponent, has at least 1.10× the points per game, is priced "
               "higher, and has SODD in its favour."),
}


# ============================================================
# PRESENTATION
# ============================================================

GLOSSARY = [
    ("Side / Pick", "Which team the selected filter favours."),
    ("Adv Odds", "Decimal odds on the favoured side."),
    ("Implied %", "Probability implied by those odds."),
    ("Signal", "Strength of the filter's signal. Meaning varies by filter."),
    ("Min Odds", "The lowest price this filter accepts at that signal "
                 "strength. Stronger signals demand shorter prices."),
    ("SODD", "Shots-on-target differential against shared opponents. "
             "Positive favours the home side."),
    ("HCOSOD / ACOSOD", "Each side's own shots-on-target differential "
                        "against those shared opponents."),
    ("XGH / XGA", "Expected goals, home and away."),
    ("ESOTH / ESOTA", "Expected shots on target, home and away."),
    ("St.Pos / St.PPG", "League position and points per game."),
    ("Model % vs Market %", "The model's win probability against the "
                            "probability implied by the odds."),
    ("Edge", "Model probability minus market probability, in percentage "
             "points. Positive means the model rates it more likely than the "
             "market does."),
]


def render_help(active: str):
    with st.expander("How this filter works, and what the columns mean"):
        st.markdown(
            f'<div style="font-size:13px;color:{TEXT_1};margin-bottom:10px;">'
            f"<b>{active}</b> — <span style='color:{TEXT_2};'>"
            f"{FILTERS[active][1]}</span></div>"
            f'<div style="font-size:13px;color:{TEXT_2};line-height:1.7;">'
            + "".join(
                f'<div><span style="color:{TEXT_1};">{t}</span> — {d}</div>'
                for t, d in GLOSSARY
            )
            + f'<div style="margin-top:10px;color:{TEXT_3};">'
              f"Signed columns are shaded by size and direction: "
              f'<span style="color:{GOOD_HEX};">green</span> for large '
              f'positive values, <span style="color:{BAD_HEX};">red</span> '
              f"for large negative ones, with zero left uncoloured. On "
              f"differentials such as SODD and COSOD, positive favours the "
              f"home side and negative the away side — the shade indicates "
              f"how pronounced the difference is, not which outcome is "
              f"preferable. Intensity is set against the spread of that "
              f"column across every upcoming fixture, so the same value "
              f"always reads the same shade.</div></div>",
            unsafe_allow_html=True,
        )


def scale_colour(value, avg, higher_is_better=True) -> str:
    """Colour by proportional deviation from a non-zero baseline."""
    try:
        value, avg = float(value), float(avg)
        if avg == 0 or pd.isna(value) or pd.isna(avg):
            return ""
        return _tint((value - avg) / abs(avg), higher_is_better)
    except (TypeError, ValueError):
        return ""


def signed_colour(value, scale, higher_is_better=True) -> str:
    """
    Colour a value where zero is neutral rather than a baseline ratio.
    SODD, edges and gaps are all signed quantities — measuring them as a
    deviation from 1.0 would paint a neutral zero bright red.
    """
    try:
        value = float(value)
        if pd.isna(value) or scale == 0:
            return ""
        return _tint(value / scale, higher_is_better)
    except (TypeError, ValueError):
        return ""


def _tint(norm: float, higher_is_better: bool) -> str:
    intensity = float(np.tanh(abs(norm) * 1.2))
    good = norm > 0 if higher_is_better else norm < 0
    rgb = np.clip(CELL_BASE + ((GOOD if good else BAD) - CELL_BASE)
                  * intensity, 0, 255).astype(int)
    lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    fg = PAGE_BG if lum > 140 else TEXT_1
    return f"background-color:rgb({rgb[0]},{rgb[1]},{rgb[2]});color:{fg};"


CORE_COLS = ["EventID", "Date", "Kickoff", "League", "HomeTeam", "AwayTeam",
             "Home", "Draw", "Away"]
SIGNAL_COLS = ["Side", "Pick", "AdvOdds", "Implied %", "Signal", "MinOdds",
               "Model %", "Market %", "Edge", "COSOD Adv", "COSOD Opp",
               "ESOT Gap", "xG Gap", "H2H SoT", "H2H Date",
               "Pos Gap", "PPG Ratio"]
DETAIL_COLS = ["ComOpp", "SODD", "HCOSOD", "ACOSOD",
               "Home St.Pos", "Away St.Pos", "Home St.PPG", "Away St.PPG",
               "XGH", "XGA", "ESOTH", "ESOTA", "XConvH", "XConvA",
               "HomeWin%", "Draw%", "AwayWin%"]

# Signed fields: zero is neutral, large positive tints green, large negative
# tints red. The intensity scale for each is derived from the spread of that
# field across all upcoming fixtures rather than a hardcoded constant, so a
# given value always reads the same shade no matter which filter is active.
SIGNED_FIELDS = [
    "SODD", "COSOD Adv", "COSOD Opp", "ESOT Gap", "xG Gap",
    "Edge", "Signal", "Pos Gap",
]

# How to reconstruct each field's distribution from the full fixture set.
# Several of these columns only exist after a filter runs, so their scale is
# computed from the underlying inputs instead.
SCALE_SOURCES = {
    "SODD": lambda d: d.get("SODD"),
    "Signal": lambda d: d.get("SODD"),
    "COSOD Adv": lambda d: _pool(d, ["HCOSOD", "ACOSOD"]),
    "COSOD Opp": lambda d: _pool(d, ["HCOSOD", "ACOSOD"]),
    "ESOT Gap": lambda d: _gap(d, "ESOTH", "ESOTA"),
    "xG Gap": lambda d: _gap(d, "XGH", "XGA"),
    "Pos Gap": lambda d: _gap(d, "Home St.Pos", "Away St.Pos"),
    "Edge": lambda d: _edge_spread(d),
}

DEFAULT_SCALES = {
    "SODD": 8.0, "Signal": 8.0, "COSOD Adv": 3.0, "COSOD Opp": 3.0,
    "ESOT Gap": 3.0, "xG Gap": 1.0, "Pos Gap": 8.0, "Edge": 15.0,
}


def _pool(d: pd.DataFrame, cols: list[str]) -> pd.Series | None:
    present = [c for c in cols if c in d.columns]
    if not present:
        return None
    return pd.concat([pd.to_numeric(d[c], errors="coerce") for c in present],
                     ignore_index=True)


def _gap(d: pd.DataFrame, a: str, b: str) -> pd.Series | None:
    if a not in d.columns or b not in d.columns:
        return None
    return (pd.to_numeric(d[a], errors="coerce")
            - pd.to_numeric(d[b], errors="coerce"))


def _edge_spread(d: pd.DataFrame) -> pd.Series | None:
    need = ["Home", "Draw", "Away", "HomeWin%", "AwayWin%"]
    if any(c not in d.columns for c in need):
        return None
    o = {c: pd.to_numeric(d[c], errors="coerce") for c in need}
    overround = 1 / o["Home"] + 1 / o["Draw"] + 1 / o["Away"]
    home = o["HomeWin%"] - (1 / o["Home"]) / overround * 100
    away = o["AwayWin%"] - (1 / o["Away"]) / overround * 100
    return pd.concat([home, away], ignore_index=True)


def compute_scales(base: pd.DataFrame) -> dict:
    """
    One scale per field, taken as the 90th percentile of absolute values so a
    handful of outliers can't wash the whole column to full saturation.
    """
    scales = {}
    for field in SIGNED_FIELDS:
        fallback = DEFAULT_SCALES.get(field, 1.0)
        source = SCALE_SOURCES.get(field)
        series = source(base) if (source and not base.empty) else None

        if series is None:
            scales[field] = fallback
            continue

        values = pd.to_numeric(series, errors="coerce").abs().dropna()
        values = values[values > 0]
        if len(values) < 5:
            scales[field] = fallback
            continue

        scale = float(values.quantile(0.90))
        scales[field] = scale if scale > 0 else fallback

    return scales


# Ratio fields are tinted by proportional deviation from a fixed baseline.
RATIO_FIELDS = {"PPG Ratio": 1.0}


def build_view(df: pd.DataFrame, show_detail: bool) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=CORE_COLS)

    df = df.copy()
    df["Date"] = df["KickoffDT"].dt.strftime("%d/%m")
    df["Kickoff"] = df["KickoffDT"].dt.strftime("%H:%M")
    if "ImpliedProb" in df.columns:
        df["Implied %"] = df["ImpliedProb"]
    if "AdvOdds" in df.columns:
        df["AdvOdds"] = pd.to_numeric(df["AdvOdds"], errors="coerce").round(2)

    present_signal = [c for c in SIGNAL_COLS if c in df.columns]
    cols = CORE_COLS + present_signal
    if show_detail:
        cols += [c for c in DETAIL_COLS if c in df.columns and c not in cols]

    return df.reindex(columns=cols)


def style_view(view: pd.DataFrame, scales: dict):
    numeric = list(view.select_dtypes(include="number").columns)
    styler = view.style
    if numeric:
        styler = styler.format(precision=2, na_rep="—", subset=numeric)

    for col in SIGNED_FIELDS:
        if col not in view.columns:
            continue
        styler = styler.map(
            lambda v, s=scales.get(col, 1.0): signed_colour(v, s, True),
            subset=[col],
        )

    for col, baseline in RATIO_FIELDS.items():
        if col not in view.columns:
            continue
        styler = styler.map(
            lambda v, b=baseline: scale_colour(v, b, True), subset=[col],
        )

    return styler


# ============================================================
# APP
# ============================================================

st.markdown(
    f'<h2 style="font-size:22px;margin:0 0 2px 0;">Pre-game finder</h2>'
    f'<div style="font-size:13px;color:{TEXT_3};margin-bottom:14px;">'
    f"Upcoming fixtures screened against a set of statistical filters. "
    f"Read-only, updated automatically.</div>",
    unsafe_allow_html=True,
)

active = st.radio(
    "Filter", list(FILTERS.keys()), horizontal=True,
    label_visibility="collapsed", key="active_filter",
)

with st.spinner("Loading fixtures…"):
    base = add_standings(apply_global_filters(load_fixtures()))
    scales = compute_scales(base)
    result = FILTERS[active][0](base)

left, right = st.columns([3, 1])
with left:
    st.markdown(
        f'<div style="font-size:15px;color:{TEXT_1};padding-top:6px;">'
        f"{len(result)} fixture{'s' if len(result) != 1 else ''} "
        f'<span style="color:{TEXT_3};">· {active}</span></div>',
        unsafe_allow_html=True,
    )
with right:
    show_detail = st.toggle("All columns", value=False)

render_help(active)

view = build_view(result, show_detail)

if view.empty:
    st.info(
        f"No fixtures currently match the {active} filter. "
        "Try another filter, or check back after the next odds refresh."
    )
else:
    st.dataframe(
        style_view(view, scales),
        use_container_width=True,
        height=min(620, 40 * len(view) + 60),
        hide_index=True,
    )

    st.download_button(
        "Export CSV",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name=(f"pre_game_finder_"
                   f"{active.lower().replace(' ', '_').replace('/', '')}_"
                   f"{datetime.now(TZ).strftime('%Y%m%d_%H%M')}.csv"),
        mime="text/csv",
    )

st.markdown(
    f'<div style="margin-top:1.4rem;padding-top:0.8rem;'
    f"border-top:1px solid {BORDER};color:{TEXT_3};font-size:12px;"
    f'text-align:center;">Fixture data last refreshed '
    f"<span style='color:{TEXT_2};'>{get_last_refresh()}</span> "
    f"London time</div>",
    unsafe_allow_html=True,
)
