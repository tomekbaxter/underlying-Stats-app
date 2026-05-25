import datetime as dtfrom zoneinfo import ZoneInfofrom urllib.parse import quote_plus

import matplotlibmatplotlib.use("Agg")

import matplotlib.pyplot as pltimport numpy as npimport pandas as pdimport streamlit as stfrom sqlalchemy import create_engine, textfrom streamlit.errors import StreamlitSecretNotFoundError

============================================================

STREAMLIT CONFIG

============================================================

st.set_page_config(page_title="Underlying Stats", layout="wide")

st.markdown("""
