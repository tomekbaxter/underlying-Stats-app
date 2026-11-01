"""
Travel & Conditions tab. Draws one fixture's saved research record (from the
fixture_context table, written by the research worker on the Mac). Nothing is
researched here: the app only reads, so selecting a fixture never waits on it.

    travel_badge(ctx)        small travel-impact pill for the header
    render(ctx, colours)     the full tab
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

IMPACT_COLOURS = {          # travel burden, cool to warm
    "VERY LOW": "#3B82F6", "LOW": "#22C55E", "MODERATE": "#EAB308",
    "HIGH": "#F97316", "VERY HIGH": "#EF4444",
}
WEATHER = {                 # WMO weather code -> (icon, words)
    0: ("☀️", "Clear"), 1: ("🌤️", "Mainly clear"), 2: ("⛅", "Partly cloudy"), 3: ("☁️", "Overcast"),
    45: ("🌫️", "Fog"), 48: ("🌫️", "Fog"), 51: ("🌦️", "Light drizzle"), 53: ("🌦️", "Drizzle"),
    55: ("🌧️", "Heavy drizzle"), 56: ("🌧️", "Freezing drizzle"), 57: ("🌧️", "Freezing drizzle"),
    61: ("🌦️", "Light rain"), 63: ("🌧️", "Rain"), 65: ("🌧️", "Heavy rain"), 66: ("🌧️", "Freezing rain"),
    67: ("🌧️", "Freezing rain"), 71: ("🌨️", "Light snow"), 73: ("🌨️", "Snow"), 75: ("❄️", "Heavy snow"),
    77: ("🌨️", "Snow grains"), 80: ("🌦️", "Showers"), 81: ("🌧️", "Showers"), 82: ("⛈️", "Heavy showers"),
    85: ("🌨️", "Snow showers"), 86: ("🌨️", "Snow showers"), 95: ("⛈️", "Thunderstorm"),
    96: ("⛈️", "Thunderstorm, hail"), 99: ("⛈️", "Thunderstorm, hail"),
}


def _weather(code):
    try:
        return WEATHER.get(int(code), ("🌡️", "—"))
    except (TypeError, ValueError):
        return ("🌡️", "—")


def _num(v, fmt="{:.0f}", none="—"):
    return none if v is None else fmt.format(v)


def _pill(text, colour, title=""):
    return (f'<span title="{title}" style="display:inline-flex;align-items:center;gap:6px;'
            f'padding:3px 10px;border-radius:999px;font-size:12px;font-weight:600;letter-spacing:.3px;'
            f'background:{colour}22;color:{colour};border:1px solid {colour}55;white-space:nowrap;">{text}</span>')


def _hours(t):
    return f"≈{t['practical_hours']:g} h" if t.get("practical_hours") else "—"


def travel_badge(ctx) -> str:
    t, level = ctx["travel"], ctx["impact"]["travel"]
    dist = f"{t['road_km']} km" if t.get("road_km") else f"{t['straight_km']} km direct"
    tip = (f"{ctx['away']['city'] or ctx['away']['team_name']} → {ctx['venue']['city'] or '?'} · {dist} · "
           f"{t['likely_transport']} {_hours(t)} · travel impact {level}")
    icon = ('<svg width="16" height="12" viewBox="0 0 16 12" aria-hidden="true">'
            '<circle cx="2.5" cy="9.5" r="2" fill="currentColor"/><circle cx="13.5" cy="2.5" r="2" fill="currentColor"/>'
            '<path d="M4 8.5 C7 8 9 4 12 3.5" stroke="currentColor" stroke-width="1.5" fill="none" '
            'stroke-dasharray="2 1.5"/></svg>')
    return _pill(f"{icon}{level} · {dist}", IMPACT_COLOURS.get(level, "#93A0B8"), tip)


def _metric(label, value, sub="", colour=None):
    c = f"color:{colour};" if colour else ""
    return (f'<div style="background:var(--s1);border:1px solid var(--b);border-radius:10px;padding:10px 12px;">'
            f'<div style="font-size:11px;color:var(--t3);text-transform:uppercase;letter-spacing:.6px;">{label}</div>'
            f'<div style="font-size:20px;font-weight:600;{c}margin-top:2px;">{value}</div>'
            f'<div style="font-size:11px;color:var(--t2);">{sub}</div></div>')


def _summary(ctx, c):
    t, w, imp = ctx["travel"], ctx.get("weather") or {}, ctx["impact"]
    alt, tz = t.get("altitude_diff_m"), t.get("timezone_diff_h")
    cells = [
        _metric("Away travel", f"{t['road_km']} km" if t.get("road_km") else f"{t['straight_km']} km",
                f"{t['straight_km']} km direct" if t.get("road_km") else "direct, no road route"),
        _metric("Journey", _hours(t), t["likely_transport"]),
        _metric("Temperature", f"{_num(w.get('temp_c'))}°C" if w else "—",
                f"feels {_num(w.get('feels_c'))}°C" if w else "no forecast yet"),
        _metric("Altitude change", "—" if alt is None else f"{alt:+,} m", "venue v away base"),
        _metric("Timezone", "—" if tz is None else f"{tz:+g} h", "same zone" if tz == 0 else ""),
        _metric("Overall impact", imp["overall"], f"travel {imp['travel'].lower()}",
                IMPACT_COLOURS.get(imp["overall"])),
    ]
    st.markdown(
        f'<div style="--s1:{c["SURFACE_1"]};--b:{c["BORDER"]};--t2:{c["TEXT_2"]};--t3:{c["TEXT_3"]};'
        f'display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;margin:4px 0 14px;">'
        + "".join(cells) + "</div>", unsafe_allow_html=True)


def _map(ctx, c):
    h, a, t = ctx["venue"], ctx["away"], ctx["travel"]
    fig = go.Figure()
    if t.get("route"):
        lons, lats = zip(*t["route"])
        fig.add_trace(go.Scattergeo(lon=lons, lat=lats, mode="lines", hoverinfo="skip",
                                    line=dict(width=3, color=IMPACT_COLOURS.get(ctx["impact"]["travel"]))))
    fig.add_trace(go.Scattergeo(lon=[a["lon"], h["lon"]], lat=[a["lat"], h["lat"]], mode="lines",
                                hoverinfo="skip", line=dict(width=1, color=c["TEXT_3"], dash="dot")))
    for p, colour, name in ((h, c["HOME"], ctx["home"]["team_name"]), (a, c["AWAY"], a["team_name"])):
        fig.add_trace(go.Scattergeo(
            lon=[p["lon"]], lat=[p["lat"]], mode="markers+text",
            marker=dict(size=11, color=colour, line=dict(width=2, color=c["PAGE_BG"])),
            text=[p.get("city") or ""], textposition="top center", textfont=dict(color=c["TEXT_1"], size=12),
            hovertemplate=f"{name}<br>{p.get('city') or ''}<extra></extra>"))
    # Pad to the panel's shape (about 1.7 : 1) so the map fills it.
    lat_pad = max(1.0, abs(h["lat"] - a["lat"]) * 0.35)
    lat_span = abs(h["lat"] - a["lat"]) + 2 * lat_pad
    lon_pad = max(abs(h["lon"] - a["lon"]) * 0.2, (lat_span * 1.7 * 1.2 - abs(h["lon"] - a["lon"])) / 2)
    fig.update_geos(
        projection_type="mercator", resolution=50, showcountries=True, countrycolor=c["BORDER"],
        showland=True, landcolor=c["SURFACE_2"], showocean=True, oceancolor=c["PAGE_BG"],
        showlakes=False, coastlinecolor=c["TEXT_3"], coastlinewidth=0.6, bgcolor=c["PAGE_BG"],
        lataxis_range=[min(h["lat"], a["lat"]) - lat_pad, max(h["lat"], a["lat"]) + lat_pad],
        lonaxis_range=[min(h["lon"], a["lon"]) - lon_pad, max(h["lon"], a["lon"]) + lon_pad])
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
                      paper_bgcolor=c["PAGE_BG"], dragmode=False)
    return fig


def _travel_card(ctx, c):
    t, h, a, v = ctx["travel"], ctx["home"], ctx["away"], ctx["venue"]
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;gap:8px;">'
        f'<div style="font-size:14px;color:{c["TEXT_1"]};"><span style="color:{c["AWAY"]};">'
        f'{a["city"] or a["team_name"]}</span> → <span style="color:{c["HOME"]};">{v["city"] or h["team_name"]}'
        f'</span></div>{travel_badge(ctx)}</div>', unsafe_allow_html=True)
    st.plotly_chart(_map(ctx, c), use_container_width=True,
                    config={"displayModeBar": False, "scrollZoom": False})
    road = (f'<b style="color:{c["TEXT_1"]};">{t["road_km"]} km</b> by road ({t["road_hours"]:g} h drive) · '
            if t.get("road_km") else "No road route · ")
    st.markdown(
        f'<div style="font-size:12px;color:{c["TEXT_2"]};line-height:1.6;">{road}'
        f'<b style="color:{c["TEXT_1"]};">{t["straight_km"]} km</b> direct · '
        f'<b style="color:{c["TEXT_1"]};">{t["likely_transport"]}</b> {_hours(t)}<br>'
        f'Altitude {_num(a.get("altitude_m"))} m → {_num(v.get("altitude_m"))} m · venue '
        f'<b>{v["confidence"].lower()}</b> ({v["note"]})<br>'
        f'<span style="color:{c["TEXT_3"]};">Club bases: {h["status"].lower()} / {a["status"].lower()}. '
        f'Route: {t["route_source"]}. Journey time is an estimate, not the team\'s actual plan.</span></div>',
        unsafe_allow_html=True)


def _weather_card(ctx, c):
    w = ctx.get("weather")
    if not w:
        st.markdown(
            f'<div style="background:{c["SURFACE_1"]};border:1px solid {c["BORDER"]};border-radius:14px;'
            f'padding:16px 18px;font-size:13px;color:{c["TEXT_2"]};">No forecast for kick-off yet '
            f'(the forecast reaches about two weeks ahead).</div>', unsafe_allow_html=True)
        return
    icon, words = _weather(w["code"])
    local = ctx["kickoff"]["local"]
    day = "Daytime" if w["is_day"] else "Night"
    temps = [p["temp"] for p in w["timeline"] if p["temp"] is not None] or [w["temp_c"]]
    lo, hi = min(temps) - 1, max(temps) + 1
    bars = []
    for p in w["timeline"]:
        hgt = 18 + 34 * ((p["temp"] if p["temp"] is not None else lo) - lo) / (hi - lo)
        ko = p["kickoff"]
        bars.append(
            f'<div style="flex:1;text-align:center;">'
            f'<div style="font-size:11px;color:{c["TEXT_1"] if ko else c["TEXT_2"]};">{_num(p["temp"])}°</div>'
            f'<div style="height:52px;display:flex;align-items:flex-end;justify-content:center;">'
            f'<div style="width:60%;height:{hgt:.0f}px;border-radius:4px 4px 0 0;'
            f'background:{c["HOME"] if ko else c["SURFACE_3"]};"></div></div>'
            f'<div style="font-size:10px;color:{c["TEXT_3"]};margin-top:3px;">{"KO" if ko else p["time"]}</div>'
            f'<div style="font-size:10px;color:{c["TEXT_3"]};">{_weather(p["code"])[0]}'
            f'{"" if p["is_day"] else "☾"}</div></div>')
    tz = ctx["travel"].get("timezone_diff_h")
    away_t = w.get("away_base_temp_c")
    foot = [f"Timezone difference {tz:+g} h" if tz is not None else "",
            f"{ctx['away']['city'] or 'away base'} {away_t:.0f}°C at the same time" if away_t is not None else ""]
    st.markdown(
        f'<div style="background:linear-gradient(160deg,{c["SURFACE_2"]},{c["SURFACE_1"]});border:1px solid {c["BORDER"]};'
        f'border-radius:14px;padding:16px 18px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px;">'
        f'<div><div style="font-size:13px;color:{c["TEXT_2"]};">{ctx["venue"]["city"] or ""} · {local[:10]}</div>'
        f'<div style="font-size:12px;color:{c["TEXT_3"]};">Kick-off {local[11:]} · {ctx["kickoff"]["uk"][11:]} UK</div></div>'
        f'{_pill(("☀ " if w["is_day"] else "☾ ") + day, c["AWAY"] if w["is_day"] else c["HOME"])}</div>'
        f'<div style="display:flex;align-items:center;gap:14px;margin:12px 0 8px;">'
        f'<div style="font-size:46px;line-height:1;">{icon}</div>'
        f'<div><div style="font-size:38px;font-weight:600;color:{c["TEXT_1"]};line-height:1;">{_num(w["temp_c"])}°C</div>'
        f'<div style="font-size:12px;color:{c["TEXT_2"]};">{words} · feels {_num(w["feels_c"])}°C</div></div></div>'
        f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;font-size:12px;color:{c["TEXT_2"]};'
        f'margin-bottom:12px;"><div>💧 {_num(w["rain_prob"])}% rain</div><div>💨 {_num(w["wind_kmh"])} km/h</div>'
        f'<div>🌬️ gusts {_num(w["gust_kmh"])}</div><div>🌡️ {_num(w["humidity"])}% hum.</div></div>'
        f'<div style="display:flex;gap:2px;border-top:1px solid {c["BORDER"]};padding-top:8px;">{"".join(bars)}</div>'
        f'<div style="font-size:11px;color:{c["TEXT_3"]};margin-top:8px;">{" · ".join(x for x in foot if x)}</div></div>',
        unsafe_allow_html=True)
    st.caption(f"{w['source']}, {w['window']}")


def render(ctx, c):
    if not ctx:
        st.caption("Travel and conditions haven't been researched for this fixture yet.")
        return
    _summary(ctx, c)
    left, right = st.columns([3, 2], gap="medium")
    with left:
        _travel_card(ctx, c)
    with right:
        _weather_card(ctx, c)
    st.caption(f"Researched {ctx['researched_at'][:16].replace('T', ' ')} UTC · confidence {ctx['confidence'].lower()}")
