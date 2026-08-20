# =============================================================================
# TEAM OVERVIEW - football-app style club overview
# =============================================================================
from __future__ import annotations

import base64
import mimetypes
import re
from datetime import date
from functools import lru_cache

import numpy as np
import pandas as pd
import streamlit as st

from utils import data, match_analysis as ma, pitch, player_analysis as pa, team_analysis as ta, ui

TEAM_BADGE_DIR = ui.ASSETS_DIR / "team_badges"

TEAM_BADGE_FILES = {
    "barnsley": "Barnsley_FC.svg.png",
    "birmingham": "Birmingham-City.png",
    "blackburn": "Blackburn_Rovers.svg.png",
    "blackpool": "Blackpool_FC_logo.svg.png",
    "bolton": "Bolton_Wanderers_FC_logo.svg.webp",
    "bristol city": "Bristol_City_crest.svg.webp",
    "bristol rovers": "Bristol_Rovers_F.C._logo.svg.png",
    "bromley": "Bromley_FC_crest.svg.png",
    "burnley": "Burnley_FC_Logo.svg.webp",
    "burton": "Burton_Albion_FC_logo.svg.png",
    "cambridge": "Cambridge_United_FC.svg.png",
    "cardiff": "Cardiff_City_crest.svg",
    "charlton": "Charlton Logo.png",
    "chelsea": "Chelsea_FC.svg.png",
    "coventry": "Coventry_City_FC_crest.svg.png",
    "crawley": "Crawley_Town_FC_crest.svg.png",
    "derby": "Derby_County_crest.svg.png",
    "exeter": "Exeter_City_FC.svg.png",
    "huddersfield": "Huddersfield_Town_AFC_crest.svg.png",
    "hull": "Hull_City_A.F.C._logo.svg.png",
    "ipswich": "Ipswich_Town.svg.png",
    "leicester": "Leicester_City_crest.svg.png",
    "leyton orient": "Leyton_Orient_F.C._logo.svg.png",
    "lincoln": "Lincoln_City_FC_2024_crest.svg.png",
    "luton": "Luton.png",
    "mansfield": "mansfield-town-fc-logo-E2BCF556D5-seeklogo.com.png",
    "middlesbrough": "Middlesbrough_FC_crest.svg.png",
    "millwall": "Millwall_FC_crest.svg.png",
    "northampton": "Northampton_Town_F.C._logo.svg.png",
    "norwich": "Norwich_City.png",
    "oxford": "Oxford_United_FC_logo.svg.png",
    "peterborough": "Peterborough_United.svg.png",
    "plymouth": "Plymouth.jpg",
    "portsmouth": "Portsmouth_FC_logo.svg.png",
    "preston": "Preston_North_End_FC.svg.png",
    "qpr": "Queens_Park_Rangers_crest.svg.png",
    "queens park rangers": "Queens_Park_Rangers_crest.svg.png",
    "reading": "Reading_FC.svg.png",
    "rotherham": "Rotherham_United_FC.svg.png",
    "sheffield united": "Sheffield_United_FC_logo.svg.png",
    "sheffield wednesday": "Sheffield_Wednesday_badge.svg.png",
    "shrewsbury": "Shrewsbury_Town_F.C._logo.svg.png",
    "southampton": "FC_Southampton.svg.png",
    "stevenage": "Stevenage_FC_crest.svg.png",
    "stockport": "Stockport_County_FC_logo_2020.svg.png",
    "stoke": "Stoke_City_FC.svg.png",
    "swansea": "Swansea_City_A.F.C._logo.png",
    "watford": "Watford.svg.png",
    "west brom": "West_Bromwich_Albion.svg.png",
    "west bromwich albion": "West_Bromwich_Albion.svg.png",
    "west ham": "West_Ham_United_FC_logo.svg",
    "west ham united": "West_Ham_United_FC_logo.svg",
    "wigan": "Wigan_Athletic.svg.png",
    "wolverhampton": "Wolverhampton_Wanderers_FC_crest.svg.webp",
    "wolverhampton wanderers": "Wolverhampton_Wanderers_FC_crest.svg.webp",
    "wolves": "Wolverhampton_Wanderers_FC_crest.svg.webp",
    "wrexham": "Wrexham_A.F.C._Logo.svg.png",
    "wycombe": "Wycombe_Wanderers_FC_logo.svg.png",
}

PAGE_SOURCE = (
    "Team overview performance, cards, rankings and radar are derived from completed match rows, "
    "not aggregate-average team tables. Squad and key-player sections use player-level season metrics "
    "where available. Fixture lineups and formations use submitted Opta F7 data; effective-play "
    "possession uses the Second Spectrum physical summary stored for the same DVMS fixture. "
    "Public app metadata such as live news, "
    "trophies, current manager and stadium are not currently stored in the app data model."
)


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .to-note { color: var(--ss-muted); font-size: .86rem; margin: -2px 0 12px; }
        .to-hero {
            background:
                radial-gradient(circle at 92% 16%, rgba(255,255,255,.16), transparent 20%),
                linear-gradient(135deg, #111 0%, #271113 48%, #9c0214 135%);
            border: 1px solid rgba(255,255,255,.12);
            border-radius: 14px;
            box-shadow: var(--ss-shadow);
            color: #fff;
            margin: 6px 0 16px;
            overflow: hidden;
            padding: 26px 28px;
            position: relative;
        }
        .to-hero:before {
            content: "";
            position: absolute;
            inset: 0;
            border-top: 5px solid var(--ss-accent);
            pointer-events: none;
        }
        .to-hero-main {
            align-items: center;
            display: flex;
            gap: 22px;
            justify-content: space-between;
            position: relative;
            z-index: 1;
        }
        .to-team-block { align-items: center; display: flex; gap: 18px; min-width: 0; }
        .to-badge {
            display: inline-grid;
            filter: drop-shadow(0 10px 16px rgba(0,0,0,.32));
            flex: 0 0 auto;
            height: 96px;
            object-fit: contain;
            place-items: center;
            width: 96px;
        }
        .to-badge-fallback {
            background: #fff;
            border: 1px solid rgba(255,255,255,.32);
            border-radius: 50%;
            box-shadow: 0 14px 22px rgba(0,0,0,.28);
            color: var(--ss-accent);
            font-size: 1.35rem;
            font-weight: 900;
            letter-spacing: -.03em;
        }
        .to-eyebrow {
            color: rgba(255,255,255,.78);
            font-size: .76rem;
            font-weight: 850;
            letter-spacing: .12em;
            margin-bottom: 6px;
            text-transform: uppercase;
        }
        .to-title {
            color: #fff;
            font-size: clamp(2rem, 3.4vw, 3.05rem);
            font-weight: 900;
            line-height: 1.02;
            margin: 0 0 12px;
        }
        .to-meta-row, .to-pill-row { display: flex; flex-wrap: wrap; gap: 8px; }
        .to-meta-pill, .to-form-pill {
            border: 1px solid rgba(255,255,255,.18);
            border-radius: 999px;
            color: rgba(255,255,255,.86);
            font-size: .84rem;
            font-weight: 700;
            padding: 7px 11px;
        }
        .to-form-pill { min-width: 34px; text-align: center; }
        .to-win { background: rgba(18,183,106,.92); border-color: rgba(18,183,106,1); }
        .to-draw { background: rgba(152,162,179,.84); border-color: rgba(152,162,179,1); }
        .to-loss { background: rgba(195,0,23,.92); border-color: rgba(195,0,23,1); }
        .to-side { flex: 0 0 285px; }
        .to-side-label {
            color: rgba(255,255,255,.70);
            font-size: .74rem;
            font-weight: 850;
            letter-spacing: .1em;
            margin-bottom: 8px;
            text-transform: uppercase;
        }
        .to-side-text { color: rgba(255,255,255,.78); font-size: .9rem; line-height: 1.45; margin-top: 9px; }
        .to-section {
            color: var(--ss-muted);
            font-size: .78rem;
            font-weight: 850;
            letter-spacing: .1em;
            margin: 28px 0 10px;
            text-transform: uppercase;
        }
        .to-card-grid, .to-player-grid {
            display: grid;
            gap: 12px;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            margin: 8px 0 14px;
        }
        .to-card-grid--snapshot {
            gap: 18px;
            grid-template-columns: repeat(auto-fit, minmax(235px, 1fr));
            justify-content: center;
            margin: 12px auto 28px;
            max-width: 1120px;
        }
        .to-player-grid { grid-template-columns: repeat(auto-fit, minmax(205px, 1fr)); }
        .to-card, .to-fixture, .to-player-card, .to-about {
            background: #fff;
            border: 1px solid var(--ss-border);
            border-radius: 12px;
            box-shadow: 0 1px 2px rgba(16,24,40,.04);
        }
        .to-card { border-top: 3px solid var(--ss-accent); padding: 15px 16px; }
        .to-card-grid--snapshot .to-card {
            min-height: 150px;
            padding: 24px 22px;
            text-align: center;
        }
        .to-card-label {
            color: var(--ss-muted);
            font-size: .78rem;
            font-weight: 800;
            letter-spacing: .04em;
            margin-bottom: 8px;
            text-transform: uppercase;
        }
        .to-card-value {
            color: var(--ss-ink);
            font-size: clamp(1.32rem, 2vw, 1.82rem);
            font-weight: 850;
            line-height: 1.1;
        }
        .to-card-grid--snapshot .to-card-value {
            font-size: clamp(2rem, 3vw, 2.75rem);
            letter-spacing: -.04em;
        }
        .to-card-sub, .to-muted {
            color: var(--ss-muted);
            font-size: .87rem;
            line-height: 1.42;
            margin-top: 7px;
        }
        .to-fixture {
            border-top: 3px solid var(--ss-accent);
            margin-bottom: 12px;
            min-height: 150px;
            padding: 17px 18px;
        }
        .to-fixture-label {
            color: var(--ss-muted);
            font-size: .76rem;
            font-weight: 850;
            letter-spacing: .09em;
            margin-bottom: 10px;
            text-transform: uppercase;
        }
        .to-fixture-main {
            align-items: center;
            color: var(--ss-ink);
            display: flex;
            font-size: 1.05rem;
            font-weight: 850;
            gap: 12px;
            justify-content: space-between;
        }
        .to-score {
            background: var(--ss-accent-soft);
            border-radius: 8px;
            color: var(--ss-accent);
            flex: 0 0 auto;
            font-weight: 900;
            padding: 7px 10px;
        }
        .to-player-card { padding: 15px 16px; }
        .to-player-name { color: var(--ss-ink); font-size: 1rem; font-weight: 850; line-height: 1.22; margin-bottom: 6px; }
        .to-player-stat { color: var(--ss-accent); font-size: 1.2rem; font-weight: 900; margin-top: 10px; }
        .to-about {
            background: linear-gradient(135deg, rgba(255,255,255,.98), rgba(255,241,243,.82));
            padding: 18px 20px;
        }
        .to-about p { color: var(--ss-ink); line-height: 1.55; margin: 0 0 10px; }
        .to-fixture-detail {
            background: #fff;
            border: 1px solid var(--ss-border);
            border-radius: 12px;
            box-shadow: 0 1px 2px rgba(16,24,40,.04);
            margin: 8px 0 14px;
            padding: 18px 20px;
        }
        .to-fixture-detail-top {
            align-items: flex-start;
            display: flex;
            gap: 14px;
            justify-content: space-between;
        }
        .to-fixture-detail-title { color: var(--ss-ink); font-size: 1.12rem; font-weight: 900; }
        .to-fixture-detail-meta { color: var(--ss-muted); font-size: .86rem; margin-top: 5px; }
        .to-fixture-score {
            background: var(--ss-accent-soft);
            border-radius: 9px;
            color: var(--ss-accent);
            flex: 0 0 auto;
            font-size: 1.15rem;
            font-weight: 900;
            padding: 8px 12px;
        }
        .to-possession-card {
            background: #fff;
            border: 1px solid var(--ss-border);
            border-radius: 12px;
            box-shadow: 0 1px 2px rgba(16,24,40,.04);
            margin: 10px 0 16px;
            padding: 20px;
        }
        .to-possession-title { color: var(--ss-ink); font-size: 1rem; font-weight: 900; }
        .to-possession-values {
            align-items: end;
            display: flex;
            gap: 18px;
            justify-content: space-between;
            margin-top: 16px;
        }
        .to-possession-team { color: var(--ss-ink); font-size: .9rem; font-weight: 800; }
        .to-possession-team:last-child { text-align: right; }
        .to-possession-number { display: block; font-size: 1.65rem; font-weight: 900; letter-spacing: -.03em; }
        .to-possession-track {
            background: #e4e7ec;
            border-radius: 999px;
            display: flex;
            height: 16px;
            margin: 11px 0 10px;
            overflow: hidden;
            width: 100%;
        }
        .to-possession-track span { display: block; height: 100%; }
        .to-possession-times {
            color: var(--ss-muted);
            display: flex;
            font-size: .82rem;
            justify-content: space-between;
        }
        .to-possession-source { color: var(--ss-muted); font-size: .8rem; line-height: 1.4; margin-top: 12px; }
        .to-lineup-heading { color: var(--ss-ink); font-size: 1.02rem; font-weight: 900; margin-top: 4px; }
        .to-lineup-shape { color: var(--ss-muted); font-size: .84rem; margin: 4px 0 10px; }
        .to-radar-key {
            background: #ffffff;
            border: 1px solid var(--ss-border);
            border-radius: 12px;
            box-shadow: 0 1px 2px rgba(16,24,40,.04);
            margin: -18px 0 22px;
            padding: 16px 18px;
        }
        .to-radar-key-title {
            color: var(--ss-ink);
            font-size: 1rem;
            font-weight: 850;
            margin-bottom: 8px;
        }
        .to-radar-key-note {
            color: var(--ss-muted);
            font-size: .88rem;
            line-height: 1.45;
            margin-bottom: 12px;
        }
        .to-radar-key-grid {
            display: grid;
            gap: 10px;
            grid-template-columns: repeat(auto-fit, minmax(235px, 1fr));
        }
        .to-radar-key-item {
            background: var(--ss-panel-soft);
            border-left: 3px solid var(--ss-accent);
            border-radius: 8px;
            padding: 10px 12px;
        }
        .to-radar-key-label {
            color: var(--ss-ink);
            font-size: .9rem;
            font-weight: 800;
            margin-bottom: 4px;
        }
        .to-radar-key-metric {
            color: var(--ss-muted);
            font-size: .84rem;
            line-height: 1.35;
        }
        @media (max-width: 880px) {
            .to-hero-main { align-items: flex-start; flex-direction: column; }
            .to-side { flex: 1 1 auto; width: 100%; }
            .to-team-block { align-items: flex-start; flex-direction: column; }
            .to-badge { height: 76px; width: 76px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _section(label: str) -> None:
    st.markdown(f'<div class="to-section">{ui.esc(label)}</div>', unsafe_allow_html=True)


def _normalise_name(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", "" if value is None else str(value).lower()).strip()
    return " ".join(word for word in text.split() if word not in {"fc", "afc", "cf", "football", "club"})


def _best_label(values: pd.Series, team_name: str) -> str:
    labels = values.dropna().astype(str).drop_duplicates().tolist() if not values.empty else []
    if team_name in labels:
        return team_name

    target = _normalise_name(team_name)
    for label in labels:
        if _normalise_name(label) == target:
            return label

    target_words = set(target.split())
    for label in labels:
        label_words = set(_normalise_name(label).split())
        if target_words and (target_words.issubset(label_words) or label_words.issubset(target_words)):
            return label
    return team_name


def _match_labels(matches: pd.DataFrame) -> pd.Series:
    if matches.empty or not {"Home", "Away"}.issubset(matches.columns):
        return pd.Series(dtype=str)
    return pd.concat([matches["Home"], matches["Away"]], ignore_index=True)


def _badge_path(team_name: str) -> object | None:
    team_key = _normalise_name(team_name)
    badge_file = TEAM_BADGE_FILES.get(team_key)
    if badge_file:
        path = TEAM_BADGE_DIR / badge_file
        if path.exists():
            return path

    for alias, filename in sorted(TEAM_BADGE_FILES.items(), key=lambda item: len(item[0]), reverse=True):
        if alias in team_key or team_key in alias:
            path = TEAM_BADGE_DIR / filename
            if path.exists():
                return path

    return None


@lru_cache(maxsize=128)
def _badge_data_uri(path_text: str) -> str:
    path = TEAM_BADGE_DIR / path_text
    if not path.exists():
        return ""
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _badge_html(team_name: str) -> str:
    path = _badge_path(team_name)
    if path is not None:
        uri = _badge_data_uri(path.name)
        if uri:
            return f'<img class="to-badge" src="{uri}" alt="{ui.esc(team_name)} badge">'

    initials = "".join(word[:1] for word in str(team_name).split()[:2]).upper() or "FC"
    return f'<div class="to-badge to-badge-fallback">{ui.esc(initials)}</div>'


def _fmt(value: object, digits: int = 1, fallback: str = "—") -> str:
    try:
        if pd.isna(value):
            return fallback
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return f"{number:,.0f}" if number.is_integer() else f"{number:,.{digits}f}"


def _metric_text(metric: str, value: object) -> str:
    return f"{_fmt(value, 1)}%" if "Pass %" in metric or metric.endswith("%") else _fmt(value, 2)


def _date_text(value: object) -> str:
    if pd.isna(value):
        return "Date unavailable"
    try:
        return pd.to_datetime(value).strftime("%a %d %b %Y")
    except (TypeError, ValueError):
        return str(value)


def _has_score(row: pd.Series) -> bool:
    return row is not None and pd.notna(row.get("Goals For")) and pd.notna(row.get("Goals Against"))


def _result_initial(result: object) -> str:
    return {"Win": "W", "Draw": "D", "Loss": "L"}.get(str(result), "—")


def _completed_team_matches(team_matches: pd.DataFrame) -> pd.DataFrame:
    if team_matches.empty or not {"Goals For", "Goals Against"}.issubset(team_matches.columns):
        return team_matches
    rows = team_matches.copy()
    rows = rows[pd.notna(rows["Goals For"]) & pd.notna(rows["Goals Against"])]
    if "Date" in rows:
        match_dates = pd.to_datetime(rows["Date"], errors="coerce", utc=True)
        today = pd.Timestamp.now(tz="UTC").normalize()
        rows = rows[match_dates.le(today)]

    return rows


def _completed_fixture_rows(matches: pd.DataFrame) -> pd.DataFrame:
    if matches.empty or not {"Home Goals", "Away Goals"}.issubset(matches.columns):
        return matches
    rows = matches.copy()
    rows = rows[pd.notna(rows["Home Goals"]) & pd.notna(rows["Away Goals"])]
    if "Date" in rows:
        match_dates = pd.to_datetime(rows["Date"], errors="coerce", utc=True)
        today = pd.Timestamp.now(tz="UTC").normalize()
        rows = rows[match_dates.le(today)]

    return rows

def _add_venue(team_matches: pd.DataFrame, team_label: str) -> pd.DataFrame:
    if team_matches.empty or "Home" not in team_matches:
        return team_matches
    rows = team_matches.copy()
    is_home = rows["Home"].astype(str) == str(team_label)
    verified = rows["Venue Verified"].astype(bool) if "Venue Verified" in rows else pd.Series(True, index=rows.index)
    rows["Venue"] = np.where(is_home, "Home", "Away")
    rows.loc[~verified & is_home, "Venue"] = "Listed home"
    rows.loc[~verified & ~is_home, "Venue"] = "Listed away"
    return rows


def _record(team_matches: pd.DataFrame) -> dict[str, float | int]:
    rows = _completed_team_matches(team_matches)
    if rows.empty:
        return {"played": 0, "wins": 0, "draws": 0, "losses": 0, "gf": 0, "ga": 0, "points": 0, "gd": 0, "ppg": np.nan}
    wins = int((rows["Team Result"] == "Win").sum())
    draws = int((rows["Team Result"] == "Draw").sum())
    losses = int((rows["Team Result"] == "Loss").sum())
    gf = int(pd.to_numeric(rows["Goals For"], errors="coerce").fillna(0).sum())
    ga = int(pd.to_numeric(rows["Goals Against"], errors="coerce").fillna(0).sum())
    points = int(pd.to_numeric(rows["Points"], errors="coerce").fillna(0).sum())
    played = int(len(rows))
    return {
        "played": played,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "gf": gf,
        "ga": ga,
        "points": points,
        "gd": gf - ga,
        "ppg": round(points / played, 2) if played else np.nan,
    }


def _form_html(team_matches: pd.DataFrame, limit: int = 5) -> str:
    rows = _completed_team_matches(team_matches).tail(limit)
    if rows.empty:
        return '<span class="to-muted">No completed-match form in selected source</span>'

    pills = []
    for _, match in rows.iterrows():
        result = _result_initial(match.get("Team Result"))
        klass = {"W": "to-win", "D": "to-draw", "L": "to-loss"}.get(result, "")
        title = (
            f'{_date_text(match.get("Date"))}: {match.get("Opponent", "Opponent")} '
            f'{_fmt(match.get("Goals For"), 0)}-{_fmt(match.get("Goals Against"), 0)}'
        )
        pills.append(f'<span class="to-form-pill {klass}" title="{ui.esc(title)}">{ui.esc(result)}</span>')
    return "".join(pills)


def _fixture_split(team_matches: pd.DataFrame) -> tuple[pd.Series | None, pd.Series | None]:
    if team_matches.empty:
        return None, None
    rows = team_matches.copy()
    if "Date" in rows:
        rows["Date"] = pd.to_datetime(rows["Date"], errors="coerce", utc=True)
        today = pd.Timestamp.now(tz="UTC").normalize()

        has_score = pd.notna(rows["Goals For"]) & pd.notna(rows["Goals Against"])
        previous = rows[rows["Date"].le(today) & has_score].tail(1)
        next_rows = rows[rows["Date"].gt(today) | ~has_score].head(1)
    else:
        previous = rows[pd.notna(rows["Goals For"]) & pd.notna(rows["Goals Against"])].tail(1)
        next_rows = pd.DataFrame()
    return (previous.iloc[0] if not previous.empty else None, next_rows.iloc[0] if not next_rows.empty else None)


def _cards(cards: list[dict[str, str]], class_name: str = "to-card-grid") -> None:
    html = []

    for card in cards:
        label = ui.esc(card.get("label", ""))
        value = ui.esc(card.get("value", "—"))
        sub = ui.esc(card.get("sub", ""))

        html.append(
            '<div class="to-card">'
            f'<div class="to-card-label">{label}</div>'
            f'<div class="to-card-value">{value}</div>'
            f'<div class="to-card-sub">{sub}</div>'
            '</div>'
        )

    st.markdown(
        f'<div class="{ui.esc(class_name)}">{"".join(html)}</div>',
        unsafe_allow_html=True,
    )


def _fixture_card(label: str, row: pd.Series | None, empty: str) -> None:
    if row is None:
        st.markdown(
            f'<div class="to-fixture"><div class="to-fixture-label">{ui.esc(label)}</div><div class="to-muted">{ui.esc(empty)}</div></div>',
            unsafe_allow_html=True,
        )
        return

    opponent = row.get("Opponent", "Opponent")
    score = f'{_fmt(row.get("Goals For"), 0)}-{_fmt(row.get("Goals Against"), 0)}' if _has_score(row) else "vs"
    result = row.get("Team Result") if _has_score(row) else ""
    sub = " · ".join(
        str(bit)
        for bit in [_date_text(row.get("Date")), row.get("Competition", ""), row.get("Venue", ""), result]
        if bit and str(bit) != "nan"
    )
    st.markdown(
        f"""
        <div class="to-fixture">
            <div class="to-fixture-label">{ui.esc(label)}</div>
            <div class="to-fixture-main"><span>{ui.esc(opponent)}</span><span class="to-score">{ui.esc(score)}</span></div>
            <div class="to-muted">{ui.esc(sub)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _form_summary_card(team_matches: pd.DataFrame) -> None:
    recent = _completed_team_matches(team_matches).tail(5)
    if recent.empty:
        summary = "No completed fixtures are available for the selected match source."
    else:
        wins = int((recent["Team Result"] == "Win").sum())
        draws = int((recent["Team Result"] == "Draw").sum())
        losses = int((recent["Team Result"] == "Loss").sum())
        points = int(pd.to_numeric(recent["Points"], errors="coerce").fillna(0).sum())
        summary = f"Last {len(recent)} matches · {points} points · {wins}W {draws}D {losses}L"

    st.markdown(
        (
            '<div class="to-fixture">'
            '<div class="to-fixture-label">Recent Form</div>'
            f'<div class="to-pill-row">{_form_html(team_matches)}</div>'
            f'<div class="to-muted">{ui.esc(summary)}</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


def _competition(matches: pd.DataFrame, team_matches: pd.DataFrame) -> str:
    source = team_matches if not team_matches.empty else matches
    if not source.empty and "Competition" in source:
        mode = source["Competition"].dropna().astype(str).mode()
        if not mode.empty:
            return mode.iloc[0]
    return "Competition unavailable"


def _league_fixture_rows(matches: pd.DataFrame) -> pd.DataFrame:
    return ta.regular_season_fixtures(matches)


def _standings(matches: pd.DataFrame) -> pd.DataFrame:
    completed = _league_fixture_rows(matches)
    table = ma.team_record_table(completed)
    if table.empty:
        return table
    table = table.copy()
    table.insert(0, "#", np.arange(1, len(table) + 1))
    forms = []
    clean_sheets = []
    scoring_matches = []
    for team_name in table["Team"].astype(str):
        team_rows = ma.team_match_rows(completed, team_name)
        forms.append("".join(_result_initial(value) for value in team_rows.tail(5)["Team Result"]))
        goals_for = pd.to_numeric(team_rows.get("Goals For", pd.Series(index=team_rows.index, dtype=float)), errors="coerce")
        goals_against = pd.to_numeric(
            team_rows.get("Goals Against", pd.Series(index=team_rows.index, dtype=float)),
            errors="coerce",
        )
        clean_sheets.append(int(goals_against.eq(0).sum()))
        scoring_matches.append(int(goals_for.gt(0).sum()))
    table["Form"] = forms
    table["Clean Sheets"] = clean_sheets
    table["Scoring Matches"] = scoring_matches
    return table


def _surrounding_table(table: pd.DataFrame, team_name: str, radius: int = 2) -> pd.DataFrame:
    if table.empty:
        return table
    label = _best_label(table["Team"], team_name)
    found = table.index[table["Team"].astype(str) == str(label)].tolist()
    if not found:
        return table.head(8)
    idx = found[0]
    start = max(0, idx - radius)
    end = min(len(table), idx + radius + 1)
    if end - start < radius * 2 + 1:
        start = max(0, end - (radius * 2 + 1))
    return table.iloc[start:end].copy()


MATCH_SNAPSHOT_METRICS = [
    "Points / Match",
    "Goals For / Match",
    "Goals Against / Match",
    "Win %",
]


def _match_metric_table(table: pd.DataFrame) -> pd.DataFrame:
    required = {"Team", "Played", "Wins", "Losses", "Points", "GF", "GA", "GD"}
    if table.empty or not required.issubset(table.columns):
        return pd.DataFrame(columns=["Team", *MATCH_SNAPSHOT_METRICS])

    out = table.copy()
    for col in ["Played", "Wins", "Losses", "Points", "GF", "GA", "GD"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    played = out["Played"].replace(0, np.nan)
    out["Points / Match"] = (out["Points"] / played).round(2)
    out["Goals For / Match"] = (out["GF"] / played).round(2)
    out["Goals Against / Match"] = (out["GA"] / played).round(2)
    out["Goal Difference / Match"] = (out["GD"] / played).round(2)
    out["Win %"] = (out["Wins"] / played * 100).round(1)
    out["Losses / Match"] = (out["Losses"] / played).round(2)
    out["Loss %"] = (out["Losses"] / played * 100).round(1)
    clean_sheets = pd.to_numeric(out.get("Clean Sheets", pd.Series(np.nan, index=out.index)), errors="coerce")
    scoring_matches = pd.to_numeric(out.get("Scoring Matches", pd.Series(np.nan, index=out.index)), errors="coerce")
    out["Clean Sheet %"] = (clean_sheets / played * 100).round(1)
    out["Scoring Match %"] = (scoring_matches / played * 100).round(1)
    out["Defence Score"] = ta.percentile(out["Goals Against / Match"], higher_is_better=False)
    out["Avoiding Defeat"] = ta.percentile(out["Losses / Match"], higher_is_better=False)

    rank_rules = {
        "Points / Match": False,
        "Goals For / Match": False,
        "Goals Against / Match": True,
        "Goal Difference / Match": False,
        "Win %": False,
        "Loss %": True,
        "Clean Sheet %": False,
        "Scoring Match %": False,
        "Defence Score": False,
        "Avoiding Defeat": False,
    }
    for metric, ascending in rank_rules.items():
        out[f"{metric} Rank"] = pd.to_numeric(out[metric], errors="coerce").rank(
            ascending=ascending,
            method="min",
        ).astype("Int64")

    return out


def _performance_scores(team_metrics: pd.DataFrame) -> pd.DataFrame:
    if team_metrics.empty:
        return pd.DataFrame(columns=["Team", "Results", "Attack", "Defence", "Goal Difference", "Win Rate", "Avoiding Defeat"])

    out = pd.DataFrame({"Team": team_metrics["Team"]})
    out["Results"] = ta.percentile(team_metrics["Points / Match"])
    out["Attack"] = ta.percentile(team_metrics["Goals For / Match"])
    out["Defence"] = ta.percentile(team_metrics["Goals Against / Match"], higher_is_better=False)
    out["Goal Difference"] = ta.percentile(team_metrics["Goal Difference / Match"])
    out["Win Rate"] = ta.percentile(team_metrics["Win %"])
    out["Avoiding Defeat"] = ta.percentile(team_metrics["Losses / Match"], higher_is_better=False)
    return out


def _performance_strengths(performance: pd.DataFrame, team_name: str) -> list[str]:
    if performance.empty or "Team" not in performance:
        return []
    performance_team = _best_label(performance["Team"], team_name)
    rows = performance[performance["Team"].astype(str) == str(performance_team)]
    if rows.empty:
        return []
    row = rows.iloc[0]
    fields = [field for field in ["Results", "Attack", "Defence", "Goal Difference", "Win Rate", "Avoiding Defeat"] if field in row]
    ranked = sorted(fields, key=lambda field: float(row.get(field, 0) or 0), reverse=True)
    return [f"{field} ({_fmt(row.get(field), 0)}th pct)" for field in ranked[:3]]


def _radar_key() -> None:
    items = [
        ("Results", "Points per match percentile"),
        ("Attack", "Goals for per match percentile"),
        ("Defence", "Goals against per match percentile, inverted so fewer conceded is better"),
        ("Goal Difference", "Goal difference per match percentile"),
        ("Win Rate", "Win percentage percentile"),
        ("Avoiding Defeat", "Losses per match percentile, inverted so fewer losses is better"),
    ]
    html = []
    for label, metric in items:
        html.append(
            '<div class="to-radar-key-item">'
            f'<div class="to-radar-key-label">{ui.esc(label)}</div>'
            f'<div class="to-radar-key-metric">{ui.esc(metric)}</div>'
            '</div>'
        )

    st.markdown(
        (
            '<div class="to-radar-key">'
            '<div class="to-radar-key-title">Performance Profile Key</div>'
            '<div class="to-radar-key-note">'
            'Radar values are percentiles from completed match results only. Higher means stronger relative performance.'
            '</div>'
            f'<div class="to-radar-key-grid">{"".join(html)}</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


def _player_subset(players: pd.DataFrame, team_name: str) -> pd.DataFrame:
    if players.empty or "Team" not in players:
        return pd.DataFrame()
    label = _best_label(players["Team"], team_name)
    return players[players["Team"].astype(str) == str(label)].copy()


def _squad_stats(squad: pd.DataFrame) -> dict[str, str]:
    if squad.empty:
        return {"size": "—", "age": "—", "nations": "—", "minutes": "—"}
    stats = {"size": _fmt(len(squad), 0)}
    if "Birthdate" in squad:
        ages = (pd.Timestamp(date.today()) - pd.to_datetime(squad["Birthdate"], errors="coerce")).dt.days / 365.25
        stats["age"] = _fmt(ages.mean(), 1)
    else:
        stats["age"] = "—"
    stats["nations"] = _fmt(squad["Nationality"].dropna().nunique(), 0) if "Nationality" in squad else "—"
    stats["minutes"] = _fmt(pd.to_numeric(squad["Minutes"], errors="coerce").fillna(0).sum(), 0) if "Minutes" in squad else "—"
    return stats


def _position_counts(squad: pd.DataFrame) -> pd.DataFrame:
    if squad.empty or "Position" not in squad:
        return pd.DataFrame(columns=["Position group", "Players"])

    positions = squad["Position"].dropna().astype(str).str.strip()
    positions = positions[~positions.str.lower().isin(["", "nan", "none", "null"])]
    if positions.empty:
        return pd.DataFrame(columns=["Position group", "Players"])

    return (
        positions.apply(pa.position_group)
        .value_counts()
        .rename_axis("Position group")
        .reset_index(name="Players")
        .sort_values(["Players", "Position group"], ascending=[False, True])
        .reset_index(drop=True)
    )


def _player_cards(players: pd.DataFrame, metric: str) -> None:
    if players.empty:
        st.info("No player rows are available for this team and player season.")
        return

    html = []
    for _, player in players.iterrows():
        name = ui.esc(player.get("Player", "Unknown player"))
        position = ui.esc(ui.clean_position(player.get("Position")))
        minutes = _fmt(player.get("Minutes"), 0)
        value = ui.esc(_metric_text(metric, player.get(metric)))
        metric_label = ui.esc(metric)

        html.append(
            '<div class="to-player-card">'
            f'<div class="to-player-name">{name}</div>'
            f'<div class="to-muted">{position} · {minutes} mins</div>'
            f'<div class="to-player-stat">{value}</div>'
            f'<div class="to-muted">{metric_label}</div>'
            '</div>'
        )

    st.markdown(
        f'<div class="to-player-grid">{"".join(html)}</div>',
        unsafe_allow_html=True,
    )


def _top_players(squad: pd.DataFrame, metric: str, count: int = 3) -> pd.DataFrame:
    if squad.empty or metric not in squad:
        return pd.DataFrame()
    out = squad.copy()
    out[metric] = pd.to_numeric(out[metric], errors="coerce")
    return out.dropna(subset=[metric]).sort_values(metric, ascending=False).head(count)


def _opta_team_fixtures(fixtures: pd.DataFrame, team_name: str) -> pd.DataFrame:
    """Return only exact normalised team matches; never fuzzy-join provider fixtures."""
    if fixtures.empty or not {"Home", "Away"}.issubset(fixtures.columns):
        return pd.DataFrame(columns=fixtures.columns)
    target = _normalise_name(team_name)
    home = fixtures["Home"].fillna("").map(_normalise_name).eq(target)
    away = fixtures["Away"].fillna("").map(_normalise_name).eq(target)
    rows = fixtures[home | away].copy()
    if "Date" in rows:
        rows["Date"] = pd.to_datetime(rows["Date"], errors="coerce")
        rows = rows.sort_values(["Date", "FixtureId"], na_position="last")
    return rows.reset_index(drop=True)


def _opta_fixture_label(row: pd.Series) -> str:
    date_value = pd.to_datetime(row.get("Date"), errors="coerce")
    date_text = date_value.strftime("%d %b %Y") if pd.notna(date_value) else "Undated"
    score = ""
    if pd.notna(row.get("Home Goals")) and pd.notna(row.get("Away Goals")):
        score = f" · {_fmt(row.get('Home Goals'), 0)}-{_fmt(row.get('Away Goals'), 0)}"
    return f"{date_text} · {row.get('Home', 'Unknown')} vs {row.get('Away', 'Unknown')}{score}"


def _format_formation(value: object) -> str:
    if value is None or pd.isna(value):
        return "Shape unavailable"
    digits = re.sub(r"\D", "", str(value))
    return "-".join(digits) if 3 <= len(digits) <= 5 else str(value)


def _is_true(value: object) -> bool:
    return value is True or str(value).strip().casefold() in {"1", "true", "yes"}


def _lineup_rows_for_team(lineups: pd.DataFrame, team_id: object, team_name: str) -> pd.DataFrame:
    if lineups.empty:
        return lineups
    rows = pd.DataFrame()
    if "TeamId" in lineups and pd.notna(team_id):
        rows = lineups[lineups["TeamId"].fillna("").astype(str).eq(str(team_id))].copy()
    if rows.empty and "Team" in lineups:
        target = _normalise_name(team_name)
        rows = lineups[lineups["Team"].fillna("").map(_normalise_name).eq(target)].copy()
    return rows


def _lineup_table(rows: pd.DataFrame, status: str) -> pd.DataFrame:
    columns = ["#", "Player", "Position"]
    if rows.empty or "Lineup Status" not in rows:
        return pd.DataFrame(columns=columns)
    selected = rows[
        rows["Lineup Status"].fillna("").astype(str).str.casefold().eq(status.casefold())
    ].copy()
    if selected.empty:
        return pd.DataFrame(columns=columns)
    selected["_order"] = pd.to_numeric(selected.get("Formation Place"), errors="coerce").fillna(999)
    selected = selected.sort_values(["_order", "Player"], na_position="last")

    shirt_numbers = pd.to_numeric(selected.get("Shirt Number"), errors="coerce").astype("Int64")
    players = selected.get("Player", pd.Series("Unknown player", index=selected.index)).fillna("Unknown player").astype(str)
    if "Is Captain" in selected:
        players = pd.Series(
            [f"{name} (C)" if _is_true(captain) else name for name, captain in zip(players, selected["Is Captain"])],
            index=selected.index,
        )

    primary_position = "Sub Position" if status.casefold() == "sub" else "Position Group"
    positions = selected.get(primary_position, pd.Series(index=selected.index, dtype=object)).copy()
    for fallback in ["Registered Position", "Position Group"]:
        if fallback in selected:
            missing = positions.isna() | positions.astype(str).str.strip().eq("")
            positions = positions.where(~missing, selected[fallback])

    return pd.DataFrame(
        {
            "#": shirt_numbers.astype(str).replace("<NA>", "—").tolist(),
            "Player": players.tolist(),
            "Position": positions.fillna("—").astype(str).tolist(),
        }
    )


def _formation_for_team(formations: pd.DataFrame, team_id: object, side: str) -> pd.Series:
    if formations.empty:
        return pd.Series(dtype=object)
    rows = pd.DataFrame()
    if "TeamId" in formations and pd.notna(team_id):
        rows = formations[formations["TeamId"].fillna("").astype(str).eq(str(team_id))]
    if rows.empty and "Side" in formations:
        rows = formations[formations["Side"].fillna("").astype(str).str.casefold().eq(side.casefold())]
    return rows.iloc[0] if not rows.empty else pd.Series(dtype=object)


def _render_fixture_header(fixture: pd.Series) -> None:
    score = "—"
    if pd.notna(fixture.get("Home Goals")) and pd.notna(fixture.get("Away Goals")):
        score = f"{_fmt(fixture.get('Home Goals'), 0)}-{_fmt(fixture.get('Away Goals'), 0)}"
    meta = " · ".join(
        value
        for value in [
            _date_text(fixture.get("Date")),
            str(fixture.get("Venue")) if pd.notna(fixture.get("Venue")) else "",
            f"Round {fixture.get('Round')}" if pd.notna(fixture.get("Round")) else "",
        ]
        if value
    )
    st.markdown(
        f"""
        <div class="to-fixture-detail">
            <div class="to-fixture-detail-top">
                <div>
                    <div class="to-fixture-detail-title">{ui.esc(fixture.get('Home'))} vs {ui.esc(fixture.get('Away'))}</div>
                    <div class="to-fixture-detail-meta">{ui.esc(meta)}</div>
                </div>
                <div class="to-fixture-score">{ui.esc(score)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_effective_possession(possession: pd.DataFrame, team_name: str) -> None:
    if possession.empty:
        st.info("Effective-play possession is unavailable for this fixture.")
        return
    row = possession.iloc[0]
    home_pct = float(row.get("Home Possession %"))
    away_pct = float(row.get("Away Possession %"))
    home_pct = min(max(home_pct, 0.0), 100.0)
    away_pct = min(max(away_pct, 0.0), 100.0)
    selected = _normalise_name(team_name)
    home_colour = "#c30017" if _normalise_name(row.get("Home")) == selected else "#344054"
    away_colour = "#c30017" if _normalise_name(row.get("Away")) == selected else "#344054"
    st.markdown(
        f"""
        <div class="to-possession-card">
            <div class="to-possession-title">Effective-play possession</div>
            <div class="to-possession-values">
                <div class="to-possession-team">{ui.esc(row.get('Home'))}<span class="to-possession-number">{home_pct:.1f}%</span></div>
                <div class="to-possession-team">{ui.esc(row.get('Away'))}<span class="to-possession-number">{away_pct:.1f}%</span></div>
            </div>
            <div class="to-possession-track" aria-label="Effective-play possession comparison">
                <span style="width:{home_pct:.1f}%;background:{home_colour}"></span>
                <span style="width:{away_pct:.1f}%;background:{away_colour}"></span>
            </div>
            <div class="to-possession-times">
                <span>Home EPT · {ui.esc(row.get('Home EPT') or '—')}</span>
                <span>Away EPT · {ui.esc(row.get('Away EPT') or '—')}</span>
            </div>
            <div class="to-possession-source">Second Spectrum tracking · {ui.esc(row.get('Effective Playing Time') or '—')} effective playing time. Percentages are calculated from the provider-delivered home and away EPT durations; this is not an F24 pass-share estimate.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_official_lineups(lineups: pd.DataFrame, formations: pd.DataFrame, fixture: pd.Series) -> None:
    if lineups.empty:
        st.info("Submitted Opta F7 lineups are unavailable for this fixture.")
        return
    team_specs = [
        (fixture.get("Home Team Id"), str(fixture.get("Home")), "Home"),
        (fixture.get("Away Team Id"), str(fixture.get("Away")), "Away"),
    ]
    columns = st.columns(2)
    for column, (team_id, team_label, side) in zip(columns, team_specs):
        team_rows = _lineup_rows_for_team(lineups, team_id, team_label)
        starters = _lineup_table(team_rows, "Start")
        substitutes = _lineup_table(team_rows, "Sub")
        formation = _formation_for_team(formations, team_id, side)
        shape = _format_formation(formation.get("Formation"))
        average_age = formation.get("Average Age")
        age_text = f" · average age {_fmt(average_age, 1)}" if pd.notna(average_age) else ""
        with column:
            st.markdown(f'<div class="to-lineup-heading">{ui.esc(team_label)}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="to-lineup-shape">Submitted XI · {ui.esc(shape)}{ui.esc(age_text)}</div>',
                unsafe_allow_html=True,
            )
            if starters.empty:
                st.info("Starting XI unavailable.")
            else:
                lineup_tab, pitch_tab = st.tabs(["List", "Pitch"])
                with lineup_tab:
                    st.dataframe(starters, width="stretch", hide_index=True, height=424)
                with pitch_tab:
                    # Attempt to get actual formation string for better mapping
                    team_form = None
                    if not formations.empty:
                        side_key = "Home" if side.lower() == "home" else "Away"
                        team_form_row = formations[formations["Side"].str.casefold() == side_key.casefold()]
                        if not team_form_row.empty:
                            team_form = team_form_row.iloc[0].get("Formation")
                    
                    st.plotly_chart(
                        pitch.formation_map(
                            team_rows, 
                            team_label, 
                            f"{team_label} Formation", 
                            formation=team_form,
                            mirror=(side.lower() == "away"),
                            marker_color=ui.get_team_color(team_label)
                        ),
                        width="stretch",
                        key=f"team_overview_formation_{side.lower()}"
                    )
            with st.expander(f"Bench · {len(substitutes)} players"):
                if substitutes.empty:
                    st.caption("No substitutes were supplied in the F7 feed.")
                else:
                    st.dataframe(substitutes, width="stretch", hide_index=True)


def _default_index(options: list[str], preferred: str | None = None) -> int:
    if not options:
        return 0
    if preferred in options:
        return options.index(preferred)
    return len(options) - 1


def _render_hero(
    team_name: str,
    team_season: str | None,
    match_season: str | None,
    competition: str,
    league_position: str,
    record: dict[str, float | int],
    strengths: list[str],
    team_matches: pd.DataFrame,
) -> None:
    meta = [
        f"Team metrics: {team_season or 'all seasons'}",
        f"Matches: {match_season or 'all seasons'}",
        competition,
        league_position,
    ]
    meta_html = "".join(f'<span class="to-meta-pill">{ui.esc(item)}</span>' for item in meta if item)
    strength_text = " · ".join(strengths) if strengths else "Strengths depend on available team metrics"
    st.markdown(
        f"""
        <div class="to-hero">
            <div class="to-hero-main">
                <div class="to-team-block">
                    {_badge_html(team_name)}
                    <div>
                        <div class="to-eyebrow">Team overview</div>
                        <h1 class="to-title">{ui.esc(team_name)}</h1>
                        <div class="to-meta-row">{meta_html}</div>
                    </div>
                </div>
                <div class="to-side">
                    <div class="to-side-label">Recent form</div>
                    <div class="to-pill-row">{_form_html(team_matches)}</div>
                    <div class="to-side-text">
                        {_fmt(record["played"], 0)} played · {_fmt(record["points"], 0)} pts · {_fmt(record["ppg"], 2)} PPG<br>
                        {ui.esc(strength_text)}
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


ui.apply_statsearch_theme()
_inject_css()

season_options = data.list_seasons()
match_seasons = season_options.get("matches", [])
player_seasons = season_options.get("players", [])

with st.expander("Data controls", expanded=True):
    st.markdown(
        '<div class="to-note">Select the match season and team. Team overview metrics are derived from completed match rows, not aggregate-average team tables.</div>',
        unsafe_allow_html=True,
    )
    control_cols = st.columns([1.15, 1.15, 1.7])

    match_season = (
        control_cols[0].selectbox("Match season", match_seasons, index=_default_index(match_seasons), key="team_overview_match_season")
        if match_seasons
        else None
    )
    player_season = (
        control_cols[1].selectbox("Player season", player_seasons, index=_default_index(player_seasons, match_season), key="team_overview_player_season")
        if player_seasons
        else None
    )

    matches = ma.load_matches(match_season) if match_season else pd.DataFrame()
    table = _standings(matches)
    if matches.empty:
        st.warning("No match data is available for this season.")
        st.stop()

    team_names = table["Team"].dropna().astype(str).tolist() if not table.empty else _match_labels(matches).dropna().astype(str).drop_duplicates().tolist()
    charlton_matches = [index for index, name in enumerate(team_names) if "charlton" in _normalise_name(name)]
    default_team = charlton_matches[0] if charlton_matches else 0
    team_name = control_cols[2].selectbox("Team", team_names, index=default_team, key="team_overview_team")

team_metrics = _match_metric_table(table)
metric_row_label = _best_label(team_metrics["Team"], team_name) if not team_metrics.empty and "Team" in team_metrics else team_name
metric_rows = team_metrics[team_metrics["Team"].astype(str) == str(metric_row_label)] if not team_metrics.empty else pd.DataFrame()
row = metric_rows.iloc[0] if not metric_rows.empty else pd.Series(dtype=object)
performance = _performance_scores(team_metrics)
team_match_label = _best_label(_match_labels(matches), team_name)
team_matches = _add_venue(ma.team_match_rows(matches, team_match_label), team_match_label) if not matches.empty else pd.DataFrame()
table_label = _best_label(table["Team"], team_name) if not table.empty and "Team" in table else team_name
table_row = table[table["Team"].astype(str) == str(table_label)] if not table.empty else pd.DataFrame()
league_position = f"#{int(table_row.iloc[0]['#'])} in table" if not table_row.empty else "Table position unavailable"
record = _record(team_matches)
previous_match, next_match = _fixture_split(team_matches)
competition = _competition(matches, team_matches)
strengths = _performance_strengths(performance, team_name)

_render_hero(team_name, "match-derived", match_season, competition, league_position, record, strengths, team_matches)

nav_cols = st.columns(6)
with nav_cols[0]:
    st.page_link("views/team_overview.py", label="Overview")
with nav_cols[1]:
    st.page_link("views/match_overview.py", label="Fixtures")
with nav_cols[2]:
    st.page_link("views/player_search.py", label="Squad")
with nav_cols[3]:
    st.page_link("views/player_data_table.py", label="Player stats")
with nav_cols[4]:
    st.page_link("views/team_data_table.py", label="Team stats")
with nav_cols[5]:
    st.page_link("views/league_rankings.py", label="League rankings")

_section("Featured")
featured_cols = st.columns(3)
with featured_cols[0]:
    _fixture_card("Next Match", next_match, "No scheduled future fixture is available in the selected match source.")

with featured_cols[1]:
    _fixture_card("Previous Match", previous_match, "No previous fixture is available in the selected match source.")

with featured_cols[2]:
    _form_summary_card(team_matches)

recent = _completed_team_matches(team_matches).tail(5) if not team_matches.empty else pd.DataFrame()
if not recent.empty and "Date" in recent:
    recent = recent.sort_values("Date", ascending=False)

with st.container(border=True):
    st.subheader("Recent Matches")
    st.caption("Latest completed fixtures from the selected match dataset.")
    if recent.empty:
        st.info("No completed fixtures available for recent-match display.")
    else:
        cols = [col for col in ["Date", "Opponent", "Venue", "Goals For", "Goals Against", "Team Result", "Points"] if col in recent]
        st.dataframe(recent[cols], width="stretch", hide_index=True)

_section("Fixture detail")
st.caption(
    "Choose a completed Opta fixture to compare effective-play possession and inspect the two submitted starting lineups."
)
try:
    opta_fixtures = data.load_opta_fixtures(match_season)
except Exception as exc:
    st.warning(f"Opta fixtures could not be loaded: {exc}")
    opta_fixtures = pd.DataFrame()

opta_team_matches = _completed_fixture_rows(_opta_team_fixtures(opta_fixtures, team_name))
if opta_team_matches.empty:
    st.info("No completed Opta fixtures match this team and season, so fixture detail is unavailable.")
else:
    fixture_rows = opta_team_matches.set_index("FixtureId", drop=False)
    fixture_options = fixture_rows.index.astype(str).tolist()
    selected_fixture_id = st.selectbox(
        "Fixture",
        fixture_options,
        index=len(fixture_options) - 1,
        format_func=lambda fixture_id: _opta_fixture_label(fixture_rows.loc[fixture_id]),
        key="team_overview_opta_fixture",
    )
    selected_opta_fixture = fixture_rows.loc[selected_fixture_id]
    fixture_id = selected_fixture_id
    _render_fixture_header(selected_opta_fixture)

    try:
        possession = data.load_fixture_effective_possession(fixture_id)
    except Exception as exc:
        st.caption(f"Effective-play possession could not be loaded: {exc}")
        possession = pd.DataFrame(columns=getattr(data, "TRACKING_POSSESSION_COLUMNS", []))
    _render_effective_possession(possession, team_name)

    st.subheader("Official submitted lineups")
    st.caption(
        "Starting XI, shirt number, captain and team shape are taken from the Opta F7 submission for this fixture."
    )
    try:
        lineups = data.load_opta_lineups(fixture_id)
    except Exception as exc:
        st.caption(f"Opta F7 lineups could not be loaded: {exc}")
        lineups = pd.DataFrame(columns=getattr(data, "OPTA_LINEUP_COLUMNS", []))
    try:
        formations = data.load_opta_formations(fixture_id)
    except Exception as exc:
        st.caption(f"Opta F7 formations could not be loaded: {exc}")
        formations = pd.DataFrame(columns=getattr(data, "OPTA_FORMATION_COLUMNS", []))
    _render_official_lineups(lineups, formations, selected_opta_fixture)

_section("Season snapshot")
cards = [
    {"label": "Record", "value": f'{_fmt(record["wins"], 0)}W {_fmt(record["draws"], 0)}D {_fmt(record["losses"], 0)}L', "sub": f'{_fmt(record["played"], 0)} matches played'},
    {"label": "Points", "value": _fmt(record["points"], 0), "sub": f'{_fmt(record["ppg"], 2)} points per match'},
    {"label": "Goals", "value": f'{_fmt(record["gf"], 0)}-{_fmt(record["ga"], 0)}', "sub": f'{_fmt(record["gd"], 0)} goal difference'},
    {"label": "League position", "value": league_position.replace(" in table", ""), "sub": competition},
]
for metric in MATCH_SNAPSHOT_METRICS:
    if metric in row:
        cards.append({"label": metric, "value": _metric_text(metric, row.get(metric)), "sub": f'Rank {_fmt(row.get(f"{metric} Rank"), 0)} of {len(team_metrics)}'})
_cards(cards, class_name="to-card-grid to-card-grid--snapshot")

_section("Performance profile")
try:
    performance_team = _best_label(performance["Team"], team_name)
    selected_performance = performance[performance["Team"].astype(str) == str(performance_team)].iloc[0]

    labels = [
        label
        for label in ["Results", "Attack", "Defence", "Goal Difference", "Win Rate", "Avoiding Defeat"]
        if label in selected_performance and pd.notna(selected_performance[label])
    ]
    values = [float(selected_performance[label]) for label in labels]

    st.plotly_chart(ta.team_radar(labels, values, team_name), width="stretch")
    _radar_key()
except Exception:
    st.info("Performance profile is unavailable for the selected team metrics.")

_section("Table context")
if table.empty:
    st.info("No league table can be built from the selected match source.")
else:
    st.caption("Table is computed from completed regular-season match rows, then trimmed around the selected team.")
    context = _surrounding_table(table, team_name)
    cols = [col for col in ["#", "Team", "Played", "Wins", "Draws", "Losses", "GF", "GA", "GD", "Points", "Form"] if col in context]
    st.dataframe(context[cols], width="stretch", hide_index=True)

_section("Squad and key players")
try:
    players = ta.load_player_data(player_season)
except Exception as exc:
    st.warning(f"Could not load player data: {exc}")
    players = pd.DataFrame()
squad = _player_subset(players, team_name)
squad_stats = _squad_stats(squad)
_cards(
    [
        {"label": "Squad size", "value": squad_stats["size"], "sub": "Players in selected player season"},
        {"label": "Average age", "value": squad_stats["age"], "sub": "Calculated from birthdates"},
        {"label": "Nationalities", "value": squad_stats["nations"], "sub": "Distinct player countries"},
        {"label": "Total minutes", "value": squad_stats["minutes"], "sub": "Player metric source minutes"},
    ]
)

squad_cols = st.columns([1, 1.35])
with squad_cols[0]:
    positions = _position_counts(squad)
    if positions.empty:
        st.info("Position breakdown is unavailable.")
    else:
        st.dataframe(positions, width="stretch", hide_index=True)

with squad_cols[1]:
    player_metric_options = [metric for metric in data.PLAYER_PROFILE_METRICS if metric in players.columns]
    if player_metric_options:
        default_metric = "Goals /90" if "Goals /90" in player_metric_options else player_metric_options[0]
        player_metric = st.selectbox("Top players by", player_metric_options, index=player_metric_options.index(default_metric), key="team_overview_player_metric")
        _player_cards(_top_players(squad, player_metric), player_metric)
    else:
        st.info("No player metric columns are available.")

_section("About and data coverage")
record_text = (
    f'{_fmt(record["wins"], 0)} wins, {_fmt(record["draws"], 0)} draws and {_fmt(record["losses"], 0)} losses'
    if record["played"]
    else "no completed-match record in the selected match source"
)
strength_text = ", ".join(strengths) if strengths else "the currently available team metrics"
st.markdown(
    f"""
    <div class="to-about">
        <p><strong>{ui.esc(team_name)}</strong> overview is structured around the same core objects users expect from FotMob and Sofascore:
        recent form, next/previous match, fixture possession, official lineups, table context, season stats, squad profile and key players.</p>
        <p>For the selected data context, the team has {ui.esc(record_text)}. Its strongest internal profile areas are {ui.esc(strength_text)}.</p>
        <p><strong>Data basis:</strong> {ui.esc(PAGE_SOURCE)}</p>
    </div>
    """,
    unsafe_allow_html=True,
)
