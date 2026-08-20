from __future__ import annotations

import base64
import mimetypes

from utils import ui


TEAM_BADGE_DIR = ui.ASSETS_DIR / "team_badges"

TEAM_BADGE_FILES = {
    "barnsley": "Barnsley_FC.svg.png",
    "barnsley fc": "Barnsley_FC.svg.png",

    "birmingham": "Birmingham-City.png",
    "birmingham city": "Birmingham-City.png",

    "blackburn": "Blackburn_Rovers.svg.png",
    "blackburn rovers": "Blackburn_Rovers.svg.png",

    "blackpool": "Blackpool_FC_logo.svg.png",
    "blackpool fc": "Blackpool_FC_logo.svg.png",

    "bolton": "Bolton_Wanderers_FC_logo.svg.webp",
    "bolton wanderers": "Bolton_Wanderers_FC_logo.svg.webp",

    "bristol": "Bristol.png",
    "bristol city": "Bristol_City_crest.svg.webp",
    "bristol rovers": "Bristol_Rovers_F.C._logo.svg.png",

    "bromley": "Bromley_FC_crest.svg.png",
    "bromley fc": "Bromley_FC_crest.svg.png",

    "burnley": "Burnley_FC_Logo.svg.webp",
    "burton": "Burton_Albion_FC_logo.svg.png",
    "burton albion": "Burton_Albion_FC_logo.svg.png",

    "cambridge": "Cambridge_United_FC.svg.png",
    "cambridge united": "Cambridge_United_FC.svg.png",

    "cardiff": "Cardiff_City_crest.svg",
    "cardiff city": "Cardiff_City_crest.svg",

    "charlton": "Charlton Logo.png",
    "charlton athletic": "Charlton Logo.png",

    "chelsea": "Chelsea_FC.svg.png",
    "chelsea fc": "Chelsea_FC.svg.png",

    "coventry": "Coventry_City_FC_crest.svg.png",
    "coventry city": "Coventry_City_FC_crest.svg.png",

    "crawley": "Crawley_Town_FC_crest.svg.png",
    "crawley town": "Crawley_Town_FC_crest.svg.png",

    "derby": "Derby_County_crest.svg.png",
    "derby county": "Derby_County_crest.svg.png",

    "exeter": "Exeter_City_FC.svg.png",
    "exeter city": "Exeter_City_FC.svg.png",

    "huddersfield": "Huddersfield_Town_AFC_crest.svg.png",
    "huddersfield town": "Huddersfield_Town_AFC_crest.svg.png",

    "hull": "Hull_City_A.F.C._logo.svg.png",
    "hull city": "Hull_City_A.F.C._logo.svg.png",

    "ipswich": "Ipswich_Town.svg.png",
    "ipswich town": "Ipswich_Town.svg.png",

    "leicester": "Leicester_City_crest.svg.png",
    "leicester city": "Leicester_City_crest.svg.png",

    "leyton orient": "Leyton_Orient_F.C._logo.svg.png",

    "lincoln": "Lincoln_City_FC_2024_crest.svg.png",
    "lincoln city": "Lincoln_City_FC_2024_crest.svg.png",

    "luton": "Luton.png",
    "luton town": "Luton.png",

    "mansfield": "mansfield-town-fc-logo-E2BCF556D5-seeklogo.com.png",
    "mansfield town": "mansfield-town-fc-logo-E2BCF556D5-seeklogo.com.png",

    "middlesbrough": "Middlesbrough_FC_crest.svg.png",
    "fc middlesbrough": "Middlesbrough_FC_crest.svg.png",

    "millwall": "Millwall_FC_crest.svg.png",
    "millwall fc": "Millwall_FC_crest.svg.png",

    "northampton": "Northampton_Town_F.C._logo.svg.png",
    "northampton town": "Northampton_Town_F.C._logo.svg.png",

    "norwich": "Norwich_City.png",
    "norwich city": "Norwich_City.png",

    "oxford": "Oxford_United_FC_logo.svg.png",
    "oxford united": "Oxford_United_FC_logo.svg.png",

    "peterborough": "Peterborough_United.svg.png",
    "peterborough united": "Peterborough_United.svg.png",

    "plymouth": "Plymouth.jpg",
    "plymouth argyle": "Plymouth.jpg",

    "portsmouth": "Portsmouth_FC_logo.svg.png",
    "portsmouth fc": "Portsmouth_FC_logo.svg.png",

    "preston": "Preston_North_End_FC.svg.png",
    "preston north end": "Preston_North_End_FC.svg.png",

    "qpr": "Queens_Park_Rangers_crest.svg.png",
    "queens park rangers": "Queens_Park_Rangers_crest.svg.png",

    "reading": "Reading_FC.svg.png",
    "reading fc": "Reading_FC.svg.png",

    "rotherham": "Rotherham_United_FC.svg.png",
    "rotherham united": "Rotherham_United_FC.svg.png",

    "sheffield united": "Sheffield_United_FC_logo.svg.png",
    "sheffield utd": "Sheffield_United_FC_logo.svg.png",

    "sheffield wednesday": "Sheffield_Wednesday_badge.svg.png",

    "shrewsbury": "Shrewsbury_Town_F.C._logo.svg.png",
    "shrewsbury town": "Shrewsbury_Town_F.C._logo.svg.png",

    "southampton": "FC_Southampton.svg.png",
    "fc southampton": "FC_Southampton.svg.png",

    "stevenage": "Stevenage_FC_crest.svg.png",
    "stevenage fc": "Stevenage_FC_crest.svg.png",

    "stockport": "Stockport_County_FC_logo_2020.svg.png",
    "stockport county": "Stockport_County_FC_logo_2020.svg.png",

    "stoke": "Stoke_City_FC.svg.png",
    "stoke city": "Stoke_City_FC.svg.png",

    "swansea": "Swansea_City_A.F.C._logo.png",
    "swansea city": "Swansea_City_A.F.C._logo.png",

    "watford": "Watford.svg.png",
    "watford fc": "Watford.svg.png",

    "west brom": "West_Bromwich_Albion.svg.png",
    "west bromwich albion": "West_Bromwich_Albion.svg.png",

    "west ham": "West_Ham_United_FC_logo.svg",
    "west ham united": "West_Ham_United_FC_logo.svg",

    "wigan": "Wigan_Athletic.svg.png",
    "wigan athletic": "Wigan_Athletic.svg.png",

    "wolverhampton": "Wolverhampton_Wanderers_FC_crest.svg.webp",
    "wolverhampton wanderers": "Wolverhampton_Wanderers_FC_crest.svg.webp",
    "wolves": "Wolverhampton_Wanderers_FC_crest.svg.webp",

    "wrexham": "Wrexham_A.F.C._Logo.svg.png",
    "afc wrexham": "Wrexham_A.F.C._Logo.svg.png",

    "wycombe": "Wycombe_Wanderers_FC_logo.svg.png",
    "wycombe wanderers": "Wycombe_Wanderers_FC_logo.svg.png",
}

def _team_key(team_name: object) -> str:
    text = str(team_name).strip().lower()
    text = text.replace(".", "").replace("&", "and")
    text = " ".join(text.split())

    for prefix in ["fc ", "afc "]:
        if text.startswith(prefix):
            text = text[len(prefix):]

    for suffix in [" fc", " afc", " football club"]:
        if text.endswith(suffix):
            text = text[: -len(suffix)]

    return text.strip()


def badge_path(team_name: object):
    team_key = _team_key(team_name)

    badge_file = TEAM_BADGE_FILES.get(team_key)
    if not badge_file:
        return None

    path = TEAM_BADGE_DIR / badge_file
    if path.exists():
        return path

    return None


def badge_data_uri(team_name: object) -> str | None:
    path = badge_path(team_name)
    if path is None:
        return None

    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
