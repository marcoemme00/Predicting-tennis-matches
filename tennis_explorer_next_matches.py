"""
Downloads from Tennis Explorer the ATP matches still to be played and writes them
in the tennis-data format (upcoming_matches.xlsx), ready for predict_upcoming_matches().
"""
import argparse
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = "https://www.tennisexplorer.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"}
TENNIS_DATA_COLUMNS = [
    "ATP", "Location", "Tournament", "Date", "Series", "Court", "Surface",
    "Round", "Best of", "Winner", "Loser", "WRank", "LRank", "WPts", "LPts",
    "W1", "L1", "W2", "L2", "W3", "L3", "W4", "L4", "W5", "L5", "Wsets", "Lsets",
    "Comment", "B365W", "B365L", "PSW", "PSL", "MaxW", "MaxL", "AvgW", "AvgL",
]
NOT_MAIN_TOUR = ("challenger", "utr", "futures", "itf")
SURFACES = {"hard": "Hard", "clay": "Clay", "grass": "Grass", "carpet": "Carpet"}
COURTS = {"indoor": "Indoor", "indoors": "Indoor", "outdoor": "Outdoor", "outdoors": "Outdoor"}
ROUNDS = {
    "quarterfinal": "Quarterfinals", "quarterfinals": "Quarterfinals",
    "semifinal": "Semifinals", "semifinals": "Semifinals",
    "final": "The Final", "the final": "The Final", "round robin": "Round Robin",
    "1st round": "1st Round", "2nd round": "2nd Round", "3rd round": "3rd Round", "4th round": "4th Round",
}


def get_soup(url):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    time.sleep(0.35)  # be gentle with the site :-)
    return BeautifulSoup(r.text, "html.parser")


def to_number(text):
    """'13.' -> 13, '1 070' -> 1070, 'info' -> nan."""
    return pd.to_numeric(text.replace(" ", "").rstrip("."), errors="coerce")


def first_number(text):
    m = re.match(r"\d+(?:\.\d+)?", text.replace(",", "."))
    return float(m[0]) if m else float("nan")


def clean_name(name):
    name = re.sub(r"\(\d+\)", "", name)  # seed
    name = re.sub(r"Live.*", "", name, flags=re.I)
    return re.sub(r"\s+", " ", name).strip()


def player_key(name):
    """Same key for 'Zverev A.' (match list) and 'Zverev Alexander' (ranking): surname + initial."""
    parts = re.sub(r"[.\-]", "", clean_name(name)).split()
    if len(parts) < 2:
        return "".join(parts).lower()
    return ("".join(parts[:-1]) + parts[-1][0]).lower()


def is_main_tour(tournament, href):
    return "atp-men" in href and not any(x in tournament.lower() for x in NOT_MAIN_TOUR)

####################################
# 1) list of the day's matches
####################################

def parse_day(soup, day):
    """Each match takes two rows: the first has time, player A, odds; the second player B."""
    matches = []
    tournament = href = ""
    for tr in soup.select("table.result tr"):
        cls = tr.get("class") or []
        tds = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
        if "head" in cls and "flags" in cls:  # tournament header row
            a = tr.find("a")
            tournament, href = (a.get_text(strip=True), a.get("href", "")) if a else (tds[0], "")
            continue
        second_tr = tr.find_next_sibling("tr")
        if "bott" not in cls or len(tds) < 4 or second_tr is None:
            continue
        second = [td.get_text(" ", strip=True) for td in second_tr.find_all("td")] + ["", ""]
        name_a, name_b = clean_name(tds[1]), clean_name(second[0])
        if not name_a or not name_b:
            continue
        hhmm = re.search(r"(\d{1,2}):(\d{2})", tds[0])
        link = tr.find("a", href=re.compile(r"match-detail"))
        match_id = re.search(r"id=(\d+)", link["href"]) if link else None
        odds = [x for x in map(to_number, tds[2:]) if x >= 1.01]  # the last two are home / away
        matches.append({
            "tournament": tournament, "tournament_href": href,
            "Winner": name_a, "Loser": name_b,
            "Date": day.replace(hour=int(hhmm[1]), minute=int(hhmm[2])) if hhmm else day,
            "match_id": match_id[1] if match_id else None,
            "completed": tds[2].isdigit() or second[1].isdigit(),  # sets won already filled in
            "list_odds_h": odds[-2] if len(odds) >= 2 else pd.NA,
            "list_odds_a": odds[-1] if len(odds) >= 2 else pd.NA,
        })
    return matches


def scrape_days(n_days):
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    rows = []
    for day in (today + timedelta(days=i) for i in range(n_days)):
        print(f"  list {day.date()} ...")
        rows += parse_day(get_soup(f"{BASE}/next/?type=atp-men&year={day:%Y}&month={day:%m}&day={day:%d}"), day)
    return rows


####################################
# 2) ranking points
####################################

def load_ranking_points(max_pages=8):
    """player_key -> (rank, points) for the first max_pages * 50 players."""
    points = {}
    for page in range(1, max_pages + 1):
        soup = get_soup(f"{BASE}/ranking/atp-men/?page={page}")
        for tr in soup.select("table.result tr"):
            tds = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(tds) >= 5 and re.fullmatch(r"\d+\.", tds[0]):
                points[player_key(tds[2])] = (to_number(tds[0]), to_number(tds[4]))
    return points

####################################
# 3) tournament page: surface, court, city
####################################
# these data are provided by the site in a differente webpage

def surface_and_court(token):
    """From the text in brackets on the tournament page, e.g. '(9,415,725 $, hard, men)'."""
    if "indoor" in token:
        return "Hard", "Indoor"
    if "outdoor" in token:
        return pd.NA, "Outdoor"
    if token in SURFACES:
        return SURFACES[token], "Indoor" if token == "carpet" else "Outdoor"
    return pd.NA, pd.NA


def parse_tournament(href):
    soup = get_soup(BASE + href)
    title = soup.select_one("h1").get_text(" ", strip=True) if soup.select_one("h1") else ""  # 'Shanghai 2026 (China)'
    m = re.search(r"\([^()\n]*(indoors?|outdoors?|hard|clay|grass|carpet)[^()\n]*\)", soup.get_text("\n", strip=True), re.I)
    surface, court = surface_and_court(m[1].lower() if m else "")
    location = re.sub(r"\s+20\d{2}.*", "", title).strip() if "(" in title else pd.NA
    return {"Surface": surface, "Court": court, "Location_guess": location}


####################################
# 4) match page: bet365 odds, rankings, round
####################################

def parse_detail(match_id):
    soup = get_soup(f"{BASE}/match-detail/?id={match_id}")
    info = {}
    for tr in soup.select("table.result.gDetail tr"):
        cells = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
        if len(cells) >= 4 and "ranking" in " ".join(cells).lower():  # ['', '13.', 'Singles ranking', '24.', '']
            info["WRank"], info["LRank"] = to_number(cells[1]), to_number(cells[3])

    for tr in soup.select("#oddsMenu-1-data tr") or soup.select("tr"):  # home/away tab, else try scraping the whole page
        k1, k2 = tr.select_one("td.k1"), tr.select_one("td.k2")
        if k1 is None or k2 is None:
            continue
        label = (tr.select_one("span.t") or tr.find("td")).get_text(" ", strip=True).lower()
        odds = [first_number((td.select_one("div.odds-in") or td).get_text(" ", strip=True)) for td in (k1, k2)]
        if pd.isna(odds).any():
            continue
        if "bet365" in label:
            info["B365W"], info["B365L"] = odds
        elif "average" in label:
            info["AvgW"], info["AvgL"] = odds

    # On the website, each match has a short info box (e.g., 'Today, 08:40, Shanghai, quarterfinal, hard') with info that 
    #takes priority over the general tournament data extracted by `surface_and_court`, 
    # because the scheduled surface or court can change (e.g., a match moved indoors due to rain).
    boxes = (re.sub(r"\s+", " ", div.get_text(" ", strip=True)) for div in soup.select("div.box"))
    header = next((t for t in boxes if re.search(r"\d{1,2}:\d{2}", t) and "," in t and len(t) < 160), "")
    for piece in [p.strip().lower() for p in header.split(",")]:
        if piece in SURFACES:
            info["Surface_detail"] = SURFACES[piece]
        elif piece in COURTS:
            info["Court_detail"] = COURTS[piece]
        elif piece in ROUNDS or piece.startswith("round of"):
            info["Round"] = ROUNDS.get(piece, piece.title().replace("Of", "of"))
    return info


####################################
# 5) integration of data
####################################

def location_history():
    #location is not available from tennis-data, but I can retrieve it from the history files, using the tournament name. I will take the most frequent location for each tournament.
    #If the tournament is not in the history, I fall back on the location scraped from the tennis explorer tournament page (Location_guess).
    #Location is needed for the calculation of the variable "home"
    files = [p for p in Path("file").glob("*.xlsx") if p.stem.isdigit()]
    if not files:
        return {}
    h = pd.concat(pd.read_excel(p, usecols=["Tournament", "Location"]) for p in files).dropna()
    h["Tournament"] = h["Tournament"].astype(str).str.lower()
    return h.groupby("Tournament")["Location"].agg(lambda s: s.mode()[0]).to_dict()


def map_location(tournament, history):
    name = tournament.lower().strip()
    if "davis" in name:
        #davis cup is not in the history and also "davis" isn't a city: skip the search below, set to None.
        return None
    if name in history:
        return history[name]
    return next((loc for t, loc in history.items() if t in name or name in t or loc.lower() in name), None)


####################################
# 6) output
####################################

def first_valid(*values):
    return next((v for v in values if pd.notna(v)), pd.NA)


def parse_or_empty(parse, what, key):
    """One page that fails (e.g. HTTP 500) must not kill the whole run: its fields stay empty."""
    try:
        return parse(key)
    except Exception as e:
        print(f"   {what} {key} failed: {e}")
        return {}


def build_excel(rows, out_file):
    #rows are the dictionaries with the scraped characteristics of each match. They are prepared by the main() function.
    df = pd.DataFrame(rows).reindex(columns=TENNIS_DATA_COLUMNS)
    df["Comment"], df["Best of"] = "Scheduled", 3
    #I put 3 , the more common value. Anyway it doesn't matter for the models.
    gaps = df[["Date", "Winner", "Loser", "Tournament"]].assign(
        missing_B365=df["B365W"].isna() | df["B365L"].isna(),
        missing_rank=df["WRank"].isna() | df["LRank"].isna(),
        missing_points=df["WPts"].isna() | df["LPts"].isna(),
        missing_location=df["Location"].isna(),
        missing_surface=df["Surface"].isna(),
        missing_court=df["Court"].isna(),
    )
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="matches", index=False)
        gaps.to_excel(writer, sheet_name="gaps", index=False)
    return df


def main(days=2, max_matches=40, out="upcoming_matches.xlsx", include_challenger=False):
    print("1) ATP match list from Tennis Explorer")
    to_play = pd.DataFrame(scrape_days(days))
    if to_play.empty:
        raise SystemExit("No rows read from the list.")
    to_play = to_play[~to_play["completed"]]
    if not include_challenger:
        #the history doesn't contain lower tiers, like challenger or utr, so I filter them out.
        main_tour = [is_main_tour(t, h) for t, h in zip(to_play["tournament"], to_play["tournament_href"])]
        to_play = to_play[main_tour]
    to_play = to_play.drop_duplicates(subset=["match_id", "Winner", "Loser"])
    print(f"   {len(to_play)} matches not yet finished (main ATP, without challenger/UTR)")
    if to_play.empty:
        print("   ATP tour draw is empty. Try again in a few days, or run with --include-challenger for a test.")
        return
    if to_play["tournament"].str.contains("Davis", case=False).all():
        print("   WARNING: all matches are Davis Cup. The tennis-data history does not contain them.")
    to_play = to_play.head(max_matches)

    print("2) ATP ranking (points) ...")
    points = load_ranking_points()
    print(f"   {len(points)} ranked players")

    print("3) Tournament pages (surface / indoor) ...")
    tournaments = {href: parse_or_empty(parse_tournament, "tournament page", href)
                   for href in set(to_play["tournament_href"]) if href}
    history = location_history()

    print("4) Match detail pages (bet365 + ranking) ...")
    rows = []
    for i, r in enumerate(to_play.itertuples(index=False), 1):
        det = parse_or_empty(parse_detail, "detail page", r.match_id) if r.match_id else {}
        meta = tournaments.get(r.tournament_href, {})
        davis = "davis" in r.tournament.lower()
        w_rank, w_pts = points.get(player_key(r.Winner), (pd.NA, pd.NA))
        l_rank, l_pts = points.get(player_key(r.Loser), (pd.NA, pd.NA))
        row = {
            "Location": map_location(r.tournament, history) or (pd.NA if davis else meta.get("Location_guess", pd.NA)),
            "Tournament": r.tournament,
            "Date": r.Date,
            "Series": "Davis Cup" if davis else pd.NA,
            "Court": first_valid(det.get("Court_detail"), meta.get("Court")),
            "Surface": first_valid(det.get("Surface_detail"), meta.get("Surface")),
            "Round": det.get("Round", pd.NA),
            "Winner": r.Winner, "Loser": r.Loser,
            "WRank": first_valid(det.get("WRank"), w_rank),
            "LRank": first_valid(det.get("LRank"), l_rank),
            "WPts": w_pts, "LPts": l_pts,
            "B365W": det.get("B365W", pd.NA), "B365L": det.get("B365L", pd.NA),
            "AvgW": det.get("AvgW", pd.NA), "AvgL": det.get("AvgL", pd.NA),
        }
        if pd.isna(row["B365W"]) or pd.isna(row["B365L"]):  # no bet365 on the match page: use the list odds
            row["B365W"], row["B365L"] = r.list_odds_h, r.list_odds_a
        print(f"   [{i}/{len(to_play)}] {r.Winner} vs {r.Loser}  B365={row['B365W']}/{row['B365L']}")
        rows.append(row)

    df = build_excel(rows, out)
    print(f"\nWritten {out}: {len(df)} matches")
    for col in ["B365W", "WRank", "WPts", "Location", "Surface", "Court"]:
        print(f"  with {col}: {df[col].notna().sum()}/{len(df)}")
    print("The 'gaps' sheet lists what is missing, to be filled in by hand if needed.")
    print("Then: predict_upcoming_matches() in main.py")


if __name__ == "__main__":
    #the script will be mainly used on a call from main.py. It can also be run from the terminal (python tennis_explorer_next_matches.py --days 3, etc). In this case Python sets the __name__ variable to __main__
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=2, help="today + N-1 days")
    p.add_argument("--max-matches", type=int, default=40)
    p.add_argument("--out", default="upcoming_matches.xlsx")
    p.add_argument("--include-challenger", action="store_true")
    main(**vars(p.parse_args()))
