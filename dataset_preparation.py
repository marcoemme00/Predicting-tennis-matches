"""Build and clean the ATP match dataset used by the models."""

import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests

PLAYERS_FILE = 'file/others/players_data_ATP Tour Ranking - Decade-wise and year-wise.xlsx'
CITY_COUNTRY_FILE = 'file/others/states.xlsx'
COUNTRY_CODES_FILE = 'file/others/states_3_letters.xlsx'

# Player keys (surname + initial) spelled differently in the matches and in the players file.
# dolgopolovo (2017 matches) is the son Aleksandr/Oleksandr (dolgopolova), not the 1964 father.
DRAW_KEY_FIXES = {'dolgopolovo': 'dolgopolova', 'mecirm': 'mecirjrm', 'dedurad': 'dedurapalomerod'}
# Same thing, fixed on the players file side.
NAME_MISMATCHES = {
    'mpetshiperricardg': 'mpetshig', 'etcheverrytm': 'etcheverryt',
    'herbertph': 'herbertp', 'burruchagara': 'burruchagar',
    'bautistaagutr': 'bautistar', 'tsongaj': 'tsongajw',
    'moronigm': 'moronig', 'olivieriga': 'olivierig',
}

# Namesakes (same surname + initial) are told apart by their age at the match date: the bio is attached
# only when exactly one candidate is between MIN_PLAYER_AGE and MAX_PLAYER_AGE, otherwise it stays empty.
MIN_PLAYER_AGE = 16
MAX_PLAYER_AGE = 45
DROP_UNRESOLVED_NAMESAKES = False  # True: drop the matches where a namesake could not be resolved
DROP_ROWS_WITHOUT_BIO = False      # True: drop every match with a player without bio (otherwise imputed)

# Season at the match location: reversed in the southern hemisphere (the Australian Open in January is summer).
SOUTHERN_HEMISPHERE = {'Argentina', 'Australia', 'Brazil', 'Chile', 'Ecuador', 'New Zealand'}
OPPOSITE_SEASON = {'Spring': 'Autumn', 'Summer': 'Winter', 'Autumn': 'Spring', 'Winter': 'Summer'}

# In 2021.xlsx three Djokovic matches have the Bet365 odds swapped (Djokovic at 29, the other at 0.967).
# Pinnacle on the same rows is consistent, so those odds are copied onto B365 before choosing the favorite.
B365_FROM_PINNACLE = [
    ('2021-02-10', 'Djokovic N.', 'Tiafoe F.'),
    ('2021-06-01', 'Djokovic N.', 'Sandgren T.'),
    ('2021-06-05', 'Djokovic N.', 'Berankis R.'),
]

# Winner/Loser columns and their Favorite/Underdog counterparts (Elo is handled in realign_elo_favorite_underdog).
COLUMN_PAIRS = [
    ('Winner', 'Loser', 'Favorite', 'Underdog'), ('WRank', 'LRank', 'fRank', 'uRank'),
    ('WPts', 'LPts', 'fPts', 'uPts'), ('WPts_class', 'LPts_class', 'fPts_class', 'uPts_class'),
    ('B365W', 'B365L', 'fB365', 'uB365'), ('PSW', 'PSL', 'fPS', 'uPS'),
    ('prob_B365W_norm', 'prob_B365L_norm', 'fprob_B365_norm', 'uprob_B365_norm'),
    ('prob_PSW_norm', 'prob_PSL_norm', 'fprob_PSW_norm', 'uprob_PSL_norm'),
    ('winner_formatted', 'loser_formatted', 'favorite_formatted', 'underdog_formatted'),
    ('hand_winner', 'hand_loser', 'fhand', 'uhand'), ('age_winner', 'age_loser', 'fage', 'uage'),
    ('height_winner', 'height_loser', 'fheight', 'uheight'),
    ('stato_en_winner', 'stato_en_loser', 'fstato_en', 'ustato_en'),
]

# Rank bands used by the ranking/points checks.
RANK_BANDS = [1, 6, 11, 51, 101, 251, 501, 1001, np.inf]
RANK_BAND_LABELS = ['1-5', '6-10', '11-50', '51-100', '101-250', '251-500', '501-1000', '1001+']


# === loading ===

TENNIS_DATA_PAGE = "http://www.tennis-data.co.uk/data.php"
SEASON_START_YEAR = 2013
_DOWNLOAD_HEADERS = {"User-Agent": "Mozilla/5.0"}


def _season_links(html):
    """ATP year -> relative workbook link. The women's files sit under a 'w' folder and are left out."""
    return {
        int(year): href
        for href, year in re.findall(r'href="([^"]+/(\d{4})/\2\.xlsx)"', html, flags=re.I)
    }


def download_season_files(folder='file', start_year=SEASON_START_YEAR):
    """
    Save any missing ATP season workbook from tennis-data.co.uk into folder.
    A file that is already there is left untouched.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    missing = [year for year in range(start_year, datetime.now().year + 1)
               if not (folder / f"{year}.xlsx").exists()]
    if not missing:
        return
    print(f"Downloading {len(missing)} season file(s) from tennis-data.co.uk into {folder}/")
    try:
        page = requests.get(TENNIS_DATA_PAGE, headers=_DOWNLOAD_HEADERS, timeout=60)
        page.raise_for_status()
    except requests.RequestException as error:
        print(f"Could not reach tennis-data.co.uk ({error}). Using the workbooks already in {folder}/.")
        return
    links = _season_links(page.text)
    for year in missing:
        href = links.get(year)
        if href is None:
            print(f"  {year}: no ATP workbook linked on the data page, skipped")
            continue
        temporary = folder / f"{year}.xlsx.part"
        try:
            with requests.get(urljoin(TENNIS_DATA_PAGE, href), headers=_DOWNLOAD_HEADERS, timeout=120, stream=True) as downloaded:
                downloaded.raise_for_status()
                with open(temporary, "wb") as handle:
                    for chunk in downloaded.iter_content(chunk_size=65536):
                        if chunk:
                            handle.write(chunk)
        except requests.RequestException as error:
            temporary.unlink(missing_ok=True)
            print(f"  {year}: download failed ({error})")
            continue
        content = temporary.read_bytes()[:2]
        if temporary.stat().st_size < 1000 or content != b"PK":
            temporary.unlink(missing_ok=True)
            print(f"  {year}: the download was not an Excel file, skipped")
            continue
        temporary.replace(folder / f"{year}.xlsx")
        print(f"  {year}: saved")
        time.sleep(1)


def merge_excel_files(folder='file'):
    """Stack the yearly tennis-data files (2013.xlsx, 2014.xlsx, ...) into one table with a Year column."""
    download_season_files(folder)
    frames = []
    for path in sorted(Path(folder).glob('*.xlsx')):
        if not path.stem.isdigit():
            print(f"Skipped {path.name}: the name is not a valid year.")
            continue
        # bookmakers that disappeared from the data over the years
        df = pd.read_excel(path).drop(columns=['EXW', 'EXL', 'LBW', 'LBL', 'SJW', 'SJL'], errors='ignore')
        frames.append(df.assign(Year=int(path.stem)))
    return pd.concat(frames, ignore_index=True) if frames else None


def load_support_tables(players_file=PLAYERS_FILE, city_country_file=CITY_COUNTRY_FILE,
                        country_codes_file=COUNTRY_CODES_FILE):
    """Players bio, city. From city is attached country and IOC code tables used by build_dataset, with some manual fixes."""
    players = pd.read_excel(players_file)
    surname = players['name_last'].str.replace(r'[\s\.\-]', '', regex=True).str.lower()
    initial = players['name_first'].str.strip().str[0].str.lower()
    players['name_formatted'] = (surname + initial).replace(NAME_MISMATCHES)

    # Santiago in tennis-data is the Chile Open, not the Costa Rica city
    city_country = pd.read_excel(city_country_file)
    city_country.loc[city_country['city'] == 'Santiago', ['country', 'state', 'country_code']] = [
        'Chile', 'Santiago Metropolitan', 'CL']

    # LIB (Lebanon) is missing from states_3_letters.xlsx
    country_codes = pd.read_excel(country_codes_file)
    country_codes.loc[len(country_codes)] = ['LIB', 'Libano', 'Lebanon']
    return players, city_country, country_codes


def format_draw_keys(names):
    """Name as in the matches file ("Federer R.") -> key surname + initial ("federerr"), with the manual fixes."""
    return pd.Series(names).astype(str).str.replace(r'[\.\s\-]', '', regex=True).str.lower().replace(DRAW_KEY_FIXES)


def to_predict_mask(df, flag_col):
    """Matches to predict (all False when there is no flag column)."""
    if flag_col in df.columns:
        return df[flag_col].fillna(False).astype(bool)
    return pd.Series(False, index=df.index)


# === players bio ===

def player_age_years(dates, dobs):
    """Age on the match date; date of birth comes from the ATP file as YYYYMMDD."""
    dob = pd.to_datetime(pd.Series(dobs).astype('Int64').astype(str), format='%Y%m%d', errors='coerce')
    lived = pd.to_timedelta(pd.to_datetime(dates).to_numpy() - dob.to_numpy())
    return (lived.days // 365).to_numpy()


def attach_player_bio(matches, bio, name_col, suffix):
    """
    Bio columns (player_id, hand, dob, ioc, height, age) of the player in name_col, one row per match.
    Namesakes are resolved by age: the bio is attached only when exactly one candidate has a plausible age.
    """
    bio_cols = ['player_id', 'hand', 'dob', 'ioc', 'height']
    cand = (matches[['Date', name_col]].assign(row=range(len(matches)))
            .merge(bio[['name_formatted'] + bio_cols], left_on=name_col, right_on='name_formatted'))
    cand['age'] = player_age_years(cand['Date'], cand['dob'])

    plausible = cand['age'].between(MIN_PLAYER_AGE, MAX_PLAYER_AGE)
    n_cand = cand.groupby('row')['row'].transform('size')
    n_plausible = plausible.groupby(cand['row']).transform('sum')
    picked = cand[(n_cand == 1) | (plausible & (n_plausible == 1))]

    out = picked.set_index('row')[bio_cols + ['age']].reindex(range(len(matches)))
    return out.add_suffix(f'_{suffix}').reset_index(drop=True)


def why_no_bio(name, date, bio):
    """Why attach_player_bio left this player empty (diagnostics only)."""
    if pd.isna(name) or str(name).strip() in ('', 'nan'):
        return 'empty name'
    cand = bio[bio['name_formatted'] == name].copy()
    if cand.empty:
        return 'not in the players file'
    cand['age'] = player_age_years(date, cand['dob'])
    plausible = cand[cand['age'].between(MIN_PLAYER_AGE, MAX_PLAYER_AGE)]

    def label(r):
        age = f"{r.age:.0f} years" if pd.notna(r.age) else 'unknown age'
        return f"{r.name_first} {r.name_last} ({r.ioc}, {age})"

    if plausible.empty:
        who = '; '.join(label(r) for r in cand.itertuples())
        return f'{len(cand)} namesakes, none between {MIN_PLAYER_AGE} and {MAX_PLAYER_AGE} years: {who}'
    who = '; '.join(label(r) for r in plausible.itertuples())
    return f'{len(plausible)} namesakes between {MIN_PLAYER_AGE} and {MAX_PLAYER_AGE} years: {who}'


def log_merge_issues(data, bio, flag_col=None):
    """Print the players, cities, and countries that could not be attached (if flag_col is provided, diagnostics are limited to the matches to predict (flagged))."""
    view = data[to_predict_mask(data, flag_col)] if flag_col else data
    print(f"\nMerges on {len(view)} matches" + (" to predict" if flag_col else ""))

    no_bio, no_country = [], []
    for side, other in (('winner', 'loser'), ('loser', 'winner')):
        missing = view[view[f'player_id_{side}'].isna()]
        no_bio += [f"  {r[side.title()]} vs {r[other.title()]}: {why_no_bio(r[f'{side}_formatted'], r['Date'], bio)}"
                   for _, r in missing.iterrows()]
        unmatched = view[view[f'player_id_{side}'].notna() & view[f'stato_en_{side}'].isna()]
        no_country += [f"  {r[side.title()]}: " + ('players file without IOC code' if pd.isna(r[f'ioc_{side}'])
                       else f"code {r[f'ioc_{side}']} missing in states_3_letters.xlsx")
                       for _, r in unmatched.iterrows()]
    locations = view.loc[view['country'].isna(), 'Location'].fillna('').astype(str).str.strip()
    no_city = list(dict.fromkeys(f'  {loc}: missing in states.xlsx' if loc else '  empty Location' for loc in locations))

    def report(lines, problem, all_good):
        print(f"{problem}: {len(lines)}" if lines else all_good)
        if lines:
            print('\n'.join(lines))

    report(no_bio, "Biographical data not attached", "Biographical data: all players attached")
    if no_bio and not DROP_UNRESOLVED_NAMESAKES and not DROP_ROWS_WITHOUT_BIO:
        print("  The match is kept: hand, height, age and country of that player stay empty.")
    report(no_city, "City not attached", "Cities: all present in states.xlsx")
    report(no_country, "Player country not attached", "Player country: all IOC codes present")


# === features ===

def season_at(date, country=None):
    """Astronomical season on the match date, flipped for southern-hemisphere countries."""
    month_day = (date.month, date.day)
    if month_day < (3, 21) or month_day >= (12, 21):
        north = 'Winter'
    elif month_day < (6, 21):
        north = 'Spring'
    elif month_day < (9, 22):
        north = 'Summer'
    else:
        north = 'Autumn'
    return OPPOSITE_SEASON[north] if country in SOUTHERN_HEMISPHERE else north


def compute_elo_rankings(data, flag_col=None, verbose=True):
    """
    Elo rating (the chess skill score, tennis is a two-player game too) of both players before the match.
    Ratings move one calendar day at a time: the matches of day X all use the ratings up to day X-1.
    Players are identified by winner_formatted / loser_formatted, so different spellings share one rating.
    Matches flagged in flag_col (to predict) get a rating but do not update it.
    """
    if verbose:
        print("Computing Elo ratings...")
    # when a player has two matches on the same day they are updated in file order
    data = data.sort_values('Date', kind='stable').reset_index(drop=True)
    winners, losers = data['winner_formatted'].to_numpy(), data['loser_formatted'].to_numpy()
    known = ~to_predict_mask(data, flag_col).to_numpy()
    day = pd.to_datetime(data['Date']).dt.normalize()

    k = 32
    elo = defaultdict(lambda: 1500.0)
    elo_w, elo_l = np.empty(len(data)), np.empty(len(data))
    for rows in data.groupby(day).indices.values():
        elo_w[rows] = [elo[p] for p in winners[rows]]
        elo_l[rows] = [elo[p] for p in losers[rows]]
        for w, l in zip(winners[rows][known[rows]], losers[rows][known[rows]]):
            expected = 1 / (1 + 10 ** ((elo[l] - elo[w]) / 400))
            elo[w] += k * (1 - expected)
            elo[l] -= k * (1 - expected)

    data['elo_winner'], data['elo_loser'] = elo_w, elo_l
    data['proba_elo'] = 1 / (1 + 10 ** ((elo_l - elo_w) / 400))
    return data


def realign_elo_favorite_underdog(df):
    """
    Elo from Winner/Loser to Favorite/Underdog. Kept apart from the other favorite/underdog columns because
    Elo depends on results: when the ratings are frozen for a period to predict only these columns change.
    """
    df = df.copy()
    winner_is_favorite = df['B365W'] < df['B365L']
    df['felo'] = df['elo_winner'].where(winner_is_favorite, df['elo_loser'])
    df['uelo'] = df['elo_loser'].where(winner_is_favorite, df['elo_winner'])
    df['proba_elo_favorite'] = df['proba_elo'].where(winner_is_favorite, 1 - df['proba_elo'])
    df['elo_gap'] = df['felo'] - df['uelo']
    return df


def normalize_odds(df, winner_col, loser_col):
    """Odds include the bookmaker's margin, so the implied probabilities sum to more than 1: rescale them to 1."""
    for col in (winner_col, loser_col):  # "2,5" -> 2.5, text like "-" -> NaN
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
    prob_w, prob_l = 1 / df[winner_col], 1 / df[loser_col]
    df[f'prob_{winner_col}_norm'] = prob_w / (prob_w + prob_l)
    df[f'prob_{loser_col}_norm'] = prob_l / (prob_w + prob_l)
    return df


def points_class(points):
    """ATP ranking points in 5 classes."""
    return pd.cut(points, [0, 560, 920, 1460, 2000, np.inf], labels=[1, 2, 3, 4, 5], right=False).astype(float)


def historical_win_percentages(df, flag_col=None):
    """
    Win percentage of the favorite and loss percentage of the underdog over the days before the match.
    Matches to predict (flag_col) count as not played, so the percentages stop at the day before the period
    to predict (to predict January 2026 they are computed up to 31 December 2025).

    Players with few matches would get extreme values (with only one match played the win_percentage could take only two values: 100% or 0%), so I add to every player
    7 (fake) matches 3.5 of which were won:
    -a debut is 50%
    -after one win (1 + 3.5) / (1 + 7) = 56%
    -after one loss 43.75%
    and so on. The values converges to the real win percentage as matches progress (a form of Bayesian smoothing).
    """
    df = df.drop(columns=['favorite_win_percentage', 'underdog_loss_percentage'], errors='ignore')
    df = df.reset_index(drop=True)
    day = pd.to_datetime(df['Date']).dt.normalize()
    played = (~to_predict_mask(df, flag_col)).astype(int)

    sides = pd.concat([
        pd.DataFrame({'player': df['winner_formatted'], 'day': day, 'won': played, 'played': played}),
        pd.DataFrame({'player': df['loser_formatted'], 'day': day, 'won': 0, 'played': played}),
    ])
    daily = sides.groupby(['player', 'day'])[['won', 'played']].sum()
    before = daily.groupby(level='player').cumsum() - daily  # results up to the day before
    win_pct = 100 * (before['won'] + 3.5) / (before['played'] + 7)
    loss_pct = 100 * (before['played'] - before['won'] + 3.5) / (before['played'] + 7)

    # if a player name is missing > win percentage is filled the 50%  (never happened on the current data)
    favorite = pd.MultiIndex.from_arrays([df['favorite_formatted'], day])
    underdog = pd.MultiIndex.from_arrays([df['underdog_formatted'], day])
    df['favorite_win_percentage'] = win_pct.reindex(favorite).fillna(50).to_numpy()
    df['underdog_loss_percentage'] = loss_pct.reindex(underdog).fillna(50).to_numpy()
    return df


def recompute_dynamic_features(df, start_date, end_date):
    """Elo and historical percentages frozen at start_date, for the matches to predict up to end_date."""
    work = df[df['Date'] <= end_date].copy()
    work['to_predict'] = work['Date'] >= start_date
    work = compute_elo_rankings(work, flag_col='to_predict', verbose=False)
    work = realign_elo_favorite_underdog(work)
    work = historical_win_percentages(work, flag_col='to_predict')
    return work.drop(columns=['to_predict'])


def build_dataset(matches, players, city_country, country_codes, flag_col=None, verbose=False):
    """
    From the raw matches to the table with the features for the models.

    players, city_country, country_codes come from load_support_tables().
    flag_col: boolean column marking the matches to predict. For them Elo and historical percentages are
    not updated and favorite_won is left missing. Matches of the same day share the same Elo and percentages.
    verbose: print the players, cities and countries that could not be attached (flagged matches only, if set).
    """
    data = matches.reset_index(drop=True).copy()
    data['winner_formatted'] = format_draw_keys(data['Winner'])
    data['loser_formatted'] = format_draw_keys(data['Loser'])

    # biographical data and the country of each player (IOC code, country name)
    data = pd.concat([data, attach_player_bio(data, players, 'winner_formatted', 'winner'),
                      attach_player_bio(data, players, 'loser_formatted', 'loser')], axis=1)
    for side in ('winner', 'loser'):
        codes = country_codes.rename(columns={'stato_it': f'stato_it_{side}', 'stato_en': f'stato_en_{side}'})
        data = data.merge(codes, left_on=f'ioc_{side}', right_on='sigla', how='left').drop(columns='sigla')
    # the country where the match was played
    data = data.merge(city_country, left_on='Location', right_on='city', how='left').drop(columns='city')

    if verbose:
        log_merge_issues(data, players, flag_col)
    if DROP_UNRESOLVED_NAMESAKES:
        namesakes = players.loc[players['name_formatted'].duplicated(keep=False), 'name_formatted']
        unresolved = ((data['winner_formatted'].isin(namesakes) & data['player_id_winner'].isna())
                      | (data['loser_formatted'].isin(namesakes) & data['player_id_loser'].isna()))
        print(f"DROP_UNRESOLVED_NAMESAKES: dropped {unresolved.sum()} of {len(data)} matches")
        data = data[~unresolved]
    if DROP_ROWS_WITHOUT_BIO:
        before = len(data)
        data = data.dropna(subset=['player_id_winner', 'player_id_loser'])
        print(f"DROP_ROWS_WITHOUT_BIO: dropped {before - len(data)} matches without biographical data")

    data['day_of_match'] = data['Date'].dt.day_name()
    data['season'] = [season_at(date, country) for date, country in zip(data['Date'], data['country'])]
    data = compute_elo_rankings(data, flag_col=flag_col)

    day = data['Date'].dt.normalize()
    for date, winner, loser in B365_FROM_PINNACLE:
        row = (day == date) & (data['Winner'] == winner) & (data['Loser'] == loser)
        if row.sum() != 1:
            print(f"B365 odds fix: expected 1 row for {date} {winner}-{loser}, found {row.sum()}")
            continue
        data.loc[row, ['B365W', 'B365L']] = data.loc[row, ['PSW', 'PSL']].to_numpy()
        print(f"B365 odds fix: {date} {winner} vs {loser}, using Pinnacle")

    data = normalize_odds(data, 'B365W', 'B365L')
    data = normalize_odds(data, 'PSW', 'PSL')
    data['WPts_class'] = points_class(data['WPts'])
    data['LPts_class'] = points_class(data['LPts'])

    # the favorite has the shorter Bet365 odds; without odds, or with equal odds, there is no favorite
    data = data.dropna(subset=['B365W', 'B365L'])
    data = data[data['B365W'] != data['B365L']].copy()
    winner_is_favorite = data['B365W'] < data['B365L']
    for winner_col, loser_col, favorite_col, underdog_col in COLUMN_PAIRS:
        data[favorite_col] = data[winner_col].where(winner_is_favorite, data[loser_col])
        data[underdog_col] = data[loser_col].where(winner_is_favorite, data[winner_col])
    data = realign_elo_favorite_underdog(data)

    # the target, missing for the matches to predict
    data['favorite_won'] = winner_is_favorite
    if flag_col in data.columns:
        data['favorite_won'] = data['favorite_won'].astype('boolean').mask(to_predict_mask(data, flag_col))

    data = historical_win_percentages(data, flag_col=flag_col)

    # home advantage: 1 only the favorite plays at home, -1 only the underdog, 0 both or neither
    data[['fstato_en', 'ustato_en']] = data[['fstato_en', 'ustato_en']].replace('Great Britain', 'United Kingdom')
    data['Hf'] = (data['fstato_en'] == data['country']).astype(int)
    data['Uf'] = (data['ustato_en'] == data['country']).astype(int)
    data['home'] = data['Hf'] - data['Uf']

    # player comparisons, NaN when one side is unknown (the imputer handles it)
    hands_known = data['fhand'].notna() & data['uhand'].notna()
    data['different_hand'] = (data['fhand'] != data['uhand']).astype(float).where(hands_known)
    heights_known = data['fheight'].notna() & data['uheight'].notna()
    data['f_taller'] = (data['fheight'] > data['uheight'] + 3).astype(float).where(heights_known)
    data['fhand_rec'] = data['fhand'].map({'R': 1, 'L': 0})
    data['uhand_rec'] = data['uhand'].map({'R': 1, 'L': 0})

    # gaps: positive = advantage for the favorite (a lower rank number is better; elo_gap is computed above)
    data['rank_gap'] = data['uRank'] - data['fRank']
    data['age_gap'] = data['fage'] - data['uage']
    data['height_gap'] = data['fheight'] - data['uheight']

    data = pd.get_dummies(data, columns=['day_of_match', 'Surface', 'Court', 'season'], drop_first=True,
                          prefix={'day_of_match': 'day_of_match_rec', 'Surface': 'surface_rec',
                                  'Court': 'court_rec', 'season': 'season_rec'})
    data['bimonthly_period'] = (data['Date'].dt.month - 1) // 2 + 1
    return data


# === data checks ===

def rank_points_long(df):
    """One row per player (favorites first, then underdogs) with year, rank, points and rank band."""
    year = pd.to_datetime(df['Date']).dt.year
    sides = pd.DataFrame({
        'year': pd.concat([year, year]).to_numpy(),
        'rank': pd.to_numeric(pd.concat([df['fRank'], df['uRank']]), errors='coerce').to_numpy(),
        'points': pd.to_numeric(pd.concat([df['fPts'], df['uPts']]), errors='coerce').to_numpy(),
    })
    sides['band'] = pd.cut(sides['rank'], RANK_BANDS, labels=RANK_BAND_LABELS, right=False)
    return sides


def rank_points_disagree(df):
    """
    A player with more points than the best player of the next better rank band in the same year
    (players already flagged do not count) points to a data entry error, like in the 2023 file.
    """
    sides = rank_points_long(df)
    bad = np.zeros(len(sides), dtype=bool)
    for _, year_rows in sides.groupby('year'):
        ceiling = np.inf
        for _, band_rows in year_rows.groupby('band', observed=True):  # best band first
            over = band_rows['points'] > ceiling
            bad[band_rows.index[over]] = True
            kept_max = band_rows.loc[~over, 'points'].max()
            if pd.notna(kept_max):
                ceiling = kept_max
    n = len(df)
    return pd.Series(bad[:n] | bad[n:], index=df.index)


def book_disagreement(df):
    """Bet365 and Pinnacle pick a different favorite. Returns the mask and the gap in implied probability. Large differences probably hide data entry errors, they will be excluded."""
    fb, ub, fp, up = (pd.to_numeric(df[c], errors='coerce') for c in ('fB365', 'uB365', 'fPS', 'uPS'))
    usable = fb.gt(1) & ub.gt(1) & fp.gt(1) & up.gt(1) & fb.ne(ub) & fp.ne(up)
    gap = (ub / (fb + ub) - up / (fp + up)).abs().fillna(0)
    return usable & ((fb < ub) != (fp < up)), gap


def score_error_masks(df):
    """Completed matches whose set count, set scores or Best of column disagree with each other."""
    completed = df['Comment'].astype(str).str.fullmatch('Completed', case=False)
    wsets = pd.to_numeric(df['Wsets'], errors='coerce')
    lsets = pd.to_numeric(df['Lsets'], errors='coerce')
    best_of = pd.to_numeric(df['Best of'], errors='coerce')
    sets_played = df[['W1', 'W2', 'W3', 'W4', 'W5']].apply(pd.to_numeric, errors='coerce').notna().sum(axis=1)
    went_long = df[['W4', 'W5']].apply(pd.to_numeric, errors='coerce').notna().any(axis=1)

    not_ahead = completed & wsets.le(lsets)
    mismatch = completed & (wsets + lsets).notna() & (wsets + lsets).ne(sets_played)
    best_bad = completed & (wsets.gt((best_of + 1) / 2) | (best_of.eq(3) & went_long))
    return not_ahead, mismatch, best_bad


def same_round_twice(df):
    """One player in two matches of the same round, tournament and day."""
    day = pd.to_datetime(df['Date']).dt.normalize()
    sides = pd.concat([
        pd.DataFrame({'day': day, 'Tournament': df['Tournament'], 'Round': df['Round'], 'player': df[col]})
        for col in ('Winner', 'Loser')
    ])
    repeated = sides.duplicated(keep=False) & sides['player'].notna()
    return repeated.groupby(level=0).any().reindex(df.index)


def duplicate_match_mask(df, keep=False):
    """Same day and the same two players, in either order."""
    players = df[['Favorite', 'Underdog']].astype(str)
    pair = pd.DataFrame({'day': pd.to_datetime(df['Date']).dt.normalize(),
                         'first': players.min(axis=1), 'second': players.max(axis=1)})
    return pair.duplicated(keep=keep)


def check_dataset(df):
    """Print the rows with suspicious data (nothing is dropped here, see clean_dataset)."""
    print("\n=== DATA CHECKS ===")

    def show(rows, columns):
        for _, r in rows[columns].head(8).iterrows():
            print("   " + " | ".join(f"{c}={r[c]}" for c in columns))

    odds_cols = ['Date', 'Favorite', 'Underdog', 'fB365', 'uB365', 'fPS', 'uPS']
    odds = df[['fB365', 'uB365', 'fPS', 'uPS']].apply(pd.to_numeric, errors='coerce')
    lowest = odds.min(axis=1)
    print(f"Odds below 1: {lowest.lt(1).sum()}")
    show(df[lowest.lt(1)], odds_cols)
    print(f"Odds exactly 1: {lowest.eq(1).sum()}")
    show(df[lowest.eq(1)], odds_cols)

    # a gap over 5 points of implied probability is not just a different pricing, it hides a data entry error
    opposite, gap = book_disagreement(df)
    wide = opposite & gap.gt(0.05)
    print(f"Bet365 and Pinnacle disagree on the favorite: {opposite.sum()} ({wide.sum()} by more than 5 points)")
    show(df[wide].assign(gap=gap).sort_values('gap', ascending=False), odds_cols)

    # favorite/underdog come from Bet365 and equal odds are dropped, so fB365 < uB365 always
    order_bad = odds['fB365'].ge(odds['uB365'])
    if order_bad.any():
        show(df[order_bad], odds_cols[:5])
        raise ValueError(f"Favorite without shorter Bet365 odds in {order_bad.sum()} rows: build_dataset should enforce this.")
    print("Favorite odds shorter than underdog: OK")

    prob_sum = df['fprob_B365_norm'] + df['uprob_B365_norm']
    print(f"Vig-free probabilities not summing to 1: {(prob_sum - 1).abs().gt(1e-6).sum()}")

    duplicated = duplicate_match_mask(df)
    print(f"Duplicate matches: {duplicated.sum()}")
    show(df[duplicated], odds_cols[:5])

    # Better-ranked players must have more points. Average points per rank band and year: a worse band
    # averaging more than the next better band means a data error in that year's file (e.g. 2023).
    by_band = (rank_points_long(df).dropna()
               .groupby(['year', 'band'], observed=True)['points'].agg(['mean', 'size']).unstack('band'))
    mean, n = by_band['mean'], by_band['size']
    inverted = (n >= 30) & (n.shift(axis=1) >= 30) & (mean > mean.shift(axis=1))
    inversions = []
    for yr, worse in inverted.stack().loc[lambda hit: hit].index:
        better = RANK_BAND_LABELS[RANK_BAND_LABELS.index(worse) - 1]
        inversions.append(f"   {yr}: rank {worse} averages {mean.loc[yr, worse]:.0f} points (n={n.loc[yr, worse]:.0f}), "
                          f"rank {better} averages {mean.loc[yr, better]:.0f} (n={n.loc[yr, better]:.0f})")
    print(f"Rank band with more points than a better band: {len(inversions)}")
    print("\n".join(inversions[:8]))

    rank_cols = ['Date', 'Favorite', 'Underdog', 'fRank', 'uRank', 'fPts', 'uPts']
    points_bad = rank_points_disagree(df)
    print(f"Points higher than the next better rank band: {points_bad.sum()}")
    show(df[points_bad], rank_cols)
    same_rank = df['fRank'].notna() & df['fRank'].eq(df['uRank'])
    print(f"Both players with the same ranking: {same_rank.sum()}")
    show(df[same_rank], rank_cols)

    # Best of and W1-L5 are not used by the models, but the few rows where they disagree are dropped to be safe.
    not_ahead, mismatch, best_bad = score_error_masks(df)
    score_cols = ['Date', 'Tournament', 'Winner', 'Loser', 'Wsets', 'Lsets', 'W1', 'L1', 'W2', 'L2', 'W3', 'L3']
    print(f"Completed match, winner not ahead in sets: {not_ahead.sum()}")
    show(df[not_ahead], score_cols)
    print(f"Completed match, set columns do not add up to the set count: {mismatch.sum()}")
    show(df[mismatch], score_cols + ['W4', 'L4', 'W5', 'L5'])
    print(f"Completed match, Best of does not match the sets played: {best_bad.sum()}")
    show(df[best_bad], ['Date', 'Tournament', 'Round', 'Best of', 'Winner', 'Loser', 'Wsets', 'Lsets', 'W4', 'L4', 'W5', 'L5'])

    same_round = same_round_twice(df)
    print(f"Same player in two matches of the same round on the same day: {same_round.sum()}")
    show(df[same_round], ['Date', 'Tournament', 'Round', 'Winner', 'Loser'])

    # ranges of the columns the code creates, plus age, height and probabilities
    age, height, pct = (MIN_PLAYER_AGE, MAX_PLAYER_AGE), (150, 220), (0, 100)
    bounds = {
        'fage': age, 'uage': age, 'age_winner': age, 'age_loser': age,
        'fheight': height, 'uheight': height, 'height_winner': height, 'height_loser': height,
        'favorite_win_percentage': pct, 'underdog_loss_percentage': pct,
        'fPts_class': (1, 5), 'uPts_class': (1, 5), 'bimonthly_period': (1, 6),
        'home': (-1, 1), 'felo': (800, 2800), 'uelo': (800, 2800),
    }
    bounds.update({c: (0, 1) for c in df.columns if 'prob' in c.lower() and 'class' not in c.lower()})
    binary = ['Hf', 'Uf', 'different_hand', 'f_taller', 'fhand_rec', 'uhand_rec', 'favorite_won']
    problems = []
    for col, (low, high) in bounds.items():
        values = pd.to_numeric(df[col], errors='coerce')
        outside = values.lt(low) | values.gt(high)
        if outside.any():
            problems.append(f"   {col}: {outside.sum()} values outside {low:g}-{high:g} "
                            f"(min {values.min():.3g}, max {values.max():.3g})")
    for col in binary:
        values = pd.to_numeric(df[col], errors='coerce')
        not_binary = values.notna() & ~values.isin([0, 1])
        if not_binary.any():
            problems.append(f"   {col}: {not_binary.sum()} values other than 0 or 1")
    print(f"Variables outside their range: {len(problems)}")
    print("\n".join(problems[:8]))


def clean_dataset(df):
    """Drop the matches flagged by the checks above: they could contain data entry errors."""
    odds = df[['fB365', 'uB365', 'fPS', 'uPS']].apply(pd.to_numeric, errors='coerce')
    opposite, gap = book_disagreement(df)
    not_ahead, mismatch, best_bad = score_error_masks(df)
    reasons = {
        'odds missing or <= 1': odds[['fB365', 'uB365']].isna().any(axis=1) | odds.le(1).any(axis=1),
        'same ranking': df['fRank'].notna() & df['fRank'].eq(df['uRank']),
        'rank and points disagree': rank_points_disagree(df),
        'books differ by more than 5 points': opposite & gap.gt(0.05),
        'winner not ahead in sets': not_ahead,
        'set score does not add up': mismatch,
        'Best of does not match': best_bad,
        'same round twice': same_round_twice(df),
        'duplicate matches': duplicate_match_mask(df, keep='first'),  # keep the first row of each pair
    }
    drop = pd.concat(reasons, axis=1).any(axis=1)
    counts = ', '.join(f'{reason}: {mask.sum()}' for reason, mask in reasons.items())
    print(f"Dropped {drop.sum()} matches ({counts}). {len(df) - drop.sum()} matches left.")
    #drop the matches (the ones with drop=False)
    return df[~drop].reset_index(drop=True)
