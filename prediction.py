"""Predict upcoming matches (live / Tennis Explorer file)."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from config import (
    CANDIDATE_FEATURES,
    EV_THRESHOLD,
    KELLY_REFERENCE_CAPITAL,
    PROB_THRESHOLD,
    SELECTION_METHOD,
    SHRINK_TO_BOOK,
    WINDOW_YEARS,
)
from dataset_preparation import (
    build_dataset,
    clean_dataset,
    format_draw_keys,
    load_support_tables,
    merge_excel_files,
)
from modeling import (
    bet_details,
    build_calibrated_pipeline,
    fit_with_shrinkage,
    kelly_stake,
    select_features_standard,
    stepwise_selection_roi,
)


def match_keys(dates, names_a, names_b):
    """Date plus the two names, sorted, so player order does not matter."""
    a, b = format_draw_keys(names_a), format_draw_keys(names_b)
    first, second = a.where(a <= b, b), b.where(a <= b, a)
    day = pd.to_datetime(pd.Series(dates)).dt.strftime('%Y-%m-%d')
    return (day + '|' + first + '|' + second).to_numpy()


def predict_upcoming_matches(upcoming_file='upcoming_matches.xlsx',
                             output_file='upcoming_predictions.xlsx',
                             candidate_features=None,
                             base_model=None,
                             window_years=WINDOW_YEARS,
                             selection_method=SELECTION_METHOD,
                             ev_threshold=EV_THRESHOLD,
                             prob_threshold=PROB_THRESHOLD):
    """
    The upcoming file has raw rows only (date, players, rankings, odds).
    They are appended to the history, features are rebuilt, and the model
    is trained on earlier matches. The output is one row per match: bet or not, on whom, stake.
    """
    candidate_features = candidate_features or CANDIDATE_FEATURES
    base_model = base_model or LogisticRegression(random_state=42, max_iter=1000)

    sheets = pd.ExcelFile(upcoming_file).sheet_names
    upcoming = pd.read_excel(upcoming_file, sheet_name='matches' if 'matches' in sheets else sheets[0])
    upcoming['Date'] = pd.to_datetime(upcoming['Date'])
    upcoming_keys = match_keys(upcoming['Date'], upcoming['Winner'], upcoming['Loser'])

    # optional column, used only to score a file whose winners are already known (for testing)
    true_outcomes = (
        dict(zip(upcoming_keys, upcoming.pop('true_winner')))
        if 'true_winner' in upcoming.columns else {}
    )
    upcoming['to_predict'] = True
    if 'Year' not in upcoming.columns:
        upcoming['Year'] = upcoming['Date'].dt.year

    raw_history = merge_excel_files()
    if raw_history is None:
        raise ValueError("No valid Excel file found in the folder")
    raw_history['Date'] = pd.to_datetime(raw_history['Date'])

    # a match still sitting in file/ would be both history and something to predict
    history_keys = match_keys(raw_history['Date'], raw_history['Winner'], raw_history['Loser'])
    overlap = np.isin(history_keys, upcoming_keys)
    if overlap.any():
        print(
            f"WARNING: {int(overlap.sum())} matches to predict are still in file/. "
            "They are excluded here, but they should be removed from the file."
        )
    else:
        print("Check passed: no match to predict is present in the history")
    raw_history = raw_history.loc[~overlap].copy()
    raw_history['to_predict'] = False

    # Elo and the historical percentages need every match before the ones we predict
    combined = pd.concat([raw_history, upcoming], ignore_index=True).sort_values('Date').reset_index(drop=True)
    players, city_country, country_codes = load_support_tables()
    data = clean_dataset(build_dataset(
        combined, players, city_country, country_codes, flag_col='to_predict', verbose=True,
    ))

    to_predict = data['to_predict'].fillna(False).astype(bool)
    df_predict = data.loc[to_predict].reset_index(drop=True)
    if df_predict.empty:
        print("No match to predict: check the player names and the odds in the file.")
        return None
    dropped = len(upcoming) - len(df_predict)
    if dropped:
        print(
            f"Warning: {dropped} matches dropped during data preparation "
            "(player without biographical data, namesake, or missing/equal odds)"
        )

    first_date = df_predict['Date'].min()
    past_limit = first_date - pd.DateOffset(years=window_years)
    df_train = data.loc[~to_predict & data['Date'].between(past_limit, first_date, inclusive='left')].copy()
    if df_train.empty:
        print("No training match available in the chosen window.")
        return None
    df_train['favorite_won'] = df_train['favorite_won'].astype(bool)
    print(f"Training on {len(df_train)} matches from {past_limit.date()} to {first_date.date()}")

    model = clone(base_model)
    if selection_method == 'roi':
        selected, estimated_roi = stepwise_selection_roi(
            df_train, candidate_features, model,
            ev_threshold=ev_threshold, prob_threshold=prob_threshold,
        )
        if selected:
            print(f"Estimated validation ROI: {float(estimated_roi) * 100:.2f}%")
    else:
        selected = select_features_standard(df_train, candidate_features, model)
    if not selected:
        print("No feature selected: no predictions produced.")
        return None
    print(f"Selected features ({selection_method}): {selected}")

    pipeline, prob_favorite, lambda_book, raw_prob = fit_with_shrinkage(
        build_calibrated_pipeline(model), df_train, df_predict, selected,
    )
    if prob_favorite is None:
        print("No training row left after the NaN filter.")
        return None
    if SHRINK_TO_BOOK:
        print(f"Shrinkage to book: lambda={lambda_book:.2f}")

    details = pd.DataFrame([
        bet_details(row, float(p), ev_threshold=ev_threshold, prob_threshold=prob_threshold)
        for (_, row), p in zip(df_predict.iterrows(), prob_favorite)
    ])
    details['Prob. favorite (raw model)'] = [round(float(p), 4) for p in raw_prob]
    details['Lambda book'] = round(float(lambda_book), 2)
    stakes = [
        kelly_stake(ev, odds, KELLY_REFERENCE_CAPITAL) if bet == 'YES' else 0.0
        for ev, odds, bet in zip(details['EV chosen'], details['Odds chosen'], details['Bet'])
    ]
    details['Stake'] = [round(s, 4) for s in stakes]

    if true_outcomes:
        keys = match_keys(df_predict['Date'], df_predict['Winner'], df_predict['Loser'])
        details['Actual winner'] = [true_outcomes.get(k) for k in keys]
        won = details['Player'] == details['Actual winner']
        details['Outcome'] = ['WON' if w else 'LOST' for w in won]
        details['Profit'] = [
            round(stake * ((odds - 1) if w else -1.0), 2) if bet == 'YES' and stake > 0 else 0.0
            for stake, odds, w, bet in zip(stakes, details['Odds chosen'], won, details['Bet'])
        ]

    details.to_excel(output_file, index=False)
    bets = details[details['Bet'] == 'YES']
    print(f"\nMatches analysed: {len(details)} | Recommended bets: {len(bets)}")
    if true_outcomes and len(bets):
        profit, staked = bets['Profit'].sum(), float(bets['Stake'].sum())
        roi = f"{profit / staked * 100:.2f}%" if staked else "n/a"
        won = int((bets['Outcome'] == 'WON').sum())
        print(
            f"Check against the actual outcome: {won}/{len(bets)} won | "
            f"profit {profit:.2f} | staked {staked:.2f} | ROI {roi}"
        )
    print(f"Written {output_file}")
    return details
