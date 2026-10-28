"""Model fitting, feature selection, betting decision helpers."""

import numpy as np
import pandas as pd
from joblib import delayed
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from auxiliary import one_core_estimator, parallel_workers, run_parallel_batch
from config import (
    BOOK_PROB_COL,
    ESTIMATE_L1_RATIO,
    EV_THRESHOLD,
    KELLY_FRACTION,
    MAX_STAKE_FRACTION,
    PROB_THRESHOLD,
    SHRINK_MIN_VAL_ROWS,
    SHRINK_TO_BOOK,
    SHRINK_VAL_FRACTION,
    USE_KELLY,
)

####################################
# betting decision
####################################

def evaluate_bet(prob_favorite, odds_favorite, odds_underdog,
                 ev_threshold=EV_THRESHOLD, prob_threshold=PROB_THRESHOLD):
    """
    Given the model probability of the favourite, decide whom to bet on.

    Expected net gain (EV) = win probability * (odds - 1) - loss probability.
    We bet when the EV is above ev_threshold and the probability of the chosen outcome
    is above prob_threshold. If both sides qualify, the favourite is preferred.
    The side is always recorded, even when we don't bet: it says who would have been the better bet.
    """
    prob_underdog = 1 - prob_favorite
    ev_favorite = prob_favorite * (odds_favorite - 1) - prob_underdog
    ev_underdog = prob_underdog * (odds_underdog - 1) - prob_favorite

    favorite_ok = ev_favorite > ev_threshold and prob_favorite >= prob_threshold
    underdog_ok = ev_underdog > ev_threshold and prob_underdog >= prob_threshold
    if favorite_ok or underdog_ok:
        side = 'Favorite' if favorite_ok else 'Underdog'
    else:
        side = 'Favorite' if ev_favorite >= ev_underdog else 'Underdog'

    prob, odds, ev = (
        (prob_favorite, odds_favorite, ev_favorite) if side == 'Favorite'
        else (prob_underdog, odds_underdog, ev_underdog)
    )
    return {
        'bet': favorite_ok or underdog_ok,
        'side': side,
        'prob_chosen': prob,
        'odds_chosen': odds,
        'ev_chosen': ev,
        'ev_favorite': ev_favorite,
        'ev_underdog': ev_underdog,
    }


def flat_stake_profits(probs, odds_fav, odds_und, favorite_won,
                       ev_threshold=EV_THRESHOLD, prob_threshold=PROB_THRESHOLD):
    """
    Profit of a 1 euro bet on every match, following evaluate_bet.
    Returns the profits and a mask saying which matches pass the betting thresholds. It is not used for the yearly roi calculation,
    just used in the model tuning.
    """
    choices = [evaluate_bet(p, f, u, ev_threshold, prob_threshold)
               for p, f, u in zip(probs, odds_fav, odds_und)]
    won = [(c['side'] == 'Favorite') == bool(y) for c, y in zip(choices, favorite_won)]
    profits = np.array([c['odds_chosen'] - 1 if w else -1.0 for c, w in zip(choices, won)])
    bet = np.array([c['bet'] for c in choices], dtype=bool)
    return profits, bet


def kelly_stake(ev, odds, capital, fraction=KELLY_FRACTION, max_fraction=MAX_STAKE_FRACTION):
    """
    Kelly stake: capital * ev / (odds - 1). For prudence only a fraction of it is staked,
    capped at capital * max_fraction. With USE_KELLY=False the stake is a fixed 1 euro.
    """
    if not USE_KELLY:
        return 1.0
    if capital <= 0 or not np.isfinite(ev) or not np.isfinite(odds) or ev <= 0 or odds <= 1:
        return 0.0
    f = fraction * ev / (odds - 1)
    if max_fraction is not None:
        f = min(f, max_fraction)
    return float(min(f, 1.0) * capital)


def bet_details(row, prob_favorite, ev_threshold=EV_THRESHOLD, prob_threshold=PROB_THRESHOLD):
    """One row of the Excel output: the decision taken on a match (the chosen side is filled even without a bet)."""
    choice = evaluate_bet(prob_favorite, row['fB365'], row['uB365'], ev_threshold, prob_threshold)
    prob_book = row['fprob_B365_norm']
    return {
        'Date': pd.to_datetime(row['Date']).date(),
        'Tournament': row.get('Tournament', ''),
        'Favorite': row['Favorite'],
        'Underdog': row['Underdog'],
        'Odds favorite': row['fB365'],
        'Odds underdog': row['uB365'],
        'Prob. favorite (model)': round(prob_favorite, 4),
        'Prob. favorite (bookmaker)': round(float(prob_book), 4) if pd.notna(prob_book) else np.nan,
        'EV favorite': round(choice['ev_favorite'], 4),
        'EV underdog': round(choice['ev_underdog'], 4),
        'Bet': 'YES' if choice['bet'] else 'NO',
        'Side': choice['side'],
        'Player': row['Favorite'] if choice['side'] == 'Favorite' else row['Underdog'],
        'Prob. chosen': round(choice['prob_chosen'], 4),
        'Odds chosen': choice['odds_chosen'],
        'EV chosen': round(choice['ev_chosen'], 4),
    }


####################################
# model fitting, with optional shrinkage towards the bookmaker
####################################

def build_calibrated_pipeline(base_model):
    """
    impute -> scale -> calibrated classifier. Used by the backtest, the live predictions
    and the ROI feature selection, so the three always share the same model.

    The calibration (sigmoid) matters because the raw probabilities of a classifier could be
    far from real frequencies: here they are compared with the bookmaker's, so they must be
    interpretable as true win probabilities.
    one_core_estimator: xgboost would grab every core, but years and models are already run in parallel outside.
    """
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', CalibratedClassifierCV(
            estimator=one_core_estimator(base_model), method='sigmoid',
            cv=TimeSeriesSplit(n_splits=3), n_jobs=1,
        )),
    ])


def estimate_lambda_to_book(y, p_mod, p_book, odds_fav, odds_und,
                            ev_threshold=EV_THRESHOLD, prob_threshold=PROB_THRESHOLD, grid=None):
    """
    Pick the lambda of the blend (1-lambda) * p_model + lambda * p_book that gives the best
    flat-stake profit on the validation matches. Only matches passing the betting thresholds count.
    When two lambdas are linked to the same profit, the lower is chosen (less shrinkage). Returns (lambda, bets placed, profit).
    """
    arrays = [np.asarray(a, dtype=float) for a in (y, p_mod, p_book, odds_fav, odds_und)]
    ok = np.all(np.isfinite(arrays), axis=0)
    y, p_mod, p_book, odds_fav, odds_und = (a[ok] for a in arrays)
    if len(y) < SHRINK_MIN_VAL_ROWS:
        return 0.0, 0, 0.0

    results = []
    for lam in (np.linspace(0, 1, 11) if grid is None else grid):
        profits, bet = flat_stake_profits((1 - lam) * p_mod + lam * p_book, odds_fav, odds_und, y,
                                          ev_threshold, prob_threshold)
        results.append((float(profits[bet].sum()), float(lam), int(bet.sum())))
    profit, lam, n_bet = max(results, key=lambda r: (r[0], -r[1]))
    return lam, n_bet, profit


def apply_shrinkage_to_book(p_mod, p_book, lam):
    """Blend model and bookmaker probabilities; where the bookmaker's is missing keep the model's."""
    p_mod, p_book = np.asarray(p_mod, dtype=float), np.asarray(p_book, dtype=float)
    return np.where(np.isfinite(p_book), (1 - lam) * p_mod + lam * p_book, p_mod)


def fit_with_shrinkage(pipeline, df_train, df_pred, features, book_col=BOOK_PROB_COL):
    """
    Fit the pipeline on df_train and predict df_pred.
    With SHRINK_TO_BOOK, lambda is first estimated on the last SHRINK_VAL_FRACTION of the training
    set (in time) with a model fitted on the rest; then the model is refitted on everything.
    Returns (pipeline, probabilities used, lambda, raw model probabilities).
    """
    df_train = df_train.sort_values('Date').dropna(subset=['favorite_won'])
    if df_train.empty:
        return pipeline, None, 0.0, None

    lam = 0.0
    if SHRINK_TO_BOOK:
        cut = int(len(df_train) * (1 - SHRINK_VAL_FRACTION))
        df_fit, df_val = df_train.iloc[:cut], df_train.iloc[cut:]
        has_book = {book_col, 'fB365', 'uB365'} <= set(df_val.columns)
        if not df_fit.empty and len(df_val) >= SHRINK_MIN_VAL_ROWS and has_book:
            pipeline.fit(df_fit[features], df_fit['favorite_won'])
            p_val = pipeline.predict_proba(df_val[features])[:, 1]
            lam, n_bet, profit = estimate_lambda_to_book(
                df_val['favorite_won'], p_val, df_val[book_col], df_val['fB365'], df_val['uB365'])
            print(f"      Shrinkage val: lambda={lam:.2f} on {n_bet} bets (PL {profit:+.1f} on {len(df_val)} matches)")

    pipeline.fit(df_train[features], df_train['favorite_won'])
    p_raw = pipeline.predict_proba(df_pred[features])[:, 1]
    p_used = p_raw
    if SHRINK_TO_BOOK and book_col in df_pred.columns:
        p_used = apply_shrinkage_to_book(p_raw, df_pred[book_col], lam)
    return pipeline, p_used, lam, p_raw


####################################
# feature selection, approach 1: stepwise on the ROI
####################################

def roi_of_features(df, features, base_model, ev_threshold=EV_THRESHOLD, prob_threshold=PROB_THRESHOLD, cv_folds=5):
    """
    Average flat-stake profit per match of this feature set, over a time-series cross validation.
    I bet on every match, not only on those passing the thresholds: filtering made the search
    too sensitive to noise (a feature would enter because of 20-30 matches with a stellar, random ROI).
    """
    df = df.dropna(subset=['favorite_won'])
    if len(df) < 50:
        return -np.inf
    x, y = df[features], df['favorite_won']
    cv = TimeSeriesSplit(n_splits=min(cv_folds, max(2, len(df) // 20)))
    #timesseriesplit is useful to avoid taking random splits. It relies on the order of the data, the time variable time is not declared.
    pipeline = build_calibrated_pipeline(base_model)
    try:
        fold_rois = []
        for train_idx, test_idx in cv.split(x):
            pipeline.fit(x.iloc[train_idx], y.iloc[train_idx])
            probs = pipeline.predict_proba(x.iloc[test_idx])[:, 1]
            test = df.iloc[test_idx]
            profits, _ = flat_stake_profits(probs, test['fB365'], test['uB365'], test['favorite_won'],
                                            ev_threshold, prob_threshold)
            fold_rois.append(profits.mean())
        return float(np.mean(fold_rois))
    except Exception as e:
        print(f"[WARNING] Feature evaluation failed ({e}). Skipping the combination.")
        return -np.inf


def stepwise_selection_roi(df_train, candidate_features, base_model,
                           ev_threshold=EV_THRESHOLD, prob_threshold=PROB_THRESHOLD,
                           max_features=15, n_jobs=None):
    """
    Bidirectional stepwise search driven by the ROI rather than by significance:
    at each step add the feature that improves the ROI most, then drop one whose removal
    does not lower it. Returns (selected features, their ROI).
    """
    df = df_train.sort_values('Date').reset_index(drop=True)
    remaining = [f for f in candidate_features if f in df.columns]
    n_jobs = parallel_workers(n_jobs)
    print(f"start | cores: {n_jobs}")

    def score(feature_sets):
        # one process per feature combination; the output keeps the input order, so ties are stable
        if n_jobs == 1 or len(feature_sets) == 1:
            #if I have only one core> standard list comprehension
            return [roi_of_features(df, feats, base_model, ev_threshold, prob_threshold) for feats in feature_sets]
        tasks = [delayed(roi_of_features)(df, feats, base_model, ev_threshold, prob_threshold) for feats in feature_sets]
        return run_parallel_batch(n_jobs, tasks)

    selected, best_roi = [], -np.inf
    for step in range(1, max_features + 1):
        # forward
        feature, roi = max(zip(remaining, score([selected + [f] for f in remaining])), key=lambda fr: fr[1])
        if roi <= best_roi:
            print("No additional feature improves the ROI. Search over.")
            break
        selected.append(feature)
        remaining.remove(feature)
        best_roi = roi
        print(f"[{step}] FORWARD  | Added: {feature} | ROI: {roi * 100:.2f}%")

        # backward: ">=" so that a feature whose removal leaves the ROI unchanged goes too (simpler model)
        if len(selected) > 1:
            without = [[f for f in selected if f != drop] for drop in selected]
            feature, roi = max(zip(selected, score(without)), key=lambda fr: fr[1])
            if roi >= best_roi:
                selected.remove(feature)
                remaining.append(feature)
                best_roi = roi
                print(f"[{step}] BACKWARD | Removed as redundant: {feature} | New ROI: {roi * 100:.2f}%")
    return selected, best_roi


####################################
# feature selection, approach 2: the models' own selection
####################################

def _above_mean(scores):
    """
    Keep the scores at or above the mean of every candidate, zeros included.
    When every score is zero the mean is zero and `0 >= 0` would keep the whole
    list, so that fit skips the period instead.
    """
    if not (scores > 1e-8).any():
        return []
    return scores[scores >= scores.mean()].index.tolist()


def select_features_standard(df_train, candidate_features, base_model, target_col='favorite_won'):
    """
    Logistic: Elastic Net on standardised features. Keep coefficients at or above the
    mean of every candidate. If the penalty has set every coefficient to zero, skip the period.
    XGBoost: same rule on gain.
    An empty selection means the period is skipped (see rolling_window_backtest).
    Unlike the ROI search, this rewards correct and confident predictions (likelihood), not profit.
    """
    available = [f for f in candidate_features if f in df_train.columns]
    df = df_train.dropna(subset=[target_col])
    if df.empty:
        return []
    x = SimpleImputer(strategy='median').set_output(transform='pandas').fit_transform(df[available])
    y = df[target_col].astype(int)

    if isinstance(base_model, LogisticRegression):
        # l1_ratio: 1 is lasso, 0 is ridge. Either fixed at 0.5 or chosen by cross validation.
        if ESTIMATE_L1_RATIO:
            model = LogisticRegressionCV(
                Cs=10, cv=TimeSeriesSplit(n_splits=3), solver='saga',
                l1_ratios=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0], scoring='accuracy',
                use_legacy_attributes=True, random_state=42, max_iter=2000,
            )
        else:
            model = LogisticRegression(solver='saga', l1_ratio=0.5, random_state=42, max_iter=2000)
        try:
            model.fit(StandardScaler().fit_transform(x), y)
        except Exception as e:
            print(f"      Warning, Elastic Net selection: {e}. Skipping the period.")
            return []
        coef = pd.Series(np.abs(model.coef_[0]), index=x.columns)
        return _above_mean(coef)

    if 'XGB' in type(base_model).__name__:
        try:
            gain = pd.Series(one_core_estimator(base_model).fit(x, y).feature_importances_, index=x.columns)
        except Exception as e:
            print(f"      Warning, XGBoost: {e}. Skipping the period.")
            return []
        return _above_mean(gain)

    return candidate_features
