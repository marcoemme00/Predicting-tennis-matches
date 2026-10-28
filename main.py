"""
Tennis betting: build the dataset, backtest the models year by year and, if asked, predict the next matches.

SELECTION TECHNIQUE
Bet selection does not aim at maximising accuracy (guessing as many matches as possible) but the
profit of the bets. A bet is placed only when its expected value (EV), computed from the probability
predicted by the model, is above a preset threshold. Working on the ratio between the model
probability and the one implied by the bookmaker's odds gives the same result.

The stake is either fixed or, through the Kelly formula, a function of the EV and of the capital.
A protection (shadow mode) suspends the bets when the capital falls below stop_loss_limit; betting
resumes when the profit the model would have made during the suspension exceeds a threshold.

Features switched off because they lowered the ROI (all in config.py):
- SHRINK_TO_BOOK: blend the model probability with the bookmaker's.
- SELECTION_METHOD='roi': stepwise selection of the features by the profit their addition would have
  produced. Standard selection (Elastic Net for the logistic model, gain for XGBoost) did better.
- restricting bets to matches with odds in specific ranges, or excluding underdogs.
- STOP_ON_REMAINING_CAPITAL: after a recovery, apply stop_loss_limit to the capital left.
- N_GRACE_BETS: after leaving shadow mode, place a minimum number of bets before the stop can trigger again.
"""

import importlib
import os
import sys
import warnings
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
from functools import partial
from io import StringIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import delayed
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
pd.set_option("display.max_rows", 100)
pd.set_option('display.max_columns', None)

####################################
# WHAT TO RUN
####################################
backtest = True            # test the models on historical data
PLOT_CANDIDATES = False    # save the plots of the candidate features and of every numeric column
predictions = False        # predict the upcoming matches

# Which seasons to simulate, and how often the model is refitted inside each year.
test_years = list(range(2016, 2027))   # 2016 through 2026
#refit_every, refit_unit = 1, 'months'  # either calendar months or days
refit_every, refit_unit = 1, 'months'  # either calendar months or days

# Parameters of tennis_explorer_next_matches.py, used only if predictions = True.
te_days = 3                        # days to predict, today included. Missing days cost nothing in computing time.
te_max_matches = 50                # matches to predict. Only the existing ones are downloaded.
te_out = "upcoming_matches.xlsx"   # where the Tennis Explorer data is saved
te_include_challenger = False      # keep False: the models are trained on the ATP tour and Grand Slams only, so it is better to exclude minor tiers like Challenger.

SHUTDOWN_AFTER = False     # power off the pc when the script is done
SHUTDOWN_DELAY_MINUTES = 1

# Mirror the prints to a text file. The parallel year×model jobs keep their text in memory and the
# parent writes it to the file in order once the pool is done, so the log stays readable.
LOG_TO_FILE = True
LOG_DIR = "backtest_results"

import config
import auxiliary
import modeling
# reloading the modules to load eventual last changes
importlib.reload(config)
importlib.reload(auxiliary)
importlib.reload(modeling)
from auxiliary import (
    _Tee, _append_job_log, _pause_run_log_file, _resume_run_log_file, close_run_log,
    concat_details_for_model, drop_old_workers, init_run_log, load_full_environment, parallel_workers, play_beep,
    print_bootstrap_summary, run_parallel_batch, save_full_environment, schedule_shutdown,
)
from config import (
    CANDIDATE_FEATURES, EV_THRESHOLD, N_GRACE_BETS, PROB_THRESHOLD, SELECTION_METHOD, SHRINK_TO_BOOK,
    STOP_LOSS_LIMIT, STOP_ON_REMAINING_CAPITAL, WINDOW_YEARS,
)
from dataset_preparation import (
    build_dataset, check_dataset, clean_dataset, load_support_tables, merge_excel_files, recompute_dynamic_features,
)
from modeling import (
    bet_details, build_calibrated_pipeline, fit_with_shrinkage, kelly_stake, select_features_standard,
    stepwise_selection_roi,
)
drop_old_workers()

init_run_log(LOG_TO_FILE, LOG_DIR)

# load_full_environment()   
# look at the last saved results without re-running


####################################
# DATASET
####################################
matches = merge_excel_files()   # download any missing season workbook, then stack the years
if matches is None:
    raise ValueError("No valid Excel file found in the folder")
players, city_country, country_codes = load_support_tables()   # players bio, city. From city I lookup in states.xlsx for country and IOC codes

dataset = build_dataset(matches, players, city_country, country_codes)
check_dataset(dataset)
dataset = clean_dataset(dataset)


####################################
# EXPLORATORY PLOTS
####################################
def columns_to_plot(df):
    """Every numeric column, including the ones built in the code; dates and ids excluded."""
    def plottable(name):
        low = str(name).lower()
        if low == "date" or low.endswith("_id") or low.startswith(("player_id", "dob")) or "wikidata" in low:
            return False
        filled = df[name].notna().sum()
        return filled > 0 and pd.to_numeric(df[name], errors="coerce").notna().sum() >= 0.5 * filled
    return [c for c in df.columns if plottable(c)]


def _panel(n, cols=4):
    """A grid of n small axes; the unused ones are switched off."""
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.6 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")
    return fig, axes[:n]


def _save(fig, path, title):
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_candidate_features(df, features, out_dir="exploratory", stem="candidate", title="Candidate features", correlation=True):
    """
    Three figures for the columns passed in:
    - distributions: bars for discrete columns (<= 8 distinct values), histograms otherwise;
    - favourite win rate by value, or by quantile bin for the continuous columns;
    - Spearman correlation heatmap (skipped if correlation=False or more than 30 columns, for readability).
      Spearman rather than Pearson for a quick look at the data: Pearson measures a straight line and is
      moved a lot by a few outliers (the underdog odds reach 30-40 while most sit around 2), and it does not
      suit the ordinal columns (fRank, uRank, fPts_class, uPts_class) or the 0/1 flags (home, different_hand,
      surface_rec_*). Spearman only asks whether the two orderings move together, so every column fits in
      one matrix. On the continuous ones (age, height, Elo) the two coefficients are close anyway.
    """
    present = [c for c in features if c in df.columns]
    missing = [c for c in features if c not in df.columns]
    if missing:
        print(f"Candidate plots: columns not in the dataset: {missing}")
    if not present:
        return
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    numeric = df[present].apply(pd.to_numeric, errors="coerce")

    fig, axes = _panel(len(present))
    for ax, name in zip(axes, present):
        values = numeric[name].dropna()
        if values.nunique() <= 8:
            counts = values.value_counts().sort_index()
            ax.bar(counts.index.astype(str), counts.to_numpy(), color="#4C78A8")
        else:
            ax.hist(values, bins=30, color="#4C78A8", edgecolor="white")
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
    _save(fig, out / f"{stem}_distributions.png", title)

    # share of matches the favourite actually won, split by the value of the feature (on the x-axis)
    if "favorite_won" in df.columns:
        target = pd.to_numeric(df["favorite_won"], errors="coerce")
        fig, axes = _panel(len(present))
        for ax, name in zip(axes, present):
            frame = pd.DataFrame({"x": numeric[name], "y": target}).dropna()
            ax.set_title(name, fontsize=9)
            if frame.empty:
                continue
            discrete = frame["x"].nunique() <= 8
            #if less then 8, one bar per values is shown. If more, the variable is split in 8 bins.
            groups = frame["x"] if discrete else pd.qcut(frame["x"], q=min(8, frame["x"].nunique()), duplicates="drop")
            rate = frame.groupby(groups, observed=True)["y"].mean()
            labels = [f"{g:g}" if discrete else f"{g.left:.2g}" for g in rate.index]
            ax.bar(range(len(rate)), rate.to_numpy(), color="#F58518")
            ax.set_xticks(range(len(rate)), labels, rotation=45, ha="right", fontsize=6)
            ax.set_ylim(0, 1)
            ax.axhline(frame["y"].mean(), color="gray", linewidth=0.8)
            ax.tick_params(labelsize=7)
        _save(fig, out / f"{stem}_win_rate.png", f"Favourite win rate by {title.lower()}")

    if correlation and len(present) <= 30:
        fig, ax = plt.subplots(figsize=(10, 8))
        image = ax.imshow(numeric.corr(method="spearman").to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(present)), present, rotation=90, fontsize=7)
        ax.set_yticks(range(len(present)), present, fontsize=7)
        fig.colorbar(image, ax=ax, fraction=0.046)
        _save(fig, out / f"{stem}_correlation.png", f"Spearman correlation of {title.lower()}")
    print(f"{title} plots saved in {out.resolve()} ({len(present)} columns)")


if PLOT_CANDIDATES:
    plot_candidate_features(dataset, CANDIDATE_FEATURES)
    plot_candidate_features(dataset, columns_to_plot(dataset), stem="all_variables", title="All numeric columns", correlation=False)


####################################
# BACKTEST
####################################
# One year of betting at a time. Every <step> <step_unit> (example 1 month) the dynamic features (Elo, win %) are refreshed,
# the features re-selected and the model refitted on the previous window_years, then applied to the period.
# 'months' cuts on calendar months, 'days' every <step> days. I observe that calendar months beat 30-day steps empirically.
PERIOD_FREQ = {'months': 'MS', 'days': 'D'}


def refit_periods(year, step, unit):
    """(start, end) of every refit period of the year."""
    first, last = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")
    length = pd.DateOffset(months=step) if unit == 'months' else pd.Timedelta(days=step)
    starts = pd.date_range(first, last, freq=f"{step}{PERIOD_FREQ[unit]}")
    return [(start, min(start + length - pd.Timedelta(days=1), last)) for start in starts]


class Bankroll:
    """
    This class provides the functions to calculate the capital of one year, moved up or down by the profit or
    loss of each bet, with the protection rules.
    rolling_window_backtest creates the object bankroll at the start of the year with the initial capital and
    calls record() after every match of every period, so the capital and the shadow mode carry over from one
    period to the next.
    In real mode the bets move the capital; when it falls to stop_loss_limit of the reference capital
    (e.g. 70 euro from 100) shadow mode begins.
    In shadow mode the bets are only virtual; when their profit reaches restart_threshold real betting
    resumes, with an optional grace of n_grace_bets during which the stop cannot trigger again.
    The stop is checked only after a placed bet. Capital moves only then, so the first trip into
    shadow mode is the same either way. After a recovery the real capital is still under the floor,
    but a match with no bet does not send it back into shadow mode.
    """

    def __init__(self, capital, stop_loss_limit, stop_on_remaining, n_grace_bets, restart_threshold=0.5):
        self.capital = capital
        self.floor = capital * stop_loss_limit
        self.stop_loss_limit = stop_loss_limit
        self.stop_on_remaining = stop_on_remaining
        self.n_grace_bets = n_grace_bets
        self.restart_threshold = restart_threshold
        self.shadow = False
        self.virtual_profit = 0.0
        self.grace_left = 0

    def mode(self):
        return 'shadow' if self.shadow else 'real'

    def record(self, placed, profit):
        """Called on every analysed match: book the profit if a bet was placed, then apply the stop / restart rules."""
        if self.shadow:
            if placed:
                self.virtual_profit += profit
            if placed and self.virtual_profit >= self.restart_threshold:
                grace = f" (grace {self.n_grace_bets} bets)" if self.n_grace_bets else ""
                print(f"      RECOVERY CONFIRMED (+{self.virtual_profit:.2f} virtual). Real betting resumes{grace}.")
                self.shadow = False
                if self.stop_on_remaining:
                    self.floor = self.capital * self.stop_loss_limit
                self.grace_left = self.n_grace_bets
            return

        if placed:
            self.capital += profit
            if self.grace_left:
                self.grace_left -= 1
                if self.grace_left == 0:
                    print(f"      End of grace ({self.n_grace_bets} bets). Stop-loss active again.")
        if placed and not self.grace_left and self.capital <= self.floor:
            print(f"      [!] ANOMALY: capital at {self.capital:.2f}. Entering Shadow Mode (virtual betting).")
            self.shadow = True
            self.virtual_profit = 0.0


def rolling_window_backtest(df, base_model, candidate_features, test_year, window_years=2, step=1, step_unit='months',
                            initial_capital=100, selection_method='standard', stop_loss_limit=STOP_LOSS_LIMIT,
                            stop_on_remaining=STOP_ON_REMAINING_CAPITAL, n_grace_bets=N_GRACE_BETS,
                            ev_threshold=EV_THRESHOLD, prob_threshold=PROB_THRESHOLD):
    """One year of betting with one model. Returns a row per analysed match, bet or not."""
    if step_unit not in PERIOD_FREQ or step < 1:
        raise ValueError(f"step_unit must be 'months' or 'days' and step >= 1 (got {step!r} {step_unit!r})")
    if selection_method not in ('standard', 'roi'):
        raise ValueError(f"selection_method must be 'standard' or 'roi' (got {selection_method!r})")
    print(f"\nStarting every-{step}-{step_unit} backtest (window: {window_years} years, method: {selection_method}) for {test_year}...")

    bankroll = Bankroll(initial_capital, stop_loss_limit, stop_on_remaining, n_grace_bets)
    rows = []
    for start, end in refit_periods(test_year, step, step_unit):
        #refit_period give the periods (for example:  (1 Jan, 31 Jan), (1 Feb, 28 Feb), etc.)
        period = recompute_dynamic_features(df, start, end)   # Elo and win % as known at the period start
        df_train = period[(period['Date'] >= start - pd.DateOffset(years=window_years)) & (period['Date'] < start)]
        df_test = period[(period['Date'] >= start) & (period['Date'] <= end)]
        if df_train.empty or df_test.empty:
            continue
        label = f"{start:%d %b} - {end:%d %b}"
        print(f"   -> Optimising period: {label}")

        model = clone(base_model)
        if selection_method == 'roi':
            features, _ = stepwise_selection_roi(df_train, candidate_features, model,
                                                 ev_threshold=ev_threshold, prob_threshold=prob_threshold)
        else:
            features = select_features_standard(df_train, candidate_features, model)
        if not features:
            print("      Skipping period: no feature selected. Capital preserved.")
            continue
        print(f"      Selected features ({selection_method}): {features}")

        # fit, estimate lambda (only with SHRINK_TO_BOOK) and predict the period
        _, probabilities, lambda_book, raw_probabilities = fit_with_shrinkage(
            build_calibrated_pipeline(model), df_train, df_test, features)
        if probabilities is None:
            print("      Skipping period: no training row left after the NaN filter.")
            continue
        if SHRINK_TO_BOOK:
            print(f"      Shrinkage to book: lambda={lambda_book:.2f}")

        for (_, match), prob, raw in zip(df_test.iterrows(), probabilities, raw_probabilities):
            details = bet_details(match, prob, ev_threshold=ev_threshold, prob_threshold=prob_threshold)
            placed = details['Bet'] == 'YES'
            won = (details['Side'] == 'Favorite') == bool(match['favorite_won'])
            stake = kelly_stake(details['EV chosen'], details['Odds chosen'], bankroll.capital) if placed else 0.0
            profit = stake * ((details['Odds chosen'] - 1) if won else -1)
            details |= {
                'Prob. favorite (raw model)': round(float(raw), 4),
                'Lambda book': round(float(lambda_book), 2),
                'Actual winner': match['Favorite'] if match['favorite_won'] else match['Underdog'],
                'Outcome': 'WON' if won else 'LOST',
                'Stake': round(stake, 4),
                'Mode': bankroll.mode(),   # did the stake move the real capital or was it only virtual?
                'Profit': round(profit, 2),
                'Period': label,
            }
            bankroll.record(placed, profit)
            details['Capital'] = round(bankroll.capital, 2)
            rows.append(details)
    return pd.DataFrame(rows)


def summarize_backtest_details(details, year, model_name, initial_capital):
    """
    One summary row per year and model. It returns a dictionary, which is stacked with the others
    and printed as a row of the final summary table.
    Example: {'Year': 2019, 'Model': 'Logistic_Regression', 'Matches analysed': 2517,
              'Bets placed': 34, 'Staked': 34.0, 'Profit': 4.96, 'Final capital': 104.96,
              'Return on stake (%)': 14.59}
    """
    if details.empty:
        real, final_capital = pd.DataFrame(columns=['Stake', 'Profit']), initial_capital
    else:
        real = details[(details['Bet'] == 'YES') & (details['Mode'] == 'real')]
        final_capital = details['Capital'].iloc[-1]
    staked, profit = float(real['Stake'].sum()), float(real['Profit'].sum())
    # profit per euro staked. With a flat 1 euro stake it is also the average return per bet.
    roi = profit / staked * 100 if staked > 0 else np.nan
    return {
        'Year': year, 'Model': model_name, 'Matches analysed': len(details), 'Bets placed': len(real),
        'Staked': round(staked, 2), 'Profit': round(profit, 2), 'Final capital': round(final_capital, 2),
        'Return on stake (%)': round(roi, 2),
    }


def backtest_year_model(df, year, model_name, base_model, **settings):
    """
    One independent job (one year, one model), safe to run in parallel.
    Its prints hit the console while the job is running, so lines from different jobs mix
    (a 2019 XGBoost line can appear before a 2016 Logistic Regression line).
    At the end of the job the same text comes back as one block, and the parent writes those blocks to the log file in job order.
    """
    log = StringIO()
    with redirect_stdout(_Tee(sys.__stdout__, log)), redirect_stderr(_Tee(sys.__stderr__, log)):
        print(f"\n{'=' * 60}\n SIMULATION YEAR: {year} | {model_name} "
              f"(refit every {settings['step']} {settings['step_unit']}, method: {settings['selection_method']})\n{'=' * 60}")
        details = rolling_window_backtest(df, base_model, test_year=year, **settings)
        row = summarize_backtest_details(details, year, model_name, settings['initial_capital'])
    return row, details, log.getvalue()


def run_backtest_summary(df, base_models, candidate_features, test_years, window_years=2, step=1, step_unit='months',
                         initial_capital=100, selection_method='standard', details_folder='backtest_results',
                         ev_threshold=EV_THRESHOLD, prob_threshold=PROB_THRESHOLD):
    """
    This is the main function: it calculates the backtest, in parallel with the standard selection, and returns the results. With the ROI selection the jobs
    run one at a time and the feature combinations are parallelized inside stepwise_selection_roi.
    Returns the summary table and {year: {model: match details}}; the details also go to Excel.
    """
    jobs = [(year, name, model) for year in test_years for name, model in base_models.items()]
    job = partial(backtest_year_model, df, candidate_features=candidate_features, window_years=window_years,
                  step=step, step_unit=step_unit, initial_capital=initial_capital, selection_method=selection_method,
                  ev_threshold=ev_threshold, prob_threshold=prob_threshold)
    n_jobs = parallel_workers() if selection_method == 'standard' else 1
    print(f"backtest {selection_method} | jobs: {len(jobs)} ({len(test_years)} years × {len(base_models)} models) | "
          f"cores: {n_jobs} | refit every {step} {step_unit}")

    if n_jobs == 1 or len(jobs) == 1:
        outputs = []
        for args in jobs:
            outputs.append(job(*args))
            _append_job_log(outputs[-1][2])
    else:
        # the workers buffer their prints: written in real time, the texts would interleave in the log
        _pause_run_log_file()
        try:
            outputs = run_parallel_batch(n_jobs, [delayed(job)(*args) for args in jobs])
        finally:
            _resume_run_log_file()
        for _, _, text in outputs:
            _append_job_log(text)

    summary = pd.DataFrame([row for row, _, _ in outputs])
    details_by_year = defaultdict(dict)
    for (year, name, _), (_, details, _) in zip(jobs, outputs):
        details_by_year[year][name] = details

    if details_folder:
        Path(details_folder).mkdir(parents=True, exist_ok=True)
        for year, by_model in details_by_year.items():
            sheets = {name: d for name, d in by_model.items() if not d.empty}
            if not sheets:
                continue
            path = Path(details_folder) / f"backtest_{year}_{selection_method}.xlsx"
            with pd.ExcelWriter(path, engine='openpyxl') as writer:
                for name, d in sheets.items():
                    d.to_excel(writer, sheet_name=name[:31], index=False)
            print(f"\n   Match details for {year} written to {path}")
    return summary, dict(details_by_year)


# === AND FINALLY... how much would I have earned over these years, betting for real? ===
if backtest:
    models_to_test = {
        'Logistic_Regression': LogisticRegression(random_state=42, max_iter=1000),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42, eval_metric='logloss'),
    }
    summary, details_by_year = run_backtest_summary(
        dataset, models_to_test, CANDIDATE_FEATURES, test_years, window_years=WINDOW_YEARS,
        step=refit_every, step_unit=refit_unit, initial_capital=100, selection_method=SELECTION_METHOD,
    )

    print(f"\n{'*' * 80}\n SUMMARY TABLE - refit every {refit_every} {refit_unit} | method: {SELECTION_METHOD.upper()}\n{'*' * 80}")
    print(summary.pivot(index='Year', columns='Model', values=['Return on stake (%)', 'Final capital', 'Bets placed']))

    # total profit / total stake over all the years, real bets only
    print(f"\n{'=' * 80}\n RETURN ON THE MONEY STAKED (profit / euros staked)\n{'=' * 80}")
    for model, t in summary.groupby('Model', sort=False)[['Staked', 'Profit']].sum().iterrows():
        result = f"{t.Profit / t.Staked * 100:+.2f}%  (profit {t.Profit:.2f} on {t.Staked:.2f} staked)" if t.Staked > 0 else "no stake"
        print(f"Model {model:<20} -> {result}")

    # the benchmark is betting 1 euro on the B365 favourite of every match without Kelly and stop.
    print(f"\n{'=' * 80}\n BOOKMAKER BENCHMARK - always on the lowest odds (1 euro per match)\n{'=' * 80}")
    book = dataset[dataset['Date'].dt.year.isin(test_years) & dataset['favorite_won'].notna()]
    fav_odds = pd.to_numeric(book['fB365'], errors='coerce')
    book_profit = pd.Series(np.where(book['favorite_won'].astype(bool), fav_odds - 1, -1.0), index=book.index)
    for year, g in book_profit.groupby(book['Date'].dt.year):
        print(f"{year}  matches {len(g):5d}  return on stake {g.sum() / len(g) * 100:+7.2f}%")
    if len(book_profit):
        print(f"Bookmaker              -> {book_profit.sum() / len(book_profit) * 100:+.2f}% on the money staked "
              f"(profit {book_profit.sum():.2f} on {len(book_profit)} staked)")

    # is the model overconfident? On the YES bets: mean model probability, share of bets won and the
    # bookmaker's vig-free probability on the same side (every bet weighs 1).
    print(f"\n{'=' * 80}\n MODEL PROBABILITY AND OBSERVED FREQUENCY (YES bets)\n{'=' * 80}")
    for model_name in models_to_test:
        details = concat_details_for_model(details_by_year, model_name)
        yes = details[details['Bet'] == 'YES'] if not details.empty else details
        if yes.empty:
            print(f"{model_name}: no bets")
            continue
        odds_fav, odds_und = (pd.to_numeric(yes[c], errors='coerce') for c in ['Odds favorite', 'Odds underdog'])
        p_fav, p_und = 1 / odds_fav.where(odds_fav > 1), 1 / odds_und.where(odds_und > 1)
        p_book = p_und.where(yes['Side'] == 'Underdog', p_fav) / (p_fav + p_und)
        ok = p_book.notna() & yes['Prob. chosen'].notna()
        p_mod, won = yes.loc[ok, 'Prob. chosen'].mean(), (yes.loc[ok, 'Outcome'] == 'WON').mean()
        print(f"{model_name:<22} n={ok.sum():5d}  model {p_mod * 100:5.1f}%  bookmaker {p_book[ok].mean() * 100:5.1f}%  "
              f"won {won * 100:5.1f}%  (model - won {(p_mod - won) * 100:+.1f} points)")

    # bootstrap 95% CI on the return per euro
    print_bootstrap_summary(details_by_year, sheets=list(models_to_test),
                            book_profits=book_profit.to_numpy() if len(book_profit) else None)


####################################
# and finally, finally... the predictions!
####################################
if predictions:
    # reloading the modules to load eventual last changes
    import tennis_explorer_next_matches
    import prediction
    importlib.reload(tennis_explorer_next_matches)
    importlib.reload(prediction)
    from prediction import predict_upcoming_matches
    tennis_explorer_next_matches.main(days=te_days, max_matches=te_max_matches, out=te_out, include_challenger=te_include_challenger)
    upcoming_predictions = predict_upcoming_matches(upcoming_file=te_out, window_years=WINDOW_YEARS, selection_method=SELECTION_METHOD)


save_full_environment()
if SHUTDOWN_AFTER:
    schedule_shutdown(SHUTDOWN_DELAY_MINUTES)
else:
    play_beep()
close_run_log()
