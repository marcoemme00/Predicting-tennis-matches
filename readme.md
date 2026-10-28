# Predicting ATP matches and challenging the bookmaker
Can models built from publicly available tennis data beat the bookmakers?

The short answer is yes, but only by a narrow margin.

The simplest possible strategy, betting on the bookmaker's favourite, loses on average **-5.30%** over the eleven seasons from 2016 to 2026, roughly €5 lost for every €100 staked. This reveals the bookmaker's edge and I take this as a baseline.

I explore whether that edge can be overcome by combining four elements:
- **Modelling:** because the available feature set is relatively small, I focus on supervised classifiers like binomial models and XGBoost. In my experiments, greater complexity tends to overfit.
- **Feature selection:** I compare established methods such as Elastic Net with a custom stepwise procedure that selects features according to their incremental contribution to profitability.
- **Bet sizing:** I determine each stake by considering both expected return and risk.
- **Downside protection:** I use stop-loss mechanisms to limit losses when a model is no longer performing as expected.

These approaches can be combined through the switches in `config.py` and `main.py`. I evaluate the main configurations using a walk-forward backtest covering 2016–2026 and putting, in `config.py`, the best-performing one: estimated `l1_ratio`, flat 1 euro, with an average return of **+10.75%** per euro staked. Even so, the result remains uncertain because the 95% confidence interval still crosses zero (`[-3.57, +24.87]`), so the apparent edge is promising but should be interpreted with caution.

## Results at a glance

| | |
|---|---|
| **Data** | 34,059 cleaned ATP main-draw matches, 2013–2026 ([tennis-data.co.uk](http://www.tennis-data.co.uk/data.php)), plus player biographical data and city/country lookups |
| **Task** | Predict if the Bet365 favourite wins (the player with the shorter closing price) |
| **Models** | Logistic regression (Elastic Net) and XGBoost, recalculated every month and trained on the previous 2 years, with calibrated probabilities. Logistic regression is the one that holds up |
| **Betting rule** | Several were tested, and the most profitable is also one of the simplest: bet a flat 1 euro when the expected gain exceeds 0.10 euro and the predicted win probability is at least 40% |
| **Backtest** | 2016–2026, one season at a time |

## Contents

- [Data and dataset construction](#data-and-dataset-construction)
- [Models](#models)
- [Betting rule](#betting-rule)
- [Backtest](#backtest)
- [Findings](#findings)
- [Leakage check](#leakage-check)
- [Lessons learnt](#lessons-learnt)
- [Setup and usage](#setup-and-usage)
- [Repository structure](#repository-structure)
- [References](#references)

## Data and dataset construction

Historical match files (2013 through the current season) come from [tennis-data.co.uk](http://www.tennis-data.co.uk/data.php) and include tournament metadata, rankings, ranking points, and closing odds from Bet365 and Pinnacle. On the first run, `download_season_files()` fetches each missing year into `file/` and leaves any workbook already there untouched. Tennis-Data keeps the copyright on those spreadsheets and offers them for private use, so they are not part of this repository.

The player file in `file/others/` comes from Jeff Sackmann's ATP player data, via the Kaggle dataset [ATP Tour Ranking - Decade-wise and year-wise](https://www.kaggle.com/datasets/kalilurrahman/atp-tennis-player-ranking-dataset) by Kalilur Rahman. `states.xlsx` and `states_3_letters.xlsx` are lookup tables written for this project (tournament city to country, and IOC country codes).

Live upcoming matches are scraped from [Tennis Explorer](https://www.tennisexplorer.com) (`tennis_explorer_next_matches.py`).

Because the raw files describe each match from the winner's point of view (`B365W` is the winner's price and `B365L` the loser's), `build_dataset()` reorganises every row around the bookmaker's prediction instead: `fB365` is the shorter price for the favourite, while `uB365` is the longer price for the underdog.

Before any model sees the data, I correct or remove the small number of clearly invalid rows:
- Three Djokovic matches in the 2021 file have their Bet365 prices reversed, with Djokovic listed at long odds and his opponent priced below 1. Because the Pinnacle prices for those matches are plausible, I use them to replace the faulty Bet365 values.
- `clean_dataset()` removes matches with missing or impossible odds, rankings that conflict with ranking points, a gap of more than five probability points between Bet365 and Pinnacle, malformed scores, or duplicate records. In the current dataset, this process removes 155 matches and leaves 34,059. It seems like a small number, but it is enough to change the results.

Features built on top of that:

- **Elo rating** (`compute_elo_rankings`). Borrowed from chess, Elo estimates a player's strength from previous match results while accounting for the strength of each opponent, so the more unexpected a result is, the more the ratings adjust after the match. I use the difference between the two ratings to calculate the favourite's Elo-based win probability (`proba_elo_favorite`).
- **Career win percentage** (`historical_win_percentages`). The share of won matches is very volatile for new players: after the first match it can only take two extreme values, 0% or 100%, which do not represent a player's real ability. To smooth this initial behaviour, I add seven prior fake matches to every player split evenly between wins and losses (3.5 wins out of 7), producing an initial win rate of 50%. As real results accumulate, their influence gradually replaces this prior and the percentage converges to the observed career rate.
- **Season and calendar period.** I derive the astronomical season from the match date and reverse it for tournaments in the Southern Hemisphere, while `bimonthly_period` groups the calendar year into six two-month blocks. Only `bimonthly_period` is kept in the default candidate features; the astronomical season is built into the dataset but left out of the models (see [Lessons learnt](#lessons-learnt)).
- **Home advantage.** I assign +1 when only the favourite is playing in their home country, -1 when only the underdog is, and 0 when both or neither are.
- **Biographical data.** I add each player's age, height, playing hand, and country to the match data. Because the two datasets are merged on surname plus first-name initial, namesakes can occur: in those cases, only the candidate with a plausible age on the day of the match (between 16 and 45) is attached, and if none or more than one qualify, the fields are left empty for the median imputer to handle later.

For the optional exploratory plots, I use **Spearman correlation** for odds, rankings, and ranking-point classes. Most underdog prices cluster around 2, but a few reach 30 or 40; these outliers can heavily distort Pearson correlation. Spearman instead measures whether two variables move in the same ranked direction.

## Models
I test two model families, each with its own default way of selecting features:

- **Logistic regression:** I use an Elastic Net model to select the features. The `l1_ratio` sets the balance between Lasso and ridge regularisation, ranging from pure ridge at 0 to pure Lasso at 1. By default `ESTIMATE_L1_RATIO` is on, so cross-validation on the training window chooses among 0.1, 0.3, 0.5, 0.7, 0.9, and 1.0. Switching it off fixes the mix at 0.5, the baseline in [Findings](#findings). Features whose absolute coefficient reaches the mean absolute coefficient are retained, and the mean is taken over every candidate. If the penalty has set every coefficient to zero, the period is skipped and the capital is left untouched.
- **XGBoost:** I use 100 trees with a maximum depth of 3 and a learning rate of 0.05. Feature selection follows the same idea, using the model's gain-based importance rather than coefficients: features whose gain reaches the average gain are kept, the average taken over every candidate, and if every gain is zero the period is skipped.

As an alternative to these model-based selectors, I also test `SELECTION_METHOD='roi'`, a bidirectional search that adds or removes features according to their effect on flat-stake returns within the training window. The idea is appealing: select variables for the outcome that ultimately matters, but in the backtest it performs worse than the default approach. I discuss the likely reason in [Findings](#findings).

Whichever model and selector I use, the selected features then pass through the same scikit-learn pipeline: median imputation, scaling, and probability calibration with `CalibratedClassifierCV`. I calibrate with a sigmoid function over a three-fold time-series split, so the chronological order of the observations is preserved. The model's estimate is directly compared with the probability implied by the bookmaker's odds, so they must be reliable.


## Betting rule

**Whether to bet.** Expected gain per euro staked on a player:

```text
EV = p * (odds - 1) - (1 - p)
```

`p` is the model's probability for that player and `odds` is his Bet365 price. A bet is placed when the EV is above 0.10 euro and `p` is at least 0.40; if both players pass the thresholds, the favourite is taken.

**How much.** I stake a flat 1 euro in the main runs. Quarter Kelly (`USE_KELLY`) is also implemented, staking `0.25 x EV / (odds - 1)` of the bankroll, capped at 5% of the capital (but it lowers the return per euro staked).

**When to stop.** Betting pauses when the bankroll falls to 70% of its reference level (`STOP_LOSS_LIMIT = 0.7`), switching to "shadow mode" where bets are tracked virtually until they accumulate at least 0.50 euro of profit and real staking resumes. 
I also tested two variants: recomputing the floor on the capital left after each recovery (`STOP_ON_REMAINING_CAPITAL`), and forcing a few bets through after a recovery before the stop can fire again (`N_GRACE_BETS`). On the flat-stake monthly runs the bankroll never reached the 70-euro floor, so neither variant was triggered and their returns match the flat baseline, whereas the floor does come into play under quarter Kelly.

## Backtest

`rolling_window_backtest` walks through 2016–2026 one period at a time, using a calendar month as the default schedule (`refit_every = 1`, `refit_unit = 'months'`), and resets the bankroll to 100 euro at the start of each year. At each step the Elo ratings and the win percentages are rebuilt as they stood at the start of the period, the features are re-selected, and the model is refit on the previous `WINDOW_YEARS` (2) of matches.

The reported return is total profit divided by total euros staked across the whole backtest, accompanied by 95% percentile bootstrap confidence intervals (2,000 resamples), while per-match details are saved to `backtest_results/`.

## Findings

Monthly refit, 2016–2026.

| Method | Model | Bets | Win rate | Return per euro | 95% CI |
|---|---|---:|---:|---:|---|
| Bookmaker's favourite | — | 26,410 | 68.6% | -5.30% | [-6.13, -4.51] |
| Baseline (flat 1 euro) | Logistic regression | 412 | 51.2% | +6.75% | [-4.06, +16.59] |
| Baseline (flat 1 euro) | XGBoost | 1,509 | 46.1% | -3.45% | [-8.80, +1.98] |
| ROI selection | Logistic regression | 1,001 | 50.6% | -2.48% | [-8.62, +3.65] |
| ROI selection | XGBoost | 2,295 | 51.1% | -1.15% | [-4.91, +3.07] |
| Quarter Kelly | Logistic regression | 412 | 51.2% | +5.30% | [-5.29, +15.19] |
| Quarter Kelly | XGBoost | 1,076 | 46.2% | -4.29% | [-10.61, +2.00] |
| Shrink toward the bookmaker | Logistic regression | 232 | 49.1% | +3.24% | [-11.08, +16.88] |
| Shrink toward the bookmaker | XGBoost | 666 | 46.4% | +1.37% | [-7.25, +9.61] |
| Estimated `l1_ratio` | Logistic regression | 204 | 54.9% | +10.75% | [-3.57, +24.87] |
| Quarter Kelly and estimated `l1_ratio` | Logistic regression | 204 | 54.9% | +8.87% | [-5.36, +22.57] |

XGBoost does not use `l1_ratio`.

**Baseline (flat 1 euro).** Standard feature selection (`l1_ratio` fixed at 0.5 for logistic regression, gain for XGBoost), flat 1-euro stake, and no shrinkage or Kelly sizing. Logistic regression is ahead (+6.75% on 412 bets) but the interval crosses zero by a wide margin. XGBoost places nearly four times as many bets and loses 3.45%.

**ROI selection.** The selector keeps a feature when it raises profit on the training window. This selection is not profitable: a few long-priced winners can decide whether a feature enters or leaves. With the standard selector each match counts once, so a long-priced winner has no more influence than any other match, and a few lucky matches cannot decide which feature is kept.

**Quarter Kelly.** Instead of a flat 1 euro, this rule sizes each bet in proportion to the model's estimated edge and the odds, staking one-quarter of the Kelly fraction (`0.25 x EV / (odds - 1)` of the bankroll, capped at 5%). 
Compared with the baseline, it turns out less profitable per euro on the same bets: +5.30% instead of +6.75% for logistic regression (and -4.29% instead of -3.45% for XGBoost). Because the model's quoted probability on these bets is a little above the share that actually won, scaling the stake with the perceived edge allocates money less efficiently than a flat euro on each bet.

**Shrink toward the bookmaker.** The probability actually used is `(1 - λ) x model + λ x bookmaker`, with `λ` chosen to maximise profit on the last fifth of the training window. Pulling the model toward the market removes most of the disagreement the betting rule is looking for. The logistic return falls to +3.24% on 232 bets.

**Estimated `l1_ratio`.** Letting the training window choose the Lasso/ridge mix, instead of fixing it at half, is the best logistic row and the default in `config.py`: +10.75% on 204 bets, interval `[-3.57, +24.87]`. Because cross-validation often picks a heavy penalty that shrinks every coefficient to zero and skips the month, the model bets on only about 20 matches a year; the return per euro and win rate (54.9%) are the highest in the table, but the smaller sample widens the confidence interval below zero.

**Quarter Kelly and estimated `l1_ratio`.** Combining Quarter Kelly sizing with the estimated `l1_ratio` selector places the same 204 logistic bets as the flat-stake run, but lowers the return per euro from +10.75% to +8.87% (`[-5.36, +22.57]`), confirming that flat staking outperforms Kelly sizing even on the strongest feature selection.

### Year by year, estimated `l1_ratio`

Logistic regression, flat 1 euro, monthly refit. The pooled +10.75% weights each year by how much was staked.

| Year | Logistic return | Bets | XGBoost return | Bets |
|---|---:|---:|---:|---:|
| 2016 | +50.00% | 1 | -38.40% | 42 |
| 2017 | +13.75% | 16 | -2.72% | 139 |
| 2018 | +25.33% | 15 | -12.42% | 178 |
| 2019 | +12.93% | 54 | +6.41% | 166 |
| 2020 | -2.11% | 27 | -12.29% | 99 |
| 2021 | +23.67% | 15 | +9.30% | 125 |
| 2022 | -1.17% | 18 | -3.72% | 78 |
| 2023 | +4.88% | 43 | +2.53% | 240 |
| 2024 | +100.00% | 3 | -4.14% | 242 |
| 2025 | +56.00% | 3 | -5.40% | 126 |
| 2026 | -12.33% | 9 | -8.86% | 74 |

Eight of the eleven seasons are positive for logistic regression, and the two seasons with the most bets (2019 and 2023, with 54 and 43 bets) return +12.9% and +4.9%. The largest percentages (2016, 2024, and 2025) sit on 1 to 3 bets because most months in those years had all coefficients shrunk to zero and were skipped. XGBoost is negative in eight of the eleven years, and 2016 alone is -38.4% on 42 bets.

### Are the probabilities honest?

On the bets that pass the thresholds, I compare the average model probability against the share of matches actually won, alongside the bookmaker's vig-free implied probability on the same side.

| Run | Bets | Model | Bookmaker | Won |
|---|---:|---:|---:|---:|
| Estimated `l1_ratio`, logistic | 204 | 56.6% | 47.0% | 54.9% |
| Baseline (flat 1 euro), logistic | 412 | 55.0% | 45.7% | 51.2% |
| Baseline (flat 1 euro), XGBoost | 1,509 | 55.3% | 45.1% | 46.1% |
| ROI selection, logistic | 1,001 | 60.4% | 49.6% | 50.6% |

At its best setting, the logistic model quotes 56.6% and wins 54.9%: the estimate is still slightly optimistic, and because Kelly sizing trusts that perceived edge, staking larger amounts gives some of the profit back. Even so, on these matches the model is much closer to the actual outcome than the bookmaker, whose prices implied 47.0% against the 54.9% observed win rate.

### Other experiments
Several side experiments were tested and left out of the default configuration because they lowered returns:

- **Refit frequency.** Refitting every calendar month (`refit_every = 1`, `refit_unit = 'months'`) gave the best results, as shortening the step barely changes the two-year training window and mostly adds noise to the feature selection. Calendar months also align naturally with the ATP tour blocks (clay, grass, North American hard court, indoor), whereas fixed day steps often cut mid-swing: switching to a fixed **30-day** step dropped the best logistic configuration (`ESTIMATE_L1_RATIO`) from +10.75% to +7.99%, and refitting every **3, 20, or 60 days** showed the same degradation.
- **Training window.** The backtest trains on two years of history (`WINDOW_YEARS = 2`), and both shorter and longer windows reduced profitability.
- **Odds band and no underdogs.** Restricting bets to a specific range of prices or blocking underdog bets entirely both worsened returns and were removed.

## Leakage check
Because the pipeline is fairly long, I also ran a dedicated check to rule out data leakage:
1. The last 60 matches were removed from the training history and set aside, with the true winner stored separately.
2. Winner and loser (along with all player-specific columns) were swapped at random and the scores were deleted, so the file looked like a card of upcoming matches not yet played.
3. `predict_upcoming_matches()` was run on that table.

The betting decisions did not depend on which player was listed first, the held-out matches were absent from the training set, and the return on those 60 matches matched the backtest over the same dates.

## Lessons learnt
Overall, publicly available match data carries limited signal that is not already reflected in the bookmaker's closing odds, so keeping the setup simple is the best way to avoid overfitting. In this framework, the rigidity of logistic regression turns out to be an advantage because it prevents the model from following noise: as shown in [Are the probabilities honest?](#are-the-probabilities-honest), both baseline models predict an average win probability of 55% against the bookmaker's 45–46%, but while the logistic model is close to its 51% observed win rate, XGBoost wins only 46% of the time, right in line with the bookmaker.

This happens because logistic regression fits one smooth line per feature and can only learn global rules across all 5,000 training matches, like: *"On average, higher Elo than the odds imply adds +2% to win probability."* 
XGBoost, instead, splits the data through depth-3 trees that combine three conditions at once, isolating a tiny subgroup of matches like: *"Favourite odds between 1.65 and 1.75 AND Underdog is older than 29 AND Hard court."* Over a two-year window a box like that might contain only 25 matches, so if by pure luck the underdog won 16 of them, XGBoost learns a 64% win probability; when a similar match comes up the next month and Bet365 prices it at 45%, the gap triggers `EV > 0.10` and places a bet on what was really just a random streak in a small sample.
Augmenting XGBoost's penalties (`min_child_weight`, `gamma`) to avoid chasing noise improves its performance, but it still stays well behind logistic regression.

This is why I do not see the need to try more complex models, like neural networks, on this type of data. 
A similar moral comes out of the betting strategies as well: a flat 1-euro stake beats more elaborate sizing rules, and refitting the models just once a month and not more often avoids chasing noise.

Adding more candidate variables can also hurt: for example, including the astronomical season (`season_rec_*`) cut the logistic return from +10.75% to +4.58%. The selector keeps season because seasonal differences appear significant in the two-year training window, yet the structural part is already priced into the odds and the leftover gap does not persist into the next month. 
Because what held in the past is not guaranteed to repeat, the positive backtest numbers must be interpreted with caution: this is further evidence that, outside the information already captured by the odds, noisy signal remains.

## Setup and usage

Python ≥ 3.10. The support files in `file/others/` come with the repository. The season workbooks do not: the first run downloads them from tennis-data.co.uk into `file/`. Install the dependencies and run `main.py`:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib openpyxl dill requests beautifulsoup4 psutil
python main.py
```

Run modes (`backtest`, `predictions`, `refit_every`, `refit_unit`) are set at the top of `main.py`, while all model, betting, stop-loss, and parallelization switches (`ESTIMATE_L1_RATIO`, `SELECTION_METHOD`, `USE_KELLY`, `SHRINK_TO_BOOK`, `PARALLEL_N_JOBS`, etc.) are documented in `config.py`.

## Repository structure

```text
.
├── main.py                            # what to run, and the walk-forward backtest
├── config.py                          # betting and model switches, candidate features
├── modeling.py                        # functions for pipelines, feature selection, EV and Kelly
├── dataset_preparation.py             # functions to merge the years, Elo, biography, favourite/underdog
├── prediction.py                      # functions to predict the upcoming matches and store them in upcoming_predictions.xlsx
├── auxiliary.py                       # other functions (logging, parallel jobs, bootstrap interval)
├── tennis_explorer_next_matches.py    # scraper of the tennis-explorer web site. It saves the results in upcoming_matches.xlsx
├── file/
│   ├── 2013.xlsx …                  # season workbooks, downloaded on first run, not in the repository
│   └── others/
│       ├── players_data_ATP Tour Ranking - Decade-wise and year-wise.xlsx
│       ├── states.xlsx                # tournament city → country
│       └── states_3_letters.xlsx      # IOC country codes
└── readme.md
```

## References

- Lisi, F., Zanella, G. (2017). *Tennis betting: can statistics beat bookmakers?* Electronic Journal of Applied Statistical Analysis, 10(3), 790–808. [DOI](https://doi.org/10.1285/i20705948v10n3p790)