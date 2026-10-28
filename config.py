"""Shared betting / model switches used by main.py and prediction.py."""

# EV is the average gain per euro staked.
# It is the average of the gains in the two possible scenarios: win (model_probability * (odds - 1))
# and loss (-prob_loss * 1). For example EV_THRESHOLD=0.10 means betting only on matches with an
# expected gain of, at least, 0.10 euro, i.e. a return of 1.10 euro.
# Raising or lowering the EV threshold does not improve the ROI.
EV_THRESHOLD = 0.1

PROB_THRESHOLD = 0.40  # probability of the outcome bet on (to avoid matches that are too uncertain)

####### kelly criterion (USE_KELLY=True)
USE_KELLY = False
KELLY_FRACTION = 0.25
# for prudence, stake a fraction of what the Kelly formula would suggest
MAX_STAKE_FRACTION = 0.05
# cap on the stake as a share of the capital, prudential
KELLY_REFERENCE_CAPITAL = 100.0

###### shrinkage procedure (SHRINK_TO_BOOK=True)
SHRINK_TO_BOOK = False
# split the training set in two parts: one to fit the model and the rest (SHRINK_VAL_FRACTION)
# to find the best lambda, then refit on the whole training set and blend.
SHRINK_VAL_FRACTION = 0.20
SHRINK_MIN_VAL_ROWS = 50
BOOK_PROB_COL = 'fprob_B365_norm'
# the ROI gets worse with shrinkage. Procedure left disabled.

###### l1_ratio estimation (ESTIMATE_L1_RATIO=True)
ESTIMATE_L1_RATIO = True
# False: Elastic Net uses a fixed l1_ratio of 0.5.
# True: l1_ratio is chosen by cross-validation on the training window. This is the default because it gives the best ROI.

###### feature selection
SELECTION_METHOD = 'standard'
# 'standard': Elastic Net for logistic, gain above mean for XGBoost.
# 'roi': bidirectional stepwise search that adds/removes a feature when training-window
#        flat-stake profit improves.

###### training window
WINDOW_YEARS = 2
# years of past matches used to train and select features before each refit / live predict.

###### stop-loss / shadow mode (backtest capital path)
STOP_LOSS_LIMIT = 0.7
# e.g. 0.7 means the maximum acceptable loss is 30% of the capital.
# When the capital falls below 70% of the starting amount, betting is suspended.
STOP_ON_REMAINING_CAPITAL = False
# False: the floor is calculated on the initial capital (e.g. 70% of the starting amount).
# True: the floor is recalculated after each recovery on the capital left; successive
# drawdowns could drive the final balance to zero. Worse performance.
N_GRACE_BETS = 0
# After leaving shadow mode, place at least this many bets before stop-loss can trigger
# again. 0 = grace off. Worse performance when > 0.

###### parallelization
# None > on first use, count idle cores (activity < PARALLEL_IDLE_BELOW %), use idle−1.
# int > fixed number of cores, capped at cpu_count−1.
PARALLEL_N_JOBS = None
PARALLEL_IDLE_BELOW = 25.0
PARALLEL_BACKEND = "loky"

CANDIDATE_FEATURES = [
    'fB365', 'uB365', 'fhand_rec', 'uhand_rec', 'fage', 'uage', 'fheight', 'uheight',
    'felo', 'uelo', 'proba_elo_favorite', 'fRank', 'uRank', 'fPts_class', 'uPts_class',
    'Hf', 'Uf', 'home', 'different_hand', 'f_taller', 'bimonthly_period',
    'favorite_win_percentage', 'underdog_loss_percentage',
    'surface_rec_Grass', 'surface_rec_Hard', 'court_rec_Outdoor'#,
    #'season_rec_Spring', 'season_rec_Summer', 'season_rec_Winter'  # autumn is the reference level
    # surface: one of the info sheets also lists carpet, but it does not appear in the data analysed.
]
