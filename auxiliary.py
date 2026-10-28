"""Helpers for main.py: run log, workspace save/load, bootstrap CI on the backtest bets, parallel workers."""

import inspect
import os
import platform
import sys
import types
from datetime import datetime
from pathlib import Path

import dill
import numpy as np
import pandas as pd
import psutil
from joblib import Parallel
from sklearn.base import clone

from config import PARALLEL_BACKEND, PARALLEL_IDLE_BELOW, PARALLEL_N_JOBS

PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_project_path(path):
    """Relative paths start from the project folder, wherever the script is launched from."""
    return Path(path) if Path(path).is_absolute() else PROJECT_ROOT / path


##########################################################
# run log: every print goes to the terminal and to a text file
##########################################################

class _Tee:
    """Write the same text to several streams (terminal + log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


_log_file = None


def init_run_log(log_to_file=False, log_dir="backtest_results"):
    """Start mirroring the console to log_dir/run_log_<timestamp>.txt."""
    global _log_file
    if not log_to_file:
        return
    log_dir = resolve_project_path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    _log_file = open(log_dir / f"run_log_{datetime.now():%Y%m%d_%H%M%S}.txt", "w", encoding="utf-8", buffering=1)
    _resume_run_log_file()
    print(f"Logging to {_log_file.name}")


def _pause_run_log_file():
    """Console only. Used while the parallel workers run, so their prints don't get mixed in the file."""
    if _log_file is not None:
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__


def _resume_run_log_file():
    """Console + log file again."""
    if _log_file is not None:
        sys.stdout = _Tee(sys.__stdout__, _log_file)
        sys.stderr = _Tee(sys.__stderr__, _log_file)


def _append_job_log(text):
    """Write the buffered prints of one finished parallel job to the log file."""
    if _log_file is not None and text:
        _log_file.write(text if text.endswith("\n") else text + "\n")


def close_run_log():
    global _log_file
    if _log_file is None:
        return
    print(f"Log saved to {_log_file.name}")
    _pause_run_log_file()
    _log_file.close()
    _log_file = None


##########################################################
# save / load the workspace, to look at the results without re-running everything
##########################################################

def _picklable(obj):
    try:
        dill.dumps(obj)
        return True
    except Exception:
        return False


def save_full_environment(out_dir="backtest_results", namespace=None):
    """Dump every data variable of the caller (DataFrames, dicts, fitted models...) into a .pkl file."""
    print("Saving environment...")
    if namespace is None:
        namespace = inspect.currentframe().f_back.f_globals
    skip_names = {"In", "Out", "get_ipython", "quit", "exit", "copyright", "license", "credits"}
    skip_types = (types.ModuleType, types.FunctionType, types.BuiltinFunctionType, type)
    env = {
        name: obj for name, obj in dict(namespace).items()
        if not name.startswith("_") and name not in skip_names
        and not isinstance(obj, skip_types) and _picklable(obj)
    }
    out = resolve_project_path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"tennis_betting_environment_FULL_{datetime.now():%Y%m%d_%H%M%S}.pkl"
    with open(path, "wb") as f:
        dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)
    print(f"Environment saved to: {path} ({len(env)} variables)")
    return path


def load_full_environment(file_name=None, search_dir="backtest_results", namespace=None):
    """Load the newest saved environment (or file_name) into the caller's globals."""
    if namespace is None:
        namespace = inspect.currentframe().f_back.f_globals
    if file_name is None:
        files = sorted(resolve_project_path(search_dir).glob("tennis_betting_environment_FULL_*.pkl"))
        if not files:
            print("No saved environment pickle found")
            return False
        file_name = files[-1]
    with open(file_name, "rb") as f:
        env = dill.load(f)
    namespace.update(env)
    print(f"Environment loaded ({len(env)} variables) from {file_name}")
    return True

##########################################################
# bootstrap 95% CI on the return per euro staked
##########################################################

def concat_details_for_model(details_by_year, model_name):
    """All the yearly bet tables of one model stacked together (the CI needs every bet)."""
    frames = [d[model_name] for d in details_by_year.values() if model_name in d and not d[model_name].empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _real_bets(details):
    """Only the bets actually placed: Bet == YES, real mode (not shadow), stake > 0."""
    if details.empty:
        return pd.DataFrame(columns=["Profit", "Stake", "Outcome"])
    stake = pd.to_numeric(details["Stake"], errors="coerce")
    placed = details["Bet"].astype(str).eq("YES") & details["Mode"].astype(str).eq("real") & (stake > 0)
    bets = details.loc[placed, ["Profit", "Stake", "Outcome"]].copy()
    bets[["Profit", "Stake"]] = bets[["Profit", "Stake"]].apply(pd.to_numeric, errors="coerce")
    return bets.dropna()


def bootstrap_roi(profits, stakes, n_boot, rng):
    """Point estimate and 95% percentile CI of sum(profit) / sum(stake), resampling the bets with replacement."""
    n = len(profits)
    if n == 0:
        return np.nan, np.nan, np.nan
    point = profits.sum() / stakes.sum()
    if n < 5:
        return point, np.nan, np.nan
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = profits[idx].sum(axis=1) / stakes[idx].sum(axis=1)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return point, lo, hi


def _ci_row(name, bets, n_boot, rng):
    """It creates one row of the CI table. print_bootstrap_summary calls it once for each model."""
    roi, lo, hi = bootstrap_roi(bets["Profit"].to_numpy(float), bets["Stake"].to_numpy(float), n_boot, rng)
    win = bets["Outcome"].astype(str).eq("WON").mean() if len(bets) else np.nan
    return {"Model": name, "Bets": len(bets), "Win rate": win, "Return per euro": roi, "CI lo": lo, "CI hi": hi}


def print_bootstrap_summary(details_by_year, sheets=None, book_profits=None, n_boot=2000, seed=42):
    """
    Print the bootstrap 95% CI table at the end of the backtest, one row per model (it ends up
    in the run log with the rest). book_profits adds the bookmaker's favourite as benchmark.
    """
    if sheets is None:
        sheets = list(dict.fromkeys(name for d in details_by_year.values() for name in d))
    rng = np.random.default_rng(seed)
    rows = [_ci_row(name, _real_bets(concat_details_for_model(details_by_year, name)), n_boot, rng) for name in sheets]
    if book_profits is not None:
        p = np.asarray(book_profits, dtype=float)
        book = pd.DataFrame({"Profit": p, "Stake": 1.0, "Outcome": np.where(p > 0, "WON", "LOST")})
        rows.insert(0, _ci_row("Bookmaker_favourite", book, n_boot, rng))
    df = pd.DataFrame(rows)

    show = df[["Model", "Bets"]].copy()
    show["Win rate"] = df["Win rate"].map(lambda x: f"{100 * x:.1f}%" if pd.notna(x) else "")
    show["Return per euro"] = df["Return per euro"].map(lambda x: f"{100 * x:+.2f}%" if pd.notna(x) else "")
    show["95% CI"] = [f"[{100 * lo:+.2f}, {100 * hi:+.2f}]" if pd.notna(hi) else ""
                      for lo, hi in zip(df["CI lo"], df["CI hi"])]
    print(f"\n{'=' * 80}\n BOOTSTRAP 95% CI - return on money staked\n{'=' * 80}")
    print(f"Resamples: {n_boot}  seed={seed}  percentile CI\n")
    print(show.to_string(index=False))
    return df

##########################################################
# parallel workers
##########################################################

def parallel_workers(requested=None):
    """
    Cores to use: PARALLEL_N_JOBS from config or, on the first call, the idle cores minus one so the computer stays usable.
    `requested` lowers it further for a single task.
    """
    global PARALLEL_N_JOBS
    cap = max(1, (os.cpu_count() or 1) - 1)
    if PARALLEL_N_JOBS is None:
        usage = psutil.cpu_percent(interval=0.5, percpu=True)  
        #in an interval of 0.5 seconds the cores load is measured
        idle = sum(u < PARALLEL_IDLE_BELOW for u in usage)
        PARALLEL_N_JOBS = max(1, min(idle - 1, cap))
        print(f"PARALLEL_N_JOBS: {PARALLEL_N_JOBS} (auto: {idle} free cores at start, one left free; {len(usage)} CPUs)")
    budget = min(int(PARALLEL_N_JOBS), cap)
    return min(int(requested), budget) if requested else budget


def drop_old_workers():
    """The year×model workers could stay alive between runs, so I shut them down."""
    from joblib.externals.loky.reusable_executor import _executor
    if _executor is not None:
        _executor.shutdown(wait=True)


def run_parallel_batch(n_jobs, tasks):
    """Run joblib tasks on n_jobs processes; if the group of workers can't start, fall back to threads. """
    global PARALLEL_BACKEND
    try:
        return Parallel(n_jobs=n_jobs, backend=PARALLEL_BACKEND)(tasks)
    except Exception as e:
        if PARALLEL_BACKEND == "threading":
            raise
        print(f"Process pool failed ({type(e).__name__}: {e}). Continuing on threads.")
        PARALLEL_BACKEND = "threading"
        return Parallel(n_jobs=n_jobs, backend="threading")(tasks)


def one_core_estimator(estimator):
    """XGBoost runs on several cores by default. This function forces it onto one core, because the parallelization is already outside: one core for each year × model."""
    model = clone(estimator)
    if "n_jobs" in model.get_params():
        model.set_params(n_jobs=1)
    return model

##########################################################
# end of run
##########################################################

def play_beep():
    if platform.system() == "Darwin":
        os.system("afplay /System/Library/Sounds/Glass.aiff")
    elif platform.system() == "Windows":
        import winsound
        winsound.Beep(1000, 500)
    else:
        print("\a")


def schedule_shutdown(minutes=1):
    """Turn the computer off a few minutes after the run (on macOS/Linux run `sudo -v` beforehand)."""
    minutes = max(1, int(minutes))
    print(f"\nScheduling shutdown in {minutes} minutes.")
    if platform.system() == "Windows":
        os.system(f"shutdown /s /t {minutes * 60}")
        print("Cancel with: shutdown /a")
    elif os.system(f"sudo -n shutdown -h +{minutes}") == 0:
        print("Cancel with: sudo shutdown -c")
    else:
        print("Shutdown not scheduled (sudo needs a password). Run `sudo -v` before the backtest, or shut down manually.")
