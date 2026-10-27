"""Expected Threat (xT), after Karun Singh (2019).

The pitch is split into a grid. Each zone's xT is the probability that a
possession currently in that zone ends in a goal within the next few actions:

    xT(z) = P(shoot | z) * P(goal | shot from z)
          + P(move | z)  * sum over z' of P(z -> z') * xT(z')

Solved by iterating until values settle. A pass or carry from zone a to zone b
is then worth xT(b) - xT(a).

Fitted on all 34 matches in the dataset (Leverkusen and their opponents).
"""
import numpy as np
import pandas as pd

L, W = 16, 12          # zones along length, across width
PITCH_X, PITCH_Y = 120, 80


def zone(x, y):
    """Map coordinates to (column, row) grid indices; NaN-safe via clipping."""
    cx = np.clip((np.asarray(x) / PITCH_X * L).astype(int), 0, L - 1)
    cy = np.clip((np.asarray(y) / PITCH_Y * W).astype(int), 0, W - 1)
    return cx, cy


def fit_xt(ev: pd.DataFrame, n_iter: int = 30) -> np.ndarray:
    """Return a (W, L) grid of xT values. ev needs x, y, end_x, end_y (from prepare_events)."""
    moves = ev[ev.type.isin(["Pass", "Carry"]) & ev.x.notna() & ev.end_x.notna()]
    moves = moves[moves.pass_type.isna() | moves.pass_type.isin(["Recovery", "Interception"])]  # open play only
    ok = (moves.type == "Carry") | moves.pass_outcome.isna()
    shots = ev[(ev.type == "Shot") & (ev.shot_type == "Open Play") & ev.x.notna()]

    mx, my = zone(moves.x, moves.y)
    sx, sy = zone(shots.x, shots.y)
    move_n = np.zeros((W, L))
    np.add.at(move_n, (my, mx), 1)
    shot_n = np.zeros((W, L))
    np.add.at(shot_n, (sy, sx), 1)
    goal_n = np.zeros((W, L))
    goals = (shots.shot_outcome == "Goal").to_numpy()
    np.add.at(goal_n, (sy[goals], sx[goals]), 1)

    total = move_n + shot_n
    with np.errstate(divide="ignore", invalid="ignore"):
        p_shoot = np.nan_to_num(shot_n / total)
        p_move = np.nan_to_num(move_n / total)
        p_goal = np.nan_to_num(goal_n / shot_n)

    # Transition matrix: from each zone, where do successful moves end?
    # Failed moves count in the denominator (they end the possession with nothing).
    ex, ey = zone(moves.end_x, moves.end_y)
    start_idx = my * L + mx
    end_idx = ey * L + ex
    T = np.zeros((W * L, W * L))
    np.add.at(T, (start_idx[ok.to_numpy()], end_idx[ok.to_numpy()]), 1)
    counts = np.bincount(start_idx, minlength=W * L)
    with np.errstate(divide="ignore", invalid="ignore"):
        T = np.nan_to_num(T / counts[:, None])

    xt = np.zeros(W * L)
    shoot_value = (p_shoot * p_goal).ravel()
    for _ in range(n_iter):
        xt = shoot_value + p_move.ravel() * (T @ xt)
    return xt.reshape(W, L)


def add_xt(ev: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    """Add xt_added to successful open-play passes and carries (0 for everything else)."""
    ev = ev.copy()
    ev["xt_added"] = 0.0
    m = (ev.type.isin(["Pass", "Carry"]) & ev.x.notna() & ev.end_x.notna()
         & ((ev.type == "Carry") | ev.pass_outcome.isna()))
    sx, sy = zone(ev.loc[m, "x"], ev.loc[m, "y"])
    exx, eyy = zone(ev.loc[m, "end_x"], ev.loc[m, "end_y"])
    ev.loc[m, "xt_added"] = grid[eyy, exx] - grid[sy, sx]
    return ev
