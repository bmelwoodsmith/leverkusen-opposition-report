"""Case study: why did Bayern's high press work when Frankfurt's and Bochum's didn't?

All four teams pressed Leverkusen's build-up high (top 4 for press height in the season),
but Bayern held Leverkusen to their two lowest build-up progression rates while
Frankfurt (A, 1-5) and Bochum (H, 4-0) were carved open.

Every metric is computed for every match, so the four can be compared with the
season baseline (the other 30 matches).
"""
import numpy as np
import pandas as pd

from analysis import (FINAL_THIRD, OWN_THIRD, TEAM, buildup_possessions,
                      exit_routes, possessions, prepare_events)
from data import load_frames
from xt import add_xt, fit_xt

CASES = {
    3895074: "Bayern (H) 2-2",
    3895232: "Bayern (A) 0-3",
    3895333: "Frankfurt (H) 1-5",
    3895182: "Bochum (A) 0-4",
}
# Labels are from the opponent's point of view: venue and score as the opponent saw it.


GOAL_SOURCE = {"From Throw In": "Throw-in", "From Corner": "Corner",
               "From Free Kick": "Free kick", "From Counter": "Counter-attack",
               "Regular Play": "Open play"}
RESTART_SECS = 10  # a restart only gets credit if the shot came within this many seconds
RESTARTS = ["From Throw In", "From Corner", "From Free Kick", "From Goal Kick", "From Keeper", "From Kick Off"]


def label(match_id):
    return CASES.get(match_id, "Other 30 matches")


def _clock(ev):
    return (ev.minute * 60 + ev.second).astype(float)


def ppda(ev: pd.DataFrame) -> pd.Series:
    """Passes Leverkusen made per opponent defensive action, in Leverkusen's first 60% of the pitch.
    Lower = more intense press."""
    lev_passes = ev[(ev.team == TEAM) & (ev.type == "Pass") & (ev.x < 72)].groupby("match_id").size()
    def_types = ["Interception", "Foul Committed", "Duel"]
    opp = ev[(ev.team != TEAM) & (ev.x > 48)]
    acts = opp[opp.type.isin(def_types) & ~((opp.type == "Duel") & (opp.duel_type == "Aerial Lost"))]
    return lev_passes / acts.groupby("match_id").size()


WON = ["Won", "Success In Play", "Success Out"]


def high_pressures(ev: pd.DataFrame) -> pd.DataFrame:
    """Every opponent pressure in Leverkusen's defensive third, flagged regain_5s if the
    pressing team won the ball within 5 seconds: their ball recovery, won interception or
    won tackle. Measured from events, because StatsBomb possessions don't split quick regains
    (the same definition as the counter-press in defending.py)."""
    ev = ev.assign(t=_clock(ev))
    opp = ev[ev.team != TEAM]
    pr = opp[(opp.type == "Pressure") & (opp.x >= FINAL_THIRD)]
    reg = opp[((opp.type == "Ball Recovery") & opp.ball_recovery_recovery_failure.isna())
              | ((opp.type == "Interception") & opp.interception_outcome.isin(WON))
              | ((opp.type == "Duel") & (opp.duel_type == "Tackle") & opp.duel_outcome.isin(WON))]
    # Kept = the pressing team also started a possession of its own within 5 seconds
    # (StatsBomb only opens a new possession once a team has control of the ball).
    starts = (ev.groupby(["match_id", "possession"])
              .agg(t=("t", "first"), period=("period", "first"), team=("possession_team", "first"))
              .reset_index())
    starts = starts[starts.team != TEAM]

    def within(times, grp, side):
        s = np.sort(times)
        idx = np.searchsorted(s, grp.t.to_numpy(), side=side)
        nxt = np.where(idx < len(s), s[np.minimum(idx, len(s) - 1)], np.inf)
        return pd.Series((nxt - grp.t.to_numpy()) <= 5, index=grp.index)

    won, kept = [], []
    for (m, per), grp in pr.groupby(["match_id", "period"]):
        won.append(within(reg[(reg.match_id == m) & (reg.period == per)].t.to_numpy(), grp, "left"))
        kept.append(within(starts[(starts.match_id == m) & (starts.period == per)].t.to_numpy(), grp, "right"))
    return pr.assign(regain_5s=pd.concat(won), kept_5s=pd.concat(kept))


def press_regains(ev: pd.DataFrame) -> pd.DataFrame:
    return high_pressures(ev).groupby("match_id").agg(
        high_pressures=("id", "size"), regain_5s=("regain_5s", "mean"), kept_5s=("kept_5s", "mean"))


def goal_sources(ev: pd.DataFrame, match_ids) -> pd.Series:
    """Leverkusen goals by source. StatsBomb tags a whole possession with how it started,
    so a goal 20 passes after a throw-in is counted as open play here, not as a throw-in."""
    ev = ev.assign(t=ev.minute * 60 + ev.second)
    start = ev.groupby(["match_id", "possession"]).t.min().rename("t0")
    g = ev[(ev.type == "Shot") & (ev.shot_outcome == "Goal") & (ev.team == TEAM)
           & ev.match_id.isin(match_ids)].join(start, on=["match_id", "possession"])
    restart = g.play_pattern.isin(RESTARTS)
    late = restart & ((g.t - g.t0) > RESTART_SECS)
    src = np.select([g.shot_type == "Penalty", late],
                    ["Penalty", "Open play"], default=g.play_pattern.map(GOAL_SOURCE).fillna("Other"))
    return pd.Series(src).value_counts()


def time_to_final_third(ev: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    """Once Leverkusen got out of their own third, how many seconds until they reached the final third?
    Short = opponent had no midfield/back line set behind the press."""
    keys = b.set_index(["match_id", "possession"]).index
    lev = ev[(ev.team == TEAM) & ev.set_index(["match_id", "possession"]).index.isin(keys)].copy()
    lev["t"] = _clock(lev)
    lev["far"] = lev[["x", "end_x"]].max(axis=1)
    out = lev[lev.far >= OWN_THIRD].groupby(["match_id", "possession"]).t.min()
    fin = lev[lev.far >= FINAL_THIRD].groupby(["match_id", "possession"]).t.min()
    d = (fin - out).dropna()
    return d[(d >= 0) & (d < 60)].groupby("match_id").median()


def marking_360(ev: pd.DataFrame) -> pd.DataFrame:
    """From 360 frames at Leverkusen passes in their own half:
    - opp_in_lev_half: opponents visible in Leverkusen's half (how many committed to the press)
    - ball_space: metres from the passer to the nearest opponent
    - teammate_space: average metres from each visible Leverkusen outfield teammate in their own half
      to their nearest opponent (how tightly the options were marked)
    StatsBomb pitch units are yards-ish; 120 x 80 is roughly 105 x 68 m, so distances are
    scaled to metres.
    """
    frames = load_frames()
    passes = ev[(ev.team == TEAM) & (ev.type == "Pass") & (ev.x < 60) & ev.pass_type.isna()]
    fr = frames[frames.id.isin(passes.id)].copy()
    xy = np.vstack(fr.location.to_numpy())
    fr["fx"], fr["fy"] = xy[:, 0] * 105 / 120, xy[:, 1] * 68 / 80

    rows = []
    for eid, g in fr.groupby("id", sort=False):
        opp = g[~g.teammate & ~g.keeper]
        mates = g[g.teammate & ~g.keeper]
        actor = g[g.actor]
        if opp.empty or actor.empty:
            continue
        o = opp[["fx", "fy"]].to_numpy()
        a = actor[["fx", "fy"]].to_numpy()[0]
        ball_space = np.min(np.hypot(*(o - a).T))
        m = mates[~mates.actor & (mates.fx < 52.5)][["fx", "fy"]].to_numpy()
        if len(m):
            d = np.hypot(m[:, None, 0] - o[None, :, 0], m[:, None, 1] - o[None, :, 1]).min(axis=1)
            mate_space = d.mean()
        else:
            mate_space = np.nan
        rows.append((eid, (opp.fx < 52.5).sum(), ball_space, mate_space))
    df = pd.DataFrame(rows, columns=["id", "opp_in_lev_half", "ball_space", "teammate_space"])
    df = df.merge(passes[["id", "match_id"]], on="id")
    return df.groupby("match_id").agg(
        frames=("id", "size"), opp_in_lev_half=("opp_in_lev_half", "mean"),
        ball_space=("ball_space", "median"), teammate_space=("teammate_space", "median"))


def match_table(ev: pd.DataFrame, pos: pd.DataFrame) -> pd.DataFrame:
    b = buildup_possessions(pos)
    grid = fit_xt(ev)
    ex = exit_routes(add_xt(ev, grid), b)
    shots = ev[ev.type == "Shot"]
    opp_shots = shots[shots.team != TEAM]
    lev_shots = shots[shots.team == TEAM]
    # Open play = not a penalty/direct set piece, and not within RESTART_SECS of a restart
    # (a long move that began with a throw-in counts as open play).
    t_ev = ev.assign(t=ev.minute * 60 + ev.second)
    t0 = t_ev.groupby(["match_id", "possession"]).t.min().rename("t0")
    ls = lev_shots.assign(t=lev_shots.minute * 60 + lev_shots.second).join(t0, on=["match_id", "possession"])
    quick_restart = ls.play_pattern.isin(RESTARTS) & ((ls.t - ls.t0) <= RESTART_SECS)
    lev_op = ls[(ls.shot_type == "Open Play") & ~quick_restart & (ls.play_pattern != "Other")]
    # Opponent shots from possessions that started in Leverkusen's half (pressing rewards)
    opp_pos = pos[(pos.team != TEAM) & (pos.start_x >= 60)]
    opp_high_xg = opp_pos.groupby("match_id").xg.sum()

    opp = ev[ev.team != TEAM]
    t = pd.DataFrame({
        "high_press_share": opp[(opp.type == "Pressure") & (opp.x >= FINAL_THIRD)].groupby("match_id").size()
                            / opp[opp.type == "Pressure"].groupby("match_id").size(),
        "ppda": ppda(ev),
        "buildups": b.groupby("match_id").size(),
        "reach_final_third": b.groupby("match_id").reached_final_third.mean(),
        "long_exit_share": ex.groupby("match_id").method.apply(lambda s: (s == "Long pass").mean()),
        "xt_per_buildup": ex.groupby("match_id").possession_xt.mean(),
        "secs_to_final_third": time_to_final_third(ev, b),
        "lev_open_play_xg": lev_op.groupby("match_id").shot_statsbomb_xg.sum(),
        "lev_xg": lev_shots.groupby("match_id").shot_statsbomb_xg.sum(),
        "opp_xg_from_high_regains": opp_high_xg,
        "opp_xg": opp_shots.groupby("match_id").shot_statsbomb_xg.sum(),
    })
    t = t.join(press_regains(ev)).join(marking_360(ev))
    t["group"] = [label(m) for m in t.index]
    return t.fillna({"opp_xg_from_high_regains": 0})


def summarise(t: pd.DataFrame) -> pd.DataFrame:
    """The four case matches side by side with the season baseline (median of other 30)."""
    cases = t[t.group != "Other 30 matches"].set_index("group").drop(columns=["frames"])
    base = t[t.group == "Other 30 matches"].drop(columns=["group", "frames"]).median().rename("Other 30 (median)")
    out = pd.concat([cases.T, base], axis=1)
    return out[list(CASES.values()) + ["Other 30 (median)"]]


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    ev = prepare_events()
    pos = possessions(ev)
    t = match_table(ev, pos)
    t.to_csv("../data/match_table.csv")
    print(summarise(t).round(3).to_string())
    print("\nLeverkusen goals in the four case matches:")
    print(goal_sources(ev, list(CASES)).to_string())
