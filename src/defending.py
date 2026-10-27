"""How Leverkusen defend, and how to attack them.

Coach's question: "When we have the ball, how do they press us, and where can we hurt them?"

Coordinates: every event is from the acting team's view (attacking left to right).
An opponent event at x=110 is near Leverkusen's goal; to put it in Leverkusen's view,
use 120 - x. Opponent possessions are described from the opponent's view unless a
column says lev_.
"""
import numpy as np
import pandas as pd

from analysis import FINAL_THIRD, OWN_THIRD, TEAM, minutes_played, possessions, prepare_events
from case_study import RESTART_SECS, RESTARTS
from xt import add_xt, fit_xt

KEY = ["match_id", "possession"]
REGAIN_SECS = 5
FAST_SECS = 15
DEF_ACTIONS = ["Interception", "Foul Committed", "Duel"]


def _clock(df):
    return (df.minute * 60 + df.second).astype(float)


# ---------------------------------------------------------------- 1. Their press

def ppda(ev: pd.DataFrame) -> pd.DataFrame:
    """Passes allowed per defensive action, in the pressing team's first 60% of the pitch
    measured from the opponent's goal. Lower = more aggressive press."""
    out = {}
    for side, presser in [("lev_ppda", ev.team == TEAM), ("opp_ppda", ev.team != TEAM)]:
        passer = ~presser
        passes = ev[passer & (ev.type == "Pass") & (ev.x < 72)].groupby("match_id").size()
        acts = ev[presser & ev.type.isin(DEF_ACTIONS) & (ev.x > 48)
                  & ~((ev.type == "Duel") & (ev.duel_type == "Aerial Lost"))].groupby("match_id").size()
        out[side] = passes / acts
    return pd.DataFrame(out)


def pressers(ev: pd.DataFrame, min_pressures: int = 100) -> pd.DataFrame:
    pr = ev[(ev.team == TEAM) & (ev.type == "Pressure")]
    t = pr.groupby("player").agg(pressures=("id", "size"),
                                 high_share=("x", lambda s: (s >= FINAL_THIRD).mean()),
                                 matches=("match_id", "nunique"))
    t["minutes"] = minutes_played(ev).reindex(t.index)
    t["per90"] = 90 * t.pressures / t.minutes
    return t[(t.pressures >= min_pressures) & (t.minutes >= 900)].sort_values("per90", ascending=False)


def press_height(ev: pd.DataFrame) -> pd.Series:
    """Share of pressures in the opponent's defensive third, Leverkusen vs their opponents."""
    pr = ev[ev.type == "Pressure"]
    return pr.groupby(pr.team == TEAM).x.apply(lambda s: (s >= FINAL_THIRD).mean()) \
             .rename(index={True: "Leverkusen", False: "Opponents"})


# ---------------------------------------------------------------- 2. Counter-press

WON = ["Won", "Success In Play", "Success Out"]


def losses(ev: pd.DataFrame) -> pd.DataFrame:
    """Every open-play ball loss, from events (StatsBomb possessions don't split quick regains),
    with whether the team that lost it won it back within REGAIN_SECS.

    Loss = misplaced pass (not out of play), miscontrol, dispossessed, failed dribble.
    Regain = that team's ball recovery, won interception or won tackle within the window."""
    ev = ev.sort_values(["match_id", "index"]).assign(t=_clock(ev))
    lost = ev[((ev.type == "Pass") & (ev.pass_outcome == "Incomplete") & ev.pass_type.isna())
              | ev.type.isin(["Miscontrol", "Dispossessed"])
              | ((ev.type == "Dribble") & (ev.dribble_outcome == "Incomplete"))].copy()
    regain_ev = ev[((ev.type == "Ball Recovery") & ev.ball_recovery_recovery_failure.isna())
                   | ((ev.type == "Interception") & ev.interception_outcome.isin(WON))
                   | ((ev.type == "Duel") & (ev.duel_type == "Tackle") & ev.duel_outcome.isin(WON))]
    flags = pd.Series(False, index=lost.index)
    for (m, per, team), g in lost.groupby(["match_id", "period", "team"]):
        rt = np.sort(regain_ev[(regain_ev.match_id == m) & (regain_ev.period == per)
                               & (regain_ev.team == team)].t.to_numpy())
        i = np.searchsorted(rt, g.t.to_numpy(), side="right")
        nxt = np.where(i < len(rt), rt[np.minimum(i, len(rt) - 1)], np.inf)
        flags[g.index] = (nxt - g.t.to_numpy()) <= REGAIN_SECS
    lost["regain_5s"] = flags
    lost["side"] = np.where(lost.team == TEAM, "Leverkusen", "Opponents")
    lost["zone"] = pd.cut(lost.x, [-1, OWN_THIRD, FINAL_THIRD, 121],
                          labels=["Own third", "Middle third", "Final third"])
    return lost


def counterpress(lost: pd.DataFrame) -> pd.DataFrame:
    t = lost.groupby(["side", "zone"], observed=True).agg(losses=("id", "size"), regain_5s=("regain_5s", "mean"))
    t["losses_per_match"] = t.losses / lost.match_id.nunique()
    return t


# ---------------------------------------------------------------- 3. Where they concede

def opponent_shots(ev: pd.DataFrame, pos: pd.DataFrame) -> pd.DataFrame:
    """Opponent shots (not penalties), labelled by how the possession started."""
    ev = ev.assign(t=_clock(ev))
    t0 = ev.groupby(KEY).t.min().rename("t0")
    sh = ev[(ev.type == "Shot") & (ev.team != TEAM) & (ev.shot_type != "Penalty")].join(t0, on=KEY)
    sh = sh.join(pos.set_index(KEY)[["start_x"]], on=KEY)
    secs = sh.t - sh.t0
    restart = sh.play_pattern.isin(RESTARTS)
    sh["origin"] = np.select(
        [restart & (secs <= RESTART_SECS) & sh.play_pattern.isin(["From Corner", "From Free Kick", "From Throw In"]),
         sh.play_pattern.isin(["Regular Play", "From Counter"]) & (sh.start_x >= FINAL_THIRD),
         sh.play_pattern.isin(["Regular Play", "From Counter"]) & (sh.start_x >= OWN_THIRD),
         sh.play_pattern.isin(["Regular Play", "From Counter"]) & (secs <= FAST_SECS)],
        ["Set piece", "Won ball in Leverkusen's third", "Won ball in midfield", "Fast break from deep"],
        default="Patient build-up")
    sh["fast"] = secs <= FAST_SECS
    sh["secs"] = secs
    return sh


def wide_transitions(ev: pd.DataFrame, sh: pd.DataFrame) -> pd.DataFrame:
    """Did open-play attacks that ended in a shot go down the side of Leverkusen's
    wing-backs? Wide channel in Leverkusen's half, from the attacker's view:
    y < 18 = attacker's left = behind Frimpong (Leverkusen right);
    y > 62 = attacker's right = behind Grimaldo (Leverkusen left)."""
    op = sh[sh.origin != "Set piece"]
    keys = op.set_index(KEY).index.unique()
    acts = ev[(ev.team != TEAM) & ev.type.isin(["Pass", "Carry"]) & ev.end_x.notna()
              & ev.set_index(KEY).index.isin(keys)]
    acts = acts[(acts.type == "Carry") | acts.pass_outcome.isna()]
    behind = acts[(acts.end_x >= 60) & (acts.end_x < 102)]
    frim = behind[behind.end_y < 18].groupby(KEY).size().rename("behind_frimpong")
    grim = behind[behind.end_y > 62].groupby(KEY).size().rename("behind_grimaldo")
    poss = op.groupby(KEY).agg(xg=("shot_statsbomb_xg", "sum"), fast=("fast", "max"), origin=("origin", "first"))
    poss = poss.join(frim).join(grim).fillna({"behind_frimpong": 0, "behind_grimaldo": 0})
    f, g = poss.behind_frimpong > 0, poss.behind_grimaldo > 0
    poss["route"] = np.select([f & g, f, g], ["Both flanks", "Down Frimpong's side (their right)",
                                              "Down Grimaldo's side (their left)"], default="Through the middle")
    return poss


# ---------------------------------------------------------------- 4. Beating their press

def opponent_exits(ev_xt: pd.DataFrame, pos: pd.DataFrame) -> pd.DataFrame:
    """Opponent build-ups from their own third: how they got out, and what it was worth."""
    b = pos[(pos.team != TEAM) & (pos.start_x < OWN_THIRD)
            & ~pos.play_pattern.isin(["From Corner", "From Kick Off"])]
    keys = b.set_index(KEY).index
    acts = ev_xt[(ev_xt.team != TEAM) & ev_xt.type.isin(["Pass", "Carry"])
                 & ev_xt.set_index(KEY).index.isin(keys)]
    acts = acts[(acts.type == "Carry") | acts.pass_outcome.isna()]
    ex = acts[(acts.x < OWN_THIRD) & (acts.end_x >= OWN_THIRD)].groupby(KEY).first().reset_index()
    ex["channel"] = pd.cut(ex.end_y, [-1, 18, 30, 50, 62, 81],
                           labels=["Left wing", "Left half-space", "Centre", "Right half-space", "Right wing"])
    ex["method"] = np.select([ex.type == "Carry", ex.pass_length >= 35], ["Carry", "Long pass"], "Short pass")
    gained = ev_xt[ev_xt.team != TEAM].groupby(KEY).xt_added.sum().rename("possession_xt")
    ex = ex.join(gained, on=KEY).merge(b[KEY + ["reached_final_third", "shot", "xg"]], on=KEY)
    lev_pr = ev_xt[(ev_xt.team == TEAM) & (ev_xt.type == "Pressure") & (ev_xt.x >= FINAL_THIRD)]
    ex["pressed"] = ex.set_index(KEY).index.isin(lev_pr.set_index(KEY).index)
    return ex, b


def exit_table(ex: pd.DataFrame, by) -> pd.DataFrame:
    return ex.groupby(by, observed=True).agg(
        n=("possession", "size"), xt_per_exit=("possession_xt", "mean"),
        reached_final_third=("reached_final_third", "mean"), shot_rate=("shot", "mean"))


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    ev = prepare_events()
    pos = possessions(ev)

    pp = ppda(ev)
    print("PPDA (median):", pp.median().round(2).to_dict())
    print("Press height (share of pressures in opponent's def third):", press_height(ev).round(3).to_dict())
    print(pressers(ev).round(3).head(10).to_string())

    lo = losses(ev)
    print("\nCounter-press:")
    print(counterpress(lo).round(3).to_string())
    print(lo.groupby("side").regain_5s.mean().round(3).to_dict())

    sh = opponent_shots(ev, pos)
    print(f"\nOpponent non-penalty shots: {len(sh)}, xG {sh.shot_statsbomb_xg.sum():.2f}")
    print(sh.groupby("origin").agg(shots=("id", "size"), xg=("shot_statsbomb_xg", "sum"),
                                   xg_per_shot=("shot_statsbomb_xg", "mean"),
                                   goals=("shot_outcome", lambda s: (s == "Goal").sum())).round(3).to_string())
    wt = wide_transitions(ev, sh)
    print("\nOpen-play shot possessions by route:")
    print(wt.groupby("route").agg(possessions=("xg", "size"), xg=("xg", "sum"), xg_per=("xg", "mean"),
                                  fast_share=("fast", "mean")).round(3).to_string())
    print(wt[wt.fast].groupby("route").agg(possessions=("xg", "size"), xg=("xg", "sum")).round(3).to_string())

    ex, b = opponent_exits(add_xt(ev, fit_xt(ev)), pos)
    print(f"\nOpponent build-ups from own third: {len(b)}, exited: {len(ex)} ({len(ex) / len(b):.0%})")
    print(f"reached final third {b.reached_final_third.mean():.1%}, shot {b.shot.mean():.1%}")
    print(exit_table(ex, "channel").round(3).to_string())
    print(exit_table(ex, "method").round(3).to_string())
    print(exit_table(ex, "pressed").round(3).to_string())
