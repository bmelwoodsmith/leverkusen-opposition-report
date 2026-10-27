"""Build-up and pressing analysis for Leverkusen 2023/24.

Coach's question: "Should we press Leverkusen high, and if so, how?"

StatsBomb coordinates: pitch is 120 x 80, and every event is recorded from the
acting team's point of view (attacking left to right, x=0 own goal line,
y=0 left touchline). So a Leverkusen event at x=10 is near their own goal,
and an opponent event at x=110 is also near Leverkusen's goal.
"""
import numpy as np
import pandas as pd

from data import TEAM, load_events, load_frames, load_matches

OWN_THIRD = 40      # x < 40 = team's defensive third
FINAL_THIRD = 80    # x >= 80 = team's attacking third


def _xy(loc):
    if isinstance(loc, (list, np.ndarray)) and len(loc) >= 2:
        return float(loc[0]), float(loc[1])
    return np.nan, np.nan


def prepare_events() -> pd.DataFrame:
    ev = load_events().sort_values(["match_id", "index"]).reset_index(drop=True)
    ev[["x", "y"]] = pd.DataFrame(ev.location.map(_xy).tolist(), index=ev.index)
    end = ev.pass_end_location.where(ev.type == "Pass", ev.carry_end_location)
    ev[["end_x", "end_y"]] = pd.DataFrame(end.map(_xy).tolist(), index=ev.index)
    ev["under_pressure"] = ev.under_pressure.fillna(False).astype(bool)
    return ev


def possessions(ev: pd.DataFrame) -> pd.DataFrame:
    """One row per possession, with where it started and how it ended."""
    on_ball = ev[ev.type.isin(["Pass", "Carry", "Shot", "Dribble", "Ball Receipt*"])
                 & (ev.team == ev.possession_team)]
    g = ev.groupby(["match_id", "possession"])
    first = g.first()
    on_ball_g = on_ball.groupby(["match_id", "possession"])
    pos = pd.DataFrame({
        "team": first.possession_team,
        "play_pattern": first.play_pattern,
        "start_x": on_ball_g.x.first(),
        "start_y": on_ball_g.y.first(),
        "max_x": on_ball_g[["x", "end_x"]].max().max(axis=1),
        "n_passes": on_ball[on_ball.type == "Pass"].groupby(["match_id", "possession"]).size(),
        "xg": ev[ev.type == "Shot"].groupby(["match_id", "possession"]).shot_statsbomb_xg.sum(),
        "shot": ev[ev.type == "Shot"].groupby(["match_id", "possession"]).size() > 0,
        "start_minute": first.minute,
    }).reset_index()
    pos["n_passes"] = pos.n_passes.fillna(0).astype(int)
    pos["xg"] = pos.xg.fillna(0.0)
    pos["shot"] = pos.shot.fillna(False).astype(bool)

    # How did the possession end? Look at the next possession's start.
    pos = pos.sort_values(["match_id", "possession"]).reset_index(drop=True)
    nxt = pos.groupby("match_id").shift(-1)
    pos["next_team"] = nxt.team
    pos["next_start_x"] = nxt.start_x
    pos["next_xg"] = nxt.xg
    # Lost the ball in own third = opponent starts their possession in their final third
    # without a set piece restart.
    open_play = nxt.play_pattern.isin(["Regular Play", "From Counter"])
    pos["lost_own_third"] = (pos.next_team != pos.team) & open_play & (pos.next_start_x >= FINAL_THIRD)
    pos["reached_final_third"] = pos.max_x >= FINAL_THIRD
    return pos


def buildup_possessions(pos: pd.DataFrame, team: str = TEAM) -> pd.DataFrame:
    """Possessions a team started in its own defensive third (goal kicks, keeper, open play)."""
    b = pos[(pos.team == team) & (pos.start_x < OWN_THIRD)
            & ~pos.play_pattern.isin(["From Corner", "From Kick Off"])].copy()
    return b


def exit_channel(ev: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """For each build-up possession, the first pass/carry that leaves the own third."""
    keys = b.set_index(["match_id", "possession"]).index
    acts = ev[ev.type.isin(["Pass", "Carry"]) & (ev.team == TEAM)
              & ev.set_index(["match_id", "possession"]).index.isin(keys)]
    # successful passes only (pass_outcome NaN = complete)
    acts = acts[(acts.type == "Carry") | acts.pass_outcome.isna()]
    crossing = acts[(acts.x < OWN_THIRD) & (acts.end_x >= OWN_THIRD)]
    first = crossing.groupby(["match_id", "possession"]).first().reset_index()
    first["channel"] = pd.cut(first.end_y, [-1, 18, 30, 50, 62, 81],
                              labels=["Left wing", "Left half-space", "Centre",
                                      "Right half-space", "Right wing"])
    first["receiver"] = first.pass_recipient.fillna(first.player)
    return first


def goal_kick_press(ev: pd.DataFrame, pos: pd.DataFrame) -> pd.DataFrame:
    """Use 360 freeze frames at Leverkusen goal kicks to count how many opponents pressed high.

    Frame locations share the acting team's coordinates, so opponents with x < 40
    are inside Leverkusen's defensive third when the kick is taken.
    """
    frames = load_frames()
    gk = ev[(ev.team == TEAM) & (ev.type == "Pass") & (ev.pass_type == "Goal Kick")]
    fr = frames[frames.id.isin(gk.id) & ~frames.teammate]
    fr = fr.assign(x=fr.location.map(lambda l: l[0]))
    high = fr[fr.x < OWN_THIRD].groupby("id").size()
    out = gk[["id", "match_id", "possession", "pass_length", "end_x", "end_y", "pass_outcome"]].copy()
    out["has_frame"] = out.id.isin(frames.id)
    out["opp_in_def_third"] = out.id.map(high).fillna(0).astype(int)
    out = out[out.has_frame]
    out["short"] = out.pass_length < 32
    out = out.merge(pos[["match_id", "possession", "reached_final_third", "lost_own_third",
                         "xg", "next_xg", "next_team"]],
                    on=["match_id", "possession"], how="left")
    out["press"] = pd.cut(out.opp_in_def_third, [-1, 2, 4, 11],
                          labels=["0-2 pressing", "3-4 pressing", "5+ pressing"])
    return out


def opponent_press_profile(ev: pd.DataFrame, pos: pd.DataFrame) -> pd.DataFrame:
    """Per match: how high did the opponent press, and what did it cost/earn them?"""
    matches = load_matches().set_index("match_id")
    opp = ev[ev.team != TEAM]
    high_press = opp[(opp.type == "Pressure") & (opp.x >= FINAL_THIRD)].groupby("match_id").size()
    all_press = opp[opp.type == "Pressure"].groupby("match_id").size()
    b = buildup_possessions(pos)
    shots = ev[ev.type == "Shot"]
    df = pd.DataFrame({
        "opponent": matches.opponent,
        "venue": matches.venue,
        "score": matches.apply(lambda r: f"{r.home_score}-{r.away_score}", axis=1),
        "high_press_share": (high_press / all_press),
        "high_pressures": high_press,
        "buildups": b.groupby("match_id").size(),
        "buildup_lost_own_third": b.groupby("match_id").lost_own_third.mean(),
        "buildup_reach_final_third": b.groupby("match_id").reached_final_third.mean(),
        "lev_xg": shots[shots.team == TEAM].groupby("match_id").shot_statsbomb_xg.sum(),
        "opp_xg": shots[shots.team != TEAM].groupby("match_id").shot_statsbomb_xg.sum(),
    })
    return df.fillna(0).sort_values("high_press_share", ascending=False)


def player_press_resistance(ev: pd.DataFrame, min_actions: int = 60) -> pd.DataFrame:
    """Leverkusen players on the ball under pressure in their own half: who keeps it, who loses it?"""
    lev = ev[(ev.team == TEAM) & ev.under_pressure & (ev.x < 60)]
    on_ball = lev[lev.type.isin(["Pass", "Carry", "Dribble", "Miscontrol", "Dispossessed"])]
    lost = (
        ((on_ball.type == "Pass") & on_ball.pass_outcome.isin(["Incomplete", "Out", "Pass Offside"]))
        | ((on_ball.type == "Dribble") & (on_ball.dribble_outcome == "Incomplete"))
        | on_ball.type.isin(["Miscontrol", "Dispossessed"])
    )
    on_ball = on_ball.assign(lost=lost, progressive=(on_ball.end_x - on_ball.x) >= 10)
    res = on_ball.groupby("player").agg(
        actions=("lost", "size"), loss_rate=("lost", "mean"),
        progressive_rate=("progressive", "mean"),
    )
    pos_mode = ev[ev.team == TEAM].groupby("player").position.agg(lambda s: s.mode().iat[0])
    res["position"] = pos_mode
    return res[res.actions >= min_actions].sort_values("loss_rate", ascending=False)


if __name__ == "__main__":
    ev = prepare_events()
    pos = possessions(ev)
    b = buildup_possessions(pos)
    print(f"Leverkusen build-up possessions: {len(b)}")
    print(f"  reached final third: {b.reached_final_third.mean():.1%}")
    print(f"  ended with shot:     {b.shot.mean():.1%}")
    print(f"  lost in own third:   {b.lost_own_third.mean():.1%}")
    print(f"  opp xG after losing it in own third: {b[b.lost_own_third].next_xg.sum():.2f}")

    ex = exit_channel(ev, b)
    print("\nWhere they exit the defensive third:")
    print(ex.channel.value_counts(normalize=True).round(3).to_string())
    print("\nMost common receivers on exit:")
    print(ex.receiver.value_counts().head(8).to_string())

    gk = goal_kick_press(ev, pos)
    print(f"\nGoal kicks with 360 frames: {len(gk)}")
    print(gk.groupby("press", observed=True).agg(
        n=("id", "size"), short=("short", "mean"),
        reached_final_third=("reached_final_third", "mean"),
        lost_own_third=("lost_own_third", "mean"), xg_for=("xg", "mean"),
    ).round(3).to_string())

    print("\nOpponent pressing profile:")
    print(opponent_press_profile(ev, pos).round(3).to_string())

    print("\nPlayer press resistance (own half, under pressure):")
    print(player_press_resistance(ev).round(3).to_string())


# ---------------------------------------------------------------- Phase 4

def exit_routes(ev_xt: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Each build-up's exit from the defensive third, labelled by channel and method,
    with the xT Leverkusen added over the rest of that possession."""
    ex = exit_channel(ev_xt, b)
    ex["method"] = np.select(
        [ex.type == "Carry", ex.pass_length >= 35],
        ["Carry", "Long pass"], default="Short pass")
    gained = (ev_xt[ev_xt.team == TEAM].groupby(["match_id", "possession"]).xt_added.sum()
              .rename("possession_xt"))
    ex = ex.merge(gained, on=["match_id", "possession"], how="left")
    ex = ex.merge(b[["match_id", "possession", "reached_final_third", "shot", "xg"]],
                  on=["match_id", "possession"], how="left")
    return ex


def route_table(ex: pd.DataFrame, by, min_n: int = 20) -> pd.DataFrame:
    t = ex.groupby(by, observed=True).agg(
        n=("possession", "size"), xt_per_exit=("possession_xt", "mean"),
        reached_final_third=("reached_final_third", "mean"),
        shot_rate=("shot", "mean"), xg_per_exit=("xg", "mean"))
    return t[t.n >= min_n].sort_values("xt_per_exit", ascending=False)


def buildup_turnovers(ev: pd.DataFrame, pos: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Build-ups where the opponent won the ball back in open play inside Leverkusen's half.

    Location is where the opponent's possession started, flipped into Leverkusen's
    coordinates (attacking left to right), plus who lost it and what it cost.
    """
    nxt = pos.groupby("match_id")[["play_pattern", "start_y"]].shift(-1)  # b keeps pos's index
    lost = b[(b.next_team != TEAM) & (b.next_start_x >= 60)].copy()
    lost["next_pattern"] = nxt.play_pattern
    lost = lost[lost.next_pattern.isin(["Regular Play", "From Counter"])]
    lost["x"] = 120 - lost.next_start_x
    lost["y"] = 80 - nxt.start_y

    # Ball receipts are left out: on a misplaced pass StatsBomb logs a receipt for the
    # intended receiver, and the loss belongs to the passer.
    on_ball = ev[(ev.team == TEAM) & ev.type.isin(
        ["Pass", "Carry", "Dribble", "Miscontrol", "Dispossessed"])]
    last = on_ball.groupby(["match_id", "possession"]).last()
    lost = lost.join(last[["player", "type", "pass_outcome"]].rename(
        columns={"type": "last_action"}), on=["match_id", "possession"])
    lost["opp_shot"] = lost.next_xg > 0
    return lost


def minutes_played(ev: pd.DataFrame, team: str = TEAM) -> pd.Series:
    """Minutes per player for a team, from starting line-ups and substitutions."""
    end = ev[ev.period <= 2].groupby("match_id").minute.max().clip(lower=90)
    mins = {}
    for _, r in ev[(ev.type == "Starting XI") & (ev.team == team)].iterrows():
        for pl in r.tactics["lineup"]:
            mins[(r.match_id, pl["player"]["name"])] = [0, end[r.match_id]]
    subs = ev[(ev.type == "Substitution") & (ev.team == team)]
    for _, r in subs.iterrows():
        if (r.match_id, r.player) in mins:
            mins[(r.match_id, r.player)][1] = r.minute
        mins[(r.match_id, r.substitution_replacement)] = [r.minute, end[r.match_id]]
    df = pd.DataFrame([(m, p, b - a) for (m, p), (a, b) in mins.items()], columns=["match_id", "player", "minutes"])
    return df.groupby("player").minutes.sum()

