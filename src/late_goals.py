"""Why did Leverkusen score so many late goals?

Compares the last part of matches (80th minute onwards, stoppage time included) with
minutes 1-79, for both teams. Tests five explanations:
  1. Opponents dropped deeper (where they defended, how high they pressed)
  2. Opponents tired (fewer pressures and defensive actions per minute)
  3. Leverkusen pushed harder (territory, box entries, shots, crosses)
  4. Substitutes made the difference
  5. Finishing luck (goals vs xG)
"""
import numpy as np
import pandas as pd

from analysis import FINAL_THIRD, TEAM, prepare_events
from game_state import add_game_state

LATE = 80
DEF_ACTIONS = ["Interception", "Duel", "Foul Committed", "Clearance", "Block", "Ball Recovery"]


def phase(ev: pd.DataFrame) -> pd.Series:
    return np.where((ev.period == 2) & (ev.minute >= LATE), "80+", "1-79")


def minutes_in_phase(ev: pd.DataFrame) -> pd.Series:
    """Playing minutes per phase, summed over matches (stoppage time included)."""
    p2_end = ev[ev.period == 2].groupby("match_id").minute.max() + 1
    p1_end = ev[ev.period == 1].groupby("match_id").minute.max() + 1
    late = (p2_end - LATE).clip(lower=0).sum()
    early = (p1_end.sum()) + (LATE - 45) * len(p2_end)
    return pd.Series({"1-79": early, "80+": late})


def phase_profile(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev.assign(phase=phase(ev))
    mins = minutes_in_phase(ev)
    lev, opp = ev.team == TEAM, ev.team != TEAM
    shots = ev[ev.type == "Shot"]
    passes = ev[(ev.type == "Pass") & ev.pass_outcome.isna()]
    rows = {}
    for ph in ["1-79", "80+"]:
        e = ev[ev.phase == ph]
        m = mins[ph]
        o_def = e[opp.loc[e.index] & e.type.isin(DEF_ACTIONS)]
        o_pr = e[opp.loc[e.index] & (e.type == "Pressure")]
        lp = passes[(passes.phase == ph) if "phase" in passes else (phase(passes) == ph)]
        f3_lev = lp[(lp.team == TEAM) & (lp.end_x >= FINAL_THIRD)]
        f3_opp = lp[(lp.team != TEAM) & (lp.end_x >= FINAL_THIRD)]
        box = lp[(lp.team == TEAM) & (lp.end_x >= 102) & lp.end_y.between(18, 62)]
        crosses = e[lev.loc[e.index] & (e.type == "Pass") & (e.pass_cross == True)]
        sh = shots[phase(shots) == ph]
        ls, os_ = sh[sh.team == TEAM], sh[sh.team != TEAM]
        rows[ph] = {
            "minutes": m,
            # 1. How deep did opponents defend? Their defensive actions' distance from their own goal (yards)
            "opp_def_action_height": o_def.x.median(),
            "opp_press_share_high": (o_pr.x >= FINAL_THIRD).mean(),
            # 2. Did they tire?
            "opp_pressures_per90": 90 * len(o_pr) / m,
            "opp_def_actions_per90": 90 * len(o_def) / m,
            # 3. Did Leverkusen push harder?
            "lev_field_tilt": len(f3_lev) / max(len(f3_lev) + len(f3_opp), 1),
            "lev_box_passes_per90": 90 * len(box) / m,
            "lev_crosses_per90": 90 * len(crosses) / m,
            "lev_shots_per90": 90 * len(ls) / m,
            "lev_xg_per90": 90 * ls.shot_statsbomb_xg.sum() / m,
            "opp_xg_per90": 90 * os_.shot_statsbomb_xg.sum() / m,
            # 5. Finishing
            "lev_goals": (ls.shot_outcome == "Goal").sum(),
            "lev_xg": ls.shot_statsbomb_xg.sum(),
        }
    t = pd.DataFrame(rows)
    t["change"] = t["80+"] / t["1-79"] - 1
    return t


def late_goal_detail(ev: pd.DataFrame) -> pd.DataFrame:
    """Every Leverkusen goal from the 80th minute: scorer, sub or starter, how it started, score before."""
    ev = add_game_state(ev)
    subs_on = ev[(ev.type == "Substitution") & (ev.team == TEAM)][["match_id", "substitution_replacement"]]
    on = set(zip(subs_on.match_id, subs_on.substitution_replacement))
    g = ev[(ev.team == TEAM) & (ev.type == "Shot") & (ev.shot_outcome == "Goal")
           & (ev.period == 2) & (ev.minute >= LATE)].copy()
    # Assister = the pass flagged as goal assist in the same possession
    ast = ev[(ev.type == "Pass") & (ev.pass_goal_assist == True)][["match_id", "possession", "player"]]
    ast = ast.rename(columns={"player": "assist"})
    g = g.merge(ast, on=["match_id", "possession"], how="left").set_index(g.index)
    g["scorer_sub"] = [(m, p) in on for m, p in zip(g.match_id, g.player)]
    g["assist_sub"] = [(m, p) in on for m, p in zip(g.match_id, g.assist)]
    g["score_before"] = g.lev_score.astype(str) + "-" + g.opp_score.astype(str)
    return g[["match_id", "minute", "player", "scorer_sub", "assist", "assist_sub", "score_before",
              "state", "play_pattern", "shot_type", "shot_statsbomb_xg"]]


def shot_cover(ev: pd.DataFrame) -> pd.DataFrame:
    """From each Leverkusen open-play shot's freeze frame: opponents (not the keeper) between the
    ball and the goal, and the nearest opponent's distance. Fewer bodies = less cover."""
    sh = ev[(ev.team == TEAM) & (ev.type == "Shot") & (ev.shot_type == "Open Play")
            & ev.shot_freeze_frame.notna()].copy()
    rows = []
    for i, r in sh.iterrows():
        x0, y0 = r.x, r.y
        opp = [p for p in r.shot_freeze_frame if not p["teammate"] and p["position"]["name"] != "Goalkeeper"]
        between = 0
        for p in opp:
            px, py = p["location"]
            # inside the triangle from the ball to the two posts (120, 36) and (120, 44)
            if px > x0:
                t = (px - x0) / (120 - x0) if 120 > x0 else 1
                lo, hi = y0 + t * (36 - y0), y0 + t * (44 - y0)
                if min(lo, hi) <= py <= max(lo, hi):
                    between += 1
        near = min((np.hypot(p["location"][0] - x0, p["location"][1] - y0) for p in opp), default=np.nan)
        rows.append((i, between, near))
    c = pd.DataFrame(rows, columns=["idx", "defenders_in_way", "nearest_yards"]).set_index("idx")
    sh = sh.join(c)
    sh["phase"] = phase(sh)
    return sh.groupby("phase").agg(shots=("id", "size"), defenders_in_way=("defenders_in_way", "mean"),
                                   no_one_in_way=("defenders_in_way", lambda s: (s == 0).mean()),
                                   nearest_yards=("nearest_yards", "median"),
                                   xg_per_shot=("shot_statsbomb_xg", "mean"))


def late_state_minutes(ev: pd.DataFrame) -> pd.Series:
    """Share of events from minute 80 onwards played at each score, from Leverkusen's side."""
    ev = add_game_state(ev)
    late = ev[(ev.period == 2) & (ev.minute >= LATE)]
    return late.state.value_counts(normalize=True)


def set_piece_rate(ev: pd.DataFrame) -> pd.DataFrame:
    """Leverkusen corners and attacking-half free kicks per 90, and goals from them, by phase."""
    ev = ev.assign(phase=phase(ev))
    mins = minutes_in_phase(ev)
    sp = ev[(ev.team == TEAM) & (ev.type == "Pass")
            & ((ev.pass_type == "Corner") | ((ev.pass_type == "Free Kick") & (ev.x >= 60)))]
    return pd.DataFrame({"set_pieces_per90": 90 * sp.groupby("phase").size() / mins})


def sub_share_of_late_minutes(ev: pd.DataFrame) -> float:
    """Share of Leverkusen outfield player-minutes from the 80th minute played by substitutes."""
    end = ev[ev.period == 2].groupby("match_id").minute.max() + 1
    subs = ev[(ev.type == "Substitution") & (ev.team == TEAM)]
    sub_min, total = 0.0, 0.0
    for m, e in end.items():
        late_len = max(e - LATE, 0)
        total += 10 * late_len
        for on in subs[subs.match_id == m].minute:
            sub_min += max(e - max(on, LATE), 0)
    return sub_min / total


def open_play_rates(ev: pd.DataFrame) -> pd.DataFrame:
    """Leverkusen non-penalty, non-set-piece shots: xG per 90 and per shot, by phase."""
    ev = ev.assign(t=ev.minute * 60 + ev.second)
    t0 = ev.groupby(["match_id", "possession"]).t.min().rename("t0")
    sh = ev[(ev.team == TEAM) & (ev.type == "Shot") & (ev.shot_type == "Open Play")].join(
        t0, on=["match_id", "possession"])
    quick_sp = sh.play_pattern.isin(["From Corner", "From Free Kick", "From Throw In"]) & ((sh.t - sh.t0) <= 10)
    sh = sh[~quick_sp].assign(phase=lambda d: phase(d))
    mins = minutes_in_phase(ev)
    g = sh.groupby("phase").agg(shots=("id", "size"), xg=("shot_statsbomb_xg", "sum"))
    g["xg_per90"] = 90 * g.xg / mins
    g["xg_per_shot"] = g.xg / g.shots
    return g


def goal_types(d: pd.DataFrame) -> pd.DataFrame:
    """Late goals by score before the goal and how the move started."""
    src = np.select([d.play_pattern == "From Counter",
                     d.play_pattern.isin(["From Corner", "From Free Kick"]) | (d.shot_type == "Penalty")],
                    ["Counter-attack", "Set piece or penalty"], default="Other open play")
    when = np.where(d.state == "Leverkusen leading", "Already leading", "Level or behind")
    return pd.crosstab(when, src)


def opp_subs_late(ev: pd.DataFrame) -> pd.Series:
    """Positions opponents brought on in the second half: more defenders = sitting deeper."""
    s = ev[(ev.type == "Substitution") & (ev.team != TEAM) & (ev.period == 2)]
    return s.groupby(s.minute >= 70).size()


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    ev = prepare_events()
    print(phase_profile(ev).round(3).to_string())
    d = late_goal_detail(ev)
    print(f"\nLate goals: {len(d)}; scored by subs {d.scorer_sub.sum()}, assisted by subs {d.assist_sub.sum()}, "
          f"sub involved {(d.scorer_sub | d.assist_sub).sum()}")
    print(d.state.value_counts().to_string())
    print(d.play_pattern.value_counts().to_string())
    print("\nSet pieces:", set_piece_rate(ev).round(2).to_dict())
    print("Share of late outfield minutes played by subs:", round(sub_share_of_late_minutes(ev), 3))
    print("\nShot cover (open play):")
    print(shot_cover(ev).round(3).to_string())
    print("\nScore state from the 80th minute:", late_state_minutes(ev).round(3).to_dict())
    print(d[["minute", "player", "scorer_sub", "assist", "assist_sub", "score_before", "play_pattern",
             "shot_statsbomb_xg"]].to_string())
