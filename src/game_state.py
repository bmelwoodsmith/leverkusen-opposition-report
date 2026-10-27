"""Game state: does the pressing finding survive, and when do Leverkusen score?

1. Robustness check: teams chasing a game press more and leave more space, which could
   inflate the "pressing high costs more than it wins" finding. Re-run it by score state.
2. Late goals: when Leverkusen scored and conceded, and how many points late goals were worth.

Score is rebuilt from events (goals + own goals) and matches the official result in all 34 matches.
"""
import numpy as np
import pandas as pd

from analysis import TEAM, possessions, prepare_events
from data import load_matches
from risk import pressed_buildups

KEY = ["match_id", "possession"]
STATES = ["Leverkusen trailing", "Level", "Leverkusen leading"]


def goals(ev: pd.DataFrame) -> pd.DataFrame:
    """One row per goal, credited to the scoring team (own goals via 'Own Goal For')."""
    g = ev[((ev.type == "Shot") & (ev.shot_outcome == "Goal")) | (ev.type == "Own Goal For")]
    g = g[["match_id", "index", "period", "minute", "second", "team", "type"]].copy()
    g["lev"] = g.team == TEAM
    return g


def add_game_state(ev: pd.DataFrame) -> pd.DataFrame:
    """Score before each event, from Leverkusen's side."""
    ev = ev.sort_values(["match_id", "index"]).copy()
    g = goals(ev)
    ev["lev_goal"] = 0
    ev["opp_goal"] = 0
    ev.loc[g.index[g.lev], "lev_goal"] = 1
    ev.loc[g.index[~g.lev], "opp_goal"] = 1
    grp = ev.groupby("match_id")
    # Goals strictly before this event
    ev["lev_score"] = grp.lev_goal.cumsum() - ev.lev_goal
    ev["opp_score"] = grp.opp_goal.cumsum() - ev.opp_goal
    diff = ev.lev_score - ev.opp_score
    ev["state"] = np.select([diff < 0, diff == 0], STATES[:2], default=STATES[2])
    # Numbers: any sending-off before this event (all five in these matches were opponents)
    red = (ev.foul_committed_card.isin(["Red Card", "Second Yellow"])
           | ev.bad_behaviour_card.isin(["Red Card", "Second Yellow"])).astype(int)
    ev["opp_reds"] = (red * (ev.team != TEAM)).groupby(ev.match_id).cumsum() - red * (ev.team != TEAM)
    ev["lev_reds"] = (red * (ev.team == TEAM)).groupby(ev.match_id).cumsum() - red * (ev.team == TEAM)
    ev["even_numbers"] = ev.opp_reds == ev.lev_reds
    return ev


def risk_by_state(ev: pd.DataFrame, pos: pd.DataFrame) -> pd.DataFrame:
    """The risk/reward and pressed-vs-not comparisons, split by score state at the start
    of each Leverkusen build-up."""
    ev = add_game_state(ev)
    b = pressed_buildups(ev, pos)
    first = ev.groupby(KEY)[["state", "even_numbers"]].first()
    b = b.join(first, on=KEY)
    rows = []
    for st in STATES + ["Level, 11 v 11"]:
        s = b[(b.state == "Level") & b.even_numbers] if st == "Level, 11 v 11" else b[b.state == st]
        p, n = s[s.pressed], s[~s.pressed]
        won = p[p.next_team != TEAM]
        rows.append({
            "state": st,
            "buildups": len(s),
            "pressed_share": len(p) / max(len(s), 1),
            "reward_xg": won.next_xg.sum(),
            "risk_xg": p[p.broke_out].xg.sum(),
            "reward_per100": 100 * won.next_xg.sum() / max(len(p), 1),
            "risk_per100": 100 * p[p.broke_out].xg.sum() / max(len(p), 1),
            "ft_pressed": p.reached_final_third.mean(),
            "ft_not": n.reached_final_third.mean(),
            "xg100_out_pressed": 100 * p[p.broke_out].xg.mean(),
            "xg100_out_not": 100 * n[n.broke_out].xg.mean(),
        })
    return pd.DataFrame(rows).set_index("state")


def minute_bin(period, minute):
    """15-minute bins, with stoppage time in its own bin."""
    if period == 1:
        return "45+" if minute >= 45 else ["0-15", "16-30", "31-45"][min(minute // 15, 2)]
    if minute >= 90:
        return "90+"
    return ["46-60", "61-75", "76-90"][min((minute - 45) // 15, 2)]


BINS = ["0-15", "16-30", "31-45", "45+", "46-60", "61-75", "76-90", "90+"]


def goals_by_time(ev: pd.DataFrame) -> pd.DataFrame:
    g = goals(ev)
    g["bin"] = [minute_bin(p, m) for p, m in zip(g.period, g.minute)]
    t = g.groupby(["bin", "lev"]).size().unstack(fill_value=0).reindex(BINS, fill_value=0)
    return t.rename(columns={True: "scored", False: "conceded"})[["scored", "conceded"]]


def points(gf, ga):
    return 3 if gf > ga else 1 if gf == ga else 0


def late_goal_points(ev: pd.DataFrame, from_minute: int = 80) -> pd.DataFrame:
    """Points won by Leverkusen's late goals: final points vs points with their goals from
    `from_minute` onwards removed (opponents' late goals kept)."""
    g = goals(ev)
    late = (g.period == 2) & (g.minute >= from_minute)
    m = load_matches().set_index("match_id")
    rows = []
    for mid, r in m.iterrows():
        gm, lm = g[g.match_id == mid], late[g.match_id == mid]
        ff, fa = gm.lev.sum(), (~gm.lev).sum()
        late_for = (gm.lev & lm).sum()
        rows.append({"match_id": mid, "opponent": r.opponent, "venue": r.venue, "final": f"{ff}-{fa}",
                     "late_scored": late_for, "late_conceded": (~gm.lev & lm).sum(),
                     "points_final": points(ff, fa), "points_without": points(ff - late_for, fa)})
    df = pd.DataFrame(rows)
    df["points_gained"] = df.points_final - df.points_without
    return df


def trailed(ev: pd.DataFrame) -> pd.DataFrame:
    """Matches where Leverkusen were behind at some point, and how they finished."""
    ev = add_game_state(ev)
    m = load_matches().set_index("match_id")
    out = []
    for mid in ev[ev.state == "Leverkusen trailing"].match_id.unique():
        r = m.loc[mid]
        lev_home = r.home_team == TEAM
        gf, ga = (r.home_score, r.away_score) if lev_home else (r.away_score, r.home_score)
        out.append({"opponent": r.opponent, "venue": r.venue, "final": f"{gf}-{ga}", "points": points(gf, ga)})
    return pd.DataFrame(out)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    ev = prepare_events()
    pos = possessions(ev)
    print(risk_by_state(ev, pos).round(3).T.to_string())
    print("\nGoals by time:")
    print(goals_by_time(ev).to_string())
    lg = late_goal_points(ev)
    print("\nLate goals (80'+):", lg.late_scored.sum(), "scored,", lg.late_conceded.sum(), "conceded;",
          "points gained after 80':", lg.points_gained.sum())
    print(lg[lg.points_gained != 0][["opponent", "venue", "final", "late_scored", "points_gained"]].to_string())
    print("\nMatches where they trailed:")
    print(trailed(ev).to_string())
