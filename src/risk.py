"""The risk side of pressing Leverkusen high.

Coach's question: "If we press them high and they play through it, how much do we concede?"

A pressed build-up = a Leverkusen possession that started in their own third and had at
least one opponent pressure inside Leverkusen's defensive third.
  - broke the press: the possession then got out of Leverkusen's own third
  - fast break: Leverkusen shot within FAST_SECS of getting out

Reward = opponent xG from the possession straight after a pressed build-up they won back.
Risk   = Leverkusen xG from pressed build-ups that broke the press.
Both are about the same situations, so they can be compared directly.
"""
import numpy as np
import pandas as pd

from analysis import FINAL_THIRD, OWN_THIRD, TEAM, buildup_possessions, possessions, prepare_events
from case_study import CASES
from data import load_matches

FAST_SECS = 15
KEY = ["match_id", "possession"]


def pressed_buildups(ev: pd.DataFrame, pos: pd.DataFrame) -> pd.DataFrame:
    ev = ev.assign(t=ev.minute * 60 + ev.second)
    b = buildup_possessions(pos).copy()
    inb = ev.set_index(KEY).index.isin(b.set_index(KEY).index)
    e = ev[inb]
    hp = e[(e.team != TEAM) & (e.type == "Pressure") & (e.x >= FINAL_THIRD)]
    b = b.join(hp.groupby(KEY).size().rename("high_pressures"), on=KEY)
    b["high_pressures"] = b.high_pressures.fillna(0).astype(int)
    b["pressed"] = b.high_pressures > 0

    lev = e[e.team == TEAM]
    far = lev[["x", "end_x"]].max(axis=1)
    t_exit = lev[far >= OWN_THIRD].groupby(KEY).t.min().rename("t_exit")
    shots = lev[lev.type == "Shot"]
    t_shot = shots.groupby(KEY).t.min().rename("t_shot")
    b = b.join(t_exit, on=KEY).join(t_shot, on=KEY)
    b["broke_out"] = b.t_exit.notna()
    b["fast_break"] = b.broke_out & b.shot & ((b.t_shot - b.t_exit) <= FAST_SECS)
    b["fast_xg"] = np.where(b.fast_break, b.xg, 0.0)
    return b


def compare_pressed(b: pd.DataFrame) -> pd.DataFrame:
    g = b.groupby("pressed").agg(
        possessions=("possession", "size"),
        broke_out=("broke_out", "mean"),
        reached_final_third=("reached_final_third", "mean"),
        shot_rate=("shot", "mean"),
        xg_per_100=("xg", lambda s: 100 * s.mean()),
        fast_break_rate=("fast_break", "mean"),
        lost_own_third=("lost_own_third", "mean"),
    )
    escaped = b[b.broke_out].groupby("pressed").agg(
        shot_rate_once_out=("shot", "mean"), xg_per_100_once_out=("xg", lambda s: 100 * s.mean()))
    out = g.join(escaped)
    out.index = out.index.map({True: "Pressed high", False: "Not pressed high"})
    return out


def risk_reward(ev: pd.DataFrame, pos: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    m = load_matches().set_index("match_id")
    won = b[b.pressed & (b.next_team != TEAM)]
    pb = b[b.pressed & b.broke_out]
    df = pd.DataFrame({
        "opponent": m.opponent, "venue": m.venue,
        "score": m.apply(lambda r: f"{r.home_score}-{r.away_score}", axis=1),
        "pressed_buildups": b[b.pressed].groupby("match_id").size(),
        "reward_xg": won.groupby("match_id").next_xg.sum(),
        "risk_xg": pb.groupby("match_id").xg.sum(),
        "fast_break_xg": pb.groupby("match_id").fast_xg.sum(),
        "fast_breaks": pb.groupby("match_id").fast_break.sum(),
    }).fillna(0)
    df["net"] = df.reward_xg - df.risk_xg
    # Opponent's view: their venue, their goals first (same convention as CASES)
    df["label"] = [f"{r.opponent} ({'A' if r.venue == 'H' else 'H'}) "
                   f"{m.loc[i, 'away_score'] if r.venue == 'H' else m.loc[i, 'home_score']}-"
                   f"{m.loc[i, 'home_score'] if r.venue == 'H' else m.loc[i, 'away_score']}"
                   for i, r in df.iterrows()]
    df["case"] = df.index.isin(list(CASES))
    return df.sort_values("net", ascending=False)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    ev = prepare_events()
    pos = possessions(ev)
    b = pressed_buildups(ev, pos)
    print(compare_pressed(b).round(3).T.to_string())
    rr = risk_reward(ev, pos, b)
    print(rr.round(2).to_string())
    print("\ncorr(pressed_buildups, risk_xg) =", round(rr.pressed_buildups.corr(rr.risk_xg), 2),
          " corr(pressed_buildups, reward_xg) =", round(rr.pressed_buildups.corr(rr.reward_xg), 2))
    print("season totals: reward", rr.reward_xg.sum().round(2), "risk", rr.risk_xg.sum().round(2),
          "fast-break xG", rr.fast_break_xg.sum().round(2))
