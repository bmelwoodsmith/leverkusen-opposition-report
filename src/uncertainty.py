"""How sure are we? 90% confidence intervals for the report's key claims.

Method: cluster bootstrap by match. Each resample draws 34 matches with replacement and
recomputes the statistic, so within-match correlation (same opponent, same game state)
is respected. 90% interval = 5th to 95th percentile of 2,000 resamples.
"""
import numpy as np
import pandas as pd

from analysis import TEAM, possessions, prepare_events
from case_study import match_table
from defending import losses, opponent_shots, wide_transitions
from game_state import add_game_state
from network import xhaka_receipts
from risk import pressed_buildups

N_BOOT = 2000
RNG = np.random.default_rng(2024)
KEY = ["match_id", "possession"]


def boot(per_match: pd.DataFrame, stat) -> tuple:
    """per_match: one row per match with summable columns; stat(df_sum) -> float."""
    ids = per_match.index.to_numpy()
    point = stat(per_match.sum())
    draws = []
    for _ in range(N_BOOT):
        sample = per_match.loc[RNG.choice(ids, len(ids), replace=True)]
        draws.append(stat(sample.sum()))
    lo, hi = np.nanpercentile(draws, [5, 95])
    return point, lo, hi


def claims(ev: pd.DataFrame) -> pd.DataFrame:
    pos = possessions(ev)
    out = []

    # Pressing risk vs reward, all states and level only
    gs = add_game_state(ev)
    b = pressed_buildups(ev, pos).join(gs.groupby(KEY).state.first(), on=KEY)
    for label, sub in [("all score states", b), ("level score only", b[b.state == "Level"])]:
        p = sub[sub.pressed]
        pm = pd.DataFrame({
            "reward": p[p.next_team != TEAM].groupby("match_id").next_xg.sum(),
            "risk": p[p.broke_out].groupby("match_id").xg.sum(),
            "out_p_xg": p[p.broke_out].groupby("match_id").xg.sum(),
            "out_p_n": p[p.broke_out].groupby("match_id").size(),
            "out_n_xg": sub[~sub.pressed & sub.broke_out].groupby("match_id").xg.sum(),
            "out_n_n": sub[~sub.pressed & sub.broke_out].groupby("match_id").size(),
            "ft_p": p.groupby("match_id").reached_final_third.sum(),
            "n_p": p.groupby("match_id").size(),
            "ft_n": sub[~sub.pressed].groupby("match_id").reached_final_third.sum(),
            "n_n": sub[~sub.pressed].groupby("match_id").size(),
        }).reindex(ev.match_id.unique()).fillna(0)
        out.append((f"Risk minus reward, xG ({label})", *boot(pm, lambda s: s.risk - s.reward)))
        out.append((f"Extra xG per 100 escaped build-ups when pressed ({label})",
                    *boot(pm, lambda s: 100 * (s.out_p_xg / s.out_p_n - s.out_n_xg / s.out_n_n))))
        out.append((f"Drop in final-third reach when pressed, pts ({label})",
                    *boot(pm, lambda s: 100 * (s.ft_n / s.n_n - s.ft_p / s.n_p))))

    # Press height vs progression correlation (per-match rows, so resample rows directly)
    t = match_table(ev, pos)
    ids = t.index.to_numpy()
    r0 = np.corrcoef(t.high_press_share, t.reach_final_third)[0, 1]
    rs = []
    for _ in range(N_BOOT):
        s = t.loc[RNG.choice(ids, len(ids), replace=True)]
        rs.append(np.corrcoef(s.high_press_share, s.reach_final_third)[0, 1])
    out.append(("Correlation: press height vs Leverkusen progression", r0, *np.percentile(rs, [5, 95])))

    # Xhaka: forward-play rate when free vs with an opponent within 6m
    rec = xhaka_receipts(ev)
    free = rec.nearest_m > 6
    pm = pd.DataFrame({
        "f_prog": rec[free].groupby("match_id").progressive.sum(), "f_n": rec[free].groupby("match_id").size(),
        "c_prog": rec[~free].groupby("match_id").progressive.sum(), "c_n": rec[~free].groupby("match_id").size(),
    }).fillna(0)
    out.append(("Xhaka forward play: free minus marked within 6m, pts",
                *boot(pm, lambda s: 100 * (s.f_prog / s.f_n - s.c_prog / s.c_n))))

    # Counter-press: Leverkusen vs opponents, final third
    lo_ = losses(ev)
    f3 = lo_[lo_.zone == "Final third"]
    pm = pd.DataFrame({
        "l_r": f3[f3.side == "Leverkusen"].groupby("match_id").regain_5s.sum(),
        "l_n": f3[f3.side == "Leverkusen"].groupby("match_id").size(),
        "o_r": f3[f3.side == "Opponents"].groupby("match_id").regain_5s.sum(),
        "o_n": f3[f3.side == "Opponents"].groupby("match_id").size(),
    }).fillna(0)
    out.append(("Counter-press, final third: Leverkusen minus opponents, pts",
                *boot(pm, lambda s: 100 * (s.l_r / s.l_n - s.o_r / s.o_n))))

    # Flanks vs middle: xG per open-play attack
    wt = wide_transitions(ev, opponent_shots(ev, pos)).reset_index()
    wide = wt.route != "Through the middle"
    pm = pd.DataFrame({
        "w_xg": wt[wide].groupby("match_id").xg.sum(), "w_n": wt[wide].groupby("match_id").size(),
        "m_xg": wt[~wide].groupby("match_id").xg.sum(), "m_n": wt[~wide].groupby("match_id").size(),
    }).fillna(0)
    out.append(("Flank minus middle, xG per open-play shot attack",
                *boot(pm, lambda s: s.w_xg / s.w_n - s.m_xg / s.m_n)))
    sides = wt[wt.route.str.startswith("Down")]
    pm = pd.DataFrame({
        "f": sides[sides.route.str.contains("Frimpong")].groupby("match_id").xg.sum(),
        "g": sides[sides.route.str.contains("Grimaldo")].groupby("match_id").xg.sum(),
    }).reindex(ev.match_id.unique()).fillna(0)
    out.append(("Frimpong side minus Grimaldo side, total xG", *boot(pm, lambda s: s.f - s.g)))

    df = pd.DataFrame(out, columns=["claim", "estimate", "low_90", "high_90"])
    df["clear_of_zero"] = (df.low_90 > 0) | (df.high_90 < 0)
    return df


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    df = claims(prepare_events())
    df.to_csv("../data/claim_intervals.csv", index=False)
    print(df.round(3).to_string(index=False))
