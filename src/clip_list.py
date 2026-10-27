"""Clip list: the exact match moments to find on video for each recommendation in the report.

Every row gives the match (opponent's view: venue, their goals first, plus date), the half and the
match clock as shown on a broadcast (second half starts at 45:00), who is involved and why the
moment was picked. Broadcast clocks can drift a few seconds from the data: start each clip about
10 seconds early.

Output: report/clip_list.csv (for a spreadsheet) and report/clip_list.md (readable).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import OWN_THIRD, TEAM, possessions, prepare_events
from case_study import high_pressures
from data import load_matches
from defending import opponent_shots, wide_transitions
from game_state import add_game_state
from late_goals import late_goal_detail
from network import short, xhaka_receipts
from risk import pressed_buildups
from xt import add_xt, fit_xt

OUT = Path(__file__).resolve().parent.parent / "report"
KEY = ["match_id", "possession"]


def match_labels() -> pd.Series:
    m = load_matches().set_index("match_id")
    lab = {}
    for mid, r in m.iterrows():
        lev_home = r.home_team == "Bayer Leverkusen"
        opp_g, lev_g = (r.away_score, r.home_score) if lev_home else (r.home_score, r.away_score)
        lab[mid] = f"{r.opponent} ({'A' if lev_home else 'H'}) {opp_g}-{lev_g}, {r.match_date}"
    return pd.Series(lab)


def clock(minute, second) -> str:
    return f"{int(minute):02d}:{int(second):02d}"


def row(rec, show, r, labels, players, detail, why):
    return {"Recommendation": rec, "What to show": show, "Match": labels[r.match_id],
            "Half": int(r.period), "Clock": clock(r.minute, r.second),
            "Players": players, "Detail": detail, "Why this clip": why}


def possession_starts(ev):
    e = ev.assign(t=ev.minute * 60 + ev.second)
    return e.groupby(KEY).agg(t0=("t", "min"), period=("period", "first"),
                              team=("possession_team", "first")).reset_index()


def own_possession_after(pr, starts, pos):
    """xG of the pressing team's possession that started within 5 seconds of each pressure."""
    pr = pr.assign(t=pr.minute * 60 + pr.second)
    xg_by = pos.set_index(KEY).xg
    out = []
    for i, r in pr.iterrows():
        s = starts[(starts.match_id == r.match_id) & (starts.period == r.period) & (starts.team == r.team)
                   & (starts.t0 >= r.t) & (starts.t0 <= r.t + 5)]
        out.append(xg_by.get((r.match_id, s.possession.iloc[0]), 0.0) if len(s) else 0.0)
    return pr.assign(xg_after=out)


def build(ev: pd.DataFrame) -> pd.DataFrame:
    labels = match_labels()
    pos = possessions(ev)
    starts = possession_starts(ev)
    gs = add_game_state(ev)
    state_of = gs.set_index("id").state
    rows = []

    # 1. Press high at level scores: presses that won AND kept the ball, at level scores
    pr = high_pressures(ev)
    pr = pr.assign(state=pr.id.map(state_of))
    kept = own_possession_after(pr[pr.kept_5s & (pr.state == "Level")], starts, pos)
    for _, r in kept.sort_values("xg_after", ascending=False).head(4).iterrows():
        rows.append(row("1. Press high at level scores", "A high press that wins the ball and keeps it",
                        r, labels, short(r.player),
                        "Pressure in Leverkusen's third; their own possession within 5s"
                        + (f", ended with {r.xg_after:.2f} xG" if r.xg_after > 0 else ""),
                        "Shows what a successful press on their build-up looks like"))

    # 2. Don't over-commit when chasing: Leverkusen broke a high press while leading, and scored or shot
    b = pressed_buildups(ev, pos).join(gs.groupby(KEY).state.first(), on=KEY)
    esc = b[b.pressed & b.broke_out & (b.state == "Leverkusen leading") & b.shot]
    first_ev = ev.sort_values("index").groupby(KEY).first()
    for _, rb in esc.sort_values("xg", ascending=False).head(4).iterrows():
        r = first_ev.loc[(rb.match_id, rb.possession)]
        r = r.copy(); r["match_id"] = rb.match_id
        shooter = ev[(ev.match_id == rb.match_id) & (ev.possession == rb.possession) & (ev.type == "Shot")].player
        rows.append(row("2. Don't over-commit when chasing", "Leverkusen play through a high press and score or shoot",
                        r, labels, "Shot: " + short(shooter.iloc[-1]) if len(shooter) else "",
                        f"Possession starts here; {rb.xg:.2f} xG" + (" (goal)" if rb.xg > 0 and
                        ((ev.match_id == rb.match_id) & (ev.possession == rb.possession) & (ev.shot_outcome == "Goal")).any() else ""),
                        "Opponent chasing the game, pressing with no cover behind"))

    # 3. Block passes into Xhaka: free receptions that went forward, tight receptions that lost the ball
    rec = xhaka_receipts(ev)
    evx = add_xt(ev, fit_xt(ev))
    xt_next = evx[(evx.player == "Granit Xhaka") & evx.type.isin(["Pass", "Carry"])].set_index(["match_id", "index"]).xt_added
    rec["xt_next"] = [xt_next.get((m, i), 0.0) for m, i in zip(rec.match_id, rec.nidx)]
    # Build-up only: open play in their own half (a second-phase corner is not a build-up)
    open_play = rec.play_pattern.isin(["Regular Play", "From Goal Kick", "From Keeper"]) & (rec.x < 55)
    free = rec[(rec.nearest_m > 6) & rec.progressive & open_play]
    free = free.assign(gain=free.nend_x - free.nx).sort_values("gain", ascending=False)
    for _, r in free.groupby("match_id").head(1).head(3).iterrows():
        rows.append(row("3. Block passes into Xhaka", "Xhaka receives free in the build-up and drives forward", r,
                        labels, "Xhaka", f"Nearest opponent {r.nearest_m:.1f}m; next action moves the ball "
                        f"{r.gain:.0f} yards forward",
                        "The danger if no one is near him"))
    # The clip used in the report: Xhaka free in the second phase of a corner (full match available)
    fre = rec[(rec.match_id == 3895121) & (rec.minute == 28) & (rec.second == 23)]
    for _, r in fre.iterrows():
        rows.append(row("3. Block passes into Xhaka", "Xhaka left free in the second phase of a corner", r, labels,
                        "Grimaldo to Xhaka, then Xhaka to Wirtz", f"Nearest opponent {r.nearest_m:.1f}m; 5 seconds "
                        "on the ball, then a pass into the box", "The danger if no one steps out to him"))
    tight = rec[(rec.nearest_m < 3) & rec.lost].sort_values("nearest_m")
    for _, r in tight.groupby("match_id").head(1).head(3).iterrows():
        rows.append(row("3. Block passes into Xhaka", "Xhaka receives under pressure and loses the ball", r, labels,
                        "Xhaka", f"Nearest opponent {r.nearest_m:.1f}m; next action loses it",
                        "What tight marking on the reception does"))

    # 4. Keep the ball after winning it: Bochum won but lost it again; Bayern/Frankfurt won and kept it
    boch = pr[(pr.match_id == 3895182) & pr.regain_5s & ~pr.kept_5s]
    for _, r in boch.iloc[np.linspace(0, len(boch) - 1, min(4, len(boch))).astype(int)].iterrows():
        rows.append(row("4. Keep the ball after winning it", "Bochum win the ball but give it straight back", r, labels,
                        short(r.player), "Ball won within 5s, but no possession of their own",
                        "Watch for loose touches and no support near the ball"))
    good = own_possession_after(pr[pr.match_id.isin([3895074, 3895333, 3895232]) & pr.kept_5s], starts, pos)
    for _, r in good.sort_values("xg_after", ascending=False).head(3).iterrows():
        rows.append(row("4. Keep the ball after winning it", "Bayern or Frankfurt win it and keep it", r, labels,
                        short(r.player), "Ball won and their own possession within 5s"
                        + (f", {r.xg_after:.2f} xG" if r.xg_after > 0 else ""),
                        "The contrast: a secure first pass after the regain"))

    # 5a. With the ball: opponents carrying out of their own third against Leverkusen's press
    evo = evx[(evx.team != TEAM)]
    carries = evo[(evo.type == "Carry") & (evo.x < OWN_THIRD) & (evo.end_x >= OWN_THIRD) & evo.under_pressure]
    poss_xt = evo.groupby(KEY).xt_added.sum()
    carries = carries.assign(pxt=[poss_xt.get((m, p), 0) for m, p in zip(carries.match_id, carries.possession)])
    for _, r in carries.sort_values("pxt", ascending=False).groupby("match_id").head(1).head(3).iterrows():
        rows.append(row("5. With the ball: carry it out", "An opponent carries out of their third under pressure",
                        r, labels, short(r.player), f"Carry out of own third while pressed; possession then gained "
                        f"{r.pxt * 1000:.0f} xT (x1000)", "Carrying beats their press best"))

    # 5b. Attacks down the flanks that ended in big chances
    sh = opponent_shots(ev, pos)
    wt = wide_transitions(ev, sh).reset_index()
    flank = wt[wt.route != "Through the middle"].sort_values("xg", ascending=False).head(3)
    for _, w in flank.iterrows():
        s = sh[(sh.match_id == w.match_id) & (sh.possession == w.possession)].sort_values("index").iloc[-1]
        rows.append(row("5. With the ball: go wide", "An open-play attack down a flank ending in a big chance",
                        s, labels, short(s.player), f"Shot {s.shot_statsbomb_xg:.2f} xG ({s.shot_outcome}); "
                        f"{w.route.lower()}", "Flank attacks were worth more than central ones"))

    # 5c. Set pieces Leverkusen conceded from
    sp = sh[sh.origin == "Set piece"].sort_values(["shot_outcome", "shot_statsbomb_xg"], ascending=[True, False])
    sp_goals = sp[sp.shot_outcome == "Goal"].head(3)
    for _, s in sp_goals.iterrows():
        rows.append(row("5. With the ball: rehearse set pieces", "A set-piece goal against Leverkusen", s, labels,
                        short(s.player), f"{s.play_pattern.replace('From ', '')}; shot {s.shot_statsbomb_xg:.2f} xG, goal",
                        "Their most common way to concede goals"))

    # 6. Stay switched on: late goals that changed the result
    d = late_goal_detail(ev)
    dec = d[d.state != "Leverkusen leading"]
    full = ev.set_index(ev.index)
    for i, g in dec.iterrows():
        r = full.loc[i]
        rows.append(row("6. Stay switched on to the end", "A late Leverkusen goal that changed the result", r, labels,
                        short(g.player), f"Score before {g.score_before}; {g.play_pattern.replace('From ', '')}"
                        + (" (penalty)" if r.shot_type == "Penalty" else ""),
                        "Set pieces and persistence in the last 10 minutes"))

    # Extra: Leverkusen pressed high and lost the ball in their own half (open-play regain)
    p_ = pos.sort_values(KEY).reset_index(drop=True)
    nx = p_.groupby("match_id")[["possession", "play_pattern", "start_x"]].shift(-1).add_prefix("n_")
    p_ = p_.join(nx)
    lb = b.merge(p_[KEY + ["n_possession", "n_play_pattern", "n_start_x"]], on=KEY)
    lost = lb[lb.pressed & (lb.next_team != TEAM) & lb.n_play_pattern.isin(["Regular Play", "From Counter"])
              & (lb.n_start_x >= 60)]
    opp_goals = ev[(ev.type == "Shot") & (ev.shot_outcome == "Goal") & (ev.team != TEAM)]
    goal_poss = set(zip(opp_goals.match_id, opp_goals.possession))
    lost = lost.assign(goal=[(m, int(q)) in goal_poss for m, q in zip(lost.match_id, lost.n_possession)])
    for _, rl in lost.sort_values(["goal", "next_xg"], ascending=False).head(5).iterrows():
        npos = int(rl.n_possession)
        r = first_ev.loc[(rl.match_id, npos)].copy()
        r["match_id"] = rl.match_id
        last = ev[(ev.match_id == rl.match_id) & (ev.possession == rl.possession) & (ev.team == TEAM)
                  & ev.type.isin(["Pass", "Carry", "Dribble", "Miscontrol", "Dispossessed"])].sort_values("index").iloc[-1]
        how = {"Pass": "misplaced pass", "Dribble": "failed dribble", "Miscontrol": "miscontrol",
               "Dispossessed": "dispossessed", "Carry": "lost on a carry"}[last.type]
        goal = rl.goal
        rows.append(row("Extra: pressed high, Leverkusen lose it", "Leverkusen's build-up is pressed and the ball is won in their half",
                        r, labels, f"{short(last.player)} loses it, {short(r.player)} wins it",
                        f"{how.capitalize()} {120 - rl.n_start_x:.0f} yards from their own goal; opponent then "
                        + (f"{rl.next_xg:.2f} xG" + (" (goal)" if goal else "") if rl.next_xg > 0 else "no shot"),
                        "The press working: where and how they give it away"))

    # Game plan extras: Frimpong as the outlet, Tapsoba carrying out
    exits = evx[(evx.team == TEAM) & (evx.type == "Pass") & evx.pass_outcome.isna() & (evx.x < OWN_THIRD)
                & (evx.end_x >= OWN_THIRD) & (evx.pass_recipient == "Jeremie Frimpong")]
    for _, r in exits.sort_values("xt_added", ascending=False).groupby("match_id").head(1).head(2).iterrows():
        rows.append(row("Game plan: their outlet", "The ball out of their third to Frimpong high on the right", r,
                        labels, f"{short(r.player)} to Frimpong", "Pass out of their defensive third",
                        "Frimpong stays high as the outlet while the build-up goes left"))
    tap = evx[(evx.player == "Edmond Fayçal Tapsoba") & (evx.type == "Carry") & (evx.x < OWN_THIRD)
              & (evx.end_x >= OWN_THIRD)]
    for _, r in tap.sort_values("xt_added", ascending=False).groupby("match_id").head(1).head(2).iterrows():
        rows.append(row("Game plan: centre-back carries", "Tapsoba carries the ball out of their third", r, labels,
                        "Tapsoba", "Carry out of the defensive third", "Step onto him before he carries"))

    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    lines = ["# Clip list: Leverkusen 2023/24 opposition report", "",
             "Exact moments to find on video for each recommendation. Matches are labelled from the opponent's side "
             "(their venue, their goals first). Clock = match clock as on a broadcast (second half starts at 45:00). "
             "Start each clip about 10 seconds early; broadcast clocks can drift a few seconds from the data.", ""]
    for rec, g in df.groupby("Recommendation", sort=False):
        lines += [f"## {rec}", "", f"*{g['Why this clip'].iloc[0]}*", "",
                  "| | Match | Half | Clock | Players | What to show | Detail |", "|---|---|---|---|---|---|---|"]
        for i, r in enumerate(g.itertuples(), 1):
            lines.append(f"| {i} | {r.Match} | {r.Half} | {r.Clock} | {r.Players} | {r._2} | {r.Detail} |")
        lines.append("")
    return "\n".join(lines)


# Shortlist: the best 1-2 clips per point. Chosen for (a) how clearly the moment shows the point,
# (b) a goal or big chance where possible, because those appear in official highlights and are
# easiest to find, and (c) high-profile or case-study matches.
SHORTLIST = {
    ("1. Press high at level scores", "38:49", "VfB Stuttgart (H) 1-1"):
        ("Highlights", "Undav's press wins the ball; Führich scores about 20 seconds later. The whole move is one clip."),
    ("2. Don't over-commit when chasing", "76:15", "Eintracht Frankfurt (H) 1-5"):
        ("Highlights", "Frankfurt chasing at 1-3; Leverkusen play through the press and Frimpong scores."),
    ("2. Don't over-commit when chasing", "91:54", "Bochum (H) 0-5"):
        ("Highlights", "Bochum chasing at 0-4; Leverkusen break the press and Grimaldo scores. Note Bochum had 10 men from the 14th minute."),
    ("3. Block passes into Xhaka", "28:23", "Freiburg (A) 1-2"):
        ("Full match", "Second phase of a corner: Freiburg all back, Xhaka 11.6m from the nearest opponent, "
                       "5 seconds on the ball, then a pass into the box."),
    ("3. Block passes into Xhaka", "38:13", "Bochum (A) 0-4"):
        ("Full match", "Opponent 0.4m away as he receives; he loses it: the fix."),
    ("4. Keep the ball after winning it", "13:59", "Bochum (A) 0-4"):
        ("Full match", "Bochum win the ball high but give it straight back."),
    ("4. Keep the ball after winning it", "12:15", "Bayern Munich (A) 0-3"):
        ("Full match", "The contrast: Sané's team win it and keep it."),
    ("5. With the ball: carry it out", "80:11", "Bayern Munich (H) 2-2"):
        ("Full match", "Sané carries out of Bayern's third under pressure and the attack gains the most threat of any carry."),
    ("5. With the ball: go wide", "57:44", "Hoffenheim (H) 2-3"):
        ("Highlights", "Flank attack ending in Weghorst's goal (0.79 xG)."),
    ("5. With the ball: rehearse set pieces", "06:34", "Bayern Munich (H) 2-2"):
        ("Highlights", "Kane scores from a corner after 6 minutes."),
    ("6. Stay switched on to the end", "96:52", "Borussia Dortmund (H) 1-1"):
        ("Highlights", "Stanišić's corner equaliser in the 97th minute."),
    ("6. Stay switched on to the end", "95:59", "VfB Stuttgart (A) 2-2"):
        ("Highlights", "Andrich's free-kick equaliser in the 96th minute."),
    ("Extra: pressed high, Leverkusen lose it", "61:17", "Augsburg (A) 1-2"):
        ("Highlights", "Adli is pressed on a dribble in Leverkusen's half; Kömür scores 4 seconds later."),
    ("Extra: pressed high, Leverkusen lose it", "57:23", "FC Heidenheim (A) 1-4"):
        ("Highlights", "Xhaka's pass is won high; Dinkçi scores 11 seconds later."),
    ("Game plan: their outlet", "46:13", "Borussia Dortmund (H) 1-1"):
        ("Full match", "Andrich plays out to Frimpong high on the right."),
    ("Game plan: centre-back carries", "73:59", "FSV Mainz 05 (H) 0-3"):
        ("Full match", "Tapsoba carries out of the defensive third."),
}


def shortlist(df: pd.DataFrame) -> pd.DataFrame:
    keys = [(r.Recommendation, r.Clock, r.Match.split(",")[0]) for r in df.itertuples()]
    hit = [SHORTLIST.get(k) for k in keys]
    df = df.assign(Shortlist=["Yes" if h else "" for h in hit],
                   Footage=[h[0] if h else "" for h in hit],
                   **{"What to look for": [h[1] if h else "" for h in hit]})
    missing = set(SHORTLIST) - set(keys)
    assert not missing, f"shortlist keys not found: {missing}"
    return df


def shortlist_markdown(df: pd.DataFrame) -> str:
    s = df[df.Shortlist == "Yes"]
    lines = ["# Clip shortlist: the best 1-2 clips per point", "",
             f"{len(s)} clips. 'Highlights' = in official match highlights (a goal or big chance). "
             "'Full match' = needs a full-match replay or a video platform such as Wyscout. "
             "Clock = match clock as on a broadcast; start about 10 seconds early.", "",
             "| Point | Match | Half | Clock | Footage | What to look for |", "|---|---|---|---|---|---|"]
    for _, r in s.iterrows():
        lines.append(f"| {r.Recommendation} | {r.Match} | {r.Half} | {r.Clock} | {r.Footage} | {r['What to look for']} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    df = build(prepare_events())
    df.insert(0, "Done", "")
    df = shortlist(df.astype({"Half": str}))
    df.to_csv(OUT / "clip_list.csv", index=False)
    (OUT / "clip_list.md").write_text(to_markdown(df.drop(columns=["Done", "Shortlist", "Footage", "What to look for"])))
    (OUT / "clip_shortlist.md").write_text(shortlist_markdown(df))
    print(df[df.Shortlist == "Yes"][["Recommendation", "Match", "Clock", "Footage"]].to_string())
