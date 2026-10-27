"""Build-up pass network and Xhaka's marking.

Coach's questions:
  1. Who does Leverkusen's build-up run through, and does that change when they're pressed?
  2. Xhaka is the hub. If we mark him tightly when he receives, does their build-up suffer?

Build-up phase = Leverkusen open-play passes that start in their own half (x < 60).
"""
import numpy as np
import pandas as pd

from analysis import TEAM, buildup_possessions, minutes_played, possessions, prepare_events
from case_study import CASES
from data import load_frames

BAYERN = [m for m, name in CASES.items() if name.startswith("Bayern")]
XHAKA = "Granit Xhaka"
BUILDUP_MAX_X = 60
TIGHT_M, FREE_M = 3, 6   # marking bands at the moment Xhaka receives (metres)

SHORT = {"Alejandro Grimaldo García": "Grimaldo", "Piero Martín Hincapié Reyna": "Hincapié",
         "Exequiel Alejandro Palacios": "Palacios", "Victor Okoh Boniface": "Boniface",
         "Edmond Fayçal Tapsoba": "Tapsoba", "Lukáš Hrádecký": "Hrádecký",
         "Gustavo Adolfo Puerta Molano": "Puerta", "Lucas Silva Melo": "Tuta"}


def short(name: str) -> str:
    return SHORT.get(name, name.split()[-1])


def buildup_passes(ev: pd.DataFrame) -> pd.DataFrame:
    p = ev[(ev.team == TEAM) & (ev.type == "Pass") & ev.pass_outcome.isna()
           & (ev.pass_type.isna() | ev.pass_type.isin(["Recovery", "Interception", "Goal Kick"]))
           & (ev.x < BUILDUP_MAX_X) & ev.pass_recipient.notna()]
    return p


def network(ev: pd.DataFrame, match_ids, n_players: int = 11, min_share: float = 0.012):
    """Nodes (avg position, involvement) and edges (pass counts per match) for a set of matches.

    Players = the 11 with the most minutes in those matches, i.e. the side that usually played,
    so every position appears even if (like Frimpong) a player is rarely in the first phase."""
    sub = ev[ev.match_id.isin(match_ids)]
    p = buildup_passes(sub)
    n_matches = len(match_ids)
    n_passes_all = len(p)
    inv = pd.concat([p.player, p.pass_recipient]).value_counts()
    players = minutes_played(sub).sort_values(ascending=False).head(n_players).index
    p = p[p.player.isin(players) & p.pass_recipient.isin(players)]

    loc = pd.concat([
        p[["player", "x", "y"]],
        p[["pass_recipient", "end_x", "end_y"]].set_axis(["player", "x", "y"], axis=1),
    ])
    nodes = loc.groupby("player")[["x", "y"]].mean()
    nodes["involvement"] = inv.reindex(nodes.index)
    nodes["share"] = nodes.involvement / n_passes_all  # % of passes the player made or received
    nodes["per_match"] = nodes.involvement / n_matches

    pair = p.apply(lambda r: tuple(sorted((r.player, r.pass_recipient))), axis=1)
    edges = pair.value_counts().rename("passes").reset_index()
    edges[["a", "b"]] = pd.DataFrame(edges["index"].tolist() if "index" in edges else edges.iloc[:, 0].tolist())
    edges["per_match"] = edges.passes / n_matches
    edges["share"] = edges.passes / len(p)
    edges = edges[edges.share >= min_share][["a", "b", "passes", "per_match", "share"]]
    return nodes, edges


def xhaka_receipts(ev: pd.DataFrame) -> pd.DataFrame:
    """Every Xhaka reception in the build-up zone with a 360 frame: how much space did he have,
    and what did he do next?"""
    rec = ev[(ev.type == "Ball Receipt*") & (ev.player == XHAKA) & ev.ball_receipt_outcome.isna()
             & (ev.x < 80)].copy()
    frames = load_frames()
    fr = frames[frames.id.isin(rec.id)]
    xy = np.vstack(fr.location.to_numpy()) * np.array([105 / 120, 68 / 80])
    fr = fr.assign(fx=xy[:, 0], fy=xy[:, 1])
    rows = []
    for eid, g in fr.groupby("id", sort=False):
        actor = g[g.actor]
        opp = g[~g.teammate & ~g.keeper]
        if actor.empty or opp.empty:
            continue
        d = np.hypot(opp.fx - actor.fx.iat[0], opp.fy - actor.fy.iat[0])
        rows.append((eid, d.min(), (d < 5).sum()))
    space = pd.DataFrame(rows, columns=["id", "nearest_m", "opp_within_5m"])
    rec = rec.merge(space, on="id")

    # His next on-ball action
    nxt = ev[(ev.player == XHAKA) & ev.type.isin(["Pass", "Carry", "Dribble", "Miscontrol", "Dispossessed"])]
    nxt = nxt[["match_id", "index", "type", "x", "end_x", "pass_outcome", "dribble_outcome"]]
    rec = rec.sort_values(["match_id", "index"])
    nxt = nxt.sort_values(["match_id", "index"]).rename(columns={"index": "nidx", "type": "next_type",
                                                                  "x": "nx", "end_x": "nend_x"})
    rec = pd.merge_asof(rec.sort_values("index"), nxt.sort_values("nidx"), left_on="index", right_on="nidx",
                        by="match_id", direction="forward")
    rec = rec[(rec.nidx - rec["index"]) <= 3]  # next action follows directly
    rec["lost"] = (rec.next_type.isin(["Miscontrol", "Dispossessed"])
                   | ((rec.next_type == "Pass") & rec.pass_outcome_y.notna())
                   | ((rec.next_type == "Dribble") & (rec.dribble_outcome_y == "Incomplete")))
    rec["progressive"] = (rec.nend_x - rec.nx >= 10) & ~rec.lost
    rec["turned"] = rec.next_type.eq("Carry") | (rec.nend_x > rec.nx + 2)
    rec["band"] = pd.cut(rec.nearest_m, [0, TIGHT_M, FREE_M, 200],
                         labels=[f"Tight (<{TIGHT_M}m)", f"{TIGHT_M}-{FREE_M}m", f"Free (>{FREE_M}m)"])
    return rec


def xhaka_by_match(ev: pd.DataFrame, rec: pd.DataFrame) -> pd.DataFrame:
    pos = possessions(ev)
    b = buildup_possessions(pos)
    p = buildup_passes(ev)
    inv = (p.player.eq(XHAKA) | p.pass_recipient.eq(XHAKA)).groupby(p.match_id).mean()
    return pd.DataFrame({
        "xhaka_space_m": rec.groupby("match_id").nearest_m.median(),
        "xhaka_tight_share": rec.groupby("match_id").nearest_m.apply(lambda s: (s < TIGHT_M).mean()),
        "xhaka_receipts": rec.groupby("match_id").size(),
        "xhaka_buildup_share": inv,
        "reach_final_third": b.groupby("match_id").reached_final_third.mean(),
    }).dropna()


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    ev = prepare_events()
    others = [m for m in ev.match_id.unique() if m not in BAYERN]
    for name, ids in [("Other 32 matches", others), ("Both Bayern matches", BAYERN)]:
        nodes, edges = network(ev, ids)
        print(f"\n{name}: nodes")
        print(nodes.sort_values("share", ascending=False).round(3).to_string())
        print(edges.sort_values("passes", ascending=False).head(8).round(3).to_string())
    rec = xhaka_receipts(ev)
    print(f"\nXhaka receptions with 360: {len(rec)}")
    print(rec.groupby("band", observed=True).agg(n=("id", "size"), lost=("lost", "mean"),
          progressive=("progressive", "mean"), turned=("turned", "mean")).round(3).to_string())
    bm = xhaka_by_match(ev, rec)
    print(bm.corr().round(2).to_string())
    print(bm.loc[BAYERN].round(3).to_string())
    print(bm.drop(BAYERN).median().round(3).to_string())
