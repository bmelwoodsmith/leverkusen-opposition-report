"""Report figures. Run: python figures.py  ->  saves PNGs in figures/."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer import Pitch

from analysis import (buildup_possessions, buildup_turnovers, exit_channel,
                      exit_routes, goal_kick_press, opponent_press_profile,
                      player_press_resistance, possessions, prepare_events)
from case_study import CASES, goal_sources, high_pressures, match_table
from late_goals import goal_types, late_goal_detail, open_play_rates, phase_profile
from network import BAYERN, XHAKA, network, short, xhaka_by_match, xhaka_receipts
from defending import (counterpress, exit_table, losses, opponent_exits, opponent_shots,
                       pressers, wide_transitions)
from game_state import goals_by_time, late_goal_points, risk_by_state
from risk import compare_pressed, pressed_buildups, risk_reward
from xt import L, W, add_xt, fit_xt

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#a3a29c"
BLUE = "#2a78d6"     # Leverkusen / main series
AQUA = "#1baf7a"
ORANGE = "#eb6834"   # highlight
SOURCE = "Data: StatsBomb Open Data, Bundesliga 2023/24"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": MUTED, "font.size": 10, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e8e7e3", "grid.linewidth": 0.6,
})


def _title(fig, title, subtitle):
    fig.text(0.04, 0.965, title, fontsize=14, fontweight="bold", va="top")
    fig.text(0.04, 0.915, subtitle, fontsize=10, color=INK_2, va="top")
    fig.text(0.04, 0.02, SOURCE, fontsize=8, color=MUTED)


def fig_exit_channels(ev, b):
    ex = exit_channel(ev, b)
    pitch = Pitch(pitch_type="statsbomb", pitch_color=SURFACE, line_color=MUTED, linewidth=1)
    fig, axs = pitch.grid(figheight=8, title_height=0.12, endnote_height=0.03,
                          title_space=0.01, endnote_space=0.01, axis=False)
    ax = axs["pitch"]
    stats = pitch.bin_statistic(ex.end_x, ex.end_y, statistic="count", bins=(12, 8))
    pitch.heatmap(stats, ax=ax, cmap="Blues", edgecolors=SURFACE, lw=1.5, alpha=0.9)
    ax.plot([40, 40], [0, 80], color=ORANGE, lw=2, ls="--")
    ax.text(40.8, 1.5, "edge of own third", color=ORANGE, fontsize=9, va="bottom")
    shares = ex.channel.value_counts(normalize=True)
    centres = {"Left wing": 9, "Left half-space": 24, "Centre": 40,
               "Right half-space": 56, "Right wing": 71}
    for ch, y in centres.items():
        ax.text(20, y, f"{ch}\n{shares[ch]:.0%}", ha="center", va="center",
                fontsize=10, color=INK, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE, ec=MUTED, lw=0.6))
    top = ex.receiver.value_counts().head(4)
    who = ", ".join(f"{n.split()[-1] if 'García' not in n else 'Grimaldo'} {c}" for n, c in top.items())
    _title(fig, "Leverkusen leave their own third through every channel",
           f"Where the ball lands when a build-up first leaves the defensive third ({len(ex)} exits). "
           f"Attacking left to right.\nMost frequent receivers: {who}")
    fig.savefig(OUT / "01_buildup_exit_channels.png", dpi=200)
    plt.close(fig)


# Matches worth naming on the chart, with label offsets (points) to avoid collisions
LABELS = {
    ("Bayern Munich", "H"): (8, -4), ("Bayern Munich", "A"): (8, -4),
    ("VfB Stuttgart", "A"): (9, -4), ("VfB Stuttgart", "H"): (-60, -18),
    ("Eintracht Frankfurt", "A"): (-50, -16), ("Bochum", "H"): (-40, 9), ("Borussia Dortmund", "H"): (8, 2),
    ("Union Berlin", "A"): (8, 2),
}


def fig_press_vs_progression(prof):
    fig, ax = plt.subplots(figsize=(10, 6.6))
    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.09, right=0.97)
    x = prof.high_press_share * 100
    y = prof.buildup_reach_final_third * 100
    ax.scatter(x, y, s=70, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
    m, c = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, m * xs + c, color=INK_2, lw=1.5, ls="--", zorder=2)
    r = np.corrcoef(x, y)[0, 1]
    for _, row in prof.assign(px=x, py=y).iterrows():
        offset = LABELS.get((row.opponent, row.venue))
        if offset:
            h, a = row.score.split("-")
            opp_view = f"{row.opponent} ({'A' if row.venue == 'H' else 'H'}) {a + '-' + h if row.venue == 'H' else row.score}"
            ax.annotate(opp_view, (row.px, row.py),
                        xytext=offset, textcoords="offset points", fontsize=8.5, color=INK_2)
    ax.set_xlim(right=x.max() + 8)
    ax.set_xlabel("Opponent pressures in Leverkusen's defensive third (% of all their pressures)")
    ax.set_ylabel("Leverkusen build-ups reaching final third (%)")
    _title(fig, "The higher teams pressed, the less often Leverkusen got up the pitch",
           f"One dot per match, 2023/24 Bundesliga (34 matches). Correlation r = {r:.2f} (90% interval -0.84 to -0.61).\n"
           f"Build-up = possession started in own third. Labels are from the opponent's side: their venue, their goals first.")
    fig.savefig(OUT / "02_press_height_vs_progression.png", dpi=200)
    plt.close(fig)


def fig_player_press(pr):
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.subplots_adjust(top=0.85, bottom=0.12, left=0.09, right=0.97)
    lx, py = pr.loss_rate * 100, pr.progressive_rate * 100
    target = (lx > lx.median()) & (py < py.median())
    ax.scatter(lx[~target], py[~target], s=pr.actions[~target] / 3, color=BLUE,
               edgecolor=SURFACE, linewidth=2, alpha=0.9, zorder=3)
    ax.scatter(lx[target], py[target], s=pr.actions[target] / 3, color=ORANGE,
               edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.axvline(lx.median(), color=MUTED, lw=1)
    ax.axhline(py.median(), color=MUTED, lw=1)
    for name, row in pr.iterrows():
        short = "Grimaldo" if "Grimaldo" in name else "Hincapié" if "Hincapié" in name else name.split()[-1]
        if "Palacios" in name:
            short = "Palacios"
        ax.annotate(short, (row.loss_rate * 100, row.progressive_rate * 100), xytext=(7, 3),
                    textcoords="offset points", fontsize=8.5, color=INK)
    ax.text(0.99, 0.02, "PRESS TARGETS\nlose it often, rarely progress", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color=ORANGE, fontweight="bold")
    ax.text(0.01, 0.98, "AVOID PRESSING\nsecure and progressive", transform=ax.transAxes,
            ha="left", va="top", fontsize=9, color=BLUE, fontweight="bold")
    ax.set_xlabel("Lose the ball when pressed in own half (%)")
    ax.set_ylabel("Actions that move the ball 10+ yards forward (%)")
    _title(fig, "Who to press: Leverkusen players on the ball under pressure",
           "Own half only, players with 60+ actions under pressure. Dot size = number of actions. "
           "Lines = squad medians.")
    fig.savefig(OUT / "03_player_press_resistance.png", dpi=200)
    plt.close(fig)


def fig_goal_kicks(gk):
    g = gk.groupby("press", observed=True).agg(
        n=("id", "size"), short=("short", "mean"), final=("reached_final_third", "mean"))
    fig, ax = plt.subplots(figsize=(10, 5.8))
    fig.subplots_adjust(top=0.82, bottom=0.2, left=0.09, right=0.97)
    idx = np.arange(len(g))
    w = 0.36
    b1 = ax.bar(idx - w / 2 - 0.01, g.short * 100, w, color=BLUE, label="Goal kick played short")
    b2 = ax.bar(idx + w / 2 + 0.01, g.final * 100, w, color=ORANGE, label="Possession reached final third")
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.0f%%", padding=3, fontsize=9, color=INK)
    ax.set_xticks(idx, [f"{lab}\n(n = {n})" for lab, n in zip(g.index, g.n)])
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of goal kicks")
    ax.set_xlabel("Opponents inside Leverkusen's defensive third at the goal kick (360 freeze frame)")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="x", visible=False)
    _title(fig, "Goal kicks: too few 360 frames to draw a conclusion yet",
           f"Only {len(gk)} goal kicks have a usable 360 frame (the camera often misses restarts). "
           "Short = under 32 yards.\nNext step: tag every goal kick from video to get a full sample.")
    fig.savefig(OUT / "04_goal_kicks_vs_press.png", dpi=200)
    plt.close(fig)


def fig_xt_grid(grid):
    pitch = Pitch(pitch_type="statsbomb", pitch_color=SURFACE, line_color=INK_2, linewidth=1, line_zorder=2)
    fig, axs = pitch.grid(figheight=8, title_height=0.12, endnote_height=0.03,
                          title_space=0.01, endnote_space=0.01, axis=False)
    ax = axs["pitch"]
    stats = pitch.bin_statistic([0], [0], statistic="count", bins=(L, W))
    stats["statistic"] = grid
    pitch.heatmap(stats, ax=ax, cmap="Blues", edgecolors=SURFACE, lw=1)
    for (r, c), v in np.ndenumerate(grid):
        if v >= 0.02:
            cx, cy = (c + 0.5) * 120 / L, (r + 0.5) * 80 / W
            ax.text(cx, cy, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color=SURFACE if v > 0.12 else INK)
    _title(fig, "Expected Threat (xT): how dangerous is having the ball in each zone?",
           "Probability that a possession in this zone becomes a goal, fitted on all 34 matches. "
           "Attacking left to right.\nA pass or carry is worth xT(end zone) minus xT(start zone). "
           "Values under 0.02 not labelled.")
    fig.savefig(OUT / "05_xt_model.png", dpi=200)
    plt.close(fig)


def fig_exit_routes(ex):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.8), sharey=False,
                                 gridspec_kw={"width_ratios": [5, 3], "wspace": 0.35})
    fig.subplots_adjust(top=0.78, bottom=0.12, left=0.14, right=0.97)
    for ax, col, order in [
        (a1, "channel", ["Left wing", "Left half-space", "Centre", "Right half-space", "Right wing"]),
        (a2, "method", ["Short pass", "Long pass", "Carry"]),
    ]:
        g = ex.groupby(col, observed=True).agg(n=("possession", "size"), xt=("possession_xt", "mean"),
                                               f3=("reached_final_third", "mean")).reindex(order)
        best = g.xt.idxmax()
        colors = [ORANGE if i == best else BLUE for i in g.index]
        bars = ax.barh(range(len(g)), g.xt * 1000, color=colors, height=0.62)
        ax.set_yticks(range(len(g)), g.index)
        ax.invert_yaxis()
        ax.bar_label(bars, labels=[f"{v * 1000:.0f}  (n={n})" for v, n in zip(g.xt, g.n)],
                     padding=4, fontsize=9, color=INK)
        ax.set_xlim(0, g.xt.max() * 1000 * 1.45)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("xT added per build-up (x1000)")
    a1.set_title("By channel", loc="left", fontsize=11, fontweight="bold")
    a2.set_title("By method", loc="left", fontsize=11, fontweight="bold")
    _title(fig, "Leverkusen are most dangerous when they exit through the half-spaces",
           f"Average xT Leverkusen added over the rest of the possession, by how the ball left their defensive "
           f"third ({len(ex)} build-ups).\nWide exits are the safest for an opponent: the least threat per build-up.")
    fig.savefig(OUT / "06_exit_routes_xt.png", dpi=200)
    plt.close(fig)


def fig_turnover_map(t):
    pitch = Pitch(pitch_type="statsbomb", pitch_color=SURFACE, line_color=MUTED, linewidth=1)
    fig, axs = pitch.grid(figheight=8, title_height=0.12, endnote_height=0.03,
                          title_space=0.01, endnote_space=0.01, axis=False)
    ax = axs["pitch"]
    ax.add_patch(plt.Rectangle((60, 0), 60, 80, color=SURFACE, alpha=0.75, zorder=2))
    calm, shot = t[~t.opp_shot], t[t.opp_shot]
    pitch.scatter(calm.x, calm.y, ax=ax, s=70, color=BLUE, edgecolors=SURFACE, lw=1.5,
                  zorder=3, label=f"No shot followed ({len(calm)})")
    pitch.scatter(shot.x, shot.y, ax=ax, s=70 + shot.next_xg * 900, color=ORANGE, edgecolors=SURFACE,
                  lw=1.5, zorder=4, label=f"Opponent shot followed ({len(shot)}), size = xG")
    ax.legend(loc="lower right", frameon=True, facecolor=SURFACE, edgecolor=MUTED, fontsize=9)
    ch = pd.cut(t.y, [-1, 18, 30, 50, 62, 81],
                labels=["Left wing", "Left half-space", "Centre", "Right half-space", "Right wing"])
    for (lab, n), yc in zip(ch.value_counts().reindex(ch.cat.categories).items(), [9, 24, 40, 56, 71]):
        ax.text(90, yc, f"{lab}: {n}", ha="center", va="center", fontsize=10, color=INK_2, zorder=5)
    top = t.player.value_counts().head(5)
    who = ", ".join(f"{short(n)} {c}" for n, c in top.items())
    _title(fig, "Where Leverkusen gave the ball away in build-up",
           f"{len(t)} open-play turnovers in their own half after build-ups from the defensive third "
           f"(opponent xG: {t.next_xg.sum():.1f}).\nMost often lost by: {who}. Attacking left to right.")
    fig.savefig(OUT / "07_buildup_turnover_map.png", dpi=200)
    plt.close(fig)


CASE_STYLE = {  # Bayern share a colour (filled = Bayern at home, hollow = Bayern away)
    "Bayern (H) 2-2": dict(color=BLUE, marker="o", fc=BLUE),
    "Bayern (A) 0-3": dict(color=BLUE, marker="o", fc=SURFACE),
    "Frankfurt (H) 1-5": dict(color=ORANGE, marker="D", fc=ORANGE),
    "Bochum (A) 0-4": dict(color=AQUA, marker="s", fc=AQUA),
}


def fig_case_comparison(t):
    metrics = [
        ("reach_final_third", "Leverkusen build-ups reaching final third", "{:.0%}", 0.1),
        ("regain_5s", "Their high pressures that won the ball within 5s", "{:.0%}", 0.05),
        ("kept_5s", "...and kept it (started their own possession within 5s)", "{:.0%}", 0.05),
        ("opp_xg_from_high_regains", "Their xG from possessions won in Leverkusen's half", "{:.2f}", 1),
        ("lev_open_play_xg", "Leverkusen open-play xG", "{:.2f}", 1),
        ("high_pressures", "Their pressures in Leverkusen's defensive third", "{:.0f}", 1),
    ]
    base = t[t.group == "Other 30 matches"]
    fig, axs = plt.subplots(len(metrics), 1, figsize=(10, 9.6))
    fig.subplots_adjust(top=0.83, bottom=0.07, left=0.05, right=0.95, hspace=1.25)
    for ax, (col, name, fmt, step) in zip(axs, metrics):
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.grid(axis="y", visible=False)
        ax.set_title(name, loc="left", fontsize=10, color=INK, pad=4)
        lo, hi = base[col].quantile([0.1, 0.9])
        ax.axvspan(lo, hi, color="#ecebe7", zorder=0)
        med = base[col].median()
        ax.plot([med, med], [-0.5, 0.5], color=INK_2, lw=1.5, zorder=1)
        vals = []
        for g, st in CASE_STYLE.items():
            v = t.loc[t.group == g, col].iat[0]
            vals.append(v)
            ax.scatter(v, 0, s=110, marker=st["marker"], facecolor=st["fc"], edgecolor=st["color"],
                       linewidth=2.2, zorder=3)
        allv = np.r_[vals, lo, hi]
        pad = (allv.max() - allv.min()) * 0.08
        ax.set_xlim(min(allv.min(), 0 if col != "high_pressures" else allv.min()) - pad, allv.max() + pad)
        ax.set_ylim(-0.6, 0.6)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _, f=fmt: f.format(v)))
        if step < 1:
            ax.xaxis.set_major_locator(plt.MultipleLocator(step))
        ax.tick_params(axis="x", labelsize=8.5)
    handles = [plt.Line2D([], [], ls="", marker=st["marker"], markersize=9, markerfacecolor=st["fc"],
                          markeredgecolor=st["color"], markeredgewidth=2, label=g) for g, st in CASE_STYLE.items()]
    handles += [plt.Line2D([], [], color=INK_2, lw=1.5, label="Other 30 matches: median"),
                plt.Rectangle((0, 0), 1, 1, color="#ecebe7", label="Other 30: middle 80%")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.04, 0.905), ncol=3, frameon=False, fontsize=9)
    _title(fig, "All four won the ball. Bayern and Frankfurt kept it; Bochum gave it straight back.",
           "Four teams that pressed Leverkusen highest. Labels are from the opponent's side (venue, their score first).")
    fig.savefig(OUT / "08_case_study_comparison.png", dpi=200)
    plt.close(fig)


def fig_press_maps(ev):
    pr = high_pressures(ev)
    pitch = Pitch(pitch_type="statsbomb", pitch_color=SURFACE, line_color=MUTED, linewidth=0.8)
    fig, axs = pitch.grid(nrows=2, ncols=2, figheight=9.5, title_height=0.11, endnote_height=0.03,
                          title_space=0.04, endnote_space=0.01, grid_height=0.79, space=0.14, axis=False)
    for ax, (mid, name) in zip(axs["pitch"].flat, CASES.items()):
        p = pr[pr.match_id == mid]
        # opponent coords -> Leverkusen's view (Leverkusen attack left to right)
        x, y = 120 - p.x, 80 - p.y
        won = p.regain_5s.to_numpy()
        kept = p.kept_5s.to_numpy()          # same definition as Figure 8
        lost_again = won & ~kept
        neither = ~won & ~kept
        pitch.scatter(x[neither], y[neither], ax=ax, s=28, color=MUTED, edgecolors=SURFACE, lw=0.8, zorder=3)
        pitch.scatter(x[lost_again], y[lost_again], ax=ax, s=55, color=SURFACE, edgecolors=ORANGE, lw=1.8, zorder=4)
        pitch.scatter(x[kept], y[kept], ax=ax, s=55, color=ORANGE, edgecolors=SURFACE, lw=1, zorder=5)
        ax.set_title(f"{name}:  {len(p)} pressures, won {won.mean():.0%}, kept {kept.mean():.0%}",
                     fontsize=10.5, loc="left", color=INK)
    _title(fig, "Where the four teams pressed Leverkusen, and when it worked",
           "Opponent pressures in Leverkusen's defensive third (Leverkusen attack left to right).\nWithin 5 seconds: "
           "orange = started their own possession (kept it), hollow = won the ball but lost it again, grey = neither.")
    fig.savefig(OUT / "09_case_study_press_maps.png", dpi=200)
    plt.close(fig)


def fig_goal_sources(ev):
    src = goal_sources(ev, list(CASES)).sort_values()
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(top=0.8, bottom=0.12, left=0.17, right=0.95)
    colors = [ORANGE if k in ("Open play", "Counter-attack") else BLUE for k in src.index]
    bars = ax.barh(src.index, src.values, color=colors, height=0.6)
    ax.bar_label(bars, padding=4, fontsize=10, color=INK)
    ax.set_xlabel("Leverkusen goals")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, src.max() + 1)
    op = src.reindex(["Open play", "Counter-attack"]).fillna(0).sum()
    pens = src.get("Penalty", 0)
    _title(fig, f"Against the high presses: {op:.0f} of Leverkusen's {src.sum()} goals from open play, {pens} penalties",
           "Both Bayern matches, Frankfurt and Bochum combined. Orange = open play or counter-attack.\n"
           "A restart only counts if the shot came within 10 seconds; longer moves after a restart count as open play.")
    fig.savefig(OUT / "10_case_study_goal_sources.png", dpi=200)
    plt.close(fig)


def fig_pass_network(ev):
    from adjustText import adjust_text
    others = [m for m in ev.match_id.unique() if m not in BAYERN]
    pitch = Pitch(pitch_type="statsbomb", pitch_color=SURFACE, line_color=MUTED, linewidth=0.8)
    fig, axs = plt.subplots(1, 2, figsize=(12, 7.6))
    fig.subplots_adjust(top=0.8, bottom=0.05, left=0.03, right=0.97, wspace=0.08)
    for ax, (name, ids) in zip(axs, [(f"Other {len(others)} matches", others),
                                     ("Both Bayern matches", BAYERN)]):
        pitch.draw(ax=ax)
        ax.set_xlim(-2, 63)
        nodes, edges = network(ev, ids)
        for _, e in edges.iterrows():
            a_, b_ = nodes.loc[e.a], nodes.loc[e.b]
            pitch.lines(a_.x, a_.y, b_.x, b_.y, ax=ax, lw=0.6 + e.per_match * 0.45, color=BLUE,
                        alpha=0.25 + min(e.share * 6, 0.55), zorder=2)
        texts = []
        for pl, n in nodes.iterrows():
            col = ORANGE if pl == XHAKA else BLUE
            pitch.scatter(n.x, n.y, ax=ax, s=80 + n.share * 2600, color=col, edgecolors=SURFACE,
                          lw=1.8, zorder=3)
            texts.append(ax.text(n.x + 2.5, n.y, f"{short(pl)} {n.share:.0%}", fontsize=8.5, color=INK,
                                 va="center", zorder=5,
                                 bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.85)))
        adjust_text(texts, ax=ax, only_move={"text": "xy"}, expand=(1.15, 1.3),
                    arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.5))
        ax.set_title(name, loc="left", fontsize=11, fontweight="bold", color=INK)
    _title(fig, "Against Bayern, the build-up dropped deeper and ran through Tapsoba",
           "Completed open-play passes starting in Leverkusen's own half, between the 11 players with the most minutes. "
           "Dot = average position;\n% = share of all these passes the player made or received.\nLine width = passes per match between the pair. Xhaka in orange. "
           "Own half shown, Leverkusen attack left to right.")
    fig.savefig(OUT / "11_buildup_pass_network.png", dpi=200)
    plt.close(fig)


def fig_xhaka_marking(ev):
    rec = xhaka_receipts(ev)
    g = rec.groupby("band", observed=True).agg(n=("id", "size"), prog=("progressive", "mean"),
                                                lost=("lost", "mean"))
    bm = xhaka_by_match(ev, rec)
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(12, 5.4), gridspec_kw={"width_ratios": [1, 1, 1.3],
                                                                          "wspace": 0.42})
    fig.subplots_adjust(top=0.76, bottom=0.17, left=0.06, right=0.97)
    labels = [f"{b}\n(n={n})" for b, n in zip(g.index, g.n)]
    cols = [ORANGE, MUTED, BLUE]
    for ax, col, title in [(a1, "prog", "Next action moves ball 10+ yards forward"),
                           (a2, "lost", "Next action loses the ball")]:
        bars = ax.bar(range(3), g[col] * 100, color=cols, width=0.62)
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9, color=INK)
        ax.set_xticks(range(3), labels, fontsize=8.5)
        ax.set_ylim(0, max(g[col]) * 100 * 1.25)
        ax.grid(axis="x", visible=False)
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
        ax.set_ylabel("% of Xhaka receptions")
    a1.set_xlabel("Nearest opponent when Xhaka receives", fontsize=9)
    a2.set_xlabel("Nearest opponent when Xhaka receives", fontsize=9)

    o = bm.drop(BAYERN, errors="ignore")
    a3.scatter(o.xhaka_space_m, o.reach_final_third * 100, s=55, color=MUTED, edgecolor=SURFACE, lw=1.5,
               zorder=3, label="Other matches")
    bb = bm.loc[[m for m in BAYERN if m in bm.index]]
    a3.scatter(bb.xhaka_space_m, bb.reach_final_third * 100, s=80, color=BLUE, edgecolor=SURFACE, lw=1.5,
               zorder=4, label="Bayern (both)")
    for m, r in bb.iterrows():
        a3.annotate(CASES[m].split(" ")[0] + " " + CASES[m].split(" ")[1], (r.xhaka_space_m, r.reach_final_third * 100),
                    xytext=(7, -3), textcoords="offset points", fontsize=8.5, color=INK_2)
    r = np.corrcoef(bm.xhaka_space_m, bm.reach_final_third)[0, 1]
    a3.set_xlabel("Xhaka's median space when receiving (m)")
    a3.set_ylabel("Build-ups reaching final third (%)")
    a3.set_title(f"Per match: weak link (r = {r:.2f})", loc="left", fontsize=10.5, fontweight="bold")
    a3.text(0.98, 0.98, "grey = other matches", transform=a3.transAxes, ha="right", va="top",
            fontsize=8.5, color=INK_2)
    _title(fig, "Keep someone within 6m of Xhaka as he receives; no team managed it for a whole match",
           f"Xhaka's receptions in Leverkusen's first two thirds with a 360 frame ({len(rec)}). "
           "Distance to nearest opponent at the moment he receives.\nBayern didn't mark him tighter than usual: "
           "their success came from elsewhere.")
    fig.savefig(OUT / "12_xhaka_marking.png", dpi=200)
    plt.close(fig)


def fig_pressed_vs_not(b):
    c = compare_pressed(b)
    n = c.possessions.astype(int)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.4), gridspec_kw={"width_ratios": [3, 1.3], "wspace": 0.35})
    fig.subplots_adjust(top=0.76, bottom=0.14, left=0.07, right=0.97)
    rows = [("Got out of own third", "broke_out"), ("Reached final third", "reached_final_third"),
            ("Lost it in own third", "lost_own_third")]
    idx = np.arange(len(rows))
    w = 0.36
    for off, grp, col in [(-w / 2 - 0.01, "Not pressed high", MUTED), (w / 2 + 0.01, "Pressed high", BLUE)]:
        bars = a1.bar(idx + off, [c.loc[grp, k] * 100 for _, k in rows], w, color=col,
                      label=f"{grp} (n={n[grp]})")
        a1.bar_label(bars, fmt="%.0f%%", padding=3, fontsize=9, color=INK)
    a1.set_xticks(idx, [r for r, _ in rows])
    a1.set_ylim(0, 108)
    a1.set_ylabel("% of Leverkusen build-ups")
    a1.grid(axis="x", visible=False)
    a1.legend(frameon=False, loc="upper right", fontsize=9)
    a1.set_title("Pressing slows them down...", loc="left", fontsize=11, fontweight="bold")
    vals = [c.loc["Not pressed high", "xg_per_100_once_out"], c.loc["Pressed high", "xg_per_100_once_out"]]
    bars = a2.bar([0, 1], vals, color=[MUTED, BLUE], width=0.6)
    a2.bar_label(bars, fmt="%.2f", padding=3, fontsize=10, color=INK)
    a2.set_xticks([0, 1], ["Not pressed\nhigh", "Pressed\nhigh"])
    a2.set_ylim(0, max(vals) * 1.3)
    a2.set_ylabel("Leverkusen xG per 100 build-ups")
    a2.grid(axis="x", visible=False)
    a2.set_title("...and if they escape,\nthe difference is unclear", loc="left", fontsize=11, fontweight="bold")
    _title(fig, "Pressing them high cuts their progression; the danger after an escape is unclear",
           "Leverkusen build-ups from their own third, 2023/24. Pressed high = at least one opponent pressure "
           "in Leverkusen's defensive third.\nRight: only build-ups that got out. The gap on the right is within "
           "the noise (90% interval: -0.4 to +1.7 xG per 100).")
    fig.savefig(OUT / "13_pressed_vs_not.png", dpi=200)
    plt.close(fig)


def fig_risk_reward(rr):
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.subplots_adjust(top=0.84, bottom=0.1, left=0.09, right=0.97)
    lim = max(rr.reward_xg.max(), rr.risk_xg.max()) * 1.1
    ax.fill_between([0, lim], [0, lim], [lim, lim], color="#f6e3da", zorder=0)
    ax.plot([0, lim], [0, lim], color=INK_2, lw=1, ls="--", zorder=1)
    ax.text(lim * 0.5, lim * 0.97, "Pressing cost more than it won", color=ORANGE, fontsize=9.5,
            fontweight="bold", va="top")
    ax.text(lim * 0.97, lim * 0.2, "Pressing won more\nthan it cost", color=BLUE, fontsize=9.5,
            fontweight="bold", ha="right")
    case = rr.case
    ax.scatter(rr.reward_xg[~case], rr.risk_xg[~case], s=30 + rr.pressed_buildups[~case] * 6, color=MUTED,
               edgecolor=SURFACE, lw=1.5, zorder=2)
    ax.scatter(rr.reward_xg[case], rr.risk_xg[case], s=30 + rr.pressed_buildups[case] * 6, color=BLUE,
               edgecolor=SURFACE, lw=1.5, zorder=3)
    show = case | (rr.net.abs() > 0.5)
    for _, r in rr[show].iterrows():
        bay_a = r.label.startswith("Bayern Munich (A)")
        ax.annotate(r.label, (r.reward_xg, r.risk_xg), textcoords="offset points",
                    xytext=(70, -45) if bay_a else (8, 3),
                    arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.7) if bay_a else None,
                    fontsize=8.5, color=INK_2)
    ax.set_xlim(-0.03, lim)
    ax.set_ylim(-0.03, lim)
    ax.set_xlabel("Reward: opponent xG straight after winning a pressed build-up back")
    ax.set_ylabel("Risk: Leverkusen xG after breaking the press")
    ax.set_aspect("equal")
    _title(fig, f"Over the season, pressing Leverkusen high gave up {rr.risk_xg.sum():.1f} xG and won {rr.reward_xg.sum():.1f}",
           "One dot per match (opponent's view: venue and score). Dot size = number of pressed build-ups. "
           "Blue = case-study matches.\nMost of the gap came when Leverkusen were already leading and opponents "
           "were chasing the game (see the game-state chart).")
    fig.savefig(OUT / "14_press_risk_reward.png", dpi=200)
    plt.close(fig)


def fig_lev_press(ev):
    pr = pressers(ev).head(8).iloc[::-1]
    cp = counterpress(losses(ev)).regain_5s.unstack(0) * 100
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 6.2), gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.42})
    fig.subplots_adjust(top=0.71, bottom=0.12, left=0.1, right=0.97)
    hi = pr.per90 * pr.high_share
    lo = pr.per90 - hi
    names = [short(n) for n in pr.index]
    a1.barh(names, hi, color=BLUE, height=0.62, label="In opponent's defensive third")
    a1.barh(names, lo, left=hi + 0.15, color=MUTED, height=0.62, label="Elsewhere")
    for i, v in enumerate(pr.per90):
        a1.text(v + 0.5, i, f"{v:.0f}", va="center", fontsize=9, color=INK)
    a1.set_xlabel("Pressures per 90 minutes")
    a1.grid(axis="y", visible=False)
    a1.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8.5)
    a1.set_title("Who leads their press", loc="left", fontsize=11, fontweight="bold", pad=24)
    idx = np.arange(len(cp))
    w = 0.36
    for off, col, colr in [(-w / 2 - 0.01, "Opponents", MUTED), (w / 2 + 0.01, "Leverkusen", BLUE)]:
        bars = a2.bar(idx + off, cp[col], w, color=colr, label=col if col == "Leverkusen" else "Their opponents")
        a2.bar_label(bars, fmt="%.0f%%", padding=3, fontsize=9, color=INK)
    a2.set_xticks(idx, [f"Lost in\n{z.lower()}" for z in cp.index])
    a2.set_ylabel("Ball won back within 5 seconds (%)")
    a2.set_ylim(0, cp.values.max() * 1.3)
    a2.grid(axis="x", visible=False)
    a2.legend(frameon=False, loc="upper left", fontsize=9)
    a2.set_title("Counter-press after losing the ball", loc="left", fontsize=11, fontweight="bold", pad=24)
    _title(fig, "Leverkusen press high through Adli, Wirtz and Hofmann, and counter-press hardest in the final third",
           "34% of Leverkusen's pressures come in the opponent's defensive third (their opponents: 19%). "
           "Players with 900+ minutes.\nWon back = their ball recovery, interception or tackle within 5s "
           "of a misplaced pass, miscontrol, dispossession or failed dribble.")
    fig.savefig(OUT / "15_leverkusen_press.png", dpi=200)
    plt.close(fig)


def fig_where_concede(ev, pos):
    sh = opponent_shots(ev, pos)
    org = sh.groupby("origin").shot_statsbomb_xg.sum().sort_values()
    wt = wide_transitions(ev, sh)
    rt = wt.groupby("route").agg(n=("xg", "size"), xg=("xg", "sum"), per=("xg", "mean"))
    order = ["Down Frimpong's side (their right)", "Down Grimaldo's side (their left)", "Both flanks",
             "Through the middle"]
    rt = rt.reindex(order)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={"wspace": 0.75})
    fig.subplots_adjust(top=0.78, bottom=0.12, left=0.19, right=0.95)
    cols = [ORANGE if o == "Set piece" else BLUE for o in org.index]
    bars = a1.barh(org.index, org.values, color=cols, height=0.6)
    a1.bar_label(bars, fmt="%.1f", padding=3, fontsize=9, color=INK)
    a1.set_xlabel("Opponent xG over the season (no penalties)")
    a1.set_xlim(0, org.max() * 1.2)
    a1.grid(axis="y", visible=False)
    a1.set_title("How chances against them started", loc="left", fontsize=11, fontweight="bold")
    labels = ["Down their right\n(Frimpong's side)", "Down their left\n(Grimaldo's side)", "Down both flanks",
              "Through the middle"]
    bars = a2.barh(labels[::-1], rt.xg.values[::-1], color=[MUTED, BLUE, BLUE, BLUE], height=0.6)
    a2.bar_label(bars, labels=[f"{x:.1f} xG  ({n} attacks)" for x, n in zip(rt.xg.values[::-1], rt.n.values[::-1])],
                 padding=3, fontsize=9, color=INK)
    a2.set_xlim(0, rt.xg.max() * 1.55)
    a2.set_xlabel("Opponent open-play xG over the season")
    a2.grid(axis="y", visible=False)
    a2.set_title("Open-play attacks: which way they came", loc="left", fontsize=11, fontweight="bold")
    _title(fig, "Set pieces and the flanks: where Leverkusen conceded",
           f"Opponent shots against Leverkusen, 2023/24 ({len(sh)} shots, {sh.shot_statsbomb_xg.sum():.1f} xG, "
           "penalties excluded). Set piece = shot within 10s of a corner, free kick or throw-in.\n"
           "Right: attacks with a completed pass or carry into a wide channel of Leverkusen's half.")
    fig.savefig(OUT / "16_where_leverkusen_concede.png", dpi=200)
    plt.close(fig)


def fig_opponent_exits(ev):
    ex, _ = opponent_exits(add_xt(ev, fit_xt(ev)), possessions(ev))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.8), gridspec_kw={"width_ratios": [5, 3], "wspace": 0.45})
    fig.subplots_adjust(top=0.78, bottom=0.12, left=0.2, right=0.97)
    # Opponent's left = Leverkusen's right
    names = {"Left wing": "Their right wing", "Left half-space": "Their right half-space", "Centre": "Centre",
             "Right half-space": "Their left half-space", "Right wing": "Their left wing"}
    for ax, col, order in [(a1, "channel", list(names)), (a2, "method", ["Short pass", "Long pass", "Carry"])]:
        g = exit_table(ex, col).reindex(order)
        best = g.xt_per_exit.idxmax()
        bars = ax.barh(range(len(g)), g.xt_per_exit * 1000, color=[ORANGE if i == best else BLUE for i in g.index],
                       height=0.62)
        ax.set_yticks(range(len(g)), [names.get(i, i) for i in g.index])
        ax.invert_yaxis()
        ax.bar_label(bars, labels=[f"{v * 1000:.0f}  (n={n})" for v, n in zip(g.xt_per_exit, g.n)],
                     padding=4, fontsize=9, color=INK)
        ax.set_xlim(0, g.xt_per_exit.max() * 1000 * 1.45)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("xT added per build-up (x1000)")
    a1.set_title("By channel (Leverkusen's view)", loc="left", fontsize=11, fontweight="bold")
    a2.set_title("By method", loc="left", fontsize=11, fontweight="bold")
    _title(fig, "Beating their press: carrying the ball out was worth the most",
           f"Opponent build-ups from their own third that got out ({len(ex)}): average xT added over the rest of "
           "the possession.\nChannels named from Leverkusen's side ('their right' = Frimpong's flank). "
           "Channel differences are small; treat them as a lean, not a rule.")
    fig.savefig(OUT / "17_beating_leverkusen_press.png", dpi=200)
    plt.close(fig)


def fig_risk_by_state(ev, pos):
    t = risk_by_state(ev, pos).loc[["Level", "Leverkusen leading", "Leverkusen trailing"]]
    n_pressed = (t.buildups * t.pressed_share).round().astype(int)
    labels = ["Level", "Leverkusen\nleading", "Leverkusen\ntrailing"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.8), gridspec_kw={"wspace": 0.3})
    fig.subplots_adjust(top=0.74, bottom=0.16, left=0.07, right=0.97)
    idx = np.arange(len(t))
    w = 0.36
    b1 = a1.bar(idx - w / 2 - 0.01, t.reward_per100, w, color=BLUE, label="Won: opponent xG after a regain")
    b2 = a1.bar(idx + w / 2 + 0.01, t.risk_per100, w, color=ORANGE, label="Gave up: Leverkusen xG after escaping")
    for b in (b1, b2):
        a1.bar_label(b, fmt="%.1f", padding=3, fontsize=9, color=INK)
    a1.set_xticks(idx, [f"{l}\n(n={n})" for l, n in zip(labels, n_pressed)])
    a1.set_ylabel("xG per 100 pressed build-ups")
    a1.set_ylim(0, t[["reward_per100", "risk_per100"]].values.max() * 1.3)
    a1.grid(axis="x", visible=False)
    a1.legend(frameon=False, loc="upper left", fontsize=9)
    a1.set_title("What the press won and gave up", loc="left", fontsize=11, fontweight="bold")
    b3 = a2.bar(idx - w / 2 - 0.01, t.ft_not * 100, w, color=MUTED, label="Not pressed high")
    b4 = a2.bar(idx + w / 2 + 0.01, t.ft_pressed * 100, w, color=BLUE, label="Pressed high")
    for b in (b3, b4):
        a2.bar_label(b, fmt="%.0f%%", padding=3, fontsize=9, color=INK)
    a2.set_xticks(idx, labels)
    a2.set_ylabel("Leverkusen build-ups reaching final third (%)")
    a2.set_ylim(0, 100)
    a2.grid(axis="x", visible=False)
    a2.legend(frameon=False, loc="upper right", fontsize=9)
    a2.set_title("How much it slowed them down", loc="left", fontsize=11, fontweight="bold")
    _title(fig, "At level scores, pressing Leverkusen high broke even and cut their progression",
           "Pressed Leverkusen build-ups split by the score when the build-up started. The cost of pressing came "
           "when Leverkusen were already ahead\nand opponents were chasing the game. All five red cards in these "
           "matches were opponents'; removing spells with a man down changes nothing at level scores.")
    fig.savefig(OUT / "18_press_risk_by_game_state.png", dpi=200)
    plt.close(fig)


def fig_goals_by_time(ev):
    t = goals_by_time(ev)
    lg = late_goal_points(ev)
    fig, ax = plt.subplots(figsize=(11, 5.4))
    fig.subplots_adjust(top=0.8, bottom=0.13, left=0.07, right=0.97)
    idx = np.arange(len(t))
    w = 0.38
    b1 = ax.bar(idx - w / 2 - 0.01, t.scored, w, color=BLUE, label=f"Scored ({t.scored.sum()})")
    b2 = ax.bar(idx + w / 2 + 0.01, t.conceded, w, color=MUTED, label=f"Conceded ({t.conceded.sum()})")
    for b in (b1, b2):
        ax.bar_label(b, padding=3, fontsize=9, color=INK)
    ax.set_xticks(idx, [f"{b}'" for b in t.index])
    ax.set_xlabel("Minute of the match (stoppage time in its own bar)")
    ax.set_ylabel("Goals")
    ax.set_ylim(0, t.values.max() * 1.25)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.axvspan(6.5, 7.5, color="#ecebe7", zorder=0)
    _title(fig, f"Leverkusen don't stop: {lg.late_scored.sum()} goals after the 80th minute won them {lg.points_gained.sum()} points",
           f"Goals by 15-minute period, 2023/24 (own goals included). From the 80th minute: {lg.late_scored.sum()} scored, "
           f"{lg.late_conceded.sum()} conceded. Second-half stoppage time: {t.loc['90+', 'scored']}-{t.loc['90+', 'conceded']}.\n"
           "Points won = final points minus points with their goals from the 80th minute removed.")
    fig.savefig(OUT / "19_goals_by_time.png", dpi=200)
    plt.close(fig)


def fig_late_goals(ev):
    prof = phase_profile(ev)
    op = open_play_rates(ev)
    change = {
        "Leverkusen open-play xG per 90": op.loc["80+", "xg_per90"] / op.loc["1-79", "xg_per90"] - 1,
        "Leverkusen xG per open-play shot": op.loc["80+", "xg_per_shot"] / op.loc["1-79", "xg_per_shot"] - 1,
        "Leverkusen shots per 90": prof.loc["lev_shots_per90", "change"],
        "Opponent pressures per 90": prof.loc["opp_pressures_per90", "change"],
        "How far up opponents defended": prof.loc["opp_def_action_height", "change"],
        "Opponent xG per 90": prof.loc["opp_xg_per90", "change"],
    }
    ch = pd.Series(change)[::-1] * 100
    gt = goal_types(late_goal_detail(ev)).reindex(columns=["Counter-attack", "Set piece or penalty",
                                                          "Other open play"], fill_value=0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.8), gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.75})
    fig.subplots_adjust(top=0.74, bottom=0.13, left=0.24, right=0.97)
    cols = [BLUE if v > 0 else MUTED for v in ch.values]
    bars = a1.barh(ch.index, ch.values, color=cols, height=0.6)
    a1.bar_label(bars, labels=[f"{v:+.0f}%" for v in ch.values], padding=4, fontsize=9, color=INK)
    a1.axvline(0, color=INK_2, lw=1)
    lim = max(abs(ch.values)) * 1.4
    a1.set_xlim(-lim, lim)
    a1.set_xlabel("Change from the 80th minute vs minutes 1-79")
    a1.grid(axis="y", visible=False)
    a1.set_title("What changed late in matches", loc="left", fontsize=11, fontweight="bold")
    bottom = np.zeros(len(gt))
    colors = {"Counter-attack": BLUE, "Set piece or penalty": ORANGE, "Other open play": MUTED}
    for col in gt.columns:
        a2.bar(gt.index, gt[col], bottom=bottom, color=colors[col], label=col, width=0.55,
               edgecolor=SURFACE, linewidth=2)
        for i, (v, b) in enumerate(zip(gt[col], bottom)):
            if v:
                a2.text(i, b + v / 2, str(v), ha="center", va="center", fontsize=10,
                        color=SURFACE if col != "Other open play" else INK)
        bottom += gt[col].values
    a2.set_ylabel("Leverkusen goals from the 80th minute")
    a2.grid(axis="x", visible=False)
    a2.legend(frameon=False, fontsize=8.5, loc="upper right")
    a2.set_ylim(0, bottom.max() * 1.45)
    a2.set_title("The 18 late goals", loc="left", fontsize=11, fontweight="bold")
    _title(fig, "Late goals came from games opening up, not luck: their chances got better as opponents chased",
           "2023/24, all 34 matches. Leverkusen were already ahead for 76% of the minutes from the 80th. "
           "Late goals (18) were close to late xG (14.5),\nthe same edge as the rest of their season. "
           "Set pieces came at the same rate late as early. Open play excludes shots within 10s of a restart.")
    fig.savefig(OUT / "20_late_goals_explained.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    ev = prepare_events()
    pos = possessions(ev)
    b = buildup_possessions(pos)
    fig_exit_channels(ev, b)
    fig_press_vs_progression(opponent_press_profile(ev, pos))
    fig_player_press(player_press_resistance(ev))
    fig_goal_kicks(goal_kick_press(ev, pos))
    grid = fit_xt(ev)
    fig_xt_grid(grid)
    fig_exit_routes(exit_routes(add_xt(ev, grid), b))
    fig_turnover_map(buildup_turnovers(ev, pos, b))
    fig_case_comparison(match_table(ev, pos))
    fig_press_maps(ev)
    fig_goal_sources(ev)
    fig_pass_network(ev)
    fig_xhaka_marking(ev)
    pb = pressed_buildups(ev, pos)
    fig_pressed_vs_not(pb)
    fig_risk_reward(risk_reward(ev, pos, pb))
    fig_lev_press(ev)
    fig_where_concede(ev, pos)
    fig_opponent_exits(ev)
    fig_risk_by_state(ev, pos)
    fig_goals_by_time(ev)
    fig_late_goals(ev)
    print("Saved:", *sorted(p.name for p in OUT.glob("*.png")), sep="\n  ")
