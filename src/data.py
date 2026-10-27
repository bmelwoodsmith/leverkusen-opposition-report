"""Load and cache StatsBomb open data for Leverkusen's 2023/24 Bundesliga season.

Data: StatsBomb Open Data (https://github.com/statsbomb/open-data).
First run downloads from GitHub; later runs read the local cache in data/.
"""
from pathlib import Path
import warnings

import pandas as pd
from statsbombpy import sb

warnings.filterwarnings("ignore", message=".*credentials.*")

COMPETITION_ID = 9   # 1. Bundesliga
SEASON_ID = 281      # 2023/2024
TEAM = "Bayer Leverkusen"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def load_matches() -> pd.DataFrame:
    path = DATA_DIR / "matches.pkl"
    if path.exists():
        return pd.read_pickle(path)
    matches = sb.matches(competition_id=COMPETITION_ID, season_id=SEASON_ID)
    matches["opponent"] = matches.apply(
        lambda r: r.away_team if r.home_team == TEAM else r.home_team, axis=1
    )
    matches["venue"] = matches.home_team.eq(TEAM).map({True: "H", False: "A"})
    matches = matches.sort_values("match_date").reset_index(drop=True)
    matches.to_pickle(path)
    return matches


def load_events() -> pd.DataFrame:
    """All events from all 34 matches, one row per event."""
    path = DATA_DIR / "events.pkl"
    if path.exists():
        return pd.read_pickle(path)
    matches = load_matches()
    frames = []
    for i, m in matches.iterrows():
        print(f"[{i + 1}/{len(matches)}] events {m.home_team} v {m.away_team}")
        ev = sb.events(match_id=m.match_id)
        ev["match_id"] = m.match_id
        frames.append(ev)
    events = pd.concat(frames, ignore_index=True)
    events.to_pickle(path)
    return events


def load_frames() -> pd.DataFrame:
    """360 freeze frames: one row per visible player per event."""
    path = DATA_DIR / "frames.pkl"
    if path.exists():
        return pd.read_pickle(path)
    matches = load_matches()
    out = []
    for i, m in matches.iterrows():
        print(f"[{i + 1}/{len(matches)}] 360 {m.home_team} v {m.away_team}")
        try:
            fr = sb.frames(match_id=m.match_id, fmt="dataframe")
        except Exception as e:  # a few matches may lack 360 data
            print(f"   no 360 data: {e}")
            continue
        fr["match_id"] = m.match_id
        out.append(fr)
    frames = pd.concat(out, ignore_index=True)
    frames.to_pickle(path)
    return frames


if __name__ == "__main__":
    m = load_matches()
    print(m[["match_date", "home_team", "home_score", "away_score", "away_team"]].to_string())
    e = load_events()
    print(f"\n{len(e):,} events, {e.match_id.nunique()} matches")
    f = load_frames()
    print(f"{len(f):,} freeze-frame rows, {f.match_id.nunique()} matches")
