"""
Generate a synthetic PUBG-style dataset with the exact PRCP-1012 schema.

Use this ONLY to test the pipeline when the real dataset (train_V2.csv) is not
available. Results on synthetic data are NOT valid project results.

    python src/generate_sample_data.py --matches 1500
"""
import argparse
import uuid

import numpy as np
import pandas as pd

from config import DATA_DIR, SAMPLE_FILE

MODES = {  # matchType: (group size, weight)
    "squad-fpp": (4, 0.39), "duo-fpp": (2, 0.22), "squad": (4, 0.14), "solo-fpp": (1, 0.12),
    "duo": (2, 0.07), "solo": (1, 0.04), "normal-squad-fpp": (4, 0.01), "crashfpp": (4, 0.005),
    "flaretpp": (4, 0.003), "normal-duo-fpp": (2, 0.002),
}


def _hex():
    return uuid.uuid4().hex[:14]


def make_match(rng):
    modes, meta = list(MODES), list(MODES.values())
    probs = np.array([m[1] for m in meta]); probs /= probs.sum()
    mtype = modes[rng.choice(len(modes), p=probs)]
    gsize = MODES[mtype][0]
    n_players = int(rng.integers(85, 100))
    n_groups = max(2, int(np.ceil(n_players / gsize)))
    max_place = n_groups + int(rng.integers(0, 3))
    duration = int(rng.normal(1580, 220) if "crash" not in mtype else rng.normal(900, 80))
    match_id = _hex()

    group_skill = rng.normal(0, 1, n_groups)
    survival = group_skill + rng.normal(0, 0.9, n_groups)  # higher = survives longer
    order = np.argsort(-survival)
    place = np.empty(n_groups, int); place[order] = np.arange(1, n_groups + 1)
    wpp = np.where(max_place > 1, (max_place - place) / (max_place - 1), 0)
    wpp = np.clip(wpp - (max_place - n_groups) / (max_place - 1) * (place > n_groups - 1), 0, 1)

    rows = []
    for g in range(n_groups):
        gid = _hex()
        members = gsize if g < n_groups - 1 else max(1, n_players - gsize * (n_groups - 1))
        alive = 1 - place[g] / n_groups  # 0..1 share of match survived
        for _ in range(members):
            s = group_skill[g] + rng.normal(0, 0.6)
            lam_k = np.exp(0.55 * s + 1.2 * alive - 1.2)
            kills = int(min(rng.poisson(lam_k), 40))
            hs = int(rng.binomial(kills, 0.22))
            dmg = max(0.0, kills * rng.normal(100, 20) + rng.gamma(1.5, 30) * (rng.random() < 0.7))
            walk = max(0.0, rng.gamma(2.0, 400 + 1400 * alive))
            ride = max(0.0, rng.gamma(0.6, 2500) * (rng.random() < 0.3 + 0.4 * alive))
            swim = max(0.0, rng.gamma(0.4, 20) * (rng.random() < 0.15))
            boosts = int(rng.poisson(0.3 + 5 * alive))
            heals = int(rng.poisson(0.4 + 4 * alive))
            rank_pts = int(rng.normal(1500, 150)) if rng.random() < 0.6 else -1
            kill_pts = 0 if rank_pts > 0 else int(rng.normal(1100, 200))
            rows.append({
                "Id": _hex(), "groupId": gid, "matchId": match_id,
                "assists": int(rng.poisson(0.15 + 0.4 * alive)) if gsize > 1 else 0,
                "boosts": boosts, "damageDealt": round(dmg, 2),
                "DBNOs": int(rng.poisson(kills * 0.8)) if gsize > 1 else 0,
                "headshotKills": hs, "heals": heals,
                "killPlace": 0,  # filled below
                "killPoints": kill_pts, "kills": kills,
                "killStreaks": int(min(kills, rng.poisson(0.6 * kills) + (kills > 0))),
                "longestKill": round(rng.gamma(1.5, 40) if kills else 0.0, 2),
                "matchDuration": duration, "matchType": mtype, "maxPlace": max_place,
                "numGroups": n_groups, "rankPoints": rank_pts,
                "revives": int(rng.poisson(0.2 + 0.4 * alive)) if gsize > 1 else 0,
                "rideDistance": round(ride, 2), "roadKills": int(rng.random() < 0.003),
                "swimDistance": round(swim, 2), "teamKills": int(rng.random() < 0.02),
                "vehicleDestroys": int(rng.random() < 0.007),
                "walkDistance": round(walk, 2),
                "weaponsAcquired": int(rng.poisson(2 + 4 * alive)),
                "winPoints": 0 if rank_pts > 0 else int(rng.normal(1500, 60)),
                "winPlacePerc": round(float(wpp[g]), 4),
            })
    df = pd.DataFrame(rows)
    df["killPlace"] = (-(df["kills"] * 1e6 + df["damageDealt"])).rank(method="first").astype(int)
    return df


def generate(n_matches=1500, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.concat([make_match(rng) for _ in range(n_matches)], ignore_index=True)
    # Inject realistic data issues found in the real dataset
    cheat = df.sample(frac=0.0015, random_state=seed).index
    df.loc[cheat, "walkDistance"] = 0; df.loc[cheat, "rideDistance"] = 0
    df.loc[cheat, "swimDistance"] = 0; df.loc[cheat, "kills"] = rng.integers(3, 20, len(cheat))
    df.loc[df.sample(n=1, random_state=1).index, "winPlacePerc"] = np.nan
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=1500)
    args = ap.parse_args()
    DATA_DIR.mkdir(exist_ok=True)
    out = DATA_DIR / SAMPLE_FILE
    data = generate(args.matches)
    data.to_csv(out, index=False)
    print(f"Synthetic sample written: {out}  shape={data.shape}")
