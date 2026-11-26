"""Data loading, cleaning and feature engineering for PUBG win-place prediction."""
import numpy as np
import pandas as pd

from config import (DATA_DIR, DROP_COLS, MATCH_TYPE_MAP, RANDOM_STATE,
                    RAW_FILE_CANDIDATES, SAMPLE_FILE, TARGET)


# ----------------------------------------------------------------------------- loading
def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns - the full file is ~1 GB in float64."""
    for col in df.select_dtypes(include="number").columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        else:
            df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def find_data_file():
    for name in RAW_FILE_CANDIDATES:
        if (DATA_DIR / name).exists():
            return DATA_DIR / name, False
    if (DATA_DIR / SAMPLE_FILE).exists():
        return DATA_DIR / SAMPLE_FILE, True
    raise FileNotFoundError(
        f"No dataset found in {DATA_DIR}. Put train_V2.csv there, or run "
        "`python src/generate_sample_data.py` to create a synthetic test file.")


def load_data(sample_matches=None, path=None):
    path, synthetic = (path, False) if path else find_data_file()
    df = reduce_memory(pd.read_csv(path))
    if sample_matches and df["matchId"].nunique() > sample_matches:
        keep = pd.Series(df["matchId"].unique()).sample(sample_matches, random_state=RANDOM_STATE)
        df = df[df["matchId"].isin(keep)].reset_index(drop=True)
    return df, synthetic, path


# ----------------------------------------------------------------------------- cleaning
def flag_anomalies(df: pd.DataFrame) -> pd.Series:
    """Rows that are physically implausible (likely cheaters / bugged records)."""
    total_dist = df["walkDistance"] + df["rideDistance"] + df["swimDistance"]
    hs_rate = df["headshotKills"] / df["kills"].replace(0, np.nan)
    return (
        ((df["kills"] > 0) & (total_dist == 0))          # kills without moving
        | (df["roadKills"] > 10)
        | (df["kills"] > 40)
        | ((hs_rate == 1) & (df["kills"] > 9))           # 100% headshots on 10+ kills
        | (df["longestKill"] >= 1000)
        | (df["walkDistance"] >= 10000)
        | (df["rideDistance"] >= 20000)
        | (df["swimDistance"] >= 2000)
        | (df["weaponsAcquired"] >= 80)
        | (df["heals"] >= 40)
    )


def clean(df: pd.DataFrame, drop_anomalies=True):
    report = {"rows_in": len(df)}
    if TARGET in df:
        report["missing_target"] = int(df[TARGET].isna().sum())
        df = df[df[TARGET].notna()]
    df = df.drop_duplicates(subset="Id")
    if drop_anomalies:
        mask = flag_anomalies(df)
        report["anomalies_removed"] = int(mask.sum())
        df = df[~mask]
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    report["rows_out"] = len(df)
    return df.reset_index(drop=True), report


# ----------------------------------------------------------------------------- features
def add_player_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["playersJoined"] = df.groupby("matchId")["matchId"].transform("count")
    norm = (100 - df["playersJoined"]) / 100 + 1
    df["killsNorm"] = df["kills"] * norm
    df["damageNorm"] = df["damageDealt"] * norm
    df["totalDistance"] = df["walkDistance"] + df["rideDistance"] + df["swimDistance"]
    df["healsAndBoosts"] = df["heals"] + df["boosts"]
    df["headshotRate"] = (df["headshotKills"] / df["kills"]).replace([np.inf, -np.inf], 0).fillna(0)
    df["killStreakRate"] = (df["killStreaks"] / df["kills"]).replace([np.inf, -np.inf], 0).fillna(0)
    df["walkPerKill"] = df["walkDistance"] / (df["kills"] + 1)
    df["boostsPerWalk"] = df["boosts"] / (df["walkDistance"] + 1)
    df["healsPerWalk"] = df["heals"] / (df["walkDistance"] + 1)
    df["weaponsPerWalk"] = df["weaponsAcquired"] / (df["walkDistance"] + 1)
    df["killPlacePerc"] = df["killPlace"] / df.groupby("matchId")["killPlace"].transform("max")
    df["distancePerSec"] = df["totalDistance"] / df["matchDuration"].clip(lower=1)
    df["matchMode"] = df["matchType"].map(MATCH_TYPE_MAP).fillna("event")
    return df


NUMERIC_BASE = [
    "assists", "boosts", "damageDealt", "DBNOs", "headshotKills", "heals", "killPlace",
    "kills", "killStreaks", "longestKill", "revives", "rideDistance", "roadKills",
    "swimDistance", "teamKills", "vehicleDestroys", "walkDistance", "weaponsAcquired",
    "killsNorm", "damageNorm", "totalDistance", "healsAndBoosts", "headshotRate",
    "killStreakRate", "walkPerKill", "boostsPerWalk", "healsPerWalk", "weaponsPerWalk",
    "killPlacePerc", "distancePerSec",
]
MATCH_LEVEL = ["matchDuration", "maxPlace", "numGroups", "playersJoined"]


def build_group_features(df: pd.DataFrame):
    """
    All players in a group share the same winPlacePerc, so we model at group level:
    mean / max / min of every player stat + percentile rank of the group's mean and max
    inside its match (relative strength is what decides placement).
    """
    df = add_player_features(df)
    g = df.groupby(["matchId", "groupId"])
    agg_mean = g[NUMERIC_BASE].mean()
    agg_max = g[NUMERIC_BASE].max().add_suffix("_max")
    agg_min = g[NUMERIC_BASE].min().add_suffix("_min")
    size = g.size().rename("groupSize")
    out = pd.concat([agg_mean.add_suffix("_mean"), agg_max, agg_min, size], axis=1)

    match_info = g[MATCH_LEVEL].first()
    out = out.join(match_info)
    out["matchMode"] = g["matchMode"].first()

    rank_cols = [c for c in out.columns if c.endswith("_mean") or c.endswith("_max")]
    ranks = out.groupby(level="matchId")[rank_cols].rank(pct=True).add_suffix("_rank")
    out = out.join(ranks)

    for m in ["solo", "duo", "squad", "event"]:
        out[f"mode_{m}"] = (out["matchMode"] == m).astype(np.int8)
    out = out.drop(columns="matchMode")

    target = g[TARGET].first() if TARGET in df.columns else None
    out = reduce_memory(out.astype(np.float32))
    return out, target, df


def postprocess(pred: np.ndarray, groups: pd.DataFrame) -> np.ndarray:
    """
    Snap predictions to valid placements: rank groups inside each match by predicted
    score, then map rank -> (maxPlace - place) / (maxPlace - 1). Keeps outputs in [0, 1]
    and consistent with the game's discrete placement scale.
    """
    tmp = groups[["maxPlace", "numGroups"]].copy()
    tmp["pred"] = pred
    tmp["rank"] = tmp.groupby(level="matchId")["pred"].rank(ascending=True, method="first")
    n = tmp.groupby(level="matchId")["pred"].transform("count")
    tmp["adj"] = (tmp["rank"] - 1) / (n - 1).replace(0, 1)
    gap = 1 / (tmp["maxPlace"] - 1).replace(0, 1)
    tmp["adj"] = (np.round(tmp["adj"] / gap) * gap)
    tmp.loc[tmp["maxPlace"] <= 1, "adj"] = 0
    return tmp["adj"].clip(0, 1).values
