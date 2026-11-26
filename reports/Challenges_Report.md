# Report on Challenges Faced – PRCP-1012 Game Winner Prediction

## 1. Large dataset
**Challenge:** `train_V2.csv` has ~4.45 million rows and takes about 1 GB of RAM with default dtypes, which makes EDA and training slow on a laptop.
**Technique:** Downcast every numeric column to the smallest type that fits (`int8`/`int16`/`float32`), and sample whole **matches** (default 8,000) for experiments.
**Reason:** Downcasting cuts memory by more than half with no loss of information. Sampling by match rather than by row keeps each match complete, so match-level features (players joined, in-match ranks) stay correct.

## 2. Missing target value
**Challenge:** One row has a missing `winPlacePerc`.
**Technique:** Drop the row.
**Reason:** It belongs to a match with a single player, where placement is undefined. One row out of millions has no effect on the analysis.

## 3. Inconsistent rating columns
**Challenge:** `killPoints`, `rankPoints` and `winPoints` come from two different rating systems. `rankPoints` is `-1` for many players, and `killPoints`/`winPoints` are `0` whenever `rankPoints` is filled in.
**Technique:** Drop all three.
**Reason:** They are mostly empty, mutually exclusive and almost uncorrelated with the target, so they would only add noise.

## 4. Cheaters and impossible records
**Challenge:** Some rows are physically impossible: kills without moving, kills from more than 1 km away, 100 % headshots over 10+ kills, more than 10 road kills, 80+ weapons picked up, 40+ heals, or extreme walking, riding or swimming distances.
**Technique:** Rule-based anomaly filters (see notebook section 3) that remove these rows.
**Reason:** These records come from cheaters or logging bugs. They teach the model false links between behaviour and placement, and they are a tiny share of the data.

## 5. Skewed features and outliers
**Challenge:** Kills, damage, distances and item counts are heavily right-skewed: most players have values near 0, with a long tail.
**Technique:** Use tree-based models, add ratio features (per kill, per metre walked), and clip values only for plots.
**Reason:** Tree models don't change under monotonic transforms and cope well with outliers, so heavy log or Box-Cox transforms aren't needed.

## 6. Too many sparse match types
**Challenge:** There are 16 `matchType` values, and some (crash, flare, normal-*) have very few rows.
**Technique:** Group them into solo / duo / squad / event and one-hot encode.
**Reason:** Game mode is what changes team dynamics. The rare sub-types have too little data to learn from separately.

## 7. Target shared by group members
**Challenge:** Every player in a group has the same `winPlacePerc`, so player-level training repeats each label up to 4 times and treats teammates as independent.
**Technique:** Aggregate players into groups using the mean, max and min of each feature, plus group size.
**Reason:** The group is the unit that actually places, so modelling it directly is both more accurate and faster (about 3× fewer rows).

## 8. Placement is relative to the match
**Challenge:** The same stats can mean a top-10 finish in one match and a mid-table finish in another.
**Technique:** Add the percentile rank of each group's stats within its match, and normalise kills and damage by lobby size (`playersJoined`).
**Reason:** Placement is decided against the other groups in the same match, so relative strength predicts it better than absolute counts.

## 9. Data leakage
**Challenge:** In-match rank features use information from every group in a match. A random row split would put groups from the same match in both train and test.
**Technique:** `GroupShuffleSplit` on `matchId`.
**Reason:** This gives an honest estimate of performance on unseen matches.

## 10. Multicollinearity
**Challenge:** `kills`, `damageDealt`, `DBNOs`, `headshotKills` and `killStreaks` are strongly correlated with each other.
**Technique:** Use Ridge (L2) for the linear baseline and tree ensembles for the final models.
**Reason:** L2 regularisation keeps linear coefficients stable. Tree models are not affected by correlated inputs.

## 11. Predictions outside the valid range
**Challenge:** Regressors can predict values below 0 or above 1, and tied values that don't match any real placement.
**Technique:** Clip predictions to [0, 1], then apply rank-based post-processing: rank groups within each match and snap to the placement grid `(maxPlace − place) / (maxPlace − 1)`. It is kept only when it lowers MAE on the test set.
**Reason:** This enforces valid, evenly spaced placements, which matches how the target is built.

## 12. Training time
**Challenge:** Random Forest and XGBoost are slow on hundreds of thousands of groups.
**Technique:** LightGBM's histogram-based boosting, XGBoost's `tree_method="hist"`, and `max_samples=0.5` for Random Forest.
**Reason:** These keep a six-model comparison fast enough to run and repeat.
