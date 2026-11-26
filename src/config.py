"""Central configuration for PRCP-1012 Game Winner Prediction."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

# Real Kaggle/PUBG file name shipped inside PRCP-1012-GameWinnerPred.zip
RAW_FILE_CANDIDATES = ["train_V2.csv", "pubg.csv", "PUBG_Game_Prediction_data.csv"]
SAMPLE_FILE = "sample_pubg.csv"

TARGET = "winPlacePerc"
ID_COLS = ["Id", "groupId", "matchId"]

# Number of matches to sample for training (None = use all; full data ~47k matches / 4.4M rows)
SAMPLE_MATCHES = 8000
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Columns dropped after analysis (see reports/Challenges_Report.md)
# Note: the brief lists "maxPlace" twice - the 18th attribute in the real file is numGroups.
DROP_COLS = ["killPoints", "rankPoints", "winPoints"]

MATCH_TYPE_MAP = {
    "solo": "solo", "solo-fpp": "solo", "normal-solo": "solo", "normal-solo-fpp": "solo",
    "duo": "duo", "duo-fpp": "duo", "normal-duo": "duo", "normal-duo-fpp": "duo",
    "squad": "squad", "squad-fpp": "squad", "normal-squad": "squad", "normal-squad-fpp": "squad",
    "crashfpp": "event", "crashtpp": "event", "flarefpp": "event", "flaretpp": "event",
}
