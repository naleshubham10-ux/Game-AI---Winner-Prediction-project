# Data folder

1. Download the dataset from the link in the project brief:
   `https://d3ilbtxij3aepc.cloudfront.net/projects/CDS-Capstone-Projects/PRCP-1012-GameWinnerPred.zip`
2. Extract it and place **`train_V2.csv`** in this folder (the code also accepts `pubg.csv`).

```
data/
└── train_V2.csv      <- ~4.45 M rows, 29 columns
```

No dataset yet? Create a synthetic test file with the same schema so you can check the code runs:

```
python src/generate_sample_data.py --matches 1500
```

This writes `data/sample_pubg.csv`. The code always prefers the real file when it is present.
Results on the synthetic file are for testing only — do not submit them.
