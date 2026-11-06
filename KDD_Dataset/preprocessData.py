import sys

import pandas as pd
from sklearn.preprocessing import LabelEncoder

INPUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "kddcup_full_dataset.csv"
OUTPUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "KDD_processed_dataset.csv"

df = pd.read_csv(INPUT_PATH)

# strip b'...' byte-string wrapper fetch_kddcup99 leaves on text columns
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].str.replace(r"^b'(.*)'$", r"\1", regex=True)

# label-encode categorical columns: protocol_type, service, flag
for col in ["1", "2", "3"]:
    df[col] = LabelEncoder().fit_transform(df[col])

# binarize label: normal. -> 0, anything else (attack) -> 1
df["label"] = (df["label"] != "normal.").astype(int)

df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved {OUTPUT_PATH}, shape={df.shape}")
