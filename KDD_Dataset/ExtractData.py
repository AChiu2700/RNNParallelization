from sklearn.datasets import fetch_kddcup99
import pandas as pd
import os

# ---------------- FULL DATASET ----------------
print("Downloading FULL dataset (~4.9M rows)...")
full = fetch_kddcup99(percent10=False, as_frame=False)
print("FULL download complete.")

df_full = pd.DataFrame(full.data)
df_full["label"] = full.target

full_path = os.path.abspath("kddcup_full_dataset.csv")
print("Saving FULL dataset...")
df_full.to_csv(full_path, index=False)
print("FULL saved to:", full_path)
print("FULL shape:", df_full.shape)

print("\n-----------------------------------\n")

# ---------------- 10% DATASET ----------------
print("Downloading 10% dataset (~494k rows)...")
ten = fetch_kddcup99(percent10=True, as_frame=False)
print("10% download complete.")

df_10 = pd.DataFrame(ten.data)
df_10["label"] = ten.target

ten_path = os.path.abspath("kddcup_10percent_dataset.csv")
print("Saving 10% dataset...")
df_10.to_csv(ten_path, index=False)
print("10% saved to:", ten_path)
print("10% shape:", df_10.shape)

print("\n-----------------------------------\n")