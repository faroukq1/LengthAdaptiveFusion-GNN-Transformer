import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path("../../data/raw")
OUT_DIR = Path("../../data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load ratings.dat
ratings = pd.read_csv(
    DATA_DIR / "ratings.dat",
    sep="::", engine="python",
    names=["user_id", "item_id", "rating", "timestamp"]
)

# Implicit: rating >=4, users >=20 interactions (repo-like)
ratings = ratings[ratings["rating"] >= 4]
user_counts = ratings.groupby("user_id").size()
ratings = ratings[ratings["user_id"].isin(user_counts[user_counts >= 20].index)]

# Sort by user & timestamp (CRITICAL for seq order)
ratings = ratings.sort_values(["user_id", "timestamp"])

# Remap IDs densely (1-based)
users = sorted(ratings["user_id"].unique())
items = sorted(ratings["item_id"].unique())
user2id = {u: i+1 for i, u in enumerate(users)}
item2id = {i: j+1 for j, i in enumerate(items)}

num_users = len(user2id)
num_items = len(item2id)

# **CSV VERSION** (same data, Pandas-friendly)
ratings['user_id_remap'] = ratings['user_id'].map(user2id)
ratings['item_id_remap'] = ratings['item_id'].map(item2id)

# Save single CSV (user_id item_id only, timestamp-sorted)
ml1m_csv = ratings[['user_id_remap', 'item_id_remap']].rename(columns={
    'user_id_remap': 'user_id', 'item_id_remap': 'item_id'
})
ml1m_csv.to_csv(OUT_DIR / "ml-1m.csv", index=False)

# Stats CSV
pd.DataFrame({"user_count": [num_users], "item_count": [num_items]}).to_csv(OUT_DIR / "stats.csv", index=False)

print("Preprocessing done (CSV format - repo-equivalent).")
print(f"Users: {num_users}, Items: {num_items}")
print(f"Interactions: {len(ratings)}")
print(f"Saved: ml-1m.csv (timestamp-sorted, remapped)")
