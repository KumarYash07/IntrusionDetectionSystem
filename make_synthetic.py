"""Creates a SYNTHETIC dataset only to test that the pipeline runs. Not 5G data."""
import numpy as np, pandas as pd, os
rng = np.random.default_rng(0)
classes = ["Benign", "UDPFlood", "SYNFlood", "SYNScan", "UDPScan"]
rows = []
for ci, c in enumerate(classes):
    mu = rng.normal(0, 1.5, 16)
    X = rng.normal(mu, 1.0, (3000, 16))
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(16)])
    df["Attack Type"] = c
    df["session_id"] = ci * 10000 + rng.integers(0, 150, len(df))
    rows.append(df)
os.makedirs("data", exist_ok=True)
pd.concat(rows).sample(frac=1, random_state=0).to_csv("data/synthetic.csv", index=False)
print("wrote data/synthetic.csv")
