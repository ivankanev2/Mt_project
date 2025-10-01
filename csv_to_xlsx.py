import pandas as pd
df = pd.read_csv("results/summary.csv")
df.to_excel("results/summary.xlsx", index=False)
print("✅ Saved results/summary.xlsx")
