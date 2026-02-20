import pandas as pd
import numpy as np

# Load cleaned data (after hard one-hot)
df = pd.read_csv("imputed_output.csv")

# One-hot groups
name_cols = [c for c in df.columns if c.startswith("name_")]
dept_cols = [c for c in df.columns if c.startswith("department_")]

# Reverse name one-hot → text
df["name"] = df[name_cols].idxmax(axis=1).str.replace("name_", "")

# Reverse department one-hot → text
df["department"] = df[dept_cols].idxmax(axis=1).str.replace("department_", "")

# Drop one-hot columns if you no longer need them
df = df.drop(columns=name_cols + dept_cols)

# Save final restored file
df.to_csv("restored_text_dataset.csv", index=False)

print("✅ Done! File saved as restored_text_dataset.csv")
