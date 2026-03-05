import pandas as pd
import ast

df = pd.read_csv("./data/mudras.csv")

# Convert string lists to actual Python lists
df["meanings"] = df["meanings"].apply(ast.literal_eval)
df["viniyoga"] = df["viniyoga"].apply(ast.literal_eval)

print(df.head())