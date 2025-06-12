import pandas as pd

# Load the CSV file
df = pd.read_csv("data/movies_record.csv")

# Drop the 'Model' column
df["Year"] = df["Year"].astype(int)

# Save the cleaned CSV
df.to_csv("data/movies_record.csv", index=False)

