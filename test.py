import pandas as pd

# Load the CSV file
df = pd.read_csv("data/carsales.csv")

# Drop the 'Model' column
df = df.drop(columns=["Model"])

# Save the cleaned CSV
df.to_csv("data/Carsales.csv", index=False)

