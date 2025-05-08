# import pandas as pd
# import json
# import altair as alt
# import altair_saver
# df = pd.read_csv("data/cas_cancer_nc.csv")
# df.drop(columns=["id"],inplace=True)
# df.drop(columns=["ICD-10"],inplace=True)
# df.to_csv("data/cas_cancer_nc.csv", index=False)

import pandas as pd

# Load CSV data
df = pd.read_csv("data/2024USA_presidential_election.csv")
df["REP_PERCENT"] = df["REP_PERCENT"].str.rstrip('%').astype(float)
df["DEM_PERCENT"] = df["DEM_PERCENT"].str.rstrip('%').astype(float)
df["OTH_PERCENT"] = df["OTH_PERCENT"].str.rstrip('%').astype(float)
df["STATE"] = df["STATE"].astype("category")

# Define region mapping
region_map = {
    "Northeast": [
        "Connecticut", "Maine", "Massachusetts", "New Hampshire",
        "Rhode Island", "Vermont", "New Jersey", "New York", "Pennsylvania"
    ],
    "Southeast": [
        "Alabama", "Arkansas", "Florida", "Georgia", "Kentucky", "Louisiana",
        "Mississippi", "North Carolina", "South Carolina", "Tennessee",
        "Virginia", "West Virginia"
    ],
    "Midwest": [
        "Illinois", "Indiana", "Iowa", "Kansas", "Michigan", "Minnesota",
        "Missouri", "Nebraska", "North Dakota", "Ohio", "South Dakota", "Wisconsin"
    ],
    "Southwest": ["Arizona", "New Mexico", "Oklahoma", "Texas"],
    "West": [
        "Alaska", "California", "Colorado", "Hawaii", "Idaho", "Montana",
        "Nevada", "Oregon", "Utah", "Washington", "Wyoming"
    ],
}

# Create reverse map: State → Region
state_to_region = {}
for region, states in region_map.items():
    for state in states:
        state_to_region[state] = region

# Add a Region column
df["Region"] = df["STATE"].map(state_to_region)

# Show grouped data (e.g., total votes per region)
grouped = df.groupby("Region").agg({
    "TOTAL_VOTES": "sum",
    "DEM_VOTES": "sum",
    "DEM_PERCENT": "mean",
    "DEM_EV": "sum",
    "REP_VOTES": "sum",
    "REP_PERCENT": "mean",
    "REP_EV": "sum",
    "OTH_VOTES": "sum",
    "OTH_PERCENT": "mean",
    "OTH_EV": "sum"}).reset_index()

print(grouped)
grouped.to_csv("regional_vote_summary.csv", index=False)