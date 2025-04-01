import pandas as pd
df = pd.read_csv("data/not_used/bike_shraing_day.csv")
df.drop(columns=["instant"],inplace=True)
df.drop(columns=["year"], inplace=True)
holiday_Mapping = {
    0: "No",
    1: "Yes"
}
df["holiday"] = df["holiday"].map(holiday_Mapping)
season_Mapping = {
    1: "Spring",
    2: "Summer",
    3: "Fall",
    4: "Winter"
}
df["season"] = df["season"].map(season_Mapping)
df["workingday"] = df["workingday"].map(holiday_Mapping)
weather_Mapping = {
    1: "Clear",
    2: "Mist",
    3: "Light Rain/Snow",
    4: "Heavy Rain/Snow"
}
df["weather"] = df["weather"].map(weather_Mapping)
month_mapping = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec"   
    
}
df["month"] = df["month"].map(month_mapping)
weekday_mapping = {
    0: "Sun",
    1: "Mon",
    2: "Tue",
    3: "Wed",
    4: "Thu",
    5: "Fri",
    6: "Sat"
}
df["weekday"] = df["weekday"].map(weekday_mapping)
df.to_csv("data/bike_sharing_day.csv", index=False)