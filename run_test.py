code = """import pandas as pd
import altair as alt

def plot(data: pd.DataFrame):
    chart = alt.Chart(data).mark_point().encode(
        x=alt.X('Median_Debt:Q', title='Median Debt (Average Cost of Attendance)'),
        y=alt.Y('Median_Earnings:Q', title='Median Earnings'),
        tooltip=['Name:N', 'Median_Debt:Q', 'Median_Earnings:Q']
    ).properties(
        title='Relationship between Median Earnings and Average Cost of Attendance'
    )

    return chart

data = pd.read_csv("data/Colleges.csv")
chart = plot(data)
chart.save('chart_2.json')"""
exec(code)

import altair as alt
import json
with open(f"chart_2.json", "r") as f:
    vega_lite_json = json.load(f)
img = alt.Chart.from_dict(vega_lite_json)
img.save(f"chart_2.png")