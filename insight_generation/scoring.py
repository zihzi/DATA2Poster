def weighted_score(significance, impact_of_focus, suitability):
    a = significance * 0.5
    b = impact_of_focus * 0.2
    c = suitability * 0.3
    return a + b + c

def score_suitability(fact_type=None, chart_type="Line"):
    return SUITABILITY[fact_type][chart_type]

SUITABILITY = {
    "Trend": {
        "Area": 10/57,
        "Pie/Donut": 0/57,
        "Bar": 5/57,
        "Line": 42/57,
        "Scatter plot": 0/57,
    },
    "Difference": {
        "Area": 25/55,
        "Pie/Donut": 12/55,
        "Bar": 16/55,
        "Line": 2/55,
        "Scatter plot": 0/55,
    },
    "Proportion": {
        "Area": 7/82,
        "Pie/Donut": 65/82,
        "Bar": 10/82,
        "Line": 0/82,
        "Scatter plot": 0/82,
    },
    "Rank": {
        "Area": 17/42,
        "Pie/Donut": 0/42,
        "Bar": 25/42,
        "Line": 0/42,
        "Scatter plot": 0/42,
    },
    "Extremum": {
        "Area": 4/11,
        "Pie/Donut": 1/11,
        "Bar": 6/11,
        "Line": 0/11,
        "Scatter plot": 0/11,
    },
    "Outlier": {
        "Area": 0/5,
        "Pie/Donut": 0/5,
        "Bar": 0/5,
        "Line": 0/5,
        "Scatter plot": 5/5,
    },
    "Value": {
        "Area": 22/36,
        "Pie/Donut": 6/36,
        "Bar": 6/36,
        "Line": 2/36,
        "Scatter plot": 0/36,
    },
    "Association": {
        "Area": 0/-1,
        "Pie/Donut": 0/-1,
        "Bar": 0/-1,
        "Line": 0/-1,
        "Scatter plot": 0/-1,
    },
}

# supported_chart_types = [
#     "Area",
#     "Pie/Donut",
#     "Bar",
#     "Line",
#     "Scatter plot",
# ]