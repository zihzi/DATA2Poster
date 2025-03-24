import logging
import altair as alt
import numpy as np
from .template import style_trend_fact
from scipy.stats import kendalltau, linregress, logistic
from .scoring import score_suitability, weighted_score
from .dataFact_scoring import score_fact

logging.basicConfig(level=logging.INFO)

significance_level = 0.05


def score_trend_fact(chart_type=None):
    significance = 0
    impact_of_focus = 0
    suitability = score_suitability("Trend")
    return weighted_score(significance, impact_of_focus, suitability)

def calculate_significance(x, y):
    """
    Computes the significance metric based on the provided data.

    Parameters:
    x (array-like): Independent variable data.
    y (array-like): Dependent variable data.

    Returns:
    significance score (float): The significance score of the outlier.
    """
    # Step 1: Perform linear regression
    regression = linregress(x, y)
    slope = regression.slope
    r_squared = regression.rvalue**2  # Goodness-of-fit value (r^2)

    # Step 2: Model slope distribution using logistic distribution
    # Assume mean=0 and scale=1 for simplicity
    loc, scale = 0, 1

    # Step 3: Compute p-value
    # p-value is the probability of slope >= observed slope
    p_value = 1 - logistic.cdf(slope, loc=loc, scale=scale)

    # Step 4: Calculate significance
    significance = r_squared * (1 - p_value)
    print(f"Significance: {significance}")
    return significance

def find_peaks_and_valleys(y):
    peaks, valleys = [], []
    for i in range(0, len(y)):
        if i == 0:
            if y[i] > y[i + 1]:
                peaks.append(i)
            elif y[i] < y[i + 1]:
                valleys.append(i)
        elif i == len(y) - 1:
            if y[i] > y[i - 1]:
                peaks.append(i)
            elif y[i] < y[i - 1]:
                valleys.append(i)
        elif y[i - 1] < y[i] and y[i] > y[i + 1]:
            peaks.append(i)
        elif y[i - 1] > y[i] and y[i] < y[i + 1]:
            valleys.append(i)
    return peaks, valleys


def analyze_segment_trend(y_subset):
    if len(y_subset) < 2:
        return None
    tau, p_value = kendalltau(range(len(y_subset)), y_subset)
    if p_value <= significance_level:
        if tau > 0:
            return "increasing"
        elif tau < 0:
            return "decreasing"
    return "flat"


def gen_basic_trend_facts(df, subject, visualizer):
    facts = []

    x = df[subject["breakdown"]].values
    y = df[subject["measure"]].values
    if x.dtype or y.dtype not in ["int64", "float64"]:
        logging.info(f"skipped: x or y not numerical")
        significance = 1
    else:
        significance = calculate_significance(x, y)
    # Detailed trend analysis
    peaks, valleys = find_peaks_and_valleys(y)
    segments = sorted(peaks + valleys)
    detailed_trends = set()
    for i in range(len(segments) - 1):
        start, end = segments[i], segments[i + 1]
        # avoid repeating with overall trend
        if start == 0 and end == len(y) - 1:
            continue
        segment_trend = None
        if y[start] < y[end]:
            segment_trend = "increasing"
        elif y[start] > y[end]:
            segment_trend = "decreasing"
        if segment_trend:
            detailed_trends.add(segment_trend)
            facts.append(
                {
                    "spec": visualizer.get_fact_visualized_chart(
                        "trend",
                        subject,
                        x[start],
                        x[end],
                        segment_trend,
                        float(end - start) / len(x),
                    ) if visualizer else None,
                    "content": style_trend_fact(
                        subject, segment_trend, [x[start], x[end]]
                    ),
                    "target": (x[start], x[end], segment_trend),
                    "score": score_trend_fact(),
                    "score_C": significance*score_fact("Trend"),
                }
            )

    # Overall trend analysis
    overall_trend = analyze_segment_trend(y)
    if overall_trend == "flat":
        if len(detailed_trends) > 1:
            overall_trend = "wavering"
    facts.append(
        {
            "spec": visualizer.get_fact_visualized_chart("trend", subject, x[0], x[-1], overall_trend, 1.0) if visualizer else None,
            "content": style_trend_fact(subject, overall_trend, [x[0], x[-1]]),
            "target": (x[0], x[-1], overall_trend),
            "score": score_trend_fact(),
            "score_C": significance*score_fact("Trend"),
        }
    )

    return facts


def gen_trend_facts(df, subject, visualizer):
    logging.info("gen_trend_facts...")
    facts = []
    if len(df) == 0:
        logging.info(f"skipped: data not enough")
        return facts

    if subject["series"] is None:
        facts += gen_basic_trend_facts(df, subject, visualizer)
    else:
        # group by series, compare among breakdown (x-axis)
        for si, si_df in df.groupby(subject["series"]):
            if len(si_df) < 2:
                continue
            groupby_subject = subject.copy()
            groupby_subject["subspace"] = f'{subject["series"]} is {si}'
            groupby_subject["subspace_pair"] = (subject["series"], si)
            facts += gen_basic_trend_facts(si_df, groupby_subject, visualizer)

    logging.info(f"{len(facts)} facts generated.")
    return facts
