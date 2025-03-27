import logging
import altair as alt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
import warnings
from .template import style_rank_fact
from .scoring import score_suitability, weighted_score
from .dataFact_scoring import score_fact

logging.basicConfig(level=logging.INFO)


def score_rank_fact(chart_type=None):
    significance = 0
    impact_of_focus = 0
    suitability = score_suitability("Rank")
    return weighted_score(significance, impact_of_focus, suitability)

def power_law(x, a, b):
        """Power law function: f(x) = ax^b"""
        return a * np.power(x, b)
    
def check_gaussian_residuals(residuals: np.ndarray) -> tuple:
        """
        Test if residuals follow Gaussian distribution using Shapiro-Wilk test
        """
        statistic, p_value = stats.shapiro(residuals)
        return statistic, p_value
def calculate_significance(X: np.ndarray) -> float:
    """
    Test if data follows a power-law distribution with Gaussian noise.
    
    Parameters:
    X: np.ndarray

    Returns:
    float: significance score
    """
    # 1. Fit power-law distribution
    # Generate x values for fitting (using indices + 1 to avoid log(0))
    x_fit = np.arange(1, len(X) + 1)
    
    # Fit the power law function
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        popt, pcov = curve_fit(power_law, x_fit, X, maxfev=2000)
    
    # Calculate predicted values
    y_pred = power_law(x_fit, *popt)
    
    # Calculate residuals
    residuals = X - y_pred
    
    # 2. Test residuals for normality and calculate p-value
    shapiro_stat, p_value = check_gaussian_residuals(residuals)
    
    # 3. Calculate significance
    significance = 1 - p_value

    return significance

def gen_basic_rank_facts(df, subject, visualizer):
    facts = []
    df = df.sort_values(by=[subject["measure"]], ascending=False)
    t1 = df.iloc[0][subject["breakdown"]]
    t2 = df.iloc[1][subject["breakdown"]]
    t3 = df.iloc[2][subject["breakdown"]]
    b1 = df.iloc[-1][subject["breakdown"]]
    b2 = df.iloc[-2][subject["breakdown"]]
    b3 = df.iloc[-3][subject["breakdown"]]

    if df[subject["breakdown"]].dtype not in ["int64", "float64"]:
        logging.info(f"skipped: breakdown not numerical")
        significance = 1
    else:
        significance = calculate_significance(df[subject["breakdown"]])

    facts.append(
        {
            "spec": visualizer.get_fact_visualized_chart("rank", subject, t1, t2, t3) if visualizer else None,
            "content": style_rank_fact(subject, t1),
            "target": (t1, t2, t3),
            "score": score_rank_fact(),
            "score_C": significance*score_fact("Rank"),
        }
    )
    facts.append(
        {
            "spec": visualizer.get_fact_visualized_chart("rank", subject, b1, b2, b3) if visualizer else None,
            "content": style_rank_fact(subject, b1,"last"),
            "target": (b1, b2, b3),
            "score": score_rank_fact(),
            "score_C": significance*score_fact("Rank"),
        }
    )
    return facts


def gen_rank_facts(df, subject, top=3, visualizer=None):
    logging.info("gen_rank_facts...")
    facts = []
    if len(df) <= top:
        logging.info(f"skipped: data not enough")
        return facts
    if subject["series"] is None:
        facts += gen_basic_rank_facts(df, subject, visualizer)
    else:
        # group by breakdown (x-axis), compare among series
        for bi, bi_df in df.groupby(subject["breakdown"]):
            if len(bi_df) <= top:
                logging.info(f"skipped: grouped data not enough")
                continue
            groupby_subject = subject.copy()
            groupby_subject["subspace"] = f'{subject["breakdown"]} is {bi}'
            groupby_subject["subspace_pair"] = (subject["breakdown"], bi)
            groupby_subject["breakdown"] = subject["series"]
            facts += gen_basic_rank_facts(bi_df, groupby_subject, visualizer)
        # group by series, compare among breakdown (x-axis)
        for si, si_df in df.groupby(subject["series"]):
            if len(si_df) <= top:
                logging.info(f"skipped: grouped data not enough")
                continue
            groupby_subject = subject.copy()
            groupby_subject["subspace"] = f'{subject["series"]} is {si}'
            groupby_subject["subspace_pair"] = (subject["series"], si)
            facts += gen_basic_rank_facts(si_df, groupby_subject, visualizer)
    logging.info(f"{len(facts)} facts generated.")
    return facts
