import logging
import altair as alt
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import warnings
from .template import style_extremum_fact
from .scoring import score_suitability, weighted_score
from .dataFact_scoring import score_fact

logging.basicConfig(level=logging.INFO)


def score_extremum_fact(chart_type=None):
    significance = 0
    impact_of_focus = 0
    suitability = score_suitability("Extremum")
    return weighted_score(significance, impact_of_focus, suitability)

def power_law(x, a):
    """Power-law function: f(x) = a * x^(1)"""
    return a * np.power(x, 1)

def calculate_significance(X: np.ndarray) -> float:
    """
    Test if data follows a power-law distribution with Gaussian noise.
    
    Parameters:
    X: numpy.ndarray
       
    Returns:
    float: significance score
    """   
    # 1. Fit power-law distribution
    # Generate x values for fitting (using indices + 1 to avoid log(0))
    x_fit = np.arange(1, len(X) + 1)
    
    # Fit the power-law function
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        popt, pcov = curve_fit(power_law, x_fit, X)
        a = popt
    
    # Calculate predicted values
    y_pred = power_law(x_fit, a)
    
    # Calculate residuals
    residuals = X - y_pred
    if len(residuals) < 3:
        return 1
    # 2. Test residuals for normality
    else:
        shapiro_stat, p_value = stats.shapiro(residuals)     
        # 3. Calculate significance
        significance = 1 - p_value
   
        return significance

def gen_basic_extremum_facts(df, subject, visualizer):
    facts = []
    df = df.sort_values(by=[subject["measure"]])
    x = df.iloc[0][subject["breakdown"]]
    significance = calculate_significance(df[subject["measure"]])
    facts.append(
        {
            "spec": visualizer.get_fact_visualized_chart("extremum", subject, x) if visualizer else None,
            "content": style_extremum_fact(subject, x, "lowest (least)"),
            "target": (x, "minimum"),
            "score": score_extremum_fact(),
            "score_C": significance*score_fact("Extremum"),
        }
    )
    x = df.iloc[-1][subject["breakdown"]]
    facts.append(
        {
            "spec": visualizer.get_fact_visualized_chart("extremum", subject, x) if visualizer else None,
            "content": style_extremum_fact(subject, x, "highest (most)"),
            "target": (x, "maximum"),
            "score": score_extremum_fact(),
            "score_C": significance*score_fact("Extremum"),
        }
    )
    return facts


def gen_extremum_facts(df, subject, visualizer):
    logging.info("gen_extremum_facts...")
    facts = []
    if len(df) == 0:
        logging.info(f"skipped: data not enough")
        return facts
    if subject["series"] is None:
        facts += gen_basic_extremum_facts(df, subject, visualizer)
    else:
        # group by breakdown (x-axis), compare among series
        for bi, bi_df in df.groupby(subject["breakdown"]):
            groupby_subject = subject.copy()
            groupby_subject["subspace"] = f'{subject["breakdown"]} is {bi}'
            groupby_subject["subspace_pair"] = (subject["breakdown"], bi)
            groupby_subject["breakdown"] = subject["series"]
            facts += gen_basic_extremum_facts(bi_df, groupby_subject, visualizer)
        # group by series, compare among breakdown (x-axis)
        for si, si_df in df.groupby(subject["series"]):
            groupby_subject = subject.copy()
            groupby_subject["subspace"] = f'{subject["series"]} is {si}'
            groupby_subject["subspace_pair"] = (subject["series"], si)
            facts += gen_basic_extremum_facts(si_df, groupby_subject, visualizer)
    logging.info(f"{len(facts)} facts generated.")
    return facts
