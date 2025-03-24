import logging
import numpy as np
import pandas as pd
from scipy.stats import t
from .template import style_association_fact
from .scoring import score_suitability, weighted_score
from .dataFact_scoring import score_fact

logging.basicConfig(level=logging.INFO)


def score_association_fact(chart_type=None):
    significance = 0
    impact_of_focus = 0
    suitability = score_suitability("Association")
    return weighted_score(significance, impact_of_focus, suitability)

def calculate_significance(r: float, n: int) -> float:
    """
    Calculate the statistical significance of the correlation between two variables.
    
    Parameters:
    r: the Pearson correlation coefficient  
    n: the number of data points
    
    Returns:
    float: significance score
    """
    # Calculate the t-statistic
    t_stat = r * np.sqrt((n - 2) / (1 - r**2))
    
    # Calculate the p-value
    p_value = 2 * (1 - t.cdf(abs(t_stat), n - 2))
    
    # Determine the significance score
    if p_value < 0.001:
        significance_score = 1.0
    elif p_value < 0.01:
        significance_score = 0.8
    elif p_value < 0.05:
        significance_score = 0.6
    elif p_value < 0.1:
        significance_score = 0.4
    else:
        significance_score = 0.2
    
    return significance_score

def gen_basic_association_facts(df, subject, visualizer):
    facts = []
    correlation = df[subject["measure"]].corr(df[subject["measure2"]])
    significance = calculate_significance(correlation,len(df[subject["measure"]]))
    facts.append(
        {
            "spec": visualizer.get_fact_visualized_chart("association", subject, correlation) if visualizer else None,
            "content": style_association_fact(subject, correlation),
            "target": (None, "association"),
            "score": score_association_fact(),
            "score_C": significance*score_fact("Association")
        }
    )
    return facts


def gen_association_facts(df, subject, visualizer):
    logging.info("gen_association_facts...")
    facts = []
    if len(df) == 0:
        logging.info(f"skipped: data not enough")
        return facts
    if subject["series"] is None:
        facts += gen_basic_association_facts(df, subject, visualizer)
    else:
        # group by series, compare among breakdown (x-axis)
        for si, si_df in df.groupby(subject["series"]):
            if len(si_df) == 0:
                continue
            groupby_subject = subject.copy()
            groupby_subject["subspace"] = f'{subject["series"]} is {si}'
            groupby_subject["subspace_pair"] = (subject["series"], si)
            facts += gen_basic_association_facts(si_df, groupby_subject, visualizer)
    logging.info(f"{len(facts)} facts generated.")
    return facts
