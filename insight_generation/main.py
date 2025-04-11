import logging
import pandas as pd
import altair as alt
from .value_fact import gen_value_facts
from .difference_fact import gen_difference_facts
from .proportion_fact import gen_proportion_facts
from .extremum_fact import gen_extremum_facts
from .rank_fact import gen_rank_facts
# from .outlier_fact import gen_outlier_facts
from .trend_fact import gen_trend_facts
from .association_fact import gen_association_facts
from .categorization_fact import gen_categorization_facts
# from .outlier_scatter_fact import gen_outlier_scatter_facts
from .visualizations import Visualizer
import json
from pathlib import Path


logging.basicConfig(level=logging.INFO)


def generate_facts(
    dataset=None,
    breakdown=None,
    measure=None, 
    measure2=None,
    series=None,
    chart_type=None,
    breakdown_type=None,
    measure_type=None,
    x_type=None,
    y_type=None,
    c_type=None,
    with_vis=None,
    only_base=False,
    orient="vertical",
):
    df = pd.read_csv(dataset)
    subject = {
        "subspace": None,
        "breakdown": breakdown,
        "measure": measure,
        "measure2": measure2,
        "series": series,
        "aggregation": None,
        "visualization": chart_type,
    }

    if orient == "horizontal":
        x_type, y_type = y_type, x_type
    
    if only_base:
        visualizer = Visualizer(df, subject, x_type, y_type, c_type, orient)
        return visualizer.get_base_chart()
    elif with_vis:
        visualizer = Visualizer(df, subject, x_type, y_type, c_type, orient)
        facts = _generate_facts(df, subject, breakdown_type, measure_type, visualizer)
        facts = [fact for fact in facts if fact.get("spec", None)]
        return visualizer.get_base_chart(), facts
    else:
        facts = _generate_facts(df, subject, breakdown_type, measure_type)
        return facts


def _generate_facts(df, subject=None, bt=None, mt=None, visualizer: Visualizer=None):
    """
    Generate facts from data.
    - bt (breadown type) should be in ["C", "T", None] (categorical, temporal, none)
    - mt (measure type) should be in ["N", "NxN", None] (1 dimension, 2 dimensions, none)
    """
    logging.info("gen_facts...")
    facts = []
    # if bt is None and mt == "N":
    if bt == "T":
        print(subject)
    if bt in ("C", "T"):
        if mt == "N":
            # facts += gen_value_facts(df.copy(), subject, visualizer)
            facts += gen_difference_facts(df.copy(), subject, visualizer)
            if not "percentage" in subject["measure"].lower():
                facts += gen_proportion_facts(df.copy(), subject, visualizer)
            # facts += gen_rank_facts(df.copy(), subject, top=3, visualizer=visualizer)
            # facts += gen_extremum_facts(df.copy(), subject, visualizer)
            # facts += gen_outlier_facts(df.copy(), subject, visualizer)
        if mt == "NxN":
            facts += gen_association_facts(df.copy(), subject, visualizer)
            # facts += gen_outlier_scatter_facts(df.dropna().copy(), subject, visualizer)
    if bt == "T" and mt == "N":
        facts += gen_trend_facts(df.copy(), subject, visualizer)
    # if bt == "C":
    #     if mt is None:
    # facts += gen_categorization_facts(df.copy(), subject)
    #     if mt == "N":
    #         facts += gen_distribution_facts(df.copy(), subject)

    return facts


# if __name__ == "__main__":
#     facts = generate_facts(
#         dataset=Path("data/seattle_monthly_precipitation_2015.csv"),
#         breakdown="month",
#         measure="precipitation",
#         series=None,
#         breakdown_type="T",
#         measure_type="N",
#         with_vis=False,
#     )
#     facts = [fact["content"] for fact in facts]
#     print(json.dumps(facts, indent=2))
