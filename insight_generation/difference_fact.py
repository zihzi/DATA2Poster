import logging
import numpy as np
from itertools import combinations
from .template import style_difference_fact
from .scoring import score_suitability, weighted_score
from .dataFact_scoring import score_fact

logging.basicConfig(level=logging.INFO)


def score_difference_fact(chart_type=None):
    significance = 0
    impact_of_focus = 0
    suitability = score_suitability("Difference")
    return weighted_score(significance, impact_of_focus, suitability)


def gen_basic_difference_facts(df, subject, visualizer):
    facts = []
    max_ = np.max(np.abs(df[subject["measure"]].values))
    min_ = np.min(np.abs(df[subject["measure"]].values))
    max_diff = max_ - min_
    for (i, r1), (j, r2) in combinations(df.iterrows(), 2):       
        if i == j:
            continue
        x1, x2 = r1[subject["breakdown"]], r2[subject["breakdown"]]
        v1, v2 = r1[subject["measure"]], r2[subject["measure"]]
        value = abs(v1-v2)
        facts.append(
            {
                "spec": visualizer.get_fact_visualized_chart("difference", subject, x1, x2) if visualizer else None,
                "content": style_difference_fact(subject, x1, x2, v1, v2),
                "target": (x1, x2, v1-v2),
                "score": score_difference_fact(),
                "score_C": (value/max_diff)*score_fact("Difference")

            }
        )
        if v1 != v2:
            facts.append(
                {
                    "spec": visualizer.get_fact_visualized_chart("difference", subject, x1, x2) if visualizer else None,
                    "content": style_difference_fact(subject, x2, x1, v2, v1),
                    "target": (x1, x2, v2-v1),
                    "score": score_difference_fact(),
                    "score_C": (value/max_diff)*score_fact("Difference")
                }
            )
    return facts


def gen_difference_facts(df, subject, visualizer):
    logging.info("gen_difference_facts...")
    facts = []
    num_focus = 2
    if len(df) > 60:
        logging.info(f"skipped: data too large (more than 60)")
        return []
    if len(df) < num_focus:
        logging.info(f"skipped: data not enough")
        return facts
    if subject["series"] is None:
        facts += gen_basic_difference_facts(df, subject, visualizer)
    else:
        # group by breakdown (x-axis), compare within same x
        for bi, bi_df in df.groupby(subject["breakdown"]):
            if len(bi_df) < num_focus:
                continue
            groupby_subject = subject.copy()
            groupby_subject["subspace"] = f"{subject['breakdown']} is {bi}"
            groupby_subject["subspace_pair"] = (subject["breakdown"], bi)
            groupby_subject["breakdown"] = subject["series"]
            facts += gen_basic_difference_facts(bi_df, groupby_subject, visualizer)
        # group by series, compare within same series
        for si, si_df in df.groupby(subject["series"]):
            if len(si_df) < num_focus:
                continue
            groupby_subject = subject.copy()
            groupby_subject["subspace"] = f"{subject['series']} is {si}"
            groupby_subject["subspace_pair"] = (subject["series"], si)
            facts += gen_basic_difference_facts(si_df, groupby_subject, visualizer)

    logging.info(f"{len(facts)} facts generated.")
    return facts
