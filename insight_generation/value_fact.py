import logging
from .template import style_value_fact
from .scoring import score_suitability, weighted_score
from .dataFact_scoring import score_fact

logging.basicConfig(level=logging.INFO)


def score_value_fact(chart_type=None):
    significance = 0
    impact_of_focus = 0
    suitability = score_suitability("Value")
    return weighted_score(significance, impact_of_focus, suitability)


def gen_basic_value_facts(df, subject, visualizer):
    facts = []
    for _, r in df.iterrows():
        x = r[subject["breakdown"]]
        v = r[subject["measure"]]
        if subject["series"] is not None:
            subject["subspace"] = f'{subject["series"]} is {r[subject["series"]]}'
            subject["subspace_pair"] = (subject["series"], r[subject["series"]])
        facts.append(
            {
                "spec": visualizer.get_fact_visualized_chart("value", subject, x) if visualizer else None,
                "content": style_value_fact(subject, x, v),
                "target": (x, v),
                "score": score_value_fact(),
                "score_C": score_fact("Value"),
            }
        )
    return facts


def gen_value_facts(df, subject, visualizer):
    logging.info("gen_value_facts...")
    facts = []
    if len(df) > 60:
        logging.info(f"skipped: data too large (more than 60)")
        return []
    facts += gen_basic_value_facts(df, subject, visualizer)
    logging.info(f"{len(facts)} facts generated.")
    return facts

