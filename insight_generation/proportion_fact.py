import logging
import altair as alt
from .template import style_proportion_fact
from .scoring import score_suitability, weighted_score
from .dataFact_scoring import score_fact

logging.basicConfig(level=logging.INFO)


def score_proportion_fact(chart_type=None):
    significance = 0
    impact_of_focus = 0
    suitability = score_suitability("Proportion")
    return weighted_score(significance, impact_of_focus, suitability)


def gen_basic_proportion_facts(df, subject, visualizer):
    facts = []
    total = df[subject["measure"]].sum()
    p = 1
    for _, row in df.iterrows():
        
        x = row[subject["breakdown"]]
        v = row[subject["measure"]]
        value = v / total if total != 0 else 0
        if value <0.5:
            p = value     
        facts.append(
            {
                "spec": visualizer.get_fact_visualized_chart("proportion", subject, x) if visualizer else None,
                "content": style_proportion_fact(subject, x, value),
                "target": (x, value),
                "score": score_proportion_fact(),
                "score_C": p*score_fact("Proportion"),
            }
        )
    return facts


def gen_proportion_facts(df, subject, visualizer):
    logging.info("gen_proportion_facts...")
    facts = []
    if len(df) == 0:
        logging.info(f"skipped: data not enough")
        return facts
    if subject["series"] is None:
        facts += gen_basic_proportion_facts(df, subject, visualizer)
    else:
        # group by breakdown (x-axis), compare among series
        for bi, bi_df in df.groupby(subject["breakdown"]):
            if len(bi_df) == 0:
                continue
            groupby_subject = subject.copy()
            groupby_subject["subspace"] = f'{subject["breakdown"]} is {bi}'
            groupby_subject["subspace_pair"] = (subject["breakdown"], bi)
            groupby_subject["breakdown"] = subject["series"]
            facts += gen_basic_proportion_facts(bi_df, groupby_subject, visualizer)
        # group by series, compare among breakdown (x-axis)
        for si, si_df in df.groupby(subject["series"]):
            if len(si_df) == 0:
                continue
            groupby_subject = subject.copy()
            groupby_subject["subspace"] = f'{subject["series"]} is {si}'
            groupby_subject["subspace_pair"] = (subject["series"], si)
            facts += gen_basic_proportion_facts(si_df, groupby_subject, visualizer)
    logging.info(f"{len(facts)} facts generated.")
    return facts
