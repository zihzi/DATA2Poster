import logging
from .template import style_value_fact
from .scoring import score_suitability, weighted_score

logging.basicConfig(level=logging.INFO)

# def gen_categorization_facts(df, subject):
   
#     logging.info("gen_categorization_facts...")
#     return []



def gen_basic_categorization_facts(df, subject):
    facts = []
    b =subject["breakdown"]
    s =subject["series"]
    counts = df[subject["breakdown"]].unique()
    focus=df[subject["breakdown"]].value_counts().idxmax()
    max_record = df[subject["breakdown"]].value_counts()[0]
    facts.append(
            {
                "content": f"There are {len(counts)} categories of {b} and {focus} has the most record which is {max_record}.",
                "score_C": 1,
            }
        )
    print(facts)
    return facts


def gen_categorization_facts(df, subject):
    logging.info("gen_categorization_facts...")
    facts = []
    facts += gen_basic_categorization_facts(df, subject)
    logging.info(f"{len(facts)} facts generated.")
    return facts
