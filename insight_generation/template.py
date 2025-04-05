from dotenv import load_dotenv

load_dotenv()


def template_subspace(subject):
    if subject['subspace']:
        return f" when {subject['subspace']}"
    return ""


def template_aggregation(subject):
    if subject['aggregation']:
        return f" {subject['aggregation']}"
    return ""


def style_value_fact(subject, x, value):
    subspace = template_subspace(subject)
    agg = template_aggregation(subject)
    if "percentage" in subject["measure"].lower():
        return f"The {subject['measure']} of {subject['breakdown']} {x} is {value}%{subspace}."
    else:
        return f"The {subject['measure']} of {subject['breakdown']} {x} is {value}{subspace}."


def style_difference_fact(subject, x1, x2, v1, v2):
    subspace = template_subspace(subject)
    agg = template_aggregation(subject)
    if v1 > 0 and v2 > 0 and max(v1/v2, v2/v1) >= 1.5:
        ratio = max(v1/v2, v2/v1)
        if v1 > v2:
            return f"The{agg} {subject['measure']} of {subject['breakdown']} {x1} is {ratio:.2f} times more than that of {x2}{subspace}."
        else:
            return f"No fact."

    value = v1 - v2
    rounded_value = f"{abs(value):.4f}".rstrip('0').rstrip('.') if '.' in f"{abs(value):.4f}" else f"{abs(value):.4f}"
    if value > 0:
        return f"The{agg} {subject['measure']} of {subject['breakdown']} {x1} is {rounded_value} more than that of {x2}{subspace}."
    return f"No fact."


def style_extremum_fact(subject, x, value):
    subspace = template_subspace(subject)
    agg = template_aggregation(subject)
    return f"The {value} value of the{agg} {subject['measure']} is {subject['breakdown']} {x}{subspace}."


def style_outlier_fact(subject, x):
    subspace = template_subspace(subject)
    agg = template_aggregation(subject)
    # return f"The{agg} {subject['measure']} of {subject['breakdown']} {x} is an outlier when compare with that of other {subject['breakdown']}(s){subspace}."
    

def style_outlier_scatter_fact(subject, x, y, name):
    subspace = template_subspace(subject)
    return f"{name} is an outlier with {subject['measure']} is {x} and {subject['measure2']} is {y}{subspace}."


def style_proportion_fact(subject, x, value):
    subspace = template_subspace(subject)
    agg = template_aggregation(subject)
    return f"The {subject['breakdown']} {x} accounts for {value:.2%} of the{agg} {subject['measure']}{subspace}."


def style_rank_fact(subject, x1, x2,x3, direction="top"):
    subspace = template_subspace(subject)
    agg = template_aggregation(subject)
    return f"In the{agg} {subject['measure']} ranking of different {subject['breakdown']}(s), the {direction} three {subject['breakdown']}(s) are {x1}, {x2}, {x3} {subspace}."


def style_trend_fact(subject, value, intervals):
    subspace = template_subspace(subject)
    agg = template_aggregation(subject)
    return f"The {value} trend of{agg} {subject['measure']} over {subject['breakdown']}(s){subspace} from {intervals[0]} to {intervals[-1]}."


def style_association_fact(subject, correlation):
    subspace = template_subspace(subject)
    level = "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.3 else "weak"
    return f"There is a {level} relationship between {subject['measure']} and {subject['measure2']}{subspace}, with pearson correlation coefficient of {correlation:.2f}."
