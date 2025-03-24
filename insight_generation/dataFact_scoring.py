import numpy as np
import math

# Importance Score(Is): Is(fi)=S(fi)⋅I(fi)
# Significance: S(fi) 
# Self-Information: I(fi) = −log2(P(fi)) where P(fi)=P(mi∣ti)⋅P(bi∣ti)⋅P(si)⋅P(xi∣si) 
# The value in score_C is only Self-Information, expect for "Value"
score_C ={
    "Value": 0,
    "Difference": 0,
    "Proportion": 0,
    "Trend": 0,
    "Rank": 0,
    "Association": 0,
    "Extremum": 0,
    "Outlier": 0,
    "Outlier_scatter": 0
       
    }
def score_importance(data_schema):
    dtype_counts = {}
    si = 0
    mi_ti = 0
    bi_ti = 0
    for i in range(len(data_schema)):
        dtype = data_schema[i]["properties"]["dtype"]
        if dtype not in dtype_counts:
            dtype_counts[dtype] = 1
        else:
            dtype_counts[dtype] += 1
    subspace_counts = 0
    for i in range(len(data_schema)):
        dtype = data_schema[i]["properties"]["dtype"]
        if dtype == "C" or dtype == "T":
            subspace_counts += 1 / data_schema[i]["properties"]["num_unique_values"]
    if "C" not in dtype_counts:
        dtype_counts["C"] = 0
    if "T" not in dtype_counts:
        dtype_counts["T"] = 0
    if (dtype_counts["C"] + dtype_counts["T"]) != 0:
        si = 1 / (dtype_counts["C"] + dtype_counts["T"]) * subspace_counts
        mi_ti = 1 / dtype_counts["N"]
        bi_ti = 1 / (dtype_counts["C"] + dtype_counts["T"])
    else:
        si = 1
        mi_ti = 1 / dtype_counts["N"]
        bi_ti = 1
    # value_fact  
        # S(fi) = the probability of the fact 
        # I(fi) = −log2(P(fi)) where P(fi)=P(mi∣ti)⋅P(si)
    score_C["Value"] = subspace_counts * (-np.log2(mi_ti*si))
    # difference_fact
    # proportion_fact
    # extremum_fact
    # outlier_fact
    # rank_fact
        # I(fi) = −log2(P(fi)) where P(fi)=P(mi∣ti)⋅P(bi∣ti)⋅P(si)
    score_C["Difference"] = (-np.log2(mi_ti*bi_ti*si))
    score_C["Proportion"] = (-np.log2(mi_ti*bi_ti*si))
    score_C["Extremum"] = (-np.log2(mi_ti*bi_ti*si))
    score_C["Outlier"] = (-np.log2(mi_ti*bi_ti*si))
    score_C["Rank"] = (-np.log2(mi_ti*bi_ti*si))
    # trend_fact
        # brekdown can only "T"
    if dtype_counts["T"] != 0:
        bi_ti_T = 1 / dtype_counts["T"]
        score_C["Trend"] = (-np.log2(mi_ti*bi_ti_T*si))       
    else:
        score_C["Trend"] = 0
    # association_fact    
    # outlier_scatter_fact
        # I(fi) = −log2(P(fi)) where P(fi)=P(mi∣ti)⋅P(bi∣ti)⋅P(si)
        # measure is "N"x"N"
    if dtype_counts["N"] >=2:
        mi_ti_N = 1 / math.comb(dtype_counts["N"], 2)
        score_C["Association"] = (-np.log2(mi_ti_N*bi_ti*si))
        score_C["Outlier_scatter"] = (-np.log2(mi_ti_N*bi_ti*si))
    else:
        score_C["Association"] = 0
        score_C["Outlier_scatter"] = 0
    


def score_fact(fact_type=None):
    return score_C[fact_type]
