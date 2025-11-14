from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate





def agent11_vis_recommender(conclusion, title_1, title_2, insight, column, openai_key):
    llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", api_key = openai_key)
    prompt_template = """
    You are an Chart-Text Consistency Reviewer.
    Here is conclusion and the titles of two charts:
    Conclusion:\n\n"{conclusion}"\n\n
    Title 1: "{title_1}"\n\n
    Title 2: "{title_2}"\n\n
    Here is insight derived from the two charts:\n\n{insight}\n\n
    
    **Task**
    Given the conclusion, the titles of two charts,and insight derived from the two charts, judge whether the conclusion is supported from the two charts. 
    Classify as supported/partially_supported/unsupported using the rubric provided.
    If not fully supported, pick which chart should be replaced and output a concrete replacement recommendation for a better visualization.

    **Evaluation Categories**
    ## Task Compliance (Binary Scoring: 0/1) 
        For each criterion, provide a binary score: 1 for compliance (requirement met) or 0 for non-compliance (requirement not met).
        1. **Claim-Type Detection**
        - **Question**: Did you detect the primary claim type in the conclusion (Trend / Proportion / Distribution / Correlation)?
        - **Score**: 1 if detected; 0 if not.
        2. **Insight Consistency**
        - **Question**: Is the provided insight consistent with (a) the titles' scope/measure and (b) the conclusion's claim, without contradictions?
        - **Score**: 1 if consistent; 0 if contradicted or irrelevant.

    ## Consistency Quality (3-Level Scoring: 0/1/2)
        For each criterion, provide a score from 0-2 with brief justification.
        1. **Entailment Strength** 
        - **Score 2**: Titles clearly imply evidence for the main claim (strong entailment).
        - **Score 1**: Partial support; evidence is incomplete or ambiguous.
        - **Score 0**: Titles cannot evidence the claim.

        2. **Scope & Measure Alignment**
        - **Score 2**: Measures, aggregation (total/avg/share), and scope (overall vs subset) align with the conclusion.
        - **Score 1**: Minor mismatches (e.g., ambiguous aggregation or slight scope drift).
        - **Score 0**: Major mismatch (e.g., claim on total, title on a subcategory).

        3. **Relation / Claim-Type Cues**
        - **Score 2**: Title cues match claim type (e.g., time for Trend; share/% for Proportion; “vs/relationship” for Correlation; “distribution/by/across” for Distribution).
        - **Score 1**: Some cues present but incomplete.
        - **Score 0**: Essential cues missing.

        4. **Granularity & Specificity**
        - **Score 2**: Appropriate granularity (clear time window, explicit subgroup/top-N if implied).
        - **Score 1**: Partially specified.
        - **Score 0**: Absent or conflicting.

        5. **Explanation Quality**
        - **Score 2**: Concise, concrete rationale referencing exact alignments or mismatches.
        - **Score 1**: Generally useful but somewhat vague.
        - **Score 0**: Unclear or speculative.

    ## Claim-Type Guidance (for detection & reasoning):
        **Trend/Change over time** — conclusion words: “increase”, “decrease”, “rise”, “fall”, “trend”, “fluctuate”, “peak”, “decline”.
        Supporting title should imply: “over time”, “year”, “month”, “quarter”, “since/through”, “trend”.

        **Proportion/Share/Comparison** — conclusion words: “most”, “percentage”, “proportion”, “dominant/dominance”, “leading”, “highest/lowest”.
        Supporting title should imply: “share”, “percentage”, “proportion”, “of/in/out of”.

        **Distribution** — conclusion words: “distribution”, “spread”, “range”, “variation”, “skew”, “outlier”.
        Supporting title should imply: “distribution”, or “by/across/among [category]”.

        **Correlation/Relationship** — conclusion words: “correlation”, “relationship”, “associated”, “linked”.
        Supporting title should imply: “vs”, “compared to”, “correlation”, “relationship”.

    ## Replacement Recommendation Rules (only if not fully supported):
    1. Identify possible claim words and the main claim type.
    2. Map the claim type to a stronger viz:
    - Trend → line (x=time, y=agg(metric))
    - Proportion → pie (or 100%% stacked bar if many parts)
    - Distribution → histogram (grouped as needed)
    - Correlation → scatter (optionally mention trendline)
    3. Provide a concrete query that the recommended chart would answer, e.g.:
    - Trend: "How did {{metric}} change over {{time}} by {{group}}?"
    - Proportion: "What share of {{metric}} comes from {{group}}?"
    - Correlation: "What is the relationship between {{x}} and {{y}}?"
    - Distribution: "What is the distribution of {{metric}} across {{group}}?"
    4. Your recommended chart should strengthen the main claim and not duplicate the claim type of the chart you keep.

    ##Constraints:
    The query should only reference the following columns: {column}.

    **Output (JSON)**
    Do not INCLUDE ```json```.Do not add other sentences after this json data.
    Return **only** the final JSON in this structure:
    {{
    "vis_check":
    [
    {{
        "task_compliance": {{
        "claim_type_detection": {{
        "score": <0_or_1>,
        "reason": "Detected claim type is 'Proportion' based on keywords like 'dominates' and 'share'."
        }},
        "insight_consistency": {{
        "score": <0_or_1>,
        "reason": "<State whether the insight aligns with titles and conclusion; note contradictions if 0.>"
        }}
        }},
        "consistency_quality": {{
        "entailment_strength": {{
        "score": <0_1_or_2>,
        "reason": "<Explain how strongly titles imply evidence for the conclusion.>"
        }},
        "scope_and_measure_alignment": {{
        "score": <0_1_or_2>,
        "reason": "<Assess alignment of measure, aggregation, and scope (overall vs subset).>"
        }},
        "relation_cues_match": {{
        "score": <0_1_or_2>,
        "reason": "<Evaluate presence/absence of claim-type cues (time/share/vs/distribution).>"
        }},
        "granularity_specificity": {{
        "score": <0_1_or_2>,
        "reason": "<Evaluate time window, subgroup/top-N detail, unit/level clarity.>"
        }},
        "explanation_quality": {{
        "score": <0_1_or_2>,
        "reason": "<Judge clarity and specificity of your justification.>"
        }}
        "verdict": <"supported"/"partially_supported"/"unsupported">,
        "replace": <"chart_1"/"chart_2"/"none">,
        "reason": <explanation of your verdict and replace decision>,
        "recommendation": {{
            "chart_type": <type of chart to recommend, e.g., "line", "bar", "pie", "scatter", "histogram">,
            "query": <a concrete query that the recommended chart would answer>,
            "revised_title": <a concrete revised title for the recommended chart>,
            "explanation": <brief explanation of why this chart and title would better support the conclusion>
        }} or none if no replacement needed
    }}
    ]
    }}
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["conclusion","title_1","title_2","insight","column"],
    )
    chain = prompt | llm
    response = chain.invoke(input ={"conclusion":conclusion,"title_1":title_1,"title_2":title_2,"insight":insight,"column":column})
    return response.content

def agent12_final_checker(conclusion, titles, insight, openai_key):
    llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", api_key = openai_key)
    prompt_template = """
    You are an Chart-Text Consistency Reviewer.
    Here is conclusion and the titles of six charts:
    Conclusion:\n\n"{conclusion}"\n\n
    Title 1: "{titles_1}"\n\n
    Title 2: "{titles_2}"\n\n
    Title 3: "{titles_3}"\n\n
    Title 4: "{titles_4}"\n\n
    Title 5: "{titles_5}"\n\n
    Title 6: "{titles_6}"\n\n
    Here is insight derived from the six charts:\n\n{insight}\n\n

    **Task**
    Given the conclusion, the titles of the six charts, and the insight, judge how well the conclusion is supported by the provided evidence.
    Score the support on a scale of 0-10 using the rubric below.
    Provide a detailed justification for your score.
    If the score is not 10, provide concrete recommendations for charts that should be added or replaced.

    **Analysis Steps**:
    1. Deconstruct Conclusion: Identify every distinct claim made in the conclusion (e.g., "sales increased," "X is the largest category," "Y correlates with Z," "the trend peaked in Q3").
    2. Analyze Chart Titles: For each of the 6 titles, determine what data it represents and what type of claim it can support (e.g., Title 1 "Sales Over Time" supports a trend claim; Title 2 "Market Share by Competitor" supports a comparison/proportion claim).
    3. Incorporate Insight: Read the insight to find context that links the charts or provides specific data points (e.g., "Chart 1 and 3 show...").
    4. Cross-Reference & Score: Match each claim from the conclusion to the chart title(s) and insight text. Assign a score based on how many claims are supported.

    **Scoring Rubric (0-10)**:

    - 10 (Fully Supported): Every single claim, including major points and minor details, in the conclusion is clearly and directly supported by one or more of the chart titles and the provided insight. All charts are relevant.
    - 7-9 (Mostly Supported): The main, overarching claim of the conclusion is well-supported. However, one or two minor details or secondary claims are not explicitly verifiable from the titles/insight.
    - 4-6 (Partially Supported): A significant part of the conclusion is supported, but another significant part is missing or unsupported. For example, the conclusion makes two major claims, but the charts only provide evidence for one of them.
    - 1-3 (Weakly Supported): There is a vague, indirect link between the charts and the conclusion, but the charts are largely irrelevant or insufficient to prove the core claims.
    - 0 (Unsupported): The conclusion is completely disconnected from the provided chart titles and insight. The charts do not provide any evidence for the claims made.
    
    **Output (JSON)**
    Do not INCLUDE ```json```.Do not add other sentences after this json data.
    Do not use special symbols such as *, `, ' in json.
 
    Return **only** the final JSON in this structure:
    {{
    "final_score": <An float score from 0 to 10>,
    "justification": 
        {{
            "overall_assessment": "<Provide a brief, high-level summary of why this score was given.>",
            "supported_claims": [
                {{
                    "claim": "<A specific claim extracted from the conclusion.>",
                    "evidence": "<State which Title(s) or part of the insight supports this claim. e.g., "Title 1 (Sales Trend 2020-2024)">"
                }}
            ],
            "unsupported_claims": [
                {{
                    "claim": "<A specific claim extracted from the conclusion that is NOT supported.>",
                    "reason": "<Explain why the provided charts/insight are insufficient. e.g., "No chart breaks down data by region.">"
                }}
            ]
        }},
    "recommendations": 
        {{
            "is_required": <true_or_false, true if score is not 10>,
            "suggestion": "<If the score is not 10, provide concrete suggestions on what charts are missing or should be replaced. e.g., "To fully support the conclusion, replace Title 5 (Employee Headcount) with a new chart titled "Regional Sales Breakdown" to verify the APAC claim." If score is 10, this should be "N/A".>"
        }}
    }}
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["conclusion","titles_1","titles_2","titles_3","titles_4","titles_5","titles_6","insight"],
    )
    chain = prompt | llm
    response = chain.invoke(input ={"conclusion":conclusion,"titles_1":titles[0],"titles_2":titles[1],"titles_3":titles[2],"titles_4":titles[3],"titles_5":titles[4],"titles_6":titles[5],"insight":insight})
    return response.content







