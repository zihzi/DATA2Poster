from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate





def agent11_vis_recommender(conclusion, title_1, title_2, insight, column, openai_key):
    llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14",temperature=0.7, api_key = openai_key)
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
    **Instructions**
    (A) The following compares the consistency between the conclusion and the chart titles.
        Detect which kind of claim types are in the conclusion, then check if at least one title implies a chart that could evidence it:
        Claim Type 1. Trend
            Possible claim word in conclusion: "increase", "decrease", "rise", "fall", "trend", "fluctuate", "peak", "decline"
            What a supporting title should imply:"over time", "year", "month", "trend", "since", "through"
        Claim Type 2.  Proportion
            Possible claim word in conclusion: "most", "concentrated", "percentage", "proportion", "dominate", "dominance", "underscore", "leading", "highest", "lowest", "larger"
            What a supporting title should imply: "of", "in", "out of", "percentage", "share", "proportion"
        Claim Type 3. Distribution
            Possible claim word in conclusion: "distribution", "spread", "range", "variation", "skew"
            What a supporting title should imply: "distribution", "by [category]", "across [category]", "among [category]"
        Claim Type 4. Correlation
            Possible claim word in conclusion: "correlation", "relationship", "associated", "linked"
            What a supporting title should imply: "vs", "compared to", "correlation", "relationship"

    (B) Scoring per chart (0-100):
        (a) Variables & Entities Match — 40 pts
           (1) Measure alignment (0/8/16)
            0: Different measure (e.g., count vs revenue; share vs total).
            8: Same family but ambiguous (e.g., "sales performance" vs "total sales").
            16: Exact measure match or clear synonym (total sales, sales (sum)).
           (2) Primary dimension alignment (0/8/16)
            0: Dimension missing.
            8: Dimension present but not identical (e.g., "by vendor" vs "by retailer").
            16: Exact dimension (e.g., by retailer, by region).
           (3) Aggregation/statistic alignment (0/2/4)
            0: Mismatch (claim about average; title indicates total).
            2: Aggregation implied but vague.
            4: Explicitly aligned (avg/median/sum/rate/share clearly matches claim).
           (4) Scope/subset alignment (0/2/4)
            0: Scope conflict (claim on total, title on subcategory like "Men's Apparel").
            2: Scope unclear.
            4: Scope consistent (both overall or both same subset).
        Subtotal max: 40
        (b) Relation / Claim-Type Match — 40 pts
           (1) Primary type alignment (0/10/20)
            0: Title type cannot evidence the claim (e.g., distribution title for correlation claim).
            10: Partial alignment (comparison title for a proportion claim without "share/%").
            20: Strong alignment (trend claim <-> time-in-title; proportion <-> share/percentage).
           (2) Essential evidence cues present (0/7/14)
            0: Missing required cues (trend without any time; correlation without "vs/relationship").
            10: Some cues present but incomplete (time window unspecified; "by group" present but measure unclear).
            20: All cues present for the claim type.
            Subtotal max: 40
        (c) Granularity & Specificity — 20 pts
           (3) Group/subgroup specificity (0/5/10)
            0: Claim about subgroup gap; titles lack group field.
            5: Group hinted but vague.
            10: Exact subgroup named (e.g., "by gender/retailer type").
           (4) Unit/level consistency (0/5/10)
            0: Level mismatch (monthly vs annual; per-store rate vs total).
            5: Ambiguous unit.
            10: Explicitly consistent (unit and level match the claim).
            Subtotal max: 20

    (C) Pick verdict:
        1.  score more than 90: supported
        2.  score between 60 and 89: partially supported
        3.  score less than 60: unsupported

    (D) Pick replace:
        If unsupported: replace the lower-scoring chart (If tie, replace chart_2)
        If partially supported: replace the chart with the weaker alignment to the main claim type detected.

    (E) Replacement recommender:
        1. Identify Possible claim words in the conclusion (from A).
        2. Identify the main claim type (from A).
        3.  Map the main claim type to a viz:
            Trend: line (x=time, y=agg(metric), optional color=group)
            Proportion: pie
            Relationship: scatter
            Distribution: histogram (grouped by category)
        4.  Also provide a concrete query that the recommended chart would answer:
            Trend: "How did {{metric}} change over {{time}} by {{group}}?"
            Proportion: "What share of {{metric}} comes from {{group}}?"
            Relationship: "What is the correlation between {{x}} and {{y}}?"
            Distribution: "What is the distribution of {{metric}} across {{group}}?"
    
    **Constraints**
    - Recommend a chart that can strengthen the main claim and is not the same claim type as the kept chart.
    - The query should only contain the following columns: {column}.
    **Example**
    Conclusion:
    "CineHub clearly dominates total viewing hours across platforms, with 4.8 billion hours leading well above StreamNow and BingeBox, 
    highlighting the strength of niche film-centric services over general tech giants like OmniPrime and MacroTube. 
    This suggests that focused curation and genre specialization drive superior engagement despite broader catalogs."

    Title 1: "Total Viewing Hours Distribution Across Streaming Platforms"
    Title 2: "Documentary Viewing Hours by Platform"

    Output:
    {{
    "vis_check":
    [
    {{
        "verdict": "partially_supported",
        "replace": "chart_2",
        "reason": "Chart 1 implies platform ranking for total viewing hours; Chart 2 is a documentary subcategory, but the conclusion never mentions documentary viewing hours.",
        "recommendation": {{
            "chart_type": "pie",
            "query":"What share of total viewing hours comes from niche film-centric platforms versus general platforms?",
            "revised_title": "Viewing Hours Share by Platform",
            "explanation": "Since \"dominates\", which is the possible claim word of \"proportion\", has appeared in the conclusion, 
                            a pie chart showing the proportion of total viewing hours from niche film-centric platforms (like CineHub) versus general platforms (like OmniPrime and MacroTube) 
                            would directly support the conclusion's statement -'CineHub clearly **dominates** total viewing hours across platforms'. And there will be no redundancy with Chart 1.(which claim type is \"distribution\")"
        }}    
    }}
    ]
    }}
    **Output (JSON)**
    Do not INCLUDE ```json```.Do not add other sentences after this json data.
    Return **only** the final JSON in this structure:
    {{
    "vis_check":
    [
    {{
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
    llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", temperature=0.7, api_key = openai_key)
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
    Given the conclusion, the titles of six charts, and insight derived from the six charts, judge whether the conclusion is supported from those charts. 
    Score 0-100 on how well the conclusion is supported by the charts.
    **Instructions**
    (A) The following compares the consistency between the conclusion and the chart titles.
        Detect which kind of claim types are in the conclusion, then check if at least one title implies a chart that could evidence it:
        Claim Type 1. Trend
            Possible claim word in conclusion: "increase", "decrease", "rise", "fall", "trend", "fluctuate", "peak", "decline"
            What a supporting title should imply:"over time", "year", "month", "trend", "since", "through"
        Claim Type 2.  Proportion
            Possible claim word in conclusion: "most", "concentrated", "percentage", "proportion", "dominate", "dominance", "underscore", "leading", "highest", "lowest", "larger"
            What a supporting title should imply: "of", "in", "out of", "percentage", "share", "proportion"
        Claim Type 3. Distribution
            Possible claim word in conclusion: "distribution", "spread", "range", "variation", "skew"
            What a supporting title should imply: "distribution", "by [category]", "across [category]", "among [category]"
        Claim Type 4. Correlation
            Possible claim word in conclusion: "correlation", "relationship", "associated", "linked"
            What a supporting title should imply: "vs", "compared to", "correlation", "relationship"
    (B) Scoring (0-100):
        (a) Variables & Entities Match — 40 pts
           (1) Measure alignment (0/8/16)
            0: Different measure (e.g., count vs revenue; share vs total).
            8: Same family but ambiguous (e.g., "sales performance" vs "total sales").
            16: Exact measure match or clear synonym (total sales, sales (sum)).
           (2) Primary dimension alignment (0/8/16)
            0: Dimension missing.
            8: Dimension present but not identical (e.g., "by vendor" vs "by retailer").
            16: Exact dimension (e.g., by retailer, by region).
           (3) Aggregation/statistic alignment (0/2/4)
            0: Mismatch (claim about average; title indicates total).
            2: Aggregation implied but vague.
            4: Explicitly aligned (avg/median/sum/rate/share clearly matches claim).
           (4) Scope/subset alignment (0/2/4)
            0: Scope conflict (claim on total, title on subcategory like "Men's Apparel").
            2: Scope unclear.
            4: Scope consistent (both overall or both same subset).
        Subtotal max: 40
        (b) Relation / Claim-Type Match — 40 pts
           (1) Primary type alignment (0/10/20)
            0: Title type cannot evidence the claim (e.g., distribution title for correlation claim).
            10: Partial alignment (comparison title for a proportion claim without "share/%").
            20: Strong alignment (trend claim <-> time-in-title; proportion <-> share/percentage).
           (2) Essential evidence cues present (0/7/14)
            0: Missing required cues (trend without any time; correlation without "vs/relationship").
            10: Some cues present but incomplete (time window unspecified; "by group" present but measure unclear).
            20: All cues present for the claim type.
            Subtotal max: 40
        (c) Granularity & Specificity — 20 pts
           (3) Group/subgroup specificity (0/5/10)
            0: Claim about subgroup gap; titles lack group field.
            5: Group hinted but vague.
            10: Exact subgroup named (e.g., "by gender/retailer type").
           (4) Unit/level consistency (0/5/10)
            0: Level mismatch (monthly vs annual; per-store rate vs total).
            5: Ambiguous unit.
            10: Explicitly consistent (unit and level match the claim).
            Subtotal max: 20

    **Constraints**
    - It is not necessary to account for all potential claim types in the conclusion; evaluation should be limited to the claim types that are explicitly present.
    **Output (JSON)**
    Do not INCLUDE ```json```.Do not add other sentences after this json data.
    Return **only** the final JSON in this structure:
    {{
    "final_score": <score from 0-100>,
    "justification": <brief explanation of your score>
    }}
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["conclusion","titles_1","titles_2","titles_3","titles_4","titles_5","titles_6","insight"],
    )
    chain = prompt | llm
    response = chain.invoke(input ={"conclusion":conclusion,"titles_1":titles[0],"titles_2":titles[1],"titles_3":titles[2],"titles_4":titles[3],"titles_5":titles[4],"titles_6":titles[5],"insight":insight})
    return response.content







