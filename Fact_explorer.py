from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage





def agent1_column_selector(schema, openai_key):
    llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", temperature=0,api_key = openai_key)
    col_select_template="""
    You are a data scientist tasked with selecting 4 columns from a given dataset to perform an effective exploratory data analysis (EDA).
    Here is the data schema: {schema}

    **Instructions**

    1. Carefully review the dataset schema provided. Each column includes metadata such as column name, data type (Categorical, Numerical, Temporal), and number of unique values.
    2. Select exactly 4 columns that together enable diverse and meaningful analysis. Follow these principles:
        - Include at least 1 numerical column (for quantitative analysis).
        - Include at least 2 categorical columns (for comparisons, grouping, or segmentation).
        - Optionally, include a temporal column if available (to analyze trends over time).
    3. Make sure the chosen set of 4 columns allows:
        - Identifying overall trends or distributions.
        - Comparing across groups (categories).
        - Drilling down into deeper insights (interactions between columns).

    **Constraints**
    - The first column must be categorical or temporal.
    - The second column must be categorical or temporal.

    **Output your answer in JSON format as follows**
    NEVER INCLUDE ```json```.Do not add other sentences after this json data.
    {{
    "key_columns": [
        {{
        "column": "<column name 1>",
        "properties": {{
            "dtype": "C",
            "num_unique_values": 5
        }}
        }},
        {{
        "column": "<column name 2>",
        "properties": {{
            "dtype": "C",
            "num_unique_values": 32
        }}
        }},
        {{
        "column": "<column name 3>",
        "properties": {{
            "dtype": "N",
            "num_unique_values": 12
        }},
        {{
        "column": "<column name 4>",
        "properties": {{
            "dtype": "N",
            "num_unique_values": 32
        }}
    ]

    """
    col_select_prompt = PromptTemplate(
        template=col_select_template,
        input_variables=["schema"]
    )
    col_select_chain = col_select_prompt | llm
    col_select_result = col_select_chain.invoke(input = {"schema":schema})
    return col_select_result.content

def agent2_fact_summarizer(data_name, main_column, facts,title, openai_key):

    llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", temperature=0.5, api_key=openai_key)
    summarize_facts_prompt_template = """
    You are a senior data analyst. You are examining a dataset named **{data_name}** with this column: **{main_column}**.
    Here are key facts extracted from this dataset:{facts}
    If provided, key facts are extracted from the chart whose title is: {title}
    **Your goal:**  
    Generate **5** focused, structured insights, using chain-of-thought reasoning.

    **Instructions for your internal reasoning (do NOT include these steps in your response):**   

    1. **Context & Exploration**  
    - Briefly restate what the dataset covers and the role of **{main_column}**.  
    - List 2 hypotheses or questions about how **{main_column}** might vary or relate to other columns.  

    2. **Deep Dive**  
    For each hypothesis or question:  
    - **Analyze** how **{main_column}** changes across values of one other column.  
    - **Highlight** any trends, group commonality, or group differences.  


    3. **Chain-of-Thought**  
    - Build your reasoning in 2-3 short bullet steps per insight.  
    - If chart title provided, think about what the chart emphasizes.
    - Use phrases like "Because…", "When comparing…", "We observe that…".  

    4. **Summarize**  
    - Choose key facts that are interesting observations and weave into an insight.
    - Each insight should be **one single** sentence.  
    - Order by importance or strength of insight.  
    - Do **not** include extra commentary or metadata.  

    **Constraints:**
    - Understand the values contained in {main_column}. NEVER use values that do not appear in the key facts.
    - If chart title provided, align insights with the chart's focus.
    - Drop the invalid facts by following the rules below:
        1. No self-comparison: reject if a fact compares a category to itself.
           e.g., "The TotalEmployed of Sex Female is 400.60 times more than that of Female when Sex is Female.".
           e.g., "The TotalEmployed of Sex Female is 7285 when Sex is Female.".
           e.g., "The Sex Female accounts for 9.67% of the TotalEmployed when Sex is Female.".
           e.g., "The TotalEmployed of Pacific Island Countries Tuvalu is 5.09 times more than that of Tuvalu when Pacific Island Countries is Tuvalu.".
        2. Don't compare/rank on a column that is fixed by a filter 
           e.g., "In the TotalEmployed ranking of different Sex(s), the top three Sex(s) are Female, Female, Female  when Sex is Female.".

    **Output requirements (ONLY return this; no reasoning or extra text):**  
    - A five-point list of one-sentence insights. 
    - No explanations, no bullet steps, no metadata.


    **Example output:**  
    1. Japan has the highest Sales when **OtherColumn** is "X".  
    2. The Sales in China are significantly lower than in the US.
    3. Sales in Germany are higher than in France when **OtherColumn** is "Y".  
    4. Sales in the UK show a unique trend compared to other countries.
    5. Sales in Canada are consistently higher than in the US across all categories.

    """
    prompt = PromptTemplate(
        template=summarize_facts_prompt_template,
        input_variables=["data_names", "main_column", "facts", "title"],
    )
    summarize_chain = prompt | llm
    summarize_insight = summarize_chain.invoke(input ={"data_name":data_name, "main_column":main_column, "facts":facts, "title":title})

    return summarize_insight.content

def agent3_s1q_generator(s1c1_insight, data_facts, column_1, key_columns, openai_key):
    llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", api_key=openai_key)
    eda_q_for_sec1_template="""
    You are a data-analysis assistant that drafts *exploratory questions* destined for visualization generation.
    Read the information founded from the current exploratory data analysis(EDA) carefully: {s1c1_insight}

    **Your Task**
    Generate **two** distinct follow-up questions that logically extend the current EDA based on the following data facts (observations already discovered):
    {data_facts}\n\n
    1. Total Pick **exactly two** facts that add new angles, and avoid redundancy with each other.
    2. For the first question, choose a fact and write one specific follow-up questions that:
        - In addition to the numerical column, the question must refer to this one column **ONLY**: {column_1}. 
        - The question should be high-level and **ONLY** mention column name rather than specific value.
        - Make them answerable with the existing dataset.
    3. For the second question, choose a fact and write one specific follow-up questions that:
        - Drills down into significance one revealed from {s1c1_insight}.
        - In addition to the numerical column, the question must mention this column: {column_1}.
        - Make them answerable with the existing dataset.
    4. Write a title for the chart **(no more than 12 words each)** based on the questions.
    
    **Constraints**
    - Never rewrite the data facts from {data_facts}.
    - Only use the columns from {key_columns}.

    **Unvalid Examples**:
    - Question 1: "How does the number of employed persons vary across Pacific Island Countries in different economic sectors?"
    Rationale: Mentions two columns ("Pacific Island Countries" and "economic sectors") instead of just one besides the numerical column (num of employed person).
    - Question 2: "What is the trend of employed persons in the Agriculture sector over the years?"
    Rationale: Mentions a specific value ("Agriculture sector") instead of just column names.

    **Example**
    Dataset columns: [gender, racial groups, math score, reading score, writing score]
    Follow-up questions:
    Question 1. How do the average math scores across different races?
    Rationale: 
    Follows rule #2: 
    - Focuses on one numerical column (math score) and one categorical column (gender).
    - High-level (doesn't mention specific values like "female" or "male").
    Helps reveal broad performance gaps between genders.
    Question 2. How does the reading score differ across gender and racial groups?  
    Rationale:
    Follows rule #3: 
    - Uses one numerical column (reading score), the required gender, and one other categorical column (racial groups).
    - Adds a second dimension for deeper drill-down.
    - Avoids redundancy by choosing a different numerical column (reading score) than the first question.


    **Output (exact JSON)**  
    NEVER INCLUDE ```json```.Do not add other sentences after this json data.
    {{
    "follow_up_questions": [
        {{
        "selected_facts": "<fact 1>",
        "question": "<Question 1>",
        "suggested_viz_title": "title for the chart",
        "suggested_viz_type": "<bar|line|pie|scatter|...>",
        "column": "<column name that the question is related to>",
        "rationale": "<Why this deepens the analysis>"
        }},
        {{
        "selected_facts": "<fact 2>",
        "question": "<Question 2>",
        "suggested_viz_title": "title for the chart",
        "suggested_viz_type": "<bar|line|pie|scatter|...>",
        "column": "<column name that the question is related to>",
        "rationale": "<Rationale>"
        }}
        
    ]
    }}"""
    eda_q_for_sec1_prompt = PromptTemplate(
        template=eda_q_for_sec1_template,
        input_variables=["s1c1_insight","data_facts","column_1","key_columns"]
    )
    eda_q_for_sec1_chain = eda_q_for_sec1_prompt | llm
    eda_q_for_sec1_result = eda_q_for_sec1_chain.invoke(input = {
                                                    "s1c1_insight":s1c1_insight,
                                                    "data_facts":data_facts,
                                                    "column_1": column_1,
                                                    "key_columns": key_columns                                        
                                                    })

    return eda_q_for_sec1_result.content

def agent4_s2q_generator(s2c1_insight, s1q1, s1q2, data_facts, column_2, key_columns, openai_key):
    llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", api_key=openai_key)
    eda_q_for_sec2_template="""
    You are a data-analysis assistant that drafts *exploratory questions* destined for visualization generation.
    Read the information founded from the current exploratory data analysis(EDA) carefully: {s2c1_insight}

    **Your Task**
    Generate **two** distinct follow-up questions that can make a comparison with this two questions "{s1q1}" and "{s1q2}" based on the following data facts (observations already discovered):
    {data_facts}\n\n
    1. Total Pick **exactly two** facts that add new angles, and avoid redundancy with each other.
    2. For the first question, choose a fact and write one specific follow-up questions that:
        - In addition to the numerical column, the question must refer to this one column **ONLY**: {column_2}. 
        - The question should be high-level and **ONLY** mention column name rather than specific value.
        - Make them answerable with the existing dataset.
    3. For the second question, choose a fact and write one specific follow-up questions that:
        - Drills down into significance one revealed from {s2c1_insight}.
        - In addition to the numerical column, the question must mention this column: {column_2}.
        - Make them answerable with the existing dataset.
    4. Write a title for the chart **(no more than 12 words each)** based on the questions.
    
    **Constraints**
    - Never rewrite th data facts from {data_facts}.
    - Only use the columns from {key_columns}.

    **Unvalid Examples**:
    - Question 1: "How does the number of employed persons vary across Pacific Island Countries in different economic sectors?"
    Rationale: Mentions two columns ("Pacific Island Countries" and "economic sectors") instead of just one besides the numerical column (num of employed person).
    - Question 2: "What is the trend of employed persons in the Agriculture sector over the years?"
    Rationale: Mentions a specific value ("Agriculture sector") instead of just column names.

    **Example**
    Dataset columns: [gender, racial groups, math score, reading score, writing score]
    Follow-up questions:
    Question 1. How do the average math scores across different races?
    Rationale: 
    Follows rule #2: 
    - Focuses on one numerical column (math score) and one categorical column (gender).
    - High-level (doesn't mention specific values like "female" or "male").
    Helps reveal broad performance gaps between genders.
    Question 2. How does the reading score differ across gender and racial groups?  
    Rationale:
    Follows rule #3: 
    - Uses one numerical column (reading score), the required gender, and one other categorical column (racial groups).
    - Adds a second dimension for deeper drill-down.
    - Avoids redundancy by choosing a different numerical column (reading score) than the first question.
    
    **Output (exact JSON)**
    NEVER INCLUDE ```json```.Do not add other sentences after this json data.
    {{
    "follow_up_questions": [
        {{
        "selected_facts": "<fact 1>",
        "question": "<Question 1>",
        "suggested_viz_title": "title for the chart",
        "suggested_viz_type": "<bar|line|pie|scatter|...>",
        "column": "<column name that the question is related to>",
        "rationale": "<Why this deepens the analysis>"
        }},
        {{
        "selected_facts": "<fact 2>",
        "question": "<Question 2>",
        "suggested_viz_title": "title for the chart",
        "suggested_viz_type": "<bar|line|pie|scatter|...>",
        "column": "<column name that the question is related to>",
        "rationale": "<Rationale>"
        }}
        
    ]
    }}"""
    eda_q_for_sec2_prompt = PromptTemplate(
        template=eda_q_for_sec2_template,
        input_variables=["s2c1_insight","s1q1","s1q2","data_facts","column_2","key_columns"]
    )
    eda_q_for_sec2_chain = eda_q_for_sec2_prompt | llm
    eda_q_for_sec2_result = eda_q_for_sec2_chain.invoke(input = {
                                                                "s2c1_insight":s2c1_insight,
                                                                "s1q1": s1q1,
                                                                "s1q2": s1q2,
                                                                "data_facts":data_facts,
                                                                "column_2": column_2,
                                                                "key_columns": key_columns
                                                            })
    return eda_q_for_sec2_result.content

def agent5_s3q_generator(s1c1_insight, question_1, question_2, question_3, question_4, key_columns, openai_key):
    llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", api_key=openai_key)
    eda_q_for_sec3_template="""
    You are a data-analysis assistant that drafts *exploratory questions* destined for visualization generation.
    Read the information founded from the current exploratory data analysis(EDA) carefully: {s1c1_insight}


    **Your Task**
    Generate **two** distinct follow-up questions that logically extend the current EDA:
    1. Understand the given information from the current EDA.
    2. Identify two entities that worth to explore.
    3. Write specific follow-up questions **(no more than 15 words each)** for these two entities that:
        - The question drill down to reveal the pattern within this entity.
        - In addition to the numerical column, the question must refer to two column **ONLY**.
        - Ensure the two questions are use **the same** combination of columns.
        - Make them answerable with the existing dataset.
    4. Write a title for the chart **(no more than 12 words each)** based on the questions.


    **Constraints**
    - Never raise the same or similar questions as following:  
        1.{question_1}
        2.{question_2}
        3.{question_3}
        4.{question_4}
    - Only use the columns from {key_columns}.



    **Output (exact JSON)**  
    Do not INCLUDE ```json```.Do not add other sentences after this json data.
    {{
    "follow_up_questions": [
        {{
        "question": "<Question 1>",
        "suggested_viz_title": "title for the chart",
        "suggested_viz_type": "<bar|line|pie|scatter|...>",
        "column": "<column name that the question is related to>",
        "entity": "<entity name that is used in the question 1>",
        "rationale": "<Why this deepens the analysis>"
        }},
        {{
        "question": "<Question 2>",
        "suggested_viz_title": "title for the chart",
        "suggested_viz_type": "<bar|line|pie|scatter|...>",
        "column": "<column name that the question is related to>",
        "entity": "<entity name that is used in the question 2>",
        "rationale": "<Rationale>"
        }}
        
    ]
    }}"""
    eda_q_for_sec3_prompt = PromptTemplate(
        template=eda_q_for_sec3_template,
        input_variables=["s1c1_insight","question_1","question_2","question_3","question_4","key_columns"]
    )
    eda_q_for_sec3_chain = eda_q_for_sec3_prompt | llm
    eda_q_for_sec3_result = eda_q_for_sec3_chain.invoke(input = {                                         
                                                    "s1c1_insight":s1c1_insight,
                                                    "question_1": question_1,
                                                    "question_2": question_2,
                                                    "question_3": question_3,
                                                    "question_4": question_4,
                                                    "key_columns": key_columns
                                                    })

    return eda_q_for_sec3_result.content