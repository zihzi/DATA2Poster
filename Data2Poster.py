import streamlit as st
import pandas as pd
import re
import base64
import json
import itertools
import altair as alt
from collections import Counter
from operator import itemgetter
from dataclasses import asdict
from nl4dv import NL4DV
from dataSchema_builder import get_column_properties
from data_cleaning import clean_csv_data
from insight_generation.dataFact_scoring import score_importance
from insight_generation.main import generate_facts
from selfAugmented_thinker import self_augmented_knowledge
from question_evaluator import expand_questions
from vis_generator import agent_improve_vis, agent_2_improve_code, agent_1_generate_code,agent_4_validate_spec, agent_consistent
import vl_convert as vlc
from pathlib import Path
from poster_generator_test import create_pdf

# Import langchain modules
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
import random



    
# Set page config
st.set_page_config(page_icon="figure/analytics.png",layout="wide",page_title="DATA2Poster")
col1, col2 = st.columns([1,23])
with col1:
    st.write('')
    st.image('figure/analytics.png', width=50)
with col2:
    st.title("DATA2Poster")

# List to hold datasets
if "datasets" not in st.session_state:
    datasets = {}
    # Preload datasets
    datasets["adidas_sale"] = pd.read_csv("data/adidas_sale.csv")
    datasets["bike_sharing_day"] = pd.read_csv("data/bike_sharing_day.csv")
    datasets["crime_safety_1"] = pd.read_csv("data/crime_safety_1.csv")
    datasets["crime_safety_2"] = pd.read_csv("data/crime_safety_2.csv")
    datasets["fashion_products"] = pd.read_csv("data/fashion_products.csv")
    datasets["restaurant_dishes"] = pd.read_csv("data/restaurant_dishes.csv")
    datasets["Sleep_health_and_lifestyle_dataset"] = pd.read_csv("data/Sleep_health_and_lifestyle_dataset.csv")
    datasets["starbucks"] = pd.read_csv("data/starbucks.csv")
    datasets["Indian_Kids_Screen_Time_1"] = pd.read_csv("data/Indian_Kids_Screen_Time_1.csv")
    datasets["Indian_Kids_Screen_Time_2"] = pd.read_csv("data/Indian_Kids_Screen_Time_2.csv")
    datasets["Indian_Kids_Screen_Time_3"] = pd.read_csv("data/Indian_Kids_Screen_Time_3.csv")
    datasets["volcano"] = pd.read_csv("data/volcano.csv")
    datasets["shopping_trends"] = pd.read_csv("data/shopping_trends.csv")




    st.session_state["datasets"] = datasets
else:
    # Use the list already loaded
    datasets = st.session_state["datasets"]
# Set left sidebar content
with st.sidebar:
    # Set area for user guide
    with st.expander("UserGuide"):
         st.write("""
            1. Input your OpenAI Key.
            2. Select dataset from the list below then you can start.
        """)
    # Set area for OpenAI key
    openai_key = st.text_input(label = "🔑 OpenAI Key:", help="Required for models.",type="password")
         
    # First we want to choose the dataset, but we will fill it with choices once we've loaded one
    dataset_container = st.empty()

    # default the radio button to the newly added dataset
    index_no = len(datasets)-1
    # Radio buttons for dataset choice
    chosen_dataset = dataset_container.radio("👉 Choose your data :",datasets.keys(),index=index_no)
    # Save column names of the dataset for gpt to generate questions
    head = datasets[chosen_dataset].columns
    # 10 rows of chosen_dataset for gpt to generate vlspec
    sample_data = datasets[chosen_dataset].head(10)

# Get the schema of the chosen dataset    
chosen_data_schema = get_column_properties(datasets[chosen_dataset])
st.write("**Dataset Schema:**",chosen_data_schema)



# Calculate the importance score of data facts
score_importance(chosen_data_schema)

# Session state variables for workflow

if "bt_try" not in st.session_state:
    st.session_state["bt_try"] = ""
if "base_fact" not in st.session_state:
    st.session_state["base_fact"] = []
if "fact" not in st.session_state:
    st.session_state["sub_fact"] = []




# page content 
st.write("Let's explore your data!✨")
try_true = st.button("Try it out!") 

# Use NL4DV to generate chosen_dataset's summary
nl4dv_instance = NL4DV(data_value = datasets[chosen_dataset])
summary = nl4dv_instance.get_metadata()


# Get 'CNC', 'TNC', 'TNT', 'CNN', 'TNN' for calculating data facts


def organize_by_dtype_combinations(data, desired_combinations):
    """
    Organize data by dtype combinations, showing columns in the dict for each combination
    with column names as keys and their dtypes as values.
    
    This implementation prevents duplicates by tracking combinations via frozensets.
    
    Args:
        data: List of column objects with properties including dtype
        desired_combinations: List of dtype pattern strings (e.g., 'CNC', 'CNN')
        
    Returns:
        List of combination objects with unique combinations
    """
    # Get all columns by their data types
    dtype_columns = {}
    for item in data:
        dtype = item["properties"]["dtype"]
        if dtype not in dtype_columns:
            dtype_columns[dtype] = []
        dtype_columns[dtype].append(item["column"])
    
    # Count occurrences of each dtype in each pattern
    pattern_dtype_counts = {}
    for pattern in desired_combinations:
        pattern_dtype_counts[pattern] = Counter(pattern)
    
    # Initialize results and track unique combinations
    results = []
    processed_combinations = set()
    
    # Generate all combinations for each pattern
    for pattern in desired_combinations:
        # Skip patterns with dtypes that don't exist in our data
        valid_pattern = True
        for dtype, count in pattern_dtype_counts[pattern].items():
            if dtype not in dtype_columns or len(dtype_columns[dtype]) < count:
                valid_pattern = False
                break
        
        if not valid_pattern:
            continue
        
        # For each dtype, get all possible combinations of the required count
        dtype_combinations = {}
        for dtype, count in pattern_dtype_counts[pattern].items():
            dtype_combinations[dtype] = list(itertools.combinations(dtype_columns[dtype], count))
        
        # Generate the Cartesian product of all dtype combinations
        dtype_keys = list(dtype_combinations.keys())
        
        # Generate and process all combinations
        for combo_parts in itertools.product(*[dtype_combinations[dtype] for dtype in dtype_keys]):
            # Create a mapping of columns to dtypes
            columns_dtype_mapping = {}
            
            # Iterate through the combination elements
            for dtype_idx, columns_tuple in enumerate(combo_parts):
                dtype = dtype_keys[dtype_idx]
                
                # Add each column with its dtype
                for col in columns_tuple:
                    columns_dtype_mapping[col] = dtype
            
            # Create a unique key for this combination to detect duplicates
            combo_key = frozenset(columns_dtype_mapping.items())
            
            # Only add if we haven't seen this exact mapping before
            if combo_key not in processed_combinations:
                processed_combinations.add(combo_key)
                results.append({
                    "combinations": pattern,
                    "columns_dtype_mapping": columns_dtype_mapping
                })
    
    return results

# Remove duplicates data facts
def remove_duplicates(strings: list) -> list:
    seen = set()
    unique = []
    for s in strings:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique
# load a prompt from a file
def load_prompt_from_file(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        return file.read()
    
# preprocess the code generated by GPT
def preprocess_json(code: str, count: str) -> str:
    """Preprocess code to remove any preamble and explanation text"""

    code = code.replace("<imports>", "")
    code = code.replace("<stub>", "")
    code = code.replace("<transforms>", "")

    # remove all text after chart = plot(data)
    if "chart = plot(data)" in code:
        index = code.find("chart = plot(data)")
        if index != -1:
            code = code[: index + len("chart = plot(data)")]

    if "```" in code:
        pattern = r"```(?:\w+\n)?([\s\S]+?)```"
        matches = re.findall(pattern, code)
        if matches:
            code = matches[0]

    if "import" in code:
        # return only text after the first import statement
        index = code.find("import")
        if index != -1:
            code = code[index:]

    code = code.replace("```", "")
    if "chart = plot(data)" not in code:
        code = code + "\nchart = plot(data)"
    if "def plot" in code:
        index = code.find("def plot")
        code = code[:index] + f"data = pd.read_csv('data/{chosen_dataset}.csv')\n\n" + code[index:] + f"\n\nchart.save('DATA2Poster_img/image_{count}.png')\n\nchart_json = chart.to_json()\n\nwith open('DATA2Poster_json/vega_lite_json_{count}.json', 'w') as f:\n\n f.write(chart_json)"
        exec(code)
    return code
# For code generated by nl4dv
def preprocess_json_2(code: str, count: str) -> str:
    """Preprocess code to remove any preamble and explanation text"""

    code = code.replace("<imports>", "")
    code = code.replace("<stub>", "")
    code = code.replace("<transforms>", "")

    # remove all text after chart = plot(data)
    if "chart = plot(data)" in code:
        index = code.find("chart = plot(data)")
        if index != -1:
            code = code[: index + len("chart = plot(data)")]

    if "```" in code:
        pattern = r"```(?:\w+\n)?([\s\S]+?)```"
        matches = re.findall(pattern, code)
        if matches:
            code = matches[0]

    if "import" in code:
        # return only text after the first import statement
        index = code.find("import")
        if index != -1:
            code = code[index:]

    code = code.replace("```", "")
    if "chart = plot(data)" not in code:
        code = code + "\nchart = plot(data)"
    if "def plot" in code:
        index = code.find("def plot")
        code = code[:index] + f"data = pd.read_csv('data/{chosen_dataset}.csv')\n\n" + code[index:] + f"\n\nchart.save('nl4DV_img/image_{count}.png')\n\nchart_json = chart.to_json()\n\nwith open('nl4DV_json/vega_lite_json_{count}.json', 'w') as f:\n\n f.write(chart_json)"
        exec(code)
    return code
def expand_vis_query(query,columns, openai_key):
    """Expand the query using OpenAI API"""
    llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", temperature=0, api_key=openai_key)
    prompt = PromptTemplate(
        template="""
                You are an expert data visualization analyst. 
                **Task**
                Analyze a natural language query and expand it about creating a visualization and expand it into a comprehensive, detailed specification that can guide the creation of an effective chart.
                **Instructions**
                When given a natural language visualization query, analyze it thoroughly and provide a detailed expansion covering all aspects needed to create the visualization.
                Think step by step and consider the following aspects:          
                Step 1. Query Understanding
                - Intent: Summarize what the user wants to visualize
                - Key Elements: Identify the main data components mentioned
                - Implicit Requirements: Note any unstated but implied needs
                Step 2. Data Requirements
                - Data Types: Specify categorical, numerical, temporal, geographical, etc.
                - Data Preprocessing: Any transformations, aggregations, or calculations needed
                Step 3. Visualization Specification
                - Chart Type: Recommend **one** that is the most appropriate visualization type 
                - Visual Encoding: Specify how data should map to visual elements (x-axis, y-axis, color, size, shape, etc.)
                - Scales and Axes: Define axis types, scales (linear, log, categorical), ranges, and labels
                Step 4. Write a comprehensive, detailed specification that can guide the creation of an effective chart.

                Now analyze the following natural language visualization query and provide a comprehensive expansion following the instruction above:
                Query: {query}
                Dataset columns available: {columns}  
                **Output ONLY the comprehensive detailed specification without intermediate step. NO extra commentary. "The expanded query is ..."**
               

                    """,
        input_variables=["query","columns"],
    )
    
    chain = prompt | llm
    expanded_query = chain.invoke(input={"query": query,"columns": columns})
    return expanded_query.content
def load_json(json_file):
    with open(json_file, "r", encoding="utf-8") as fh:

        return json.load(fh)
 
# Check if the user has tried and entered an OpenAI key
api_keys_entered = True  # Assume the user has entered an OpenAI key
if try_true or (st.session_state["bt_try"] == "T"):
    if not openai_key.startswith("sk-"):
                st.error("Please enter a valid OpenAI API key.")
                api_keys_entered = False

    st.session_state["bt_try"] = "T"
    
    
    if api_keys_entered:
        # use GPT as llm
        llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", temperature=0,api_key = openai_key)
        # use OpenAIEmbeddings as embedding model
        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key = openai_key)
        # Create a vector store
               


        data_list = load_json('vis_corpus.json')
        docs = [Document(page_content=json.dumps(item)) for item in data_list]
        vectorstore = FAISS.from_documents(
        docs,
        OpenAIEmbeddings(model="text-embedding-3-small", api_key = openai_key),
        )
      
                
        
        # To generate facts and questions by user-selected column
        # For user to select a column
        st.write("Select a column:")
        selected_column = st.radio("Select one column:", head, index=None, label_visibility="collapsed")            
        skip_true = st.button("Skip")
    
        user_selected_column = ""
        if skip_true:
            llm_select_column_template = load_prompt_from_file("prompt_templates/llm_select_column.txt")
            prompt_select_column = PromptTemplate(
                                    template=llm_select_column_template,
                                    input_variables=["chosen_dataset", "chosen_data_schema"],
                                    
                                    )
            
            chain_llm_selected_column = prompt_select_column | llm 
            selected_column_from_gpt = chain_llm_selected_column.invoke(input = {"chosen_dataset":chosen_dataset, "chosen_data_schema":chosen_data_schema})
            user_selected_column = selected_column_from_gpt.content
            st.write(f'**Selected column:**', f'**{user_selected_column}**')
        elif selected_column: 
            user_selected_column = selected_column
        # Call GPT to select columns related to user-selected column in JSON format 
        if skip_true or selected_column:
            
            facts_list = []
            # USE ALL COLUMNS TO GENERATE FACTS#####################################################################
            desired_combinations = ['CNC', 'TNC', 'TNT', 'CNN', 'TNN']

            # Get and print results
            output = organize_by_dtype_combinations(chosen_data_schema, desired_combinations)

            # Categorization_facts ? to be fixed, not suitable for dataset type
            # for item in output:
            #     for key, value in item["columns_dtype_mapping"].items():
            #         if value == "C":
            #             ratio = round(datasets[chosen_dataset][key].value_counts()[0]/len(datasets[chosen_dataset]),3)*100
            #             focus = datasets[chosen_dataset][key].value_counts().idxmax()
            #             facts_list.append({"content":f"In {key}, {focus} accounts for the largest proportion ({ratio}%).","score":1})
            base_fact = []
            for item in output:
                breakdown = []  
                measure = [] 
                for key, value in item["columns_dtype_mapping"].items():
                    if value == "C" or value == "T":
                        breakdown.append(key)
                    else:
                        measure.append(key)
            
                
                # list all possible combinations of breakdown and measure to generate facts
                if len(breakdown) == 2: 
                    
                    if item["columns_dtype_mapping"][breakdown[0]] == "C":
                        facts_1 = generate_facts(
                                        dataset=Path(f"data/{chosen_dataset}.csv"),
                                        breakdown=breakdown[0],
                                        measure=measure[0],
                                        series=breakdown[1],
                                        breakdown_type="C",
                                        measure_type="N",
                                        with_vis=False,
                                    )
                        for fact in facts_1:
                            facts_list.append({"content":fact["content"], "score":fact["score_C"]})
                        facts_2 = generate_facts(
                                        dataset=Path(f"data/{chosen_dataset}.csv"),
                                        breakdown=breakdown[0],
                                        measure=measure[0],
                                        series=None,
                                        breakdown_type="C",
                                        measure_type="N",
                                        with_vis=False,
                                    )
                        for fact in facts_2:
                            facts_list.append({"content":fact["content"], "score":fact["score_C"]})                                    
                    elif item["columns_dtype_mapping"][breakdown[0]] == "T":
                        facts_1 = generate_facts(
                                        dataset=Path(f"data/{chosen_dataset}.csv"),
                                        breakdown=breakdown[0],
                                        measure=measure[0],
                                        series=breakdown[1],
                                        breakdown_type="T",
                                        measure_type="N",
                                        with_vis=False,
                                    )
                        for fact in facts_1:
                            facts_list.append({"content":fact["content"], "score":fact["score_C"]})
                        facts_2 = generate_facts(
                                        dataset=Path(f"data/{chosen_dataset}.csv"),
                                        breakdown=breakdown[0],
                                        measure=measure[0],
                                        series=None,
                                        breakdown_type="T",
                                        measure_type="N",
                                        with_vis=False,
                                    )
                        for fact in facts_2:
                            facts_list.append({"content":fact["content"], "score":fact["score_C"]})
                                        
                if len(measure) == 2:
                    if item["columns_dtype_mapping"][breakdown[0]] == "C":
                        facts_1 = generate_facts(
                            dataset=Path(f"data/{chosen_dataset}.csv"),
                            breakdown=breakdown[0],
                            measure=measure[0],
                            measure2=measure[1],
                            series=None,
                            breakdown_type="C",
                            measure_type="NxN",
                            with_vis=False,
                        )
                        for fact in facts_1:
                            facts_list.append({"content":fact["content"], "score":fact["score_C"]})
                        facts_2 = generate_facts(
                                dataset=Path(f"data/{chosen_dataset}.csv"),
                                breakdown=breakdown[0],
                                measure=measure[0],
                                series=None,
                                breakdown_type="C",
                                measure_type="N",
                                with_vis=False,
                            )
                        for fact in facts_2:
                            facts_list.append({"content":fact["content"], "score":fact["score_C"]})
                        facts_3 = generate_facts(
                                dataset=Path(f"data/{chosen_dataset}.csv"),
                                breakdown=breakdown[0],
                                measure=measure[1],
                                series=None,
                                breakdown_type="C",
                                measure_type="N",
                                with_vis=False,
                            )
                        for fact in facts_3:
                            facts_list.append({"content":fact["content"], "score":fact["score_C"]})
                    elif item["columns_dtype_mapping"][breakdown[0]] == "T": 
                        facts_1 = generate_facts(
                                dataset=Path(f"data/{chosen_dataset}.csv"),
                                breakdown=breakdown[0],
                                measure=measure[0],
                                measure2=measure[1],
                                series=None,
                                breakdown_type="T",
                                measure_type="NxN",
                                with_vis=False,
                            )
                        for fact in facts_1:
                            facts_list.append({"content":fact["content"], "score":fact["score_C"]})  
                        facts_2 = generate_facts(
                                dataset=Path(f"data/{chosen_dataset}.csv"),
                                breakdown=breakdown[0],
                                measure=measure[0],
                                series=None,
                                breakdown_type="T",
                                measure_type="N",
                                with_vis=False,
                            )
                        for fact in facts_2:
                            facts_list.append({"content":fact["content"], "score":fact["score_C"]})
                        facts_3 = generate_facts(
                                dataset=Path(f"data/{chosen_dataset}.csv"),
                                breakdown=breakdown[0],
                                measure=measure[1],
                                series=None,
                                breakdown_type="T",
                                measure_type="N",
                                with_vis=False,
                            )
                        for fact in facts_3:
                            facts_list.append({"content":fact["content"], "score":fact["score_C"]})
            if "fact" in st.session_state:
                st.session_state["sub_fact"] = []
            if "base_fact" in st.session_state:
                st.session_state["base_fact"] = []
            # Remove duplicates from facts_list
            contents = [fact["content"] for fact in facts_list]
                # Ranking facts by score
            facts_list = sorted(facts_list, key=itemgetter('score'), reverse=True)  
            seen = set()
            seen_1 = set()
            for item in facts_list:
                if f"when {user_selected_column}" in item["content"] and item["content"] not in seen:
                    seen.add(item["content"])
                    st.session_state["sub_fact"].append(item["content"])
                elif f"{user_selected_column}" in item["content"] and "when" in item["content"] and item["content"] not in seen:
                    seen_1.add(item["content"])
                    st.session_state["sub_fact"].append(item["content"])
                elif item["content"] != "No fact." and "when" not in item["content"] and item["content"] not in seen_1 :
                        seen_1.add(item["content"])
                        st.session_state["base_fact"].append(item["content"])
            # Randomly select 100 facts
            # if len(st.session_state["sub_fact"]) > 100:
            #     st.session_state["sub_fact"] = random.sample(st.session_state["sub_fact"], 100)
            st.write("Base Facts:",st.session_state["base_fact"])
            st.write("Subspace Facts:",st.session_state["sub_fact"])

            base_col_com_set = set()
            base_fact_col_com = []
            for c in chosen_data_schema:
                if c["properties"]["dtype"] == "C":
                    for c2 in chosen_data_schema:
                        if c2["properties"]["dtype"] == "N" and (c["column"], c2["column"]) not in base_col_com_set:
                            base_col_com_set.add((c["column"], c2["column"]))
                            base_fact_col_com.append({"x": c["column"], "y": c2["column"]})
                elif c["properties"]["dtype"] == "N" :
                    for c2 in chosen_data_schema:
                        if c2["column"] != c["column"] and (c2["column"], c["column"]) not in base_col_com_set:
                            base_col_com_set.add((c2["column"], c["column"]))
                            base_fact_col_com.append({"x": c2["column"], "y": c["column"]})
            st.write("Base Facts Column Combinations:",base_fact_col_com)
                
            # Index for base chart  
            b_idx = 1
            img_num = 1
            chart_title = []
            chart_query = []
            with open ("json_schema/base_fact.json", "r") as f:
                base_fact_json = json.load(f)
            # Randomly select 3 base facts to generate base charts
            new_base_fact_col_com = random.sample(base_fact_col_com, 2)
            for bf in new_base_fact_col_com:
                # base_fact_json["data"]["url"] = f"https://raw.githubusercontent.com/zihzi/DATA2Poster/refs/heads/main/data/{chosen_dataset}.csv"
                # base_fact_json["encoding"]["x"]["field"] = bf["x"]
                # base_fact_json["encoding"]["x"]["axis"]["title"] = bf["x"]
                # base_fact_json["transform"][0]["aggregate"][0]["field"] = bf["y"]
                # base_fact_json["transform"][0]["groupby"][0] = bf["x"]
                # base_fact_json["encoding"]["y"]["axis"]["title"] = bf["y"]   
                # base_fact_json["encoding"]["color"]["field"] = bf["x"]                
                # base_fact_json["title"]["text"] = f"{bf['y']} across {bf['x']}"
                # if "year" in bf["x"].lower():
                #     base_fact_json["encoding"]["x"]["sort"] = "x"
                # base_png = vlc.v(vl_spec=base_fact_json,scale=3.0)
                # with open(f"DATA2Poster_img/base/base_img_{b_idx}.png", "wb") as f:
                #     f.write(base_png)
                # st.image(f"DATA2Poster_img/base/base_img_{b_idx}.png", caption="Base Chart")
                # Aggregate total Y(ex.employed persons) by X(ex.sex)
                agg = datasets[chosen_dataset].groupby(bf["x"])[bf["y"]].sum().reset_index()
                # b_idx += 1
            
            # To iterate through the base facts and generate first-level follow-up questions
            # for i in range(1,b_idx):
                # binary_base_chart     = open(f"DATA2Poster_img/base/base_img_{i}.png", 'rb').read()  
                # base64_utf8_base_chart = base64.b64encode(binary_base_chart).decode('utf-8')
                # base_img_url = f'data:image/png;base64,{base64_utf8_base_chart}' 

                # Use llm to generate the base chart description 
                bc_des_prompt = """
                                    You are a data visualization expert.

                                    Below is the raw data that will populate a **bar chart**:

                                    {data_table}\n\n
                                    
                                    Read the numbers carefully, pretending you are *seeing* the chart bars and their exact heights.

                                    **Your task**

                                    Write a concise narrative that to describe the chart, focusing on the most important insights.

                                    **Instructions**

                                    1. Think step-by-step:
                                    - Identify high, low, and median bars.  
                                    - Note any obvious clusters, gaps.  
                                    - Calculate at least one simple statistic if useful (e.g., top vs bottom difference).

                                    2. Write a concise narrative that:
                                    - Starts with a strong headline insight.  
                                    - Highlights 2-3 secondary observations (comparisons, rank order, surprising values).  
                                    - Uses actual numbers or percentages from the data where appropriate.  
                                    - Uses plain language suited to the audience.

                                    **Output**

                                    Return **only** plain text in this structure:

                                    <Headline insight>
                                    • <Bullet 1>
                                    • <Bullet 2>
                                    • ...
                                """
                
                bc_prompt = PromptTemplate(
                    template=bc_des_prompt,
                    input_variables=["data_table"]
                )
                bc_chain = bc_prompt | llm
                bc_result = bc_chain.invoke(input = {"data_table": agg.to_string(index=False)})
                st.write("base chart description:",bc_result.content)
                

                # ////zero shot//// Use llm to read the base chart description to generate section 1 follow-up questions(and provide with the base facts)
                eda_q_for_sec1_template="""
                You are a data-analysis assistant that drafts *exploratory questions* destined for visualization generation.
                The user has just viewed the following **bar chart**:
                **What it shows (summary):** {chart_summary}  

                Additional dataset columns may not represented in the chart: {key_columns}

                **Your Task**
                Generate **two** distinct follow-up questions that logically extend the current EDA based on the following data facts (observations already discovered):
                {data_facts}\n\n
                1. Briefly note which fact most directly extend insights from the current chart.
                2. Pick **exactly two** facts that add new angles, and avoid redundancy with each other.
                3. For the first question, choose a fact and write **one** specific follow-up questions (≤25 words each) that:
                    - The question must refer to **ONLY** one column from {key_columns}. 
                    - The question should be high-level and only mention column from {key_columns} rather than specific value.
                    - Make them answerable with the existing dataset.
                4. For the second question, choose a fact and write **one** specific follow-up questions (≤25 words each) that:
                    - ONLY choose rank facts from {data_facts}.
                    - The question must refer to  **two** column from {key_columns}.
                    - In addition to the column from the first question, the second column should be a new column which "dtype" is "C" from {key_columns}.
                    - The "num_unique_values" of the second column should be **no more than 3**.
                    - Ensure you understand the number of unique values in the column and clearly specify a valid N, either 'Top N' or 'Bottom N' in your question.
                    - **ALWAYS use the same metric** for ranking(i.e. y-axis) from the first question.
                    - Make them answerable with the existing dataset.
                5. Write a title for the chart **(≤7 words each)** based on the question.
                
                **Constraints**
                - Never rewrite th data facts from {data_facts}.
                - The second question should be in the form of **""What are the top N {{first column that the first question refer to}} by {{second column}} in {{the same metric as the first question}}?"**.
                - The title for second question should be in the form of **"Top N {{first column that the first question refer to}} by {{second column}}"**.

                **Example**
                Dataset columns: [gender, racial groups, math score, reading score, writing score]
                Follow-up questions:
                Question 1. How do the average math scores across different races?
                Question 2. Which are top 5 racial groups show the largest gender gap in average math scores?
                Rationale: 
                The first question provide an overview of students from different races on math scores performance.
                The second question extends the analysis by comparing the average math scores(same metric as Question 1) between gender(both male and female) and racial groups, which is a new angle to the analysis.

                **Avoid**
                Question: What are the TOP 3 reading score by writing score for Male and Female in the top racial groups?
                Rationale: It contains three columns: 'writing score', 'gender (Male and Female)' and 'racial groups', which is not allowed.

                **Output (exact JSON)**  
                Do not INCLUDE ```json```.Do not add other sentences after this json data.
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
                    input_variables=["chart_summary","key_columns","data_facts"]
                )
                eda_q_for_sec1_chain = eda_q_for_sec1_prompt | llm
                eda_q_for_sec1_result = eda_q_for_sec1_chain.invoke(input = {"chart_summary":bc_result.content,
                                                                "key_columns":chosen_data_schema,
                                                                "data_facts":str(st.session_state["base_fact"])})
                eda_q_for_sec1_json = json.loads(eda_q_for_sec1_result.content)
                st.write("Follow-up Questions sec 1:",eda_q_for_sec1_json)
                chart_query.append(eda_q_for_sec1_json["follow_up_questions"][0]["question"])
                chart_query.append(eda_q_for_sec1_json["follow_up_questions"][1]["question"])
                chart_title.append(eda_q_for_sec1_json["follow_up_questions"][0]["suggested_viz_title"])
                chart_title.append(eda_q_for_sec1_json["follow_up_questions"][1]["suggested_viz_title"])
                


               # ////one shot//// Use llm to read the base chart description to generate section 2 follow-up questions(and provide with the base facts)
                eda_q_for_sec2_template="""
                You are a data-analysis assistant that drafts *exploratory questions* destined for visualization generation.

                **Your Task**
                Generate **two** distinct follow-up questions that logically extend the current EDA based on the following data facts (observations already discovered):
                {data_facts}\n\n

                Additional dataset columns may not represented in the chart: {key_columns}
                The column that you **SHOULD NEVER** use for follow-up questions: {not_use_column}

                1. Pick **exactly two** facts that add new angles, and avoid redundancy with each other.
                2. For the first question, choose a fact and write **one** specific follow-up questions (≤25 words each) that:
                    - The question must refer to **ONLY** one column from {key_columns}. 
                    - The question should be high-level and only mention column from {key_columns} rather than specific value.
                    - Make them answerable with the existing dataset.
                3. For the second question, choose a fact and write **one** specific follow-up questions (≤25 words each) that:
                    - ONLY choose rank facts from {data_facts}.
                    - The question must refer to  **two** column from {key_columns}.
                    - In addition to the column from the first question, the second column should be a new column which "dtype" is "C" from {key_columns}.
                    - The "num_unique_values" of the second column should be **no more than 3**.
                    - Ensure you understand the number of unique values in the column and clearly specify a valid N, either 'Top N' or 'Bottom N' in your question.
                    - **ALWAYS use the same metric** for ranking(i.e. y-axis) from the first question.
                    - Make them answerable with the existing dataset.
                4. Write a title for the chart **(≤7 words each)** based on the question.
                
                **Constraints**
                - Never rewrite th data facts from {data_facts}.
                - The second question should be in the form of **""What are the top N {{first column that the first question refer to}} by {{second column}} in {{the same metric as the first question}}?"**.
                - The title for second question should be in the form of **"Top N {{first column that the first question refer to}} by {{second column}}"**.

                **Example**
                Dataset columns: [gender, racial groups, math score,reading score, writing score]
                Follow-up questions:
                Question 1. How do the average math scores across different racial groups?
                Question 2. Which are top 5 racial groups show the largest gender gap in average math scores?
                Rationale: 
                The first question provide an overview of students from different racial groups on math scores performance.
                The second question extends the analysis by comparing the average math scores between gender(both male and female) and racial groups, which is a new angle to the analysis.

                **Avoid**
                Question: What are the TOP 3 reading score by writing score for Male and Female in the top racial groups?
                Rationale: It contains three columns: 'writing score', 'gender (Male and Female)' and 'racial groups', which is not allowed.

                **Output (exact JSON)**  
                Do not INCLUDE ```json```.Do not add other sentences after this json data.
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
                    input_variables=["data_facts","key_columns","not_use_column"]
                )
                eda_q_for_sec2_chain = eda_q_for_sec2_prompt | llm
                eda_q_for_sec2_result = eda_q_for_sec2_chain.invoke(input = {
                                                                "data_facts":str(st.session_state["base_fact"]),
                                                                "key_columns":chosen_data_schema,
                                                                "not_use_column":eda_q_for_sec1_json["follow_up_questions"][0]["column"]
                                                                })
                eda_q_for_sec2_json = json.loads(eda_q_for_sec2_result.content)
                st.write("Follow-up Questions sec 2:",eda_q_for_sec2_json)
                chart_query.append(eda_q_for_sec2_json["follow_up_questions"][0]["question"])
                chart_query.append(eda_q_for_sec2_json["follow_up_questions"][1]["question"])
                chart_title.append(eda_q_for_sec2_json["follow_up_questions"][0]["suggested_viz_title"])
                chart_title.append(eda_q_for_sec2_json["follow_up_questions"][1]["suggested_viz_title"])

            # Section 1 Chart 1
                # with open ("vis_nl_pair.json", "r") as f:
                #     vis_nl_json = json.load(f)
                #     query = eda_q_for_sec1_json["follow_up_questions"][0]["question"]
                #     # expanded_query = expand_vis_query(query,list(head), openai_key)
                #     # st.write(f'Expanded Query for Chart:',f'**{expanded_query}**')
                #     # RAG to extract vlspec of similar questions from the vectorstore
                #     st.write(f'**Question for Section 1 Chart 1:**',f'**{query}**')
                    
                #     result = vectorstore.similarity_search(
                #                         query,
                #                         k=1,
                #                     )
                #     result_json = json.loads(result[0].page_content)
                #     st.write("RAG Result:",result_json)
                #     target_nl = ""
                #     for key, value in result_json.items():
                #         target_nl = value
                #     st.write("Target NL:",target_nl)
                #     for nl in vis_nl_json.values():
                #         if nl["nl"] == target_nl:
                #             sample_vlspec = nl["spec"]
                #             st.write("vlspec:",sample_vlspec)
                #             break
                    
                # vlspec = agent_1_generate_code(chosen_dataset,query,eda_q_for_sec1_json["follow_up_questions"][0]["suggested_viz_title"],chosen_data_schema,sample_data, sample_vlspec, openai_key)
                # st.write("Vega-Lite Specification:",vlspec)
                # json_code = json.loads(vlspec)
                # with open(f"DATA2Poster_json/vlspec1_{img_num}.json", "w") as f:
                #     json.dump(json_code, f, indent=2)
                # json_code["height"] =600
                # json_code["width"] =800
                # try:
                #     chart_1 = alt.Chart.from_dict(json_code)
                #     chart_1.save(f"DATA2Poster_img/image{img_num}.png")
                #     # fl_png = vlc.vegalite_to_png(vl_spec=json_code,scale=3.0) # fl aka first_level
                #     # with open(f"DATA2Poster_img/image{img_num}.png", "wb") as f:
                #     #     f.write(fl_png)
                #     st.image(f"DATA2Poster_img/image{img_num}.png", caption="Section 1 Chart 1")
                # except Exception as e:
                #     st.error("The code is failed to execute.")
                
                # # Evaluate the generated first-level chart
                # binary_fl       = open(f"DATA2Poster_img/image{img_num}.png", 'rb').read()  
                # base64_utf8_fl = base64.b64encode(binary_fl).decode('utf-8')
                # fl_url = f'data:image/png;base64,{base64_utf8_fl}'
                # feedback = agent_2_improve_code(query, fl_url, openai_key)
                # st.write(feedback)

                # # Improve the first-level chart based on feedback
                # improved_code = agent_improve_vis(vlspec, feedback,sample_data, openai_key)
                # st.write("Improved Vega-Lite Specification:",improved_code)
                # improved_json = json.loads(improved_code)
                # with open(f"DATA2Poster_json/vlspec2_{img_num}.json", "w") as f:
                #     json.dump(json_code, f, indent=2)      
                # st.write("Improved Vega-Lite JSON:",improved_json)
                # improved_json["height"] =600
                # improved_json["width"] =800
                # try:
                #     chart_1 = alt.Chart.from_dict(improved_json)
                #     chart_1.save(f"DATA2Poster_img/image{img_num}.png")
                #     st.image(f"DATA2Poster_img/image{img_num}.png", caption="Improved Section 1 Chart 1")
                # except Exception as e:
                #     st.error("The code is failed to execute.")

               
                img_num += 1

                # Section 1 Chart 2
                with open ("vis_nl_pair.json", "r") as f:
                    vis_nl_json = json.load(f)
                    query = eda_q_for_sec1_json["follow_up_questions"][1]["suggested_viz_title"]
                    # expanded_query = expand_vis_query(query,list(head), openai_key)
                    # st.write(f'Expanded Query for Chart:',f'**{expanded_query}**')
                    # RAG to extract vlspec of similar questions from the vectorstore
                    st.write(f'**Question for Section 1 Chart 2:**',f'**{query}**')

                    result = vectorstore.similarity_search(
                                        query,
                                        k=1,
                                    )
                    result_json = json.loads(result[0].page_content)
                    # st.write("RAG Result:",result_json)
                    target_nl = ""
                    for key, value in result_json.items():
                        target_nl = value
                    # st.write("Target NL:",target_nl)
                    for nl in vis_nl_json.values():
                        if nl["nl"] == target_nl:
                            sample_vlspec = nl["spec"]
                            st.write("vlspec:",sample_vlspec)
                            break
                    
                vlspec = agent_1_generate_code(chosen_dataset,query,eda_q_for_sec1_json["follow_up_questions"][1]["suggested_viz_title"],chosen_data_schema,sample_data, sample_vlspec, openai_key)
                # st.write("Vega-Lite Specification:",vlspec)
                json_code = json.loads(vlspec)
                with open(f"DATA2Poster_json/vlspec1_{img_num}.json", "w") as f:
                    json.dump(json_code, f, indent=2)
                trans_json = json_code["transform"]
                # json_code["height"] =600
                # json_code["width"] =800
                # st.write("Vega-Lite JSON:",json_code)
                # try:
                #     chart_2 = alt.Chart.from_dict(json_code)
                #     chart_2.save(f"DATA2Poster_img/image{img_num}.png")
                #     # fl_png = vlc.vegalite_to_png(vl_spec=json_code,scale=3.0) # fl aka first_level
                #     # with open(f"DATA2Poster_img/image{img_num}.png", "wb") as f:
                #     #     f.write(fl_png)
                #     st.image(f"DATA2Poster_img/image{img_num}.png", caption="Section 1 Chart 2")
                # except Exception as e:
                #     st.error("The code is failed to execute.") 
                
                #  # Evaluate the generated first-level chart
                # binary_fl       = open(f"DATA2Poster_img/image{img_num}.png", 'rb').read()  
                # base64_utf8_fl = base64.b64encode(binary_fl).decode('utf-8')
                # fl_url = f'data:image/png;base64,{base64_utf8_fl}'
                # feedback = agent_2_improve_code(query, fl_url, openai_key)
                # st.write(feedback)

                # # Improve the first-level chart based on feedback
                # improved_code = agent_improve_vis(vlspec, feedback,sample_data, openai_key)
                # st.write("Improved Vega-Lite Specification:",improved_code)
                # improved_json = json.loads(improved_code)
                # with open(f"DATA2Poster_json/vlspec2_{img_num}.json", "w") as f:
                #     json.dump(json_code, f, indent=2)
                # st.write("Improved Vega-Lite JSON:",improved_json)
                # improved_json["height"] =600
                # improved_json["width"] =800
                # try:
                #     chart_2 = alt.Chart.from_dict(improved_json)
                #     chart_2.save(f"DATA2Poster_img/image{img_num}.png")
                #     st.image(f"DATA2Poster_img/image{img_num}.png", caption="Improved Section 1 Chart 2")
                # except Exception as e:
                #     st.error("The code is failed to execute.")
            
               

                img_num += 1

                # Extract data table from section 1 chart 2 and send it for section 3 question generation
                data_transform_prompt = """You are a Python coding expert.
                You are given the following Vega-Lite transformation specification:{trans_json}
                **Your Task**
                Write the equivalent Python code using Pandas to perform the same transformation on a DataFrame named df. 
                The output should be a new DataFrame after transformed.
                Only return the Python code without any explanation or additional text.
                NEVER include ```python``` in your response.
                Only modify the code in the middle of the following code snippet:        
                    def trans_data(df):\n\n
                    # Assuming "df" is the input DataFrame\n\n
                    # Grouping and aggregating\n\n
                    result_df =         ####only modify this line#####
                    \n\n
                    return result_df
                """
                data_transform_prompt = PromptTemplate(
                    template=data_transform_prompt,
                    input_variables = {"trans_json"})
                data_transform_chain = data_transform_prompt | llm
                data_transform_result = data_transform_chain.invoke(input={"trans_json":json.dumps(trans_json)})
                trans_code = "import pandas as pd\n\n"+f"df = pd.read_csv('data/{chosen_dataset}.csv')\n\n"+data_transform_result.content+"\n\ntrans_df = trans_data(df)\n\ntrans_df.to_csv('DATA2Poster_df/transformed_df.csv', index=False)\n\n"
                exec(trans_code)
                new_df = pd.read_csv('DATA2Poster_df/transformed_df.csv')

               
                
                # ////one shot//// Use llm to read the base chart description to generate section 3 follow-up questions
                    
                eda_q_for_sec3_template="""
                You are a data-analysis assistant that drafts *exploratory questions* destined for visualization generation.
                Here is the data from the visualization chart:{new_df}
                This title of chart is:{title}
                Additional dataset columns may not represented in the chart: {key_columns}
                The column that you **MUST** use for follow-up questions: {use_column}

                **Your Task**
                Generate **two** distinct follow-up questions that logically extend the current EDA:
                1. Read the data carefully. Understand the significant value of the data.
                    - Identify high and low statistic.
                2. Only focus the top N entities in the title.
                3. For the first question, choose the entity with highest statistic from the top N and write one specific follow-up questions (≤25 words each) that:
                    - This question drill down to reveal the pattern within this entity.
                    - The question must refer to **two** column from {key_columns}. 
                    - Make them answerable with the existing dataset.
                4. For the second question, choose the entity with lowest statistic from the top N and write one specific follow-up questions (≤25 words each) that:
                    - This question drill down to reveal the pattern within this entity.
                    - The question must refer to **two** column from {key_columns}. 
                    - Make them answerable with the existing dataset.

                **Example**
                Dataset columns: [gender, racial groups, math score,reading score, writing score]
                Follow-up questions:
                Question 1. How do the math scores by gender in white people group?
                Question 2. How do the math scores by gender in black people group?
                Rationale: 
                The  white people group has the highest average math scores from the top N racial groups. The first question provide an overview of students of different gender on math scores performance in this group.
                The  black people group has the lowest average math scores from the top N racial groups. The second question provide an overview of students of different gender on math scores performance in this group.

                **Output (exact JSON)**  
                Do not INCLUDE ```json```.Do not add other sentences after this json data.
                {{
                "follow_up_questions": [
                    {{
                    "question": "<Question 1>",
                    "suggested_viz_title": "title for the chart",
                    "suggested_viz_type": "<bar|line|pie|scatter|...>",
                    "column": "<column name that the question is related to>",
                    "rationale": "<Why this deepens the analysis>"
                    }},
                    {{
                    "question": "<Question 2>",
                    "suggested_viz_title": "title for the chart",
                    "suggested_viz_type": "<bar|line|pie|scatter|...>",
                    "column": "<column name that the question is related to>",
                    "rationale": "<Rationale>"
                    }}
                    
                ]
                }}"""
                eda_q_for_sec3_prompt = PromptTemplate(
                    template=eda_q_for_sec3_template,
                    input_variables=["new_df","query","key_columns"]
                )
                eda_q_for_sec3_chain = eda_q_for_sec3_prompt | llm
                eda_q_for_sec3_result = eda_q_for_sec3_chain.invoke(input = {
                                                                "new_df":new_df.to_string(index=False),
                                                                "title":eda_q_for_sec1_json["follow_up_questions"][1]["suggested_viz_title"],
                                                                "key_columns":list(head),
                                                                "use_column":eda_q_for_sec2_json["follow_up_questions"][0]["column"]
                                                               
                                                                })
                eda_q_for_sec3_json = json.loads(eda_q_for_sec3_result.content)
                st.write("Follow-up Questions sec 3:",eda_q_for_sec3_json)
                chart_query.append(eda_q_for_sec3_json["follow_up_questions"][0]["question"])
                chart_query.append(eda_q_for_sec3_json["follow_up_questions"][1]["question"])
                chart_title.append(eda_q_for_sec3_json["follow_up_questions"][0]["suggested_viz_title"])
                chart_title.append(eda_q_for_sec3_json["follow_up_questions"][1]["suggested_viz_title"])

                # Section 2 Chart 1
                # with open ("vis_nl_pair.json", "r") as f:
                #     vis_nl_json = json.load(f)
                #     query = eda_q_for_sec2_json["follow_up_questions"][0]["question"]
                #     chart_title.append(eda_q_for_sec2_json["follow_up_questions"][0]["suggested_viz_title"])
                #     # expanded_query = expand_vis_query(query,list(head), openai_key)
                #     # st.write(f'Expanded Query for Chart:',f'**{expanded_query}**')
                #     # RAG to extract vlspec of similar questions from the vectorstore
                #     st.write(f'**Question for Section 2 Chart 1:**',f'**{query}**')
                    
                #     result = vectorstore.similarity_search(
                #                         query,
                #                         k=1,
                #                     )
                #     result_json = json.loads(result[0].page_content)
                #     st.write("RAG Result:",result_json)
                #     target_nl = ""
                #     for key, value in result_json.items():
                #         target_nl = value
                #     st.write("Target NL:",target_nl)
                #     for nl in vis_nl_json.values():
                #         if nl["nl"] == target_nl:
                #             sample_vlspec = nl["spec"]
                #             st.write("vlspec:",sample_vlspec)
                #             break
                    
                # vlspec = agent_1_generate_code(chosen_dataset,query,eda_q_for_sec2_json["follow_up_questions"][0]["suggested_viz_title"],chosen_data_schema,sample_data, sample_vlspec, openai_key)
                # st.write("Vega-Lite Specification:",vlspec)
                # json_code = json.loads(vlspec)
                # with open(f"DATA2Poster_json/vlspec1_{img_num}.json", "w") as f:
                #     json.dump(json_code, f, indent=2)
                # json_code["height"] =600
                # json_code["width"] =800
                # st.write("Vega-Lite JSON:",json_code)
                # try:
                #     chart_3 = alt.Chart.from_dict(json_code)
                #     chart_3.save(f"DATA2Poster_img/image{img_num}.png")
                #     st.image(f"DATA2Poster_img/image{img_num}.png", caption="Section 2 Chart 1")
                #     # fl_png = vlc.vegalite_to_png(vl_spec=json_code,scale=3.0) # fl aka first_level
                #     # with open(f"DATA2Poster_img/image{img_num}.png", "wb") as f:
                #     #     f.write(fl_png)
                # except Exception as e:
                #     st.error("The code is failed to execute.")
                
                #  # Evaluate the generated first-level chart
                # binary_fl       = open(f"DATA2Poster_img/image{img_num}.png", 'rb').read()  
                # base64_utf8_fl = base64.b64encode(binary_fl).decode('utf-8')
                # fl_url = f'data:image/png;base64,{base64_utf8_fl}'
                # feedback = agent_2_improve_code(query, fl_url, openai_key)
                # st.write(feedback)

                # # Improve the first-level chart based on feedback
                # improved_code = agent_improve_vis(vlspec, feedback,sample_data, openai_key)
                # st.write("Improved Vega-Lite Specification:",improved_code)
                # improved_json = json.loads(improved_code)
                # with open(f"DATA2Poster_json/vlspec2_{img_num}.json", "w") as f:
                #     json.dump(json_code, f, indent=2)
                # st.write("Improved Vega-Lite JSON:",improved_json)
                # improved_json["height"] =600
                # improved_json["width"] =800
                # try:
                #     chart_3 = alt.Chart.from_dict(improved_json)
                #     chart_3.save(f"DATA2Poster_img/image{img_num}.png")
                #     st.image(f"DATA2Poster_img/image{img_num}.png", caption="Improved Section 2 Chart 1")
                # except Exception as e:
                #     st.error("The code is failed to execute.")
            
                img_num += 1

                # Section 2 Chart 2
                # with open ("vis_nl_pair.json", "r") as f:
                #     vis_nl_json = json.load(f)
                #     query = eda_q_for_sec2_json["follow_up_questions"][1]["question"]
                #     # expanded_query = expand_vis_query(query,list(head), openai_key)
                #     # st.write(f'Expanded Query for Chart:',f'**{expanded_query}**')
                #     # RAG to extract vlspec of similar questions from the vectorstore
                #     st.write(f'**Question for Section 2 Chart 2:**',f'**{query}**')
                    
                #     result = vectorstore.similarity_search(
                #                         query,
                #                         k=1,
                #                     )
                #     result_json = json.loads(result[0].page_content)
                #     st.write("RAG Result:",result_json)
                #     target_nl = ""
                #     for key, value in result_json.items():
                #         target_nl = value
                #     st.write("Target NL:",target_nl)
                #     for nl in vis_nl_json.values():
                #         if nl["nl"] == target_nl:
                #             sample_vlspec = nl["spec"]
                #             st.write("vlspec:",sample_vlspec)
                #             break
                    
                # vlspec = agent_1_generate_code(chosen_dataset,query,eda_q_for_sec2_json["follow_up_questions"][1]["suggested_viz_title"],chosen_data_schema,sample_data, sample_vlspec, openai_key)
                # st.write("Vega-Lite Specification:",vlspec)
                # json_code = json.loads(vlspec)
                # with open(f"DATA2Poster_json/vlspec1_{img_num}.json", "w") as f:
                #     json.dump(json_code, f, indent=2)
                # json_code["height"] =600
                # json_code["width"] =800
                # st.write("Vega-Lite JSON:",json_code)
                # try:
                #     chart_4 = alt.Chart.from_dict(json_code)
                #     chart_4.save(f"DATA2Poster_img/image{img_num}.png")
                #     # fl_png = vlc.vegalite_to_png(vl_spec=json_code,scale=3.0) # fl aka first_level
                #     # with open(f"DATA2Poster_img/image{img_num}.png", "wb") as f:
                #     #     f.write(fl_png)
                #     st.image(f"DATA2Poster_img/image{img_num}.png", caption="Section 2 Chart 2")
                # except Exception as e:
                #     st.error("The code is failed to execute.")
                # # Evaluate the generated first-level chart
                # binary_fl       = open(f"DATA2Poster_img/image{img_num}.png", 'rb').read()  
                # base64_utf8_fl = base64.b64encode(binary_fl).decode('utf-8')
                # fl_url = f'data:image/png;base64,{base64_utf8_fl}'
                # feedback = agent_2_improve_code(query, fl_url, openai_key)
                # st.write(feedback)

                # # Improve the first-level chart based on feedback
                # improved_code = agent_improve_vis(vlspec, feedback,sample_data, openai_key)
                # st.write("Improved Vega-Lite Specification:",improved_code)
                # improved_json = json.loads(improved_code)
                # with open(f"DATA2Poster_json/vlspec2_{img_num}.json", "w") as f:
                #     json.dump(json_code, f, indent=2)
                # st.write("Improved Vega-Lite JSON:",improved_json)
                # improved_json["height"] =600
                # improved_json["width"] =800
                # try:
                #     chart_4 = alt.Chart.from_dict(improved_json)
                #     chart_4.save(f"DATA2Poster_img/image{img_num}.png")
                #     st.image(f"DATA2Poster_img/image{img_num}.png", caption="Improved Section 2 Chart 2")
                # except Exception as e:
                #     st.error("The code is failed to execute.")
                

                img_num += 1

                # Section 3 Chart 1
                # with open ("vis_nl_pair.json", "r") as f:
                #     vis_nl_json = json.load(f)
                #     query = eda_q_for_sec3_json["follow_up_questions"][0]["question"]
                #     # expanded_query = expand_vis_query(query,list(head), openai_key)
                #     # st.write(f'Expanded Query for Chart:',f'**{expanded_query}**')
                #     # RAG to extract vlspec of similar questions from the vectorstore
                #     st.write(f'**Question for Section 3 Chart 1:**',f'**{query}**')
                    
                #     result = vectorstore.similarity_search(
                #                         query,
                #                         k=1,
                #                     )
                #     result_json = json.loads(result[0].page_content)
                #     st.write("RAG Result:",result_json)
                #     target_nl = ""
                #     for key, value in result_json.items():
                #         target_nl = value
                #     st.write("Target NL:",target_nl)
                #     for nl in vis_nl_json.values():
                #         if nl["nl"] == target_nl:
                #             sample_vlspec = nl["spec"]
                #             st.write("vlspec:",sample_vlspec)
                #             break
                    
                # vlspec = agent_1_generate_code(chosen_dataset,query,eda_q_for_sec3_json["follow_up_questions"][0]["suggested_viz_title"],chosen_data_schema,sample_data, sample_vlspec, openai_key)
                # st.write("Vega-Lite Specification:",vlspec)
                # json_code = json.loads(vlspec)
                # with open(f"DATA2Poster_json/vlspec1_{img_num}.json", "w") as f:
                #     json.dump(json_code, f, indent=2)
                # json_code["height"] =600
                # json_code["width"] =800
                # st.write("Vega-Lite JSON:",json_code)
           
                # try:
                #     chart_5 = alt.Chart.from_dict(json_code)
                #     chart_5.save(f"DATA2Poster_img/image{img_num}.png")
                #     st.image(f"DATA2Poster_img/image{img_num}.png", caption="Section 3 Chart 1")
                # except Exception as e:
                #     st.error("The code is failed to execute.")
                
                # # Evaluate the generated first-level chart
                # binary_fl       = open(f"DATA2Poster_img/image{img_num}.png", 'rb').read()  
                # base64_utf8_fl = base64.b64encode(binary_fl).decode('utf-8')
                # fl_url = f'data:image/png;base64,{base64_utf8_fl}'
                # feedback = agent_2_improve_code(query, fl_url, openai_key)
                # st.write(feedback)

                # # Improve the first-level chart based on feedback
                # improved_code = agent_improve_vis(vlspec, feedback,sample_data, openai_key)
                # st.write("Improved Vega-Lite Specification:",improved_code)
                # improved_json = json.loads(improved_code)
                # with open(f"DATA2Poster_json/vlspec2_{img_num}.json", "w") as f:
                #     json.dump(json_code, f, indent=2)
                # st.write("Improved Vega-Lite JSON:",improved_json)
                # improved_json["height"] =600
                # improved_json["width"] =800
                # try:
                #     chart_5 = alt.Chart.from_dict(improved_json)
                #     chart_5.save(f"DATA2Poster_img/image{img_num}.png")
                #     st.image(f"DATA2Poster_img/image{img_num}.png", caption="Improved Section 3 Chart 1")
                # except Exception as e:
                #     st.error("The code is failed to execute.")
            

                img_num += 1

                # Section 3 Chart 2
                # with open ("vis_nl_pair.json", "r") as f:
                #     vis_nl_json = json.load(f)
                #     query = eda_q_for_sec3_json["follow_up_questions"][1]["question"]
                #     # expanded_query = expand_vis_query(query,list(head), openai_key)
                #     # st.write(f'Expanded Query for Chart:',f'**{expanded_query}**')
                #     # RAG to extract vlspec of similar questions from the vectorstore
                #     st.write(f'**Question for Section 3 Chart 2:**',f'**{query}**')
                    
                #     result = vectorstore.similarity_search(
                #                         query,
                #                         k=1,
                #                     )
                #     result_json = json.loads(result[0].page_content)
                #     st.write("RAG Result:",result_json)
                #     target_nl = ""
                #     for key, value in result_json.items():
                #         target_nl = value
                #     st.write("Target NL:",target_nl)
                #     for nl in vis_nl_json.values():
                #         if nl["nl"] == target_nl:
                #             sample_vlspec = nl["spec"]
                #             st.write("vlspec:",sample_vlspec)
                #             break
                    
                # vlspec = agent_1_generate_code(chosen_dataset,query,eda_q_for_sec3_json["follow_up_questions"][1]["suggested_viz_title"],chosen_data_schema,sample_data, sample_vlspec, openai_key)
                # st.write("Vega-Lite Specification:",vlspec)
                # json_code = json.loads(vlspec)
                # with open(f"DATA2Poster_json/vlspec1_{img_num}.json", "w") as f:
                #     json.dump(json_code, f, indent=2)
                # json_code["height"] =600
                # json_code["width"] =800
                # st.write("Vega-Lite JSON:",json_code)
  
                # try:
                #     chart_6 = alt.Chart.from_dict(json_code)
                #     chart_6.save(f"DATA2Poster_img/image{img_num}.png")
                #     st.image(f"DATA2Poster_img/image{img_num}.png", caption="Section 3 Chart 2")
                # except Exception as e:
                #     st.error("The code is failed to execute.")
                
                # # Evaluate the generated first-level chart
                # binary_fl       = open(f"DATA2Poster_img/image{img_num}.png", 'rb').read()  
                # base64_utf8_fl = base64.b64encode(binary_fl).decode('utf-8')
                # fl_url = f'data:image/png;base64,{base64_utf8_fl}'
                # feedback = agent_2_improve_code(query, fl_url, openai_key)
                # st.write(feedback)

                # # Improve the first-level chart based on feedback
                # improved_code = agent_improve_vis(vlspec, feedback,sample_data, openai_key)
                # st.write("Improved Vega-Lite Specification:",improved_code)
                # improved_json = json.loads(improved_code)
                # with open(f"DATA2Poster_json/vlspec2_{img_num}.json", "w") as f:
                #     json.dump(improved_json, f, indent=2)
                # st.write("Improved Vega-Lite JSON:",improved_json)
                # improved_json["height"] =600
                # improved_json["width"] =800
                # try:
                #     chart_6 = alt.Chart.from_dict(improved_json)
                #     chart_6.save(f"DATA2Poster_img/image{img_num}.png")
                #     st.image(f"DATA2Poster_img/image{img_num}.png", caption="Improved Section 3 Chart 2")
                # except Exception as e:
                #     st.error("The code is failed to execute.")

            

                img_num += 1
                    
                    
                    
        
                
            #     # # Inspect logic error of the vlspec for final validation(at most 3 times)
            #     # pre_final_code = agent_4_validate_spec(improved_code,sample_data, openai_key)
            #     # pre_final_json = json.loads(pre_final_code)
            #     # pre_final_json["height"] =600
            #     # pre_final_json["width"] =800
            #     # st.write("Final Vega-Lite JSON:",pre_final_json)
            #     # exec_count=0
            #     # try:
            #     #     pre_final_png_data = vlc.vegalite_to_png(vl_spec= pre_final_json,scale=3.0)
            #     #     with open(f"DATA2Poster_img/image_{idx}.png", "wb") as f:
            #     #         f.write(pre_final_png_data)
            #     #     st.image(f"DATA2Poster_img/image_{idx}.png", caption="Final Chart")
            #     # except Exception as e:
            #     #     exec_count += 1
            #     #     error = str(e)
            #     #     error_code =  pre_final_code
            #     #     print(f"\n🔴 Error encountered: {error}\n")
                    
            #     #     # Load error-handling prompt
            #     #     error_prompt_template = load_prompt_from_file("prompt_templates/error_prompt.txt")
            #     #     error_prompt = PromptTemplate(
            #     #         template=error_prompt_template,
            #     #         input_variables=["error_code", "error"],
            #     #     )    

            #     #     error_chain = error_prompt | llm 
                    
            #     #     # Invoke the chain to fix the code
            #     #     corrected_code = error_chain.invoke(input={"error_code": error_code, "error": error})
            #     #     final_code = corrected_code.content  # Update with the corrected code
                    
            #     #     if exec_count == 3:
            #     #         st.error("The code is failed to execute.")
            #     #     else:
                    
            #     #         final_json = json.loads(final_code)
            #     #         final_json["height"] =600
            #     #         final_json["width"] =800
            #     #         final_png_data = vlc.vegalite_to_png(vl_spec=final_json,scale=3.0)
            #     #         with open(f"DATA2Poster_img/image_{idx}.png", "wb") as f:
            #     #             f.write(final_png_data)
            #     #         st.image(f"DATA2Poster_img/image_{idx}.png", caption="Final Chart")
                        
                        
            #     # binary_chart     = open(f"DATA2Poster_img/image_{idx}.png", 'rb').read()  # fc aka file_content
            #     # base64_utf8_chart = base64.b64encode(binary_chart ).decode('utf-8')
            #     # img_url = f'data:image/png;base64,{base64_utf8_chart}' 
            #     # chart_prompt_template = load_prompt_from_file("prompt_templates/chart_prompt.txt")
            #     # chart_des_prompt = [
            #     #     SystemMessage(content=chart_prompt_template),
            #     #     HumanMessage(content=[
            #     #         {
            #     #             "type": "text", 
            #     #             "text": f"This chart is ploted  based on this question:\n\n {query}.\n\n"
            #     #         },
            #     #         {
            #     #                 "type": "text", 
            #     #                 "text": "Here is the chart to describe:"
            #     #         },
            #     #         {
            #     #                 "type": "image_url",
            #     #                 "image_url": {
            #     #                     "url": img_url
            #     #                 },
            #     #         },
            #     #     ])
            #     # ] 
                
            #     # chart_des =  llm.invoke(chart_des_prompt)

                
            #     # st.write(f'**Chart Description:**', f'**{chart_des.content}**')
            #     # # st.write(f'**Supported Data Fact:**', f'**{supported_fact}**')
            #     # # st.write(f'**Data Fact after RAG:**', f'**{retrieved_fact}**')
                
            #     # # call GPT to generate insight description
            #     # insight_prompt_template = load_prompt_from_file("prompt_templates/insight_prompt.txt")
            #     # insight_prompt_input_template = load_prompt_from_file("prompt_templates/insight_prompt_input.txt")
            #     # insight_prompt_input = PromptTemplate(
            #     #             template=insight_prompt_input_template,
            #     #             input_variables=["query", "chart_des"],
                                                    
            #     # ) 
            #     # insight_prompt = ChatPromptTemplate.from_messages(
            #     #         messages=[
            #     #             SystemMessage(content = insight_prompt_template),
            #     #             HumanMessagePromptTemplate.from_template(insight_prompt_input.template)
            #     #         ]
            #     #     )
                
            #     # insight_chain = insight_prompt | llm
            #     # insight = insight_chain.invoke(input= {"query":query, "chart_des":chart_des})
            #     # insight_list.append(insight.content)
            #     # st.write(f'**Insight Description:**', f'**{insight.content}**')
            #     # chart_des_list.append(chart_des.content)

            # # # Test chart combiner#############################
            img_to_llm_list = []
            for i in range(1,img_num):
                
            #         binary_chart     = open(f"DATA2Poster_img/image{i}.png", 'rb').read()
            #         base64_utf8_chart = base64.b64encode(binary_chart).decode('utf-8')
            #         img_url = f'data:image/png;base64,{base64_utf8_chart}'
                    img_to_llm_list.append({"chart_id": i, "title": chart_title[i-1]})
            #         img_to_llm_list.append({"type": "image_url", "image_url": {"url": img_url}})
         
        

            # ////zero shot//// Use llm to select 6 charts at once             
            chart_check_prompt ="""
                

                                        You are “ChartSelector-Bot”, an assistant that builds a 3-section poster.
                                        You are given only the chart titles and its id listed below:{img_to_llm_list}
                                        
                                        ** Your Tasks(Think step by step)**
                                        1. Read and evaluate each chart title for subject matter, variables, and implied insight.
                                        2. Identify which charts title are identical or nearly identical.
                                        3. Select **exactly six chart titles** that are distinct to each other and no duplication.
                                        4. Group the selected chart titles into three sections with the following roles and order:
                                        SECTION 1
                                            • Chart 1 — Overall trend / distribution.          
                                            • Chart 2 — Top-N ranking that extends Chart 1.    

                                        SECTION 2
                                            • Chart 3 — Overall trend using a DIFFERENT dimension than Section 1.                      
                                            • Chart 4 — Top-N ranking that extends Chart 3.    

                                        SECTION 3
                                            • Chart 5 — Drill-down on the HIGHEST entity from SECTION 1 Chart 2.
                                            • Chart 6 — Drill-down on the LOWEST  entity from SECTION 1 Chart 2. 
                                        
                                        **EXAMPLE**
                                        Chart 1  "Total Sales Trend 2010-2024"
                                        Chart 2  "Top 5 Countries by Total Sales"
                                        Chart 3  "Sales by Product Category 2010-2024"
                                        Chart 4  "Top 5 Product Categories by Sales"
                                        Chart 5  "Category Breakdown for USA (highest)"
                                        Chart 6  "Category Breakdown for Iceland (lowest)"
                                            
                                        
                                        **RULES**
                                        1. Do not repeat a chart ID.
                                        2. Section 1 and Section 2 must use different categorical dimensions.
                                        3. Charts in the same section are closely related and, when viewed together, communicate a single, higher-level insight.
                                        4. Create a concise section heading that captures the shared insight of each group.
                                        5. For each section, write a one sentence (shorter than 20 words) that captures the key insight conveyed in that section.
                                      
                                        **Output (JSON)**
                                        Do not INCLUDE ```json```.Do not add other sentences after this json data.
                                        Return **only** the final JSON in this structure:
                                        {{
                                        "sections": [
                                            {{
                                            "section": "A",
                                            "heading": "<section heading>",
                                            "charts_id": ["<chart id 1>", "<chart id 2>"],
                                            "charts_title": ["<chart title 1>", "<chart title 2>"],
                                            "insight": "<one-sentence synthesis>"
                                            }},
                                            {{
                                            "section": "B",
                                            "heading": "...",
                                            "charts_id": ["<chart id 1>", "<chart id 2>"],
                                            "charts_title": ["<chart title 1>", "<chart title 2>"],
                                            "insight": "..."
                                            }},
                                            {{
                                            "section": "C",
                                            "heading": "...",
                                            "charts_id": ["<chart id 1>", "<chart id 2>"],
                                            "charts_title": ["<chart title 1>", "<chart title 2>"],
                                            "insight": "..."
                                            }}
                                        ]
                                        }}"""
                                        
            chart_check_prompt = PromptTemplate(
                            template=chart_check_prompt,
                            input_variables=["img_to_llm_list"]
                        )
            chart_check_chain = chart_check_prompt | llm
            chart_check = chart_check_chain.invoke(input = {"img_to_llm_list":img_to_llm_list})
            chart_check_json = json.loads(chart_check.content)
            st.write("Chart Check JSON:",chart_check_json)
            chart_id_list = []
            insight_list = []

            for section in chart_check_json["sections"]:
                insight_list.append(section["insight"])
                for chart_id in section["charts_id"]:
                    chart_id_list.append(int(chart_id))
            rag_spec = []
            for id in chart_id_list:
                with open ("vis_nl_pair.json", "r") as f:
                    vis_nl_json = json.load(f)
                    query = chart_title[id-1]
                    # st.write(f'**Question for Chart {id}:**',f'**{query}**')
                    # st.write("Chart Title:",chart_title[id-1])
                    # RAG to extract vlspec of similar questions from the vectorstore
                    result = vectorstore.similarity_search(
                                        query,
                                        k=1,
                                    )
                    result_json = json.loads(result[0].page_content)
                    st.write("RAG Result:",result_json)
                    target_nl = ""
                    for key, value in result_json.items():
                        target_nl = value
                    st.write("Target NL:",target_nl)
                    for nl in vis_nl_json.values():
                        if nl["nl"] == target_nl:
                            sample_vlspec = nl["spec"]
                            # st.write("vlspec:",sample_vlspec)
                            rag_spec.append(sample_vlspec)
                            break

            vlspec = agent_consistent(chosen_dataset,chart_query,chart_title,chosen_data_schema,sample_data, rag_spec, openai_key)
            st.write("Vega-Lite Specification:",vlspec)
            json_vlspec = json.loads(vlspec)
            spec_id = 0
            for spec in json_vlspec["visualizations"]:
                with open(f"DATA2Poster_json/vlspec1_{spec_id}.json", "w") as f:
                    json.dump(spec, f, indent=2)
                spec["height"] =600
                spec["width"] =800
                try:
                    chart_3 = alt.Chart.from_dict(spec)
                    chart_3.save(f"DATA2Poster_chart/image{spec_id}.png")
                    st.image(f"DATA2Poster_chart/image{spec_id}.png", caption="Chart "+str(spec_id))

                except Exception as e:
                    error = str(e)
                    print(f"\n🔴 Error encountered: {error}\n")
                    
                    # Load error-handling prompt
                    error_prompt_template = load_prompt_from_file("prompt_templates/error_prompt.txt")
                    error_prompt = PromptTemplate(
                        template=error_prompt_template,
                        input_variables=["error_code", "error"],
                    )    

                    error_chain = error_prompt | llm 
                    
                    # Invoke the chain to fix the code
                    corrected_vlspec = error_chain.invoke(input={"error_code": spec, "error": error})
                    json_corrected_vlspec = json.loads(corrected_vlspec.content)
                    json_corrected_vlspec["height"] =600
                    json_corrected_vlspec["width"] =800
                    final_chart = alt.Chart.from_dict(json_corrected_vlspec)
                    final_chart.save(f"DATA2Poster_chart/image{spec_id}.png")
                    st.image(f"DATA2Poster_chart/image{spec_id}.png", caption="Chart "+str(spec_id))
                spec_id += 1
            
            # for i in range(0,6):          
            # # Evaluate the generated first-level chart
            #     binary_fl       = open(f"DATA2Poster_chart/image{i}.png", 'rb').read()
            #     base64_utf8_fl = base64.b64encode(binary_fl).decode('utf-8')
            #     fl_url = f'data:image/png;base64,{base64_utf8_fl}'
            #     feedback = agent_2_improve_code(chart_query[i], fl_url, openai_key)
            #     st.write(feedback)

            #     # Improve the first-level chart based on feedback
            #     improved_code = agent_improve_vis(json_vlspec["visualizations"][i], feedback, sample_data, openai_key)
            #     st.write("Improved Vega-Lite Specification:",improved_code)
            #     improved_json = json.loads(improved_code)
            #     with open(f"DATA2Poster_json/vlspec2_{id}.json", "w") as f:
            #         json.dump(improved_json, f, indent=2)
            #     st.write("Improved Vega-Lite JSON:",improved_json)
            #     improved_json["height"] =600
            #     improved_json["width"] =800
            #     try:
            #         chart_ = alt.Chart.from_dict(improved_json)
            #         chart_.save(f"DATA2Poster_chart/image{id}.png")
            #         st.image(f"DATA2Poster_chart/image{id}.png", caption="Improved Chart "+str(id))
            #     except Exception as e:
            #         st.error("The code is failed to execute.")
                

            chartid_for_pdf = []
            chart_for_pdf = []
           
            for id in range(0,spec_id):
                chartid_for_pdf.append(id)
                binary_chart     = open(f"DATA2Poster_chart/image{id}.png", 'rb').read()
                base64_utf8_chart = base64.b64encode(binary_chart).decode('utf-8')
                img_url = f'data:image/png;base64,{base64_utf8_chart}'              
                chart_for_pdf.append(img_url)
            
            # chart_dedup_prompt = [
            #         SystemMessage(content="""
            #                                 You are a senior data-visualisation reviewer.
            #                                 You will receive **three groups**, each containing two chart link to Vega-Lite specifications.  
            #                                 For every group:

            #                                 1. **Check visual consistency** between the two charts in that group only.  
            #                                 Consider at least these aspects:  
            #                                 - colour palette / category ordering  
            #                                 - font family & font sizes (title, axis, legend)  
            #                                 - axis & gridline style (position, tick count, rotation, grid visibility)  
            #                                 - legend placement & formatting  
            #                                 - mark style (strokeWidth, cornerRadius, opacity)  
            #                                 2. **Rate** the pair on a 5-point scale where 5 = perfectly consistent, 1 = very inconsistent.  
            #                                 3. **List every inconsistency** you find (bullet points).  
            #                                 4. **Recommend precise changes** (bullet points) to make the two charts visually consistent, using the smaller change principle.

            #                                 Output format (strict markdown):
            #                                 Group A (score: X/5)
            #                                 Inconsistencies
            #                                 ...


            #                                 Recommendations

            #                                 ...

            #                                 Group B (score: X/5)

            #                                 ...

            #                                 Group C (score: X/5)
            #                                 ..."""),
            # # You are a data-visualization expert. You will receive three groups, each with two chart IDs. 
            # #                       For each pair, if the chart is blank, replace the id of duplicate one with "99".
            # #                       Additionally, if the charts are identical, keep one id and replace the id of duplicate one with "99". 
            # #                       Return exactly six chart IDs, preserving their original order. Replace any duplicate with 99. 
            # #                       Output only the JSON object in this form:{ "chart": [id1, id2, id3, id4, id5, id6] }. 
            # #                       Do not INCLUDE ```json```. No additional text after this json."""),
            #         HumanMessage(content=[
            #             {
            #                 "type": "text", 
            #                 "text": f"Group 1.\n\n"
            #             },
            #             {
            #                     "type": "image_url",
            #                     "image_url": {
            #                         "url": chart_for_pdf[0]
            #                     },
            #             },
            #             {
            #                     "type": "image_url",
            #                     "image_url": {
            #                         "url": chart_for_pdf[1]
            #                     },
            #             },
                       
            #             {
            #                 "type": "text", 
            #                 "text": f"Group 2.\n\n"
            #             },
            #             {
            #                     "type": "image_url",
            #                     "image_url": {
            #                         "url": chart_for_pdf[2]
            #                     },
            #             },
            #             {
            #                     "type": "image_url",
            #                     "image_url": {
            #                         "url": chart_for_pdf[3]
            #                     },
            #             },
            #              {
            #                 "type": "text", 
            #                 "text": f"Group 3.\n\n"
            #             },
            #             {
            #                     "type": "image_url",
            #                     "image_url": {
            #                         "url": chart_for_pdf[4]
            #                     },
            #             },
            #             {
            #                     "type": "image_url",
            #                     "image_url": {
            #                         "url": chart_for_pdf[5]
            #                     },
            #             },
            #         ])
            #     ] 
                
            # chart_dedup =  llm.invoke(chart_dedup_prompt)
            # # chart_dedup_json = json.loads(chart_dedup.content)
            # # chartid_for_pdf = chart_dedup_json["chart"]
            # st.write("Chart improve consistency feedback:",chart_dedup.content)
        
            # Reset session state
            st.session_state["bt_try"] = ""  
            st.session_state["sub_fact"] = []
            st.session_state["questions_for_poster"] = []
            # Create pdf and download
            pdf_title = f"{chosen_dataset} Poster"
            create_pdf(chosen_dataset,insight_list,chartid_for_pdf,chart_for_pdf,openai_key)
            st.success("Poster has been created successfully!🎉")
            with open(f"pdf/{chosen_dataset}_summary.pdf", "rb") as f:
                st.download_button("Download Poster as PDF", f, f"""{chosen_dataset}_summary.pdf""")

# Display chosen datasets 
if chosen_dataset :
    st.subheader(chosen_dataset)
    st.dataframe(datasets[chosen_dataset],hide_index=True)

# Insert footer to reference dataset origin  
footer="""<style>.footer {position: fixed;left: 0;bottom: 0;width: 100%;text-align: center;}</style><div class="footer">
<p> <a style="display: block; text-align: center;"> Datasets courtesy of NL4DV, nvBench and ADVISor </a></p></div>"""
st.caption("Datasets courtesy of NL4DV and kaggle")

# Hide menu and footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

