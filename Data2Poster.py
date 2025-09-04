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
from json_sanitizer import parse_jsonish, normalize_jsonish

# Import langchain modules
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import random
import os







    
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
    datasets["Occupation_gender_gap2"] = pd.read_csv("data/Occupation_gender_gap2.csv")
    datasets["Occupation_by_gender"] = pd.read_csv("data/Occupation_by_gender.csv")
    datasets["adidas_sale"] = pd.read_csv("data/adidas_sale.csv")
    datasets["volcano"] = pd.read_csv("data/volcano.csv")
    datasets["Indian_Kids_Screen_Time"] = pd.read_csv("data/Indian_Kids_Screen_Time.csv")
    datasets["billionaires"] = pd.read_csv("data/billionaires.csv")
    datasets["flower"] = pd.read_csv("data/flower.csv")
    datasets["Coffee_Chain"] = pd.read_csv("data/Coffee_Chain.csv")
    datasets["movies_record"] = pd.read_csv("data/movies_record.csv")




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
    # Set area for OpenAI key //Gemeini key
    openai_key = st.text_input(label = "🔑 OpenAI Key:", help="Required for models.",type="password")

    # Set area for OpenAPI key
    openapi_key = st.text_input(label = "🔑 OpenAPI Key:", help="Required for models.",type="password")

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
if "s1c1_fact" not in st.session_state:
    st.session_state["s1c1_fact"] = []
if "s2c1_fact" not in st.session_state:
    st.session_state["s2c1_fact"] = []




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
# def expand_vis_query(query,columns, openai_key):
#     """Expand the query using OpenAI API"""
#     llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", temperature=0, api_key=openai_key)
#     prompt = PromptTemplate(
#         template="""
#                 You are an expert data visualization analyst. 
#                 **Task**
#                 Analyze a natural language query and expand it about creating a visualization and expand it into a comprehensive, detailed specification that can guide the creation of an effective chart.
#                 **Instructions**
#                 When given a natural language visualization query, analyze it thoroughly and provide a detailed expansion covering all aspects needed to create the visualization.
#                 Think step by step and consider the following aspects:          
#                 Step 1. Query Understanding
#                 - Intent: Summarize what the user wants to visualize
#                 - Key Elements: Identify the main data components mentioned
#                 - Implicit Requirements: Note any unstated but implied needs
#                 Step 2. Data Requirements
#                 - Data Types: Specify categorical, numerical, temporal, etc.
#                 - Data Preprocessing: Any transformations, aggregations, or calculations needed
#                 Step 3. Visualization Specification
#                 - Chart Type: Recommend **one** that is the most appropriate visualization type 
#                 - Visual Encoding: Specify how data should map to visual elements (x-axis, y-axis, color, size, shape, etc.)
#                 - Scales and Axes: Define axis types, scales (linear, log, categorical), ranges, and labels
#                 Step 4. Write a comprehensive, detailed specification that can guide the creation of an effective chart.

#                 Now analyze the following natural language visualization query and provide a comprehensive expansion following the instruction above:
#                 Query: {query}
#                 Dataset columns available: {columns}  
#                 **Output ONLY the comprehensive detailed specification without intermediate step. NO extra commentary. "The expanded query is ..."**
               

#                     """,
#         input_variables=["query","columns"],
#     )
    
#     chain = prompt | llm
#     expanded_query = chain.invoke(input={"query": query,"columns": columns})
#     return expanded_query.content
def load_json(json_file):
    with open(json_file, "r", encoding="utf-8") as fh:

        return json.load(fh)
 
# Check if the user has tried and entered an OpenAI key
api_keys_entered = True  # Assume the user has entered an OpenAI key
if try_true or (st.session_state["bt_try"] == "T"):
    # if not openai_key.startswith("sk-"):
    #             st.error("Please enter a valid OpenAI API key.")
    #             api_keys_entered = False

    st.session_state["bt_try"] = "T"
    
    if api_keys_entered:
        # use GPT as llm
        # llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", temperature=0,api_key = openapi_key)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            max_tokens=None,
            api_key = openai_key
            # other params...
        )
        # use OpenAIEmbeddings as embedding model
        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key = openapi_key)
        # embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", api_key = openai_key)
        # Create a vector store
               
        data_list = load_json('vis_corpus.json')
        docs = [Document(page_content=json.dumps(item)) for item in data_list]
        vectorstore = FAISS.from_documents(
        docs,
        embeddings_model
        )
        
      
        # ////zero shot//// Use llm to select vaild columns for 3 sections
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
        col_select_result = col_select_chain.invoke(input = {"schema": chosen_data_schema})
        col_select_json = parse_jsonish(col_select_result.content)
        st.write("Selected Columns:",col_select_json)
        # Extract column names by dtype from col_select_json
        dtype_columns_dict = {}
        dtype_columns_dict["breakdown"] = []
        dtype_columns_dict["measure"] = []
        for col in col_select_json["key_columns"]:
            dtype = col["properties"]["dtype"]
            if dtype =="C" or dtype =="T":
                dtype_columns_dict["breakdown"].append(col["column"])
            else:
                dtype_columns_dict["measure"].append(col["column"])
        st.write("Columns grouped by dtype:", dtype_columns_dict)

        
        facts_list = []
        # USE ALL COLUMNS TO GENERATE FACTS#####################################################################
        desired_combinations = ['CNC', 'TNC', 'CNN', 'TNN']
        # desired_combinations = ['CN', 'TN']
        result = organize_by_dtype_combinations(chosen_data_schema, desired_combinations)
        for item in result:
            if item["combinations"] == "CNC":
                breakdown = [col for col, dtype in item["columns_dtype_mapping"].items() if dtype == "C"]
                measure = [col for col, dtype in item["columns_dtype_mapping"].items() if dtype == "N"]
                facts_1 = generate_facts(
                                dataset=Path(f"data/{chosen_dataset}.csv"),
                                breakdown=breakdown[0],
                                measure=measure[0],
                                series=None,
                                breakdown_type="C",
                                measure_type="N",
                                with_vis=False,
                            )
                for fact in facts_1:
                    facts_list.append({"content":fact["content"], "score":fact["score_C"]})
                facts_2 = generate_facts(
                                dataset=Path(f"data/{chosen_dataset}.csv"),
                                breakdown=breakdown[1],
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
                                measure=measure[0],
                                series=breakdown[1],
                                breakdown_type="C",
                                measure_type="N",
                                with_vis=False,
                            )
                for fact in facts_3:
                    facts_list.append({"content":fact["content"], "score":fact["score_C"]})
                facts_4 = generate_facts(
                                dataset=Path(f"data/{chosen_dataset}.csv"),
                                breakdown=breakdown[1],
                                measure=measure[0],
                                series=breakdown[0],
                                breakdown_type="C",
                                measure_type="N",
                                with_vis=False,
                            )
                for fact in facts_4:
                    facts_list.append({"content":fact["content"], "score":fact["score_C"]})
            elif item["combinations"] == "TNC":
                breakdown_T = [col for col, dtype in item["columns_dtype_mapping"].items() if dtype == "T"]
                breakdown_C = [col for col, dtype in item["columns_dtype_mapping"].items() if dtype == "C"]
                measure = [col for col, dtype in item["columns_dtype_mapping"].items() if dtype == "N"]
                facts_1 = generate_facts(
                                dataset=Path(f"data/{chosen_dataset}.csv"),
                                breakdown=breakdown_T[0],
                                measure=measure[0],
                                series=None,
                                breakdown_type="T",
                                measure_type="N",
                                with_vis=False,
                            )
                for fact in facts_1:
                    facts_list.append({"content":fact["content"], "score":fact["score_C"]})
                facts_2 = generate_facts(
                                dataset=Path(f"data/{chosen_dataset}.csv"),
                                breakdown=breakdown_C[0],
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
                                breakdown=breakdown_T[0],
                                measure=measure[0],
                                series=breakdown_C[0],
                                breakdown_type="T",
                                measure_type="N",
                                with_vis=False,
                            )
                for fact in facts_3:
                    facts_list.append({"content":fact["content"], "score":fact["score_C"]})
                facts_4 = generate_facts(   
                                dataset=Path(f"data/{chosen_dataset}.csv"),
                                breakdown=breakdown_C[0],
                                measure=measure[0],
                                series=breakdown_T[0],
                                breakdown_type="C",
                                measure_type="N",
                                with_vis=False,
                            )
                for fact in facts_4:
                    facts_list.append({"content":fact["content"], "score":fact["score_C"]})
            elif item["combinations"] == "CNN":
                breakdown = [col for col, dtype in item["columns_dtype_mapping"].items() if dtype == "C"]
                measure = [col for col, dtype in item["columns_dtype_mapping"].items() if dtype == "N"]
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
            elif item["combinations"] == "TNN":
                breakdown = [col for col, dtype in item["columns_dtype_mapping"].items() if dtype == "T"]
                measure = [col for col, dtype in item["columns_dtype_mapping"].items() if dtype == "N"]
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
        
        # Preprocess the facts into insight fact list and send it to llm afterwards
        if "s1c1_fact" not in st.session_state:
            st.session_state["s1c1_fact"] = []
        if "s2c1_fact" not in st.session_state:
            st.session_state["s2c1_fact"] = []
        
        # Ranking facts by score
        facts_list = sorted(facts_list, key=itemgetter('score'), reverse=True) 
        facts_list = [fact for fact in facts_list if fact["score"] > 1]
        st.write("Filtered and Ranked Facts:",facts_list)
        seen = set()
        seen_1 = set()
        for item in facts_list:
            if item["content"] != "No fact." and col_select_json["key_columns"][0]["column"] in item["content"] and item["content"] not in seen:
                seen.add(item["content"])
                st.session_state["s1c1_fact"].append(item["content"])
            elif item["content"] != "No fact." and col_select_json["key_columns"][1]["column"] in item["content"] and item["content"] not in seen_1:
                seen_1.add(item["content"])
                st.session_state["s2c1_fact"].append(item["content"])
        # top 100 facts
        # st.session_state["base_fact"] = st.session_state["base_fact"][0:100]
        # st.session_state["sub_fact"] = st.session_state["sub_fact"][0:100]
        st.write("s1c1 Facts:",st.session_state["s1c1_fact"])
        st.write("s2c1 Facts:",st.session_state["s2c1_fact"])

        insight_from_fact_to_llm=[]
        for i in range(0,2):   
            if i == 0:
                insight_from_s1c1 = self_augmented_knowledge(openai_key, chosen_dataset, col_select_json["key_columns"][i]["column"], st.session_state["s1c1_fact"], )
                insight_from_fact_to_llm.append(insight_from_s1c1)
            elif i == 1:
                insight_from_s2c1 = self_augmented_knowledge(openai_key, chosen_dataset, col_select_json["key_columns"][i]["column"], st.session_state["s2c1_fact"], )
                insight_from_fact_to_llm.append(insight_from_s2c1)
        st.write("Insights from Facts:",insight_from_fact_to_llm)

        # //// Random sample 100 facts for passing to LLM ////
        st.session_state["s1c1_fact"] = random.sample(st.session_state["s1c1_fact"], min(100, len(st.session_state["s1c1_fact"])))
        st.session_state["s2c1_fact"] = random.sample(st.session_state["s2c1_fact"], min(100, len(st.session_state["s2c1_fact"])))
          

        chart_title = []
        chart_query = []

        # ////zero shot//// Use llm to read the base chart description to generate section 1 follow-up questions(and provide with the base facts)
        eda_q_for_sec1_template="""
        You are a data-analysis assistant that drafts *exploratory questions* destined for visualization generation.
        Additional dataset columns information: {key_columns}
        Read the information founded from the current exploratory data analysis(EDA) carefully: {s1c1_insight}

        **Your Task**
        Generate **two** distinct follow-up questions that logically extend the current EDA based on the following data facts (observations already discovered):
        {data_facts}\n\n
        1. Total Pick **exactly two** facts that add new angles, and avoid redundancy with each other.
        2. For the first question, choose a fact and write one specific follow-up questions that:
            - In addition to the numerical column(for Y-axis), the question must refer to this one column(for X-axis) **ONLY**: {column_1}. 
            - The question should be high-level and **ONLY** mention column name rather than specific value.
            - Make them answerable with the existing dataset.
        3. For the second question, choose a fact and write one specific follow-up questions that:
            - In addition to the numerical column(for Y-axis), the question must refer to these two column(for X-axis) **ONLY**: {column_1} and the other different column.
            - Make them answerable with the existing dataset.
        4. Write a title for the chart **(no more than 12 words each)** based on the questions.
        
        **Constraints**
        - Never rewrite th data facts from {data_facts}.
    
        
        **Example**
        Dataset columns: [gender, racial groups, math score, reading score, writing score]
        Follow-up questions:
        Question 1. How do the average math scores across different races?
        Rationale: 
        Follows rule #2: 
        - Focuses on one numerical column (math score) and one categorical column (gender).
        - High-level (doesn't mention specific values like “female” or “male”).
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
            input_variables=["key_columns","s1c1_insight","data_facts","column_1"]
        )
        eda_q_for_sec1_chain = eda_q_for_sec1_prompt | llm
        eda_q_for_sec1_result = eda_q_for_sec1_chain.invoke(input = {
                                                        "key_columns":col_select_json["key_columns"],
                                                        "s1c1_insight":insight_from_fact_to_llm[0],
                                                        "data_facts":str(st.session_state["s1c1_fact"]),
                                                        "column_1":col_select_json["key_columns"][0]["column"],
                                                        })
        eda_q_for_sec1_json = parse_jsonish(eda_q_for_sec1_result.content)
        # st.write(eda_q_for_sec1_result.content)
        # eda_q_for_sec1_json = json.loads(eda_q_for_sec1_result.content)
        st.write("Follow-up Questions sec 1:",eda_q_for_sec1_json)
        chart_query.append(eda_q_for_sec1_json["follow_up_questions"][0]["question"])
        chart_query.append(eda_q_for_sec1_json["follow_up_questions"][1]["question"])
        chart_title.append(eda_q_for_sec1_json["follow_up_questions"][0]["suggested_viz_title"])
        chart_title.append(eda_q_for_sec1_json["follow_up_questions"][1]["suggested_viz_title"])
        


    #     # ////one shot//// Use llm to read the base chart description to generate section 2 follow-up questions(and provide with the base facts)
        eda_q_for_sec2_template="""
        You are a data-analysis assistant that drafts *exploratory questions* destined for visualization generation.
        Additional dataset columns information: {key_columns}
        Read the information founded from the current exploratory data analysis(EDA) carefully: {s2c1_insight}


        **Your Task**
        Generate **two** distinct follow-up questions that can make a comparison with this two questions "{s1q1}" and "{s1q2}" based on the following data facts (observations already discovered):
        {data_facts}\n\n
        1. Total Pick **exactly two** facts that add new angles, and avoid redundancy with each other.
        2. For the first question, choose a fact and write one specific follow-up questions that:
            - In addition to the numerical column(for Y-axis), the question must refer to one column(for X-axis) **ONLY**:
            - The question should be high-level and **ONLY** mention column name rather than specific value.
            - Make them answerable with the existing dataset.
        3. For the second question, choose a fact and write one specific follow-up questions that:
            - In addition to the numerical column(for Y-axis), the question must refer to these two column(for X-axis) **ONLY**: the column used in the first question and the other different column.
            - Avoid using the same combination of columns as following: {column_from_s1q2}.
            - Make them answerable with the existing dataset.
        4. Write a title for the chart **(no more than 12 words each)** based on the questions.
        
        **Constraints**
        - Never rewrite th data facts from {data_facts}.
     
        **Example**
        Dataset columns: [gender, racial groups, math score, reading score, writing score]
        Follow-up questions:
        Question 1. How do the average math scores across different races?
        Rationale: 
        Follows rule #2: 
        - Focuses on one numerical column (math score) and one categorical column (gender).
        - High-level (doesn't mention specific values like “female” or “male”).
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
            input_variables=["key_columns","s2c1_insight","s1q1","s1q2","data_facts","column_from_s1q2"]
        )
        eda_q_for_sec2_chain = eda_q_for_sec2_prompt | llm
        eda_q_for_sec2_result = eda_q_for_sec2_chain.invoke(input = {
                                                        
                                                        "key_columns":col_select_json["key_columns"],
                                                        "s2c1_insight":insight_from_fact_to_llm[1],
                                                        "s1q1": eda_q_for_sec1_json["follow_up_questions"][0]["question"],
                                                        "s1q2": eda_q_for_sec1_json["follow_up_questions"][1]["question"],
                                                        "data_facts":str(st.session_state["s2c1_fact"]),
                                                        "column_from_s1q2":eda_q_for_sec1_json["follow_up_questions"][1]["column"]
                                                        })
        eda_q_for_sec2_json = parse_jsonish(eda_q_for_sec2_result.content)
        # st.write(eda_q_for_sec2_json)
        # eda_q_for_sec2_json = `json.load`s(eda_q_for_sec2_result.content)
        st.write("Follow-up Questions sec 2:",eda_q_for_sec2_json)
        chart_query.append(eda_q_for_sec2_json["follow_up_questions"][0]["question"])
        chart_query.append(eda_q_for_sec2_json["follow_up_questions"][1]["question"])
        chart_title.append(eda_q_for_sec2_json["follow_up_questions"][0]["suggested_viz_title"])
        chart_title.append(eda_q_for_sec2_json["follow_up_questions"][1]["suggested_viz_title"])


        # Section 1 Chart 2
        # with open ("vis_nl_pair.json", "r") as f:
        #     vis_nl_json = json.load(f)
        #     query = eda_q_for_sec1_json["follow_up_questions"][1]["suggested_viz_title"]
        #     # expanded_query = expand_vis_query(query,list(head), openai_key)
        #     # st.write(f'Expanded Query for Chart:',f'**{expanded_query}**')
        #     # RAG to extract vlspec of similar questions from the vectorstore
        #     st.write(f'**Question for Section 1 Chart 2:**',f'**{query}**')

        #     result = vectorstore.similarity_search(
        #                         query,
        #                         k=1,
        #                     )
        #     result_json = json.loads(result[0].page_content)
        #     # st.write("RAG Result:",result_json)
        #     target_nl = ""
        #     for key, value in result_json.items():
        #         target_nl = value
        #     # st.write("Target NL:",target_nl)
        #     for nl in vis_nl_json.values():
        #         if nl["nl"] == target_nl:
        #             sample_vlspec = nl["spec"]
        #             st.write("vlspec:",sample_vlspec)
        #             break
            
        # vlspec = agent_1_generate_code(chosen_dataset,query,eda_q_for_sec1_json["follow_up_questions"][1]["suggested_viz_title"],chosen_data_schema,sample_data, sample_vlspec, openai_key)
        # # st.write("Vega-Lite Specification:",vlspec)
        # json_code = json.loads(vlspec)
        # with open(f"DATA2Poster_json/vlspec1_1.json", "w") as f:
        #     json.dump(json_code, f, indent=2)
        # trans_json = json_code["transform"]

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
        # data_transform_prompttemplate = PromptTemplate(
        #     template=data_transform_prompt,
        #     input_variables = {"trans_json"})
        # data_transform_chain = data_transform_prompttemplate | llm
        # data_transform_result = data_transform_chain.invoke(input={"trans_json":json.dumps(trans_json)})
        # trans_code = "import pandas as pd\n\n"+f"df = pd.read_csv('data/{chosen_dataset}.csv')\n\n"+data_transform_result.content+"\n\ntrans_df = trans_data(df)\n\ntrans_df.to_csv('DATA2Poster_df/transformed_df.csv', index=False)\n\n"
        # exec(trans_code)
        # new_df = pd.read_csv('DATA2Poster_df/transformed_df.csv')
        # # Drop columns whose name contains "rank"
        # cols_to_drop = [col for col in new_df.columns if "rank" in col]
        # new_df = new_df.drop(columns=cols_to_drop)
        # st.write("Transformed DataFrame:",new_df)
       

        # ////zero shot//// Use llm to read the insights from s1c1 facts to generate section 3 follow-up questions
            
        eda_q_for_sec3_template="""
        You are a data-analysis assistant that drafts *exploratory questions* destined for visualization generation.
        Additional dataset columns information: {key_columns}
        Read the information founded from the current exploratory data analysis(EDA) carefully: {s1c1_insight}


        **Your Task**
        Generate **two** distinct follow-up questions that logically extend the current EDA:
        1. Understand the given information from the current EDA.
        2. Identify two entities that worth to explore.
        3. Write specific follow-up questions **(no more than 15 words each)** for these two entities that:
            - The question drill down to reveal the pattern within this entity.
            - In addition to the numerical column(for Y-axis), the question must refer to two column(for X-axis) **ONLY**.
            - Ensure the two questions are use **the same** combination of columns.
            - Make them answerable with the existing dataset.
        4. Write a title for the chart **(no more than 12 words each)** based on the questions.

        **Constraints**
        - Never raise the same or similar questions as following:  
          1.{question_1}
          2.{question_2}
          3.{question_3}
          4.{question_4}



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
            input_variables=["key_columns","s1c1_insight","question_1","question_2","question_3","question_4"]
        )
        eda_q_for_sec3_chain = eda_q_for_sec3_prompt | llm
        eda_q_for_sec3_result = eda_q_for_sec3_chain.invoke(input = {

                                                        "key_columns":chosen_data_schema,
                                                        "s1c1_insight":insight_from_fact_to_llm[0],
                                                        "question_1": chart_query[0],
                                                        "question_2": chart_query[1],
                                                        "question_3": chart_query[2],
                                                        "question_4": chart_query[3],

                                                        })
        eda_q_for_sec3_json = parse_jsonish(eda_q_for_sec3_result.content)
        # st.write(eda_q_for_sec3_result.content)
        # eda_q_for_sec3_json = json.loads(eda_q_for_sec3_result.content)
        st.write("Follow-up Questions sec 3:",eda_q_for_sec3_json)
        chart_query.append(eda_q_for_sec3_json["follow_up_questions"][0]["question"])
        chart_query.append(eda_q_for_sec3_json["follow_up_questions"][1]["question"])
        chart_title.append(eda_q_for_sec3_json["follow_up_questions"][0]["suggested_viz_title"])
        chart_title.append(eda_q_for_sec3_json["follow_up_questions"][1]["suggested_viz_title"])

        
        llm_1 = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", temperature=0,api_key = openapi_key)
        # Use llm to describe the charts based on the chart image
        chart_pattern = []
        for i in range(0,6):            
            binary_chart     = open(f"DATA2Poster_chart/image{i}.png", 'rb').read()  # fc aka file_content
            base64_utf8_chart = base64.b64encode(binary_chart ).decode('utf-8')
            img_url = f'data:image/png;base64,{base64_utf8_chart}' 
            chart_prompt_template = load_prompt_from_file("prompt_templates/chart_prompt.txt")
            chart_des_prompt = [
                SystemMessage(content=chart_prompt_template),
                HumanMessage(content=[
                    {
                        "type": "text", 
                        "text": f"This chart is ploted  based on this question:\n\n {chart_query[i]}.\n\n"
                    },
                    {
                            "type": "text", 
                            "text": "Here is the chart to describe:"
                    },
                    {
                            "type": "image_url",
                            "image_url": {
                                "url": img_url
                            },
                    },
                ])
            ]
            chart_des =  llm_1.invoke(chart_des_prompt)
            st.write(f'**Description for Chart {i}:**', f'**{chart_des.content}**')
            chart_pattern.append(chart_des.content)


        

        img_to_llm_list = []
        for i in range(0,6):
            img_to_llm_list.append({"chart_id": i, "title": chart_title[i]})
           
        
        # ////zero shot//// Use llm to write section header and section insight based on the chart titles           
        chart_check_prompt ="""
                                    You are an expert at capture interesting insight from visualization charts.
                                    You are building a 3-section poster.
                                    You are given only the chart titles and its id listed below:{img_to_llm_list}
                                    
                                    ** Your Tasks(Think step by step)**
                                    1. Read and evaluate each chart title for subject matter, variables, and implied insight.  
                                    2. Create a concise section heading that captures the shared insight of each group.
                                    3. For each section, write a one sentence (shorter than 25 words) that captures the key insight conveyed in that section.

                                    **Output (JSON)**
                                    Do not INCLUDE ```json```.Do not add other sentences after this json data.
                                    Return **only** the final JSON in this structure:
                                    {{
                                    "sections": [
                                        {{
                                        "section": "A",
                                        "heading": "<section heading>",
                                        "charts_id": ["0", "1"],
                                        "charts_title": ["<chart title 1>", "<chart title 2>"],
                                        "insight": "<one-sentence synthesis>"
                                        }},
                                        {{
                                        "section": "B",
                                        "heading": "...",
                                        "charts_id": ["2", "3"],
                                        "charts_title": ["<chart title 1>", "<chart title 2>"],
                                        "insight": "..."
                                        }},
                                        {{
                                        "section": "C",
                                        "heading": "...",
                                        "charts_id": ["4", "5"],
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
        chart_check_json = parse_jsonish(chart_check.content)
        st.write("Chart Check JSON:",chart_check_json)
        chart_id_list = []
        section_insight_list = []
        section_header_list = []

        for section in chart_check_json["sections"]:
            section_insight_list.append(section["insight"])
            section_header_list.append(section["heading"])
            for chart_id in section["charts_id"]:
                chart_id_list.append(int(chart_id))
        rag_spec = []
        for id in chart_id_list:
            with open ("vis_nl_pair.json", "r") as f:
                vis_nl_json = json.load(f)
                query = chart_title[id]
                # st.write(f'**Question for Chart {id}:**',f'**{query}**')
                st.write("Chart Title:",chart_title[id])
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
        
                       
     
        vlspec = agent_consistent(chosen_dataset,chart_query,chart_title,chosen_data_schema,sample_data, rag_spec, openapi_key)
        st.write("Vega-Lite Specification:",vlspec)
        cor_vlspec = parse_jsonish(vlspec)  # -> dict/list or raises clear ValueError
        # json_vlspec = json.loads(cor_vlspec)
        spec_id = 0
        for spec in cor_vlspec["visualizations"]:
            with open(f"DATA2Poster_json/vlspec1_{spec_id}.json", "w") as f:
                json.dump(spec, f, indent=2)
            spec["height"] =600
            spec["width"] =800
            # Sort s1c1 and s2c1 charts by y-axis
            if spec_id ==0 or spec_id == 2:
                spec["encoding"]["x"]["sort"] = "-y"
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

                error_chain = error_prompt | llm_1 
                
                # Invoke the chain to fix the code
                corrected_vlspec = error_chain.invoke(input={"error_code": spec, "error": error})
                json_corrected_vlspec = parse_jsonish(corrected_vlspec.content)
                json_corrected_vlspec["height"] =600
                json_corrected_vlspec["width"] =800
                final_chart = alt.Chart.from_dict(json_corrected_vlspec)
                final_chart.save(f"DATA2Poster_chart/image{spec_id}.png")
                st.image(f"DATA2Poster_chart/image{spec_id}.png", caption="Chart "+str(spec_id))
            spec_id += 1
       
        
        # Generate data facts from s1c2, s2c2, s3c1, s3c2 charts 
        insight_from_fact = []

        for i in range(0,6):
                facts_1_1 = []
                facts_2_1 = []
                title = chart_title[i]
                st.write("chart_title:",title)
            # if i == 2:
            #     continue
            # else:
                with open(f"DATA2Poster_json/vlspec1_{i}.json", "r") as f:
                    spec_json = json.load(f)
      
                data_transform_prompttemplate = PromptTemplate(
                    template=data_transform_prompt,
                    input_variables = {"trans_json" })
                data_transform_chain = data_transform_prompttemplate | llm_1
                data_transform_result = data_transform_chain.invoke(input={"trans_json":spec_json["transform"]})
                trans_code = "import pandas as pd\n\n"+f"df = pd.read_csv('data/{chosen_dataset}.csv')\n\n"+data_transform_result.content+"\n\ntrans_df = trans_data(df)\n\ntrans_df.to_csv('DATA2Poster_df/transformed_df.csv', index=False)\n\n"
                exec(trans_code)
                new_df = pd.read_csv('DATA2Poster_df/transformed_df.csv')
                # Drop columns whose name contains "rank"
                cols_to_drop = [col for col in new_df.columns if "rank" in col]
                new_df = new_df.drop(columns=cols_to_drop)
                st.write("Transformed DataFrame:",new_df)
                new_df_schema = get_column_properties(new_df)        
                raw_facts =[]
                C_col =[ col for col in new_df_schema if col["properties"]["dtype"] == "C" ]
                T_col =[ col for col in new_df_schema if col["properties"]["dtype"] == "T" ]
                N_col =[ col for col in new_df_schema if col["properties"]["dtype"] == "N" ]
                for j in range(len(C_col)):
                    if len(N_col) >=2:
                        for k in range(len(N_col)):
                            facts_1_1 = generate_facts(
                                        dataset=Path(f"DATA2Poster_df/transformed_df.csv"),
                                        breakdown=C_col[j]["column"],
                                        measure=N_col[k]["column"],
                                        series=C_col[j]["column"],
                                        breakdown_type="C",
                                        measure_type="N",
                                        with_vis=False,
                                    )
                        
                            if len(T_col) >=1:
                                facts_2_1 = generate_facts(
                                            dataset=Path(f"DATA2Poster_df/transformed_df.csv"),
                                            breakdown=C_col[j]["column"],
                                            measure=N_col[k]["column"],
                                            series=T_col[0]["column"],
                                            breakdown_type="T",
                                            measure_type="N",
                                            with_vis=False,
                                        )
                        for fact in facts_1_1:
                            raw_facts.append({"content":fact["content"], "score":fact["score_C"]})
                        if len(T_col) >=1:
                            for fact in facts_2_1:
                                raw_facts.append({"content":fact["content"], "score":fact["score_C"]})
                    elif len(N_col) ==1:
                        facts_1_1 = generate_facts(
                                        dataset=Path(f"DATA2Poster_df/transformed_df.csv"),
                                        breakdown=C_col[j]["column"],
                                        measure=N_col[0]["column"],
                                        series=C_col[j]["column"],
                                        breakdown_type="C",
                                        measure_type="N",
                                        with_vis=False,
                                    )
                        if len(T_col) >=1:
                            facts_2_1 = generate_facts(
                                        dataset=Path(f"DATA2Poster_df/transformed_df.csv"),
                                        breakdown=C_col[j]["column"],
                                        measure=N_col[0]["column"],
                                        series=T_col[0]["column"],
                                        breakdown_type="T",
                                        measure_type="N",
                                        with_vis=False,
                                        )
                        for fact in facts_1_1:
                            raw_facts.append({"content":fact["content"], "score":fact["score_C"]})
                        if len(T_col) >=1:
                            for fact in facts_2_1:
                                raw_facts.append({"content":fact["content"], "score":fact["score_C"]})
                
                # Ranking facts by score
                raw_facts = sorted(raw_facts, key=itemgetter('score'), reverse=True)
                seen = set()
                
                clean_facts = []
                for item in raw_facts:
                    if i==0 or i ==1:
                        if item["content"] != "No fact." and col_select_json["key_columns"][0]["column"] in item["content"] and item["content"] not in seen:
                            seen.add(item["content"])
                            clean_facts.append(item["content"])
                    else:
                        if item["content"] != "No fact." and col_select_json["key_columns"][0]["column"] not in item["content"] and col_select_json["key_columns"][1]["column"] in item["content"] and item["content"] not in seen:
                            seen.add(item["content"])
                            clean_facts.append(item["content"])
                st.write("Clean Facts:",clean_facts)
                
                if i ==0 or i == 1:
                    insight_from_sc = self_augmented_knowledge(openai_key, chosen_dataset, col_select_json["key_columns"][0]["column"], clean_facts)
                    insight_from_fact.append(insight_from_sc)
                else:
                    insight_from_sc = self_augmented_knowledge(openai_key, chosen_dataset, col_select_json["key_columns"][1]["column"], clean_facts)
                    insight_from_fact.append(insight_from_sc)
        st.write("Insight from all charts:",insight_from_fact)
            
            
            

        chartid_for_pdf = []
        chart_for_pdf = []
        
        for id in range(0,spec_id):
            chartid_for_pdf.append(id)
            binary_chart     = open(f"DATA2Poster_chart/image{id}.png", 'rb').read()
            base64_utf8_chart = base64.b64encode(binary_chart).decode('utf-8')
            img_url = f'data:image/png;base64,{base64_utf8_chart}'              
            chart_for_pdf.append(img_url)
        
    
        # Reset session state
        st.session_state["bt_try"] = ""  
        st.session_state["s1c1_fact"] = []
        st.session_state["s2c1_fact"] = []
        
        # Create pdf and download
        pdf_title = f"{chosen_dataset} Poster"
        create_pdf(chosen_dataset, chart_pattern,insight_from_fact,section_insight_list, chartid_for_pdf,chart_for_pdf,col_select_json["key_columns"][0]["column"], col_select_json["key_columns"][1]["column"], eda_q_for_sec3_json["follow_up_questions"][0]["entity"],eda_q_for_sec3_json["follow_up_questions"][1]["entity"], section_header_list,openapi_key)
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

