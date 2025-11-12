import streamlit as st
import pandas as pd
import re
import base64
import json
import itertools
import altair as alt
import random
from collections import Counter
from operator import itemgetter
from nl4dv import NL4DV
from dataSchema_builder import get_column_properties
from data_cleaner import clean_csv_data
from json_sanitizer import parse_jsonish
from insight_generation.dataFact_scoring import score_importance
from insight_generation.main import generate_facts
from Fact_explorer import agent1_column_selector, agent2_fact_summarizer, agent3_s1q_generator, agent4_s2q_generator, agent5_s3q_generator
from Poster_generator import agent6_vis_generator, agent6_sec_vis_generator, agent6_vis_corrector, agent6_vis_refiner, agent7_vis_describer, agent8_dt_extractor, agent9_section_designer, agent10_pdf_creator, agent10_sec_pdf_creator
from Poster_evaluator import agent11_vis_recommender, agent12_final_checker
from pathlib import Path
# Import langchain modules
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
import os, json, time, datetime
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
   
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
    datasets["cars"] = pd.read_csv("data/cars.csv")
    datasets["onlinefoods"] = pd.read_csv("data/onlinefoods.csv")


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

    # Default the radio button to the newly added dataset
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

# Page Content 
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

# Preprocess the code generated by GPT //////suspended
def preprocess_json(code: str, count: str) -> str:
    """Preprocess code to remove any preamble and explanation text"""

    code = code.replace("<imports>", "")
    code = code.replace("<stub>", "")
    code = code.replace("<transforms>", "")

    # Remove all text after chart = plot(data)
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
        # Return only text after the first import statement
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

# For code generated by nl4dv //////suspended
def preprocess_json_2(code: str, count: str) -> str:
    """Preprocess code to remove any preamble and explanation text"""

    code = code.replace("<imports>", "")
    code = code.replace("<stub>", "")
    code = code.replace("<transforms>", "")

    # Remove all text after chart = plot(data)
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
        # Return only text after the first import statement
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

def load_json(json_file):
    with open(json_file, "r", encoding="utf-8") as fh:

        return json.load(fh)

def insight_list_generator():
       # Generate data facts from each chart and summarize the insights using llm
        insight_from_fact = []
        for i in range(0,6):
                facts_1_1 = []
                facts_2_1 = []
                title = chart_title[i]
                st.write("chart_title:",title)
                with open(f"DATA2Poster_json/vlspec1_{i+1}.json", "r") as f:
                    spec_json = json.load(f)
                data_transform_result = agent8_dt_extractor(spec_json["transform"], openai_key)
                trans_code = "import pandas as pd\n\n"+f"df = pd.read_csv('data/{chosen_dataset}.csv')\n\n"+data_transform_result+"\n\ntrans_df = trans_data(df)\n\ntrans_df.to_csv('DATA2Poster_df/transformed_df.csv', index=False)\n\n"
                exec(trans_code)
                new_df = pd.read_csv('DATA2Poster_df/transformed_df.csv')
                # Drop columns whose name contains "rank"
                cols_to_drop = [col for col in new_df.columns if "rank" in col]
                new_df = new_df.drop(columns=cols_to_drop)
                # st.write("Transformed DataFrame:",new_df)
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
                    if "highest" in item["content"] or "lowest" in item["content"] or "100%" in item["content"] or "100.00%" in item["content"]:
                        continue
                    if i ==4 or i ==5:
                        if item["content"] != "No fact." not in item["content"] and f"""when {col_select_json["key_columns"][0]["column"]}""" not in item["content"] and item["content"] not in seen:
                            seen.add(item["content"])
                            clean_facts.append(item["content"])
                    elif item["content"] != "No fact."  and item["content"] not in seen:
                        seen.add(item["content"])
                        clean_facts.append(item["content"])           
                st.write("Clean Facts:",clean_facts)
                if i ==2 or i ==3:
                    insight_from_sc = agent2_fact_summarizer(chosen_dataset, col_select_json["key_columns"][1]["column"], clean_facts, chart_title[i], openai_key)
                    insight_from_fact.append(insight_from_sc)
                else:
                    insight_from_sc = agent2_fact_summarizer(chosen_dataset, col_select_json["key_columns"][0]["column"], clean_facts, chart_title[i], openai_key)
                    insight_from_fact.append(insight_from_sc)
        return insight_from_fact
 
# Check if the user has tried and entered an OpenAI key
api_keys_entered = True  # Assume the user has entered an OpenAI key
if try_true or (st.session_state["bt_try"] == "T"):
    # if not openai_key.startswith("sk-"):
    #             st.error("Please enter a valid OpenAI API key.")
    #             api_keys_entered = False

    st.session_state["bt_try"] = "T"
    
    if api_keys_entered:
        # Use GPT as llm
        llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14",api_key = openai_key)
        # llm = ChatGoogleGenerativeAI(
        #     model="gemini-2.5-flash",
        #     temperature=0,
        #     max_tokens=None,
        #     api_key = openai_key
        #     # other params...
        # )

        # Use OpenAIEmbeddings as embedding model
        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key = openai_key)
        # Create a vector store
        data_list = load_json('vis_corpus.json')
        docs = [Document(page_content=json.dumps(item)) for item in data_list]
        vectorstore = FAISS.from_documents(
        docs,
        embeddings_model
        )

        # ////zero shot//// Select 4 columns from the chosen dataset
        col_select_result = agent1_column_selector(chosen_data_schema, openai_key)
        col_select_json = parse_jsonish(col_select_result)
        st.write("Selected Columns:",col_select_json)
        
        facts_list = []
        # Use 4 columns to generate facts
        desired_combinations = ['CNC', 'TNC', 'CNN', 'TNN']
        result = organize_by_dtype_combinations(col_select_json["key_columns"], desired_combinations) 
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
        
        # Ranking facts by score (meaning?)
        facts_list = sorted(facts_list, key=itemgetter('score'), reverse=True) 
        facts_list = [fact for fact in facts_list if fact["score"] > 1]
        # st.write("Filtered and Ranked Facts:",facts_list)
        seen = set()
        seen_1 = set()
        for item in facts_list:
            if item["content"] != "No fact." and col_select_json["key_columns"][0]["column"] in item["content"] and item["content"] not in seen:
                seen.add(item["content"])
                st.session_state["s1c1_fact"].append(item["content"])
            elif item["content"] != "No fact." and col_select_json["key_columns"][1]["column"] in item["content"] and item["content"] not in seen_1:
                seen_1.add(item["content"])
                st.session_state["s2c1_fact"].append(item["content"])
      
        st.write("s1c1 Facts:",st.session_state["s1c1_fact"])
        st.write("s2c1 Facts:",st.session_state["s2c1_fact"])

        insight_from_fact_to_llm=[]

        # Use llm to summarize the facts to insights
        insight_from_s1c1 = agent2_fact_summarizer(chosen_dataset, col_select_json["key_columns"][0]["column"], st.session_state["s1c1_fact"], "", openai_key)
        insight_from_fact_to_llm.append(insight_from_s1c1)
        insight_from_s2c1 = agent2_fact_summarizer(chosen_dataset, col_select_json["key_columns"][1]["column"], st.session_state["s2c1_fact"], "", openai_key)
        insight_from_fact_to_llm.append(insight_from_s2c1)
        st.write("Insights from Facts:",insight_from_fact_to_llm)

        # Random sample 100 facts for passing to LLM 
        if len(st.session_state["s1c1_fact"]) > 100:
            st.session_state["s1c1_fact"] = random.sample(st.session_state["s1c1_fact"], 100)
        if len(st.session_state["s2c1_fact"]) > 100:
            st.session_state["s2c1_fact"] = random.sample(st.session_state["s2c1_fact"], 100)

        chart_title = []
        chart_query = []

        # ////one shot//// Use llm to read the base chart description to generate section 1 follow-up questions
        eda_q_for_sec1_result = agent3_s1q_generator(insight_from_fact_to_llm[0],str(st.session_state["s1c1_fact"]),col_select_json["key_columns"][0]["column"],str([col["column"] for col in col_select_json["key_columns"]]),openai_key)                                               
        eda_q_for_sec1_json = parse_jsonish(eda_q_for_sec1_result)
        st.write("Follow-up Questions sec 1:",eda_q_for_sec1_json)
        chart_query.append(eda_q_for_sec1_json["follow_up_questions"][0]["question"])
        chart_query.append(eda_q_for_sec1_json["follow_up_questions"][1]["question"])
        chart_title.append(eda_q_for_sec1_json["follow_up_questions"][0]["suggested_viz_title"])
        chart_title.append(eda_q_for_sec1_json["follow_up_questions"][1]["suggested_viz_title"])
        
        #////one shot//// Use llm to read the base chart description to generate section 2 follow-up questions
        eda_q_for_sec2_result = agent4_s2q_generator(insight_from_fact_to_llm[1],eda_q_for_sec1_json["follow_up_questions"][0]["question"],eda_q_for_sec1_json["follow_up_questions"][1]["question"],str(st.session_state["s2c1_fact"]),col_select_json["key_columns"][1]["column"],str([col["column"] for col in col_select_json["key_columns"]]),openai_key)
        eda_q_for_sec2_json = parse_jsonish(eda_q_for_sec2_result)
        st.write("Follow-up Questions sec 2:",eda_q_for_sec2_json)
        chart_query.append(eda_q_for_sec2_json["follow_up_questions"][0]["question"])
        chart_query.append(eda_q_for_sec2_json["follow_up_questions"][1]["question"])
        chart_title.append(eda_q_for_sec2_json["follow_up_questions"][0]["suggested_viz_title"])
        chart_title.append(eda_q_for_sec2_json["follow_up_questions"][1]["suggested_viz_title"])

        # ////zero shot//// Use llm to read the insights from s1c1 facts to generate section 3 follow-up questions       
        eda_q_for_sec3_result = agent5_s3q_generator(insight_from_fact_to_llm[0],chart_query[0],chart_query[1],chart_query[2],chart_query[3],str([col["column"] for col in col_select_json["key_columns"]]),openai_key)
        eda_q_for_sec3_json = parse_jsonish(eda_q_for_sec3_result)
        st.write("Follow-up Questions sec 3:",eda_q_for_sec3_json)
        chart_query.append(eda_q_for_sec3_json["follow_up_questions"][0]["question"])
        chart_query.append(eda_q_for_sec3_json["follow_up_questions"][1]["question"])
        chart_title.append(eda_q_for_sec3_json["follow_up_questions"][0]["suggested_viz_title"])
        chart_title.append(eda_q_for_sec3_json["follow_up_questions"][1]["suggested_viz_title"])

        logs_dir = "logs"
        os.makedirs(logs_dir, exist_ok=True)

        log_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "col_select_json": col_select_json,
            "eda_q_for_sec1_json": eda_q_for_sec1_json,
            "eda_q_for_sec2_json": eda_q_for_sec2_json, 
            "eda_q_for_sec3_json": eda_q_for_sec3_json,
            "insight_from_fact_to_llm": insight_from_fact_to_llm,
           
        }

        filename = f"eda_q_{chosen_dataset}_{int(time.time())}.json"
        path = os.path.join(logs_dir, filename)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(log_entry, fh, ensure_ascii=False, indent=2)

        # RAG + llm to generate vlspec for each chart based on the chart title  
        chart_id_list = [1,2,3,4,5,6] 
        rag_spec = []
        for id in chart_id_list:
            with open ("vis_nl_pair.json", "r") as f:
                vis_nl_json = json.load(f)
                query = chart_title[id-1]
                # st.write("Chart Title:",chart_title[id-1])
                # RAG to extract vlspec of similar questions from the vectorstore
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
                        # st.write("vlspec:",sample_vlspec)
                        rag_spec.append(sample_vlspec)
                        break
        
        vlspec = agent6_vis_generator(chosen_dataset,chart_query,chart_title,chosen_data_schema,sample_data, rag_spec, openai_key)
        st.write("Vega-Lite Specification:",vlspec)
        cor_vlspec = parse_jsonish(vlspec)  
        # Save each chart's vlspec to a json file and generate chart image using altair
        spec_id = 1
        for spec in cor_vlspec["visualizations"]:
            with open(f"DATA2Poster_json/vlspec1_{spec_id}.json", "w") as f:
                json.dump(spec, f, indent=2)
            spec["height"] =600
            spec["width"] =800
            # Sort s1c1 and s2c1 charts by y-axis???#####################################
            # if spec_id ==0 or spec_id == 2:
            #     spec["encoding"]["x"]["sort"] = "-y"
            try:
                chart = alt.Chart.from_dict(spec)
                chart.save(f"DATA2Poster_chart/image{spec_id}.png")
                st.image(f"DATA2Poster_chart/image{spec_id}.png", caption="Chart "+str(spec_id))

            except Exception as e:
                error = str(e)
                print(f"\n🔴 Error encountered: {error}\n")
                corrected_vlspec = agent6_vis_corrector(spec, error, openai_key)
                json_corrected_vlspec = parse_jsonish(corrected_vlspec)
                json_corrected_vlspec["height"] =600
                json_corrected_vlspec["width"] =800
                final_chart = alt.Chart.from_dict(json_corrected_vlspec)
                final_chart.save(f"DATA2Poster_chart/image{spec_id}.png")
                with open(f"DATA2Poster_json/vlspec1_{spec_id}.json", "w") as f:
                    json.dump(json_corrected_vlspec, f, indent=2)
                st.image(f"DATA2Poster_chart/image{spec_id}.png", caption="Chart "+str(spec_id))
            spec_id += 1
        
        # Refine the chart  using llm
        for i in range(1, 7):
            binary_chart = open(f"DATA2Poster_chart/image{i}.png", 'rb').read()
            base64_utf8_chart = base64.b64encode(binary_chart).decode('utf-8')
            img_url = f'data:image/png;base64,{base64_utf8_chart}'
            with open(f"DATA2Poster_json/vlspec1_{i}.json", "r") as f:
                vlspec = json.load(f)
            refined_vlspec = agent6_vis_refiner(vlspec, chart_query[i-1], img_url, openai_key)
            
            try:
                chart = alt.Chart.from_dict(refined_vlspec)
                chart.save(f"DATA2Poster_chart/image{i}.png")
                st.image(f"DATA2Poster_chart/image{i}.png", caption="Chart "+str(i))

            except Exception as e:
                error = str(e)
                print(f"\n🔴 Error encountered: {error}\n")
                corrected_vlspec = agent6_vis_corrector(refined_vlspec, error, openai_key)
                json_corrected_vlspec = parse_jsonish(corrected_vlspec)
                json_corrected_vlspec["height"] =600
                json_corrected_vlspec["width"] =800
                final_chart = alt.Chart.from_dict(json_corrected_vlspec)
                final_chart.save(f"DATA2Poster_chart/image{i}.png")
                with open(f"DATA2Poster_json/vlspec1_{i}.json", "w") as f:
                    json.dump(json_corrected_vlspec, f, indent=2)
                st.image(f"DATA2Poster_chart/image{i}.png", caption="Chart "+str(i))

        # Use llm to describe the charts based on the chart image
        chart_pattern = []
        for i in range(1,7):            
            binary_chart     = open(f"DATA2Poster_chart/image{i}.png", 'rb').read()  
            base64_utf8_chart = base64.b64encode(binary_chart ).decode('utf-8')
            img_url = f'data:image/png;base64,{base64_utf8_chart}' 
            chart_des =  agent7_vis_describer(chart_query[i-1], img_url, openai_key)
            st.write(f'**Description for Chart {i}:**', f'**{chart_des}**')
            chart_pattern.append(chart_des)
        
        # Generate data facts from each chart and summarize the insights using llm
        insight_from_fact = insight_list_generator()
        # st.write("Insight from all charts:", insight_from_fact)

        # Log chart_pattern and insight_from_fact to JSON
        logs_dir = "logs"
        os.makedirs(logs_dir, exist_ok=True)

        payload = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "chart_title_list": chart_title,
            "chart_pattern": chart_pattern,
            "insight_from_fact": insight_from_fact
        }

        filename = f"chartdes_insight_{chosen_dataset}_{int(time.time())}.json"
        path = os.path.join(logs_dir, filename)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

        # Image titles for llm
        img_to_llm_list = []
        for i in range(0,6):
            img_to_llm_list.append(chart_title[i])
        # ////zero shot//// Use llm to write section header and section insight based on the chart titles           
        section_result = agent9_section_designer(img_to_llm_list, insight_from_fact, openai_key)
        st.write("section_result:",section_result)
        section_json = parse_jsonish(section_result)
        st.write("Section Check JSON:",section_json)

        # Create pdf and download
        section_insight_list = []
        section_header_list = []   
        for section in section_json["sections"]:
            section_insight_list.append(section["insight"])
            section_header_list.append(section["heading"])
        chartid_for_pdf = []
        chart_for_pdf = []    
        for id in range(1,7):
            chartid_for_pdf.append(id)
            binary_chart     = open(f"DATA2Poster_chart/image{id}.png", 'rb').read()
            base64_utf8_chart = base64.b64encode(binary_chart).decode('utf-8')
            img_url = f'data:image/png;base64,{base64_utf8_chart}'              
            chart_for_pdf.append(img_url)
        conclusion_json,poster_Q = agent10_pdf_creator(chosen_dataset, chart_pattern, insight_from_fact, section_insight_list, chartid_for_pdf, chart_for_pdf, eda_q_for_sec3_json["follow_up_questions"][0]["entity"], eda_q_for_sec3_json["follow_up_questions"][1]["entity"], section_header_list, openai_key)
        # Log section_json, conclusion_json and poster_Q
        logs_dir = "logs"
        os.makedirs(logs_dir, exist_ok=True)

        log_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "section_json": section_json,
            "conclusion_json": conclusion_json,
            "poster_Q": poster_Q,
        }

        filename = f"section_conclusion_{chosen_dataset}_{int(time.time())}.json"
        path = os.path.join(logs_dir, filename)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(log_entry, fh, ensure_ascii=False, indent=2)

        st.write(f"Saved section/conclusion JSON -> {path}")


        # Iterative visualization check
        sec_query = []
        sec_title = []
        sec_rag_spec = []
        sec_chartid_for_pdf = [1,2,3,4,5,6]
        sec_chart_for_pdf = []
        conc_for_pdf = ""
        st.write("Conclusion JSON:",conclusion_json)
        st.write("Poster Title:",poster_Q)
        conc_for_pdf = conclusion_json["conclusion"][0]["content"] + conclusion_json["conclusion"][1]["content"] + conclusion_json["conclusion"][2]["content"]
        vis_checked_1 = agent11_vis_recommender(conclusion_json["conclusion"][0]["content"], eda_q_for_sec1_json["follow_up_questions"][0]["suggested_viz_title"], eda_q_for_sec1_json["follow_up_questions"][1]["suggested_viz_title"], section_insight_list[0], str([col["column"] for col in col_select_json["key_columns"]]), openai_key)
        vis_checked_2 = agent11_vis_recommender(conclusion_json["conclusion"][1]["content"], eda_q_for_sec2_json["follow_up_questions"][0]["suggested_viz_title"], eda_q_for_sec2_json["follow_up_questions"][1]["suggested_viz_title"], section_insight_list[1], str([col["column"] for col in col_select_json["key_columns"]]), openai_key)
        # vis_checked_3 = agent11_vis_recommend(conclusion_json["conclusion"][2]["content"], eda_q_for_sec3_json["follow_up_questions"][0]["suggested_viz_title"], eda_q_for_sec3_json["follow_up_questions"][1]["suggested_viz_title"], openai_key)
        vis_checked_1_json = parse_jsonish(vis_checked_1)
        vis_checked_2_json = parse_jsonish(vis_checked_2)
        # vis_checked_3_json = parse_jsonish(vis_checked_3)
        st.write("Visualization Check Section 1 JSON:",vis_checked_1_json)
        st.write("Visualization Check Section 2 JSON:",vis_checked_2_json)      
        # st.write("Visualization Check Section 3 JSON:",vis_checked_3_json)
        if vis_checked_1_json["vis_check"][0]["replace"] != "none":
            if vis_checked_1_json["vis_check"][0]["replace"] == "chart_1":
                sec_chartid_for_pdf[0] = 11
                img_to_llm_list[0] = vis_checked_1_json["vis_check"][0]["recommendation"]["revised_title"]
            if vis_checked_1_json["vis_check"][0]["replace"] == "chart_2":
                sec_chartid_for_pdf[1] = 11
                img_to_llm_list[1] = vis_checked_1_json["vis_check"][0]["recommendation"]["revised_title"]
            sec_query.append(vis_checked_1_json["vis_check"][0]["recommendation"]["query"])
            sec_title.append(vis_checked_1_json["vis_check"][0]["recommendation"]["revised_title"])
        if vis_checked_2_json["vis_check"][0]["replace"] != "none":
            if vis_checked_2_json["vis_check"][0]["replace"] == "chart_1":
                sec_chartid_for_pdf[2] = 22
                img_to_llm_list[2] = vis_checked_2_json["vis_check"][0]["recommendation"]["revised_title"]
            if vis_checked_2_json["vis_check"][0]["replace"] == "chart_2":
                sec_chartid_for_pdf[3] = 22
                img_to_llm_list[3] = vis_checked_2_json["vis_check"][0]["recommendation"]["revised_title"]
            sec_query.append(vis_checked_2_json["vis_check"][0]["recommendation"]["query"])
            sec_title.append(vis_checked_2_json["vis_check"][0]["recommendation"]["revised_title"])
        # if vis_checked_3_json["vis_check"][0]["replace"] != "none":
        #     if vis_checked_3_json["vis_check"][0]["replace"] == "chart_1":
        #         sec_chartid_for_pdf[4] = 33
        #         img_to_llm_list[4] = vis_checked_3_json["vis_check"][0]["recommendation"]["revised_title"]
        #     if vis_checked_3_json["vis_check"][0]["replace"] == "chart_2":
        #         sec_chartid_for_pdf[5] = 33
        #         img_to_llm_list[5] = vis_checked_3_json["vis_check"][0]["recommendation"]["revised_title"]
        #     sec_query.append(vis_checked_3_json["vis_check"][0]["recommendation"]["query"])
        #     sec_title.append(vis_checked_3_json["vis_check"][0]["recommendation"]["revised_title"])

        # Log visualization check JSONs
        logs_dir = "logs"
        os.makedirs(logs_dir, exist_ok=True)

        vis_check_log = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "vis_checked_1_json": vis_checked_1_json,
            "vis_checked_2_json": vis_checked_2_json,
        }

        log_filename = f"vis_checked_{chosen_dataset}_{int(time.time())}.json"
        log_path = os.path.join(logs_dir, log_filename)
        with open(log_path, "w", encoding="utf-8") as fh:
            json.dump(vis_check_log, fh, ensure_ascii=False, indent=2)
       

        for id in range(len(sec_query)):
            with open ("vis_nl_pair.json", "r") as f:
                    vis_nl_json = json.load(f)
                    query = sec_title[id]
                    # st.write(f'**Question for Chart {id}:**',f'**{query}**')
                    # st.write("Chart Title:",sec_title[id])
                    # RAG to extract vlspec of similar questions from the vectorstore
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
                            # st.write("vlspec:",sample_vlspec)
                            sec_rag_spec.append(sample_vlspec)
                            break
        sec_spec_list = []
        for i in range(len(sec_query)):
            sec_vlspec = agent6_sec_vis_generator(chosen_dataset,sec_query[i],sec_title[i],chosen_data_schema,sample_data, sec_rag_spec, openai_key)
            st.write("Vega-Lite Specification:",sec_vlspec)
            sec_cor_vlspec = parse_jsonish(sec_vlspec)  # -> dict/list or raises clear ValueError
            sec_spec_list.append(sec_cor_vlspec["visualizations"][0])

        for spec_id in range(len(sec_spec_list)):
            with open(f"DATA2Poster_json/vlspec2_{spec_id}.json", "w") as f:
                json.dump(sec_spec_list[spec_id], f, indent=2)
            sec_spec_list[spec_id]["height"] =600
            sec_spec_list[spec_id]["width"] =800
            try:
                chart = alt.Chart.from_dict(sec_spec_list[spec_id])
                chart.save(f"DATA2Poster_chart/image{spec_id+1}{spec_id+1}.png")
                st.image(f"DATA2Poster_chart/image{spec_id+1}{spec_id+1}.png", caption="Chart "+str(spec_id+1))

            except Exception as e:
                error = str(e)
                print(f"\n🔴 Error encountered: {error}\n")
                corrected_vlspec = agent6_vis_corrector(spec, error, openai_key)
                json_corrected_vlspec = parse_jsonish(corrected_vlspec)
                json_corrected_vlspec["height"] =600
                json_corrected_vlspec["width"] =800
                final_chart = alt.Chart.from_dict(json_corrected_vlspec)
                final_chart.save(f"DATA2Poster_chart/image{spec_id+1}{spec_id+1}.png")
                with open(f"DATA2Poster_json/vlspec2_{spec_id}.json", "w") as f:
                    json.dump(json_corrected_vlspec , f, indent=2)
                st.image(f"DATA2Poster_chart/image{spec_id+1}{spec_id+1}.png", caption="Chart "+str(spec_id+1))

        # Refine the chart  using llm
        for spec_id in range(len(sec_spec_list)): 
            binary_chart = open(f"DATA2Poster_chart/image{spec_id+1}{spec_id+1}.png", 'rb').read()
            base64_utf8_chart = base64.b64encode(binary_chart).decode('utf-8')
            img_url = f'data:image/png;base64,{base64_utf8_chart}'
            with open(f"DATA2Poster_json/vlspec2_{spec_id+1}.json", "r") as f:
                vlspec = json.load(f)
            refined_vlspec = agent6_vis_refiner(vlspec, sec_query[spec_id], img_url, openai_key)

            try:
                chart = alt.Chart.from_dict(refined_vlspec)
                chart.save(f"DATA2Poster_chart/image{spec_id+1}{spec_id+1}.png")
                st.image(f"DATA2Poster_chart/image{spec_id+1}{spec_id+1}.png", caption="Chart "+str(spec_id+1))

            except Exception as e:
                error = str(e)
                print(f"\n🔴 Error encountered: {error}\n")
                corrected_vlspec = agent6_vis_corrector(refined_vlspec, error, openai_key)
                json_corrected_vlspec = parse_jsonish(corrected_vlspec)
                json_corrected_vlspec["height"] =600
                json_corrected_vlspec["width"] =800
                final_chart = alt.Chart.from_dict(json_corrected_vlspec)
                final_chart.save(f"DATA2Poster_chart/image{spec_id+1}{spec_id+1}.png")
                with open(f"DATA2Poster_json/vlspec2_{spec_id+1}.json", "w") as f:
                    json.dump(json_corrected_vlspec , f, indent=2)
                st.image(f"DATA2Poster_chart/image{spec_id+1}{spec_id+1}.png", caption="Chart "+str(spec_id+1))

        st.write("chartid_for_pdf:",sec_chartid_for_pdf )
        for id in sec_chartid_for_pdf:
            binary_chart     = open(f"DATA2Poster_chart/image{id}.png", 'rb').read()
            base64_utf8_chart = base64.b64encode(binary_chart).decode('utf-8')
            img_url = f'data:image/png;base64,{base64_utf8_chart}'              
            sec_chart_for_pdf.append(img_url)
        
        insight_from_fact = insight_list_generator()
        st.write("Insight from all charts:", insight_from_fact)

        #  Second generation section header and section insight based on the chart titles
        sec_section_result = agent9_section_designer(img_to_llm_list, insight_from_fact, openai_key)
        sec_section_json = parse_jsonish(sec_section_result)
        st.write("Chart Check JSON:",sec_section_json)
        sec_section_insight_list = []
        sec_section_header_list = []
        for section in sec_section_json["sections"]:
            sec_section_insight_list.append(section["insight"])
            sec_section_header_list.append(section["heading"])
        # Score the poster quality
        iter_flag = 1
        poster_score = agent12_final_checker(conc_for_pdf, img_to_llm_list, sec_section_insight_list, openai_key)
        poster_score_json = parse_jsonish(poster_score)
        st.write("Poster Score JSON:",poster_score_json)
        # Log poster score JSON
        logs_dir = "logs"
        os.makedirs(logs_dir, exist_ok=True)
        filename = f"poster_score_{chosen_dataset}_{int(time.time())}.json"
        path = os.path.join(logs_dir, filename)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(poster_score_json, fh, ensure_ascii=False, indent=2)
        
        if poster_score_json["final_score"] >= 90:
            agent10_sec_pdf_creator(chosen_dataset, sec_section_insight_list, sec_chartid_for_pdf, sec_chart_for_pdf,conc_for_pdf,sec_section_header_list, openai_key)
            st.success("Poster has been created successfully!🎉")
            with open(f"pdf/{chosen_dataset}_summary_2.pdf", "rb") as f:  
                st.download_button("Download Poster as PDF", f, f"""{chosen_dataset}_summary_2.pdf""")
        else:
            # Iterative 2 more times to create final poster pdf
            for i in range(2):
                iter_flag += 1
                sec_query = []
                sec_title = []
                sec_rag_spec = []
                sec_chartid_for_pdf = [1,2,3,4,5,6]
                sec_chart_for_pdf = []
                vis_checked_1 = agent11_vis_recommender(conclusion_json["conclusion"][0]["content"], eda_q_for_sec1_json["follow_up_questions"][0]["suggested_viz_title"], eda_q_for_sec1_json["follow_up_questions"][1]["suggested_viz_title"], section_insight_list[0], str([col["column"] for col in col_select_json["key_columns"]]), openai_key)
                vis_checked_2 = agent11_vis_recommender(conclusion_json["conclusion"][1]["content"], eda_q_for_sec2_json["follow_up_questions"][0]["suggested_viz_title"], eda_q_for_sec2_json["follow_up_questions"][1]["suggested_viz_title"], section_insight_list[1], str([col["column"] for col in col_select_json["key_columns"]]), openai_key)
                # vis_checked_3 = agent11_vis_recommend(conclusion_json["conclusion"][2]["content"], eda_q_for_sec3_json["follow_up_questions"][0]["suggested_viz_title"], eda_q_for_sec3_json["follow_up_questions"][1]["suggested_viz_title"], openai_key)
                vis_checked_1_json = parse_jsonish(vis_checked_1)
                vis_checked_2_json = parse_jsonish(vis_checked_2)
                # vis_checked_3_json = parse_jsonish(vis_checked_3)
                st.write("Visualization Check Section 1 JSON:",vis_checked_1_json)
                st.write("Visualization Check Section 2 JSON:",vis_checked_2_json)      
                # st.write("Visualization Check Section 3 JSON:",vis_checked_3_json)
                if vis_checked_1_json["vis_check"][0]["replace"] != "none":
                    if vis_checked_1_json["vis_check"][0]["replace"] == "chart_1":
                        sec_chartid_for_pdf[0] = 11
                        img_to_llm_list[0] = vis_checked_1_json["vis_check"][0]["recommendation"]["revised_title"]
                    if vis_checked_1_json["vis_check"][0]["replace"] == "chart_2":
                        sec_chartid_for_pdf[1] = 11
                        img_to_llm_list[1] = vis_checked_1_json["vis_check"][0]["recommendation"]["revised_title"]
                    sec_query.append(vis_checked_1_json["vis_check"][0]["recommendation"]["query"])
                    sec_title.append(vis_checked_1_json["vis_check"][0]["recommendation"]["revised_title"])
                if vis_checked_2_json["vis_check"][0]["replace"] != "none":
                    if vis_checked_2_json["vis_check"][0]["replace"] == "chart_1":
                        sec_chartid_for_pdf[2] = 22
                        img_to_llm_list[2] = vis_checked_2_json["vis_check"][0]["recommendation"]["revised_title"]
                    if vis_checked_2_json["vis_check"][0]["replace"] == "chart_2":
                        sec_chartid_for_pdf[3] = 22
                        img_to_llm_list[3] = vis_checked_2_json["vis_check"][0]["recommendation"]["revised_title"]
                    sec_query.append(vis_checked_2_json["vis_check"][0]["recommendation"]["query"])
                    sec_title.append(vis_checked_2_json["vis_check"][0]["recommendation"]["revised_title"])
                # if vis_checked_3_json["vis_check"][0]["replace"] != "none":
                #     if vis_checked_3_json["vis_check"][0]["replace"] == "chart_1":
                #         sec_chartid_for_pdf[4] = 33
                #         img_to_llm_list[4] = vis_checked_3_json["vis_check"][0]["recommendation"]["revised_title"]
                #     if vis_checked_3_json["vis_check"][0]["replace"] == "chart_2":
                #         sec_chartid_for_pdf[5] = 33
                #         img_to_llm_list[5] = vis_checked_3_json["vis_check"][0]["recommendation"]["revised_title"]
                #     sec_query.append(vis_checked_3_json["vis_check"][0]["recommendation"]["query"])
                #     sec_title.append(vis_checked_3_json["vis_check"][0]["recommendation"]["revised_title"])

                for id in range(len(sec_query)):
                    with open ("vis_nl_pair.json", "r") as f:
                            vis_nl_json = json.load(f)
                            query = sec_title[id]
                            # st.write(f'**Question for Chart {id}:**',f'**{query}**')
                            # st.write("Chart Title:",sec_title[id])
                            # RAG to extract vlspec of similar questions from the vectorstore
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
                                    # st.write("vlspec:",sample_vlspec)
                                    sec_rag_spec.append(sample_vlspec)
                                    break
                sec_spec_list = []
                for i in range(len(sec_query)):
                    sec_vlspec = agent6_sec_vis_generator(chosen_dataset,sec_query[i],sec_title[i],chosen_data_schema,sample_data, sec_rag_spec, openai_key)
                    st.write("Vega-Lite Specification:",sec_vlspec)
                    sec_cor_vlspec = parse_jsonish(sec_vlspec)  # -> dict/list or raises clear ValueError
                    sec_spec_list.append(sec_cor_vlspec["visualizations"][0])

                for spec_id in range(len(sec_spec_list)):
                    with open(f"DATA2Poster_json/vlspec2_{spec_id+1}.json", "w") as f:
                        json.dump(sec_spec_list[spec_id], f, indent=2)
                    sec_spec_list[spec_id]["height"] =600
                    sec_spec_list[spec_id]["width"] =800
                    try:
                        chart = alt.Chart.from_dict(sec_spec_list[spec_id])
                        chart.save(f"DATA2Poster_chart/image{spec_id+1}{spec_id+1}.png")
                        st.image(f"DATA2Poster_chart/image{spec_id+1}{spec_id+1}.png", caption="Chart "+str(spec_id+1))

                    except Exception as e:
                        error = str(e)
                        print(f"\n🔴 Error encountered: {error}\n")
                        corrected_vlspec = agent6_vis_corrector(spec, error, openai_key)
                        json_corrected_vlspec = parse_jsonish(corrected_vlspec)
                        json_corrected_vlspec["height"] =600
                        json_corrected_vlspec["width"] =800
                        final_chart = alt.Chart.from_dict(json_corrected_vlspec)
                        final_chart.save(f"DATA2Poster_chart/image{spec_id+1}{spec_id+1}.png")
                        st.image(f"DATA2Poster_chart/image{spec_id+1}{spec_id+1}.png", caption="Chart "+str(spec_id+1))

                # Refine the chart  using llm
                for spec_id in range(len(sec_spec_list)): 
                    binary_chart = open(f"DATA2Poster_chart/image{spec_id+1}{spec_id+1}.png", 'rb').read()
                    base64_utf8_chart = base64.b64encode(binary_chart).decode('utf-8')
                    img_url = f'data:image/png;base64,{base64_utf8_chart}'
                    with open(f"DATA2Poster_json/vlspec2_{spec_id+1}.json", "r") as f:
                        vlspec = json.load(f)
                    refined_vlspec = agent6_vis_refiner(vlspec, sec_query[spec_id], img_url, openai_key)

                    try:
                        chart = alt.Chart.from_dict(refined_vlspec)
                        chart.save(f"DATA2Poster_chart/image{spec_id+1}{spec_id+1}.png")
                        st.image(f"DATA2Poster_chart/image{spec_id+1}{spec_id+1}.png", caption="Chart "+str(spec_id+1))

                    except Exception as e:
                        error = str(e)
                        print(f"\n🔴 Error encountered: {error}\n")
                        corrected_vlspec = agent6_vis_corrector(refined_vlspec, error, openai_key)
                        json_corrected_vlspec = parse_jsonish(corrected_vlspec)
                        json_corrected_vlspec["height"] =600
                        json_corrected_vlspec["width"] =800
                        final_chart = alt.Chart.from_dict(json_corrected_vlspec)
                        final_chart.save(f"DATA2Poster_chart/image{spec_id+1}{spec_id+1}.png")
                        with open(f"DATA2Poster_json/vlspec2_{spec_id+1}.json", "w") as f:
                            json.dump(json_corrected_vlspec , f, indent=2)
                        st.image(f"DATA2Poster_chart/image{spec_id+1}{spec_id+1}.png", caption="Chart "+str(spec_id+1))

                st.write("chartid_for_pdf:",sec_chartid_for_pdf )
                for id in sec_chartid_for_pdf:
                    binary_chart     = open(f"DATA2Poster_chart/image{id}.png", 'rb').read()
                    base64_utf8_chart = base64.b64encode(binary_chart).decode('utf-8')
                    img_url = f'data:image/png;base64,{base64_utf8_chart}'              
                    sec_chart_for_pdf.append(img_url)
                
                insight_from_fact = insight_list_generator()
                st.write("Insight from all charts:", insight_from_fact)

                #  Second generation section header and section insight based on the chart titles
                sec_section_result = agent9_section_designer(img_to_llm_list, insight_from_fact, openai_key)
                sec_section_json = parse_jsonish(sec_section_result)
                st.write("Chart Check JSON:",sec_section_json)
                sec_section_insight_list = []
                sec_section_header_list = []
                for section in sec_section_json["sections"]:
                    sec_section_insight_list.append(section["insight"])
                    sec_section_header_list.append(section["heading"])
                # Score the poster quality
                poster_score = agent12_final_checker(conc_for_pdf, img_to_llm_list, sec_section_insight_list, openai_key)
                poster_score_json = parse_jsonish(poster_score)
                st.write("Poster Score JSON:",poster_score_json)

                # Log poster score JSON
                logs_dir = "logs"
                os.makedirs(logs_dir, exist_ok=True)
                filename = f"vischeck_poster_score_iter_{chosen_dataset}_{int(time.time())}.json"
                path = os.path.join(logs_dir, filename)
                with open(path, "w", encoding="utf-8") as fh:
                    log_payload = {
                        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                        "vis_checked_1_json": vis_checked_1_json,
                        "vis_checked_2_json": vis_checked_2_json,
                        "poster_score_json": poster_score_json,
                    }
                    json.dump(log_payload, fh, ensure_ascii=False, indent=2)


                if poster_score_json["final_score"] >= 90:
                    agent10_sec_pdf_creator(chosen_dataset, sec_section_insight_list, sec_chartid_for_pdf, sec_chart_for_pdf,conc_for_pdf,sec_section_header_list, openai_key)
                    st.success("Poster has been created successfully!🎉")
                    with open(f"pdf/{chosen_dataset}_summary_2.pdf", "rb") as f:  
                        st.download_button("Download Poster as PDF", f, f"""{chosen_dataset}_summary_2.pdf""")
                    break
                else:
                    continue
            if iter_flag ==3 and poster_score_json["final_score"] < 90:
                st.info("Fail to create poster. Please try again.")
        st.write("iter_flag:", iter_flag)
        # Reset session state
        st.session_state["bt_try"] = ""  
        st.session_state["s1c1_fact"] = []
        st.session_state["s2c1_fact"] = []
        chart_pattern = []
        insight_from_fact = []
        iter_flag = 1

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

