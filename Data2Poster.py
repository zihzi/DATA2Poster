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
from vis_generator import agent_improve_vis, agent_2_improve_code, agent_3_fix_code, agent_1_generate_code,agent_4_validate_spec
import vl_convert as vlc
from pathlib import Path
from poster_generator import create_pdf

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
    datasets["movies"] = pd.read_csv("data/movies.csv")
    datasets["StudentsPerformance"] = pd.read_csv("data/StudentsPerformance.csv") #NO vis_corpus
    datasets["Cars"] = pd.read_csv("data/Cars.csv") #NO vis_corpus
    datasets["Japan_life_expectancy"] = pd.read_csv("data/Japan_life_expectancy.csv")
    datasets["Sleep_health_and_lifestyle_dataset"] = pd.read_csv("data/Sleep_health_and_lifestyle_dataset.csv")
    datasets["personality_dataset"] = pd.read_csv("data/personality_dataset.csv")
    datasets["Korean_demographics"] = pd.read_csv("data/Korean_demographics.csv")
    datasets["regional_presidential_election"] = pd.read_csv("data/regional_presidential_election.csv")
    datasets["2024USA_presidential_election"] = pd.read_csv("data/2024USA_presidential_election.csv")
    datasets["Occupational gaps by gender"] = pd.read_csv("data/Occupational gaps by gender.csv")
    
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




# Calculate the importance score of data facts
score_importance(chosen_data_schema)

# Session state variables for workflow
def select_question():
    st.session_state["stage"] = "question_selected"
if "bt_try" not in st.session_state:
    st.session_state["bt_try"] = ""
if "stage" not in st.session_state:
    st.session_state["stage"] = "initial"
if "df" not in st.session_state:
    st.session_state["df"] = pd.DataFrame()
if "fact" not in st.session_state:
    st.session_state["fact"] = []
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = []
if "questions_for_pster" not in st.session_state:
    st.session_state["questions_for_pster"] = []
if "Q_from_gpt" not in st.session_state:
    st.session_state["Q_from_gpt"] = {}
if "selection" not in st.session_state:
    st.session_state["selection"] = ""


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

 
# Check if the user has tried and entered an OpenAI key
api_keys_entered = True  # Assume the user has entered an OpenAI key
if try_true or (st.session_state["bt_try"] == "T"):
    if not openai_key.startswith("sk-"):
                st.error("Please enter a valid OpenAI API key.")
                api_keys_entered = False

    st.session_state["bt_try"] = "T"
    
    
    if api_keys_entered:
        # use GPT as llm
        llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", temperature=1,api_key = openai_key)
        # use OpenAIEmbeddings as embedding model
        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key = openai_key)
        # To generate facts and questions by user-selected column
        if st.session_state["stage"] == "initial": 
            # For user to select a column
            st.write("Select a column:")
            selected_column = st.radio("Select one column:", head, index=None, label_visibility="collapsed")            
            skip_true = st.button("Skip")
        
            # column_des_prompt_template = """
            # You are a senior data analyst. You are analyzing a dataset named: {data_name} and it has columns: {data_columns}.
            # Here are summary of the dataset:\n\n{chosen_data_schema}\n\n
            # Describe this dataset to a user the structure of the table and the data it contains.
            # Explain these column names WITHOUT mention the statistical value given by the summary in natural language description that can help the user understand the dataset better.
            # Please do not add any extra prose to your response.
            # You should list reponse as below format:\n\n
            # "\"Data Description(IN BOLD):\n\n
            #   "Your description here."
            #   Columns Overview(IN BOLD):\n\n
            #   1.Column_Name: "Your description here."

            #   2.Column_Name: "Your description here."

            #   3.Column_Name: "Your description here."

            #   4.Column_Name: "Your description here."\"
            # """
            # prompt = PromptTemplate(
            # template=column_des_prompt_template,
            # input_variables=["data_names","data_columns","chosen_data_schema"],
            # )
            # column_des_chain = prompt | llm
            # column_des = column_des_chain.invoke(input ={"data_name": chosen_dataset, "data_columns":head,"chosen_data_schema":chosen_data_schema})

            # st.write(column_des.content)

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
                # user_select_column_template = load_prompt_from_file("prompt_templates/user_select_column.txt")       
                # prompt_column = PromptTemplate(
                #                             template=user_select_column_template,
                #                             input_variables=["user_selected_column", "chosen_dataset", "chosen_data_schema"],        
                #                 )
                # with open("json_schema/column_json_schema.json", "r") as f:
                #     column_json_schema = json.load(f)
                # chain_column = prompt_column | llm.with_structured_output(column_json_schema)
                # columns_from_gpt = chain_column.invoke(input = {"user_selected_column":user_selected_column, "chosen_dataset":chosen_dataset, "chosen_data_schema":chosen_data_schema})
                
                # # log the selected column and related columns
                # st.subheader(f"related column set:",divider = True)
                # for i in range(1,3):
                #     st.write("\"", f'**{columns_from_gpt[f"related_column_{i}"]["name"]}**',"\"")
                #     st.write(f'**Reason:**', f'**{columns_from_gpt[f"related_column_{i}"]["reason"]}**')
                        
                # # Extract dataFrame by user_selected_column and the related columns from gpt 
                # related_column = [columns_from_gpt["related_column_1"]["name"],columns_from_gpt["related_column_2"]["name"]]
                # df_for_cal = datasets[chosen_dataset][[user_selected_column] + related_column]   

                # # Produce columns to generate fact list
                # column_list_for_Q = []
                facts_list = []
                # columns_dic = {columns_from_gpt["selected_column"]["name"]: columns_from_gpt["selected_column"]["dtype"]}
                # for i in range(1,3):
                #     columns_dic[columns_from_gpt[f"related_column_{i}"]["name"]] = columns_from_gpt[f"related_column_{i}"]["dtype"]         
                # breakdown = [col for col, dtype in columns_dic.items() if dtype == "C" or dtype == "T"]   
                # measure = [col for col, dtype in columns_dic.items() if dtype == "N"]
                # column_list_for_Q.append(columns_dic)
                # st.write("Columns for generating facts:",column_list_for_Q)
                # TEST: USE ALL COLUMNS TO GENERATE FACTS#####################################################################
                desired_combinations = ['CNC', 'TNC', 'TNT', 'CNN', 'TNN']

                # Get and print results
                output = organize_by_dtype_combinations(chosen_data_schema, desired_combinations)

                # Categorization_facts ? to be fixed, not suitable for dataset type
                for item in output:
                    for key, value in item["columns_dtype_mapping"].items():
                        if value == "C":
                            ratio = round(datasets[chosen_dataset][key].value_counts()[0]/len(datasets[chosen_dataset]),3)*100
                            focus = datasets[chosen_dataset][key].value_counts().idxmax()
                            facts_list.append({"content":f"In {key}, {focus} accounts for the largest proportion of records({ratio}%).","score":1})
                
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
                    st.session_state["fact"] = []
                # Remove duplicates from facts_list
                contents = [fact["content"] for fact in facts_list]
                # facts_list = sorted(facts_list, key=itemgetter('score'), reverse=True)  
                seen = set()
                for item in facts_list:
                    if item["content"] != "No fact." and item["content"] not in seen:
                        seen.add(item["content"])
                        st.session_state["fact"].append(item["content"])
                # for item in st.session_state["fact"]:
                #     if user_selected_column not in item:
                #         st.session_state["fact"].remove(item)
                # Randomly select 100 items from st.session_state["fact"]
                if len(st.session_state["fact"]) > 100:
                    st.session_state["fact"] = random.sample(st.session_state["fact"], 200)
                st.write("Facts:",st.session_state["fact"])
                # TEST#########
                eda_prompt=PromptTemplate(template="""
                        You are an expert exploratory data analysis assistant. You will help synthesize a large set of individual data facts into a concise, insightful narrative closely related to **{column}** by following these steps:

                        **Task**:
                        1. **Enrich & Tag**  
                        For each fact in facts_list, generate or extract:
                        - dimensions: list of relevant dimensions (e.g. “Genre”, “Year”, “Region”)
                        - pattern_type: one of (trend, outlier, correlation, ranking, other)
                        - direction: one of (increasing, decreasing, high, low, none)
                        - magnitude: a short quantitative descriptor (e.g. “+25%”, “rank 5 of 10”, “p-value 0.01”)
                        - time_window: the years or range involved
                        - value_tag: one of (actionable, contextual, anomalous)

                        2. **Filter & Prioritize**  
                        - Discard facts with magnitude below noise threshold (e.g. <10%% change).  
                        - Keep only facts whose value_tag is "actionable" or matches priority_tags.  

                        3. **Cluster by Theme**  
                        - Group the remaining facts into 3-5 themes, based on shared dimensions or pattern_types.  
                        - Label each theme with a short name (e.g. "Genre Polarization", "Budget vs. Return").

                        4. **Select Representatives**  
                        - Within each theme, pick 1-2 facts that are both strong (largest magnitude) and novel.  

                        5. **Synthesize Patterns**  
                        - For each theme, write a 1-2 sentence insight summarizing the pattern (e.g. "Since 2010, Action's share of total gross has doubled…").  
                        - Recommend one visualization per theme (type of chart, key fields to encode).

                        6. **Output Format**  
                        Do not INCLUDE ```json```.Do not add other sentences after this json data.
                        Return a JSON object with:
                            {{
                            "themes": [
                                {{
                                "name": "<theme name>",
                                "fact_ids": [<indexes of chosen facts>],
                                "insight": “<1-2 sentence takeaway>”,
                                "visualization": {{
                                    "type": "line|bar|scatter|boxplot|…", 
                                    "x": "<field>", 
                                    "y": "<field>", 
                                    "color": "<optional field>"
                                }}
                                }},
                                ...
                            ]
                            }}
                        **** Inputs: {facts_list}*******"""
                        ,
                        input_variables=["column","facts_list"])
                eda_chain = eda_prompt | llm
                eda_result = eda_chain.invoke(input ={"column":user_selected_column,"facts_list":st.session_state["fact"]})
                eda_result_json = json.loads(eda_result.content)
                st.write("EDA Result:",eda_result_json)

                # eda_goal_prompt = PromptTemplate(template="""You are a senior data analyst. You are analyzing a dataset named: {data_name} and it has columns: {data_columns}.
                # Here are key facts extracted from this dataset:{fact}
                # **Your goal:**  
                # Generate a focused, structured EDA goal, using chain-of-thought reasoning.
                # **Instructions for your internal reasoning (do NOT include these steps in your response):**
                # 1. **Context & Exploration**
                # - Briefly restate what the dataset covers.
                # - List 1–2 hypotheses or questions.
                # 2. **Deep Dive**
                # For each hypothesis or question:
                # - **Analyze** how a column changes across values of one other column.
                # - **Highlight** any trends, correlations, anomalies, or group differences.
                # - **Limit** each point to **two** column names (the target plus one comparator).
                # 3. **Chain-of-Thought**
                # - Show your reasoning in 2–3 short bullet steps per insight.
                # - Use phrases like “Because…”, “When comparing…”, “We observe that…”.
                # **Output requirements (ONLY return this; no reasoning or extra text):**
                # - ONE sentence to describe your EDA GOAL.
                # - No explanations, no bullet steps, no metadata.
                # """,
                # input_variables=["data_name","data_columns","fact"])
                # eda_goal_chain = eda_goal_prompt | llm
                # eda_goal = eda_goal_chain.invoke(input ={"data_name":chosen_dataset, "data_columns":chosen_data_schema,"fact":st.session_state["fact"]})
                
                # st.write("EDA Goal:",eda_goal.content)
                # eda_task_prompt = PromptTemplate(template="""
                #                                             Your are a senior data analyst. You are analyzing a dataset: {data_columns}.
                #                                             Your need to come up with a short plan to help a user
                #                                             accomplish the goal:{goal}. Please recommend 3 exploratory tasks. For task type, you may consider the trend,
                #                                             correlation, category, distribution, etc., to explore the goal
                #                                             from different aspects. Do not use advanced modeling
                #                                             methods.Each task is a sentence. For data variables, the task may not correspond
                #                                             to the data, so you need to go to the dataset and find the
                #                                             corresponding data variables, and even if transformation is
                #                                             required, you need to write the original column name. There
                #                                             may be more than one column of data so it is an array. No
                #                                             need to say Here is the requested JSON format listing the
                #                                             tasks:. Do not INCLUDE ```json```.Do not add other sentences after this json data.
                #                                             The following is output JSON example:
                #                                             {{
                #                                                 "tasks": ["","",""],
                #                                                 "data variables":[],
                #                                                 "task type": ["","",""]
                #                                                 }}
                #                                  """,
                # input_variables=["data_columns","goal"])
                # eda_task_chain = eda_task_prompt | llm
                # eda_task = eda_task_chain.invoke(input ={"data_columns":chosen_data_schema,"goal":eda_goal.content})
                # task_json = json.loads(eda_task.content)
                # st.write("EDA Task:",task_json)
                # q_prompt = PromptTemplate(
                #                         template="""You are a senior data analyst. You are analyzing a dataset named: {data_name} and it has columns: {data_columns}.
                #                         Your goal is to generate a set of concise EDA questions for solving the following tasks:{eda_task}
                #                         Please generate questions that are specific to each tasks and these questions can be answered through visualizaing the data.
                #                         Do not INCLUDE ```json```.Do not add other sentences after this json data.
                #                         The following is output JSON example:
                #                         {{
                #                             "questions": [
                #                                 {{"task": "task_1", "question": "question_1"}},
                #                                 {{"task": "task_2", "question": "question_2"}},
                #                                 {{"task": "task_3", "question": "question_3"}}
                #                             ]
                #                         }}
                #                         """,
                # input_variables=["data_name","data_columns","eda_task"])
                # q_chain = q_prompt | llm
                # q_from_gpt = q_chain.invoke(input ={"data_name":chosen_dataset,"data_columns":chosen_data_schema,"eda_task":eda_task.content})
                # q_json = json.loads(q_from_gpt.content)
                # st.write("EDA Questions:",q_json)

                # # Create a vector store
                # def load_json(json_file):
                #     with open(json_file, "r", encoding="utf-8") as fh:
                #         return json.load(fh)


                # data_list = load_json('vis_corpus.json')
                # docs = [Document(page_content=json.dumps(item)) for item in data_list]
                # vectorstore = FAISS.from_documents(
                # docs,
                # OpenAIEmbeddings(model="text-embedding-3-small", api_key = openai_key),
                # )
                # st.session_state["vectorstore"] = vectorstore
                
                # Create intermediate output as knowledge
                # knowledge = self_augmented_knowledge(openai_key, chosen_dataset, list(head),user_selected_column, st.session_state["fact"])
                # st.write("KnowledgeBase:",knowledge)

                # Combine facts into interesting patterns
                # llm_pattern_template = load_prompt_from_file("prompt_templates/fact_idea_prompt.txt")
                # prompt_pattern = PromptTemplate(
                #                         template=llm_pattern_template,
                #                         input_variables=["facts","user_selected_column"],

                #             )
                # with open("json_schema/fact_idea_schema.json", "r") as f:
                #     pattern_json_schema = json.load(f)
                # chain_pattern = prompt_pattern | llm.with_structured_output(pattern_json_schema)
                # patterns_from_gpt = chain_pattern.invoke(input = {"facts":st.session_state["fact"],"user_selected_column":user_selected_column})
                # st.write("Interesting Patterns:",patterns_from_gpt)

                question_list=[]
                # supported_fact= []
                # Generate EDA questions based on interesting patterns
                for pattern in eda_result_json["themes"]:
                    # supported_fact.append(patterns_from_gpt[pattern]["supporting_facts"][0])
                    # supported_fact.append(patterns_from_gpt[pattern]["supporting_facts"][1])
                    # supported_fact.append(patterns_from_gpt[pattern]["supporting_facts"][2])
           
                    llm_Q_template = load_prompt_from_file("prompt_templates/llm_question.txt")
                    prompt_llm_Q = PromptTemplate(
                                            template=llm_Q_template,
                                            input_variables=["pattern_1","columns_set_1"],

                                )
                    with open ("json_schema/llm_question_schema.json", "r") as f:
                        llm_Q_schema = json.load(f)
                    llm_Q_chain = prompt_llm_Q | llm.with_structured_output(llm_Q_schema)
                    llm_Q_from_gpt = llm_Q_chain.invoke(input = {"pattern_1":pattern["insight"],"columns_set_1":list(head)})
                    st.write(llm_Q_from_gpt)
                    question_list.append(llm_Q_from_gpt["query_object"]["question"]+llm_Q_from_gpt["query_object"]["action"])
                Q_advice_prompt = PromptTemplate(
                template="""You are an assistant specialized in exploratory data analysis (EDA). Evaluate the following three queries:{query}
                            Identify if the queries are duplicates, similar, or distinct.
                            If any queries are duplicates or similar (leading to similar visualizations), suggest replacing the redundant query with a new distinct query that contributes to a coherent exploratory data analysis narrative.
                            Provide the new queries set with exactly **3** query if you find redundancy; otherwise, only return the original queries as they are.
                            """,
                input_variables=["query"],
                )

                Q_advice_chain = Q_advice_prompt | llm
                Q_advice = Q_advice_chain.invoke(input ={"query":question_list})
                st.write("Question Advice:",Q_advice.content)
                new_Q_prompt = PromptTemplate(
                template="""You are an assistant specialized in exploratory data analysis (EDA). 
                            Provide the new queries based on the following advice for refining EDA query:{advice}
                            Please approach this task methodically and reason step by step.
                            ONLY return the queries in JSON format without your reasoning or any additional text.
                            YOUR RESPONSE SHOULD NEVER INCLUDE "```json```".Please do not add any extra prose to your response.
                            """,
                input_variables=["advice"],
                )
                with open("json_schema/new_question_schema copy.json", "r") as f:
                    new_Q_schema = json.load(f)
                new_Q_chain = new_Q_prompt | llm.with_structured_output(new_Q_schema)
                new_Q = new_Q_chain.invoke(input ={"advice":Q_advice.content})
                st.write("New Question:",new_Q)    
                # # log the llm question
                # def log_response_to_json(knowledgebase, response):
                #     log_data = {"knowkedgebase": knowledgebase, "response": response}
                    
                #     with open("log/COT_Q_logs.json", "a") as f:
                #         f.write(json.dumps(log_data,indent=2))  # Append new log entry
                
                # # Save log
                # log_response_to_json(knowledge, llm_Q_from_gpt)

                st.write(question_list)
                # st.session_state["Q_from_gpt"] = {"questions":question_list}
                
                
            # To visualize the EDA questions
       
                st.session_state["bt_try"] = ""     
           
                # Call GPT to generate initial vlspec
                insight_list = []
                chart_des_list = []
                idx=1
                for query in new_Q["query_set"]:
                    
                #     st.write(f'**Question for Chart:**',f'**{query}**')
                #     print("\n🟢 Step 1: Generating Initial Code...\n")
                #     result = st.session_state["vectorstore"].similarity_search(
                #                         query,
                #                         k=1,
                #                     )
                # st.write("RAG Vega-Lite JSON:",result[0].page_content)
                    st.write(f'**Question for Chart {idx}:**',f'**{query}**')
                    vlspec = agent_1_generate_code(chosen_dataset,query,summary, openai_key)
                    json_code = json.loads(vlspec)
                    json_code["height"] = 400
                    json_code["width"] = 600
                    st.write("Vega-Lite JSON:",json_code)
                    png_data = vlc.vegalite_to_png(vl_spec=json_code,scale=3.0)
                    with open(f"DATA2Poster_img/image_{idx}.png", "wb") as f:
                        f.write(png_data)
                    st.image(f"DATA2Poster_img/image_{idx}.png", caption="Generated Chart")
                    # Evaluate the generated vlspec
                    binary_fc       = open(f"DATA2Poster_img/image_{idx}.png", 'rb').read()  # fc aka file_content
                    base64_utf8_str = base64.b64encode(binary_fc).decode('utf-8')
                    url = f'data:image/png;base64,{base64_utf8_str}'
                    feedback = agent_2_improve_code(query, url, openai_key)
                    st.write(feedback)
                    # Improve the vlspec based on feedback
                    improved_code = agent_improve_vis(vlspec, feedback,summary, openai_key)
                    improved_json = json.loads(improved_code)
                    improved_json["height"] = 400
                    improved_json["width"] = 600
                    st.write("Improved Vega-Lite JSON:",improved_json)
                    exec_count=0
                    try: 
                        improved_png_data = vlc.vegalite_to_png(vl_spec=improved_json,scale=3.0)
                        with open(f"DATA2Poster_img/image_{idx}.png", "wb") as f:
                            f.write(improved_png_data)
                        st.image(f"DATA2Poster_img/image_{idx}.png", caption="Improved Chart")
                    except Exception as e:
                        st.error("The code is failed to execute.")
            
                    
                    # Inspect logic error of the vlspec for final validation(at most 3 times)
                    pre_final_code = agent_4_validate_spec(improved_code,summary, openai_key)
                    pre_final_json = json.loads(pre_final_code)
                    pre_final_json["height"] = 400
                    pre_final_json["width"] = 600
                    st.write("Final Vega-Lite JSON:",pre_final_json)
                    try:
                        pre_final_png_data = vlc.vegalite_to_png(vl_spec= pre_final_json,scale=3.0)
                        with open(f"DATA2Poster_img/image_{idx}.png", "wb") as f:
                            f.write(pre_final_png_data)
                        st.image(f"DATA2Poster_img/image_{idx}.png", caption="Final Chart")
                    except Exception as e:
                        exec_count += 1
                        error = str(e)
                        error_code =  pre_final_code
                        print(f"\n🔴 Error encountered: {error}\n")
                        
                        # Load error-handling prompt
                        error_prompt_template = load_prompt_from_file("prompt_templates/error_prompt.txt")
                        error_prompt = PromptTemplate(
                            template=error_prompt_template,
                            input_variables=["error_code", "error"],
                        )    

                        error_chain = error_prompt | llm 
                        
                        # Invoke the chain to fix the code
                        corrected_code = error_chain.invoke(input={"error_code": error_code, "error": error})
                        final_code = corrected_code.content  # Update with the corrected code
                        
                        if exec_count == 3:
                            st.error("The code is failed to execute.")
                        else:
                        
                            final_json = json.loads(final_code)
                            final_json["height"] = 400
                            final_json["width"] = 600
                            final_png_data = vlc.vegalite_to_png(vl_spec=final_json,scale=3.0)
                            with open(f"DATA2Poster_img/image_{idx}.png", "wb") as f:
                                f.write(final_png_data)
                            st.image(f"DATA2Poster_img/image_{idx}.png", caption="Final Chart")
                           
                            
                    binary_chart     = open(f"DATA2Poster_img/image_{idx}.png", 'rb').read()  # fc aka file_content
                    base64_utf8_chart = base64.b64encode(binary_chart ).decode('utf-8')
                    img_url = f'data:image/png;base64,{base64_utf8_chart}' 
                    chart_prompt_template = load_prompt_from_file("prompt_templates/chart_prompt.txt")
                    chart_des_prompt = [
                        SystemMessage(content=chart_prompt_template),
                        HumanMessage(content=[
                            {
                                "type": "text", 
                                "text": f"This chart is ploted  based on this question:\n\n {query}.\n\n"
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
                    
                    chart_des =  llm.invoke(chart_des_prompt)

                    # supported_fact = st.session_state["Q_from_gpt"]["supported_fact"][idx-1]

                    
                    st.write(f'**Chart Description:**', f'**{chart_des.content}**')
                    # st.write(f'**Supported Data Fact:**', f'**{supported_fact}**')
                    # st.write(f'**Data Fact after RAG:**', f'**{retrieved_fact}**')
                    
                    # call GPT to generate insight description
                    insight_prompt_template = load_prompt_from_file("prompt_templates/insight_prompt.txt")
                    insight_prompt_input_template = load_prompt_from_file("prompt_templates/insight_prompt_input.txt")
                    insight_prompt_input = PromptTemplate(
                                template=insight_prompt_input_template,
                                input_variables=["query", "chart_des"],
                                                      
                    ) 
                    insight_prompt = ChatPromptTemplate.from_messages(
                            messages=[
                                SystemMessage(content = insight_prompt_template),
                                HumanMessagePromptTemplate.from_template(insight_prompt_input.template)
                            ]
                        )
                    
                    insight_chain = insight_prompt | llm
                    insight = insight_chain.invoke(input= {"query":query, "chart_des":chart_des})
                    insight_list.append(insight.content)
                    st.write(f'**Insight Description:**', f'**{insight.content}**')
                    chart_des_list.append(chart_des.content)
                    idx+=1
                
         
                # Reset session state
                st.session_state["df"] = pd.DataFrame()
                st.session_state["fact"] = []
                st.session_state["vectorstore"] = []
                st.session_state["questions_for_poster"] = []
                # st.session_state["Q_from_gpt"] = {}
                st.session_state["selection"] = ""
                # Create pdf and download
                pdf_title = f"{chosen_dataset} Poster"
                create_pdf(chosen_dataset, question_list, pdf_title, insight_list, openai_key)
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

