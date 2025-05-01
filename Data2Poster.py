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
from vis_generator import agent_1_generate_code, agent_2_improve_code, agent_3_fix_code
from pathlib import Path
from poster_generator import create_pdf

# Import langchain modules
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document



    
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
    datasets["Sleep_health_and_lifestyle"] = pd.read_csv("data/Sleep_health_and_lifestyle.csv")
    datasets["movies"] = pd.read_csv("data/movies.csv")
    datasets["StudentsPerformance"] = pd.read_csv("data/StudentsPerformance.csv") #NO vis_corpus
    datasets["Cars"] = pd.read_csv("data/Cars.csv") #NO vis_corpus
    datasets["Iris"] = pd.read_csv("data/Iris.csv")
    datasets["Sleep_health_and_lifestyle_dataset"] = pd.read_csv("data/Sleep_health_and_lifestyle_dataset.csv")
    datasets["Employee"] = pd.read_csv("data/Employee.csv")
    datasets["Housing"] = pd.read_csv("data/Housing.csv")
    datasets["adidas_sale"] = pd.read_csv("data/adidas_sale.csv")
    datasets["UberDataset"] = pd.read_csv("data/UberDataset.csv")
    
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
        llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", temperature=0,api_key = openai_key)
        # use OpenAIEmbeddings as embedding model
        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key = openai_key)
        # Initial stage to generate facts and questions by user-selected column
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
                facts_list = sorted(facts_list, key=itemgetter('score'), reverse=True)
                seen = set()
                for item in facts_list[:200]:
                    if item["content"] != "No fact." and item["content"] not in seen:
                        seen.add(item["content"])
                        st.session_state["fact"].append(item["content"])
                st.write("Facts:",st.session_state["fact"])

                # Create a vector store
                def load_json(json_file):
                    with open(json_file, "r", encoding="utf-8") as fh:
                        return json.load(fh)


                data_list = load_json('vis_corpus.json')
                docs = [Document(page_content=json.dumps(item)) for item in data_list]
                vectorstore = FAISS.from_documents(
                docs,
                OpenAIEmbeddings(model="text-embedding-3-small", api_key = openai_key),
                )
                st.session_state["vectorstore"] = vectorstore
                
                # Create intermediate output as knowledge
                knowledge = self_augmented_knowledge(openai_key, chosen_dataset, list(head),user_selected_column, st.session_state["fact"])
                st.write("KnowledgeBase:",knowledge)

                # Combine facts into interesting patterns
                llm_pattern_template = load_prompt_from_file("prompt_templates/fact_idea_prompt.txt")
                prompt_pattern = PromptTemplate(
                                        template=llm_pattern_template,
                                        input_variables=["knowledgebase","facts","user_selected_column"],

                            )
                with open("json_schema/fact_idea_schema.json", "r") as f:
                    pattern_json_schema = json.load(f)
                chain_pattern = prompt_pattern | llm.with_structured_output(pattern_json_schema)
                patterns_from_gpt = chain_pattern.invoke(input = {"knowledgebase":knowledge,"facts":st.session_state["fact"],"user_selected_column":user_selected_column})
                st.write("Interesting Patterns:",patterns_from_gpt)

                question_list=[]
                vis_q=[]
                supported_fact= []
                # Generate poster question based on interesting patterns
                for pattern in patterns_from_gpt:
                    supported_fact.append(patterns_from_gpt[pattern]["supporting_facts"][0])
                    supported_fact.append(patterns_from_gpt[pattern]["supporting_facts"][1])
                    supported_fact.append(patterns_from_gpt[pattern]["supporting_facts"][2])
                    support_fact_list = [patterns_from_gpt[pattern]["supporting_facts"][0],patterns_from_gpt[pattern]["supporting_facts"][1],patterns_from_gpt[pattern]["supporting_facts"][2]]  
                    vis_q.append(patterns_from_gpt[pattern]["extracted_pattern"])
                    llm_Q_template = load_prompt_from_file("prompt_templates/llm_question.txt")
                    prompt_llm_Q = PromptTemplate(
                                            template=llm_Q_template,
                                            input_variables=["pattern_1","pattern_1_fact_1","pattern_1_fact_2","pattern_1_fact_3","columns_set_1","knowledgebase"],

                                )
                    with open ("json_schema/llm_question_schema.json", "r") as f:
                        llm_Q_schema = json.load(f)
                    llm_Q_chain = prompt_llm_Q | llm.with_structured_output(llm_Q_schema)
                    llm_Q_from_gpt = llm_Q_chain.invoke(input = {"pattern_1":patterns_from_gpt[pattern]["extracted_pattern"],"pattern_1_fact_1":support_fact_list[0],"pattern_1_fact_2":support_fact_list[1],"pattern_1_fact_3":support_fact_list[2],"columns_set_1":list(head),"knowledgebase":knowledge})
                    st.write(llm_Q_from_gpt)
                    question_list.append(llm_Q_from_gpt["questions"]["question"])
                poster_Q_prompt = PromptTemplate(
                template="You are a senior data analyst. You are writing a poster title in question format to present the data analysis results. This poster want to convey the following insights {vis_q}.Think step by step to raise a question that can be covered by the insights. Do not use columnar formulas. ONLY respond with the question.",
                input_variables=["vis_q"],
                )
                poster_Q_chain = poster_Q_prompt | llm
                poster_Q = poster_Q_chain.invoke(input ={"vis_q": vis_q})
                    
                # # log the llm question
                # def log_response_to_json(knowledgebase, response):
                #     log_data = {"knowkedgebase": knowledgebase, "response": response}
                    
                #     with open("log/COT_Q_logs.json", "a") as f:
                #         f.write(json.dumps(log_data,indent=2))  # Append new log entry
                
                # # Save log
                # log_response_to_json(knowledge, llm_Q_from_gpt)

                st.write(question_list)
                questions_for_poster = [poster_Q.content]
                # supported_fact = [llm_Q_from_gpt["questions"]["fact_1"],llm_Q_from_gpt["questions"]["fact_2"],llm_Q_from_gpt["questions"]["fact_3"]]
                Q_for_vis = question_list
                st.session_state["Q_from_gpt"] = {"supported_fact":supported_fact,"questions":Q_for_vis}
                st.session_state["questions_for_poster"] = questions_for_poster
                st.write("Select a poster question:")
                selected_question=st.selectbox("Select a poster question:", questions_for_poster , on_change=select_question, label_visibility="collapsed",index=None,placeholder="Select one...", key="selection")      
        # Second stage to score related facts based on the selected question
        elif st.session_state["stage"] == "question_selected":
            st.session_state["bt_try"] = ""
            st.session_state["stage"] = "initial"            
            selected_poster_question = st.session_state["selection"]
            st.subheader(selected_poster_question)

            
            q_for_nl4DV = st.session_state["Q_from_gpt"]["questions"]
           
            code_template = load_prompt_from_file("prompt_templates/code_template.txt")
            # Call GPT to generate vlspec
            insight_list = []
            chart_des_list = []
            idx=1
            for query in q_for_nl4DV:
                st.write(f'**Question for Chart:**',f'**{query}**')
                print("\n🟢 Step 1: Generating Initial Code...\n")
                results = st.session_state["vectorstore"].similarity_search(
                                    query,
                                    k=1,
                                )
                chart_vega_json = json.loads(results[0].page_content)
                st.write("RAG Vega-Lite JSON:",chart_vega_json)
                chart_test_template = load_prompt_from_file("prompt_templates/chart_test_prompt.txt")
                prompt_chart_test = PromptTemplate(
                                    template=chart_test_template,
                                    input_variables=["query","label","chart","column","filter","aggregate","mark","encoding","sort","column_list"],

                        )
                chart_test_chain = prompt_chart_test | llm
                chart_code = chart_test_chain.invoke(input = {"query":query,"code_template":code_template,"label":chart_vega_json["label"],"chart":chart_vega_json["chart"],"column":chart_vega_json["column"],"filter":chart_vega_json["filter"],"aggregate":chart_vega_json["aggregate"],"mark":chart_vega_json["mark"],"encoding":chart_vega_json["encoding"],"sort":chart_vega_json["sort"],"column_list":datasets[chosen_dataset].columns.tolist()})
                print(chart_code.content)
                print("\n🟡 Step 2: Improving Code Quality...\n")
                improved_code = agent_2_improve_code(query, chart_code.content, openai_key)
                print(improved_code)
                exec_count=0
                try:
                    print("\n🔵 Step 3: Ensuring Code is Executable...\n")
                    final_code = agent_3_fix_code(improved_code, openai_key)
                    while exec_count < 3:  # Loop until the code executes successfully
                        try:
                            print("\n🟢 Trying to execute the code...\n")
                            code_executed = preprocess_json(final_code, idx)
                            print("\n✅ Code executed successfully!\n")
                            break  # Exit loop when execution is successful
                        
                        except Exception as e:
                            exec_count += 1
                            error = str(e)
                            error_code = final_code
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

                except Exception as final_exception:
                    print("\n❌ Failed to generate executable code after multiple attempts.")
                show_chart_flag = 1
                if exec_count == 3:
                    st.error("The code is failed to execute.")
                    show_chart_flag = 0
                
                # load the vega_lite_json for insight_prompt
                with open(f"DATA2Poster_json/vega_lite_json_{idx}.json", "r") as f:
                        chart = json.load(f)
                        # json for chart_description
                        # if "layer" in chart:
                        #     chart_type = chart["layer"][0]["mark"]["type"]
                        #     if "title" in chart["layer"][0]:
                        #         chart_title = chart["layer"][0]["title"]
                        #     else:
                        #         chart_title = chart["title"]
                        #     x_field = "Title: " + chart["layer"][0]["encoding"]["x"]["title"] + " Type: " + chart["layer"][0]["encoding"]["x"]["type"]
                        #     y_field = "Title: " + chart["layer"][0]["encoding"]["y"]["title"] + " Type: " + chart["layer"][0]["encoding"]["y"]["type"]
                        # elif "hconcat" in chart:
                        #     if "spec" in chart["hconcat"][0]:
                        #         if "layer" in chart["hconcat"][0]["spec"]:
                        #             chart_type = chart["hconcat"][0]["spec"]["layer"][0]["mark"]["type"]                             
                        #             chart_title = chart["spec"]["title"]
                        #         else:   
                        #             chart_type = chart["hconcat"][0]["spec"]["mark"]["type"]
                        #             chart_title = chart["hconcat"][0]["spec"]["title"]
                        #     else:
                        #         chart_type = chart["hconcat"][0]["mark"]["type"]
                        #         chart_title = chart["hconcat"][0]["title"]
                        #         x_field = "Title: " + chart["hconcat"][0]["encoding"]["x"]["title"] + " Type: " + chart["hconcat"][0]["encoding"]["x"]["type"]
                        #         y_field = "Title: " + chart["hconcat"][0]["encoding"]["y"]["title"] + " Type: " + chart["hconcat"][0]["encoding"]["y"]["type"]
                        # elif "spec" in chart:
                        #     if "layer" in chart["spec"]:
                        #         chart_type = chart["spec"]["layer"][0]["mark"]["type"]
                        #         chart_title = chart["spec"]["title"]
                        #     else:
                        #         chart_type = chart["spec"]["mark"]["type"]
                        #         chart_title = chart["spec"]["title"]
                        # else:
                        #     chart_type = chart["mark"]["type"]
                        #     if "title" in chart:
                        #         chart_title = chart["title"]
                        #     if "x" in chart["encoding"]:
                        #         if "axis" in chart["encoding"]["x"] and "title" in chart["encoding"]["x"]["axis"]:
                        #             x_field = "Title: " + chart["encoding"]["x"]["axis"]["title"] + " Type: " + chart["encoding"]["x"]["type"]
                        #         else:
                        #             x_field = "Title: " + chart["encoding"]["x"]["title"] + " Type: " + chart["encoding"]["x"]["type"]
                        #     if "y" in chart["encoding"]:
                        #         if "axis" in chart["encoding"]["y"] and "title" in chart["encoding"]["y"]["axis"]:
                        #             y_field = "Title: " + chart["encoding"]["y"]["axis"]["title"] + " Type: " + chart["encoding"]["y"]["type"]
                        #         else:
                        #             y_field = "Title: " + chart["encoding"]["y"]["title"] + " Type: " + chart["encoding"]["y"]["type"]
                        #     # It's Pie Chart
                        #     else:
                        #         x_field = "Title: " + chart["encoding"]["color"]["field"] + " Type: " + chart["encoding"]["color"]["type"]
                        #         y_field = "Title: " + chart["encoding"]["theta"]["field"] + " Type: " + chart["encoding"]["theta"]["type"]
                        for key in chart["datasets"]:
                            chart_data = chart["datasets"][key]
                            
                if show_chart_flag == 1:
                    st.vega_lite_chart(chart, theme = None)
                chart_prompt_template = load_prompt_from_file("prompt_templates/chart_prompt.txt")
                prompt_chart = PromptTemplate(
                                        template=chart_prompt_template,
                                        input_variables=["query","chart_type","chart_title","x_field","y_field","chart_data"],
                            )
                
                chain_pattern = prompt_chart | llm
                chart_des = chain_pattern.invoke(input = {"query":query,"chart_data":chart_data})

                supported_fact = st.session_state["Q_from_gpt"]["supported_fact"][idx-1]

                
                st.write(f'**Chart Description:**', f'**{chart_des.content}**')
                st.write(f'**Supported Data Fact:**', f'**{supported_fact}**')
                # st.write(f'**Data Fact after RAG:**', f'**{retrieved_fact}**')
                
                # call GPT to generate insight description
                insight_prompt_template = load_prompt_from_file("prompt_templates/insight_prompt.txt")
                insight_prompt_input_template = load_prompt_from_file("prompt_templates/insight_prompt_input.txt")
                insight_prompt_input = PromptTemplate(
                            template=insight_prompt_input_template,
                            input_variables=["query", "chart_des"],
                            # response_format=Insight,                         
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
            st.session_state["Q_from_gpt"] = {}
            st.session_state["selection"] = ""
            # Create pdf and download
            pdf_title = selected_poster_question
            create_pdf(chosen_dataset, q_for_nl4DV , pdf_title, chart_des_list,insight_list, openai_key)
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

