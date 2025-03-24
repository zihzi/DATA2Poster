import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import altair as alt
import random
import operator
from operator import itemgetter
from dataclasses import asdict
from nl4dv import NL4DV
from dataSchema_builder import get_column_properties
from insight_generation.dataFact_scoring import score_importance
# from dataFact_generator import fact_generator
from insight_generation.main import generate_facts
from typing import List, Dict, Union
from pathlib import Path
from itertools import product
# from viz_generator import VizGenerator
# from viz_executor import ChartExecutor
# from vispilot_for_poster import data_scope_generator, bar_data_generator, utility_calculator
# from PIL import Image
# import io
# import base64 
from dataFact_embedding import search_by_rag
from poster_generator import create_pdf
# import warnings
# warnings.filterwarnings("ignore")
st.set_option("deprecation.showPyplotGlobalUse", False)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from sklearn.cluster import KMeans
from langchain_community.vectorstores.utils import cosine_similarity, maximal_marginal_relevance


    
# Set page config
st.set_page_config(page_icon="analysis.png",layout="wide",page_title="DATA2POSTER")
st.title("📊 DATA2POSTER")

# List to hold datasets
if "datasets" not in st.session_state:
    datasets = {}
    # Preload datasets
    datasets["Movies"] = pd.read_csv("data/Movies.csv")
    datasets["Cars"] = pd.read_csv("data/Cars.csv")
    datasets["Customers & Products"] = pd.read_csv("data/Customers & Products.csv")
    datasets["Energy Production"] = pd.read_csv("data/Energy Production.csv")
    datasets["seattle_monthly_precipitation_2015"] = pd.read_csv("data/seattle_monthly_precipitation_2015.csv")
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
            2. Select dataset from the list below or upload your own dataset.
        """)
    # Set area for OpenAI key
    openai_key = st.text_input(label = "🔑 OpenAI Key:", help="Required for models.",type="password")
         
    # First we want to choose the dataset, but we will fill it with choices once we've loaded one
    dataset_container = st.empty()

    # Upload a dataset(!can only use the latest uploaded dataset for now)
    try:
        uploaded_file = st.file_uploader("📂 Load a CSV file:", type="csv")
        index_no = 0
        if uploaded_file:
            # Read in the data, add it to the list of available datasets.
            file_name = uploaded_file.name[:-4]
            datasets[file_name] = pd.read_csv(uploaded_file)
            # Save the uploaded dataset as a CSV file to the data folder
            datasets[file_name].to_csv(f"data/{file_name}.csv", index=False)
            # default the radio button to the newly added dataset
            index_no = len(datasets)-1
    except Exception as e:
        st.error("File failed to load. Please select a valid CSV file.")
        print("File failed to load.\n" + str(e))
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
if "question" not in st.session_state:
    st.session_state["question"] = []
if "Q_from_gpt" not in st.session_state:
    st.session_state["Q_from_gpt"] = {}
if "selection" not in st.session_state:
    st.session_state["selection"] = ""
if "set" not in st.session_state:
    st.session_state["set"] = {}

# page content 
st.write("Let's explore your data!✨")
try_true = st.button("Try it out!") 

# Use NL4DV to generate chosen_dataset's summary
nl4dv_instance = NL4DV(data_value = datasets[chosen_dataset])
summary = nl4dv_instance.get_metadata()

# preprocess the code generated by gpt-4o-mini
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
        code = code[:index] + f"data = pd.read_csv('data/{chosen_dataset}.csv')\n\n" + code[index:] + f"\n\nchart.save('data2poster_json/vega_lite_json_{count}.json')"
        exec(code)
    return code

# Format response from gpt-4o-mini as JSON
class Columns(BaseModel):
    selected_column: str = Field()
    related_columns: list = Field()
# class Questions(BaseModel):
#     Main_Question: str = Field()
#     Theme: str = Field()
#     Question: str = Field()
#     Purpose: str = Field()
#     Insight: str = Field()
class Questions(BaseModel):
    set1: dict = Field()
    set2: dict = Field()
    set3: dict = Field()
    Main_Question: str = Field() 
    Decomposed_Question: str = Field()
    Data_fact: str = Field()

class FactSubject(BaseModel):
    dataset: str
    breakdown: str
    measure: str
    measure2: Union[str, None] = None
    series: Union[str, None] = None
    chart_type: str
    disable_cache: bool = False
class Score(BaseModel):
    fact_index : int = Field()
    fact_text: str = Field()
    relevance: int = Field()  
    clarity: int = Field()
    contribution: int = Field()
    score: int = Field()
    reason: dict = Field()
class nl4DV_JSON(BaseModel):
    data_fact: str = Field()
    data_fact_raw: str = Field()
    dataset: str = Field()
    visList: list = Field()
    attributeMap: dict = Field()
    taskMap: dict = Field()      
parser_column_json = JsonOutputParser(pydantic_object = Columns)
parser_Q_json = JsonOutputParser(pydantic_object = Questions)
parser_score_json = JsonOutputParser(pydantic_object = Score)
parser_nl4dv_json = JsonOutputParser(pydantic_object = nl4DV_JSON) 
 
# Check if the user has tried and entered an OpenAI key
api_keys_entered = True  # Assume the user has entered an OpenAI key
if try_true or (st.session_state["bt_try"] == "T"):
    # use gpt-4o-mini as llm
    llm = ChatOpenAI(model_name="gpt-4o-mini-2024-07-18", api_key = openai_key)
    # use OpenAIEmbeddings as embedding model
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key = openai_key)
    st.session_state["bt_try"] = "T"
    
    if not openai_key.startswith("sk-"):
                st.error("Please enter a valid OpenAI API key.")
                api_keys_entered = False
    if api_keys_entered:
        # Initial stage to generate facts and questions by user-selected column
        if st.session_state["stage"] == "initial": 
            # For user to select a column
            st.write("Select a column:")
            selected_column = st.radio("Select one column:", head, index=None, label_visibility="collapsed")
            if selected_column: 
                user_selected_column = selected_column
            # Call gpt-4o-mini to select columns related to user-selected column in JSON format         
                prompt_column = PromptTemplate(
                                        template="""You are an expert in analyzing and understanding data relationships in structured datasets. 
                                                    Based on the user-selected column {user_selected_column}, dataset name {chosen_dataset}, and data summary {chosen_data_schema} provided, 
                                                    identify all columns related to the selected column based on logical, semantic, or contextual connections.
                                                    [Instructions] 
                                                    1) Understand the Schema: Review the schema carefully to understand the data structure and types of columns available.
                                                    2) Identify Insights: Think about the different types of insights we want to uncover, such as relationships between columns, trends or anomalies.
                                                    3) Identify breakdown and measure dimensions:
                                                    Insights are obtained when a measure is compared across a breakdown dimension.
                                                    The measure is a quantity of interest expressed in terms of variables of the table. It consists of
                                                    - A measure function (aggregation) - COUNT, MEAN, MIN, MAX
                                                    - A measure column - a numerical(N) column of the table
                                                    The breakdown dimension is a variable of the table across which we would like to compare values of measure to obtain meaningful insights. It is
                                                    - A breakdown column - a categorical(C) or temporal(T) column of the table
                                                    [/Instructions] 
                                                    You can ONLY SELECT TWO related columns,and at least one of whose dtype is "C" or "T". 
                                                    Return the result in JSON format, clearly listing the TWO related columns and the reason why you select the column.
                                                    The following is example JSON that you should response:
                                                    {{
                                                        "selected_column": {{"name":"user_selected_column","dtype":"N"}},
                                                        "related_columns": [
                                                            {{
                                                            "name": "related_column_1",
                                                            "dtype":"N",
                                                            "reason": "Description of the meaningful information can be revealed."
                                                            }},
                                                            {{
                                                            "name": "related_column_2",
                                                            "dtype":"C",
                                                            "reason": "Description of the meaningful information can be revealed."
                                                            }}
                                                        ]
                                                    }}\n{format_instructions}""",
                                        input_variables=["user_selected_column", "chosen_dataset", "chosen_data_schema"],
                                        partial_variables={"format_instructions": parser_column_json.get_format_instructions()},
                            )
                chain_column = prompt_column | llm | parser_column_json
                columns_from_gpt = chain_column.invoke(input = {"user_selected_column":user_selected_column, "chosen_dataset":chosen_dataset, "chosen_data_schema":chosen_data_schema})
                st.write("Columns related to the selected column:", columns_from_gpt)
                # Extract dataFrame by user_selected_column and the related columns from gpt 
                related_column = []
                for column in columns_from_gpt["related_columns"]:
                    related_column.append(column["name"])
                df_for_cal = datasets[chosen_dataset][[user_selected_column] + related_column]   
                # Produce columns to generate fact list
                columns_dic = {columns_from_gpt["selected_column"]["name"]: columns_from_gpt["selected_column"]["dtype"]}
                for column in columns_from_gpt["related_columns"]:
                    columns_dic[column["name"]] = column["dtype"]           
                breakdown = [col for col, dtype in columns_dic.items() if dtype == "C"or dtype == "T"]   
                measure = [col for col, dtype in columns_dic.items() if dtype == "N"]
                st.write(columns_dic)
                combination = list(product(breakdown, measure))
                facts_list = []
            
                for b, m in combination:                 
                    if columns_dic[b] == "C":
                        facts = generate_facts(
                                dataset=Path(f"data/{chosen_dataset}.csv"),
                                breakdown=b,
                                measure=m,
                                series=None,
                                breakdown_type="C",
                                measure_type="N",
                                with_vis=False,
                            )
                        for fact in facts:
                            facts_list.append({"content":fact["content"], "score":fact["score_C"]})

                    elif columns_dic[b] == "T":
                        facts = generate_facts(
                                dataset=Path(f"data/{chosen_dataset}.csv"),
                                breakdown=b,
                                measure=m,
                                series=None,
                                breakdown_type="T",
                                measure_type="N",
                                with_vis=False,
                            )
                        for fact in facts:
                            facts_list.append({"content":fact["content"], "score":fact["score_C"]})
                                
                if len(breakdown) == 2: 
                    if columns_dic[breakdown[0]] == "C":
                        facts = generate_facts(
                                        dataset=Path(f"data/{chosen_dataset}.csv"),
                                        breakdown=breakdown[0],
                                        measure=measure[0],
                                        series=breakdown[1],
                                        breakdown_type="C",
                                        measure_type="N",
                                        with_vis=False,
                                    )
                        for fact in facts:
                            facts_list.append({"content":fact["content"], "score":fact["score_C"]})
                                
                    elif columns_dic[breakdown[0]] == "T":
                        facts = generate_facts(
                                        dataset=Path(f"data/{chosen_dataset}.csv"),
                                        breakdown=breakdown[0],
                                        measure=measure[0],
                                        series=breakdown[1],
                                        breakdown_type="T",
                                        measure_type="N",
                                        with_vis=False,
                                    )
                        for fact in facts:
                            facts_list.append({"content":fact["content"], "score":fact["score_C"]})
                                
                if len(measure) == 2:
                    if columns_dic[breakdown[0]] == "C":
                        facts = generate_facts(
                            dataset=Path(f"data/{chosen_dataset}.csv"),
                            breakdown=breakdown[0],
                            measure=measure[0],
                            measure2=measure[1],
                            series=None,
                            breakdown_type="C",
                            measure_type="NxN",
                            with_vis=False,
                        )
                        for fact in facts:
                            facts_list.append({"content":fact["content"], "score":fact["score_C"]})

                    elif columns_dic[breakdown[0]] == "T": 
                        facts = generate_facts(
                                dataset=Path(f"data/{chosen_dataset}.csv"),
                                breakdown=breakdown[0],
                                measure=measure[0],
                                measure2=measure[1],
                                series=None,
                                breakdown_type="T",
                                measure_type="NxN",
                                with_vis=False,
                            )
                        for fact in facts:
                            facts_list.append({"content":fact["content"], "score":fact["score_C"]})  

                facts_list_sorted = sorted(facts_list, key=itemgetter('score'), reverse=True)
                st.write(facts_list_sorted[:10])

                for item in facts_list_sorted:
                    st.session_state["fact"].append(item["content"])
                
                # for item in facts_list_sorted: 
                #     emb = embeddings_model.embed_documents(item["content"])
                    
                #     st.write(emb)
                # Call gpt-4o-mini to generate poster question based on the fact list
                prompt_Q = PromptTemplate(
                                        template="""You are an expert in the domain of given dataset and a data science expert conducting exploratory data analysis.
                                                    Here are column name {columns_dic} you should only use and data summary {chosen_data_schema} and observed data patterns {facts} from the dataset for your reference.
                                                    First, use your extensive knowledge to generate a main question that can reveal the observed data patterns.
                                                    Then, think step by step to decompose the main question into THREE sub-questions.
                                                    Lastly, provide decomposed question and related data facts.
                                                    [Instructions]
                                                    Two type of main questions can be decomposed based on the following reasoning methods:
                                                    1)Type-I Main Questions: Provide a breakdown of the corresponding decomposed sub-questions that contribute to answering it.
                                                        Method 1) Comparison: Generate a question that compares facts within different data scopes based on the same measurement.
                                                        Method 2) Intersection: Generate a question that seeks data elements that satisfy multiple conditions specified by different data facts.
                                                        Method 3) Bridging: Generate a question that asks for a data fact satisfying a prior condition specified by another fact.
                                                    
                                                    2)Type-II Main Questions: Generate three additional questions by omitting specific components, and list all relevant data facts that could contribute to answering them.
                                                        Method 1) A question that does not mention an analysis task (no fact type).
                                                        Method 2) A question that does not mention the aspect to be estimated (no measure).
                                                        Method 3) A question that does not mention data divisions (no breakdown methods).
                                                    [/Instructions]
                                                    [Example] 
                                                        Given Data Facts:
                                                        Fact 1: The average monthly sales revenue of Product A in 2023 was $50,000.
                                                        Fact 2: The average monthly sales revenue of Product B in 2023 was $60,000.
                                                        Fact 3: In Q4 2023, the monthly sales revenue of Product A increased by 10%% compared to Q3 2023.
                                                        Fact 4: The total sales revenue from all products in 2023 was $2,000,000.
                                                        Fact 5: The number of units sold for Product A in December 2023 was 1,200.
                                                        Fact 6: The average price per unit of Product A in 2023 was $42.
                                                        
                                                        Generating Main Question:
                                                        1.(Type-I) Comparison Question: "Which product had higher average monthly sales revenue in 2023, and by how much?"
                                                          Decomposed Questions:
                                                          What was the average monthly sales revenue of Product A in 2023? (Fact 1)
                                                          What was the average monthly sales revenue of Product B in 2023? (Fact 2)
                                                          What is the difference in average monthly sales revenue between Product A and Product B?                
                                                        2.(Type-I) Intersection Question: "Did any product have both a high total sales revenue and a significant sales increase in Q4 2023?"
                                                          Decomposed Questions:
                                                          What was the sales revenue trend for Product A in Q4 2023? (Fact 3)
                                                          What was the total sales revenue from all products in 2023? (Fact 4)
                                                          Which products contributed most to the total sales revenue? (Fact 2)
                                                        3.(Type-I) Bridging Question: " In the month with the highest units sold for Product A, what was the total revenue generated from it?"
                                                          Decomposed Questions:
                                                          In which month were the most units of Product A sold? (Fact 5)
                                                          What was the average price per unit of Product A in 2023? (Fact 6)
                                                        4.(Type-II) No fact type: "What do we know about Product A's performance in 2023?"
                                                          Decomposed Questions:
                                                          What was the average monthly sales revenue of Product A in 2023? (Fact 1)
                                                          How did the sales revenue of Product A change during Q4 2023? (Fact 3)
                                                          How many units of Product A were sold in December 2023? (Fact 5)
                                                        5.(Type-II) No measure: "How did Product A and Product B perform in 2023?"
                                                          Decomposed Questions:
                                                          What was the average monthly sales revenue of Product A in 2023? (Fact 1)
                                                          What was the average monthly sales revenue of Product B in 2023? (Fact 2)
                                                          What was the total revenue generated by all products in 2023? (Fact 4)
                                                        6.(Type-II) No breakdown methods: "What was the overall sales revenue performance in 2023?"
                                                          Decomposed Questions:
                                                          What was the total revenue from all products in 2023? (Fact 4)
                                                          What was the total monthly sales revenue for Product A? (Fact 1)
                                                          What was the total monthly sales revenue for Product B? (Fact 2)
                                                    [/Example]
                                                        You should provide three sets of main question and corresponding decomposed questions, data facts in JSON format.
                                                        YOUR RESPONSE SHOULD NEVER INCLUDE "```json```".Please do not add any extra prose to your response.
                                                        The following is example JSON that you should response:
                                                        {{
                                                            "set1":
                                                            {{
                                                                "Main_Question": " ",
                                                                "1":{{
                                                                "Decomposed_Question": " ",
                                                                "Data_fact": " ",
                                                                }},
                                                                "2":{{
                                                                "Decomposed_Question": " ",
                                                                "Data_fact": " ",
                                                                }},
                                                                "3":{{
                                                                "Decomposed_Question": " ",
                                                                "Data_fact": " ",
                                                                }},
                                                            }},
                                                            "set2":
                                                            {{
                                                                "Main_Question": " ",
                                                                "1":{{
                                                                "Decomposed_Question": " ",
                                                                "Data_fact": " ",
                                                                }},
                                                                "2":{{
                                                                "Decomposed_Question": " ",
                                                                "Data_fact": " ",
                                                                }},
                                                                "3":{{
                                                                "Decomposed_Question": " ",
                                                                "Data_fact": " ",
                                                                }},
                                                            }},
                                                            "set3":
                                                            {{
                                                                "Main_Question": " ",
                                                                "1":{{
                                                                "Decomposed_Question": " ",
                                                                "Data_fact": " ",
                                                                }},
                                                                "2":{{
                                                                "Decomposed_Question": " ",
                                                                "Data_fact": " ",
                                                                }},
                                                                "3":{{
                                                                "Decomposed_Question": " ",
                                                                "Data_fact": " ",
                                                                }},
                                                            }},
                                                        }}\n\n{format_instructions}""",
                                                        
                                        input_variables=["chosen_data_schema","facts"],
                                        partial_variables={"format_instructions": parser_Q_json.get_format_instructions()},
                            )

                chain_Q = prompt_Q | llm | parser_Q_json
                Q_from_gpt = chain_Q.invoke(input = {"columns_dic":columns_dic, "chosen_data_schema":chosen_data_schema, "facts":facts_list_sorted[:10]})
                st.session_state["Q_from_gpt"] = Q_from_gpt
                st.write("Poster Question And Sub Questions:", Q_from_gpt)
                questions = [Q_from_gpt["set1"]["Main_Question"], Q_from_gpt["set2"]["Main_Question"], Q_from_gpt["set3"]["Main_Question"]]
                st.session_state["question"] = questions
                st.write("Select a poster question:")
                selected_question=st.selectbox("Select a poster question:", questions , on_change=select_question, label_visibility="collapsed",index=None,placeholder="Select one...", key="selection")      
        # Second stage to score related facts based on the selected question
        elif st.session_state["stage"] == "question_selected":            
            selected_question = st.session_state["selection"]
            st.subheader(selected_question)
            st.session_state["stage"] = "initial"

            index = 1
            for i, q in enumerate(st.session_state["question"]):
                if q == selected_question:
                    index += i
            select_set = st.session_state["Q_from_gpt"][f"set{index}"]
            st.session_state["set"] = select_set
            q_for_nl4DV = []
            for i in range(1,4):
                q_for_nl4DV.append(select_set[str(i)]["Decomposed_Question"])
            st.write(q_for_nl4DV)
 
            # Call gpt-4o-mini to generate vlspec
            insight_list = []
            for query in q_for_nl4DV:
                idx = q_for_nl4DV.index(query)+1
                prompt_nl4DV = """
                                    Consider the below JSON array of objects describing low-level analytic tasks (fundamental operations that users perform when interacting with data visualizations) as a list of their "Name", "Description" and "Pro Forma Abstract" (a concise summary outlining the main elements for a given natural language query), with "Examples" (example natural language queries about different datasets), "Attribute Data Types and Visual Encodings" (the data type of the column titles in the provided dataset along with the preferred visual encodings in the recommended visualization), "Attributes and Visual Encoding Descriptions", and "Recommended Visualizations":
                                    [
                                    {{
                                    "Name": "Correlation",
                                    "Description": "Given a set of data cases and two attributes, determine useful relationships between the values of those attributes.",
                                    "Pro Forma Abstract": "What is the correlation between attributes X and Y over a given set S of data cases?",
                                    "Examples": ["There is a negative moderate relationship between Cylinders and MPG when 'Cylinders' is 'greater' than mean, with pearson correlation coefficient of -0.57.", "There is a negative weak relationship between Cylinders and MPG when 'MPG' is 'greater' than mean, with pearson correlation coefficient of -0.15.", "There is a strong relationship between Cylinders and MPG when 'Year' is '1972', with pearson correlation coefficient of 0.86."],
                                    "taskMap Encoding": "correlation",
                                    "Attribute Data Types and Visual Encodings":
                                        {{
                                            "X axis": "Quantitative",
                                            "Y axis": "Quantitative",
                                            "Other Encoding": <choose from : ["Nominal", "Ordinal", "Quantitative", "Temporal"]>,
                                            "Sort": null,
                                            "Filter": null,
                                        }},
                                    "Attributes and Visual Encodings Description": "The "X axis" key indicates that the first attribute is used for the horizontal or x-axis of the visualization. Similarly the "Y axis" key is used for the vertical or y-axis of the visualization. The attributes in the "Other Encoding" key are optional and are to be used in encodings in color, shape, size, and opacity. The "Sort" and "Filter" keys should not be used in this task.",
                                    "Recommended Visualization":["Scatterplot"]
                                    }},
                                    {{
                                    "Name": "Derived Value",
                                    "Description": "Given a set of data cases, compute an aggregate numeric representation of those data cases.",
                                    "Pro Forma Abstract": "What is the value of aggregation function F over a given set S of data cases?",
                                    "Examples": ["The max of 'MPG' is 38.0 when 'Cylinders' is greater than mean.", "The min of 'MPG' is 18.0 when 'Cylinders' is less than mean.", "The sum of 'Displacement' is 39757.0 when 'Cylinders' is greater than mean.","The min 'Rotten_Tomatoes_Rating' of 'Content_Rating'=PG-13 is 5.00 times more than that of 'Content_Rating'=R when 'Genre' is 'Horror'."],
                                    "taskMap Encoding": "derived_value",
                                    "Attribute Data Types and Visual Encodings":
                                        {{
                                            "X axis": "Nominal" or "Ordinal",
                                            "Y axis": "Quantitative",
                                            "Other Encoding": <choose from : ["Nominal", "Ordinal", "Quantitative", "Temporal"]>,
                                            "Sort": null,
                                            "Filter": null,
                                        }},
                                    "Attributes and Visual Encodings Description": "The "X axis" indicates that the first attribute is used for the horizontal or x-axis of the visualization. Similarly the "Y axis" is used for the vertical or y-axis of the visualization. The attributes in the "Other Encoding" key are optional and are to be used in encodings in color, shape, size, and opacity. The "Sort" and "Filter" keys should not be used in this task.",
                                    "Recommended Visualization" : ["Bar Chart"]
                                    }},
                                    {{
                                    "Name": "Filter",
                                    "Description": "Given some concrete conditions on attribute values, find data cases satisfying those conditions.",
                                    "Pro Forma Abstract": "Which data cases satisfy conditions [A, B, C, ...]?",
                                    "Examples": ["The 'Creative_Type'=Contemporary Fiction accounts for 84.97%% of the sum 'Rotten_Tomatoes_Rating' when 'Genre' is 'Romantic Comedy'."],
                                    "taskMap Encoding": "filter",
                                    "Attribute Data Types and Visual Encodings":
                                        {{
                                            "X axis": null
                                            "Y axis": null
                                            "Other Encoding": null,
                                            "Sort": null,
                                            "Filter": True,
                                        }},
                                    "Attributes and Visual Encodings Description":  "The "X axis", "Y axis", "Other Encoding", and "Sort" keys should not be used for this task. However, the "Filter" key should be used. As the "Filter" key is set to True, this indicates that the ensuing visualization must satisfy the conditions requested by the input natural language query. As such, the ensuing visualization must only display data points that satisfy these conditions. The attribute requested for the filter task can be any datatype (Quantiative, Nominal, Ordinal, or Temporal). If there is a "Filter" task detected in the input natural language query, please add the "transform" property in Vega-Lite to the Vega-Lite specification. This action will apply the filter specified in the natural language query. (Link to Vega-Lite "transform" property: https://vega.github.io/vega-lite/docs/filter.html)",
                                    "Visualization Recommendation":["Line Chart", "Scatter Plot", "Strip Plot", "Histogram", "Bar Chart", "Heatmap"]
                                    }},
                                    {{
                                    "Name": "Trend",
                                    "Description": "Trend is the direction of the data over time, which may be increasing, decreasing, or flat",
                                    "Pro Forma Abstract": "What is the direction of values for attribute(a) in the span of Time(t)?",
                                    "Examples": ["The wavering trend of sum 'MPG' over 'Year' when 'Cylinders' is greater than mean.", "The increasing trend of max 'Cylinders' over 'Year' when 'Displacement' is greater than mean."],
                                    "taskMap Encoding": "trend",
                                    "Attribute Data Types and Visual Encodings":
                                        {{
                                            "X axis": "Temporal",
                                            "Y axis": "Quantitative",
                                            "Other Encoding": <choose from : ["Quantitative", "Nominal", "Ordinal"]>,
                                            "Sort": null,
                                            "Filter": null,
                                        }},
                                    "Attributes and Visual Encodings Description": "The "X axis" indicates that the first attribute is used for the horizontal or x-axis of the visualization. Similarly the "Y axis" is used for the vertical or y-axis of the visualization. Attributes in the curly brackets {} are optional and are to be used in the following encodings: color, shape, size, and opacity. The "Sort" and "Filter" keys are not to be used for this task.",
                                    "Visualization Recommendation":["Line Chart"]
                                    }},
                                    {{
                                    "Name": "Distribution",
                                    "Description": "Given a set of data cases and a quantitative attribute of interest, characterize the distribution of that attribute’s values over the set",
                                    "Pro Forma Abstract": "What is the distribution of values of attribute A in a set S of data cases?",
                                    "Examples": ["The distribution of the min 'Gas' over 'Year' when 'Population_M_' is less than mean is not normal."],
                                    "taskMap Encoding": "distribution",
                                    "Attribute Data Types and Visual Encodings":
                                        {{
                                            "X axis": "Quantitative" or "Nominal" or "Ordinal",
                                            "Y axis": "Quantitative",
                                            "Other Encoding": <choose from : ["Nominal", "Ordinal", "Quantitative", "Temporal"]>,
                                            "Sort": null,
                                            "Filter": null,
                                        }},
                                    "Attributes Description": "The "X axis" indicates that the first attribute is used for the horizontal or x-axis of the visualization. Similarly the "Y axis" is used for the vertical or y-axis of the visualization. Attributes in the curly brackets {} are optional and are to be used in encodings in color, shape, size, and opacity.",
                                    "Visualization Recommendation":["Histogram"]
                                    }},
                                    {{
                                    "Name": "Sort",
                                    "Description": "Given a set of data cases, rank them according to some ordinal metric.",
                                    "Pro Forma Abstract": "What is the sorted order of a set S of data cases according to their value of attribute A?",
                                    "Examples": ["In the min 'MPG' ranking of different 'Year', the top three are [1982, 1980, 1981] when 'Origin' is 'US'.", "There are 3 categories of 'Origin' which are ['Europe' 'Japan' 'US'], when 'Cylinders' is less than mean, among which 'US' is the most frequent category."],
                                    "taskMap Encoding": "sort",
                                    "Attribute Data Types and Visual Encodings": 
                                        {{
                                            "X axis": null
                                            "Y axis": null
                                            "Other Encoding": null,
                                            "Sort": True
                                            "Filter": null
                                        }},
                                    "Attributes and Visual Encodings Description": "The "X axis", "Y axis", "Other Encoding", and "Filter" keys should not be used for this task. However, the "Sort" key should be used. As the "Sort" key is set to True, this indicates that the ensuing visualization must satisfy the order/ranking requested by the input natural language query. The attribute requested for the sort task must be of the Quantitative datatype.",
                                    "Visualization Recommendation":["Bar Chart"]
                                    }},
                                    {{
                                    "Name": "Find Extremum",
                                    "Description": "Find data cases possessing an extreme value of an attribute over its range within the data set.",
                                    "Pro Forma Abstract": "What are the top/bottom N data cases with respect to attribute A?",
                                    "Examples": ["The mean 'MPG' of '1982' is an outlier when compare with that of other 'Year' when 'Cylinders' is greater than mean.", "The maximum value of the min 'MPG' is 21.0 from 'Origin'=US when 'Cylinders' is less than mean."],
                                    "taskMap Encoding": "find_extremum",
                                    "Attribute Data Types and Visual Encodings":
                                        {{
                                            "X axis": null
                                            "Y axis": null
                                            "Other Encoding": null,
                                            "Sort": True
                                            "Filter": null
                                        }},
                                    "Attributes and Visual Encodings Description": "The "X axis", "Y axis", "Other Encoding", and "Filter" keys should not be used for this task. However, the "Sort" key should be used. As the "Sort" key is set to True, this indicates that the ensuing visualization must satisfy the order/ranking requested by the input natural language query. The attribute requested for the sort task must be of the Quantitative datatype.",
                                    "Visualization Recommendation":["Bar Chart"]
                                    }}
                                    ]\n\n

                                    Using the above definitions, classify the below natural language queries into the respective analytic tasks they map to. There can be one or more analytic tasks detected in the input natural language query. Return the visualization type in the form of a Vega-Lite specification. PLEASE ensure that the schema used in your Vega-Lite specification is https://vega.github.io/schema/vega-lite/v4.json.
                                    Here's a subset of the original dataset with actual columns and rows for reference.

                                    <INSERT DATASET HERE>

                                    Detect any attributes, tasks, and visualizations in the dataset that the provided data fact references, and place the detected dataset columns in the attributeMap, taskMap and visList property of the JSON below. Each Query can have more than one task and visualization type they can map to. Each property in the "attributeMap" JSON should be populated with the extracted dataset column (e.g. "Worldwide Gross"). There can be multiple attributes, tasks, and visualizations that are detected, but make sure that each attribute, task, and visualization in the attributeMap, taskMap, and visList is unique. Put each attribute into the attributeMap, task into taskMap, and Visualization into visList JSON. There can also be multiple possible visualization specifications as well. For each possible visualization specification, you can include a dictionary that has the required contents to the "visList" list.
                                    Furthermore, note that the input natural language (NL) query can also use ambiguous language with partial references to data attributes. In these cases, the attributeMap also includes an "isAmbiguous" field. The field can either be True or False. Set the field to True if the queryPhrase could refer to other attributes in the dataset. Otherwise set the field to False. If there are ambiguous attributes detected, generate Vega-Lite specifications for each ambiguous attribute that encode the ambiguous attribute in the visualization. Furthermore, if there are no tasks detected in the NL query, infer the task that is best suited with the detected attributes' datatypes. Generate a visualization specification using this inferred task and detected attributes.
                                    Here is the JSON object that the response should be returned :

                                    {{
                                    "query": <Add data fact that is being parsed here>,
                                    "qyery_raw": <Add data fact that is being parsed here>,
                                    "dataset": <Add dataset URL here>,
                                    "visList": [
                                    {{
                                    "attributes": [List of dataset attributes detected],
                                    "queryPhrase": [<Keywords found in query that were used to detect the taskMap and the recommended visualization>],
                                    "visType": <Put the visualization type that was explicitly specified in the query. Put the string "None" if vis type was not specified in the query.>,
                                    "visQueryPhrase": <Keywords found in query that were used to detect the visType. Put the string "None" if vis type was not specified in the query.>,
                                    "tasks": [Add the list of tasks detected here. Utilize the value from the "taskMap Encoding" key in the analytic task JSON array to populate this list],
                                    "inferenceType": <Can be one of two values: "explicit" or "implicit". Set the value to "explicit" if the visualization's "queryPhrase" explicitly references a visualization type. Otherwise set the value to "implicit".>
                                    "vlSpec": <Add the Vega-Lite specification of the visualization recommended here.><ALWAYS include '$schema': 'https://vega.github.io/schema/vega-lite/v4.json','data': {{'url': '<Add dataset URL here>'}}>
                                    }}
                                    ],
                                    "attributeMap": {{
                                    <Dataset column that was detected (should have same value as "name")>: {{
                                    "name": <Dataset column that was detected>,
                                    "queryPhrase": [<Keywords found in query that were used to detect the dataset attribute>]
                                    "encode": <Boolean value depending on if the attribute appears on either of the axes or color in the Vega-Lite specification. The boolean value should be output as true or false in all lowercase letters.>
                                    }},
                                    "metric": <[Can be one of two values: "attribute_exact_match" or "attribute_similarity_match".  Set the value to "attribute_exact_match" if the attribute was found directly in the query. Set the value to "attribute_similarity_match" if the query uses a synonym for the attribute.]>,
                                    "inferenceType": <Can be one of two values: "explicit" or "implicit". Set the value to "explicit" if the attribute’s "queryPhrase" references an attribute name. Set the value to "implicit" if the queryPhrase directly references values found in the attribute’s values.>
                                    "isAmbiguous": <Can be either True or False. Set the field to True if the queryPhrase could refer other attributes in the dataset. Otherwise set the field to False.>
                                    "ambiguity": [<Populate this list with all the different attributes in the dataset that the queryPhrase can refer to  if isAmbiguous is set to True. Otherwise keep this list empty.]
                                    }},
                                    "taskMap": {{
                                    <Task that was detected. Utilize the value from the "taskMap Encoding" key in the analytic task JSON array to populate this key>: [
                                    {{
                                    "task": <Task that was detected. You must utilize the value from the "taskMap Encoding" key in the analytic task JSON array to populate this key>,
                                    "queryPhrase": [<Keywords found in query that were used to detect the task>],
                                    "values": [<If the "Filter" task was detected, put the filter value in here>],
                                    "attributes": [<Populate with the attributes that the task is mapped to]>],
                                    "operator": "<Can be one of "IN", "GT", "EQ", "AVG", "SUM", "MAX", or "MIN". "GT" is greater than. "EQ" is equals. "GT" and "EQ" are used for quantitative filters. "IN" is used for nominal filters. "AVG" and "SUM" are used for derived value tasks. "SUM" is for summation and "AVG" is for average. "MAX" and "MIN" are to be used for the sort and find extremum tasks. "MAX" indicates that the highest value must be displayed first. "MIN" indicates that the lowest value must be displayed first. Keep the string empty otherwise.>"
                                    "inferenceType": <Can be one of two values: "explicit" or "implicit". Set the value to "explicit" if the "queryPhrase" directly requests for a task to be applied. Set the value to "implicit" if the task is derived implicitly from the "queryPhrase".>
                                    }}
                                    ]
                                    }}
                                    
                                    }}
                                    YOUR RESPONSE SHOULD NEVER INCLUDE "```json```".Please do not add any extra prose to your response. I only want to see the JSON output.
                                """
                code_template = \
                    f"""
                        import altair as alt
                        import pandas as pd
                        <imports>
                        def plot(data: pd.DataFrame):
                    

                            <stub> # only modify this section
                        
                            return chart
                        chart = plot(data) Always include this line. No additional code beyond this line.
                    """
                prompt_input = PromptTemplate(
                        template="""
                        Here's a subset of the original dataset with actual columns and rows for reference.\n\n
                        {sample_data}\n\n
                        Here is metadata for the dataset.\n\n
                        {summary}\n\n
                        Here is the query.\n\n
                        {query}\n\n
                        """,
                        input_variables=["sample_data", "summary", "query"]
            )
                nl4DV_prompt = ChatPromptTemplate.from_messages(
                        messages=[
                            HumanMessage(content = prompt_nl4DV),
                            HumanMessagePromptTemplate.from_template(prompt_input.template)
                        ]
                    )
                nl4DV_chain = nl4DV_prompt | llm | parser_nl4dv_json
                nl4DV_json = nl4DV_chain.invoke(input= {"sample_data":sample_data, "summary": summary, "query":query} )
                # Call gpt-4o-mini to generate vis code
                code1_prompt = PromptTemplate(
                            template="""
                                        You are a helpful assistant highly skilled in writing PERFECT code for visualizations in python. 
                                        Here is the dataset {dataset} ,and some information for your reference to write visualization code {nl4DV_json}.
                                        ALWAYS make sure viualize each attributes and the task in the information 
                                        Given the code template, you complete the template to generate a visualization. 
                                        The visualization CODE MUST BE EXECUTABLE and MUST NOT CONTAIN ANY SYNTAX OR LOGIC ERRORS (e.g., it must consider the data types and use them correctly). 
                                        You MUST first generate a brief plan for how you would solve the task e.g. what transformations you would apply if you need to construct a new column, 
                                        what attributes you would use for what fields, what aesthetics you would use, etc.
                                        Based on the {vlSpec}, the GENERATED CODE SOLUTION SHOULD BE CREATED BY MODIFYING THE SPECIFIED PARTS OF THE TEMPLATE BELOW.\n\n {code_template} 
                                        Please do not add any extra prose to your response. I only want to see the EXECUTABLE CODE output.\n\n.
                                        The FINAL COMPLETED CODE BASED ON THE TEMPLATE above is ...""",
                            input_variables=["dataset", "nl4DV_json", "vl_spec", "code_template"],
                )    
                code1_chain = code1_prompt | llm 
                code1 = code1_chain.invoke(input= {"dataset":datasets[chosen_dataset], "nl4DV_json":nl4DV_json, "vlSpec":nl4DV_json["visList"][0]["vlSpec"] ,"code_template":code_template} )
                # repair the processed_code
                code2_prompt = PromptTemplate(
                            template="""
                                        You are a helpful assistant highly skilled in revising visualization code to improve the quality of the code and visualization based on a given query.
                                        Your task is to make the visualization code explain the query for the best.
                                        Here is the query {query} and the visualization code {processed_code} for your reference.
                                        Consider the following information {nl4DV_json}, focus on "attributeMap" and "taskMap", and evaluate how well is the code that applies any kind of data transformation (filtering, aggregation, grouping, null value handling etc).
                                        DATA IS ALREADY IN THE GIVEN CODE, SO NEVER USE "<Add dataset URL here>" or LOAD ANY DATA ELSE.
                                        ONLY revise the code between the lines of "def plot(data):" and "return chart".
                                        You MUST return a full EXECUABLE program. DO NOT include any preamble text. Do not include explanations or prose.""",
                            input_variables=["data_fact", "processed_code", "nl4DV_json"],
                )    
                code2_chain = code2_prompt | llm 
                code2 = code2_chain.invoke(input= {"query":query, "processed_code":code1.content, "nl4DV_json":nl4DV_json} )
                # make the code EXECUABLE.
                code3_prompt = PromptTemplate(
                            template="""
                                        You are a helpful assistant highly skilled in revising visualization code to make the code EXECUABLE.
                                        Your task is to check the code and find out problems that may cause it to fail to execute, such as logic errors, undefined parameters, etc.
                                        Here is the visualization code {processed_code} you have to revise.ONLY revise the code between the lines of "def plot(data):" and "return chart".
                                        DATA IS ALREADY IN THE GIVEN CODE, SO NEVER USE "<Add dataset URL here>" or LOAD ANY DATA ELSE.
                                        NEVER USE THE ATTRIBUTE "configure_background" in the code.
                                        You MUST return a full EXECUABLE program. DO NOT include any preamble text. Do not include explanations or prose.""",
                            input_variables=["processed_code"],
                )    
                code3_chain = code3_prompt | llm 
                code3 = code3_chain.invoke(input= {"processed_code":code2.content} )
                final_code = preprocess_json(code3.content, idx)
                #  RAG
                result = search_by_rag(st.session_state["fact"], query, openai_key)

                # load the vega_lite_json for insight_prompt
                with open(f"data2poster_json/vega_lite_json_{idx}.json", "r") as f:
                        chart = json.load(f)
                
                insight_prompt = PromptTemplate(
                            template="""
                                        You are an expert data analyst. Below is a observed data facts, followed by a question. 
                                        Your task is use the data fact to generate a concise and clear insight description for charts. 
                                        Provide an insight description that:
                                        - Highlights the most relevant data points.
                                        - Avoid speculative claims without statistical support.
                                        - Your description MUST BE TWO SENTENCES.

                                        Data Facts: {fact}

                                        Charts: {chart}
                                        
                                        Question: {query}
                                
                                        """,
                            input_variables=["fact", "chart", "query"],
                )    
                st.write(result)
                insight_chain = insight_prompt | llm 
                insight = insight_chain.invoke(input= {"fact":result, "chart":chart, "query":query})
                insight_list.append(insight.content)
                col1, col2 = st.columns(2)
                with col1:
                    with open(f"data2poster_json/vega_lite_json_{idx}.json", "r") as f:
                        vega_lite_json = json.load(f)
                        st.vega_lite_chart(vega_lite_json, theme = None)
                        # image for pdf
                        img = alt.Chart.from_dict(vega_lite_json)
                        img.save(f"image_{idx}.png")
                with col2:
                    st.write(insight.content)
            st.session_state["fact"] = []
            # Create pdf and download
            pdf_title = selected_question
            introduction_materials = st.session_state["set"]
            # poster_code=create_pdf(chosen_dataset, introduction_materials, pdf_title, insight_list, summary, openai_key)
            # exec(poster_code)
            create_pdf(chosen_dataset, introduction_materials, pdf_title, insight_list, summary, openai_key)
            st.success(f"""Poster has been created successfully!🎉""")
            with open(f"""{chosen_dataset}_summary.pdf""", "rb") as f:
                st.download_button("Download Poster as PDF", f, f"""{chosen_dataset}_summary.pdf""")

##################################################################################################################################                
            
            # col1, col2, col3 = st.columns(3)
            # with open(f"data2poster_json/vega_lite_json_1.json", "r") as f:
            #     vega_lite_json = json.load(f)
            #     with col1:
            #         st.vega_lite_chart(vega_lite_json, theme = None)
            #         # image for pdf
            #         img = alt.Chart.from_dict(vega_lite_json)
            #         img.save(f"image_1.png")
            # with open(f"data2poster_json/vega_lite_json_2.json", "r") as f:
            #     vega_lite_json = json.load(f)
            #     with col2:
            #         st.vega_lite_chart(vega_lite_json, theme = None)
            #         # image for pdf
            #         img = alt.Chart.from_dict(vega_lite_json)
            #         img.save(f"image_2.png")
            # with open(f"data2poster_json/vega_lite_json_3.json", "r") as f:
            #     vega_lite_json = json.load(f)
            #     with col3:
            #         st.vega_lite_chart(vega_lite_json, theme = None)
            #         # image for pdf
            #         img = alt.Chart.from_dict(vega_lite_json)
            #         img.save(f"image_3.png")
                
                    
        #         clean_fact = [
        #                 s for s in fact
        #                 if s is not None
        #                 and "No significant difference." not in s
        #                 and "No clear trend." not in s
        #                 and "is empty." not in s
        #                 and "The normal distribution" not in s
        #                 and "There is no outlier" not in s
        #                 ] 
####################################################################################################################################            
        #         # Call gpt-4o-mini to generation poster question based on the fact list
        #         embeddings = np.array(embeddings_model.embed_documents(facts))
        #         kmeans = KMeans(n_clusters=10, init="k-means++", random_state=42)
        #         kmeans.fit(embeddings)
        #         labels = kmeans.labels_
        #         df = pd.DataFrame(facts, columns=["fact"])
        #         df["fact_embedding"] = embeddings.tolist()
        #         df["cluster"] = labels
        #         st.dataframe(df)
        #         st.session_state["df"] = df
        #         # Calculate cosine similarity between facts in each cluster and extract top-10 facts
        #         # grouped = df.groupby("cluster")
        #         # for i in range(10):
        #         #     arr = grouped.get_group(i).drop("cluster", axis=1).to_numpy()
        #         #     similarity_matrix = cosine_similarity(arr)
        #         #     st.write(similarity_matrix)
        #         #     pairwise_similarities = [
        #         #     (j, k, similarity_matrix[j, k])
        #         #     for j in range(len(arr))
        #         #     for k in range(j + 1, len(arr))  # Iterate over the upper triangular part of the matrix (excluding diagonal) to avoid duplicates and self-comparisons.
        #         #     ]
        #         #     top_10_fact = sorted(pairwise_similarities, key=lambda x: x[2], reverse=True)[:10]
        #         #     st.write(top_10_fact)
        #         questions=[]
        #         for i in range(10):
        #             facts = "\n".join(
        #                 df[df["cluster"] == i]["fact"]
        #                 .str.replace(".", ".\n")  
        #                 .sample(2, random_state=42) # Random select 10 facts from each cluster
        #                 .values
        #             )
        #             prompt_question = PromptTemplate(
        #                                 template="""You are an expert in data science. You are tasked with generating a question based on the following data facts from a multi-dimentional data.
        #                                             ONLY return a high-level and concise question that encapsulates the shared theme or key insight represented by these data facts without any preamble or explanation.\n\n
        #                                             Data facts:\n\n{facts}\n\n.
        #                                         """,
        #                                 input_variables=["facts"]
        #                     )
        #             chain_question = prompt_question | llm 
        #             question_from_gpt = chain_question.invoke(input = {"facts":facts})
        #             questions.append(question_from_gpt.content)
        #         st.session_state["question"] = questions
        #         st.write("Select a poster question:")
        #         selected_question=st.selectbox("Select a poster question:", random.sample(population=questions, k=5) , on_change=select_question, label_visibility="collapsed",index=None,placeholder="Select one...", key="selection")      
        # # Second stage to score related facts based on the selected question
        # elif st.session_state["stage"] == "question_selected":
        #     st.subheader(st.session_state["selection"])
        #     selected_question = st.session_state["selection"]
        #     df = st.session_state["df"]
        #     i=0
        #     # Use MMR to select top-5 facts that most related to the question
        #     question_embedding = embeddings_model.embed_query(selected_question)
        #     for q in st.session_state["question"]:
        #         if q == selected_question:
        #             i = st.session_state["question"].index(q)
        #             break
        #     documents = df[df["cluster"] == i]["fact"].values.tolist()
        #     documents_embeddings = embeddings_model.embed_documents(documents)
        #     selected_indices = maximal_marginal_relevance(np.array(question_embedding), documents_embeddings, lambda_mult=0.1, k=5)
        #     facts_to_score = {}
        #     for idx in selected_indices:
        #         facts_to_score[f"fact {idx+1}"] = documents[idx]
        #     st.write(facts_to_score)
        #     # Call gpt-4o-mini to score the selected facts
        #     prompt_score = PromptTemplate(
        #                         template="""You are an expert in data analysis and natural language interpretation. 
        #                                     Your task is to score a set of data facts based on how well they contribute to interpreting the given question from a data exploration perspective. 
        #                                     The goal is to assess each fact's relevance, clarity, and ability to address the question.
        #                                     Think of rationale for how to assign score first and then score for each fact.\n\n 
        #                                     The following is an example of question and data facts:
        #                                     [EXAMPLE]
        #                                         Question: "How do variations in temperature and wind conditions influence the occurrence of different weather types, such as rain and sun?"
        #                                         Data Facts:
        #                                             {{
        #                                                 "fact 1": "The maximum value of the sum 'wind' is 321.0 from 'weather'=rain when 'temp_min' is less than mean."
        #                                                 "fact 2": "The minimum value of the mean 'temp_min' is 6.486046511627906 from 'weather'=fog when 'wind' is less than mean."
        #                                                 "fact 3": "The maximum value of the min 'temp_min' is 10.0 from 'weather'=drizzle when 'wind' is less than mean."
        #                                                 "fact 4": "The minimum value of the mean 'wind' is 2.5142857142857147 from 'weather'=drizzle when 'temp_min' is greater than mean."
        #                                                 "fact 5": "The minimum value of the max 'wind' is 6.5 from 'weather'=sun when 'temp_max' is less than mean."
        #                                             }}
        #                                     [/EXAMPLE]\n\n 
        #                                     For each data fact, score its relevance to the question based on the following guide:
        #                                         Scoring Criteria:
        #                                             1. Relevance (0-5): Does the fact directly relate to the question’s theme (variations in temperature and wind conditions influencing weather types)?
        #                                             2. Clarity (0-5): Is the fact clearly expressed and easy to interpret?
        #                                             3. Contribution (0-5): Does the fact provide unique or useful insight for interpreting the question?
        #                                         Instructions for Scoring:
        #                                             1. Evaluate the semantic alignment of each fact with the question using embeddings or semantic similarity measures.
        #                                             2. Assign scores for relevance, clarity, and contribution, ensuring each score is justified with a brief explanation.
        #                                     The following is example JSON that you should response:
        #                                     {{
        #                                         "fact_index": 1,
        #                                         "fact_text": "The maximum value of the sum 'wind' is 321.0 from 'weather'=rain when 'temp_min' is less than mean.",
        #                                         "relevance": 5,
        #                                         "clarity": 5,
        #                                         "contribution": 4,
        #                                         "score": 14,
        #                                         "reason": 
        #                                         {{
        #                                             "relevance": "Directly links wind values and temperature variations to the occurrence of rain, matching the question's theme.",
        #                                             "clarity": "The fact is expressed clearly with well-defined conditions and values.",
        #                                             "contribution": "Provides a unique insight into the relationship between low temperatures and high wind sums in rainy weather."
        #                                         }}
        #                                     }}\n\n
        #                                     Question:\n\n{question}\n\n.
        #                                     Data facts:\n\n{facts}\n\n.
        #                                     \n\n{format_instructions}""",               
        #                         input_variables=["question", "facts"],
        #                         partial_variables={"format_instructions": parser_score_json .get_format_instructions()},

        #             )
        #     chain_score = prompt_score | llm | parser_score_json 
        #     score_from_gpt = chain_score.invoke(input = {"question":selected_question, "facts":facts_to_score})
        #     # !!!!!!TypeError: string indices must be integers, not 'str'
        #     sorted_score = sorted(score_from_gpt, key=lambda x: x["score"], reverse=True)
        #     # temp position for renew stage
        #     st.session_state["stage"] = "initial"

####################################################################################################################################


        #     # Visualize the top-3 facts after scoring
        #     for fact in sorted_score[0:3]:
        #         index = sorted_score.index(fact) + 1
        #         data_fact = fact["fact_text"]
        
####################################################################################################################################
       
       
                

        
    # st.button("Skip")


# Display chosen datasets 
if chosen_dataset :
   st.subheader(chosen_dataset)
   st.dataframe(datasets[chosen_dataset],hide_index=True)

# Insert footer to reference dataset origin  
footer="""<style>.footer {position: fixed;left: 0;bottom: 0;width: 100%;text-align: center;}</style><div class="footer">
<p> <a style="display: block; text-align: center;"> Datasets courtesy of NL4DV, nvBench and ADVISor </a></p></div>"""
st.caption("Datasets courtesy of NL4DV, nvBench and ADVISor")

# Hide menu and footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)




############################################ OLD CODE ########################################################
# # Call gpt-4o-mini and return 1 mainquestion, 3 subquestion in JSON format
# class Questions(BaseModel):
#     Main_Question: str = Field()
#     Theme: str = Field()
#     Question: str = Field()
#     Purpose: str = Field()
#     Insight: str = Field()
# parser_json = JsonOutputParser(pydantic_object = Questions)
# prompt_Q = PromptTemplate(
#                         template="""[EXAMPLE] 
#                                         Consider a tabluar data containing about learning status of students in different grades in covid-19.
#                                         There are column name like: "instructional mode", "grade", "learning hours", "economic condition", "internet access", "living situation", "social class", etc.
#                                         And you use three viualization charts to discuss the main question: "How children are learning in public schools in the United States since covid-19 pandemic?".
#                                         Here is the JSON array of the main question and the three charts:
#                                         [
#                                         "Main_Question": "How children are learning in public schools in the United States since covid-19 pandemic?",
#                                         "1":[
#                                         "Theme": "Percentage Distribution of Students by Enrollment in Instructional Modes",
#                                         "Question": "What percentage of students in differt grades were enrolled in different instructional modes (in-person, hybrid, remote) during the pandemic?",
#                                         "Purpose": "This question can represented by a pie chart provide an overview of how students were distributed across various learning modes, helping to assess the extent of remote learning adoption.",
#                                         "Insight": "Highlights the overall impact of the pandemic on instructional mode preferences, revealing which mode was most common and how it changed over time.",
#                                         ],
#                                         "2":[
#                                         "Theme": "Average Learning Hours by Month",
#                                         "Question": "How did the average learning hours change over time during the pandemic?",
#                                         "Purpose": "It shows trends in student learning hours, highlighting periods of reduced learning due to school closures or transitions between instructional modes.",
#                                         "Insight": "Identifies specific months where students faced challenges in maintaining consistent learning hours.",
#                                         ],
#                                         "3":[
#                                         "Theme": "Distribution of Instructional Model for Disadvantaged Groups",
#                                         "Question": "How did instructional modes vary among disadvantaged groups (e.g., low-income students, minority groups)?",
#                                         "Purpose": "To examine whether disadvantaged groups had equal access to in-person learning or were disproportionately placed in remote settings.",
#                                         "Insight": "Reveals potential inequities in educational access, showing that some groups may have faced more barriers to in-person learning than others.",
#                                         ],
#                                         
#                                         ]
#                                     [/EXAMPLE]
#                                         You are an expert in the domain of given dataset.
#                                         Here are column name {head} from the dataset for your reference.
#                                         First, use your extensive knowledge to think about a main question to discuss.
#                                         Then, think step by step and construct three viualization charts to discuss the main question.\n{format_instructions}""",
#                         input_variables=["head"],
#                         partial_variables={"format_instructions": parser_json.get_format_instructions()},
#             )
# llm = ChatOpenAI(model_name="gpt-4o-mini", api_key = openai_key)
# chain_Q = prompt_Q | llm | parser_json
# Q_from_gpt = chain_Q.invoke(input = {"head":head})
# st.write("Poster Question:", Q_from_gpt["Main_Question"])