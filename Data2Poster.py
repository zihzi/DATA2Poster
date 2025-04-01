import streamlit as st
import pandas as pd
import re
import base64
import json
import altair as alt
from operator import itemgetter
from dataclasses import asdict
from nl4dv import NL4DV
from dataSchema_builder import get_column_properties
from data_cleaning import clean_csv_data
from insight_generation.dataFact_scoring import score_importance
from insight_generation.main import generate_facts
from selfAugmented_thinker import self_augmented_knowledge
from question_evaluator import generate_questions
from vis_generator import agent_1_generate_code, agent_2_improve_code, agent_3_fix_code
from pathlib import Path
from poster_generator import create_pdf

# Import langchain modules
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores.faiss import FAISS



    
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
    datasets["Cars"] = pd.read_csv("data/Cars.csv")
    datasets["Sleep_health_and_lifestyle"] = pd.read_csv("data/Sleep_health_and_lifestyle.csv")
    datasets["bike_sharing_day"] = pd.read_csv("data/bike_sharing_day.csv")
    datasets["cancer_by_year"] = pd.read_csv("data/cancer_by_year.csv")
    datasets["social_media"] = pd.read_csv("data/social_media.csv")
   
    
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
            # Clean the dataset
            datasets[file_name] = clean_csv_data(datasets[file_name])
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

# load a prompt from a file
def load_prompt_from_file(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        return file.read()
    
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
        code = code[:index] + f"data = pd.read_csv('data/{chosen_dataset}.csv')\n\n" + code[index:] + f"\n\nchart.save('DATA2Poster_json/vega_lite_json_{count}.json')"
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
        # use gpt-4o-mini as llm
        llm = ChatOpenAI(model_name="gpt-4o-mini-2024-07-18", temperature=0.2,api_key = openai_key)
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
            # Explain these column names WITHOUT mention the statistical value given by the summary in natural language description that can help the user understand the dataset better.
            # Please do not add any extra prose to your response.
            # You should list reponse as below format:\n\n
            # \"Overview of Columns(IN BOLD):\n\n
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
            # Call gpt-4o-mini to select columns related to user-selected column in JSON format 
            if skip_true or selected_column:
                user_select_column_template = load_prompt_from_file("prompt_templates/user_select_column.txt")       
                prompt_column = PromptTemplate(
                                            template=user_select_column_template,
                                            input_variables=["user_selected_column", "chosen_dataset", "chosen_data_schema"],        
                                )
                with open("json_schema/column_json_schema.json", "r") as f:
                    column_json_schema = json.load(f)
                chain_column = prompt_column | llm.with_structured_output(column_json_schema)
                columns_from_gpt = chain_column.invoke(input = {"user_selected_column":user_selected_column, "chosen_dataset":chosen_dataset, "chosen_data_schema":chosen_data_schema})
                
                # log the selected column and related columns
                st.subheader(f"related column set:",divider = True)
                for i in range(1,3):
                    st.write("\"", f'**{columns_from_gpt[f"related_column_{i}"]["name"]}**',"\"")
                    st.write(f'**Reason:**', f'**{columns_from_gpt[f"related_column_{i}"]["reason"]}**')
                        
                # Extract dataFrame by user_selected_column and the related columns from gpt 
                related_column = [columns_from_gpt["related_column_1"]["name"],columns_from_gpt["related_column_2"]["name"]]
                df_for_cal = datasets[chosen_dataset][[user_selected_column] + related_column]   

                # Produce columns to generate fact list
                column_list_for_Q = []
                facts_list = []
                columns_dic = {columns_from_gpt["selected_column"]["name"]: columns_from_gpt["selected_column"]["dtype"]}
                for i in range(1,3):
                    columns_dic[columns_from_gpt[f"related_column_{i}"]["name"]] = columns_from_gpt[f"related_column_{i}"]["dtype"]         
                breakdown = [col for col, dtype in columns_dic.items() if dtype == "C" or dtype == "T"]   
                measure = [col for col, dtype in columns_dic.items() if dtype == "N"]
                column_list_for_Q.append(columns_dic)
                st.write("Columns for generating facts:",column_list_for_Q)
                # Categorization_facts ? to be fixed, not suitable for dataset type
                for col, dtype in columns_dic.items():
                    if dtype == "C":
                        ratio = round(df_for_cal[col].value_counts()[0]/len(df_for_cal),3)*100
                        focus = df_for_cal[col].value_counts().idxmax()
                        facts_list.append({"content":f"In {col}, {focus} accounts for the largest proportion of records({ratio}%).","score":1})
                    
                # list all possible combinations of breakdown and measure to generate facts
                if len(breakdown) == 2: 
                    if columns_dic[breakdown[0]] == "C":
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
                    elif columns_dic[breakdown[0]] == "T":
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
                    if columns_dic[breakdown[0]] == "C":
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
                    elif columns_dic[breakdown[0]] == "T": 
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
                facts_list = sorted(facts_list, key=itemgetter('score'), reverse=True)
                seen = set()
                for item in facts_list[:200]:
                    if item["content"] != "No fact." and item["content"] not in seen:
                        seen.add(item["content"])
                        st.session_state["fact"].append(item["content"])
                st.write("Facts:",st.session_state["fact"])

                # Create a vector store from facts
                # vectorstore = FAISS.from_texts(
                # st.session_state["fact"],
                # OpenAIEmbeddings(model="text-embedding-3-small", api_key = openai_key),
                # )
                # st.session_state["vectorstore"] = vectorstore
                
                # Create intermediate output as knowledge
                knowledge = self_augmented_knowledge(openai_key, chosen_dataset, column_list_for_Q, st.session_state["fact"])
                st.write("KnowledgeBase:",knowledge)

                # Combine facts into interesting patterns
                llm_pattern_template = load_prompt_from_file("prompt_templates/fact_idea_prompt.txt")
                prompt_pattern = PromptTemplate(
                                        template=llm_pattern_template,
                                        input_variables=["knowledgebase","facts"],

                            )
                with open("json_schema/fact_idea_schema.json", "r") as f:
                    pattern_json_schema = json.load(f)
                chain_pattern = prompt_pattern | llm.with_structured_output(pattern_json_schema)
                patterns_from_gpt = chain_pattern.invoke(input = {"knowledgebase":knowledge,"facts":st.session_state["fact"]})
                st.write("Interesting Patterns:",patterns_from_gpt)

                
                # Generate poster question based on interesting patterns
                support_fact_list = [patterns_from_gpt["supporting_facts"][0],patterns_from_gpt["supporting_facts"][1],patterns_from_gpt["supporting_facts"][2]]  
                
                llm_Q_template = load_prompt_from_file("prompt_templates/llm_question.txt")
                prompt_llm_Q = PromptTemplate(
                                        template=llm_Q_template,
                                        input_variables=["pattern_1","pattern_1_fact_1","pattern_1_fact_2","pattern_1_fact_3","columns_set_1","knowledgebase"],

                            )
                with open ("json_schema/llm_question_schema.json", "r") as f:
                    llm_Q_schema = json.load(f)
                llm_Q_chain = prompt_llm_Q | llm.with_structured_output(llm_Q_schema)
                llm_Q_from_gpt = llm_Q_chain.invoke(input = {"pattern_1":patterns_from_gpt["extracted_pattern"],"pattern_1_fact_1":support_fact_list[0],"pattern_1_fact_2":support_fact_list[1],"pattern_1_fact_3":support_fact_list[2],"columns_set_1":column_list_for_Q[0],"knowledgebase":knowledge})
                st.write("Question:",llm_Q_from_gpt)
                
                # log the llm question
                def log_response_to_json(knowledgebase, response):
                    log_data = {"knowkedgebase": knowledgebase, "response": response}
                    
                    with open("log/COT_Q_logs.json", "a") as f:
                        f.write(json.dumps(log_data,indent=2))  # Append new log entry
                
                # Save log
                log_response_to_json(knowledge, llm_Q_from_gpt)


                

                # Refine the questions
                 
                q_for_refine = [llm_Q_from_gpt["questions"]["question_1"],llm_Q_from_gpt["questions"]["question_2"],llm_Q_from_gpt["questions"]["question_3"]]
                actions = [llm_Q_from_gpt["questions"]["action_1"],llm_Q_from_gpt["questions"]["action_2"],llm_Q_from_gpt["questions"]["action_3"]]
                    
                advice, new_Q = generate_questions(openai_key,llm_Q_from_gpt["conclusion"], column_list_for_Q, q_for_refine,actions)
                st.write("advice:", advice)
                st.write("refine_Q:",new_Q)
                    
                questions_for_poster = [new_Q["poster_question"]]
                supported_fact = [llm_Q_from_gpt["questions"]["fact_1"],llm_Q_from_gpt["questions"]["fact_2"],llm_Q_from_gpt["questions"]["fact_3"]]
                Q_for_vis =[new_Q["questions"][0]+new_Q["actions"][0],new_Q["questions"][1]+new_Q["actions"][1],new_Q["questions"][2]+new_Q["actions"][2]]
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
            # q_for_nl4DV.append(st.session_state["Q_from_gpt"]["questions"]["question_1"]+" "+st.session_state["Q_from_gpt"]["questions"]["action_1"])
            # q_for_nl4DV.append(st.session_state["Q_from_gpt"]["questions"]["question_2"]+" "+st.session_state["Q_from_gpt"]["questions"]["action_2"])
            # q_for_nl4DV.append(st.session_state["Q_from_gpt"]["questions"]["question_3"]+" "+st.session_state["Q_from_gpt"]["questions"]["action_3"])

 
            # Call gpt-4o-mini to generate vlspec
            insight_list = []
            for query in q_for_nl4DV:
                idx = q_for_nl4DV.index(query)+1
                nl4DV_prompt_template = load_prompt_from_file("prompt_templates/nl4DV_prompt.txt")
                code_template = load_prompt_from_file("prompt_templates/code_template.txt")
                nl4_DV_prompt_input = load_prompt_from_file("prompt_templates/nl4DV_prompt_input.txt")
                nl4DV_prompt_input = PromptTemplate(
                        template=nl4_DV_prompt_input,
                        input_variables=["sample_data", "summary", "query"]
                    )
                nl4DV_prompt = ChatPromptTemplate.from_messages(
                        messages=[
                            SystemMessage(content = nl4DV_prompt_template),
                            HumanMessagePromptTemplate.from_template(nl4DV_prompt_input.template)
                        ]
                    )
                with open("json_schema/nl4DV_json_schema.json", "r") as f:
                    nl4DV_json_schema = json.load(f)
                nl4DV_chain = nl4DV_prompt | llm.with_structured_output(nl4DV_json_schema)      
                nl4DV_json = nl4DV_chain.invoke(input= {"sample_data":sample_data, "summary": summary, "query":query})

                # Call gpt-4o-mini to generate vis code
                print("\n🟢 Step 1: Generating Initial Code...\n")
                initial_code = agent_1_generate_code(query, datasets[chosen_dataset], nl4DV_json, nl4DV_json["visList"][0]["vlSpec"], code_template, openai_key)
                print(initial_code)
                print("\n🟡 Step 2: Improving Code Quality...\n")
                improved_code = agent_2_improve_code(query, initial_code, nl4DV_json, openai_key)
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
                    print(f"Final Error: {str(final_exception)}")
                
                # load the vega_lite_json for insight_prompt
                with open(f"DATA2Poster_json/vega_lite_json_{idx}.json", "r") as f:
                        chart = json.load(f)
                        # image for pdf
                        img = alt.Chart.from_dict(chart)
                        img.save(f"DATA2Poster_img/image_{idx}.png")
                        # json for chart_description
                        chart_type = chart["mark"]["type"]
                        chart_title = chart["title"]
                        x_field = "Title: " + chart["encoding"]["x"]["title"] + " Type: " + chart["encoding"]["x"]["type"]
                        y_field = "Title: " + chart["encoding"]["y"]["title"] + " Type: " + chart["encoding"]["y"]["type"]
                        for key in chart["datasets"]:
                            chart_data = chart["datasets"][key]
        
                chart_prompt_template = load_prompt_from_file("prompt_templates/chart_prompt.txt")
                prompt_chart = PromptTemplate(
                                        template=chart_prompt_template,
                                        input_variables=["query","chart_type","chart_title","x_field","y_field","chart_data"],
                            )
                
                chain_pattern = prompt_chart | llm
                chart_des = chain_pattern.invoke(input = {"query":query,"chart_type":chart_type,"chart_title":chart_title,"x_field":x_field,"y_field":y_field,"chart_data":chart_data})

                            
                # #  RAG
                # retrieve_docs = st.session_state["vectorstore"].max_marginal_relevance_search(query, k=3,fetch_k=25,lambda_mult=0.4)
                # retrieved_fact = [doc.page_content for doc in retrieve_docs] # retrieve_fact is a list of facts
                # vectorstore = st.session_state["vectorstore"]
                
                # fact_for_insight = retrieved_fact
                supported_fact = st.session_state["Q_from_gpt"]["supported_fact"][idx-1]

                st.write(f'**Question for Chart:**',f'**{query}**')
                st.write(f'**Chart Description:**', f'**{chart_des.content}**')
                st.write(f'**Supported Data Fact:**', f'**{supported_fact}**')
                # st.write(f'**Data Fact after RAG:**', f'**{retrieved_fact}**')

                # call gpt-4o-mini to generate insight description
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
                # insight_llm = ChatOpenAI(model_name="gpt-4o-mini-2024-07-18", api_key = openai_key, max_tokens=15)
                insight_chain = insight_prompt | llm
                insight = insight_chain.invoke(input= {"query":query, "chart_des":chart_des})
                st.write(f'**Insight Description:**', f'**{insight.content}**')
                insight_list.append(insight.content)
                st.vega_lite_chart(chart, theme = None)
                
         
            # Reset session state

            st.session_state["df"] = pd.DataFrame()
            st.session_state["fact"] = []
            st.session_state["vectorstore"] = []
            st.session_state["questions_for_poster"] = []
            st.session_state["Q_from_gpt"] = {}
            st.session_state["selection"] = ""
            # Create pdf and download
            pdf_title = selected_poster_question
            create_pdf(chosen_dataset, q_for_nl4DV , pdf_title, insight_list, openai_key)
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

