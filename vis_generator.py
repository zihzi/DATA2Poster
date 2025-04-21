from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate,SystemMessagePromptTemplate
from langchain_core.messages import SystemMessage


# **Agent 1: Generates Initial Visualization Code**
def agent_1_generate_code(query, dataset, nl4DV_json, code_template, openai_key):
    prompt = """
    You are an expert Python programmer specializing in data visualization.
    Here is the query, dataset, visualization taskinformation, and code template for the viusalization chart.
    Refer to the vega-lite specification to generate full vega-lite visualization code BY MODIFYING THE SPECIFIED PARTS OF THE TEMPLATE. 
    ALWAYS make sure visualize each attributes and the task in the information. 
    The visualization code MUST BE EXECUTABLE and MUST NOT CONTAIN ANY SYNTAX OR LOGIC ERRORS (e.g., it must consider the data types and use them correctly). 
    You MUST first generate a brief plan for how you would solve the task e.g. how to set the parameters in each function correctly, how to prepare the data, how to manipulate the data so that it becomes appropriate for later functions to call etc,.
    ONLY return the code. DO NOT include any preamble text. Do not include explanations or prose.\n\n. 
    """                      
    prompt_input = PromptTemplate(
                        template="""
                        Here is the query.\n\n
                        {query}\n\n
                        Here is the dataset.\n\n
                        {dataset}\n\n
                        Here is the information.\n\n
                        {nl4DV_json}\n\n
                        Here is the code template.\n\n
                        {code_template}\n\n
                        """,
                        input_variables=["query", "dataset", "nl4DV_json", "code_template"],
            )
    # interact with LLM
    llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", api_key = openai_key)
    prompt_for_chain = ChatPromptTemplate.from_messages(
                        messages=[SystemMessage(content=prompt),                         
                        SystemMessagePromptTemplate.from_template(prompt_input.template)
                        ]
                    )
    chain = prompt_for_chain | llm      
    response = chain.invoke(input= {"query":query, "dataset":dataset, "nl4DV_json":nl4DV_json, "code_template":code_template})
    return response.content
    

# **Agent 2: Improves Code Quality**
def agent_2_improve_code(query, code, openai_key):
    prompt = """
    You are a senior data visualization expert highly skilled in revising visualization code based on a given query.
    Your task is to make the visualization code intepret the query for better readability, aesthetics, and effectiveness.
    Here is the query and code.
    DATA IS ALREADY IN THE GIVEN CODE, SO NEVER USE "<Add dataset URL here>" or LOAD ANY DATA ELSE.
    ONLY revise the code between the lines of "def plot(data):" and "return chart".
    You MUST return a full EXECUABLE code. DO NOT include any preamble text. Do not include explanations or prose.
    [\Instructions]
    - Carefully read and analyze the user query to understand the specific requirements. 
    - Examine the provided code to understand how the current plot is generated. 
    - Assess the plot type, the data it represents, labels, titles, colors, and any other visual elements. 
    - ALWAYS ADD LEGEND TO THE PLOT.
    - Evaluate how well is the code that applies any kind of data transformation (filtering, aggregation, grouping, null value handling etc).
    - Improvements for better visualization practices, such as clarity, readability, and aesthetics, while ensuring the primary focus is on meeting the user's specified requirements.
    - ONLY USE COLORS from this color palette: 'D9ED92','B5E48C','99D98C','76C893','52B69A','34A0A4','168AAD','1A759F','1E6091','184E77'.
    - Return only the improved code, without explanations 
    [\Instructions]                                        
    """
    prompt_input = PromptTemplate(
                        template="""
                        Here is the query.\n\n
                        {query}\n\n
                        Here is the code.\n\n
                        {code}\n\n
                        """,
                       input_variables=["query", "code"],
            )
    # interact with LLM
    llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", api_key = openai_key)
    prompt_for_chain = ChatPromptTemplate.from_messages(
                        messages=[SystemMessage(content=prompt),
                                  SystemMessagePromptTemplate.from_template(prompt_input.template)
                                  ]                         
                    )
    chain = prompt_for_chain | llm      
    response = chain.invoke(input= {"query":query, "code":code})
    return response.content

# **Agent 3: Ensures Code is Executable**
def agent_3_fix_code(code,openai_key):
    prompt = """
    You are a Python debugging expert highly skilled in revising visualization code to make the code EXECUABLE.
    The following Python visualization code needs to be checked for execution errors.
    Your task is to fix any issues that may cause the code to fail to execute, such as syntax, import errors, incorrect function usage, logic errors, undefined parameters, etc.
    ONLY revise the code between the lines of "def plot(data):" and "return chart".
    DATA IS ALREADY IN THE GIVEN CODE, SO NEVER USE "<Add dataset URL here>" or LOAD ANY DATA ELSE.
    NEVER USE THE ATTRIBUTE "configure_background" in the code.
    You MUST return a full EXECUABLE code. DO NOT include any preamble text. Do not include explanations or prose.
    Instructions:
    - Ensure all necessary imports are included
    - Fix incorrect function calls or missing parameters
    - Return only the fixed code, without explanations
    """
    prompt_input = PromptTemplate(
                        template="""
                        Here is the code.\n\n
                        {code}\n\n
                        """,
                        input_variables=["code"]
            )
    # interact with LLM
    llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", api_key = openai_key)
    prompt_for_chain = ChatPromptTemplate.from_messages(
                        messages=[SystemMessage(content=prompt),
                                  SystemMessagePromptTemplate.from_template(prompt_input.template)
                                  ]                         
                    )
    chain = prompt_for_chain | llm      
    response = chain.invoke(input= {"code":code})
    return response.content



