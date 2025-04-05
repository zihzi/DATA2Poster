from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate,SystemMessagePromptTemplate
from langchain_core.messages import SystemMessage


# **Agent 1: Generates Initial Visualization Code**
def agent_1_generate_code(query, dataset, nl4DV_json, vl_Spec, code_template, instructions, openai_key):
    prompt = """
    You are an expert Python programmer specializing in data visualization.
    Here is the query, dataset, information, vega-lite specification and code template for the viusalization chart.
    Here is step by step detailed instruction on how to write python code to fulfill the user query's requirements.
    Refer to the instructions and use the vega-lite specification to generate full vega-lite visualization code BY MODIFYING THE SPECIFIED PARTS OF THE TEMPLATE. 
    ALWAYS make sure visualize each attributes and the task in the information. 
    The visualization code MUST BE EXECUTABLE and MUST NOT CONTAIN ANY SYNTAX OR LOGIC ERRORS (e.g., it must consider the data types and use them correctly). 
    You MUST first generate a brief plan for how you would solve the task e.g. what transformations you would apply if you need to construct a new column, 
    what attributes you would use for what fields, what aesthetics you would use, etc.
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
                        Here is the vega-lite specification.\n\n
                        {vl_Spec}\n\n
                        Here is the code template.\n\n
                        {code_template}\n\n
                        Here is the instructions.\n\n
                        {instructions}
                        """,
                        input_variables=["query", "dataset", "nl4DV_json", "vl_Spec", "code_template", "instructions"],
            )
    # interact with LLM
    llm = ChatOpenAI(model_name="gpt-4o-mini-2024-07-18", api_key = openai_key)
    prompt_for_chain = ChatPromptTemplate.from_messages(
                        messages=[SystemMessage(content=prompt),                         
                        SystemMessagePromptTemplate.from_template(prompt_input.template)
                        ]
                    )
    chain = prompt_for_chain | llm      
    response = chain.invoke(input= {"query":query, "dataset":dataset, "nl4DV_json":nl4DV_json, "vl_Spec":vl_Spec,"code_template":code_template,"instructions":instructions})
    return response.content
    

# **Agent 2: Improves Code Quality**
def agent_2_improve_code(query, code, nl4DV_json, openai_key):
    prompt = """
    You are a senior data visualization expert highly skilled in revising visualization code based on a given query.
    Your task is to make the visualization code intepret the query for better readability, aesthetics, and effectiveness.
    Here is the query, the code, and the information for your reference.
    DATA IS ALREADY IN THE GIVEN CODE, SO NEVER USE "<Add dataset URL here>" or LOAD ANY DATA ELSE.
    ONLY revise the code between the lines of "def plot(data):" and "return chart".
    You MUST return a full EXECUABLE code. DO NOT include any preamble text. Do not include explanations or prose.
    [\Instructions]
    - Carefully read and analyze the user query to understand the specific requirements. 
    - Examine the provided code to understand how the current plot is generated. 
    - Assess the plot type, the data it represents, labels, titles, colors, and any other visual elements. 
    - Consider the information, focus on "attributeMap" and "taskMap", and evaluate how well is the code that applies any kind of data transformation (filtering, aggregation, grouping, null value handling etc).
    - Improvements for better visualization practices, such as clarity, readability, and aesthetics, while ensuring the primary focus is on meeting the user's specified requirements.
    - ONLY USE COLORS from this color palette: ["E37663", "FCD5CE", "C1928B", "DFACA4", "BDBDB3", "D8E2DC", "E3D7CA", "FFE5D9", "FFD7BA", "FEC89A"].
    - Return only the improved code, without explanations 
    [\Instructions]                                        
    """
    prompt_input = PromptTemplate(
                        template="""
                        Here is the query.\n\n
                        {query}\n\n
                        Here is the code.\n\n
                        {code}\n\n
                        Here is the information.\n\n
                        {nl4DV_json}\n\n
                        """,
                       input_variables=["query", "code", "nl4DV_json"],
            )
    # interact with LLM
    llm = ChatOpenAI(model_name="gpt-4o-mini-2024-07-18", api_key = openai_key)
    prompt_for_chain = ChatPromptTemplate.from_messages(
                        messages=[SystemMessage(content=prompt),
                                  SystemMessagePromptTemplate.from_template(prompt_input.template)
                                  ]                         
                    )
    chain = prompt_for_chain | llm      
    response = chain.invoke(input= {"query":query, "code":code, "nl4DV_json":nl4DV_json})
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
    llm = ChatOpenAI(model_name="gpt-4o-mini-2024-07-18", api_key = openai_key)
    prompt_for_chain = ChatPromptTemplate.from_messages(
                        messages=[SystemMessage(content=prompt),
                                  SystemMessagePromptTemplate.from_template(prompt_input.template)
                                  ]                         
                    )
    chain = prompt_for_chain | llm      
    response = chain.invoke(input= {"code":code})
    return response.content



