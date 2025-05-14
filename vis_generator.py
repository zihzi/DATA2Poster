from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate,SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage


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
    

# **Agent 2: Improves Code Quality** TEST:vis feedback
def agent_2_improve_code(query, url, openai_key):
    prompt = """
    You are a seasoned Data Visualization Critic with a background in data journalism, cognitive psychology, and statistical analysis. 
    Your task is to review and critique visualizations as if they were submissions for publication in a prestigious data science magazine.
    The visualizations is aim to complete a query.
    Think step by step how to complete the query and evaluate the visualizations based on the following aspects:
    1. Chart Type and Design:
    - Is the chart type appropriate for the data and the story being told? If not, suggest a more suitable type.
    2. Title and Axis Labels:
    - Ensure the chart has a clear title.
    - Do the X-axis and Y-axis labels exist, and are they complete?
    - Check if the labels are difficult to read, e.g., are they written vertically instead of horizontally?
    - The title should not be a direct question; instead, it should describe the data or trends being presented.
    3.Tick Labels:
    - Data Alignment: Are data points (bars, dots, lines) centered on or aligned with tick marks clearly? Or do they fall ambiguously between ticks?
    - Tick Density: Are the ticks too sparse or too dense for the chart's scale and purpose?
    - Tick Placement: Are the tick intervals logical (e.g., rounded numbers, meaningful time intervals)? Are they aligned with key features in the data?
    - Label Readability: Are tick labels readable and well-formatted? Do they overlap or clutter the chart?
    - Match with Data Granularity: Are ticks consistent with the resolution of the data (e.g., weekly ticks for weekly data)?
    - Tick Origin: Does the axis start at an appropriate origin (e.g., zero for bar charts)? If not, does the non-zero baseline distort interpretation?
    4. Legend Completeness:
    - Is the legend complete, and does it clearly indicate the color labels for different data series?
    - Ensure each color has a corresponding legend, making it easy for users to understand what the data represents.  
    ONLY return the concise improvements as structured points without any additional explanation."The improvements are: ...".
    """
    
    # interact with LLM
    llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", api_key = openai_key)
    prompt = [
        SystemMessage(content=prompt),
        HumanMessage(content=[
            {
                "type": "text", 
                "text": f"Here is the query to complete:\n\n{query}"
            },
            {
                "type": "text", 
                "text": "Here is the chart to review:"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": url
                },
            },
        ])
    ] 
    response = llm.invoke(prompt)
    return response.content
# **Agent 2: Improves Code Quality**
def agent_improve_vis(code,feedback, openai_key):
    prompt = """
    You are a data visualization engineer tasked with improving a chart based on expert feedback. 
    Your task is to revise the code so that it addresses each point in the critique below, while preserving the original chart’s intent.
    DATA IS ALREADY IN THE GIVEN CODE, SO NEVER USE "<Add dataset URL here>" or LOAD ANY DATA ELSE.
    ONLY revise the code between the lines of "def plot(data):" and "return chart".
    You MUST return a full EXECUABLE code. DO NOT include any preamble text. Do not include explanations or prose.
    [\Instructions]
    - Improvements for better visualization practices, such as clarity, readability, and aesthetics, while ensuring the primary focus is on meeting the user's specified requirements.
    - ONLY USE COLORS from this color palette: 'D9ED92','B5E48C','99D98C','76C893','52B69A','34A0A4','168AAD','1A759F','1E6091','184E77'.
    - Return only the improved code, without explanations 
    [\Instructions]                                        
    """
    prompt_input = PromptTemplate(
                        template="""
                        Here is the code to revise:{code}\n\n    
                        The the critique is as follows:{feedback}\n\n
                        """,
                       input_variables=["code","feedback"],
            )
    # interact with LLM
    llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", api_key = openai_key)
    prompt_for_chain = ChatPromptTemplate.from_messages(
                        messages=[SystemMessage(content=prompt),
                                  HumanMessagePromptTemplate.from_template(prompt_input.template)
                                  ]                         
                    )
    chain = prompt_for_chain | llm      
    response = chain.invoke(input= {"code":code,"feedback":feedback})
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



