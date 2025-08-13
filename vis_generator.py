from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate,SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage



def agent_consistent(table_name, query,title, data_schema,sampled_data, vlspec, openai_key):
    prompt = """
    You are a data analysis assistant that uses Vega-Lite to create data visualizations.
    Your task is to create **six** optimal visualizations for the six query {query} based on the {table_name} data table using Vega-Lite.
    The visualizations title for the six charts should be the following titles in order:{title}
    The data content overview of the {table_name} table is as follows:{data_schema}
    The {table_name} sample data table is as follows:{sampled_data}
    Here is the Vega-Lite specification suitable for these queries for your reference:{vlspec}
    Create six Vega-Lite specifications for each query that obey the following rules:
    [\Rules]
    Rule 1: The "$schema" property should be: "https://vega.github.io/schema/vega-lite/v5.json".
    Rule 2: The "transform" property should be put ahead of the "encoding" property.
    Rule 3: Always include "data" attribute as following in the Vega-Lite output:{{url:"https://raw.githubusercontent.com/zihzi/DATA2Poster/refs/heads/main/data/{table_name}.csv"}}.
    Rule 4: Pay attention to the query description to determine whether you should use "filter" transformation in the "transform" property.
    Rule 5: If you use "aggregate" operation in the "transform" property, the "groupby" property of "aggregate" should be correctly specified.
    Rule 6: If you use "window" operation to rank data in the "transform" property, make sure **NO "groupby" property** exists after "window" operation in the "transform" property.
    Rule 7: Make sure no "sort" operations exist in the "transform" property, you should define the order of axes only in the "encoding" property.
    Rule 8: Make sure no "false" and "true" is used in the Vega-Lite specification.
    Rule 9: Make sure  no "aggregate: None" is used in the Vega-Lite specification.
    Rule 10: If title is including "Top N", make sure to **add "xOffset" operation in the encoding property**.
    [\Rules]

    Instructions for generating the Vega-Lite specification:
    [\Instructions]
    - Read the data content overview and the sample data table carefully to understand the data type and valid value for each column in the data table.
    - The Vega-Lite specification should be a valid JSON object.
    - ONLY return the Vega-Lite specification. DO NOT include any preamble text. Do not include explanations or prose.\n\n. 
    [\Instructions]

    Visual-design contract that apply to **every** spec:
    [\Visual-design contract]
    1. Tick labels & counts
     - Always use -30° rotation for x-axis categorical labels.
     - Max six ticks per axis.

    2. Colour encoding
     - The six charts should use the same color scheme.
     - The same column should use the same color across all charts.
     - The first chart and the third chart should **only use the same one color**, and do not add a redundant legend.
     - If there is **NO "xOffset"** in the Vega-Lite specification, **do not add "color" operation**, and do not add a redundant legend.
     - Make sure the colors are harmonious.

    [\Visual-design contract]

    [\Constraints]
    - Using a grouped bar chart to compare different value if one column. For example, compare male vs. female employment across economic sectors.
    - NEVER generate multiple subplots.
    - NEVER USE stacked bar chart.
    - If title is including "Top N", **do not modify "sort": [{{"field": "..", "order": "descending"}}]** in the "window" property from {vlspec}.
    - ALWAYS add the following properties in the output Vega-Lite specification and do not revise them:
        {{
            "config": {{
            "title": {{ "fontSize": 44 }},
            "axis": {{
            "titleFontSize": 44,
            "labelFontSize": 30,
            "tickCount": 6,
            "labelLimit": 0,
            "titlePadding": 10
            }},
            "legend": {{
            "titleFontSize": 26,
            "labelFontSize": 26,
            "labelLimit": 0
            }}
            
        }}
    [\Constraints]
   
    **Output (exact JSON)**  
    Do not INCLUDE ```json```.Do not add other sentences after this json data.
    Return **only** the final JSON in this structure:
        {{
        "visualizations": [
            {{
            "<Vega-Lite JSON specification for chart 1>",
            }},
            {{
            "<Vega-Lite JSON specification for chart 2>",
            }},
            {{
            "<Vega-Lite JSON specification for chart 3>",
            }},
            {{
            "<Vega-Lite JSON specification for chart 4>",
            }},
            {{
            "<Vega-Lite JSON specification for chart 5>",
            }},
            {{
            "<Vega-Lite JSON specification for chart 6>",
            }}
        ]
        }}"""                      
    prompt_input = PromptTemplate(
                        template=prompt,
                        input_variables=["table_name", "query", "title","sampled_data", "vlspec"],
            )
    # interact with LLM
    llm = ChatOpenAI(model_name="gpt-4.1-2025-04-14", api_key = openai_key)
    prompt_for_chain = ChatPromptTemplate.from_messages(
                        messages=[SystemMessagePromptTemplate.from_template(prompt_input.template)
                        ]
                    )
    chain = prompt_for_chain | llm      
    response = chain.invoke(input= {"table_name":table_name, "query":query, "title":title,"data_schema":data_schema,"sampled_data":sampled_data,"vlspec":vlspec})
    return response.content


# **Agent 1: Generates Initial Visualization Code**
def agent_1_generate_code(table_name, query,title, data_schema,sampled_data, vlspec, openai_key):
    prompt = """
    You are a data analysis assistant that uses Vega-Lite to create data visualizations.
    Your task is to create the optimal visualization for the {table_name} data table using Vega-Lite to complete this query:{query}
    The visualization chart title should be:{title}
    The data content overview of the {table_name} table is as follows:{data_schema}
    The {table_name} sample data table is as follows:{sampled_data}
    Here is the Vega-Lite specification suitable for this query for your reference:{vlspec}
    Create a Vega-Lite specification obey the following rules:
    [\Rules]
    Rule 1: The "$schema" property should be: "https://vega.github.io/schema/vega-lite/v5.json".
    Rule 2: The "transform" property should be put ahead of the "encoding" property.
    Rule 3: Always include "data" attribute as following in the Vega-Lite output:{{url:"https://raw.githubusercontent.com/zihzi/DATA2Poster/refs/heads/main/data/{table_name}.csv"}}.
    Rule 4: Pay attention to the query description to determine whether you should use "filter" transformation in the "transform" property.
    Rule 5: If you use "aggregate" operation in the "transform" property, the "groupby" property of "aggregate" should be correctly specified.
    Rule 6: If you use "window" operation to rank data in the "transform" property, make sure no "groupby" operations exist after "window" operation in the "transform" property.
    Rule 7: Make sure no "sort" operations exist in the "transform" property, you should define the order of axes only in the "encoding" property.
    Rule 8: **Always add "sort" order as "descending" in the "encoding" property.
    Rule 9: Make sure no "false" and "true" is used in the Vega-Lite specification.
    Rule 10: Make sure  no "aggregate: None" is used in the Vega-Lite specification.
    Rule 11: If title is including "Top N", make sure to **add "xOffset" operation in the encoding property**.
    [\Rules]

    Instructions for generating the Vega-Lite specification:
    [\Instructions]
    - Read the data content overview and the sample data table carefully to understand the data type and valid value for each column in the data table.
    - You MUST first generate a brief plan for how you would solve the query e.g.  how to prepare the data, how to manipulate the data.
    - The Vega-Lite specification should be a valid JSON object.
    - ONLY return the Vega-Lite specification. DO NOT include any preamble text. Do not include explanations or prose.\n\n. 
    [\Instructions]

    [\Constraints]
    - Using a grouped bar chart to compare different value if one column. For example, compare male vs. female employment across economic sectors.
    - NEVER generate multiple subplots.
    - Never use stacked bar chart.
    - ALWAYS add the following properties in the output Vega-Lite specification and do not revise them:
        {{
            "config": {{
            "title": {{ "fontSize": 44 }},
            "axis": {{
            "titleFontSize": 44,
            "labelFontSize": 28,
            "tickCount": 6,
            "labelLimit": 0,
            "titlePadding": 10
            }},
            "legend": {{
            "titleFontSize": 26,
            "labelFontSize": 26,
            "labelLimit": 0
            }}
            
        }}
    [\Constraints]
   
    **Output (exact JSON)**  
    Do not INCLUDE ```json```.Do not add other sentences after this json data.
    """                      
    prompt_input = PromptTemplate(
                        template=prompt,
                        input_variables=["table_name", "query", "title","sampled_data", "vlspec"],
            )
    # interact with LLM
    llm = ChatOpenAI(model_name="gpt-4.1-2025-04-14", api_key = openai_key)
    prompt_for_chain = ChatPromptTemplate.from_messages(
                        messages=[SystemMessagePromptTemplate.from_template(prompt_input.template)
                        ]
                    )
    chain = prompt_for_chain | llm      
    response = chain.invoke(input= {"table_name":table_name, "query":query, "title":title,"data_schema":data_schema,"sampled_data":sampled_data,"vlspec":vlspec})
    return response.content
    

# **Agent 2: Improves Code Quality** TEST:vis feedback
def agent_2_improve_code(query, url, openai_key):
    prompt = """
    You are a seasoned Data Visualization Critic with a background in data journalism and statistical analysis. 
    Your task is to review and critique visualization and clearly outline concrete steps to improve it, avoiding vague or suggestive phrasing.
    The visualization is aim to complete a query.
    Think step by step how to complete the query and evaluate the visualization based on the following aspects:
    1. Title and Axis Labels:
    - Check if the x-axis label is "year" or "time", it should not be expressed with a decimal point.
    2. Number of Subplots:
    - Make sure the chart is a single plot **without any subplots**.
    3. Color Scheme:
    - There should be no redundant colors if the chart has only one category.
    4. Data Integrity:
    - When the title is including 'Top N', ensure the chart **only** displays the correct number of top N items.

    [\Instructions]
    - Provide an ordered list of actionable steps (e.g., Step 1, Step 2…). 
    - Each step should be specific and implementable. Avoid weak suggestions like "Consider using...", "You might want to...", "If possible," or "It may help to...".
    - Focus on visual clarity and correctness. 
    - Assume the visualization is built using vega-lite declarative charting library.
    - ONLY return the concise and clear steps about how to improve the visualization as structured points without any additional explanation."The steps are: ...".

    [\Instructions]
    [\Constraints]
    - Limit the list of necessary improvements up to five items.
    - NEVER suggest to add text or annotations on the bar.
    [\Constraints]
    """
    
    # interact with LLM
    llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", temperature=0, api_key = openai_key)
    prompt = [
        SystemMessage(content=prompt),
        HumanMessage(content=[
            {
                "type": "text", 
                "text": f"The chart is generated from the following query:\n\n{query}"
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
def agent_improve_vis(code,feedback,sampled_data, openai_key):
    prompt = """
    You are a data visualization engineer tasked with improving a chart based on expert feedback. 
    Your task is to revise the Vega-Lite specification so that it addresses each point in the critique below, while preserving the original chart's intent.


    The Vega-Lite specification must follow the following rules:
    [\Rules]
    Rule 1: The "$schema" property should be: "https://vega.github.io/schema/vega-lite/v5.json".
    Rule 2: The "transform" property should be put ahead of the "encoding" property.
    Rule 3: Always include "data" attribute as following in the Vega-Lite output:{{url:"https://raw.githubusercontent.com/zihzi/DATA2Poster/refs/heads/main/data/{table_name}.csv"}}.
    Rule 4: Pay attention to the query description to determine whether you should use "filter" transformation in the "transform" property.
    Rule 5: If you use "aggregate" operation in the "transform" property, the "groupby" property of "aggregate" should be correctly specified.
    Rule 6: If you use "window" operation to rank data in the "transform" property, make sure no "groupby" operations exist after "window" operation in the "transform" property.
    Rule 7: Make sure no "sort" operations exist in the "transform" property, you should define the order of axes only in the "encoding" property.
    Rule 8: Make sure no "false" and "true" is used in the Vega-Lite specification.
    Rule 9: Make sure  no "aggregate: None" is used in the Vega-Lite specification.
    [\Rules]

    Instructions for improving the chart:
    [\Instructions]
    - Improvements for better visualization practices, ensure it addresses each point in the critique.
    - The Vega-Lite specification should be a valid JSON object.
    - ONLY return improved the Vega-Lite specification. DO NOT include any preamble text. Do not include explanations or prose.
    [\Instructions] 


    Visual-design contract that apply to **every** spec:
    [\Visual-design contract]
    1. Tick labels & counts
     - Always use -30° rotation for x-axis categorical labels.
     - Max six ticks per axis.

    2. Colour encoding
     - DO NOT CHANGE the color if the feedback does not mention color.
     - If the x-axis has only one single category, render it with just **ONE** solid color, and do not add a redundant legend. 
  
    [\Visual-design contract]

    [\Constraints]
    - Using a grouped bar chart to compare different value if one column. For example, compare male vs. female employment across economic sectors.
    - NEVER generate multiple subplots.
    - NEVER USE stacked bar chart.
    - ALWAYS add the following properties in the output Vega-Lite specification and do not revise them:
        {{
            "config": {{
            "title": {{ "fontSize": 44 }},
            "axis": {{
            "titleFontSize": 44,
            "labelFontSize": 30,
            "tickCount": 6,
            "labelLimit": 0,
            "titlePadding": 10
            }},
            "legend": {{
            "titleFontSize": 36,
            "labelFontSize": 30,
            "labelLimit": 0
            }}
            
        }}
    [\Constraints]
 
    **Output (exact JSON)**  
    Do not INCLUDE ```json```.Do not add other sentences after this json data.                                      
    """
    prompt_input = PromptTemplate(
                        template="""
                        Here is the vega-lite specification to revise:{code}\n\n    
                        The the critique is as follows:{feedback}\n\n
                        The  sample data table for the Vega-Lite specification is as follows:{sampled_data}\n\n
                        """,
                       input_variables=["code","feedback","sampled_data"],
            )
    # interact with LLM
    llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", temperature=0, api_key = openai_key)
    prompt_for_chain = ChatPromptTemplate.from_messages(
                        messages=[SystemMessage(content=prompt),
                                  HumanMessagePromptTemplate.from_template(prompt_input.template)
                                  ]                         
                    )
    chain = prompt_for_chain | llm      
    response = chain.invoke(input= {"code":code,"feedback":feedback,"sampled_data":sampled_data})
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

# ** Agent 4:Validator Agent**
def agent_4_validate_spec(vlspec, sampled_data, openai_key):
    prompt = """
    You are a validator for Vega-Lite specifications.   
    Your task is to check a Vega-Lite JSON specification and correct if it has error. 
    Please approach this task methodically and reason step by step.
    [/Background]
    A valid Vega-Lite spec typically includes the following components:
    {
    "data": { "name" / "url": ... },
    "mark": "bar" | "line" | "point" | "area" | ...,
    "encoding": {
        "x": { "field": ..., "type": ..., "aggregate": ... },
        "y": { "field": ..., "type": ..., "aggregate": ... },
        "color": { "field": ..., "type": ... },
        ...
    },
    "transform": [...],
    "width": ..., 
    "height": ...
    }
    [/Background]

    [/Rules for Improvement]
    a. Must include "x" and "y" encodings with clear "type" and "field"
    b. For grouped or colored charts
       - Add "color" or "column" encodings for categorical breakdowns
       - Do not include unnecessary visual channels
    c. Binning and Time Units
       - Use "bin": true for numeric histograms
       - Use "timeUnit" for temporal groupings — prefer over transforming raw dates
    d. Include "tooltip" encoding for better readability
    e. Adjust axis ticks, titles, and labels for clarity
    f. Make sure no "false" and "true" is used in the Vega-Lite specification.
    g. Make sure "redyellowblue" color scheme is used in the Vega-Lite specification.
    [/Rules for Improvement]

    [/Constraints]
    1. Mark encodings
        - line, area, trail marks must have both x and y.
        - bar marks must have at least one positional (x or y) and one size/ordinal encoding.
    2. Scale and type consistency
        - scale.type == "log"|"pow" requires the field's type == "quantitative".
    3. Aggregate functions
        - Allowed aggregates: count, sum, mean, median, min, max, rank, stdev, variance,.
    4. Binning
        - bin: true (or object) only on type == "quantitative".
        - Cannot combine bin with continuous scales like "log".
    5. Stacking
        - stack: "zero"|"normalize"|"center" only on bar/area/line and requires a quantitative axis.
    6. Color/Opacity/Size channels
        - Field-based color/opacity/size must be quantitative or ordinal.
        - Literal encodings must use value.
    7. Sort order
        - sort must reference an encoded field of matching type.
    8. Parameter binding.
        - bind on parameters must use a supported input type (legend, scales, or form controls).
    9. Output must be a valid JSON object
    10. ONLY return the full corrected Vega-Lite JSON specification without any additional text
    [/Constraints]
    **Output (exact JSON)**  
    Do not INCLUDE ```json```.Do not add other sentences after this json data.
    """
    prompt_input = PromptTemplate(
                        template="""
                        Here is the current Vega-Lite Specification should be checked:{vlspec}\n\n    
                        The  sample data table for the Vega-Lite specification is as follows:{sampled_data}\n\n
                        """,
                       input_variables=["vlspec"],
            )
    # interact with LLM
    llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", temperature=0, api_key = openai_key)
    prompt_for_chain = ChatPromptTemplate.from_messages(
                        messages=[SystemMessage(content=prompt),
                                  HumanMessagePromptTemplate.from_template(prompt_input.template)
                                  ]                         
                    )
    chain = prompt_for_chain | llm      
    response = chain.invoke(input= {"vlspec":vlspec,"sampled_data":sampled_data})
    return response.content
