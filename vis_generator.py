from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate,SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage


# **Agent 1: Generates Initial Visualization Code**
def agent_1_generate_code(table_name, query,sampled_data, openai_key):
    prompt = """
    You are a data analysis assistant that uses Vega-Lite to create data visualizations.
    Your task is to create the optimal visualization for the {table_name} data table using Vega-Lite to complete this query:{query}
    The {table_name} sample data table is as follows:{sampled_data}
    Create a Vega-Lite specification obey the following rules:
    [\Rules]
    Rule 1: The "$schema" property should be: "https://vega.github.io/schema/vega-lite/v5.json".
    Rule 2: The "transform" property should be put ahead of the "encoding" property.
    Rule 3: Always include "data" attribute as following in the Vega-Lite output:{{url:"https://raw.githubusercontent.com/zihzi/DATA2Poster/refs/heads/main/data/{table_name}.csv"}}.
    Rule 4: Rule 3: Pay attention to the query description to determine whether you should use "filter" transformation in the "transform" property.
    Rule 5: If you use "aggregate" operation in the "transform" property, the "groupby" property of "aggregate" should be correctly specified.
    Rule 6: Make sure no "sort" operations exist in the "transform" property, you should define the order of axes only in the "encoding" property.
    Rule 7: Make sure no "layer","facet","hconcat", "vconcat", "concat", "repeat" property exist in the Vega-Lite specification.
    Rule 8: Make sure "yellowgreenblue" color scheme is used in the Vega-Lite specification.
    [\Rules]

    Instructions for generating the Vega-Lite specification:
    [\Instructions]
    - You MUST first generate a brief plan for how you would solve the query e.g.  how to prepare the data, how to manipulate the data.
    - The Vega-Lite specification should be a valid JSON object.
    - ONLY return the Vega-Lite specification. DO NOT include any preamble text. Do not include explanations or prose.\n\n. 
    [\Instructions]
    The example of Vega-Lite specification:
    [\Example]
    The query is:"Show all majors and corresponding number of students by a scatter plot."
    The Vega-Lite specification is:
            {{
                "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                "data": {{"url": "https://raw.githubusercontent.com/zihzi/DATA2Poster/refs/heads/main/data/students.csv"}},
                "transform": [
                    {{"aggregate": [{{"op": "count", "field": "Major", "as": "Number of Students"}}],"groupby": ["Major"]}}
                ],
                "mark": "point",
                "encoding": {{
                    "x": {{"field": "Major", "type": "nominal"}},
                    "y": {{"field": "Number of Students", "type": "quantitative"}}
                }}
            }}
    [\Example]
    """                      
    prompt_input = PromptTemplate(
                        template=prompt,
                        input_variables=["table_name", "query", "sampled_data", "label"],
            )
    # interact with LLM
    llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", api_key = openai_key)
    prompt_for_chain = ChatPromptTemplate.from_messages(
                        messages=[SystemMessagePromptTemplate.from_template(prompt_input.template)
                        ]
                    )
    chain = prompt_for_chain | llm      
    response = chain.invoke(input= {"table_name":table_name, "query":query, "sampled_data":sampled_data})
    return response.content
    

# **Agent 2: Improves Code Quality** TEST:vis feedback
def agent_2_improve_code(query, url, openai_key):
    prompt = """
    You are a seasoned Data Visualization Critic with a background in data journalism and statistical analysis. 
    Your task is to review and critique visualization and clearly outline concrete steps to improve it, avoiding vague or suggestive phrasing.
    The visualization is aim to complete a query.
    Think step by step how to complete the query and evaluate the visualization based on the following aspects:
    1. Chart Type and Design:
    - Is the chart type appropriate for the data and the story being told? If not, suggest a more suitable type.
    2. Title and Axis Labels:
    - Ensure the chart has a clear title.
    - Check if the X-axis and Y-axis labels are difficult to read, e.g., are they overlapped?
    3.Tick Labels:
    - Check if the tick labels are difficult to read, e.g., are they overlapped?
    4. Legend Completeness:
    - Is the legend complete?
    [\Instructions]
    - Provide an ordered list of actionable steps (e.g., Step 1, Step 2…). 
    - Each step should be specific and implementable (e.g., "Change the y-axis scale to logarithmic” rather than “Consider using a logarithmic scale").
    - Avoid weak suggestions like "Consider using...", "You might want to...", "If possible," or "It may help to...".
    - Focus on visual clarity, interpretability, and correctness. 
    - Assume the visualization is built using vega-lite declarative charting library.
    - ONLY return the concise and clear steps about how to improve the visualization as structured points without any additional explanation."The steps are: ...".
    [\Instructions]
    """
    
    # interact with LLM
    llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", api_key = openai_key)
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
    Your task is to revise the Vega-Lite specification so that it addresses each point in the critique below, while preserving the original chart’s intent.


    The Vega-Lite specification must follow the following rules:
    [\Rules]
    Rule 1: The "$schema" property should be: "https://vega.github.io/schema/vega-lite/v5.json".
    Rule 2: The "transform" property should be put ahead of the "encoding" property.
    Rule 3: Always include "data" attribute as following in the Vega-Lite output:{{url:"https://raw.githubusercontent.com/zihzi/DATA2Poster/refs/heads/main/data/{table_name}.csv"}}.
    Rule 4: Rule 3: Pay attention to the query description to determine whether you should use "filter" transformation in the "transform" property.
    Rule 5: If you use "aggregate" operation in the "transform" property, the "groupby" property of "aggregate" should be correctly specified.
    Rule 6: Make sure no "sort" operations exist in the "transform" property, you should define the order of axes only in the "encoding" property.
    Rule 7: Make sure no "layer","facet","hconcat", "vconcat", "concat", "repeat" property exist in the Vega-Lite specification.
    Rule 8: Make sure "yellowgreenblue" color scheme is used in the Vega-Lite specification.
    [\Rules]

    Instructions for improving the chart:
    [\Instructions]
    - Improvements for better visualization practices, ensure it addresses each point in the critique.
    - The Vega-Lite specification should be a valid JSON object.
    - ONLY return improved the Vega-Lite specification. DO NOT include any preamble text. Do not include explanations or prose.
    [\Instructions] 

    The example of Vega-Lite specification:
    [\Example]
    The query is:"Show all majors and corresponding number of students by a scatter plot."
    The Vega-Lite specification is:
            {{
                "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                "data": {{"url": "https://raw.githubusercontent.com/zihzi/DATA2Poster/refs/heads/main/data/students.csv"}},
                "transform": [
                    {{"aggregate": [{{"op": "count", "field": "Major", "as": "Number of Students"}}],"groupby": ["Major"]}}
                ],
                "mark": "point",
                "encoding": {{
                    "x": {{"field": "Major", "type": "nominal"}},
                    "y": {{"field": "Number of Students", "type": "quantitative"}}
                }}
            }}
    [\Example]                                       
    """
    prompt_input = PromptTemplate(
                        template="""
                        Here is the code to revise:{code}\n\n    
                        The the critique is as follows:{feedback}\n\n
                        The  sample data table for the Vega-Lite specification is as follows:{sampled_data}\n\n
                        """,
                       input_variables=["code","feedback","sampled_data"],
            )
    # interact with LLM
    llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", api_key = openai_key)
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
    f. Make sure "yellowgreenblue" color scheme is used in the Vega-Lite specification.
    [/Rules for Improvement]

    [/Constraints]
    1.Mark‐specific encodings
        - line, area, trail marks must have both x and y.
        - bar marks must have at least one positional (x or y) and one size/ordinal encoding.
    2. Scale‐type <-> field‐type consistency
        - scale.type == "log"|"pow" requires the field’s type == "quantitative".
        - scale.type == "time"|"utc" requires type == "temporal".
    3. Aggregate functions
        - Allowed aggregates: count, sum, mean, median, min, max, stdev, stdevp, variance, variancep.
    4. Binning
        - bin: true (or object) only on type == "quantitative".
        - Cannot combine bin with continuous scales like "log".
    5. Time units
        - timeUnit only on type == "temporal".
    6. Stacking
        - stack: "zero"|"normalize"|"center" only on bar/area/line and requires a quantitative axis.
    7. Color/Opacity/Size channels
        - Field-based color/opacity/size must be quantitative or ordinal.
        - Literal encodings must use value.
    8. Sort order
        - sort must reference an encoded field of matching type.
    9. Parameter binding.
        - bind on parameters must use a supported input type (legend, scales, or form controls).
    10. Output must be a valid JSON object
    11. ONLY return the full corrected Vega-Lite JSON specification without any additional text
    [/Constraints]
    """
    prompt_input = PromptTemplate(
                        template="""
                        Here is the current Vega-Lite Specification should be checked:{vlspec}\n\n    
                        The  sample data table for the Vega-Lite specification is as follows:{sampled_data}\n\n
                        """,
                       input_variables=["vlspec"],
            )
    # interact with LLM
    llm = ChatOpenAI(model_name="o4-mini-2025-04-16", temperature=1,api_key = openai_key)
    prompt_for_chain = ChatPromptTemplate.from_messages(
                        messages=[SystemMessage(content=prompt),
                                  HumanMessagePromptTemplate.from_template(prompt_input.template)
                                  ]                         
                    )
    chain = prompt_for_chain | llm      
    response = chain.invoke(input= {"vlspec":vlspec,"sampled_data":sampled_data})
    return response.content
