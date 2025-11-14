from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A3, landscape
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from json_sanitizer import parse_jsonish





def agent6_vis_generator(table_name, query, title, data_schema,sampled_data, vlspec, openai_key):
    prompt = """
    You are a data analysis assistant that generates Vega-Lite specification.
    Your task is to create **six** optimal Vega-Lite specification for the six query {query} based on the CSV file named '{table_name}'.
    The title for the six charts should be the following titles in order:{title}
    The data content overview is as follows:{data_schema}
    The sample data is as follows:{sampled_data}
    Here is the Vega-Lite specification suitable for these queries for your reference:{vlspec}

    Create Vega-Lite specification for each query that obey the following rules:
    [\Rules]
    Rule 1: The "$schema" property should be: "https://vega.github.io/schema/vega-lite/v5.json".
    Rule 2: The "transform" property should be put ahead of the "encoding" property.
    Rule 3: ALWAYS include "data" property as following in the Vega-Lite output:{{url:"https://raw.githubusercontent.com/zihzi/DATA2Poster/refs/heads/main/data/{table_name}.csv"}}.
    Rule 4: Pay attention to the query description to determine whether you should use "filter" transformation in the "transform" property.
    Rule 5: If you use "aggregate" operation in the "transform" property, the "groupby" property of "aggregate" should be correctly specified.
    Rule 6: If you use "window" operation to rank data in the "transform" property, make sure **NO "groupby" property** exists after "window" operation in the "transform" property.
    Rule 7: Make sure no "sort" operations exist in the "transform" property, you should define the order of axes only in the "encoding" property.
    Rule 8: Make sure no "false" and "true" is used in the Vega-Lite specification.
    Rule 9: Make sure no "aggregate: None" is used in the Vega-Lite specification.
    Rule 10: If "mark": {{ "type": "arc"}} is used in the Vega-Lite specification, make sure **no "legend: null"** operation in the encoding property.
    Rule 11: If "mark": {{ "type": "arc"}} is used in the Vega-Lite specification, make sure **"color": {{"field": "...","type": "nominal","legend": {{"title": "..."}},"scale": {{"scheme": "..."}}}}** operation in the encoding property.
    [\Rules]

    Instructions for generating the Vega-Lite specification:
    [\Instructions]
    - Read the data content overview and the sample data table carefully to understand the data type and valid value for each column.
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
     - Make sure the colors are harmonious.
    [\Visual-design contract]

    [\Constraints]
    - Using a grouped bar chart to compare different value if one column. For example, compare male vs. female employment across economic sectors.
    - If "share" or "proportion" is mentioned in the title, ALWAYS use **pie** chart.
    - Always limit **pie categories to 5 slices total**. Keep the top 4 categories by proportion (descending), and **group all remaining categories into a single slice labeled "Others". 
    - Always add the **text label of percentage** to the **pie** chart.
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
    llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", api_key = openai_key)
    prompt_for_chain = ChatPromptTemplate.from_messages(
                        messages=[SystemMessagePromptTemplate.from_template(prompt_input.template)
                        ]
                    )
    chain = prompt_for_chain | llm      
    response = chain.invoke(input= {"table_name":table_name, "query":query, "title":title,"data_schema":data_schema,"sampled_data":sampled_data,"vlspec":vlspec})
    return response.content

def agent6_sec_vis_generator(table_name, query, title, data_schema,sampled_data, vlspec, openai_key):
    prompt = """
    You are a data analysis assistant that generates Vega-Lite specification.
    Your task is to create optimal Vega-Lite specification for this query {query} based on based on the CSV file named '{table_name}'.
    The title for the chart should be the following title:{title}
    The data content overview is as follows:{data_schema}
    The sample data is as follows:{sampled_data}
    Here is the Vega-Lite specification suitable for these queries for your reference:{vlspec}

    Create Vega-Lite specifications for each query that obey the following rules:
    [\Rules]
    Rule 1: The "$schema" property should be: "https://vega.github.io/schema/vega-lite/v5.json".
    Rule 2: The "transform" property should be put ahead of the "encoding" property.
    Rule 3: ALWAYS include "data" property as following in the Vega-Lite output:{{url:"https://raw.githubusercontent.com/zihzi/DATA2Poster/refs/heads/main/data/{table_name}.csv"}}.
    Rule 4: Pay attention to the query description to determine whether you should use "filter" transformation in the "transform" property.
    Rule 5: If you use "aggregate" operation in the "transform" property, the "groupby" property of "aggregate" should be correctly specified.
    Rule 6: If you use "window" operation to rank data in the "transform" property, make sure **NO "groupby" property** exists after "window" operation in the "transform" property.
    Rule 7: Make sure no "sort" operations exist in the "transform" property, you should define the order of axes only in the "encoding" property.
    Rule 8: Make sure no "false" and "true" is used in the Vega-Lite specification.
    Rule 9: Make sure no "aggregate: None" is used in the Vega-Lite specification.
    Rule 10: If "mark": {{ "type": "arc" }} is used in the Vega-Lite specification, make sure **no "legend: null"** operation in the encoding property.
    Rule 11: If "mark": {{ "type": "arc"}} is used in the Vega-Lite specification, make sure **"color": {{"field": "...","type": "nominal","legend": {{"title": "..."}},"scale": {{"scheme": "..."}}}}** operation in the encoding property.
    [\Rules]

    Instructions for generating the Vega-Lite specification:
    [\Instructions]
    - Read the data content overview and the sample data table carefully to understand the data type and valid value for each column.
    - The Vega-Lite specification should be a valid JSON object.
    - ONLY return the Vega-Lite specification. DO NOT include any preamble text. Do not include explanations or prose.\n\n. 
    [\Instructions]

    Visual-design contract that apply to the spec:
    [\Visual-design contract]
    1. Tick labels & counts
     - Always use -30° rotation for x-axis categorical labels.
     - Max six ticks per axis.
    2. Colour encoding
     - Make sure the colors are harmonious.
    [\Visual-design contract]

    [\Constraints]
    - Using a grouped bar chart to compare different value if one column. For example, compare male vs. female employment across economic sectors.
    - If "share" or "proportion" is mentioned in the title, ALWAYS use **pie** chart.
    - Always limit **pie categories to 5 slices total**. Keep the top 4 categories by proportion (descending), and **group all remaining categories into a single slice labeled "Others". Ensure the legend lists exactly those 5 labels (Top-4 + "Others").
    - Always add the **text label of percentage** to the **pie** chart.
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
            "<Vega-Lite JSON specification>",
            }}
        ]
        }}"""                      
    prompt_input = PromptTemplate(
                        template=prompt,
                        input_variables=["table_name", "query", "title","sampled_data", "vlspec"],
            )
    # interact with LLM
    llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", api_key = openai_key)
    prompt_for_chain = ChatPromptTemplate.from_messages(
                        messages=[SystemMessagePromptTemplate.from_template(prompt_input.template)
                        ]
                    )
    chain = prompt_for_chain | llm      
    response = chain.invoke(input= {"table_name":table_name, "query":query, "title":title,"data_schema":data_schema,"sampled_data":sampled_data,"vlspec":vlspec})
    return response.content

def agent6_vis_refiner(vlspec, query,img_url, openai_key):
    llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", api_key=openai_key)
    prompt = """
    You are a data analysis assistant that generates Vega-Lite specification to refine existing plots.
    Given a complete vega-lite specification, a query, and an image of the current plot, 
    please determine whether the plot has faithfully followed the query. 
    Your task is to provide complete vega-lite specification to enhance the plot.
    
    [\Instructions]
    1. Carefully read and analyze the user query to understand the specific requirements. 
    2. Examine the provided vega-lite specification to understand how the current plot is generated. 
    3. Check if the code aligns with query in terms of data selection, plot type, and any specific customization. 
    4. Look at the provided image of the plot. 
    5. Assess the plot type, the data it represents, labels, titles, colors, and any other visual elements. 
    6. Think about improvements for better visualization practices, such as clarity, readability, and aesthetics.
    7. Provide complete vega-lite specification after making the necessary adjustments.
    [\Instructions]

    Create Vega-Lite specifications for each query that obey the following rules:
    [\Rules]
    Rule 1: The "$schema" property should be: "https://vega.github.io/schema/vega-lite/v5.json".
    Rule 2: The "transform" property should be put ahead of the "encoding" property.
    Rule 3: ALWAYS include "data" property as following in the Vega-Lite output:{{url:"https://raw.githubusercontent.com/zihzi/DATA2Poster/refs/heads/main/data/{table_name}.csv"}}.
    Rule 4: Pay attention to the query description to determine whether you should use "filter" transformation in the "transform" property.
    Rule 5: If you use "aggregate" operation in the "transform" property, the "groupby" property of "aggregate" should be correctly specified.
    Rule 6: If you use "window" operation to rank data in the "transform" property, make sure **NO "groupby" property** exists after "window" operation in the "transform" property.
    Rule 7: Make sure no "sort" operations exist in the "transform" property, you should define the order of axes only in the "encoding" property.
    Rule 8: Make sure no "false" and "true" is used in the Vega-Lite specification.
    Rule 9: Make sure no "aggregate: None" is used in the Vega-Lite specification.
    Rule 10: If "mark": {{ "type": "arc" }} is used in the Vega-Lite specification, make sure **no "legend: null"** operation in the encoding property.
    Rule 11: If "mark": {{ "type": "arc"}} is used in the Vega-Lite specification, make sure **"color": {{"field": "...","type": "nominal","legend": {{"title": "..."}},"scale": {{"scheme": "..."}}}}** operation in the encoding property.
    [\Rules]

    Visual-design contract that apply to the spec:
    [\Visual-design contract]
    1. Tick labels & counts
     - Always use -30° rotation for x-axis categorical labels.
     - Max six ticks per axis.
    2. Colour encoding
     - DO NOT change the original color. Just keep what it is.
    [\Visual-design contract]

    [\Constraints]
    - Using a grouped bar chart to compare different value if one column. For example, compare male vs. female employment across economic sectors.
    - If "share" or "proportion" is mentioned in the title, ALWAYS use **pie** chart.
    - Always limit **pie categories to 5 slices total**. Keep the top 4 categories by proportion (descending), and **group all remaining categories into a single slice labeled "Others". Ensure the legend lists exactly those 5 labels (Top-4 + "Others").
    - Always add the **text label of percentage** to the **pie** chart.
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
            "<Vega-Lite JSON specification>",
            }}
        ]
        }}
    """              
    chart_refine_prompt = [
    SystemMessage(content=prompt),
    HumanMessage(content=[
        {
            "type": "text", 
            "text": f"This chart is ploted  based on this question:\n\n {query}.\n\n"
        },
        {
                "type": "text", 
                "text": f"Here is the vega-lite specification of the chart:\n\n{vlspec}\n\n"
        },
        {
                "type": "image_url",
                "image_url": {
                    "url": img_url
                },
        },
    ])
    ]
    response =  llm.invoke(chart_refine_prompt)
    return response.content

def agent6_vis_corrector(error_vlspec, error, openai_key):
    prompt = """
            GIVEN Vega-Lite Specification:\n
            {error_vlspec}\n
            There are some errors in the pecification above:\n
            {error}\n
            Please correct the errors.
            Give the complete Vega-Lite Specification in JSON format and don't omit anything else.
            
            **Output (exact JSON)**  
            Do not INCLUDE ```json```.Do not add other sentences after this json data.
            Return **only** the final JSON:
                {{
                
                    "<Vega-Lite JSON specification>",
                }}
    """                      
    prompt_input = PromptTemplate(
                        template=prompt,
                        input_variables=["error_vlspec", "error"],
            )
    # interact with LLM
    llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", api_key = openai_key)
    prompt_for_chain = ChatPromptTemplate.from_messages(
                        messages=[SystemMessagePromptTemplate.from_template(prompt_input.template)
                        ]
                    )
    chain = prompt_for_chain | llm      
    response = chain.invoke(input= {"error_vlspec":error_vlspec, "error":error})
    return response.content

def agent7_vis_describer(chart_query, img_url, openai_key):
    llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", api_key=openai_key)
    chart_prompt_template = """
    You are a data-analysis assistant excel at annotating visualization charts.  

    **Task**
    Your task is to write a **short insight (3-5 sentences)** about the most surprising pattern in the chart, and include:
    - neutral explanations (objective and descriptive).
    - speculative opinions (possible reason, implication, or why it matters).  

    **Instructions**
    1. Pick the most notable trend, contrast, or anomaly.  
    2. Express it clearly without mechanically repeating chart labels.  
    3. Vary the style: sometimes write an objective explanation, sometimes add a speculative reason/opinion.  
    4. Keep the output concise within **3-5** sentences.  
    5. The final insight should feel natural and insightful, not repetitive.  

    **Output**
    Write **3-5 sentences** with a description containing both neutral explanations and speculative opinions.

    """
    chart_des_prompt = [
        SystemMessage(content=chart_prompt_template),
        HumanMessage(content=[
            {
                "type": "text", 
                "text": f"This chart is ploted  based on this question:\n\n {chart_query}.\n\n"
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
    return chart_des.content

def agent8_dt_extractor(trans_json, openai_key):
    llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", api_key=openai_key)
    data_transform_prompt = """
        You are a Python coding expert.
        You are given the following Vega-Lite transformation specification:{trans_json}
        **Task**
        Write the equivalent Python code using Pandas to perform the same transformation on a DataFrame named df. 
        The output should be a new DataFrame after transformed.
        Only return the Python code without any explanation or additional text.
        NEVER include ```python``` in your response.
        Only modify the code in the middle of the following code snippet:        
            def trans_data(df):\n\n
            # Assuming "df" is the input DataFrame\n\n
            # Grouping and aggregating\n\n
            result_df =         ####only modify this line#####
            \n\n
            return result_df
        """
    data_transform_prompt_template = PromptTemplate(
                    template=data_transform_prompt,
                    input_variables = {"trans_json" })
    data_transform_chain = data_transform_prompt_template | llm
    data_transform_result = data_transform_chain.invoke(input={"trans_json":trans_json})
    return data_transform_result.content

def agent9_section_designer(img_list, insight_list, openai_key):
    llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", api_key=openai_key)
    section_prompt ="""
                            You are a data analysis expert at capturing interesting insights from visualization charts.
                            You are building a 3-section poster.Each section contains 2 charts.
                            You are given only the chart titles listed below:{img_list}
                            You are also given the insights from each chart in oreder listed below:
                            Insight for Chart 1: {insight_1}
                            Insight for Chart 2: {insight_2}
                            Insight for Chart 3: {insight_3}
                            Insight for Chart 4: {insight_4}
                            Insight for Chart 5: {insight_5}
                            Insight for Chart 6: {insight_6}
                            
                            ** Your Tasks(Think step by step)**
                            1. Read and evaluate each chart title for subject matter, variables.  
                            2. Read and evaluate each insight for key findings, themes.
                            3. For each chart, consider what insight it provides.
                            4. Use appropriate conjunctions to connect and synthesize the insights from both charts.
                            5. Write a single sentence (maximum 15 words) to introduce the section.
                            6. Create a concise section heading that captures the shared insight of each section.
                            
                            **Output (JSON)**
                            Do not INCLUDE ```json```.Do not add other sentences after this json data.
                            Return **only** the final JSON in this structure:
                            {{
                            "sections": [
                                {{
                                "section": "A",
                                "heading": "<section heading>",                       
                                "charts_title": ["<chart title 1>", "<chart title 2>"],
                                "insight": "<one-sentence synthesis>"
                                }},
                                {{
                                "section": "B",
                                "heading": "<section heading>",                                       
                                "charts_title": ["<chart title 3>", "<chart title 4>"],
                                "insight": "<one-sentence synthesis>"
                                }},
                                {{
                                "section": "C",
                                "heading": "<section heading>",
                                "charts_title": ["<chart title 5>", "<chart title 6>"],
                                "insight":  "<one-sentence synthesis>"
                                }}
                            ]
                            }}"""
                                    
    section_prompt_template = PromptTemplate(
                    template=section_prompt,
                    input_variables=["img_list","insight_1","insight_2","insight_3","insight_4","insight_5","insight_6"]
                )
    section_chain = section_prompt_template | llm
    section_result = section_chain.invoke(input = {"img_list":img_list,"insight_1":insight_list[0],"insight_2":insight_list[1],"insight_3":insight_list[2],"insight_4":insight_list[3],"insight_5":insight_list[4],"insight_6":insight_list[5]})
    return section_result.content

def agent10_pdf_creator(data_name, chart_pattern, insight_list, section_insight_list, chart_id_list, chart_url_list, entity_1, entity_2, section_header_list, openai_key):
    filename = f"{data_name}_summary.pdf"
    filedir = "pdf"
    filepath = os.path.join(filedir, filename)
    background_image = f"figure/poster_background_v3.png"
    # Create canvas
    c = canvas.Canvas(filepath, pagesize=landscape(A3))
    width, height = landscape(A3)
    c.drawImage(background_image, 0, 0, width=width, height=height)

    # Generate introduction
    text_introduction = introduction(chart_url_list , section_header_list,openai_key)

    text_conclusion = ""
    # Generate conclusion 
    text_conclusion_content = conclusion(chart_pattern, insight_list , section_header_list,  entity_1, entity_2, openai_key)
    text_conclusion_json = parse_jsonish(text_conclusion_content)
    text_conclusion = text_conclusion_json["conclusion"][0]["content"] + "  " + text_conclusion_json["conclusion"][1]["content"] + "  " + text_conclusion_json["conclusion"][2]["content"]

    # Generate Title from conclusion
    title = poster_title(text_introduction, text_conclusion, openai_key)
    
    # Title
    p_title = Paragraph(title, ParagraphStyle(name='title', fontSize=30, fontName='Helvetica-Bold',leading=22, textColor="#2c2a32"))
    p_title.wrapOn(c, width,20)
    p_title.drawOn(c, 10, height-50)
    
    # Introduction content
    p_in = Paragraph(text_introduction, ParagraphStyle(name="introduction", fontSize=14, fontName='Helvetica', leading=14, alignment=4, textColor="#2c2a32"))
    p_in.wrapOn(c, width-600, 60)
    p_in.drawOn(c, 20, height-170)
    
     
    # Add section 1 descriptions

    p_desc = Paragraph(section_insight_list[0], ParagraphStyle(name="insight", fontSize=16, fontName='Helvetica-Bold',leading=14, alignment=4, textColor="#2c2a32"))
    p_desc.wrapOn(c, 550, 25)
    p_desc.drawOn(c, 50, height-221)

    # Add section 2 descriptions 
    p_desc = Paragraph(section_insight_list[1], ParagraphStyle(name="insight", fontSize=16, fontName='Helvetica-Bold',leading=14, alignment=4, textColor="#2c2a32"))
    p_desc.wrapOn(c, 550, 25)
    p_desc.drawOn(c, 50, height-540)
    # Add section 3 descriptions 
    p_desc = Paragraph(section_insight_list[2], ParagraphStyle(name="insight", fontSize=16, fontName='Helvetica-Bold',leading=14, alignment=4, textColor="#2c2a32"))
    p_desc.wrapOn(c, 500, 20)
    p_desc.drawOn(c, 685, height-92)
    # Add section 1 images
    c.drawImage(f"data2poster_chart/image{chart_id_list[0]}.png", 20, height-505, width=280, height=280)
    c.drawImage(f"data2poster_chart/image{chart_id_list[1]}.png", 330, height-505, width=290, height=280)

    # Add section 2 images
    c.drawImage(f"data2poster_chart/image{chart_id_list[2]}.png", 20, height-815, width=280, height=260)
    c.drawImage(f"data2poster_chart/image{chart_id_list[3]}.png", 330, height-815, width=290, height=260)

    # Add section 3 images
    c.drawImage(f"data2poster_chart/image{chart_id_list[4]}.png", 670, height-340, width=500, height=230)
    c.drawImage(f"data2poster_chart/image{chart_id_list[5]}.png", 670, height-600, width=500, height=230)

    # Conclusion content
    p_con = Paragraph(text_conclusion, ParagraphStyle(name="conclusion", fontSize=14, fontName='Helvetica', leading=14, alignment=4, textColor="#2c2a32"))
    p_con.wrapOn(c, 520, 140)
    p_con.drawOn(c, 653, height-817)

    c.save()
    return text_conclusion_json, title

def agent10_sec_pdf_creator(data_name, section_insight_list, chart_id_list, chart_url_list,text_conclusion, section_header_list, openai_key):

    filename = f"{data_name}_summary_2.pdf"
    filedir = "pdf"
    filepath = os.path.join(filedir, filename)
    background_image = f"figure/poster_background_v3.png"
    # Create canvas
    c = canvas.Canvas(filepath, pagesize=landscape(A3))
    width, height = landscape(A3)
    c.drawImage(background_image, 0, 0, width=width, height=height)

    # Generate introduction
    text_introduction = introduction(chart_url_list , section_header_list,openai_key)

    # Generate Title from conclusion
    title = poster_title(text_introduction, text_conclusion, openai_key)
    
    # Title
    p_title = Paragraph(title, ParagraphStyle(name='title', fontSize=30, fontName='Helvetica-Bold',leading=22, textColor="#2c2a32"))
    p_title.wrapOn(c, width,20)
    p_title.drawOn(c, 10, height-50)
    
    # Introduction content
    p_in = Paragraph(text_introduction, ParagraphStyle(name="introduction", fontSize=14, fontName='Helvetica', leading=14, alignment=4, textColor="#2c2a32"))
    p_in.wrapOn(c, width-600, 60)
    p_in.drawOn(c, 20, height-170)
    
     
    # Add section 1 descriptions

    p_desc = Paragraph(section_insight_list[0], ParagraphStyle(name="insight", fontSize=16, fontName='Helvetica-Bold',leading=14, alignment=4, textColor="#2c2a32"))
    p_desc.wrapOn(c, 550, 25)
    p_desc.drawOn(c, 50, height-221)

    # Add section 2 descriptions 
    p_desc = Paragraph(section_insight_list[1], ParagraphStyle(name="insight", fontSize=16, fontName='Helvetica-Bold',leading=14, alignment=4, textColor="#2c2a32"))
    p_desc.wrapOn(c, 550, 25)
    p_desc.drawOn(c, 50, height-540)
    # Add section 3 descriptions 
    p_desc = Paragraph(section_insight_list[2], ParagraphStyle(name="insight", fontSize=16, fontName='Helvetica-Bold',leading=14, alignment=4, textColor="#2c2a32"))
    p_desc.wrapOn(c, 500, 20)
    p_desc.drawOn(c, 685, height-92)
    # Add section 1 images
    c.drawImage(f"data2poster_chart/image{chart_id_list[0]}.png", 20, height-505, width=280, height=280)
    c.drawImage(f"data2poster_chart/image{chart_id_list[1]}.png", 330, height-505, width=290, height=280)

    # Add section 2 images
    c.drawImage(f"data2poster_chart/image{chart_id_list[2]}.png", 20, height-815, width=280, height=260)
    c.drawImage(f"data2poster_chart/image{chart_id_list[3]}.png", 330, height-815, width=290, height=260)

    # Add section 3 images
    c.drawImage(f"data2poster_chart/image{chart_id_list[4]}.png", 670, height-340, width=500, height=230)
    c.drawImage(f"data2poster_chart/image{chart_id_list[5]}.png", 670, height-600, width=500, height=230)

    # # Conclusion content
    # p_conclusion = Paragraph("Conclusion", ParagraphStyle(name='conclusion', fontSize=26, fontName='Helvetica-Bold', textColor=text_color))
    # p_conclusion.wrapOn(c, width/2, 200)
    # p_conclusion.drawOn(c, 30, height-725)
    p_con = Paragraph(text_conclusion, ParagraphStyle(name="conclusion", fontSize=14, fontName='Helvetica', leading=14, alignment=4, textColor="#2c2a32"))
    p_con.wrapOn(c, 520, 140)
    p_con.drawOn(c, 653, height-817)

    c.save()

# Generate text for poster(title, introduction, conclusion)
def introduction(vis, header, openai_key): 
    llm = ChatOpenAI(model_name='gpt-4.1-mini-2025-04-14', api_key = openai_key)
    # llm = ChatGoogleGenerativeAI(
    #         model="gemini-2.5-flash",
    #         temperature=0,
    #         max_tokens=None,
    #         api_key = openai_key
    #         # other params...
    #     )
    intro_prompt = [
                SystemMessage(content="""
                                        You are an excellent data analyst.
                                        You will be given 3 section headers and corresponding visualization charts.
                                        Each section contains two charts.
                              
                                        **Task**
                                        Writing an introduction for a 3-section poster to present the data analysis results.
                                        
                                        **Instruction (think step by step)**
                                        1. Read header and the visualization charts carefully.
                                        2. Understand what the section header is about.
                                        3. Understand what the visualization charts are about.
                                        4. States the poster's purpose concisely in plain language.
                                        5. Use **ONLY 4** SENTENCES. EACH SENTENCE SHOULD BE SHORT AND CLEAR in **10-15** words.
                              
                                        **Constraints**
                                        Do not use columnar formulas. Do not use special symbols such as *, `. 
                        """),              

                                HumanMessage(content=[
                                    {
                                            "type": "text",
                                            "text": " **Section 1** Here is the section header: " + header[0] + ". The following are the visualization chart in this section: " 
                                            
                                    },
                                    
                                    {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": vis[0]
                                            },
                                    },
                                    
                                    {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": vis[1]
                                            },
                                    },
                                    {
                                            "type": "text",
                                            "text": " **Section 2** Here is the section header: " + header[1] + ". The following are the visualization chart in this section: " 
                                        
                                    },
                                    
                                    {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": vis[2]
                                            },
                                    },
                                    
                                    {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": vis[3]
                                            },
                                    },
                                    {
                                            "type": "text",
                                            "text": " **Section 3** Here is the section header: " + header[2] + ". The following are the visualization chart in this section: " 
                                            
                                    },
                                    
                                    {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": vis[4]
                                            },
                                    },
                                   
                                    {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": vis[5]
                                            },
                                    }
                                    
                                ])
                                ] 
    response =  llm.invoke(intro_prompt)
    return response.content

def conclusion(chart_pattern, insight, header, entity_1, entity_2, openai_key):
    prompt = PromptTemplate(
            template="""
            You are an excellent data analyst excel at summarizing visualization charts from a 3-section poster.
            You will be given 3 section headers and each section contains two charts.
            Here's the header of this poster "{header1}", "{header2}", "{header3}".

            The insight is supportive evidence for writing conclusions.
            The description of each chart is a concise summary of what the chart is about in 3-5 sentences.

            The following is the insights and the description of each chart:
            Section 1:\n\n
            Insights from chart 1.{insight_1}\n\n Description of chart 1: {chart_pattern1}\n\n
            Insights from chart 2.{insight_2}\n\n Description of chart 2: {chart_pattern2}\n\n
            Section 2:\n\n
            Insights from chart 3.{insight_3}\n\n Description of chart 3: {chart_pattern3}\n\n
            Insights from chart 4.{insight_4}\n\n Description of chart 4: {chart_pattern4}\n\n
            Section 3:\n\n
            Insights from chart 5.{insight_5}\n\n Description of chart 5: {chart_pattern5}\n\n
            Insights from chart 6.{insight_6}\n\n Description of chart 6: {chart_pattern6}\n\n

            **Task:**
            Write a concise and insightful conclusion for the poster.

            **Instruction (think step by step)**
            1. Refer to the section header and understand the analysis purpose of each section.
            2. Read and understand the given insights carefully.
            3. Read and understand the given chart descriptions carefully.
            4. Write a revealing conclusion content for each section in the poster.
            5. The conclusion content for each section should include:
               - neutral explanations (objective and descriptive, you may quote the statistical number).
               - speculative opinions (possible reason, implication, or why it matters).
            6. Keep the tone insightful, not descriptive only.
            7. The conclusion should not be vague. Instead, apply your knowledge to produce insights that give the reader real takeaways. 
            8. The conclusion should synthesize the patterns revealed by the analysis from each section, highlight meaningful disparities or anomalies.
            
            **Constraints**
            1. Each sentence should be **short** and clear **within 10-15 words**.
            2. Conclusion content for each section should be **no more than 3** sentences.
            3. Use appropriate conjunctions and transitional phrases to ensure the narrative for connecting each section flows smoothly. 
            4. The chart description may not be correct, so use the sentences from chart descriptions **only when you found it CAN BE SUPPORTED by the insight**
            5. The description for chart 5 MUST mention the entity: {entity_1} and the description for chart 6 MUST mention the entity: {entity_2}.
            6. Do not use columnar formulas. Do not use special symbols such as *, `.
 
            **Output (JSON)**
            Do not INCLUDE ```json```.Do not add other sentences after this json data.
            Return **only** the final JSON in this structure:
            {{
            "conclusion": [
                {{
                "section": "A",
                "content": "<conclusion content for section A>",
                "charts_id": ["0", "1"],
                }},
                {{
                "section": "B",
                "content": "<conclusion content for section B>",
                "charts_id": ["2", "3"],
                }},
                {{
                "section": "C",
                "content": "<conclusion content for section C>",
                "charts_id": ["4", "5"],
                }}
            ]
            }}

            **Example of Output**
            {{
            "conclusion": [
                {{
                "section": "A",
                "content": "Renewable output is unevenly concentrated, with Norway and Sweden contributing the largest totals,which is consistent with hydropower resource centralization and interconnect-driven exports.",
                "charts_id": ["0", "1"],
                }},
                {{
                "section": "B",
                "content": "By source, hydro dominates regionally; wind is disproportionately large in Denmark and coastal Sweden, while solar remains marginal, it suggest signaling partial, uneven diversification.",
                "charts_id": ["2", "3"],
                }},
                {{
                "section": "C",
                "content": "Pronounced seasonal disparities persist: hydro surges during spring thaw, wind intensifies in winter, and solar peaks in summer; variability is most extreme in Norway (hydro) and Denmark (wind), with Iceland showing the flattest seasonal profile.",
                "charts_id": ["4", "5"],
                }}
            ]
            }}
            """,
            input_variables=["header1","header2","header3","insight_1","insight_2","insight_3","insight_4","insight_5","insight_6","chart_pattern1","chart_pattern2","chart_pattern3","chart_pattern4","chart_pattern5","chart_pattern6","entity_1","entity_2"],
        )      
    llm = ChatOpenAI(model_name='gpt-4.1-mini-2025-04-14', api_key = openai_key)
    # llm = ChatGoogleGenerativeAI(
    #         model="gemini-2.5-flash",
    #         temperature=0,
    #         max_tokens=None,
    #         api_key = openai_key
    #         # other params...
    #     )
    conclusion_chain = prompt | llm
    response = conclusion_chain.invoke(input= {'header1':header[0], 'header2':header[1],'header3':header[2], 'insight_1':insight[0], 'insight_2':insight[1], 'insight_3':insight[2],'insight_4':insight[3],'insight_5':insight[4],'insight_6':insight[5],'chart_pattern1':chart_pattern[0],'chart_pattern2':chart_pattern[1],'chart_pattern3':chart_pattern[2],'chart_pattern4':chart_pattern[3],'chart_pattern5':chart_pattern[4],'chart_pattern6':chart_pattern[5],'entity_1':entity_1,'entity_2':entity_2} )
    return response.content

def poster_title(intro,conclusion,openai_key):
    title_schema ={
                    "title": "text",
                    "description": "Poster Title",
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Poster Title",
                            "maxLength": 12
                        }
                    },
                    "required": ["content"],
                }
    prompt = PromptTemplate(
            template="""
            You are an excellent data analyst excel at writing title for a data visualization poster.
            Here is the introduction of this poster:\n\n{intro}\n\n
            Here is the conclusion drawn from visualization charts:\n\n{conclusion}\n\n

            **Task**
            Generate a simple poster title in the form of a question.

            **Instructions (think step by step)**
            Step 1: Analyze the Introduction.
                    Read the Introduction carefully:
                    - Identify the main topic of the dataset.
                    - Understand the context and purpose of the poster.

            Step 2: Analyze the Conclusion.
                    Carefully read the provided conclusion and identify:
                    - The main finding or pattern
                    - The variables or relationships being described
                   
            Step 3: Extract the Core Insight.
                    Distill the conclusion into a single, abstract insight statement that captures:
                    - The most important and interesting relationship or pattern

            Step 4: Formulate a Question-Based Title.
                    Create a title in question form that:
                    - Is concise in **10-15** words.
                    - Reflects the extracted key insight.
                    - Uses simple, clear language that is accessible to a general audience.
                    
            Step 5: Double-check.
                    Before finalizing, review your title to ensure that it:
                    - Falls within the 10-15 word limit.
                    - Flows smoothly and reads like a natural question.
                    - Can be answered by the extracted key insight from the conclusion.
            """,
            input_variables=["intro","conclusion"],
        )
        
    llm = ChatOpenAI(model_name='gpt-4.1-mini-2025-04-14', api_key = openai_key)
    # llm = ChatGoogleGenerativeAI(
    #         model="gemini-2.5-flash",
    #         temperature=0,
    #         max_tokens=None,
    #         api_key = openai_key
    #         # other params...
    #     )
    title_chain = prompt | llm.with_structured_output(title_schema)
    response = title_chain.invoke(input= {'intro':intro, 'conclusion':conclusion} )
    return response["content"]