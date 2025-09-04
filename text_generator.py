from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

introduction_schema ={
                    "title": "text",
                    "description": "Introduction Description",
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Introduction Description",
                            "maxLength": 80
                        }
                    },
                    "required": ["content"],
                }
conclusion_schema ={
                    "title": "text",
                    "description": "Conclusion Description",
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Conclusion Description",
                            "maxLength": 80
                        }
                    },
                    "required": ["content"],
                }

def introduction(vis, header, openai_key):
    
    llm = ChatOpenAI(model_name='gpt-4.1-mini-2025-04-14', temperature=0, api_key = openai_key)
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
                                        You are writing an introduction for a 3-section poster to present the data analysis results.
                                        You will be given 3 section headers and their visualization charts.
                                        Each section contains two charts.
                                        Read header and the visualization charts carefully.
                                        **Instruction (think step by step)**
                                        Write a concise **poster introduction** that:
                                        1. Understand what the section header is about.
                                        2. Understand what the visualization charts are about.
                                        3. States the poster's purpose concisely in plain language.
                                        4. Use **ONLY 4** SENTENCES. EACH SENTENCE SHOULD BE SHORT AND CLEAR in **10-15** words.
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
    section_des =  llm.invoke(intro_prompt)
    return section_des.content
        
    # prompt = PromptTemplate(
    #         template="""You are an excellent data scientist. 
    #                     You are writing an introduction for a poster to present the data analysis results.
    #                     Acoording to the information from {vis_q}, there are two or three questions which will be represented as viusalization charts.
    #                     These charts are designed to interpret the main question(i.e. poster title) in different aspects.
    #                     Think step by step about what is the purpose to explore the main question(i.e. poster title) and write a brief introduction for the poster IN THREE SENTENCES.
    #                     EACH SENTENCE SHOULD BE SHORT AND CLEAR in 10-15 words.
    #                     Do not use columnar formulas. Do not use special symbols such as *, `. 
    #                     LIMIT your response to 100 words.""",
    #         input_variables=["title", "vis_q"],
    #     )
        
    
    # structured_llm = llm.with_structured_output(Text)
    # introduction_chain = prompt | llm.with_structured_output(introduction_schema)
    # response = introduction_chain.invoke(input= {"title":title, "vis_q":vis_q})
    
   


def conclusion(chart_pattern, insight, header, entity_1, entity_2, openai_key):

    prompt = PromptTemplate(
            template="""
            You are an excellent data analyst.
            You are summarizing visualization charts from a 3-section poster.
            You will be given 3 section headers and each section is represented by two charts.
            Here's the header of this poster "{header1}", "{header2}", "{header3}".

            The insights is supportive evidence for writing conclusions.
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

            **Your task:**
            Write a concise and insightful conclusion for the poster.

            **Instruction (think step by step)**
            1. Refer to the section header and understand the analysis purpose of each section.
            2. Read and understand the given insights carefully.
            3. Read and understand the given chart descriptions carefully.
            4. Write a revealing conclusion for the poster.
            5. The conclusion should include:
               - neutral explanations (objective and descriptive, you may quote the statistical number).
               - speculative opinions (possible reason, implication, or why it matters).
            6. Keep the tone insightful, not descriptive only.
            7. The conclusion should not be conservative or vague. Instead, apply your knowledge to produce insights that give the reader real takeaways. The conclusion should synthesize the patterns revealed by the analysis, highlight meaningful disparities or anomalies, and connect them to broader implications or future actions. Aim for clarity, depth, and value, so the reader gains new understanding rather than just a summary.
            8. The conclusion SHOULD **NO MORE THAN 85 words**.
            9. DO NOT use special symbols such as *, `


            **Constraints**
            1. Use appropriate conjunctions and transitional phrases to ensure the narrative for connecting each section flows smoothly. Connect sentences logically so that each sentence leads naturally to the next, creating a coherent and easy-to-follow conclusion.
            2. The chart descriptions may not be correct, so use the sentences from chart descriptions **only when you found it CAN BE SUPPORTED by the insights**
            3. The description for chart 5 MUST mention the entities: {entity_1} and the description for chart 6 MUST mention the entities: {entity_2}.


            **Example of Good Conclusion**
            Employment patterns across Pacific Island countries reveal a heavy reliance on agriculture and fishing, coupled with persistent gender imbalances in several sectors. These findings highlight the structural dependence on primary industries and the need for targeted strategies to address unequal opportunities across genders and sectors.
            **Avoid**
            The following is example of conclusion you should NOT write:
            1."Solomon Islands leads Pacific Island countries with nearly 100,000 employed persons. Male employment is consistently higher than female across all countries. Agriculture and Fishing is the largest employment sector with 35.25%% share. Female employment exceeds male by 3130 in the Education sector. Male employment in Solomon Islands is up to 315.94 times that of female. Male employment in Samoa is 1.6 to over 11 times higher than female across sectors."
            Rationale: This conclusion is too detailed, contains specific numbers, and does not focus on the main insights. It also fails to provide a clear, concise summary of the key findings.
            2."Employment levels differ significantly among Pacific Island countries and between genders. Key economic sectors show concentrated employment with clear gender disparities. Solomon Islands and Tonga exhibit unique employment patterns by sex and sector. Understanding these differences aids targeted employment policies in the region."
            Rationale: This conclusion is too vague, lacks specific insights, and does not directly address the key findings from the charts. It also does not provide a clear summary of the main insights.
            """,
            input_variables=["header1","header2","header3","insight_1","insight_2","insight_3","insight_4","insight_5","insight_6","chart_pattern1","chart_pattern2","chart_pattern3","chart_pattern4","chart_pattern5","chart_pattern6","entity_1","entity_2"],
        )
        
    llm = ChatOpenAI(model_name='gpt-4.1-mini-2025-04-14', temperature=0, api_key = openai_key)
    # llm = ChatGoogleGenerativeAI(
    #         model="gemini-2.5-flash",
    #         temperature=0,
    #         max_tokens=None,
    #         api_key = openai_key
    #         # other params...
    #     )
    conclusion_chain = prompt | llm.with_structured_output(conclusion_schema)
    response = conclusion_chain.invoke(input= {'header1':header[0], 'header2':header[1],'header3':header[2], 'insight_1':insight[0], 'insight_2':insight[2], 'insight_3':insight[1],'insight_4':insight[3],'insight_5':insight[4],'insight_6':insight[5],'chart_pattern1':chart_pattern[0],'chart_pattern2':chart_pattern[1],'chart_pattern3':chart_pattern[2],'chart_pattern4':chart_pattern[3],'chart_pattern5':chart_pattern[4],'chart_pattern6':chart_pattern[5],'entity_1':entity_1,'entity_2':entity_2} )
    return response["content"]

def improve_title(intro,conclusion,openai_key):
    prompt = PromptTemplate(
            template="""
            You are an excellent data analyst writing a data visualization poster title in a question.
            Here is the introduction of this poster:\n\n{intro}\n\n
            Here is the conclusion drawn from data visualization charts:\n\n{conclusion}\n\n
            **Task**
            Generate an engaging poster title in the form of a question.

            **Instructions (think step by step)**
            Step 1: Analyze the Introduction.
                    Read the Introduction carefully:
                    - Identify the main topic of the dataset.
                    - Understand the context and purpose of the poster.

            Step 2: Analyze the Visualization Conclusion.
                    Carefully read the provided conclusion and identify:
                    - The main finding or pattern
                    - The variables or relationships being described
                   

            Step 3: Extract the Core Insight.
                    Distill the conclusion into a single, clear insight statement that captures:
                    - The most important and interesting relationship or pattern

            Step 4: Formulate a Question-Based Title.
                    Create a title in question form that:
                    - Is concise and attention-grabbing in **10-15** words.
                    - Reflects the key insight (from the conclusion).
                    - Uses simple, clear language that is accessible to a general audience.
                    

            Step 5: Double-check.
                    Before finalizing, review your title to ensure that it:
                    - Falls within the 10-15 word limit.
                    - Flows smoothly and reads like a natural question.
                    - Can be answered by the key insight from the conclusion.
                  
     
            """,
            input_variables=["intro","conclusion"],
        )
        
    llm = ChatOpenAI(model_name='gpt-4.1-mini-2025-04-14', temperature=0, api_key = openai_key)
    # llm = ChatGoogleGenerativeAI(
    #         model="gemini-2.5-flash",
    #         temperature=0,
    #         max_tokens=None,
    #         api_key = openai_key
    #         # other params...
    #     )
    title_chain = prompt | llm.with_structured_output(conclusion_schema)
    response = title_chain.invoke(input= {'intro':intro, 'conclusion':conclusion} )
    return response["content"]