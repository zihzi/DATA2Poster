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
    
    # llm = ChatOpenAI(model_name='gpt-4.1-mini-2025-04-14', temperature=0, api_key = openai_key)
    llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            max_tokens=None,
            api_key = openai_key
            # other params...
        )
    intro_prompt = [
                SystemMessage(content="""
                                        You are an excellent data scientist. 
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
    
   


def conclusion(insight, intro, column_1, column_2, entity_1, entity_2, openai_key):

    prompt = PromptTemplate(
            template="""
            You are an assistant that helps people to summarize given visualization charts from a poster.
            Here's the introduction of this poster {intro}.
            The following is the insights of six visualization charts:
            Insights from chart 1.{insight_1}\n\n
            Insights from chart 2.{insight_2}\n\n
            Insights from chart 3.{insight_3}\n\n
            Insights from chart 4.{insight_4}\n\n
            Insights from chart 5.{insight_5}\n\n
            Insights from chart 6.{insight_6}\n\n
            The insights from each chart is summarized in 5 bullet points.
            **Your task:**
            Select **one** the most important bullet point from the insights for each charts and write a concise conclusion for the poster.
            **Instruction (think step by step)**
            1. Refer to the introduction and understand the analysis purpose of this poster.
            2. Read and understand the given insights carefully.
            3. Select **one** the most important bullet point from the insights for each chart.
            4. Use the sentense from the selected bullet points to write a concise conclusion for the poster.
            5. Do not restate or reference all the sentences directly. Instead, quote numbers only when they are essential to strengthen the insight, but avoid overwhelming the audience with raw figures.
            6. Write the conclusion in clear, plain language that is easy for the audience to understand.
            7. EACH SENTENCE SHOULD BE SHORT AND CLEAR in **10-15** words.
            8. DO NOT use special symbols such as *, `

            **Attention**
            1. The insight from chart 1 are facts about the column name {column_1} and the insight from chart 3 are facts about {column_2}. Ensure you add the word of {column_1} and {column_2} in your conclusion.
            2. The insight from chart 5 are facts about {entity_1} and the insight from chart 6 are facts about {entity_2}.
            3. Insight from chart 5 and chart 6 are about the same topic, so you should compare these insights.
            4. After comparing the insights from chart 5 and chart 6, you highlight the key differences and similarities in your conclusion.
            5. Ensure you add the word of {entity_1} and {entity_2} in your conclusion.
               - Example: If entity_1 is "China" and entity_2 is "Japan", you might say: "China has the highest sales in electronics, while Japan has the highest sales in automobiles."

            **Constraints**
            1. Write the conclusion for each charts **in order**.
            2. Make sure insight from each chart is clearly expressed without confuse with insights of other charts.
            3. Use appropriate conjunctions and transitional phrases to ensure the narrative flows smoothly. Connect insights logically so that each sentence leads naturally to the next, creating a coherent and easy-to-follow conclusion.

            **Avoid**
            The following is example of conclusion you should NOT write:
            1."Solomon Islands leads Pacific Island countries with nearly 100,000 employed persons. Male employment is consistently higher than female across all countries. Agriculture and Fishing is the largest employment sector with 35.25%% share. Female employment exceeds male by 3130 in the Education sector. Male employment in Solomon Islands is up to 315.94 times that of female. Male employment in Samoa is 1.6 to over 11 times higher than female across sectors."
            Rationale: This conclusion is too detailed, contains specific numbers, and does not focus on the main insights. It also fails to provide a clear, concise summary of the key findings.
            2."Employment levels differ significantly among Pacific Island countries and between genders. Key economic sectors show concentrated employment with clear gender disparities. Solomon Islands and Tonga exhibit unique employment patterns by sex and sector. Understanding these differences aids targeted employment policies in the region."
            Rationale: This conclusion is too vague, lacks specific insights, and does not directly address the key findings from the charts. It also does not provide a clear summary of the main insights.
            """,
            input_variables=["intro","insight_1","insight_2","insight_3","insight_4","insight_5","insight_6"],
        )
        
    # llm = ChatOpenAI(model_name='gpt-4.1-mini-2025-04-14', temperature=0, api_key = openai_key)
    llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            max_tokens=None,
            api_key = openai_key
            # other params...
        )
    conclusion_chain = prompt | llm.with_structured_output(conclusion_schema)
    response = conclusion_chain.invoke(input= {'intro':intro, 'insight_1':insight[0], 'insight_2':insight[2], 'insight_3':insight[1],'insight_4':insight[3],'insight_5':insight[4],'insight_6':insight[5],'column_1':column_1, 'column_2':column_2, 'entity_1':entity_1,'entity_2':entity_2} )
    return response["content"]

def improve_title(intro,conclusion,openai_key):
    prompt = PromptTemplate(
            template="""
            You are an excellent data scientist writing a data visualization poster title in a question.
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
                    - Any contextual information about magnitude, trends, or comparisons
                    - The implications of the finding

            Step 3: Extract the Core Insight.
                    Distill the conclusion into a single, clear insight statement that captures:
                    - The most important and interesting relationship or pattern
                    - Specific metrics or comparisons when relevant
                    - The significance of the finding

            Step 4: Formulate a Question-Based Title.
                    Create a title in question form that:
                    - Is concise and attention-grabbing in **10-15** words.
                    - Reflects both the main topic (from the introduction) and the key insight (from the conclusion).
                    - Uses simple, clear language that is accessible to a general audience.
                    - Highlights curiosity or surprise to engage the audience.   

            Step 5: Double-check.
                    Before finalizing, review your title to ensure that it:
                    - Falls within the 10-15 word limit.
                    - Clearly combines the dataset's topic and the key insight.
                    - Flows smoothly and reads like a natural question.
                    - Is engaging and sparks curiosity for a general audience.
     
            """,
            input_variables=["intro","conclusion"],
        )
        
    # llm = ChatOpenAI(model_name='gpt-4.1-mini-2025-04-14', temperature=0, api_key = openai_key)
    llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            max_tokens=None,
            api_key = openai_key
            # other params...
        )
    title_chain = prompt | llm.with_structured_output(conclusion_schema)
    response = title_chain.invoke(input= {'intro':intro, 'conclusion':conclusion} )
    return response["content"]