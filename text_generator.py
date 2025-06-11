from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


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
                            "maxLength": 100
                        }
                    },
                    "required": ["content"],
                }

def introduction(title, vis_q, openai_key):

    prompt = PromptTemplate(
            template="""You are an excellent data scientist. 
                        You are writing an introduction for a poster {title} to present the data analysis results.
                        Acoording to the information from {vis_q}, there is three questions which will be represented as viusalization charts.
                        These charts are designed to interpret the main question(i.e. poster title) in different aspects.
                        Think step by step about what is the purpose to explore the main question(i.e. poster title) and write a brief introduction for the poster IN THREE SENTENCES.
                        EACH SENTENCE SHOULD BE SHORT AND CLEAR in 10-15 words.
                        Do not use columnar formulas. Do not use special symbols such as *, `. 
                        LIMIT your response to 100 words.""",
            input_variables=["title", "vis_q"],
        )
        
    llm = ChatOpenAI(model_name='gpt-4.1-mini-2025-04-14', api_key = openai_key)
    # structured_llm = llm.with_structured_output(Text)
    introduction_chain = prompt | llm.with_structured_output(introduction_schema)
    response = introduction_chain.invoke(input= {"title":title, "vis_q":vis_q})
    return response["content"]


def conclusion(title,insight, intro, openai_key):
   
    prompt = PromptTemplate(
            template="""
            You are an assistant that helps people to summarize given visualization charts.
            You are writing an conclusion for a poster {title}.
            The following is the insight of three charts:
            1.{insight_1}\n\n
            2.{insight_2}\n\n
            3.{insight_3}\n\n
            Here's the introduction of this poster {intro}.
            First, refer to the introduction and understand the purpose of this poster
            Second, think carefully how to add the given three insight in sequence in the conclusion smoothly.
            Finally, cite your rich knowledge and write conclusion to anwser the poster question.
            DO NOT use special symbols such as *, `
            EACH SENTENCE SHOULD BE SHORT AND CLEAR in 10-15 words.
            LIMIT your response to 100 words.""",
            input_variables=["title","insight", "intro"],
        )
        
    llm = ChatOpenAI(model_name='gpt-4.1-mini-2025-04-14', api_key = openai_key)
    conclusion_chain = prompt | llm.with_structured_output(conclusion_schema)
    response = conclusion_chain.invoke(input= {'title':title,'insight_1':insight[0], 'insight_2':insight[1], 'insight_3':insight[2], 'intro':intro} )
    return response["content"]

def improve_title(conclusion,openai_key):
    prompt = PromptTemplate(
            template="""
            You are an excellent data scientist writing a data visualization poster title in a question.
            You will be given a conclusion drawn from data visualization charts. 
            Your task is to:Generate an engaging poster title in the form of a question whose answer is the extracted insight.
            [\Instructions]
            Step 1: Analyze the Visualization Conclusion.
                    Carefully read the provided conclusion and identify:
                    - The main finding or pattern
                    - The variables or relationships being described
                    - Any contextual information about magnitude, trends, or comparisons
                    - The implications of the finding

            Step 2: Extract the Core Insight.
                    Distill the conclusion into a single, clear insight statement that captures:
                    - The most important and interesting relationship or pattern
                    - Specific metrics or comparisons when relevant
                    - The significance of the finding

            Step 3: Formulate a Question-Based Title.
                    Create a title in question form that:
                    - Is concise and attention-grabbing (ideally 5-12 words)
                    - Directly leads to the insight as its answer
                    - Creates curiosity or tension that the visualization resolves
            [\Instructions]

            [\Examples]
            Given Conclusion:\n\n
            "Among all renewable energy sources, solar power installation costs decreased most dramatically, falling 82%% between 2010 and 2022, while wind power costs fell by only 39%% in the same period."
            
            The Core Insight before generating the title:\n\n
            "Solar power installation costs decreased most dramatically."
            
            Finally generate  the Question-Based Poster Title:\n\n
            "Is Solar Power Winning the Renewable Cost Reduction Race?"
            [\Examples]
            Here is conclusion for writing a poster title :\n\n{conclusion}
            """,
            input_variables=["conclusion"],
        )
        
    llm = ChatOpenAI(model_name='gpt-4.1-mini-2025-04-14', api_key = openai_key)
    title_chain = prompt | llm.with_structured_output(conclusion_schema)
    response = title_chain.invoke(input= {'conclusion':conclusion} )
    return response["content"]