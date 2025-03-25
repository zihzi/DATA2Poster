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
                        You are writing an introduction for a poster whose title is a question {title} to present the data analysis results.
                        Acoording to the information from {vis_q}, there is three questions which will be represented as viusalization charts.
                        These charts are designed to interpret the main question(i.e. poster title) in different aspects.
                        Think step by step about what is the purpose to explore the main question(i.e. poster title) and write a brief introduction for the poster IN THREE SENTENCES.
                        EACH SENTENCE SHOULD BE SHORT AND CLEAR in 10-15 words.
                        Do not use columnar formulas. Do not use special symbols such as *, `. 
                        LIMIT your response to 100 words.""",
            input_variables=["title", "vis_q"],
        )
        
    llm = ChatOpenAI(model_name='gpt-4o-mini-2024-07-18', api_key = openai_key)
    # structured_llm = llm.with_structured_output(Text)
    introduction_chain = prompt | llm.with_structured_output(introduction_schema)
    response = introduction_chain.invoke(input= {"title":title, "vis_q":vis_q})
    return response["content"]


def conclusion(title,insight, intro, openai_key):
   
    prompt = PromptTemplate(
            template="""
            You are an assistant that helps people to summarize given visualization charts.
            You are writing an conclusion for a poster which is aim to answer the question {title}.
            This is the list containing the insight of three charts {insight} and the introduction of this poster {intro}.
            Refer to the information and cite your rich knowledge to anwser the poster question.
            DO NOT use special symbols such as *, `
            EACH SENTENCE SHOULD BE SHORT AND CLEAR in 10-15 words.
            LIMIT your response to 100 words.""",
            input_variables=["title","insight", "intro"],
        )
        
    llm = ChatOpenAI(model_name='gpt-4o-mini-2024-07-18', api_key = openai_key)
    conclusion_chain = prompt | llm.with_structured_output(conclusion_schema)
    response = conclusion_chain.invoke(input= {'title':title,'insight':insight, 'intro':intro} )
    return response["content"]