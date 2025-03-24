from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate




def introduction(title, topic, openai_key):

    prompt = PromptTemplate(
            template="""You are an excellent data scientist. 
                        You are an expert in the domain of given dataset.
                        You are writing an introduction for a poster whose title is a question {title} to present the data analysis results.
                        Acoording to the information from {topic}, there is a main question and three objects which will be represented as viusalization charts.
                        These charts are designed to interpret the main question(i.e. title) in different aspects.
                        Think step by step about how well are these charts doing and write a brief introduction for the poster in three sentences.
                        Please do not use columnar formulas. Do not use special symbols such as *, `. BE CONCISE.""",
            input_variables=["title", "topic"]
        )
        
    llm = ChatOpenAI(model_name='gpt-4o-mini', api_key = openai_key)
    gpt4_image_chain = prompt | llm 
    response = gpt4_image_chain.invoke(input= {"title":title, "topic":topic})
    return response.content

def description(my_json_list, openai_key):

    prompt = PromptTemplate(
            template="""
            You are an AI assistant that helps people understand unfamiliar visualizations. 
            You can assume the user struggles to develop the most important insight. 
            You can make the most out of your factual knowledge to interpret the chart.
            And HIGHTLIGHT the most important insights for the chart in one sentence. 
            YOUR INSIGHT SHOULD BE CURTKY and don't need to mention the real number in data.
            This is the the chart in vega-lite format {my_json_list}.
            DO NOT use special symbols such as *, `""",
            input_variables=["my_json_list"]
        )
        
    llm = ChatOpenAI(model_name='gpt-4o-mini', api_key = openai_key)
    gpt4_image_chain = prompt | llm 
    response = gpt4_image_chain.invoke(input= {'my_json_list':my_json_list})
    return response.content

def conclusion(final_distribution, summary, openai_key):
   
    prompt = PromptTemplate(
            template="""
            You are an AI assistant that helps people to summarize given visualization charts.
            This is the list contain the insight of three charts {final_distribution} and the original data summary {summary}.
            Refer to the insight of each chart and the data, and cite your rich knowledge to conclude what user can learn from these charts in two sentenses.
            DO NOT use special symbols such as *, `""",
            input_variables=["final_distribution", "summary"]
        )
        
    llm = ChatOpenAI(model_name='gpt-4o-mini', api_key = openai_key)
    gpt4_image_chain = prompt | llm 
    response = gpt4_image_chain.invoke(input= {'final_distribution':final_distribution, 'summary':summary} )
    
    return response.content