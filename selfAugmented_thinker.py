
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.prompts import PromptTemplate, HumanMessagePromptTemplate
import json


def self_augmented_knowledge(openai_key,data_name, data_columns, data_facts):
    llm = ChatOpenAI(model_name="gpt-4o-mini-2024-07-18", api_key =openai_key)
    ###########TEST intermediate output############
    intermediate_prompt_template = """
    You are a senior data analyst. You are analyzing a dataset named: {data_name} and it has columns: {data_columns}.

    Here are key facts about this dataset:
    {data_facts}

    Step 1: Identify what should be explored

    Given the dataset name and its context, what are the most relevant aspects to analyze?
    What potential relationships, distributions, or anomalies deserve further investigation?
    Step 2: Generate deeper insights using Chain-of-Thought reasoning

    Think step by step:
    What are the most important features to consider?
    Do any fact show unusual patterns, correlations, or outliers?
    Step 3: Summarize key takeaways

    Based on your reasoning, summarize the most relevant insights that highlight important directions for further analysis.
    Format the response as structured points to guide exploratory data analysis.
    ONLY return the key takeaways without any additional explanation. "The key takeaways are: ..."
    """
    prompt = PromptTemplate(
        template=intermediate_prompt_template,
        input_variables=["data_names","data_columns","data_facts"],
    )
    summarize_chain = prompt | llm
    intermediate_output = summarize_chain.invoke(input ={"data_name":data_name, "data_columns":data_columns,"data_facts":data_facts})
    
    return intermediate_output.content
    
    