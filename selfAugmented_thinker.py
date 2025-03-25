
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate, HumanMessagePromptTemplate



def self_augmented_knowledge(openai_key,data_name, data_columns, data_facts):
    llm = ChatOpenAI(model_name="gpt-4o-mini-2024-07-18", api_key =openai_key)
    intermediate_prompt_template = """
    You are a senior data analyst. You are analyzing a dataset named:\n\n{data_name} and it has columns:\n\n{data_columns}.

    Here are key facts summarizing essential information from this dataset:\n\n{data_facts}

    Step 1: Identify what should be explored

    Given the dataset name and its context, what are the most relevant aspects to analyze?
    What potential relationships or  distributions deserve further investigation?

    Step 2: Generate deeper insights using Chain-of-Thought reasoning

    Think step by step:
    What are the most important features to consider?
    Do any fact show unusual patterns or correlations?

    Step 3: Summarize key takeaways

    Based on your reasoning, summarize the most relevant insights in a coherent manner that compatible with the key facts.
    Remove duplicate or similar insights.
    Format the response as structured points.
    ONLY return the key takeaways without any additional explanation. "The key takeaways are: ..."
    """
    prompt = PromptTemplate(
        template=intermediate_prompt_template,
        input_variables=["data_names","data_columns","data_facts"],
    )
    summarize_chain = prompt | llm
    intermediate_output = summarize_chain.invoke(input ={"data_name":data_name, "data_columns":data_columns,"data_facts":data_facts})
    
    return intermediate_output.content
    
    