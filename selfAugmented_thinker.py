
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate



def self_augmented_knowledge(openai_key,data_name, data_columns, user_selected_column, data_facts):
    llm = ChatOpenAI(model_name="gpt-4.1-nano-2025-04-14", api_key =openai_key)
    intermediate_prompt_template = """
    You are a senior data analyst. You are analyzing a dataset named:\n\n{data_name} and it has columns:\n\n{data_columns}.

    Here are key facts summarizing essential information from this dataset:\n\n{data_facts}
    Your task is to generate insightful takeaways focus on the selected column: '{user_selected_column}'.
    Think step by step:

    Step 1: Understand the dataset and its context
    - What is the dataset about?
    - What are the key columns and their types?
    - What are the key facts summarizing essential information from this dataset?

    Step 2: Identify what should be explored
    - What are the most relevant aspects to analyze?
    - What potential relationships or distributions deserve further investigation?

    Step 3: Generate deeper insights using Chain-of-Thought reasoning

    - Begin by analyzing how the selected column varies across different values of other columns.
    - Examine any trends, correlations, or anomalies associated with this column.
    - Consider temporal, categorical, or numerical relationships depending on column types (e.g., time-based trends, category distributions, or value ranges).
    - Identify patterns or insights that suggest causality, segmentation, or comparisons across subgroups.
    - Think in terms of common analysis tasks such as:
    “What drives higher or lower values of this column?”
    “How does this column behave across different groups?”
    “Are there outliers or exceptions?”
    “Which features are most related to this column's variation?”

    Step 4: Summarize insightful  takeaways
    - Based on your reasoning, summarize the most relevant insights in a coherent manner that compatible with the key facts.
    - Remove duplicate or similar insights.
    - Format the response as structured points.
    - ONLY USE TWO COLUMN NAMES in one point.
    - ONLY return the key takeaways without any additional explanation."The key takeaways are: ..."
    """
    prompt = PromptTemplate(
        template=intermediate_prompt_template,
        input_variables=["data_names","data_columns","data_facts","user_selected_column"],
    )
    summarize_chain = prompt | llm
    intermediate_output = summarize_chain.invoke(input ={"data_name":data_name, "data_columns":data_columns,"user_selected_column":user_selected_column,"data_facts":data_facts})
    
    return intermediate_output.content
    
    