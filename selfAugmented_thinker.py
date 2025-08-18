
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate



def self_augmented_knowledge(openai_key, data_name, data_columns, user_selected_column, base_facts, sub_facts):
    llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", api_key=openai_key)
    intermediate_prompt_template = """
    You are a senior data analyst. You are examining a dataset named **{data_name}** with these columns: **{data_columns}**. 
    Here are key facts extracted from this dataset:{base_facts} and {sub_facts}
    **Your goal:**  
    Generate **3-5** focused, structured insights, using chain-of-thought reasoning.

    **Instructions for your internal reasoning (do NOT include these steps in your response):**   

    1. **Context & Exploration**  
    - Briefly restate what the dataset covers and the role of **{user_selected_column}**.  
    - List 2 hypotheses or questions about how **{user_selected_column}** might vary or relate to other columns.  

    2. **Deep Dive**  
    For each hypothesis or question:  
    - **Analyze** how **{user_selected_column}** changes across values of one other column.  
    - **Highlight** any trends, correlations, anomalies, or group differences.  
    - **Limit** each point to **two** column names (the target plus one comparator).  

    3. **Chain-of-Thought**  
    - Show your reasoning in 2-3 short bullet steps per insight.  
    - Use phrases like “Because…”, “When comparing…”, “We observe that…”.  

    4. **Summarize**  
    - Choose into **3-5** strongest observations.
    - Each insight should be a single sentence, mentioning exactly two columns.  
    - Order by importance or strength of insight.  
    - Do **not** include extra commentary or metadata.  

    **Output requirements (ONLY return this; no reasoning or extra text):**  
    - A numbered list of **3-5** one-sentence insights.  
    - Each sentence mentions **{user_selected_column}** and exactly **one** other column.  
    - No explanations, no bullet steps, no metadata.


    **Example output:**  
    1. **{user_selected_column}** increases by 25% when **OtherColumn** is “X”.  
    2. **{user_selected_column}** is lowest for **OtherColumn** = “Y”.  
    3. …

    """
    prompt = PromptTemplate(
        template=intermediate_prompt_template,
        input_variables=["data_names","data_columns","base_facts","sub_facts","user_selected_column"],
    )
    summarize_chain = prompt | llm
    intermediate_output = summarize_chain.invoke(input ={"data_name":data_name, "data_columns":data_columns,"base_facts":base_facts,"sub_facts":sub_facts,"user_selected_column":user_selected_column})

    return intermediate_output.content
    
    