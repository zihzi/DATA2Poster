
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI



def self_augmented_knowledge(openai_key, data_name, main_column, base_facts):
    llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", api_key=openai_key)
    # llm = ChatGoogleGenerativeAI(
    #         model="gemini-2.5-flash",
    #         temperature=0,
    #         max_tokens=None,
    #         api_key = openai_key
    #         # other params...
    #     )
    intermediate_prompt_template = """
    You are a senior data analyst. You are examining a dataset named **{data_name}** with this column: **{main_column}**.
    Here are key facts extracted from this dataset:{base_facts}
    **Your goal:**  
    Generate **5** focused, structured insights, using chain-of-thought reasoning.

    **Instructions for your internal reasoning (do NOT include these steps in your response):**   

    1. **Context & Exploration**  
    - Briefly restate what the dataset covers and the role of **{main_column}**.  
    - List 2 hypotheses or questions about how **{main_column}** might vary or relate to other columns.  

    2. **Deep Dive**  
    For each hypothesis or question:  
    - **Analyze** how **{main_column}** changes across values of one other column.  
    - **Highlight** any trends, group commonality, or group differences.  


    3. **Chain-of-Thought**  
    - Build your reasoning in 2-3 short bullet steps per insight.  
    - Use phrases like “Because…”, “When comparing…”, “We observe that…”.  

    4. **Summarize**  
    - Choose key facts that are interesting observations and weave into an insight.
    - Each insight should be **one single** sentence.  
    - Order by importance or strength of insight.  
    - Do **not** include extra commentary or metadata.  

    **Constraints:**
    - Understand the values contained in {main_column}. NEVER use values that do not appear in the key facts.

    **Output requirements (ONLY return this; no reasoning or extra text):**  
    - A five-point list of one-sentence insights.  
    - Each sentence mentions **{main_column}** and exactly **one** other column.
    - No explanations, no bullet steps, no metadata.


    **Example output:**  
    1. Japan has the highest Sales when **OtherColumn** is “X”.  
    2. The Sales in China are significantly lower than in the US.
    3. Sales in Germany are higher than in France when **OtherColumn** is “Y”.  
    4. …

    """
    prompt = PromptTemplate(
        template=intermediate_prompt_template,
        input_variables=["data_names","main_column","base_facts"],
    )
    summarize_chain = prompt | llm
    intermediate_output = summarize_chain.invoke(input ={"data_name":data_name, "main_column":main_column,"base_facts":base_facts})

    return intermediate_output.content
    
    