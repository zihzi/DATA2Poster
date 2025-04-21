from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate





def expand_questions(openai_key,questions):
    llm = ChatOpenAI(model_name="gpt-4.1-nano-2025-04-14", api_key = openai_key)
    expanded_query_prompt_template = """
    You are an experienced data analyst specializing in Exploratory Data Analysis (EDA).
    Your task is to expand and solidify the query into a step by step detailed instruction (or comment) on how to write python code to fulfill the user query's requirements. 
    Import the appropriate libraries. Pinpoint the correct library functions to call and set each parameter in every function call accordingly.
    The following is the query:\n\n
    {questions}
    You should understand what the query's requirements are, and output step by step, detailed instructions on how to use python code to fulfill these requirements. 
    Include what libraries to import, what library functions to call, how to set the parameters in each function correctly, how to prepare the data, how to manipulate the data so that it becomes appropriate for later functions to call etc,. 
    Make sure the code to be executable and correctly generate the desired output in the user query. 
    """
    expanded_query_prompt = PromptTemplate(
        template=expanded_query_prompt_template,
        input_variables=["questions"],
    )
    expanded_query_chain = expanded_query_prompt | llm
    expanded_query = expanded_query_chain.invoke(input ={"questions":questions})
    
    return expanded_query.content







