from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import json

Q_json_schema = {
    "title": "Questions_Evaluation",
    "type": "object", 
    "description": "A set of questions for EDA.",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },
        "actions": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },
        "poster_question": {
            "type": "string"
        },
        
    },
    "required": ["questions","actions","poster_question"]
}


def generate_questions(openai_key,poster_question, data_columns, questions,actions):
   
    llm = ChatOpenAI(model_name="gpt-4o-mini-2024-07-18", api_key =openai_key)
    advice_prompt_template = """
    You are an experienced data analyst specializing in Exploratory Data Analysis (EDA). 
    Your task is to critically evaluate whether the provided questions effectively guide the EDA process and contribute to meaningful insights.
    The following questions were used to generate visualization charts:\n\n
    {questions}
    The following are simple and straightforward directive to transform questions into visualization chart:\n\n
    {actions}
    These questions were formulated based on the main research question:\n\n
    {poster_question}
    You may reference the dataset columns when evaluating the questions:\n\n
    {data_columns}
    
    Carefully assess the provided questions based on the following criteria:
    [/Criteria] 
    (1) Uniqueness & Redundancy: Do any questions overlap or repeat without adding new insights?
    (2) Relevance: Are these questions directly connected to the research question?
    (3) EDA Flow & Coherence: Do the questions guide a structured and logical progression of EDA?
    (4) Clarity & Precision: Are the questions well-defined, unambiguous, and concise?
    (5) Visualization: Can these visualization directive transform questions into visualization chart well?
    [/Criteria] 
    
    Your Response Should Include:
    [/Instructions]
    (1) Step-by-step evaluation of the provided questions based on the above criteria.
    (2) Think critically to ensure the questions facilitate insightful, structured, and visualization-friendly EDA.
    (3) Provide constructive advice on how to refine or improve the questions.
    [/Instructions]
    
    """
    advice_prompt = PromptTemplate(
        template=advice_prompt_template,
        input_variables=["questions","poster_question","data_columns"],
    )
    advice_chain = advice_prompt | llm
    advice_for_Q = advice_chain.invoke(input ={"questions":questions,"actions":actions, "poster_question":poster_question, "data_columns":data_columns})
    new_Q_prompt_template = """
    You are an experienced data analyst specializing in Exploratory Data Analysis (EDA).
    Your task is to revise or formulate new questions that effectively guide the EDA process and contribute to meaningful insights.
    The following questions were used to generate visualization charts:\n\n
    {questions}
    The following are simple and straightforward directive to transform questions into visualization chart:\n\n
    {actions}
    These questions were formulated based on the main research question:\n\n
    {poster_question}

    You may reference the dataset columns when refining or rewritng the questions:\n\n
    {data_columns}

    You have received the following advice for improving the given questions:\n\n
    {advice_for_Q}
    
    [/Instructions]
    (1) Carefully review the advice provided on the initial set of questions.
    (2) Think critically to ensure these THREE questions facilitate insightful, structured, and visualization-friendly EDA.
    (3) Revised the given three questions that better align with the research goal based on given advice.
    (4) Revised visualzation directive for corresponding questions.
    (5) Revised the research question, if necessary, to ensure clarity and alignment with the newly formulated questions.
    (6) Your response should consist of THREE revised questions and ONE refined research question.
    [/Instructions]

    
    """
    new_Q_prompt = PromptTemplate(
        template=new_Q_prompt_template,
        input_variables=["questions","poster_question","data_columns","advice_for_Q"],
    )
    new_Q_chain = new_Q_prompt | llm.with_structured_output(Q_json_schema)
    new_Q = new_Q_chain.invoke(input ={"questions":questions, "actions":actions, "poster_question":poster_question, "data_columns":data_columns, "advice_for_Q":advice_for_Q.content})
    
    return advice_for_Q.content,new_Q




