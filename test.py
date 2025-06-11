# import pandas as pd
# # import json
# import altair as alt
# # import altair_saver
# df = pd.read_csv("data/2024USA_presidential_election.csv")
# df['DEM_PERCENT'] = df['DEM_PERCENT'].str.replace('%', '').astype(float)
# df['REP_PERCENT'] = df['REP_PERCENT'].str.replace('%', '').astype(float)
# df['OTH_PERCENT'] = df['OTH_PERCENT'].str.replace(',', '').astype(float)
# df.to_csv("data/2024USA_presidential_election.csv", index=False)


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="o4-mini", use_responses_api=True)

llm_with_tools = llm.bind_tools(
    [
        {
            "type": "code_interpreter",
            # Create a new container
            "container": {"type": "auto"},
        }
    ]
)   
response = llm_with_tools.invoke(
    "Write and run code to answer the question: what is 3^3?"
)
print(response)

