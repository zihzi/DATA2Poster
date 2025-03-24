from langchain_openai import  OpenAIEmbeddings
from langchain.vectorstores import Chroma


def search_by_rag(fact, query, openai_key):
    # Create a vector store from facts
    vectorstore = Chroma.from_texts(
    fact,
    OpenAIEmbeddings(model="text-embedding-3-small", api_key = openai_key),
    )
    retrieve_docs = vectorstore.max_marginal_relevance_search(query, k=1,fetch_k=25,lambda_mult=0.4)
    retrieve_fact = [doc.page_content for doc in retrieve_docs] # retrieve_fact is a list of facts
    
    return retrieve_fact

