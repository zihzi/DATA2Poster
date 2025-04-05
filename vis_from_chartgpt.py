from langchain_community.vectorstores.faiss import FAISS
# from langchain_core.documents import Document
import json
from langchain_openai.embeddings import OpenAIEmbeddings
openai_key = "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"   

def load_json(json_file):
    with open(json_file, "r", encoding="utf-8") as fh:
        return json.load(fh)


data_list = load_json('train_set.json')
docs = [Document(page_content=item["content"],
                 metadata={"source": item["source"]})
        for item in data_list]
docs = [item for item in data_list]
vectorstore = FAISS.from_texts(
                docs,
                OpenAIEmbeddings(model="text-embedding-3-small", api_key = openai_key),
                )



        