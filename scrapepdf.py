from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

import requests
import os

from dotenv import load_dotenv

load_dotenv()

# given the list of urls, download the pdfs
urls = ["https://arxiv.org/pdf/2410.23661"]


def get_pdfs():
    for url in urls:
        response = requests.get(url, stream=True)
        os.makedirs(dir.__name__ + "/docs", exist_ok=True)
        if response.status_code == 200:
            with open("docs/" + url.split("/")[-1] + ".pdf", "wb") as f:
                f.write(response.content)
                print(f"Downloaded: {url.split('/')[-1]}")


get_pdfs()
reader = SimpleDirectoryReader(input_dir="docs")
docs = reader.load_data()
text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
nodes = text_splitter.get_nodes_from_documents(docs, show_progress=True)
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

Settings.llm = Groq(
    model="llama-3.1-8b-instant", api_key=os.environ.get("GROQ_API_KEY")
)
Settings.node_parser = text_splitter
Settings.embed_model = embed_model


vector_index = VectorStoreIndex.from_documents(
    documents=docs,
    embed_model=embed_model,
    show_progress=True,
)
vector_index.storage_context.persist(persist_dir="metabo")

# Test the index
storage_context = StorageContext.from_defaults(persist_dir="metabo")
index = load_index_from_storage(storage_context=storage_context)
query_engine = index.as_query_engine(llm=Settings.llm)
query = "The metabolomics of PA literature falls in to which group?"
resp = query_engine.query(query)
print(resp)
