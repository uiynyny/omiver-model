from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    ServiceContext,
    load_index_from_storage
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq
import warnings
warnings.filterwarnings('ignore')

import os
os.environ["GROQ_KEY"] = "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"

reader=SimpleDirectoryReader(input_dir="docs")
docs=reader.load_data()
text_splitter=SentenceSplitter(chunk_size=1024,chunk_overlap=200)
nodes = text_splitter.get_nodes_from_documents(docs,show_progress=True)
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = Groq(model="llama-3.1-8b-instant", api_key=os.environ.get("GROQ_KEY"))

service_context=ServiceContext.from_defaults(embed_model=embed_model, llm=llm)

vector_index=VectorStoreIndex.from_documents(documents=docs,node_parser=nodes,service_context=service_context,show_progress=True)
vector_index.storage_context.persist(persist_dir="metabo")
storage_context=StorageContext.from_defaults(persist_dir="metabo")
index=load_index_from_storage(storage_context=storage_context, service_context=service_context)
query_engine = index.as_query_engine(service_context=service_context)
query="The metabolomics of PA literature falls in to which group?"
resp=query_engine.query(query)
print(resp)
