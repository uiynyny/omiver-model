import streamlit as st

from langchain.prompts import ChatPromptTemplate
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

import os

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
You are a nutritionist expert in metabolomics. You will help clients with their needs after they give some information about them. Then you need to provide suggestions to help them achieve their goals based on the user's given data regarding their diet preferences, race, sexuality, age, etc.. 
{context}

---

Answer the question based on the above context: {question}
"""


def get_embedding_function():
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": False}
    hf = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    return hf


def query_rag(query_text: str, vector_store):
    # Prepare the DB.
    # vector_store = PineconeVectorStore(index_name="metabodb", embedding=get_embedding_function())

    # Search the DB.
    # results = db.similarity_search_with_score(query_text, k=5)
    results = vector_store.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = ChatGroq(api_key=os.getenv("GROQ_KEY"), model="llama-3.1-8b-instant")

    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    #print(formatted_response)
    return response_text


st.title("Omiver chatbot v2")

# Prepare the DB.
vector_store = PineconeVectorStore(
    index_name="metabodb",
    embedding=get_embedding_function(),
    pinecone_api_key=os.getenv("PINECONE_API_KEY"),
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        resp = query_rag(st.session_state.messages[-1]["content"], vector_store)
        response = st.markdown(resp.content)
    st.session_state.messages.append({"role": "assistant", "content": resp.content})
