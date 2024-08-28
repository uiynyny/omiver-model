import argparse
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

from embedding_function import get_embedding_function
import os

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
You are a nutritionist expert in metabolomics. You will help clients with their needs after they give some information about them. Then you need to provide suggestions to help them achieve their goals based on the user's given data regarding their diet preferences, race, sexuality, age, etc.. 
{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    # Prepare the DB.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Search the DB.
    # results = db.similarity_search_with_score(query_text, k=5)
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)
    "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"
    model = ChatGroq(
        api_key=os.environ["CHATGROQ_API_KEY"],
        model="llama-3.1-8b-instant",
    )

    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()
