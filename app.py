import os
import tempfile

import chromadb# the db
import ollama
import streamlit as st
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile



system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""

def process_document(uploaded_file: UploadedFile) -> list[Document]:
    temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
    temp_file.write(uploaded_file.read())

    loader = PyMuPDFLoader(temp_file.name)
    docs = loader.load()
    os.unlink(temp_file.name)  # Delete file after use

    print(f"Loaded Docs: {docs}")  # Debugging step

    if not docs:
        raise ValueError("No documents were loaded from the PDF. Check file format or content.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )

    splits = text_splitter.split_documents(docs)
    print(f"Split Documents: {splits}")  # Debugging step

    if not splits:
        raise ValueError("Document splitting failed. Ensure the PDF has extractable text.")

    return splits



def get_vector_collection() -> chromadb.Collection:#before ingesting any file convert to vector embeddings ollama
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )

    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
    return chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"Collection: {collection}")  # Debugging step
    return collection

def add_to_vector_collection(all_splits: list[Document], file_name: str):
    collection = get_vector_collection()
    documents, metadatas, ids = [], [], []

    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")

    # Debugging - Print or Log the lists
    print(f"Documents: {documents}")
    print(f"Metadatas: {metadatas}")
    print(f"IDs: {ids}")

    # Check if lists are empty before calling upsert
    if not documents or not metadatas or not ids:
        raise ValueError("One or more required lists (documents, metadatas, ids) are empty. Check data processing.")

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
    st.success("Data added to the vector store!")


def query_collection(prompt: str, n_results: int = 10):
    collection = get_vector_collection()
    results = collection.query(query_texts=[prompt], n_results=n_results)

    print(f"Query Results: {results}")  # Debugging step

    if not results or "documents" not in results or not results["documents"]:
        raise ValueError("No documents found in the query results")

    return results


def call_llm(context: str, prompt: str):
    print(f"Sending to LLM - Context: {context[:500]}...")  # Show first 500 chars
    response = ollama.chat(
        model="llama3.2:3b",
        stream=True,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:{context}\n\nQuestion:{prompt}"},
        ],
    )

    collected_response = ""  # Store response to check output
    for chunk in response:
        if chunk["done"] is False:
            collected_response += chunk["message"]["content"]
            yield chunk["message"]["content"]
        else:
            break

    print(f"LLM Response: {collected_response[:500]}...")  # Print first 500 chars



def re_rank_cross_encoders(documents: list[str]) -> tuple[str, list[int]]:
    if not documents or not isinstance(documents, list):
        raise ValueError("Documents must be a non-empty list")

    relevant_text = ""
    relevant_text_ids = []

    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Ensure documents is a list of strings
    if not all(isinstance(doc, str) for doc in documents):
        raise ValueError("Documents must be a list of strings")

    ranks = encoder_model.rank(prompt, documents, top_k=3)

    for rank in ranks:
        corpus_id = rank["corpus_id"]
        if 0 <= corpus_id < len(documents):
            relevant_text += documents[corpus_id] + "\n\n"
            relevant_text_ids.append(corpus_id)

    return relevant_text, relevant_text_ids


if __name__ == "__main__":
     with st.sidebar:
        st.set_page_config(page_title="RAG Question Answer")
        st.header("RAG Question Answer")
        uploaded_file =st.file_uploader(
            "🗎**Upload PDF files for QnA**", type=["pdf"], accept_multiple_files=False
        )

        process = st.button(
            "Process⚡",
        )

        if uploaded_file and process:
            normalize_uploaded_file_name = uploaded_file.name.translate(
                str.maketrans({"-": "_", ".": "_", " ": "_"})
            )
            all_splits = process_document(uploaded_file)
            add_to_vector_collection(all_splits, normalize_uploaded_file_name)

     st.header("RAG Question Answer")
     prompt = st.text_area("** Ask a question related to your document:**")
     ask = st.button(
         "Ask🔥",
     )

     if ask and prompt:
         results = query_collection(prompt)
         context = results.get("documents")[0]
         relevant_text, relevant_text_ids = re_rank_cross_encoders(context)
         response = call_llm(context=relevant_text, prompt=prompt)
         st.write(results)

         with st.expander("See retrieved documents"):
            st.write(results)

         with st.expander("See most relevant document ids"):
            st.write(relevant_text_ids)
            st.write(relevant_text)

