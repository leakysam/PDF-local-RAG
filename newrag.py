import logging
import os
import warnings
from typing import Any

import ollama
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

warnings.filterwarnings('ignore', category=UserWarning, message='.*torch.classes.*')
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
PERSIST_DIRECTORY = os.path.join("data", "vectors")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def create_vector_db_from_file(file_path: str) -> Chroma:
    logger.info(f"Creating vector DB from: {file_path}")
    loader = UnstructuredPDFLoader(file_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_name=f"pdf_{hash(file_path)}"
    )
    return vector_db

def process_question(question: str, vector_db: Chroma) -> str:
    logger.info(f"Processing question: {question} using model: tinyllama")
    llm = ChatOllama(model="tinyllama")
    query_prompt = PromptTemplate(
        input_variables=["question"],
        template="""You are a helpful assistant. Use ONLY the context provided below to answer the question.
        If the answer is not in the context, reply: "The information is not available in the document."
        
        Format your answer clearly using:
        - Bullet points for lists
        - Paragraphs for detailed insights
        
        Context:
        {context}
        
        Question: {question}"""

    )
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=query_prompt
    )
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant. Answer the question using ONLY the context provided below.
        If the answer is not in the context, reply: "The information is not available in the document."

        Context:
        {context}

        Question: {question}"""
    )
    chain = (
        {"context": retriever, "question": RunnablePassthrough()} |
        prompt |
        llm |
        StrOutputParser()
    )
    return chain.invoke(question)
