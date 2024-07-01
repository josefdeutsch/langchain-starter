import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)
from langchain_community.document_loaders import TextLoader
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_tool_calling_agent

load_dotenv()

# Set environment variables
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv('LANGCHAIN_TRACING_V2')
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')

if not os.getenv('USER_AGENT'):
    os.environ['USER_AGENT'] = 'MyLangChainApp/1.0 (macOS; Python 3.9) LangChain/0.8'


# Load and split PDF
def load_pdf_from_disk(path: str):
    loader = PyPDFLoader(path)
    pages = loader.load_and_split()
    return pages

@st.cache_resource
def get_database(_pages):
    db = FAISS.from_documents(_pages, OpenAIEmbeddings())
    return db

@st.cache_data
def get_retriever_from_database(_db):
    retriever = _db.as_retriever()
    response = retriever.invoke("how to upload a dataset")[0]
    return response

@st.cache_resource
def get_language_model():
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    return llm

@st.cache_resource
def get_vector_store_info(_db):
    vectorstore_info = VectorStoreInfo(
        name="annual_report",
        description="a banking annual report as a pdf",
        vectorstore=_db
    )
    return vectorstore_info

@st.cache_resource
def get_vector_store_tools(_llm, _vectorstore_info):
    vector_tool = VectorStoreToolkit(vectorstore_info=_vectorstore_info, llm=_llm)
    return vector_tool

@st.cache_resource
def get_vector_store_agent(_llm, _vector_tool):
    PREFIX = """You are an agent designed to answer questions about sets of documents.
    You have access to tools for interacting with the documents, and the inputs to the tools are questions. Write long detail text.
    """
    agent_executor = create_vectorstore_agent(
        llm=_llm, toolkit=_vector_tool, prefix=PREFIX, verbose=True)
    return agent_executor
