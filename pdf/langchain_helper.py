import os
from pprint import pprint
from dotenv import load_dotenv
import asyncio
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
def load_pdf_from_disk(path:str):
    loader = PyPDFLoader(path)
    pages = loader.load_and_split()
    return pages

def get_database(pages):
    db = FAISS.from_documents(pages, OpenAIEmbeddings())
    return db

def get_retriever_from_database(db):
    retriever = db.as_retriever()
    response = retriever.invoke("how to upload a dataset")[0]
    return response

def get_language_model():
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    return llm

def get_vector_store_info(db):
    vectorstore_info = VectorStoreInfo(
        name="annual_report",
        description="a banking annual report as a pdf",
        vectorstore=db
    )
    return vectorstore_info

def get_vector_store_tools(llm, vectorstore_info):
    vector_tool = VectorStoreToolkit(vectorstore_info=vectorstore_info,llm=llm)
    return vector_tool


def get_vector_store_agent(llm, vector_tool):
    PREFIX = """You are an agent designed to answer questions about sets of documents.
    You have access to tools for interacting with the documents, and the inputs to the tools are questions. Write long detail text.
    """
    agent_executor = create_vectorstore_agent(
    llm=llm, toolkit=vector_tool, prefix=PREFIX, verbose=True)
        
    return agent_executor


#prints = await db.asimilarity_search_with_score(response)

