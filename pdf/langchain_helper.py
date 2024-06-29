import os
from pprint import pprint
from dotenv import load_dotenv
import asyncio
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
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_tool_calling_agent
load_dotenv()


async def main():
    # Set environment variables
    os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
    os.environ["LANGCHAIN_TRACING_V2"] = os.getenv('LANGCHAIN_TRACING_V2')
    os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')

    if not os.getenv('USER_AGENT'):
        os.environ['USER_AGENT'] = 'MyLangChainApp/1.0 (macOS; Python 3.9) LangChain/0.8'

    # Load and split PDF
    loader = PyPDFLoader("/Users/Joseph/Documents/langchain-starter/annualreport.pdf")
    pages = loader.load_and_split()

    # Create FAISS database from documents
    db = FAISS.from_documents(pages, OpenAIEmbeddings())
    retriever = db.as_retriever()

    # Test retriever
    response = retriever.invoke("how to upload a dataset")[0] # nur das Relevanteste!

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

    vectorstore_info = VectorStoreInfo(
        name="annual_report",
        description="a banking annual report as a pdf",
        vectorstore=db
    )
    # Convert the document store into a langchain toolkit
    vector_tool = VectorStoreToolkit(vectorstore_info=vectorstore_info,llm=llm)
    #toolkit = [retriever_tool,vector_tool]


    # Define your query and document page content
    query = "provide data about Macquarie and list all data in great detail"
    docs_page_content = response

    # Create the agent executor
    PREFIX = """You are an agent designed to answer questions about sets of documents.
    You have access to tools for interacting with the documents, and the inputs to the tools are questions. Write long detail text.
    """

    agent_executor = create_vectorstore_agent(
        llm=llm, toolkit=vector_tool, prefix=PREFIX, verbose=True)

    response = agent_executor.run(query)

    #prints = await db.asimilarity_search_with_score(response)

    #pprint(prints[0][0].page_content)

asyncio.run(main())
