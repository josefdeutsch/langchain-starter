import os
from dotenv import load_dotenv

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

retriever_tool = create_retriever_tool(
    retriever,
    "pdf_search",
    "Search for information about Annual Report. For any questions about Report, you must use this tool!",
)

toolkit = [retriever_tool]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("placeholder", "{chat_history}"),
        ("human", "Question: {question} Use the following transcript to answer the question:{docs}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
# Define your query and document page content
query = "Who is Macquarie?"
docs_page_content = response

agent = create_tool_calling_agent(llm, toolkit, prompt)

agent_executor = AgentExecutor(agent=agent, tools=toolkit, verbose=True)

result = agent_executor.invoke({"question": query, "docs": docs_page_content})
