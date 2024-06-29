import os
from pprint import pprint
from dotenv import load_dotenv

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langchain.agents import AgentExecutor

from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_tool_calling_agent


load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv('LANGCHAIN_TRACING_V2')
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')
os.environ["TAVILY_API_KEY"] = os.getenv('TAVILY_API_KEY')


if not os.getenv('USER_AGENT'):
    os.environ['USER_AGENT'] = 'MyLangChainApp/1.0 (macOS; Python 3.9) LangChain/0.8'

search = TavilySearchResults()
search.invoke("what is the weather in SF")

loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)
vector = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vector.as_retriever()
retriever.invoke("how to upload a dataset")[0] #nur das Relevanteste!

retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Search for information about Annual Report. For any questions about Report, you must use this tool!",
)

tools = [search, retriever_tool]

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")


agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": "hi!"})

agent_executor.invoke({"input": "how can langsmith help with testing?"})

agent_executor.invoke({"input": "whats the weather in sf?"})

