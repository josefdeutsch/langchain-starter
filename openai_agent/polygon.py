import os
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.agent_toolkits.polygon.toolkit import PolygonToolkit
from langchain_community.utilities.polygon import PolygonAPIWrapper
from langchain_community.tools.polygon.last_quote import PolygonLastQuote
from langchain import hub
#https://python.langchain.com/v0.2/docs/integrations/tools/polygon/
#https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/utilities/polygon.py
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv('LANGCHAIN_TRACING_V2')
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')
os.environ["TAVILY_API_KEY"] = os.getenv('TAVILY_API_KEY')
os.environ["POLYGON_API_KEY"] = os.getenv('POLYGON_API_KEY')

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

instructions = """You are an assistant."""
base_prompt = hub.pull("langchain-ai/openai-functions-template")
prompt = base_prompt.partial(instructions=instructions)

polygon = PolygonAPIWrapper()

toolkit = PolygonToolkit.from_polygon_api_wrapper(polygon)

agent = create_openai_functions_agent(llm, toolkit.get_tools(), prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=toolkit.get_tools(),
    verbose=True,
)

# Adjust the date range to a recent period within the last 2 years
agent_executor.invoke({"input": "What is the latest stock price for AAPL?"})

