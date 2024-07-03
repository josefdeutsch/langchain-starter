import json
import os
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.agent_toolkits.polygon.toolkit import PolygonToolkit
from langchain_community.utilities.polygon import PolygonAPIWrapper
from langchain_community.tools.polygon.last_quote import PolygonLastQuote
from langchain import hub

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv('LANGCHAIN_TRACING_V2')
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')
os.environ["TAVILY_API_KEY"] = os.getenv('TAVILY_API_KEY')
os.environ["POLYGON_API_KEY"] = os.getenv('POLYGON_API_KEY')

#"gpt-3.5-turbo-1106"
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)


#instructions = """A highly accurate and precise agent, it processes all input data and returns the output data in JSON format. The JSON format must be well-structured, and no duplicates are allowed."""
#base_prompt = hub.pull("langchain-ai/openai-functions-template")
#prompt = base_prompt.partial(instructions=instructions)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "A highly accurate and precise agent, it processes all input data and returns the output data in JSON format. The JSON format must be well-structured, and no duplicates are allowed."),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

polygon = PolygonAPIWrapper()

toolkit = PolygonToolkit.from_polygon_api_wrapper(polygon)

agent = create_openai_tools_agent(llm, toolkit.get_tools(), prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=toolkit.get_tools(),
    verbose=True,
)

input_data = {
    "input": "Get aggregate data for X:BTCUSD ticker with a timespan of 1 day from 2023-01-09 to 2023-02-10.",
}

response = agent_executor.invoke(input=input_data)

