import os
import numpy as np
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from numpy import cumsum, log, polyfit, sqrt, std, subtract
from numpy.random import randn

from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.agent_toolkits.polygon.toolkit import PolygonToolkit
from langchain_community.utilities.polygon import PolygonAPIWrapper

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv('LANGCHAIN_TRACING_V2')
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')
os.environ["TAVILY_API_KEY"] = os.getenv('TAVILY_API_KEY')
os.environ["POLYGON_API_KEY"] = os.getenv('POLYGON_API_KEY')


#Introduction to the Hurst exponent — with code in Python
#https://medium.com/@jpolec_72972/the-rolling-hurst-exponent-in-python-19b7b908e251



@tool
def calculate_hurst_exponent(ts):
    """
    Calculate the Hurst exponent of a time series.


    Parameters:
    ts (list): A list representing the time series data.

    Returns:
    float: The Hurst exponent of the time series. Returns NaN if the time series is too short 
           or if the computation fails.
    """
    ts = np.array(ts)  # Ensure ts is a NumPy array
    lags = range(2, 100)
    tau = []

    for lag in lags:
        if len(ts) <= lag:
            break

        lagged_diff = ts[lag:] - ts[:-lag]
        if len(lagged_diff) > 1:
            tau.append(np.std(lagged_diff))

    tau = np.array(tau)

    # Filter out invalid (zero or NaN) tau values
    valid = (tau > 0)
    lags = np.array(lags[:len(tau)])[valid]
    tau = tau[valid]

    if len(tau) < 2:
        return np.nan

    # Perform linear regression on log-log scale
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0



@tool
def calculate_cumulative_sum(params: list) -> float: 
    """
    Calculate the logarithm of the cumulative sum of a list.

    Args:
        params (List[float]): A list of numerical values.

    Returns:
        float: The logarithm of the cumulative sum of the list.
    """
    return log(cumsum(params))





polygon = PolygonAPIWrapper()

polytool = PolygonToolkit.from_polygon_api_wrapper(polygon)


prompt = ChatPromptTemplate.from_messages(
    [
       ("system", "A reliable and exact assistant, capable of utilizing tools to answer questions. If a tool is unavailable, the assistant will notify you accordingly."),
       ("user", "{input}"),
       MessagesPlaceholder("chat_history", optional=True),
       MessagesPlaceholder(variable_name="agent_scratchpad"),
   ]
)
# Choose the LLM that will drive the agent
llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)

# setup the toolkit
# Combine all tools into a single list
tools = [calculate_cumulative_sum,calculate_hurst_exponent] + polytool.get_tools()


# Construct the OpenAI Tools agent
agent = create_openai_tools_agent(llm, tools, prompt)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

#result = agent_executor.invoke({"input": "Get aggregate data for X:BTCUSD ticker with a timespan of 1 day from 2023-01-09 to 2023-02-10."})
#print(result['output'])

#Determine the Hurst exponent using opening prices
#Determine the calculate_cumulative_sum of opening prices

agent_executor.invoke(
    {
        "chat_history": [
            HumanMessage(content="Get aggregate data for X:BTCUSD ticker with a timespan of 1 day from 2024-06-06 to 2024-07-07.")
        ],
        "input": "Calculate the Hurst exponent using opening prices by using the cumulative sum of them"
    }
)


#Error code: 400 - {'error': {'message': "This model's maximum context length is 16385 tokens. However, your messages resulted in 29429 tokens (28993 in the messages, 436 in the functions). Please reduce the length of the messages or functions.", 'type': 'invalid_request_error', 'param': 'messages', 'code': 'context_length_exceeded'}}