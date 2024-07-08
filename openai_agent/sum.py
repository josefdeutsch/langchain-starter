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

@tool
def calculate_hurst_exponent(ts) -> float:
    """
    Calculate the Hurst Exponent of a time series vector `ts`.
    
    The Hurst Exponent indicates whether a series is mean-reverting, trending, or 
    following a Geometric Brownian Motion. It is determined by computing variances of 
    lagged differences and fitting a linear model to estimate the exponent. An H value 
    near 0 indicates strong mean reversion, while H near 1 indicates strong trending. 
    The Hurst Exponent is useful for identifying time series behavior, particularly 
    for mean-reverting trading strategies.
    
    Parameters:
    ts (array-like): The time series vector.
    
    Returns:
    float: The estimated Hurst Exponent.
    """

    lags = range(2, 100)  # A practical range of lags
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0


@tool
def get_result_of_hurst_exponent(a: int) -> np.ndarray:
    """
    Generate a log-cumulative sum of random numbers plus 1000.

    This function generates an array by taking the cumulative sum of random numbers
    and adding 1000 to each element. The logarithm of each element is then returned.

    Parameters:
    a (int): The number of random numbers to generate.

    Returns:
    np.ndarray: The resulting array after log-cumulative summation.
    """
    if a <= 0:
        raise ValueError("The input 'a' must be a positive integer.")
    
    random_numbers = np.random.random(a)
    cumulative_sum = np.cumsum(random_numbers) + 1000
    
    # Ensure no zero or negative values are passed to log
    if np.any(cumulative_sum <= 0):
        raise ValueError("Cumulative sum contains non-positive values.")
    
    log_cumulative_sum = np.log(cumulative_sum)
    
    return log_cumulative_sum





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
tools = [calculate_hurst_exponent, get_result_of_hurst_exponent] + polytool.get_tools()


# Construct the OpenAI Tools agent
agent = create_openai_tools_agent(llm, tools, prompt)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

#result = agent_executor.invoke({"input": "Get aggregate data for X:BTCUSD ticker with a timespan of 1 day from 2023-01-09 to 2023-02-10."})
#print(result['output'])

agent_executor.invoke(
    {
        "chat_history": [
            HumanMessage(content="Get aggregate data for X:BTCUSD ticker with a timespan of 1 day from 2024-06-06 to 2024-07-07."),
            AIMessage(content="Hello Bob! How can I assist you today?"),
        ],
        "input": "Process the keys labeled: 'c' in every entry and calculate the Hurst exponent.",
    }
)


#Error code: 400 - {'error': {'message': "This model's maximum context length is 16385 tokens. However, your messages resulted in 29429 tokens (28993 in the messages, 436 in the functions). Please reduce the length of the messages or functions.", 'type': 'invalid_request_error', 'param': 'messages', 'code': 'context_length_exceeded'}}