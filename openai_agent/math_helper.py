from pprint import pprint
from numpy import cumsum, log, polyfit, sqrt, std, subtract
from numpy.random import randn
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_community.tools import MoveFileTool
from langchain_core.messages import HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import ChatOpenAI

@tool
def hurst(ts) -> float:
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
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = polyfit(log(lags), log(tau), 1)
    return poly[0] * 2.0

# Generate a time series with 1000 items
#ts = log(cumsum(randn(1000)) + 1000)

#hurst_exponent = hurst(ts)
#print("Hurst Data:", ts)
#print("Hurst Exponent:", hurst_exponent)
