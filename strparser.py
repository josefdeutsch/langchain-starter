# Import necessary libraries
import os
from pprint import pprint
from dotenv import load_dotenv

# Import classes from langchain for OpenAI model, messages, and output parsing
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from a .env file
load_dotenv()

# Set the environment variables for the API keys from the loaded .env file
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv('LANGCHAIN_TRACING_V2')
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')

# Initialize the output parser
parser = StrOutputParser()

# Initialize the OpenAI model with the specified model name
model = ChatOpenAI(model="gpt-4")

# Define the messages to be sent to the model
messages = [
    SystemMessage(content="Translate the following from English into Italian"),
    HumanMessage(content="hi!"),
]

# Invoke the model with the defined messages and get the result
result = model.invoke(messages)

# Parse the result using the output parser and pretty print the output
pprint(parser.invoke(result))

# Chain the model invocation and parser together
chain = model | parser

# Invoke the chain with the defined messages and pretty print the parsed output
pprint(chain.invoke(messages))