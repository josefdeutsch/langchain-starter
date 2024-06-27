# Import necessary libraries
import os
from pprint import pprint
from dotenv import load_dotenv

# Import classes from langchain for OpenAI model, messages, and output parsing
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

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

# Define the system message template for translation
system_template = "Translate the following into {language}:"

# Create a prompt template from system and user messages
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

# Invoke the prompt template with the input data
result = prompt_template.invoke({"language": "italian", "text": "hi"})

# Convert the result to messages
result.to_messages()

# Chain the prompt template, model, and parser together
chain = prompt_template | model | parser

# Invoke the chain with the input data and pretty print the parsed output
pprint(chain.invoke({"language": "italian", "text": "hi"}))
