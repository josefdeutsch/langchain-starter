import getpass
import os
from langchain_openai import ChatOpenAI

# Prompt the user for the OpenAI API key
openai_api_key = getpass.getpass(prompt="Enter your OpenAI API key: ")

# Set the environment variable for the API key
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"

# Function to verify the OpenAI API key
def verify_openai_api_key(api_key):
    try:
        # Initialize the OpenAI model with the provided API key
        model = ChatOpenAI(api_key=api_key, model="gpt-4")

        # Make a test call to verify the key
        response = model.invoke([{"role": "system", "content": "Translate the following from English into Italian"}, {"role": "user", "content": "hi!"}])
        print("API Key is valid.")
        print(response)
        return True
    except Exception as e:
        print(f"Invalid API Key or other error occurred: {e}")
        return False

# Verify the OpenAI API key
is_valid = verify_openai_api_key(openai_api_key)
if not is_valid:
    raise ValueError("Invalid OpenAI API Key. Please check your API key.")
