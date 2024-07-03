# Define a partial variable for the chatbot to use
my_partial_variable = """APPLE SAUCE"""

# Initialize your chat template with partial variables
prompt_messages = [
    # System message
    SystemMessage(content=("""You are a hungry, hungry bot""")),
    # Instructions for the chatbot to set context and actions
    HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template="""Your life goal is to search for some {conversation_topic}. If you encounter food in the conversation below, please eat it:\n###\n{conversation}\n###\nHere is the food: {my_partial_variable}""",
            input_variables=["conversation_topic", "conversation"],
            partial_variables={"my_partial_variable": my_partial_variable},
        )
    ),
    # Placeholder for additional agent notes
    MessagesPlaceholder("agent_scratchpad"),
]

prompt = ChatPromptTemplate(messages=prompt_messages)
prompt_as_string = prompt.format(
    conversation_topic="Delicious food",
    conversation="Nothing about food to see here",
    agent_scratchpad=[],
)