from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize OpenAI Chat model
llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-3.5-turbo"
)

# Setup Wikipedia Search Tool
wikipedia = WikipediaAPIWrapper()
wiki_tool = Tool(
    name="Wikipedia Search",
    func=wikipedia.run,
    description="Useful for searching Wikipedia for background information"
)

# Create a simple calculator tool
def calculator(expression: str) -> str:
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error in calculation: {str(e)}"

calc_tool = Tool(
    name="Calculator",
    func=calculator,
    description="Useful for performing mathematical calculations"
)

# Create tools list
tools = [wiki_tool, calc_tool]

# Configure memory
memory = ConversationBufferMemory(return_messages=True)

# Initialize the agent
agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

def interact_with_agent(query: str) -> str:
    try:
        response = agent_executor.invoke({"input": query})
        return response["output"]
    except Exception as e:
        return f"An error occurred: {str(e)}"

def main():
    print("AI Assistant initialized. Type 'quit' to exit.")
    while True:
        query = input("\nYour question: ")
        if query.lower() == 'quit':
            break
        result = interact_with_agent(query)
        print(f"\nResponse: {result}")

if __name__ == "__main__":
    main()
