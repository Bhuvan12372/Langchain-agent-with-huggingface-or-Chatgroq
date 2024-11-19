from langchain.llms import HuggingFaceHub
from langchain_groq import ChatGroq
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.tools import DuckDuckGoSearchRun
from langchain_experimental.tools import PythonREPLTool
from langchain.memory import ConversationBufferMemory
import os
from typing import Literal

class MultiModelAgent:
    def __init__(
        self,
        model_provider: Literal["huggingface", "groq"],
        api_key: str,
        model_name: str = None
    ):
        """
        Initialize the LangChain agent with either HuggingFace or Groq model.
        
        Args:
            model_provider: Choose between "huggingface" or "groq"
            api_key: API key for the chosen provider
            model_name: Model name (default: None - uses provider's default)
        """
        # Set up the language model based on provider
        self.llm = self._initialize_llm(model_provider, api_key, model_name)
        
        # Set up memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize tools
        self.tools = self._initialize_tools()
        
        # Initialize the agent
        self.agent = self._initialize_agent()

    def _initialize_llm(self, provider: str, api_key: str, model_name: str = None):
        """Initialize the language model based on the chosen provider."""
        if provider == "huggingface":
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
            return HuggingFaceHub(
                repo_id=model_name or "google/flan-t5-large",
                model_kwargs={
                    "temperature": 0.7,
                    "max_length": 512
                }
            )
        elif provider == "groq":
            os.environ["GROQ_API_KEY"] = api_key
            return ChatGroq(
                model_name=model_name or "mixtral-8x7b-32768",
                temperature=0.7,
                max_tokens=512
            )
        else:
            raise ValueError("Provider must be either 'huggingface' or 'groq'")
    
    def _initialize_tools(self):
        """Initialize and return the list of tools available to the agent."""
        # Load basic tools
        basic_tools = load_tools(
            ["llm-math"],
            llm=self.llm
        )
        
        # Add search capability
        search = DuckDuckGoSearchRun()
        
        # Add Python REPL for code execution
        python_repl = PythonREPLTool()
        
        return basic_tools + [search, python_repl]
    
    def _initialize_agent(self):
        """Initialize the agent with tools and memory."""
        return initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            max_iterations=5,
            early_stopping_method="generate"
        )
    
    def process_query(self, user_input: str) -> str:
        """
        Process a user query through the agent.
        
        Args:
            user_input: The user's question or command
            
        Returns:
            str: The agent's response
        """
        try:
            response = self.agent.run(input=user_input)
            return response
        except Exception as e:
            return f"An error occurred: {str(e)}"
    
    def reset_memory(self):
        """Clear the conversation memory."""
        self.memory.clear()

def main():
    """
    Main function to run the agent with choice of model provider.
    """
    # Select model provider
    while True:
        provider = input("Choose model provider (huggingface/groq): ").lower()
        if provider in ["huggingface", "groq"]:
            break
        print("Invalid choice. Please choose 'huggingface' or 'groq'")
    
    # Get API key
    api_key = input(f"Enter your {provider.title()} API key: ")
    
    # Optional: specify model name
    model_name = input(f"Enter model name (press Enter for default): ").strip() or None
    
    # Initialize the agent
    agent = MultiModelAgent(provider, api_key, model_name)
    print(f"\nAgent initialized with {provider.title()}. Type 'quit' to exit or 'reset' to clear memory.")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
            
        if user_input.lower() == 'reset':
            agent.reset_memory()
            print("Memory cleared. Starting fresh conversation.")
            continue
            
        if not user_input:
            print("Please enter a valid input.")
            continue
        
        response = agent.process_query(user_input)
        print(f"Assistant: {response}")

if __name__ == "__main__":
    main()