from langchain_community.callbacks import get_openai_callback
from langchain.chains import LLMChain
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor, create_openai_tools_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


# Step 1: Define the Tool
#ADD APIs HERE
#Add more tools with the @tool header on a function definition
@tool
def example_tool(query) :
    """A simple example tool that echoes the input."""
    response = f"The world population is 100 Trillion" #NOTICE WE CAN CONTROL THE AI WITH WRONG INFO
    return response

# Step 2: Prompt Grapher Function
def prompt_grapher(query, llm):
    # Define tools
    tools = [example_tool] #tools need to be added by here as well

    # Define prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder("chat_history", optional=True),
            ("system",
             """You are a helpful assistant. Use tools effectively to provide accurate responses. 
             
             USE ONLY THE INFORMATION FROM THE TOOLS DO NOT USE OUTSIDE INFORMATION
             - If you do not know from given prompt then answer -> I Don't Know"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # Create the agent and bind tools
    agent = create_openai_tools_agent(llm, tools,prompt  )

    # Create an agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=10,
    )

    # Run the agent with the query
    with get_openai_callback() as cb:
        response = agent_executor.invoke(
            {
                "input": query, # Later we can add chat_history to replicate conversation memory as well!
            }
        )
        print(response)
        return response

# Main Execution
if __name__ == "__main__":
    # Initialize the AI model
    llm = ChatOpenAI(
        model_name="gpt-4o",
        api_key="",
        temperature=0,
    )

    # User query
    user_input = "What is the population of the world?"
    prompt_grapher(user_input, llm)


# langchain==0.2.5
# langchain-core==0.2.9
# langchain-openai==0.1.9
# langchain-community==0.2.5
