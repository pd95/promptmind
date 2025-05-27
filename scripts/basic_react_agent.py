import sys
from langchain_ollama import ChatOllama
from langchain.agents import initialize_agent
from langchain_community.tools import DuckDuckGoSearchRun

# Get query and optional model from command line
if len(sys.argv) < 2:
    print("Usage: python scripts/basic_react_agetn.py \"<your question>\" [model]")
    sys.exit(1)

user_query = sys.argv[1]
model = sys.argv[2] if len(sys.argv) > 2 else "granite3.1-dense:8b"


llm = ChatOllama(model=model)

search_tool = DuckDuckGoSearchRun()

tools = [search_tool]

agent = initialize_agent(tools=tools, llm=llm, agent="chat-zero-shot-react-description", verbose=True)

agent.invoke(user_query)
