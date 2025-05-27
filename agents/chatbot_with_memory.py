# chatbot_with_memory.py
# based on https://langchain-ai.github.io/langgraph/tutorials/get-started/3-add-memory
# adapted for Ollama using phi4

from typing import Annotated

from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages

from langgraph.checkpoint.memory import MemorySaver

import sys

# Default values
default_model = "phi4"

# Check command line arguments
model = default_model
for arg in sys.argv[1:]:
    if arg.startswith("--model="):
        model = str(arg.split("=", 1)[1])


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

llm = init_chat_model("ollama:"+model)

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")

# The memory saver and checkpointer system ensures that chatbot can remember facts during the current conversation. 
# E.g. your name. But after restarting the chatbot, the memory is lost.
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "1"}}


def stream_graph_updates(user_input: str):
    for event in graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config
    ):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
