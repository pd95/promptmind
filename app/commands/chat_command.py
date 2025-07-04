from app.settings import Settings
from app.embeddings import get_embedding
from app.query import load_vector_store
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage, AIMessageChunk
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.retrievers import BaseRetriever
from typing import Annotated, TypedDict, Any, List
import datetime
import json

# 1. Define the retrieval tool
def make_semantic_search_tool(retriever: BaseRetriever):
    @tool
    def semantic_search(query: str) -> str:
        """
        Retrieve relevant documents from the knowledge base using semantic search.
        Returns the sources and relevant content.
        """
        docs = retriever.invoke(query)
        result = {
            "sources": [doc.metadata.get("source", "unknown") for doc in docs],
            "context": [doc.page_content for doc in docs]
        }
        return json.dumps(result)
    return semantic_search

# 2. Prepare the system prompt
def get_llm_prompt() -> str:
    local_tz = datetime.datetime.now().astimezone().tzinfo
    now = datetime.datetime.now(local_tz)
    return (
        f"You are a helpful assistant. Use tools if needed to answer the user's question.\n"
        f"Current date and time is {now.strftime('%A, %d %B %Y %H:%M:%S %Z')}.\n"
        f"Use the semantic_search tool to access the knowledge base when appropriate."
    )

# 3. Define the agent state
class AgentState(TypedDict):
    messages: Annotated[List[Any], add_messages]

# 4. Define the LLM and tool nodes
def make_run_llm(settings: Settings, tools):
    llm = ChatOllama(model=settings.llm_model, base_url=settings.ollama_url, temperature=0).bind_tools(tools)

    def run_llm(state: AgentState) -> dict:
        messages = state['messages']
        message = llm.invoke(messages)
        return {'messages': [message]}
    return run_llm

def tool_exists(state: AgentState) -> bool:
    result = state['messages'][-1]
    return hasattr(result, "tool_calls") and result.tool_calls and len(result.tool_calls) > 0

def stream_graph(graph: CompiledStateGraph, messages: List[AnyMessage], config):
    """"This function invokes the graph with the given messages and configuration and prints incoming chunks!"""

    state = {"messages": messages}

    print("")
    for s in graph.stream(state, config=config, stream_mode="messages"):
        chunk, _ = s

        curr_chunk_type = chunk.type
        if curr_chunk_type == "AIMessageChunk" and isinstance(chunk, AIMessageChunk):
            if chunk.tool_calls:
                print(f"triggering tool call(s) {len(chunk.tool_calls)}: ", chunk.tool_calls)
            else:
                print(chunk.content, end = "")

        elif curr_chunk_type == "tool":
            print(f"------- '{chunk.name}' completed with {chunk.status}: -------\n", chunk.content[:400], "\n-------\n")

        else:
            print(chunk.type, "********** ??? ********")

        if hasattr(chunk, "response_metadata") and isinstance(chunk.response_metadata, dict) and chunk.response_metadata.get("done"):
            print("")

# 5. Build the LangGraph agent
def chat_command(settings: Settings) -> None:
    print("Entering agentic chat mode (type 'exit' to quit)...")

    embedding = get_embedding(settings)
    db = load_vector_store(embedding)
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",  # or "similarity"
        search_kwargs={"k": 5, "score_threshold": 0.3}
    )
    tools = [make_semantic_search_tool(retriever)]
    run_llm = make_run_llm(settings, tools)

    memory = MemorySaver()
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("llm", run_llm)
    graph_builder.add_node("tools", ToolNode(tools))
    graph_builder.add_conditional_edges(
        "llm",
        tool_exists,
        {True: "tools", False: END}
    )
    graph_builder.add_edge("tools", "llm")
    graph_builder.add_edge(START, "llm")

    graph = graph_builder.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": "1"}}

    # Initial system prompt
    system_prompt = SystemMessage(content=get_llm_prompt())

    # Chat loop
    messages = [system_prompt]
    try:
        while True:
            user_input = input("\nYou: ")
            if user_input.strip().lower() in ["exit", "quit", "q"]:
                print("Exiting chat.")
                break
            messages.append(HumanMessage(content=user_input))
            stream_graph(graph, messages, config)
    except (KeyboardInterrupt, EOFError):
        print("\nExiting chat.")