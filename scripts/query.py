import sys
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv 
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow
from opentelemetry import trace

# Load environment
if not(load_dotenv(verbose=True)):
    print("ERROR: .env not found!")
    exit(1)

# Setup tracing (see .env file and https://traceloop.com/docs/openllmetry/configuration)
Traceloop.init(
    app_name="Query RAG",   # Specify a custom name. Otherwise sys.argv[0] will be used
    disable_batch=True,             # Don't batch the telemetry to make it immediately visible
)

tracer = trace.get_tracer(__name__)

# Get query and optional model from command line
if len(sys.argv) < 2:
    print("Usage: python scripts/query.py [--model=<model name>] [--embed-model=<model name>] \"<your question>\"")
    sys.exit(1)

# Check command line arguments
model = "llama3.2"
user_query = ""
for arg in sys.argv[1:]:
    if arg.startswith("--"):
        if arg.startswith("--model="):
            model = arg.split("=", 1)[1]
    elif len(user_query) == 0:
        user_query = arg

if len(user_query) == 0:
    print("No query given")
    exit(1)

@workflow(name="retrieve_documents")
def retrieve_documents(query: str, top: int = 10):
    docs = db.similarity_search(query, k=top)
    sources = {doc.metadata.get("source") for doc in docs if "source" in doc.metadata}
    print("search_knowledge_base executed, found ", len(docs), "fragments from", sources)
    return (doc.page_content for doc in docs)

@workflow(name="generate_rag_response")
def generate_rag_response(query: str):
    docs = retrieve_documents(query)
    context = "\n---\n".join(docs)
    prompt = f"""You are a helpful assistant. 
    Use only the following context to answer the question. 
    If the answer isn't in the context, say 'I don't know'.
    
    Context: {context}
    
    Question: {query} 
    
    Answer:"""
    llm = ChatOllama(model=model, temperature=0)
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

with tracer.start_as_current_span("main part") as span:
    with tracer.start_as_current_span("initialization") as span:
        print("Loading Embeddings")
        from embeddings import embedding

        with tracer.start_as_current_span("Loading Vector Store") as span:
            db = FAISS.load_local("vector_store", embedding, allow_dangerous_deserialization=True)


    answer = generate_rag_response(user_query)
    print(f"Q: {user_query}\nA: {answer}")
