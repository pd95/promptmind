import sys
from opentelemetry import trace

tracer = trace.get_tracer(__name__)


with tracer.start_as_current_span("Load Embeddings") as span:

    model = "nomic-embed-text"
    for arg in sys.argv[1:]:
        if arg.startswith("--embed-model="):
            model = arg.split("=", 1)[1]
            print("Using embedding model", model)

    from langchain_ollama.embeddings import OllamaEmbeddings
    embedding = OllamaEmbeddings(model=model)

    #from langchain_huggingface import HuggingFaceEmbeddings
    #embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
