from opentelemetry import trace

tracer = trace.get_tracer(__name__)


with tracer.start_as_current_span("Load Embeddings") as span:

    from langchain_huggingface import HuggingFaceEmbeddings
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
