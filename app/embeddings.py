from langchain_ollama.embeddings import OllamaEmbeddings
from app.settings import Settings

def get_embedding(settings: Settings) -> OllamaEmbeddings:
    return OllamaEmbeddings(model=settings.embedding_model, base_url=settings.ollama_url)