from app.embeddings import get_embedding
from app.query import load_vector_store, retrieve_documents, generate_rag_response
from app.settings import Settings
import argparse
from langchain_ollama import ChatOllama

def query_command(args: argparse.Namespace, settings: Settings) -> None:
    embedding = get_embedding(settings)
    db = load_vector_store(embedding)
    llm = ChatOllama(model=settings.llm_model, temperature=0)
    docs = retrieve_documents(db, args.prompt)
    answer = generate_rag_response(llm, docs, args.prompt)
    print(f"\nQ: {args.prompt}\nA: {answer}")