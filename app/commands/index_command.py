from app.embeddings import get_embedding
from app.ingest import load_folder, load_file, load_url, chunk_documents, build_vector_store
from app.settings import Settings
import os
import argparse

def index_command(args: argparse.Namespace, settings: Settings) -> None:
    embedding = get_embedding(settings)
    all_documents = []
    for src in args.sources:
        if os.path.isdir(src):
            print(f"Importing from directory: {src}")
            all_documents.extend(load_folder(src))
        elif os.path.isfile(src):
            doc = load_file(src)
            if doc:
                all_documents.append(doc)
        elif src.startswith("http://") or src.startswith("https://"):
            print(f"Importing from URL: {src}")
            all_documents.extend(load_url(src))
        else:
            print(f"Skipping unknown source: {src}")

    if not all_documents:
        print("No documents found. Exiting.")
        return

    print(f"Chunking {len(all_documents)} documents with chunk_size={settings.chunk_size}, overlap={settings.overlap}")
    chunks = chunk_documents(all_documents, chunk_size=settings.chunk_size, overlap=settings.overlap)

    print("Building vector store...")
    build_vector_store(chunks, embedding)
    print("Indexing complete.")