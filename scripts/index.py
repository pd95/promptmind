# This python script is producing the vector store based on the PDF files in the "docs" folder

import sys
import os
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from pypdf import PdfReader
from langchain_community.document_loaders import WebBaseLoader

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
    app_name="Indexer",   # Specify a custom name. Otherwise sys.argv[0] will be used
    disable_batch=True,             # Don't batch the telemetry to make it immediately visible
)

tracer = trace.get_tracer(__name__)

# Default values
default_chunk_size = 600
default_overlap = 100
default_doc_sources = ["docs"]

@workflow(name="load_pdf_text")
def load_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

@workflow(name="load_file")
def load_file(path: str, filename: str) -> Document:
    """A helper to extract for each known file type the text into a `Document`"""
    if filename.lower().endswith(".pdf"):
        # read PDF from path
        print("  reading PDF", path)
        # Workaround: We should not return "plain text" in load_pdf_text and wrap it again as a document
        return Document(page_content=load_pdf_text(path), metadata={"source": path})
    elif filename.lower().endswith(".md"):
        # read plain content from path
        print("  reading Markdown", path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return Document(page_content=content, metadata={"source": path})
    elif filename.lower().endswith(".txt"):
        # read plain text content from path
        print("  reading Text", path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return Document(page_content=content, metadata={"source": path})
    return None

@workflow(name="load_all_texts")
def load_folder(path: str) -> list[Document]:
    """A helper to extract for each known file type the text into a `Document`"""
    all_docs = []
    print("importing from", path)
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path):
            content = load_file(file_path, filename)
            if content:
                all_docs.append(content)
        elif os.path.isdir(file_path):
            docs = load_folder(file_path)
            if len(docs) > 0:
                all_docs.extend(docs)
    return all_docs

@workflow(name="load_url")
def load_url(url: str) -> list[Document]:
    print("  downloading from URL", url)
    docs = [WebBaseLoader(url).load()]
    docs_list = [item for sublist in docs for item in sublist]
    return docs_list


@workflow(name="chunk_documents")
def chunk_documents(docs_list: str, chunk_size=600, overlap=100):
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=overlap
    )
    doc_splits = text_splitter.split_documents(docs_list)
    return doc_splits


# Parse CLI arguments
doc_sources = []
chunk_size = default_chunk_size
overlap = default_overlap

for arg in sys.argv[1:]:
    if arg.startswith("--"):
        if arg.startswith("--chunk_size="):
            chunk_size = int(arg.split("=", 1)[1])
        elif arg.startswith("--overlap="):
            overlap = int(arg.split("=", 1)[1])
    else:
        doc_sources.append(arg)

if not doc_sources:
    doc_sources = default_doc_sources

with tracer.start_as_current_span("Indexer main part") as span:

    # Process all documents
    span.add_event("Process all documents")
    all_documents = []
    for doc_src in doc_sources:

        if os.path.isdir(doc_src):
            all_documents.extend(load_folder(doc_src))
        elif os.path.isfile(doc_src):
            all_documents.append(load_file(doc_src, doc_src))
        elif doc_src.startswith("http://") or doc_src.startswith("https://"):
            docs = load_url(doc_src)
            all_documents.extend(docs)

    # Ensure data has been loaded
    if not all_documents:
        print("No documents found. Exiting.")
        sys.exit(1)

    print(f"Chunking {len(all_documents)} documents with chunk_size={chunk_size}, overlap={overlap}")
    all_documents = chunk_documents(all_documents, chunk_size=chunk_size, overlap=overlap)

    print("Initializing embedding model")
    from embeddings import embedding

    # Build the FAISS vector store
    print("Embedding ", len(all_documents), "chunks")
    with tracer.start_as_current_span("Vector Store") as span:
        db = FAISS.from_documents(all_documents, embedding)
        span.add_event("all documents added")
        db.save_local("vector_store/")
        span.add_event("vector_store saved")
