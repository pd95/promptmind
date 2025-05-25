# This python script is producing the vector store based on the PDF files in the "docs" folder

import sys
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from pypdf import PdfReader
from langchain_community.document_loaders import WebBaseLoader

def load_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

def load_file(path: str, filename: str) -> Document:
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

def load_all_texts(folder_path: str) -> list[Document]:
    all_docs = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        content = load_file(file_path, filename)
        if content:
            all_docs.append(content)
    return all_docs

def load_url(url: str) -> list[Document]:
    print("  downloading from URL", url)
    docs = [WebBaseLoader(url).load()]
    docs_list = [item for sublist in docs for item in sublist]
    return docs_list


def chunk_documents(docs_list: str, chunk_size=600, overlap=100):
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=overlap
    )
    doc_splits = text_splitter.split_documents(docs_list)
    return doc_splits


doc_sources = sys.argv[1::] if len(sys.argv) > 1 else ["docs"]
all_documents = []
for doc_src in doc_sources:

    if os.path.isdir(doc_src):
        print("importing from", doc_src)
        docs = load_all_texts(doc_src)
        all_documents.extend(docs)
    elif os.path.isfile(doc_src):
        all_documents.append(load_file(doc_src, doc_src))
    elif doc_src.startswith("http://") or doc_src.startswith("https://"):
        docs = load_url(doc_src)
        all_documents.extend(docs)

# Ensure data has been loaded
if not all_documents:
    print("No documents found. Exiting.")
    sys.exit(1)

print("Chunking", len(all_documents), "documents")
all_documents = chunk_documents(all_documents)

# Use high-quality sentence transformer embeddings
print("Initializing embedding model")
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build the FAISS vector store
print("Embedding ", len(all_documents), "chunks")
db = FAISS.from_documents(all_documents, embedding)
db.save_local("vector_store/")