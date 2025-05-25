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

def load_file(path: str, filename: str) -> str:
    if filename.lower().endswith(".pdf"):
        # read PDF from path
        print("  reading PDF", path)
        return (path, load_pdf_text(path))
    elif filename.lower().endswith(".md"):
        # read plain content from path
        print("  reading Markdown", path)
        with open(path, "r", encoding="utf-8") as f:
            return (path, f.read())
    elif filename.lower().endswith(".txt"):
        # read plain text content from path
        print("  reading Text", path)
        with open(path, "r", encoding="utf-8") as f:
            return (path, f.read())
    return None

def load_all_texts(folder_path: str):
    all_texts = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        content = load_file(file_path, filename)
        if content:
            all_texts.append(content)
    return all_texts

def load_url(url: str) -> list[Document]:
    print("  downloading from URL", url)
    docs = WebBaseLoader(url).load()

    # Workaround because currently we can only handle text and not documents!
    docs_list = "\n\n".join(item.page_content for item in docs)
    return (url, docs_list)


def chunk_text(text: str, chunk_size=600, overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text)


doc_sources = sys.argv[1::] if len(sys.argv) > 1 else ["docs"]
all_documents = []
for doc_src in doc_sources:

    all_texts = []
    if os.path.isdir(doc_src):
        print("importing from", doc_src)
        all_texts = load_all_texts(doc_src)
    elif os.path.isfile(doc_src):
        all_texts.append(load_file(doc_src, doc_src))
    elif doc_src.startswith("http://") or doc_src.startswith("https://"):
        all_texts.append(load_url(doc_src))

    for filename, text in all_texts:
        chunks = chunk_text(text)
        # Add metadata={'source': filename} to each Document
        doc_chunks = [Document(page_content=chunk, metadata={'source': filename}) for chunk in chunks]
        all_documents.extend(doc_chunks)

# Ensure data has been loaded
if not all_documents:
    print("No documents found. Exiting.")
    sys.exit(1)

# Use high-quality sentence transformer embeddings
print("Initializing embedding model")
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build the FAISS vector store
print("Embedding ", len(all_documents), "chunks")
db = FAISS.from_documents(all_documents, embedding)
db.save_local("vector_store/")