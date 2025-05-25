# This python script is producing the vector store based on the PDF files in the "docs" folder

import sys
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from pypdf import PdfReader

def load_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

def load_all_pdfs_texts(folder_path: str):
    pdf_texts = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            print("  reading", path)
            pdf_texts.append((filename, load_pdf_text(path)))
    return pdf_texts

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
    print("importing from", doc_src)
    pdf_texts = load_all_pdfs_texts(doc_src)
    for filename, text in pdf_texts:
        chunks = chunk_text(text)
        # Add metadata={'source': filename} to each Document
        doc_chunks = [Document(page_content=chunk, metadata={'source': filename}) for chunk in chunks]
        all_documents.extend(doc_chunks)

# Use high-quality sentence transformer embeddings
print("Initializing embedding model")
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build the FAISS vector store
print("Embedding ", len(all_documents), "chunks")
db = FAISS.from_documents(all_documents, embedding)
db.save_local("vector_store/")