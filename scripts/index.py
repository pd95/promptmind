# This python script is producing the vector store based on the PDF files in the "docs" folder

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
            pdf_texts.append((filename, load_pdf_text(path)))
    return pdf_texts

def chunk_text(text: str, chunk_size=600, overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text)

# Load and split text
documents_folder = "docs"
pdf_texts = load_all_pdfs_texts(documents_folder)
all_documents = []
for filename, text in pdf_texts:
    print(filename)
    chunks = chunk_text(text)
    # Add metadata={'source': filename} to each Document
    all_documents.extend([Document(page_content=chunk, metadata={'source': filename}) for chunk in chunks])

# Use high-quality sentence transformer embeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build the FAISS vector store
db = FAISS.from_documents(all_documents, embedding)
db.save_local("vector_store/")