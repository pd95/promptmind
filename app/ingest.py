import os
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.embeddings import Embeddings

def load_pdf_text(path: str) -> List[Document]:
    loader = PyPDFLoader(path, mode = "page", extraction_mode="layout")
    return loader.load()

def load_file(path: str) -> List[Document]:
    filename = os.path.basename(path)
    ext = filename.lower().split('.')[-1]
    if ext == "pdf":
        return load_pdf_text(path)
    elif ext in ("md", "txt"):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return [Document(page_content=content, metadata={"source": path})]
    return []

def load_all_texts(folder_path: str) -> List[Document]:
    all_docs: List[Document] = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        contents = load_file(file_path)
        if contents:
            all_docs.extend(contents)
    return all_docs

def load_url(url: str) -> List[Document]:
    docs: List[List[Document]] = [WebBaseLoader(url).load()]
    docs_list: List[Document] = [item for sublist in docs for item in sublist]
    return docs_list

def chunk_documents(docs_list: List[Document], chunk_size: int = 600, overlap: int = 100) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=overlap
    )
    return text_splitter.split_documents(docs_list)

def build_vector_store(
    documents: List[Document],
    embedding: Embeddings,
    vector_store_path: str = "vector_store/"
) -> FAISS:
    db = FAISS.from_documents(documents, embedding)
    db.save_local(vector_store_path)
    return db