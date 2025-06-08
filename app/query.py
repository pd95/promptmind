from typing import List
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import FAISS

def load_vector_store(embedding: Embeddings, path: str = "vector_store/") -> FAISS:
    return FAISS.load_local(path, embedding, allow_dangerous_deserialization=True)

def retrieve_documents(db: FAISS, query: str, top_k: int = 10) -> List[Document]:
    docs = db.similarity_search(query, k=top_k)
    sources = {doc.metadata.get("source") for doc in docs if "source" in doc.metadata}
    print("Distinct source files:", sources)
    return docs

def generate_rag_response(llm: ChatOllama, docs: List[Document], query: str) -> str:
    context = "\n---\n".join(doc.page_content for doc in docs)
    prompt = (
        f"You are a helpful assistant. Use only the following context to answer the question. "
        f"If the answer isn't in the context, say 'I don't know'.\n"
        f"Context: {context}\nQuestion: {query}\nAnswer:"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content