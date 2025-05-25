import sys
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Get query and optional model from command line
if len(sys.argv) < 2:
    print("Usage: python scripts/query.py \"<your question>\" [model]")
    sys.exit(1)

user_query = sys.argv[1]
model = sys.argv[2] if len(sys.argv) > 2 else "phi4-mini"

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("vector_store", embedding, allow_dangerous_deserialization=True)

def retrieve_documents(query: str, top: int = 10):
    docs = db.similarity_search(query, k=top)
    print("search_knowledge_base executed, returning num_docs=", len(docs))
    return (doc.page_content for doc in docs)

def generate_rag_response(query: str):
    docs = retrieve_documents(query)
    context = "\n---\n".join(docs)
    prompt = f"""You are a helpful assistant. Use only the following context to answer the question. If the answer isn't in the context, say 'I don't know'.
    Context: {context} Question: {query} Answer:"""
    llm = ChatOllama(model=model, temperature=0)
    response = llm([HumanMessage(content=prompt)])
    return response.content

answer = generate_rag_response(user_query)
print(f"Q: {user_query}\nA: {answer}")
