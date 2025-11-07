from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.llms import Ollama
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_classic.memory.buffer import ConversationBufferMemory

import os

def load_documents(path="knowledge"):
    docs = []
    for file in os.listdir(path):
        full_path = os.path.join(path, file)
        if file.endswith(".pdf"):
            docs.extend(PyPDFLoader(full_path).load())
        elif file.endswith(".txt"):
            docs.extend(TextLoader(full_path).load())
    return docs

def build_rag():
    print("🔹 Cargando documentos...")
    documents = load_documents()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = splitter.split_documents(documents)
    
    print("🔹 Generando embeddings...")
    embeddings = HuggingFaceEmbeddings()
    
    print("🔹 Creando base vectorial...")
    db = Chroma.from_documents(texts, embeddings, persist_directory="db/chroma")

    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = Ollama(model="gemma:2b", base_url="http://localhost:11434")

    print("🔹 Creando memoria conversacional...")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    print("🔹 Construyendo cadena RAG con memoria...")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        chain_type="stuff",
        verbose=True
    )
    return qa_chain

rag_agent = build_rag()
