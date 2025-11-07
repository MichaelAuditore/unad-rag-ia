import os
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.llms import Ollama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


def load_documents(path="knowledge"):
    docs = []
    for file in os.listdir(path):
        full_path = os.path.join(path, file)
        if file.endswith(".pdf"):
            docs.extend(PyPDFLoader(full_path).load())
        elif file.endswith(".txt"):
            docs.extend(TextLoader(full_path).load())
    return docs


def build_rag(reindex=False):
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "db/chroma")

    if not reindex and os.path.exists(persist_dir):
        print("✅ Cargando base vectorial existente...")
        embeddings = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
        db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    else:
        print("🔹 Cargando documentos...")
        documents = load_documents()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        texts = splitter.split_documents(documents)

        print("🔹 Generando embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))

        print("🔹 Creando base vectorial...")
        db = Chroma.from_documents(texts, embeddings, persist_directory=persist_dir)

    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = Ollama(model=os.getenv("OLLAMA_MODEL", "gemma:2b"),
                 base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))

    print("🔹 Creando memoria conversacional...")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    print("🔹 Construyendo cadena RAG...")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True
    )
    return qa_chain


# Construye el agente RAG
rag_agent = build_rag(reindex=os.getenv("REINDEX", "false").lower() == "true")
