import os
import google.generativeai as genai
import cohere
from config.prompts import PROMPTS
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

# --- INITIAL SETUP ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
cohere_key = os.getenv("CO_API_KEY")

genai.configure(api_key=api_key)
co = cohere.Client(cohere_key)

DB_DIR = "./chroma_db"
PDF_PATH = "satcom-ngp.pdf"

# --- CORE FUNCTIONS ---

def load_pdf(path):
    reader = PdfReader(path)
    documents = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        documents.append(
            Document(
                page_content=text,
                metadata={"source": path, "page": page_num + 1}
            )
        )
    return documents

def initialize_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    
    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        print("Loading existing Vector DB from disk...")
        vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        # Load chunks for BM25
        docs = load_pdf(PDF_PATH)
        chunks = splitter.split_documents(docs)
    else:
        print(" No Vector DB found. Ingesting PDF...")
        docs = load_pdf(PDF_PATH)
        chunks = splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(
            chunks,
            embedding=embeddings,
            persist_directory=DB_DIR
        )
        print(f" Vector DB created and saved to {DB_DIR}")
    
    return vectorstore, chunks

# --- GLOBAL INITIALIZATION ---
vectorstore, chunks = initialize_vectorstore()

# Initialize retrievers ONCE
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
chunk_texts = [doc.page_content for doc in chunks]
tokenized_chunks = [text.split() for text in chunk_texts]
bm25 = BM25Okapi(tokenized_chunks)

# --- RAG PIPELINE ---

def hybrid_retrieve(query, k=10):
    vector_docs = vector_retriever.invoke(query)
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]
    bm25_docs = [chunks[i] for i in top_bm25_indices]
    
    # Combined and Deduplicated
    unique_docs = list({doc.page_content: doc for doc in (vector_docs + bm25_docs)}.values())
    return unique_docs

def rerank(query, docs, top_n=3):
    texts = [doc.page_content for doc in docs]
    results = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=texts,
        top_n=top_n
    )
    return [docs[result.index] for result in results.results]

def expand_query(query):
    prompt = PROMPTS["query_expansion"].format(question=query)
    model = genai.GenerativeModel("models/gemma-3-27b-it") 
    response = model.generate_content(prompt)
    return response.text.strip()

def ask_question(question):
    expanded_query = expand_query(question)
    retrieved_docs = hybrid_retrieve(expanded_query)
    docs = rerank(question, retrieved_docs)

    context = ""
    for i, doc in enumerate(docs):
        page = doc.metadata.get("page", "Unknown")
        context += f"[Source {i+1} - Page {page}]\n{doc.page_content}\n\n"

    prompt = PROMPTS["rag_answer"].format(context=context, question=question)
    model = genai.GenerativeModel("models/gemma-3-27b-it")
    response = model.generate_content(prompt)
    return response.text, docs

# --- ENTRY POINT ---
if __name__ == "__main__":
    while True:
        query = input("\nAsk a question (type exit to quit): ")
        if query.lower() == "exit": break
        answer, docs = ask_question(query)
        print("\nAnswer:\n", answer)