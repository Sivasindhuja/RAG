import os
import logging
from dotenv import load_dotenv
from pypdf import PdfReader
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from rank_bm25 import BM25Okapi

import google.generativeai as genai
import cohere
from config.prompts import PROMPTS


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
cohere_key = os.getenv("CO_API_KEY")

genai.configure(api_key=api_key)
co = cohere.Client(cohere_key)

DB_DIR = "./chroma_db"
STORE_DIR = "./parent_store"
PDF_PATH = "satcom-ngp.pdf"


# 1. Setup Logging 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_system.log"), # Saves errors to a file
        logging.StreamHandler() # Prints to terminal
    ]
)
logger = logging.getLogger(__name__)

# 2. Define the Retry Logic
# This will wait 4s, 8s, 16s... up to 5 attempts if the API fails.
retry_logic = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    before_sleep=lambda retry_state: logger.warning(f"🔄 Rate limit hit. Retrying in {retry_state.next_action.sleep}s...")
)

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

def initialize_pdr_system():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

    # Note: Use a separate collection or directory for PDR to avoid metadata conflicts
    vectorstore = Chroma(
        collection_name="satcom_pdr",
        embedding_function=embeddings,
        persist_directory="./pdr_db" 
    )

    fs = LocalFileStore("./pdr_parent_store")
    store = create_kv_docstore(fs)

    # IMPROVEMENT: Enable MMR search type here
    # fetch_k: gets 20 candidates, k: picks 5 most diverse from them
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_type="mmr", 
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5} 
    )

    if not os.path.exists("./pdr_parent_store") or len(vectorstore.get()['ids']) == 0:
        logger.info("🚀 Indexing PDR System...")
        docs = load_pdf(PDF_PATH)
        retriever.add_documents(docs)
    
    return retriever



# Global Instance
pdr_retriever = initialize_pdr_system()

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

@retry_logic
def expand_query(query):
    try:
            prompt = PROMPTS["query_expansion"].format(question=query)
            # Using flash-lite for cost/speed efficiency
            model = genai.GenerativeModel("models/gemini-2.0-flash-lite") 
            response = model.generate_content(prompt)
            return response.text.strip()
    
    except Exception as e:
        logger.error(f"Failed to expand query: {e}")
        raise # Raise so tenacity can catch and retry


@retry_logic
def ask_question(question):
    try:
        # Step A: Query Expansion
        expanded_query = expand_query(question)
        logger.info(f" Searching for: {expanded_query}")

        # Step B: PDR Retrieval (Implicitly uses MMR)
        # Finds tiny child matches -> returns large parent articles
        parent_docs = pdr_retriever.invoke(expanded_query)

        # Step C: Reranking (Ensures the top 3 of the 5 parents are the best)
        docs = rerank(question, parent_docs, top_n=3)

        context = ""
        for i, doc in enumerate(docs):
            page = doc.metadata.get("page", "Unknown")
            # These are now full articles, so context is much richer!
            context += f"[Full Policy Article - Page {page}]\n{doc.page_content}\n\n"

        prompt = PROMPTS["rag_answer"].format(context=context, question=question)
        
        model = genai.GenerativeModel("models/gemma-3-27b-it")
        response = model.generate_content(prompt)
        
        return response.text, docs
    except Exception as e:
        logger.error(f"Error in ask_question: {e}")
        raise

# --- ENTRY POINT ---
if __name__ == "__main__":
    while True:
        query = input("\nAsk a question (type exit to quit): ")
        if query.lower() == "exit": break
        answer, docs = ask_question(query)
        print("\nAnswer:\n", answer)