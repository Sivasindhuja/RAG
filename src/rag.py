import os
try:
    from src.config.prompts import PROMPTS
except ImportError:
    from config.prompts import PROMPTS
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import google.generativeai as genai
from rank_bm25 import BM25Okapi
import cohere

# Load Environment Keys
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
cohere_key = os.getenv("CO_API_KEY")

PERSIST_DIR = "./chroma_db"

# Initialize Clients
genai.configure(api_key=api_key)
co = cohere.Client(cohere_key)

# Step 1: Load PDF
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

# Step 2: Ingestion & VectorStore (Persistent)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def get_vectorstore(documents):
    """
    Checks if Vector DB exists on disk. 
    If not, creates it using 'Small' chunks for better matching.
    """
    if os.path.exists(PERSIST_DIR) and len(os.listdir(PERSIST_DIR)) > 0:
        print(" Loading existing Vector DB from disk...")
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    
    print(" Creating new Vector DB (One-time process)...")
    # Using 'Small' chunks (300) for the actual Vector Search index
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    child_chunks = child_splitter.split_documents(documents)
    
    return Chroma.from_documents(
        child_chunks, 
        embedding=embeddings, 
        persist_directory=PERSIST_DIR
    )

# --- Initializing the Knowledge Base ---
current_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(current_dir, "satcom-ngp.pdf")
raw_docs = load_pdf(pdf_path)
vectorstore = get_vectorstore(raw_docs)

# Step 3: Retrieval Setup
# We increase 'k' because reranking will filter the noise later
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

# BM25 Setup (using same chunks as Vector Store for consistency)
# Note: In production, you'd persist BM25 indices too, but this is fast.
child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks_for_bm25 = child_splitter.split_documents(raw_docs)
chunk_texts = [doc.page_content for doc in chunks_for_bm25]
tokenized_chunks = [text.split() for text in chunk_texts]
bm25 = BM25Okapi(tokenized_chunks)

def hybrid_retrieve(query, k=10):
    # Vector retrieval
    vector_docs = vector_retriever.invoke(query)

    # BM25 retrieval
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]
    bm25_docs = [chunks_for_bm25[i] for i in top_bm25_indices]

    # Combine and Deduplicate
    combined = vector_docs + bm25_docs
    unique_docs = list({doc.page_content: doc for doc in combined}.values())
    return unique_docs

# Step 4: Reranking with Relevance Guardrails
def rerank(query, docs, top_n=3):
    texts = [doc.page_content for doc in docs]
    results = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=texts,
        top_n=top_n
    )

    reranked_docs = []
    for result in results.results:
        # PRODUCTION GUARDRAIL: Only include if the relevance score is > 0.2
        # This helps in "Not Found" cases
        if result.relevance_score > 0.20:
            reranked_docs.append(docs[result.index])
    
    return reranked_docs

# Step 5: Query Expansion
def expand_query(query):
    prompt = PROMPTS["query_expansion"].format(question=query)
    model = genai.GenerativeModel("models/gemma-3-27b-it") 
    response = model.generate_content(prompt)
    return response.text.strip()

# Step 6: Main QA Logic
def ask_question(question):
    # 1. Expand
    expanded_query = expand_query(question)
    print("\nExpanded Query:", expanded_query)

    # 2. Retrieve (Hybrid)
    retrieved_docs = hybrid_retrieve(expanded_query)

    # 3. Rerank & Filter
    docs = rerank(question, retrieved_docs)
    
    # 4. Handle "Not Found" Early
    if not docs:
        return "The document does not contain this information.", []

    # 5. Build context with Source Citations
    context = ""
    # unique_pages = set()
    seen_pages = set()
    for i, doc in enumerate(docs):
        page = doc.metadata.get("page", "Unknown")
        # If we have multiple chunks from the same page, we merge them
        if page not in seen_pages:
            context += f"--- Document Section: Page {page} ---\n"
            seen_pages.add(page)
        
        context += f"{doc.page_content}\n\n"
   
    # 6. Generate final answer
    prompt = PROMPTS["rag_answer"].format(context=context, question=question)
    model = genai.GenerativeModel("models/gemma-3-27b-it")
    response = model.generate_content(prompt)

    return response.text, docs

if __name__ == "__main__":
    print("SatCom Intelligence Agent Stage 2 Ready.")
    while True:
        query = input("\nAsk a question (type exit to quit): ")
        if query.lower() == "exit":
            break
        answer, docs = ask_question(query)
        print("\nAnswer:\n", answer)