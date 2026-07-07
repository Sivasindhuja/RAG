import os
import re
import pickle
from pathlib import Path
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.documents import Document
from langgraph.store.memory import InMemoryStore
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai
from rank_bm25 import BM25Okapi
import cohere
# Langfuse setup for query tracking and execution tracking
from langfuse.decorators import langfuse_context, observe

# ---------------- PATHS & CONFIG ---------------- #
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PERSIST_DIR = ROOT_DIR / "chroma_db"
DOCSTORE_PATH = ROOT_DIR / "parent_docstore.pkl"
CHUNKS_CACHE_PATH = ROOT_DIR / "child_chunks_cache.pkl"
PDF_PATH = DATA_DIR / "satcom-ngp.pdf"

# Load Environment Keys
load_dotenv(ROOT_DIR / ".env")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
co = cohere.Client(os.getenv("CO_API_KEY"))

# BAAI/bge-large-en-v1.5 provides optimal legal/regulatory domain text embeddings 
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

# --- Default Fallback Prompts if src.config is absent ---
PROMPTS = {
    "query_expansion": (
        "Rewrite the user question into a better search query for retrieving information from a technical document.\n"
        "The rewritten query should preserve meaning, include relevant technical terms, and be concise.\n"
        "Return ONLY the improved query text.\n\n"
        "Question:\n{question}"
    ),
    "rag_answer": (
        "You must answer ONLY using the provided context.\n"
        "Rules:\n"
        "- Do not use outside knowledge.\n"
        "- If the context does not contain the answer, reply exactly: 'The document does not contain this information.'\n"
        "- Cite sources using the format [Source 1], [Source 2], etc.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}"
    )
}

# ---------------- LOAD PDF ---------------- #
def load_pdf(path):
    """Extracts text from PDF page by page attaching explicit page tracking metadata."""
    reader = PdfReader(path)
    documents = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": str(path), "page": page_num + 1}
                )
            )
    return documents

# ---------------- RETRIEVER ---------------- #
class HierarchicalRetriever:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        # Recursive text splitting tailored for regulatory structures
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        self.vectorstore = None
        self.docstore = InMemoryStore()
        self.ns = ("parents",)
        self.all_children = []

    def add_documents(self, documents):
        parent_docs = self.parent_splitter.split_documents(documents)
        self.all_children = []

        for i, parent in enumerate(parent_docs):
            parent_id = f"parent_{i}"
            
            self.docstore.put(
                self.ns,
                parent_id,
                {
                    "page_content": parent.page_content,
                    "metadata": parent.metadata
                }
            )

            sub_children = self.child_splitter.split_documents([parent])
            for child in sub_children:
                child.metadata["parent_id"] = parent_id
                self.all_children.append(child)

        # ChromaDB setup and local persistence integration
        self.vectorstore = Chroma.from_documents(
            self.all_children,
            embedding=self.embeddings,
            persist_directory=str(PERSIST_DIR),
            collection_name="hierarchical_children",
        )
        self._persist_docstore()
        
        # Cache child chunks persistently to maintain exact index match alignment with BM25 across re-runs
        with open(CHUNKS_CACHE_PATH, "wb") as f:
            pickle.dump(self.all_children, f)
            
        print(f"Indexed {len(parent_docs)} parents and {len(self.all_children)} children.")

    def _persist_docstore(self):
        items = self.docstore.search(self.ns)
        data = {
            item.key: {
                "page_content": item.value["page_content"],
                "metadata": item.value["metadata"],
            }
            for item in items
        }
        with open(DOCSTORE_PATH, "wb") as f:
            pickle.dump(data, f)

    def _load_docstore(self):
        if os.path.exists(DOCSTORE_PATH):
            with open(DOCSTORE_PATH, "rb") as f:
                data = pickle.load(f)
            for k, v in data.items():
                self.docstore.put(self.ns, k, v)
        
        if os.path.exists(CHUNKS_CACHE_PATH):
            with open(CHUNKS_CACHE_PATH, "rb") as f:
                self.all_children = pickle.load(f)
        print("Loaded parent contexts and aligned child chunks from persistent local files.")

    def get_relevant_documents(self, query, k=6):
        if self.vectorstore is None:
            return []
        
        results = self.vectorstore.similarity_search(query, k=k * 3)
        parents = []
        seen_ids = set()

        for child in results:
            pid = child.metadata.get("parent_id")
            if pid and pid not in seen_ids:
                item = self.docstore.get(self.ns, pid)
                if item and item.value:
                    parents.append(
                        Document(
                            page_content=item.value["page_content"],
                            metadata=item.value["metadata"],
                        )
                    )
                seen_ids.add(pid)
        return parents[:k]

def get_hierarchical_components(documents):
    retriever = HierarchicalRetriever(embeddings)
    if os.path.exists(PERSIST_DIR) and os.path.exists(DOCSTORE_PATH) and os.path.exists(CHUNKS_CACHE_PATH):
        print("Loading existing local ChromaDB and cache instances...")
        retriever.vectorstore = Chroma(
            persist_directory=str(PERSIST_DIR),
            embedding_function=embeddings,
            collection_name="hierarchical_children",
        )
        retriever._load_docstore()
    else:
        print("Creating new hierarchical index partitions...")
        retriever.add_documents(documents)
    return retriever

# --- Pipeline Initialization Sequence ---
if PDF_PATH.exists():
    raw_docs = load_pdf(PDF_PATH)
    hier_retriever = get_hierarchical_components(raw_docs)
    child_chunks = hier_retriever.all_children
    bm25 = BM25Okapi([doc.page_content.split() for doc in child_chunks])
else:
    print(f"Warning: PDF file missing at {PDF_PATH}. System running in detached state.")
    hier_retriever = None
    bm25 = None
    child_chunks = []

# ---------------- RETRIEVAL ---------------- #

def hybrid_retrieve(query, k=5):
    # Retrieve top candidates via vector space match (Dense)
    vector_parents = hier_retriever.get_relevant_documents(query, k=k)
    
    # Retrieve top candidates via BM25 Okapi string tokens (Lexical)
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    bm25_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    bm25_children = [child_chunks[i] for i in bm25_indices]
    
    # Combine lists and unique-deduplicate via context body strings
    combined = vector_parents + bm25_children
    return list({doc.page_content: doc for doc in combined}.values())

def rerank(query, docs, top_n=3):
    if not docs:
        return []
    texts = [doc.page_content for doc in docs]
    results = co.rerank(model="rerank-english-v3.0", query=query, documents=texts, top_n=top_n)
    return [docs[res.index] for res in results.results if res.relevance_score > 0.20]

def expand_query(query):
    prompt = PROMPTS["query_expansion"].format(question=query)
    model = genai.GenerativeModel("models/gemma-3-27b-it")
    return model.generate_content(prompt).text.strip()

# ---------------- QA PIPELINE WITH OBSERVABILITY ---------------- #

@observe(name="SatCom_Inference_Pipeline")
def ask_question(question):
    # Set run metadata attributes directly within your traced execution context
    langfuse_context.update_current_trace(name="SatCom_QA_Inference", user_id="siva_dev")
    
    expanded_query = expand_query(question)
    raw_candidates = hybrid_retrieve(expanded_query)
    docs = rerank(question, raw_candidates)
    
    if not docs:
        langfuse_context.update_current_trace(tags=["refusal"])
        return "The document does not contain this information.", []
        
    context = ""
    for i, doc in enumerate(docs):
        page = doc.metadata.get("page", "Unknown")
        # Unified structured tagging scheme aligning to prompt expectations
        context += f"--- Context Block: [Source {i+1}] (Page {page}) ---\n"
        context += f"{doc.page_content}\n\n"
        
    prompt = PROMPTS["rag_answer"].format(context=context, question=question)
    model = genai.GenerativeModel("models/gemma-3-27b-it")
    response = model.generate_content(prompt)
    answer_text = response.text
    
    # Track citation density to assess quality metrics natively
    sentences = [s.strip() for s in re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', answer_text) if len(s.strip()) > 10]
    cited_count = sum(1 for s in sentences if re.search(r"\[Source \d+\]", s))
    coverage = cited_count / len(sentences) if sentences else 0
    
    langfuse_context.update_current_trace(
        metadata={"citation_coverage": round(coverage, 2), "chunk_type": "recursive_hierarchical"},
        tags=["high_grounding"] if coverage > 0.7 else ["low_grounding_warning"],
    )
    
    return answer_text, docs

if __name__ == "__main__":
    print("SatCom Intelligence Agent Ready.")
    while True:
        query = input("\nAsk a question: ")
        if query.lower() == "exit":
            break
        answer, _ = ask_question(query)
        print(f"\n======== ANSWER ========\n{answer}\n============")