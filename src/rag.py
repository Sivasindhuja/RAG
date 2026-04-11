import os
import re
import pickle
from pathlib import Path
from langfuse.decorators import observe, langfuse_context
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

try:
    from src.config.prompts import PROMPTS
except ImportError:
    from config.prompts import PROMPTS
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PERSIST_DIR = ROOT_DIR / "chroma_db"
DOCSTORE_PATH = ROOT_DIR / "parent_docstore.pkl"
PDF_PATH = DATA_DIR / "satcom-ngp.pdf"

# Load Environment Keys
load_dotenv(ROOT_DIR / ".env") # Ensure .env is loaded from the root
api_key = os.getenv("GEMINI_API_KEY")
cohere_key = os.getenv("CO_API_KEY")
PERSIST_DIR_STR = str(PERSIST_DIR)
DOCSTORE_PATH_STR = str(DOCSTORE_PATH)


genai.configure(api_key=api_key)
co = cohere.Client(cohere_key)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def load_pdf(path):
    """Extracts text from PDF and returns a list of LangChain Documents."""
    reader = PdfReader(path)
    documents = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        documents.append(
            Document(
                page_content=text, 
                metadata={"source": str(path), "page": page_num + 1}
            )
        )
    return documents

class HierarchicalRetriever:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        self.vectorstore = None
        self.docstore = InMemoryStore()
        self.ns = ("parents",)

    def add_documents(self, documents):
        parent_docs = self.parent_splitter.split_documents(documents)
        persist_directory=str(PERSIST_DIR)
        all_children = []

        for i, parent in enumerate(parent_docs):
            parent_id = f"parent_{i}"

            self.docstore.put(
                self.ns,
                parent_id,
                {
                    "page_content": parent.page_content,
                    "metadata": parent.metadata,
                },
            )

            sub_children = self.child_splitter.split_documents([parent])
            for child in sub_children:
                child.metadata["parent_id"] = parent_id
                all_children.append(child)

        self.vectorstore = Chroma.from_documents(
            all_children,
            embedding=self.embeddings,
            persist_directory=PERSIST_DIR,
            collection_name="hierarchical_children",
        )

        self._persist_docstore()
        print(f"Indexed {len(parent_docs)} parents and {len(all_children)} children.")

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
            print(" Loaded parent context from disk.")

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
    if os.path.exists(PERSIST_DIR) and os.path.exists(DOCSTORE_PATH):
        print("Loading existing hierarchical index...")
        retriever.vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
            collection_name="hierarchical_children",
        )
        retriever._load_docstore()
    else:
        print("Creating new hierarchical index...")
        retriever.add_documents(documents)
    return retriever


# --- Initialize System ---
if PDF_PATH.exists():
    raw_docs = load_pdf(PDF_PATH)
    hier_retriever = get_hierarchical_components(raw_docs)
    if hasattr(hier_retriever, 'all_children') and hier_retriever.all_children:
        child_chunks = hier_retriever.all_children
    else:
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        child_chunks = child_splitter.split_documents(raw_docs)
    # BM25 still uses small children for keyword precision 
    bm25 = BM25Okapi([doc.page_content.split() for doc in child_chunks])
else:
    print(f"Warning: SatCom PDF not found at {PDF_PATH}. Retrieval will be disabled.")
    hier_retriever = None
    bm25 = None


@observe()
def hybrid_retrieve(query, k=5):
    vector_parents = hier_retriever.get_relevant_documents(query)

    tokenized_query = query.split()
    bm25_indices = sorted(
        range(len(bm25.get_scores(tokenized_query))),
        key=lambda i: bm25.get_scores(tokenized_query)[i],
        reverse=True,
    )[:k]
    bm25_children = [child_chunks[i] for i in bm25_indices]

    combined = vector_parents + bm25_children
    return list({doc.page_content: doc for doc in combined}.values())


@observe()
def rerank(query, docs, top_n=3):
    texts = [doc.page_content for doc in docs]
    results = co.rerank(model="rerank-english-v3.0", query=query, documents=texts, top_n=top_n)
    return [docs[res.index] for res in results.results if res.relevance_score > 0.20]


@observe()
def expand_query(query):
    prompt = PROMPTS["query_expansion"].format(question=query)
    model = genai.GenerativeModel("models/gemma-3-27b-it")
    return model.generate_content(prompt).text.strip()


@observe()
def ask_question(question):
    langfuse_context.update_current_trace(name="SatCom_QA_Inference", user_id="siva_dev")
    expanded_query = expand_query(question)
    docs = rerank(question, hybrid_retrieve(expanded_query))

    if not docs:
        langfuse_context.update_current_trace(tags=["refusal"])
        return "The document does not contain this information.", []

    context = ""
    seen_pages = set()
    for doc in docs:
        page = doc.metadata.get("page", "Unknown")
        if page not in seen_pages:
            context += f"--- Document Section: Page {page} ---\n"
            seen_pages.add(page)
        context += f"{doc.page_content}\n\n"

    prompt = PROMPTS["rag_answer"].format(context=context, question=question)
    model = genai.GenerativeModel("models/gemma-3-27b-it")
    response = model.generate_content(prompt)
    answer_text = response.text

    # Phase 5 Metrics logic
    sentences = [
        s.strip()
        for s in re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', answer_text)
        if len(s.strip()) > 10
    ]
    cited_count = sum(1 for s in sentences if re.search(r"\[Page \d+\]", s))
    coverage = cited_count / len(sentences) if sentences else 0

    langfuse_context.update_current_trace(
        metadata={"citation_coverage": round(coverage, 2), "chunk_type": "hierarchical_parent"},
        tags=["high_grounding"] if coverage > 0.7 else ["low_grounding_warning"],
    )
    langfuse_context.update_current_observation(input=prompt, output=answer_text, model="gemma-3-27b-it")
    return answer_text, docs


if __name__ == "__main__":
    print("SatCom Intelligence Agent - Modern Hierarchical RAG Ready!")
    while True:
        query = input("\nAsk a question: ")
        if query.lower() == "exit":
            break
        answer, _ = ask_question(query)
        print("\n" + "=" * 80 + "\nANSWER:\n", answer + "\n" + "=" * 80)