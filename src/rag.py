import os
import pickle
from pathlib import Path

import fitz
import cohere
import google.generativeai as genai

from dotenv import load_dotenv
from langfuse.decorators import observe, langfuse_context
from langchain_core.documents import Document
from langgraph.store.memory import InMemoryStore
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

try:
    from src.config.prompts import PROMPTS
except ImportError:
    from config.prompts import PROMPTS

for model in client.models.list():
    print(f"Name: {model.name}")
    print(f"Supported Methods: {model.supported_generation_methods}\n")


# ---------------- PATHS ---------------- #

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"

PERSIST_DIR = ROOT_DIR / "chroma_db"

DOCSTORE_PATH = ROOT_DIR / "parent_docstore.pkl"
print("ROOT_DIR:", ROOT_DIR)
print("DATA_DIR:", DATA_DIR)

DATA_DIR = ROOT_DIR / "data"

# ---------------- ENV ---------------- #

load_dotenv(ROOT_DIR / ".env")

api_key = os.getenv("GEMINI_API_KEY")
cohere_key = os.getenv("CO_API_KEY")

genai.configure(api_key=api_key)

co = cohere.Client(cohere_key)


# ---------------- EMBEDDINGS ---------------- #

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
pdf_files = list(DATA_DIR.glob("*.pdf"))

# ---------------- LOAD PDFs ---------------- #

def load_all_pdfs():

    all_documents = []

    print("ROOT_DIR:", ROOT_DIR)
    print("DATA_DIR:", DATA_DIR)

    # SEARCH RECURSIVELY
    pdf_files = list(DATA_DIR.rglob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF files")

    # DEBUG
    for file in pdf_files:
        print("PDF FOUND:", file)

    if len(pdf_files) == 0:
        raise Exception(
            f"No PDF files found inside {DATA_DIR}"
        )

    for pdf_path in pdf_files:

        print(f"\nLoading: {pdf_path.name}")

        try:

            pdf = fitz.open(pdf_path)

            for page_num in range(len(pdf)):

                page = pdf[page_num]

                text = page.get_text()

                # VERY IMPORTANT
                if text and text.strip():

                    all_documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "paper": pdf_path.stem,
                                "page": page_num + 1,
                                "source": str(pdf_path)
                            }
                        )
                    )

        except Exception as e:

            print(f"ERROR loading {pdf_path.name}: {e}")

    print(f"\nLoaded {len(all_documents)} document pages")

    if len(all_documents) == 0:
        raise Exception(
            "PDFs were found but NO TEXT could be extracted."
        )

    return all_documents


# ---------------- RETRIEVER ---------------- #

class HierarchicalRetriever:

    def __init__(self, embeddings):

        self.embeddings = embeddings

        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2500,
            chunk_overlap=300
        )

        self.vectorstore = None

        self.docstore = InMemoryStore()

        self.ns = ("parents",)

    def add_documents(self, documents):

        parent_docs = self.parent_splitter.split_documents(documents)

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
            persist_directory=str(PERSIST_DIR),
            collection_name="research_papers",
        )

        self._persist_docstore()

        print(f"Indexed {len(parent_docs)} parents and {len(all_children)} children.")

        return all_children

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

            print("Loaded parent context from disk.")

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


# ---------------- INITIALIZE ---------------- #

raw_docs = load_all_pdfs()

hier_retriever = HierarchicalRetriever(embeddings)

if os.path.exists(PERSIST_DIR) and os.path.exists(DOCSTORE_PATH):

    print("Loading existing hierarchical index...")

    hier_retriever.vectorstore = Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=embeddings,
        collection_name="research_papers",
    )

    hier_retriever._load_docstore()

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    child_chunks = child_splitter.split_documents(raw_docs)

else:

    print("Creating new hierarchical index...")

    child_chunks = hier_retriever.add_documents(raw_docs)


bm25 = BM25Okapi(
    [doc.page_content.split() for doc in child_chunks]
)


# ---------------- RETRIEVAL ---------------- #

@observe()
def hybrid_retrieve(query, k=5):

    vector_parents = hier_retriever.get_relevant_documents(query)

    tokenized_query = query.split()

    scores = bm25.get_scores(tokenized_query)

    bm25_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:k]

    bm25_children = [child_chunks[i] for i in bm25_indices]

    combined = vector_parents + bm25_children

    return list({doc.page_content: doc for doc in combined}.values())


@observe()
def rerank(query, docs, top_n=3):

    texts = [doc.page_content for doc in docs]

    results = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=texts,
        top_n=top_n
    )

    return [
        docs[res.index]
        for res in results.results
        if res.relevance_score > 0.20
    ]


# ---------------- QUERY EXPANSION ---------------- #

@observe()
def expand_query(query):

    prompt = PROMPTS["query_expansion"].format(question=query)


    model = genai.GenerativeModel("gemini-3.5-flash")

    return model.generate_content(prompt).text.strip()


# ---------------- QA ---------------- #

@observe()
def ask_question(question):

    langfuse_context.update_current_trace(
        name="Research_Paper_RAG",
        user_id="siva_dev"
    )

    expanded_query = expand_query(question)

    retrieved_docs = hybrid_retrieve(expanded_query)

    docs = rerank(question, retrieved_docs)

    if not docs:

        return "The knowledge base does not contain this information.", []

    context = ""

    for doc in docs:

        paper = doc.metadata.get("paper", "Unknown")

        page = doc.metadata.get("page", "Unknown")

        context += f"\n--- Paper: {paper} | Page: {page} ---\n"

        context += doc.page_content + "\n\n"

    prompt = PROMPTS["rag_answer"].format(
        context=context,
        question=question
    )

   
    model = genai.GenerativeModel("gemini-3.5-flash")

    response = model.generate_content(prompt)

    answer_text = response.text

    return answer_text, docs


# ---------------- TERMINAL MODE ---------------- #

if __name__ == "__main__":

    print("AI Research Intelligence Agent Ready!")

    while True:

        query = input("\nAsk a question: ")

        if query.lower() == "exit":
            break

        answer, _ = ask_question(query)

        print("\n" + "=" * 80)

        print(answer)

        print("=" * 80)
