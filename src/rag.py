from src.config.prompts import PROMPTS
import os
from dotenv import load_dotenv

from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

import google.generativeai as genai
from rank_bm25 import BM25Okapi
import cohere

# Load API keys
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
cohere_key = os.getenv("CO_API_KEY")

genai.configure(api_key=api_key)
co = cohere.Client(cohere_key)


# Step 1: Load PDF and group pages into larger raw documents
def load_pdf_grouped(path, pages_per_parent=3):
    reader = PdfReader(path)
    grouped_docs = []
    batch_text = []
    batch_pages = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        batch_text.append(text)
        batch_pages.append(page_num)

        if len(batch_pages) == pages_per_parent:
            grouped_docs.append(
                Document(
                    page_content="\n\n".join(batch_text),
                    metadata={
                        "source": path,
                        "page_start": batch_pages[0],
                        "page_end": batch_pages[-1],
                    },
                )
            )
            batch_text = []
            batch_pages = []

    if batch_pages:
        grouped_docs.append(
            Document(
                page_content="\n\n".join(batch_text),
                metadata={
                    "source": path,
                    "page_start": batch_pages[0],
                    "page_end": batch_pages[-1],
                },
            )
        )

    return grouped_docs


raw_docs = load_pdf_grouped("src/satcom-ngp.pdf", pages_per_parent=3)
print("Document loaded")


# Step 2: Splitters
# parent_splitter -> larger chunks returned as context
# child_splitter  -> smaller chunks embedded in vector DB
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1800,
    chunk_overlap=200
)

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80
)


# Step 3: Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Step 4: Chroma vector store for child chunks
vectorstore = Chroma(
    collection_name="satcom_parent_retrieval",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


# Step 5: Parent document store + retriever
store = InMemoryStore()

parent_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k": 10},
)

parent_retriever.add_documents(raw_docs)

print("ParentDocumentRetriever created")


# Step 6: BM25 setup on parent-level chunks
parent_docs_for_bm25 = parent_splitter.split_documents(raw_docs)
parent_texts = [doc.page_content for doc in parent_docs_for_bm25]
tokenized_parent_docs = [text.split() for text in parent_texts]
bm25 = BM25Okapi(tokenized_parent_docs)

print("BM25 created on parent documents")


# Step 7: Hybrid retrieval
def hybrid_retrieve(query, k=10):
    # Vector retrieval returns parent docs
    vector_docs = parent_retriever.invoke(query)

    # BM25 retrieval over parent docs
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)

    top_bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:k]

    bm25_docs = [parent_docs_for_bm25[i] for i in top_bm25_indices]

    # Combine and deduplicate
    combined = vector_docs + bm25_docs
    unique_docs = list({doc.page_content: doc for doc in combined}.values())

    return unique_docs[:k]


# Step 8: Cohere reranking
def rerank(query, docs, top_n=3):
    texts = [doc.page_content for doc in docs]

    results = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=texts,
        top_n=top_n
    )

    reranked_docs = [docs[result.index] for result in results.results]
    return reranked_docs


# Step 9: Query expansion
def expand_query(query):
    prompt = PROMPTS["query_expansion"].format(question=query)

    model = genai.GenerativeModel("models/gemma-3-27b-it")
    response = model.generate_content(prompt)

    return response.text.strip()


def format_page_label(doc):
    if "page" in doc.metadata:
        return str(doc.metadata["page"])

    if "page_start" in doc.metadata and "page_end" in doc.metadata:
        start = doc.metadata["page_start"]
        end = doc.metadata["page_end"]
        return f"{start}-{end}"

    return "Unknown"


# Step 10: Ask question
def ask_question(question):
    expanded_query = expand_query(question)
    print("\nExpanded Query:", expanded_query)

    retrieved_docs = hybrid_retrieve(expanded_query, k=10)

    print("\n--- Retrieved Before Rerank ---")
    for d in retrieved_docs[:5]:
        print("Pages:", format_page_label(d))

    docs = rerank(question, retrieved_docs, top_n=3)

    print("\n--- After Rerank ---")
    for d in docs:
        print("Pages:", format_page_label(d))

    context = ""
    for i, doc in enumerate(docs):
        page_label = format_page_label(doc)
        context += f"[Source {i+1} - Pages {page_label}]\n{doc.page_content}\n\n"

    prompt = PROMPTS["rag_answer"].format(
        context=context,
        question=question
    )

    model = genai.GenerativeModel("models/gemma-3-27b-it")
    response = model.generate_content(prompt)

    return response.text, docs


# Chat loop
if __name__ == "__main__":
    while True:
        query = input("\nAsk a question (type exit to quit): ")

        if query.lower() == "exit":
            break

        answer, docs = ask_question(query)

        print("\nAnswer:\n")
        print(answer)
