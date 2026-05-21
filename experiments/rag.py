import argparse
import os
import shutil
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv()

# Path Setup
RAG_DIR = Path(__file__).resolve().parent
PERSIST_DIRECTORY = RAG_DIR / "chroma_db"
DOCUMENTS_DIRECTORY = RAG_DIR / "documents" # Adjusted to find docs relative to script
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL = "gemini-2.0-flash" # Stable version

def list_pdf_paths() -> list[Path]:
    if not DOCUMENTS_DIRECTORY.exists():
        DOCUMENTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    pdf_paths = sorted(DOCUMENTS_DIRECTORY.glob("*.pdf"))
    return pdf_paths

def build_documents(pdf_paths: list[Path]) -> list[Document]:
    documents: list[Document] = []
    for pdf_path in pdf_paths:
        reader = PdfReader(str(pdf_path))
        # Identify the scheme from filename for better retrieval context
        scheme_name = pdf_path.stem.replace("-", " ").title()
        
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "scheme": scheme_name,
                        "source": pdf_path.name,
                        "page": page_number,
                    },
                )
            )
    return documents

def build_chunks(documents: list[Document]) -> list[Document]:
    # Smaller chunks with overlap work better for policy documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_documents(documents)

@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

@lru_cache(maxsize=1)
def get_genai_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")
    return genai.Client(api_key=api_key)

def get_vectorstore(force_rebuild: bool = False) -> Chroma:
    embeddings = get_embeddings()
    if force_rebuild and PERSIST_DIRECTORY.exists():
        shutil.rmtree(PERSIST_DIRECTORY)

    if not PERSIST_DIRECTORY.exists() or not any(PERSIST_DIRECTORY.iterdir()):
        print("Indexing documents... please wait.")
        paths = list_pdf_paths()
        if not paths:
            raise FileNotFoundError(f"Place your PMKVY PDFs in {DOCUMENTS_DIRECTORY}")
        docs = build_documents(paths)
        chunks = build_chunks(docs)
        return Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(PERSIST_DIRECTORY),
        )
    
    return Chroma(persist_directory=str(PERSIST_DIRECTORY), embedding_function=embeddings)

def ask_question(question: str):
    vectorstore = get_vectorstore()
    # Retrieval
    docs = vectorstore.similarity_search(question, k=4)
    
    # Augmentation
    context_entries = []
    for doc in docs:
        source_info = f"[Source: {doc.metadata.get('scheme')} - Page {doc.metadata.get('page')}]"
        context_entries.append(f"{source_info}\n{doc.page_content}")
    
    context = "\n\n---\n\n".join(context_entries)
    
    prompt = f"""You are a PMKVY Policy Expert. Answer the question using ONLY the context provided. 
If the answer is not in the context, state that the information is not available in the provided documents.

Context:
{context}

Question: {question}

Answer:"""

    # Generation
    client = get_genai_client()
    response = client.models.generate_content(model=GENERATION_MODEL, contents=prompt)
    
    return response.text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    
    if args.rebuild:
        get_vectorstore(force_rebuild=True)
    else:
        while True:
            user_input = input("\nPMKVY Query: ").strip()
            if user_input.lower() in ["exit", "quit"]: break
            print("\n" + ask_question(user_input))