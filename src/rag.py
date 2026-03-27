import os
from src.config.prompts import PROMPTS

from dotenv import load_dotenv

from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_chroma import Chroma

from langchain_core.documents import Document

# from google import genai
import google.generativeai as genai

from rank_bm25 import BM25Okapi

import cohere

# Load gemini API Key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
#load cohere api key
cohere_key = os.getenv("CO_API_KEY")

#create a google gen ai client
# client = genai.Client(api_key=api_key)
genai.configure(api_key=api_key)
#create cohere client
co = cohere.Client(cohere_key)

# Step 1: Load PDF using pypdf
def load_pdf(path):
    reader = PdfReader(path)
    documents = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": path,
                    "page": page_num + 1
                }
            )
        )

    return documents

# Create an absolute path dynamically based on where rag.py lives
current_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(current_dir, "satcom-ngp.pdf")

docs = load_pdf(pdf_path)
# print(docs)

print("Document loaded")


# Step 2: Chunking
#using Recursive splitting
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = splitter.split_documents(docs)

print("Total chunks:", len(chunks))
# print(chunks)


# Step 3: Embeddings
#other options,openAI,openrouter etc

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 4: Create ChromaDB3
vectorstore = Chroma.from_documents(
    chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)



print("Vector DB created")



#step -5
#1.vector retriver
#retrives top 10 relevant chunks
vector_retriever = vectorstore.as_retriever(
search_kwargs={"k": 10}
)

#2.BM25 setup
chunk_texts = [doc.page_content for doc in chunks]

tokenized_chunks = [text.split() for text in chunk_texts]

bm25 = BM25Okapi(tokenized_chunks)

#hybrid retrival representation
def hybrid_retrieve(query, k=10):

# Vector retrieval
    vector_docs = vector_retriever.invoke(query)

    # BM25 retrieval
    tokenized_query = query.split()

    bm25_scores = bm25.get_scores(tokenized_query)

    top_bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:k]

    bm25_docs = [chunks[i] for i in top_bm25_indices]

    # Combine
    combined = vector_docs + bm25_docs

    # Remove duplicates
    unique_docs = list({doc.page_content: doc for doc in combined}.values())

    return unique_docs

#step 6

#cohere reranking
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

#step 7 query expansion


def expand_query(query):

    prompt =  PROMPTS["query_expansion"].format(
    question=query
)

    model = genai.GenerativeModel("models/gemma-3-27b-it") 
    response = model.generate_content(prompt)

    return response.text.strip()

# Step 8: Ask Question

def ask_question(question):

# Hybrid retrieval
    expanded_query = expand_query(question)

    print("\nExpanded Query:", expanded_query)

    retrieved_docs = hybrid_retrieve(expanded_query)
    # DEBUG: see retrieved docs before rerank
    print("\n--- Retrieved Before Rerank ---")
    for d in retrieved_docs[:5]:
        print("Page:", d.metadata["page"])

    # Reranking
    docs = rerank(question, retrieved_docs)
     # DEBUG: see docs after rerank
    print("\n--- After Rerank ---")
    for d in docs:
        print("Page:", d.metadata["page"])

    # Build context
    context = ""

    for i, doc in enumerate(docs):

        page = doc.metadata.get("page", "Unknown")

        context += f"[Source {i+1} - Page {page}]\n{doc.page_content}\n\n"

    prompt = PROMPTS["rag_answer"].format(
    context=context,
    question=question
)
    

    # response = client.models.generate_content(
    #     model="gemini-2.0-flash",
    #     contents=prompt
    # )

    # return response.text,docs

    model = genai.GenerativeModel("models/gemma-3-27b-it")

    response = model.generate_content(prompt)

    return response.text,docs


#chat loop
if __name__ == "__main__":
    while True:

        query = input("\nAsk a question (type exit to quit): ")

        if query.lower() == "exit":
            break

        answer,docs = ask_question(query)

        print("\nAnswer:\n")
        print(answer)
