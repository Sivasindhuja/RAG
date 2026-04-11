import pandas as pd
import os
import time
from pathlib import Path
from datasets import Dataset
from src.rag import ask_question 

from ragas.run_config import RunConfig
from ragas.metrics import (
    Faithfulness,
    AnswerCorrectness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

# --- Path Configuration ---
ROOT_DIR = Path(__file__).resolve().parent.parent
GOLDEN_DATASET_PATH = ROOT_DIR / "tests" / "golden_dataset.csv"

# --- Step 1: Evaluation Setup ---
# Using the same high-capability model (Gemma 3) as the "Judge"
eval_llm = ChatGoogleGenerativeAI(model="models/gemma-3-27b-it")
eval_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

ragas_llm = LangchainLLMWrapper(eval_llm)
ragas_emb = LangchainEmbeddingsWrapper(eval_embeddings)
run_config = RunConfig(max_workers=1, timeout=180)

def run_evaluation_pipeline():
    """
    Executes the full RAG evaluation cycle:
    1. Inference on Golden Dataset
    2. Ragas metric calculation
    3. CI/CD Threshold Gating
    """
    print(f" Starting Evaluation Pipeline using: {GOLDEN_DATASET_PATH.name}")
    
    # Step 2: Inference over Golden Dataset
    df = pd.read_csv(GOLDEN_DATASET_PATH)
    results = []

    # Head(5) is used for CI speed; in production/nightly, use the full set [cite: 10]
    for i, row in df.head(5).iterrows():
        question = row["question"]
        ground_truth = row["answer"]
        
        try:
            answer, docs = ask_question(question)
            contexts = [doc.page_content for doc in docs]
            
            results.append({
                "question": question, 
                "answer": answer,
                "contexts": contexts, 
                "ground_truth": ground_truth
            })
            print(f" Generated answer for Q{i+1}")
            
        except Exception as e:
            print(f" Error generating answer for Q{i+1}: {e}")
            continue
            
        # Rate limiting sleep to prevent API exhaustion during batch runs [cite: 79, 115]
        time.sleep(5) 

    # Step 3: Ragas Evaluation
    print("\n Running Ragas metrics calculation...")
    dataset = Dataset.from_dict({
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "ground_truth": [r["ground_truth"] for r in results]
    })

    eval_result = evaluate(
        dataset,
        metrics=[
            Faithfulness(llm=ragas_llm),
            AnswerCorrectness(llm=ragas_llm),
            LLMContextPrecisionWithReference(llm=ragas_llm),
            LLMContextRecall(llm=ragas_llm),
        ],
        embeddings=ragas_emb, 
        run_config=run_config
    )

  # Step 4: CI/CD Gatekeeper logic 
    # Extract lists of scores from the results dictionary 
    f_scores = eval_result["faithfulness"]
    r_scores = eval_result["context_recall"]
    c_scores = eval_result["answer_correctness"] # Extracting correctness

    # Calculate averages manually to avoid TypeError
    avg_faithfulness = sum(f_scores) / len(f_scores) if f_scores else 0
    avg_recall = sum(r_scores) / len(r_scores) if r_scores else 0
    avg_correctness = sum(c_scores) / len(c_scores) if c_scores else 0

    print(f"\n--- CI/CD Quality Report ---")
    print(f"Average Faithfulness: {avg_faithfulness:.4f}")
    print(f"Average Context Recall: {avg_recall:.4f}")
    print(f"Average Answer Correctness: {avg_correctness:.4f}")

    # Threshold gating: Focus on Faithfulness to prevent hallucinations [cite: 10, 11]
    THRESHOLD_FAITHFULNESS = 0.85
    
    if avg_faithfulness < THRESHOLD_FAITHFULNESS:
        print(f" FAIL: Faithfulness {avg_faithfulness:.2f} is below target {THRESHOLD_FAITHFULNESS}!")
        exit(1) # Build fails in GitHub Actions [cite: 11]
    else:
        print(f"PASS: Quality standards met.")
        exit(0) # Build passes [cite: 11]
if __name__ == "__main__":
    run_evaluation_pipeline()