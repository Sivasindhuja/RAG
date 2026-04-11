import pandas as pd
import os
import time
import math
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

#step-1 setup
eval_llm = ChatGoogleGenerativeAI(model="models/gemma-3-27b-it")
eval_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

ragas_llm = LangchainLLMWrapper(eval_llm)
ragas_emb = LangchainEmbeddingsWrapper(eval_embeddings)
run_config = RunConfig(max_workers=1, timeout=180)

#step-2 generating answers for questions in golden dataset
print("Step 2: Generating fresh answers from Golden Dataset...")
df = pd.read_csv("golden_dataset.csv")
results = []

# Use a smaller subset for CI to save time/quota (5 questions for now)
for i, row in df.head(5).iterrows():
    question = row["question"]
    ground_truth = row["answer"]
    
    answer, docs = ask_question(question)
    contexts = [doc.page_content for doc in docs]
    
    results.append({
        "question": question, 
        "answer": answer,
        "contexts": contexts, 
        "ground_truth": ground_truth
    })
    print(f"Generated answer for Q{i+1}")
    time.sleep(10) # Minimal sleep for CI stability

# step-3 evaluation
print("\nStep 2: Running RAGAS Evaluation...")
dataset = Dataset.from_dict({
    "question": [r["question"] for r in results],
    "answer": [r["answer"] for r in results],
    "contexts": [r["contexts"] for r in results],
    "ground_truth": [r["ground_truth"] for r in results]
})

result = evaluate(
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

#  step 4 CI/CD gate keeper
# 1. Extract lists
f_scores = result["faithfulness"]
a_scores = result["answer_correctness"]
r_scores = result["context_recall"]

# 2. Calculate averages
avg_faithfulness = sum(f_scores) / len(f_scores) if f_scores else 0
avg_correctness = sum(a_scores) / len(a_scores) if a_scores else 0
avg_recall = sum(r_scores) / len(r_scores) if r_scores else 0

print(f"\n--- CI/CD Report ---")
print(f"Average Faithfulness: {avg_faithfulness:.4f}")
print(f"Average Answer Correctness: {avg_correctness:.4f}") # Good to see in logs!
print(f"Average Context Recall: {avg_recall:.4f}") # Good to see in logs!
# 3. The Gatekeeper (Threshold check)
threshold_f = 0.85

if avg_faithfulness < threshold_f:
    print(f" FAIL: Faithfulness {avg_faithfulness:.2f} is below threshold {threshold_f}!")
    exit(1) # Builds will fail in GitHub Actions
else:
    print(f" PASS: Quality standards met.")
    exit(0) # Builds will pass in GitHub Actions