PROMPTS = {

    "query_expansion": """
Rewrite the user question into a better search query for retrieving information from a technical document.

The rewritten query should:
- Preserve the original meaning
- Include relevant technical terms if possible
- Be concise and optimized for search

Return ONLY the improved query text.

Question:
{question}
""",


    "rag_answer": """
You must answer ONLY using the provided context.

Rules:
- Do not use outside knowledge.
- If the context does not contain the answer, reply exactly:
"The document does not contain this information."
- Cite sources using the format [Source 1], [Source 2], etc.

Context:
{context}

Question:
{question}
"""
}