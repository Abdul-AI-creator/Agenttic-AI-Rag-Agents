GRADE_PROMPT = """
Question: {question}
Context: {context}

Is the context relevant? Answer yes or no.
"""

ANSWER_PROMPT = """
Answer concisely (max 3 sentences).

Question: {question}
Context: {context}
"""

REWRITE_PROMPT = """
Improve this question for better semantic search:

{question}
"""
