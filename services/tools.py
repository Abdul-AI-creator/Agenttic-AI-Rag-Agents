from langchain_core.tools import tool
from db.chroma import get_vectorstore

retriever = get_vectorstore().as_retriever()

@tool
def vector_search(query: str) -> str:
    """Search knowledge base using semantic vector search."""
    docs = retriever.invoke(query)
    return "\n\n".join(d.page_content for d in docs)

@tool
def calculator(expression: str) -> str:
    """Evaluate math expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return str(e)
