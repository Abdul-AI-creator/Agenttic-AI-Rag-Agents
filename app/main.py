from fastapi import FastAPI
import config  # Load environment variables
from app.routes import router

app = FastAPI(
    title="Agentic RAG API",
    description="RAG-powered Q&A agent using LangGraph and Claude",
    version="1.0.0"
)

app.include_router(router)

@app.get("/health")
def health_check():
    return {"status": "healthy"}
