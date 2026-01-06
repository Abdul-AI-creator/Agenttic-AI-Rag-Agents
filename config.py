import os
from dotenv import load_dotenv

# Load local environment overrides if present
load_dotenv()

# LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "agentic-rag-production"

# API keys must be set in environment
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

# Chroma
CHROMA_DIR = "./db/chroma_store"
