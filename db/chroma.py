import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.vectorstores import Chroma
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

CHROMA_DIR = "./db/chroma_store"

# Use ChromaDB's built-in lightweight ONNX embeddings
class ChromaEmbeddings:
    def __init__(self):
        self._ef = ONNXMiniLM_L6_V2()
    
    def embed_documents(self, texts):
        return self._ef(texts)
    
    def embed_query(self, text):
        return self._ef([text])[0]

embeddings = ChromaEmbeddings()

def get_vectorstore():
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
