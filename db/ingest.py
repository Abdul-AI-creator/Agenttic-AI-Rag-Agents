import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from db.chroma import get_vectorstore

urls = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
]

def ingest():
    docs = []
    for url in urls:
        docs.extend(WebBaseLoader(url).load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
    )

    splits = splitter.split_documents(docs)

    vs = get_vectorstore()
    vs.add_documents(splits)
    vs.persist()

if __name__ == "__main__":
    ingest()
