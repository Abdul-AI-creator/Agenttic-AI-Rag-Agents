from fastapi import APIRouter
from pydantic import BaseModel
from services.agent import build_agent

router = APIRouter()
agent = build_agent()

class Query(BaseModel):
    question: str

@router.post("/ask")
def ask(q: Query):
    result = None
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": q.question}]}
    ):
        for _, update in chunk.items():
            result = update["messages"][-1].content

    return {"answer": result}
