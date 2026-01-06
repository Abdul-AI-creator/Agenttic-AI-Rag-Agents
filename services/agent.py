import config  # Load environment variables first

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field
from typing import Literal

from services.tools import vector_search, calculator
from services.prompts import GRADE_PROMPT, ANSWER_PROMPT, REWRITE_PROMPT

llm = ChatAnthropic(
    model_name="claude-sonnet-4-20250514",
    temperature=0,
)

TOOLS = [vector_search, calculator]

class Grade(BaseModel):
    binary_score: str = Field(description="yes or no")

def agent_node(state: MessagesState):
    resp = llm.bind_tools(TOOLS).invoke(state["messages"])
    return {"messages": [resp]}

def grade_docs(state: MessagesState) -> Literal["answer", "rewrite"]:
    q = state["messages"][0].content
    ctx = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(question=q, context=ctx)
    result = llm.with_structured_output(Grade).invoke(
        [{"role": "user", "content": prompt}]
    )

    return "answer" if result.binary_score == "yes" else "rewrite"

def rewrite_question(state: MessagesState):
    q = state["messages"][0].content
    resp = llm.invoke(
        [{"role": "user", "content": REWRITE_PROMPT.format(question=q)}]
    )
    return {"messages": [resp]}

def generate_answer(state: MessagesState):
    q = state["messages"][0].content
    ctx = state["messages"][-1].content
    resp = llm.invoke(
        [{"role": "user", "content": ANSWER_PROMPT.format(question=q, context=ctx)}]
    )
    return {"messages": [resp]}

def build_agent():
    graph = StateGraph(MessagesState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("rewrite", rewrite_question)
    graph.add_node("answer", generate_answer)

    graph.add_edge(START, "agent")

    graph.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: END},
    )

    graph.add_conditional_edges("tools", grade_docs)
    graph.add_edge("rewrite", "agent")
    graph.add_edge("answer", END)

    return graph.compile()
