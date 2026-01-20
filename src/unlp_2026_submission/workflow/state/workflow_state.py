from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class WorkflowState(TypedDict):
    query: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
