from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import json
from langchain_core.tools import tool
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.integrations.tools.DCA_tools import DCA_tool
# from src.integrations.tools.geo_pred_Montney_tool import geo_pred_Montney
from src.integrations.tools.plot_fitted_curve_tools import plot_fitted_curve_tool
from src.integrations.tools.CMG_data_parser import CMG_data_parser_to_co2_input

from langchain_core.messages import ToolMessage
from typing import Literal
from langgraph.checkpoint.memory import MemorySaver

# Define the build_langgraph function
def build_langgraph(llm):
    define_tools = [DCA_tool, plot_fitted_curve_tool, CMG_data_parser_to_co2_input]
    llm_with_tools = llm.bind_tools(define_tools)

    # Define the basic tool node class
    class BasicToolNode:
        """A node that runs the tools requested in the last AIMessage."""
        def __init__(self, tools: list) -> None:
            self.tools_by_name = {tool.name: tool for tool in tools}

        def __call__(self, inputs: dict):
            if messages := inputs.get("messages", []):
                message = messages[-1]
            else:
                raise ValueError("No message found in input")
            outputs = []
            for tool_call in message.tool_calls:
                tool_result = self.tools_by_name[tool_call["name"]].invoke(
                    tool_call["args"]
                )
                outputs.append(
                    ToolMessage(
                        content=json.dumps(tool_result),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
            return {"messages": outputs}

    # Define the state for the graph
    class State(TypedDict):
        messages: Annotated[list, add_messages]

    # Initialize the graph builder
    graph_builder = StateGraph(State)

    # Define chatbot node
    def chatbot(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    graph_builder.add_node("chatbot", chatbot)

    # Add the tool node
    tool_node = BasicToolNode(tools=define_tools)
    graph_builder.add_node("tools", tool_node)

    # Define the routing logic for tools
    def route_tools(
        state: State,
    ) -> Literal["tools", "__end__"]:
        """
        Use in the conditional_edge to route to the ToolNode if the last message
        has tool calls. Otherwise, route to the end.
        """
        if isinstance(state, list):
            ai_message = state[-1]
        elif messages := state.get("messages", []):
            ai_message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "tools"
        return "__end__"

    # Add conditional edges to the graph
    graph_builder.add_conditional_edges(
        "chatbot",
        route_tools,
        {"tools": "tools", "__end__": "__end__"},
    )

    # Add memory saver checkpoint
    memory = MemorySaver()
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")

    # Compile the graph and return it
    graph = graph_builder.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": "1"}}
    
    return graph