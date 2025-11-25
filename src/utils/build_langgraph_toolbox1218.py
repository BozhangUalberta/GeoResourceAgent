from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, trim_messages
from langchain_core.messages.tool import ToolMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode

from openai import OpenAI

from decouple import config
from typing import Annotated, Union, Literal, Optional
from typing_extensions import TypedDict

from src.utils.toolbox_management import define_toolbox, define_econ_toolbox, define_prod_toolbox
from src.utils.db_utils import get_db_json, get_all_db_json
from src.utils.memory_utils import buffer_msg
from src.integrations.tools.RAG_tools import retriever, rewrite, generate, grade_documents


def pretty_print_messages(messages, show_details=False):
    for i, message in enumerate(messages):
        # Print the message type (e.g., AIMessage, HumanMessage)
        message_type = type(message).__name__
        print(f"{message_type} {i+1}:")
        print(f"  Content: {message.content}")
        
        # if show_details:
        #     print(f"  Additional kwargs: {message.additional_kwargs}")
        #     print(f"  Response metadata: {message.response_metadata}")
        
        print()  # Print a blank line for better readability



def build_TB_langgraph(llm, assistant_prompt):
    # Initialize toolbox
    toolbox = define_toolbox()

    # Build mapping of tool names to categories and collect all tools
    tool_name_to_category = {}
    all_tools = []
    for category_name, tools in toolbox.items():
        for tool in tools:
            tool_name_to_category[tool.name] = category_name
            all_tools.append(tool)


    ###########################
    ### Define Assistant
    ###########################
    class State(TypedDict):
        messages: Annotated[list, add_messages]
        userID: str
        summary: str

    class Assistant:
        def __init__(self, runnable: Runnable):
            self.runnable = runnable

        def __call__(self, state: State, config: RunnableConfig):
            while True:
                # new_msg = buffer_msg(state, detect_length=10 , trim_length=5)
                new_msg = {"summary": [], "messages": []}
                # Inject DB info
                running_info = get_db_json(db_name=state["userID"], table_name="running")
                state = {
                    "messages": [AIMessage(content=running_info)] + [AIMessage(content=new_msg["summary"])] + add_messages(state["messages"], new_msg["messages"]),
                    "userID": state["userID"],
                    "summary": new_msg["summary"],
                }
                
                pretty_print_messages(state["messages"], show_details=True)
                # print(state["summary"])

                result = self.runnable.invoke(state)
                # If the LLM happens to return an empty response, we will re-prompt it
                # for an actual response.
                if not result.tool_calls and (
                    not result.content
                    or isinstance(result.content, list)
                    and not result.content[0].get("text")
                ):
                    messages = state["messages"] + [("user", "Respond with a real output.")]
                    state = {**state, "messages": messages}
                else:
                    break
            return {"messages": result}

    assistant_runnable = assistant_prompt | llm.bind_tools(all_tools)

    def handle_tool_error(state) -> dict:
        error = state.get("error")
        tool_calls = state["messages"][-1].tool_calls
        return {
            "messages": [
                ToolMessage(
                    content=f"Error: {repr(error)}\n please fix your mistakes.",
                    tool_call_id=tc["id"],
                )
                for tc in tool_calls
            ]
        }

    def create_tool_node_with_fallback(tools: list) -> dict:
        return ToolNode(tools).with_fallbacks(
            [RunnableLambda(handle_tool_error)], exception_key="error"
        )

    # Graph Build
    graph_builder = StateGraph(State)

    ###########################
    ### Define Nodes
    ###########################
    # Node: assistant
    graph_builder.add_node("assistant", Assistant(assistant_runnable))

    # Add nodes for each category dynamically
    # Node: geostats_tools, sql_tools, clean_tools ...
    for category_name, tools in toolbox.items():
        graph_builder.add_node(
            category_name,
            create_tool_node_with_fallback(tools)
        )

    # Add node for RAG
    # Node: retriever created in tools
    # Node: rewrite
    graph_builder.add_node("rewrite", rewrite)

    # Node: generate
    # graph_builder.add_node("generate", generate)


    ###########################
    ### Define Edges
    ###########################
    # Define Starting Point
    graph_builder.add_edge(START, "assistant")

    # Edge(optional): assistant -> xx_tools (if tools called)
    def route_tools(state: State) -> Union[str, Literal["__end__"]]:
        next_node = tools_condition(state)
        # If no tools are invoked, return to the user
        if next_node == END:
            return END
        ai_message = state["messages"][-1]
        first_tool_call = ai_message.tool_calls[0]
        tool_name = first_tool_call["name"]
        category_name = tool_name_to_category.get(tool_name)

        if not category_name:
            # Handle the case where the tool name is not in the mapping
            return END
        else:
            return category_name

    graph_builder.add_conditional_edges("assistant", route_tools)

    # Edge: xx_tools -> assistant
    for category_name in toolbox.keys():
        # Edge: retriever -> generate / rewrite
        if category_name == "RAG_retriever_node":
            graph_builder.add_conditional_edges("RAG_retriever_node", grade_documents)
        else:
            graph_builder.add_edge(category_name, "assistant")

        
    # Edge: generate -> assistant
    # graph_builder.add_edge("generate", "assistant")

    # Edge: rewrite -> assistant
    graph_builder.add_edge("rewrite", "assistant")


    # memory = MemorySaver()
    # graph = graph_builder.compile(
    #     checkpointer=memory,
    # )

    return graph_builder
    # return graph
