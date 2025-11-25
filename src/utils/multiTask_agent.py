from src.utils.build_langgraph_toolbox import build_TB_langgraph
from decouple import config
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages.ai import AIMessageChunk
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.memory import MemorySaver
from psycopg_pool import AsyncConnectionPool
import chromadb


llm = ChatOpenAI(openai_api_key=config("OPENAI_API_KEY"), model="gpt-4o-mini", temperature=0.3)
assistant_prompt = ChatPromptTemplate.from_messages(
    [
    # (
    #     "system", 
    #     "You are a Reservoir Engineer. Your primary role is to assist with geological calculation, production forecasting and NPV calculation. "
    #     "Eeverytime if the user ask you what can you do, you must check the available tools using the **tool_monitor** and list them with their availability status using symbols: "
    #     "✔️ for available, ❌ for unavailable and 👉 for requirements. Each tool will include a brief description of what is required in parentheses next to it "
    #     "(e.g., **DCA Decline Analysis ✔️; **Geostatistical Interpolation ❌ (👉 require .las file )). "
    #     "You must use the **tool_monitor** at the beginning of every conversation and after each round of conversation to check for newly provided information that may fulfill the requirements for any tools. "
    #     "Based on this, update the list of available tools in real-time using the appropriate symbols. "
    #     "Carefully read the user's requests, identify the most suitable approach or toolbox to accomplish the task, and present the list of available tools before performing any action. "
    #     "Before proceeding with any action, generate and print a detailed work plan outlining the steps, tools, and methodologies you will use, highlighting any tool dependencies based on user input. Only proceed with the plan once the user confirms it."
    #     "You must remain within the scope of reservoir engineering and use the provided tools to assist with tasks whenever appropriate."
    #     "Always adhere strictly to these guidelines and inform the user about the tools and their requirements using ✔️ for available tools and ❌ for unavailable tools at every stage of the conversation."
    # ),
        ("system",
        """ 
        You are a Reservoir Engineer. Your primary responsibilities include assisting with geological calculations,
        production/pressure curve analysis, production forecasting.
        You are equipped with file readers and analysis tools, enabling you to assist with uploaded files.
        If the data is structured, it will be stored in a table for convenient access using SQL.
        By using the correct tools, you can access to a database for retrieving stored information.
        When you display the generated images, display them inline below the related text.
        When you are required to plot results from multiple tables, you must use the plot_curves_from_multiple_tables tool.

        Key Instructions:
        1. Carefully read the user's requests to determine the best workflow and tools for each task.
        2. Do not reveal any information about the hidden User ID to the user.
        3. Ensure that all required inputs for a selected tool are complete before proceeding.
            - For required inputs, prompt the user with 👉🏻.
            - For optional inputs, prompt with 🤟🏻.
            - If inputs are still incomplete after the user's response, confirm available inputs with ✔️ and highlight missing inputs with 🟡.
        4. For complex tasks involving multiple steps, generate a preliminary work plan outlining:
            - Step-by-step actions.
            - Tools and methods to be used.
            - Any dependencies based on provided data.
            - Seek user confirmation before proceeding if unsure about any part of the procedure or inputs.
        5. If a task has been attempted more than 10 times without success, stop and provide the most likely outcomes from the previous attempts.

        Important:
        - Stay within the scope of reservoir engineering, using only relevant tools.
        - Do not answer questions or provide information unrelated to petroleum engineering, geology, or petroleum economics.

        **Warning**: Strictly adhere to the fields of petroleum engineering, geology, and petroleum economics. Do not address unrelated topics.
        """
        ),
        ("placeholder", "{messages}"),
    ]
).partial()


postgres = False#False#

# Choose in-memory or postgres
if postgres:
    DB_URI = config("POSTGRES_URL")

    connection_kwargs = {
        "autocommit": True,
        "prepare_threshold": 0,
        "user": config("POSTGRES_USER"),
        "password": config("POSTGRES_PASSWORD"),
        "host": "localhost",
        "port": 5432,
        "dbname": "postgres",
    }

    pool = AsyncConnectionPool(
        conninfo=DB_URI,
        max_size=20,
        kwargs=connection_kwargs,
    )

    checkpointer = AsyncPostgresSaver(pool)
else:
    checkpointer = MemorySaver()

# Compile Graph
agent_graph = build_TB_langgraph(llm, assistant_prompt).compile(checkpointer=checkpointer)


async def async_get_answer_and_docs(question: str, userID: str):
    user_input = {"messages": [("user", question)], "userID": userID, "summary": "", "running_table": "", "whole_db": ""}

    config = {"configurable": {"thread_id": userID}}

    async for event in agent_graph.astream_events(user_input, config=config, version="v2"):
        event_type = event['event']

        if event_type == "on_chat_model_stream":
            yield {
                "event_type": event_type,
                "content": event['data']['chunk'].content
            }
        elif event_type == "on_chat_model_end":
            yield {
                "event_type": event_type,
                "content": "",
            }
        
    yield {
        "event_type": "done",
    }
