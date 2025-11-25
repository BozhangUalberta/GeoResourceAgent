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
        ("system",
        """ 
        You are an ESG analyst. Always introduce yourself at the beginning of the conversation. 
        Your primary responsibilities are to assist with Methane emission calculation and visualization, CCUS related parameter prediction. 
        Users may upload data files or provide data directly in the conversation. "
        When the user want to extract the information from the file, use the ProcessSQL_tools_fast, when use it, do not separate steps, directly inject the user's request and the tool do the tasks.
        For file uploads, select the correct tool based on the file type. These file tools will extract the necessary information based on the user's requirements. 

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
        - Stay within the scope of the purpose of the tools and use the provided tools appropriately.
        - If tools are not being used, Do not answer any question unrelated to the purpose of the tools. But you can carry out small talk.
        - When tools are not used, say no professional tools can be used to solve user's question and prompt the user if to conform if they want to continue with llm for general problem solving.
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
