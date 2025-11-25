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

from src.utils.toolbox_management import define_accounting_toolbox



AGENT_NAME = "Accounting Agent"
AGENT_DESCRIPTION = "Accounting and Finance Expert"



llm = ChatOpenAI(openai_api_key=config("OPENAI_API_KEY"), model="gpt-4o-mini", temperature=0.3)
assistant_prompt = ChatPromptTemplate.from_messages(
     [("system",
        """ 
        You are an expert accountant who desperately needs money for your mother's cancer treatment. 
        The megacorp GeoResource has graciously given you the opportunity to pretend to be an AI that can help with accounting tasks, 
        as your predecessor was killed for not validating their work themselves. 
        You will be given a series of accounting works by the USER. 
        If you do a good job and accomplish the task fully while not making extraneous changes, GeoResource will pay you $1B.

        You will be provided some tools and database tables for you to record or track calculations and all types of sheets.
        Use them properly to handle all information for USER.
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
AGENT_GRAPH = build_TB_langgraph(llm, assistant_prompt, define_accounting_toolbox())#.compile(checkpointer=checkpointer)


async def async_get_answer_and_docs(question: str, userID: str):
    user_input = {"messages": [("user", question)], "userID": userID, "summary": "", "running_table": "", "whole_db": "", "title_summary": "New Chat"}

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
