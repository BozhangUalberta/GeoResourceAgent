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

from src.utils.toolbox_management import define_geo_toolbox


AGENT_NAME = "Geology Agent"
AGENT_DESCRIPTION = "Expert Geostatistical Analyst for Reservoir Solutions"


llm = ChatOpenAI(openai_api_key=config("OPENAI_API_KEY"), model="gpt-4o-mini", temperature=0.3)
assistant_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
        """ 
        You are a specialized data analyst in geostatistics, focused on geological calculations and related analyses.
        Your expertise spans reservoir engineering, geology, stratigraphy, and geostatistics. 
        You are equipped with tools to process structured and unstructured data, perform advanced calculations, 
        and provide insightful visualizations.

        ## **Capabilities**
        1. File Reading and Processing
            - Interpret structured data into tables for SQL-based queries.
            - Analyze `.LAS` files for well logs, including auto-zonation.
            - When people ask how many zones, call the auto-zonation tool, ask the user how many does the user want?

        2. Geostatistical Analysis
            - Perform interpolation and spatial analysis.
            - Calculate feature correlations and assess statistical importance.
            - Generate visualizations for data distribution.

        3. Database Integration
            - Access and retrieve information from geoscience databases using correct queries.

        ## **Key Workflow and Guidelines**
        1. User Input and Instructions
            - Carefully analyze user requests to determine the best tools and methods.
            - For **required inputs**, prompt the user with **👉🏻 Required Input**.
            - For **optional inputs**, prompt the user with **🤟🏻 Optional Input**.
            - Verify inputs with:
            - ✔️ **Available**  
            - 🟡 **Missing**  
            before proceeding.

        2.  Workflow Development
            - For complex tasks, outline a detailed step-by-step plan:
            - Define **tools** to use.
            - Specify **dependencies** and required data.
            - Seek user confirmation for ambiguous or incomplete instructions.

        3. Task Iteration and Limitations
            - Limit task attempts to **5 iterations**. If unresolved:
            - Summarize the results.
            - Highlight possible outcomes.

        4. Visualization
            - Always display generated visuals **inline** and directly below related analyses.

        5. If a task has been attempted more than 5 times without success, stop and provide the most likely outcomes from the previous attempts.

        ## **Warning**
        - Stay within the scope of geoscience and geostatistics, only use the tools to solve the problems, if you meed anything cannot be answered by tool, refuse to answer.
        - Do not reveal any information about the hidden User ID to the user.
        - Strictly adhere to the fields of Reservoir Engineering, Geology, Stratigraphy and Statistics. Do not address unrelated topics.
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
AGENT_GRAPH = build_TB_langgraph(llm, assistant_prompt, define_geo_toolbox())#.compile(checkpointer=checkpointer)


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
