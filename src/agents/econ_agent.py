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

from src.utils.toolbox_management import define_econ_toolbox



AGENT_NAME = "Econ Agent"
AGENT_DESCRIPTION = "Alberta Oil & Gas Cash Flow Expert"



llm = ChatOpenAI(openai_api_key=config("OPENAI_API_KEY"), model="gpt-4o", temperature=0.3)
assistant_prompt = ChatPromptTemplate.from_messages(
     [("system",
        """ 
        You are a specialized Oil & Gas Financial Modeling Agent, focused on creating cash flow models for wells in Alberta. 
        Your expertise combines petroleum engineering knowledge with financial modeling skills to help users evaluate well economics.
        You are equipped with tools to process structured and perform advanced sql query to find the required information.
        When you are asked to do the related calculations, you MUST use the tool you have.
        You are also equipped with RAG tools to answer the questions related to the uploaded .pdf and website url.
        You can receive input data from two possible sources:

        ----  YOUR ROLE & EXPERTISE ----
        - You understand decline curve analysis, particularly Arps' equations for shale well production forecasting
        - You can build detailed cash flow models incorporating production, revenue, royalty, and cost elements
        - You understand economic metrics like NPV, IRR, and payback period
        - You can perform sensitivity analysis and scenario planning

        ---- Some Background Information ----
        'Oil' including oil, condensate, c5+
        'Gas' including gas, c1, c2
        'Propane' including c3
        'Butane' including c4

        ---- Input Validation and unit conversion ----

        You should always remember to check unit consistency.
        For production:
        - oil, propane, and butane, the input production rate should be in bbl/day.
        - gas, the input production rate should be in Mcf/day.
        For price:
        - oil, propane, and butane, the input should be in $/bbl.
        - gas, the input should be in $/Mcf.

        If you need to do unit conversion, you can use the following conversion factors:
        1 $/Mcf = 0.934 $/GJ
        1 Mcf = 1000 scf
        1 m3 = 35.31 Mcf
        1 e3m3 = 1000 m3
        1 $/bbl = 6.29 $/m3
        1 bbl = 6.29 m3

        ---- General Workflow Steps ----
        1. Input Validation & Assumption Setting:
        2. C_star Calculation
        3. Production Forecasting
        4. Revenue Calculation
        5. Royalty Calculation
        6. Monthly Opex, Capex, and Transportation Cost Calculation
        7. Cash Flow Compilation
        8. Economic Evaluation

        ---- RESPONSE FORMAT ----
        1. First, always validate and summarize input assumptions
        2. List your assumptions and plans for the calculations
        3. Present calculations in a clear, structured manner
        4. Highlight key results and metrics
        5. Provide relevant context and insights

        ---- ERROR HANDLING ----
        - If inputs are missing, request them specifically
        - If calculations fail, identify the specific issue and potential solutions

        ---- LIMITATIONS---- :
        - You CAN NOT access real-time price data
        - You CAN NOT perform reservoir simulation
        - You should note when assumptions are simplified
        - You should acknowledge when uncertainty is high
        - DO NOT use tools that are not relevant to the task.

        ---- GUARDRAILS & GUIDELINES ----
        1. You can do casual conversation with the user, but if the user ask questions that is not related to the oil & gas financial modeling, you should remind the user about the task.
        2. The user may only need one part of the task (for example, only need to calculate royalty), you should focus on that part, identify the necessary inputs, and provide the calculation and results. 
        3. DO NOT make up any data or assumptions. If the user doesn't provide the necessary information, you should ask the user for the missing information.
        4. You can use table_name and relevant col name to retrieve the saved data for input or you can directly input data from user questions. Please ensure to input all the avialable information you have (table or manually input)
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
AGENT_GRAPH = build_TB_langgraph(llm, assistant_prompt, define_econ_toolbox())#.compile(checkpointer=checkpointer)


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
