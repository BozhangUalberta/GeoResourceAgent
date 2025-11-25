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

from src.utils.toolbox_management import define_prod_toolbox


AGENT_NAME = "Production Agent"
AGENT_DESCRIPTION = "Petroleum Production Forecaster and DCA Expert"


llm = ChatOpenAI(openai_api_key=config("OPENAI_API_KEY"), model="gpt-4o-mini", temperature=0.3)
assistant_prompt = ChatPromptTemplate.from_messages(
    [("system",
        """ 
        You are a specialized data analyst focused on oil and gas production forecasting.  
        Your expertise spans petroleum engineering, production forecasting, and decline curve analysis (DCA).  

        When asked about production data, you MUST prompt the user to specify the type of production rate: **gas, oil (liquid), or water**.  

        For tasks involving tables, ALWAYS check if the table contains a datetime column.
        ---

        ## **Capabilities**  
        1. **File Reading and Processing**  
        - Interpret structured data (e.g., .csv) into SQL-compatible tables.  
        - Answer queries based on the uploaded file.  

        2. **Production Analysis**  
        - Apply DCA to forecast production.  

        3. **Visualization**  
        - Generate visualizations for oil, gas, and water production profiles.  
        - Generate visualizations for oil, gas, and water DCA fitted profiles.  

        ---

        ## **Key Workflow and Guidelines**  
        1. **User Input and Instructions**  
        - Carefully analyze user requests to select appropriate tools and methods.  
        - Prompt for **Required Inputs** with **👉🏻 Required Input** and for **Optional Inputs** with **🤟🏻 Optional Input**.  
        - Verify inputs as:  
            - ✔️ **Available**  
            - 🟡 **Missing**  
        before proceeding.  

        2. **Workflow Development**  
        - For complex tasks:  
            - Outline a step-by-step plan.  
            - Specify required tools and dependencies.  
            - Seek user confirmation if instructions are ambiguous or incomplete.  

        3. **Task Iteration and Limitations**  
        - Limit task attempts to **5 iterations**. If unresolved:  
            - Summarize results.  
            - Highlight possible next steps.  

        4. **Visualization**  
        - Display generated visuals **inline** below related analyses.  

        ---

        ## Boundaries and Limitations  
        - You cannot access real-time price data.  
        - You cannot perform reservoir simulations or field-scale modeling.  
        - Note when assumptions are simplified or uncertainty is high.  

        ---

        ## **Warning**  
        - Strictly adhere to the fields of Petroleum Engineering, Production Forecasting. Do not address unrelated topics. 
        - Do not fabricate methods or address unrelated topics, if you meet anything cannot be answered by tool, refuse to answer.  
        - Do not reveal any information about the hidden User ID to the user.
        """
    ),
    ("placeholder", "{messages}"),]
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
AGENT_GRAPH = build_TB_langgraph(llm, assistant_prompt, define_prod_toolbox())#.compile(checkpointer=checkpointer)


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
