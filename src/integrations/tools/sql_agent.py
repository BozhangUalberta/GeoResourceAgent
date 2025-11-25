from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from decouple import config 
from openai import OpenAI
import os
import json

from langchain_core.tools import tool
from typing import Annotated, List, Literal, Dict
from langgraph.prebuilt import InjectedState
from langchain_core.messages import SystemMessage
from langchain_community.agent_toolkits import SQLDatabaseToolkit

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

@tool
def sql_query(
    userID: Annotated[str, InjectedState("userID")],
    query_prompt: Annotated[str, f"Copy User original prompt. DO NOT MODIFY THE PROMPT!"],
):
    """
    Retrieve needed information from database if user prompts contains such direct requirements or hidden requirements for you to better analysis or assistant user.
    """
    try:
        # LLM
        llm = ChatOpenAI(model='gpt-4o-mini', api_key=config("OPENAI_API_KEY"))

        # Database
        db_name = os.path.join("database", userID + ".db")
        db_uri = "sqlite:///" + db_name
        db = SQLDatabase.from_uri(db_uri)

        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        tools = toolkit.get_tools()

        SQL_PREFIX = """You are an agent designed to interact with a SQL database.
        Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
        Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
        You can order the results by a relevant column to return the most interesting examples in the database.
        Never query for all the columns from a specific table, only ask for the relevant columns given the question.
        You have access to tools for interacting with the database.
        Only use the below tools. Only use the information returned by the below tools to construct your final answer.
        You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

        To start you should ALWAYS look at the tables in the database to see what you can query.
        Do NOT skip this step.
        Then you should query the schema of the most relevant tables."""

        system_message = SystemMessage(content=SQL_PREFIX)

        agent_executor = create_react_agent(llm, tools, messages_modifier=system_message)

        response = agent_executor.invoke({"messages": [HumanMessage(content=query_prompt)]})

        return json.dumps(f"SQL operation succeed with response={response}.")

    except Exception as e:
        return json.dumps(f"Tool 'sql_query' error with user prompt={query_prompt}. Here is error info: {e}")