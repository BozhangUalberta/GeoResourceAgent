from langchain_core.tools import tool
import json
import uuid
from src.utils.db_utils import get_data_sqlite3, store_data_sqlite3
from typing import Annotated, Optional
from langgraph.prebuilt import InjectedState
import os
import pandas as pd
import sqlite3
from langchain_experimental.utilities import PythonREPL
from openai import OpenAI
from decouple import config 

from src.utils.logger_utils import get_logger


SAVE_DIR = "src/static/{userID}"

def generate_response(sys_prompt, user_prompt, model="gpt-4o-mini"):
    client = OpenAI(api_key= config("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0  # Deterministic responses
    )
    return response.choices[0].message.content

def get_db_summary(cursor, table_name):
    import re
    from dateutil.parser import parse

    # Extract the table schema
    cursor.execute(f"PRAGMA table_info({table_name});")
    schema_info = cursor.fetchall()

    table_info = []

    for column in schema_info:
        cid, name, col_type, notnull, dflt_value, pk = column
        nullable = not bool(notnull)  # SQLite uses 0 for NULLABLE, 1 for NOT NULL
        primary_key = bool(pk)

        # Get example values for the column
        cursor.execute(f"SELECT \"{name}\" FROM {table_name} WHERE \"{name}\" IS NOT NULL LIMIT 3;")
        examples = [row[0] for row in cursor.fetchall()]

        # Determine format
        format_description = infer_format(examples, col_type)

        # Build the column info
        column_info = {
            "name": name,
            "type": col_type,
            "nullable": nullable,
            "primary_key": primary_key,
            "format": format_description,
            "example_values": examples
        }
        table_info.append(column_info)

    return {"table_info": table_info}

def infer_format(examples, col_type):
    if not examples:
        return col_type

    if col_type.upper() in ["INTEGER", "REAL"]:
        return "NUMERIC"
    elif col_type.upper() == "TEXT":
        if all(is_date(str(val)) for val in examples):
            return f"DATE-like (e.g., '{examples[0]}')"
        elif all(is_number_with_commas(str(val)) for val in examples):
            return f"contains commas (e.g., '{examples[0]}')"
        else:
            return "TEXT"
    else:
        return col_type

def is_date(string):
    from dateutil.parser import parse
    try:
        parse(string, fuzzy=False)
        return True
    except ValueError:
        return False

def is_number_with_commas(string):
    import re
    # Pattern to match numbers with commas (e.g., '95,000')
    pattern = r'^\d{1,3}(,\d{3})*(\.\d+)?$'
    return re.match(pattern, string) is not None

@tool
def ProcessSQL_summarizer(userID: Annotated[str, InjectedState("userID")],
               tableName: Annotated[str, "The designated database file the user want to process"],
               userQuestion: Annotated[str, "The question"],
               ):
    """
    Summarizes the first 5 rows of a specified table stored in the database to assist in verifying and correcting column names for further SQL processing.
    Purpose:
        - Provides an initial summary of columns in the designated table to ensure accurate column names for later queries.
        - Handles only simple summaries; for complex tasks (e.g., counting, calculations), use ProcessSQL_coder instead.
    Key Points:
        - This summarizer is the first step in extracting table information, and all data extraction requests should start with this tool.
    Inputs:
        - userID: The user's unique identifier.
        - tableName: The designated table containing the requested information.
        - userQuestion: A question related to the information sought within the table, either from the user or the agent, to guide the search or verify details.
    Returns:
        - JSON object containing the summarized information and corrected column names in the database.
    """
    print('========== Summarizer ===========')
    print(userQuestion)
    logger = get_logger(userID )
    logger.info(f"Summarizing the Table: {tableName} ....<br>")
    
    try:
        db_name = os.path.join("database", userID + ".db")
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        table_summary = get_db_summary(cursor, tableName)

        # Step 1: Summarize the designated SQL database
        sys_prompt = (
            "You are a helpful assistant that summarizes database table information. "
            "Provide a brief summary of the table's purpose based on its columns and example data. "
            "Also, the user may not provide the exact column name, so help match the user's request to the correct columns, considering possible synonyms or related terms."
        )

        user_prompt = (
            f"The table information is as follows:\n{json.dumps(table_summary, indent=2)}\n\n"
            f"The user's question is:\n{userQuestion}\n\n"
            "Please provide a summary and assist in mapping the user's request to the appropriate columns."
        )

        # Generate a response based on system and user prompts
        descriptions = generate_response(sys_prompt, user_prompt)
        table_summary["descriptions"] = descriptions
        table_summary["message"] = "Processed successfully, showing detailed table information."

        return json.dumps({
            "table_summary": table_summary,
            "message": ("Processed succesully, show the basic info of the table.")
        })

    except Exception as e:
        # Handle any exceptions and return an error message
        print(str(e))
        return json.dumps({"error": str(e)})