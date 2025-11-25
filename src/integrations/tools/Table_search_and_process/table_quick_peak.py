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
    client = OpenAI(api_key=config("OPENAI_API_KEY"))
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
    # Extract the table schema
    cursor.execute(f"PRAGMA table_info({table_name});")
    schema_info = cursor.fetchall()

    table_info = []

    for column in schema_info:
        cid, name, col_type, notnull, dflt_value, pk = column
        nullable = not bool(notnull)
        primary_key = bool(pk)

        # Get example values for the column
        cursor.execute(f"SELECT \"{name}\" FROM {table_name} WHERE \"{name}\" IS NOT NULL LIMIT 3;")
        examples = [row[0] for row in cursor.fetchall()]

        # Determine format
        format_description = infer_format(examples, col_type)

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
    pattern = r'^\d{1,3}(,\d{3})*(\.\d+)?$'
    return re.match(pattern, string) is not None

@tool
def table_quick_peak(
    userID: Annotated[str, InjectedState("userID")],
    tableName: Annotated[str, "The designated database table to be examined"],
    userQuestion: Annotated[str, "A general or clarifying question about the table"]
):
    """
    The `table_quick_peak` tool provides a quick overview of a database table, including column metadata 
    and the first and last 5 rows. It helps users understand the table's structure and content before 
    performing deeper analysis.

    Purpose:
        - Offers a preliminary glimpse into the table: the correct column names, data types, example values, 
          plus the first and last 5 rows.
        - Helps verify the existence and format of columns, guiding the user to ask more precise questions 
          or use more advanced tools (like table_deep_query) later.

    Returns:
        - JSON object containing:
            - Column metadata (types, nullable, primary keys, examples).
            - First and last 5 rows of the table.
            - A summary description to help answer the user's query.
    """

    print('========== Quick Peak ===========')
    print(userQuestion)
    logger = get_logger(userID)
    logger.info(f"Quickly peeking the Table: {tableName} ....<br>")

    try:
        db_name = os.path.join("database", userID + ".db")
        print(f"Name of the database: {db_name}")
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        print(f"Name of the table: {tableName}")
        table_summary = get_db_summary(cursor, tableName)

        print(f"Connection successful and the summary is: {table_summary}")

        # Fetch first and last 5 rows
        df = pd.read_sql_query(f'SELECT * FROM "{tableName}"', conn)
        first_5_rows = df.head(5).to_dict(orient='records')
        last_5_rows = df.tail(5).to_dict(orient='records')

        # Updated system prompt to mention first and last 5 rows are provided
        sys_prompt = (
            "You are a helpful assistant that summarizes database table information. "
            "You have access to the table's column schema and some sample data. "
            "Provide a brief summary of the table's purpose and content based on its columns and example data. "
            "The user may not know the exact column names, so help map their question to the correct columns, "
            "considering synonyms or related concepts.\n\n"
            "You have been given:\n"
            "- Table schema with column formats and example values.\n"
            "- The first 5 rows of the table.\n"
            "- The last 5 rows of the table.\n\n"
            "Use this information to produce a concise overview."
        )

        # Include first and last 5 rows in the user prompt for better context
        user_prompt = (
            f"Table schema and column details:\n{json.dumps(table_summary, indent=2)}\n\n"
            f"First 5 rows:\n{json.dumps(first_5_rows, indent=2)}\n\n"
            f"Last 5 rows:\n{json.dumps(last_5_rows, indent=2)}\n\n"
            f"The user's question is:\n{userQuestion}\n\n"
            "Please provide a summary of what this table might be about and how the user's question relates to its columns."
        )

        # Generate the descriptive summary
        descriptions = generate_response(sys_prompt, user_prompt)
        table_summary["descriptions"] = descriptions
        table_summary["first_5_rows"] = first_5_rows
        table_summary["last_5_rows"] = last_5_rows
        table_summary["message"] = "Processed successfully, showing detailed table information."

        return json.dumps({
            "table_summary": table_summary,
            "message": "Successfully retrieved table overview with first and last 5 rows."
        })

    except Exception as e:
        print(str(e))
        return json.dumps({"error": str(e)})
    finally:
        conn.close()
