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

@tool
def ProcessSQL_coder(userID: Annotated[str, InjectedState("userID")],
               tableName: Annotated[str, "The designated database file the user want to process"],
               userQuestion: Annotated[str, "The user's original question"],
               table_info: Annotated[str, "The summarized infomation form the sql summarizer"],
                validation_feedback: Annotated[Optional[str], "Feedback from validation step or None if no feedback"]  = None,
                previous_solution: Annotated[Optional[str], "Previous solution if available, or None if not applicable"]  = None,
               ):
    """
    This tool generates SQL query to extract information from the tables for the uploaded files.
    The generated SQL codes fullfuill user's requests. Always couple used with ProessSQL_summarizer.

    Inputs:
        - userID: The user's unique identifier.
        - tableName: The designated table containing the requested information
        - userQuestion: The user's original question indicating what they want to do.
        - table_info: What is this table about.
        - validation_feedback: Feedback from validation step, guiding query improvements or None if no feedback.
        - previous_solution: The last generated solution, if available, to ensure continuity in improvements.

    Returns:
        - JSON object containing the generted SQL query used to address the user's problem.
    """
    print('========== Coder ===========')
    print("question:", userQuestion)
    print("descriptions:", table_info)
    print("feedback:",validation_feedback)
    logger = get_logger(userID )
    logger.info("Writing a SQL query to check....<br>")

    try:
        db_name = os.path.join("database", f"{userID}.db")
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        # Fetch column names from the specified table
        cursor.execute(f"PRAGMA table_info({tableName})")
        column_info = cursor.fetchall()
        column_names = [column[1] for column in column_info]  # Extract column names from PRAGMA output

        conn.close()
        # Step 2A: Generate SQL to Process the Database
        sys_prompt = (
            "You are an expert SQL query generator tasked with creating SQL queries that fulfill the user's request based on the provided table information. "
            "First rule: generate the SQLite code only, without any additional explanation. "
            "Match the features in the user's request with the correct table columns using the detailed `table_info`, which includes column names, data types, formats, and example values. "
            "Use the column data types provided in the `table_info` to guide your query generation. "
            "Apply CAST operations if the data in a column is incorrectly labeled in `table_info`. For example, if numeric values such as '55,0' or '1,000' are stored as `TEXT`, use CAST to convert them to the correct numeric type for calculations or sorting."
            "Exclude NULL or None values when performing operations such as sorting, calculations, minimum, and maximum evaluations. "
            "To avoid issues when a columns names contain special characters (like commas), you must enclose them in double quotes (e.g., \"column name\") when referencing them in SQL queries."
            "If the task is related to production related to production data and the user **does not** specify gas, oil, or water, count all available production data."
            "Incorporate suggestions from `validation_feedback` and reference the `previous_solution` for improvement continuity if applicable."
        )

        user_prompt = (
            f"The table information is as follows: {table_info}\n"
            f"The correct column names in the tables are: {column_names}.\n"
            f"The user's requirement (question) is: {userQuestion}.\n"
            f"The table to be processed is named '{tableName}' in the database '{db_name}'.\n"
            f"Previous solution: {previous_solution if previous_solution else 'None available'}.\n"
            f"Validation feedback: {validation_feedback if validation_feedback else 'No feedback provided.'}"
        )

        sql_query = generate_response(sys_prompt, user_prompt, model = "gpt-4o")
        print(sql_query)
        sql_query = sql_query.strip()
        if sql_query.startswith("```") and sql_query.endswith("```"):
            sql_query = sql_query[sql_query.find('\n') + 1:-3]

        return json.dumps({
            "sql_query": sql_query,
            "message": "SQL quetry generated. "
        })

    except Exception as e:
        # Handle any exceptions and return an error message
        print(str(e))
        return json.dumps({"error": str(e)})