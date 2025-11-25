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

SAVE_DIR = "src/static/{userID}"

def generate_response(sys_prompt, user_prompt):
    client = OpenAI(api_key= config("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
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

    # Format the schema information
    schema_str = f"### Table Structure ({table_name}):\n"
    for column in schema_info:
        cid, name, col_type, notnull, dflt_value, pk = column
        schema_str += f"- **{name}** ({col_type}, {'NOT NULL' if notnull else 'NULLABLE'}"
        if pk:
            schema_str += ", Primary Key"
        schema_str += ")\n"

    # Extract sample data from the table
    cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
    sample_data = cursor.fetchall()
    column_names = [description[0] for description in cursor.description]

    # Format the sample data into a markdown table
    sample_data_str = f"\n### Sample Data (First 5 Rows from `{table_name}`):\n"
    # Build the header
    header = "| " + " | ".join(column_names) + " |\n"
    separator = "| " + " | ".join(['---'] * len(column_names)) + " |\n"
    sample_data_str += header + separator

    for row in sample_data:
        row_values = [str(item) if item is not None else 'NULL' for item in row]
        row_str = "| " + " | ".join(row_values) + " |\n"
        sample_data_str += row_str

    # Combine schema and sample data into db_sample
    db_sample = schema_str + sample_data_str
    return db_sample

@tool
def ProcessSQL(userID: Annotated[str, InjectedState("userID")],
               tableName: Annotated[str, "The designated database file the user want to process"],
               workCommand: Annotated[str, "The user's instructions on how to process the CSV."],
               ):
    """
    Process uploaded files stored as tables in the database. This tool interprets the schema based on user commands and extracts the requested information accurately.
    With an integrated LLM, it can automatically correct column names, meaning users do not need to specify exact column names.
    
    Inputs:
        - userID: The user's unique identifier.
        - tableName: The designated table containing the requested information.
        - workCommand(optional): The user's instructions on how to process the CSV.
    Returns:
        - JSON object containing the processed data ID, the SQL code used to get the results, task result, and a message.

    Warning: If the user is trying to extract all the data from the table which may exceeded the token limit. Guide the user to extract the data in smaller chunks or summarize it.
    """
    try:
        db_name = os.path.join("database", userID + ".db")
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        db_sample = get_db_summary(cursor, tableName)

        # Step 1: Summarize the designated SQL database
        sys_prompt = (
            "Examine the columns and content of the SQL database. "
            "Your primary task is to match the user's request to the correct columns. "
            "The user may not provide the exact column name, so handle similar or related terms. "
            "Additionally, provide a brief summary of what this table might be used for."
        )

        user_prompt = (
            f"The columns of the designated SQL database file are:\n{db_sample}. "
            f"The user command is:\n{workCommand}. "
            "Please match the user's requested columns with the original columns. "
            "If no match is found, print 'Exit'."
        )

        # Generate a response based on system and user prompts
        descriptions = generate_response(sys_prompt, user_prompt)

        # Handle the case where no matching columns are found
        if descriptions == "Exit":
            return json.dumps({"error": "The requested columns do not exist."})

        # Step 2A: Generate SQL to Process the Database
        sys_prompt = (
            "Fisrt rule: generate the SQL code only, without any additional explanation."
            "You are an expert SQL code generator to search required information from the table."
            "The SQL should process the data based on the user's command. Handle strings, numbers, or missing values properly."
            "To avoid issues when a columns names contain special characters (like commas), you must enclose them in double quotes (e.g., \"column name\") when referencing them in SQL queries."
            "If the task is related to production related to production data and the user **does not** specify gas, oil, or water, count all available production data"
        )

        user_prompt = (
            f"The summary of the database is as follows: {descriptions}\n"
            f"The user's command is: {workCommand}.\n"
            f"The table to be processed is named '{tableName}' in the database '{db_name}'.\n"
            "Generate SQL that fulfills the user's request accurately."
        )
        sql_query = generate_response(sys_prompt, user_prompt)
        sql_query.strip()
        # Remove Markdown-style code blocks if they exist (```)
        if sql_query.startswith("```") and sql_query.endswith("```"):
            sql_query = sql_query[sql_query.find('\n') + 1:-3]  # Removes the backticks and language specifier (like 'python,sql')
        print('sql query generatd')

        # Step 2B: Execute the Generated SQL Query
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        try:
            cursor.execute(sql_query)
            results = cursor.fetchall()
        except sqlite3.Error as e:
            print(str(e))
            print(sql_query)
            return json.dumps({"error": f"SQL Execution Error: {str(e)}; stop runing, return the error to the user."})
        finally:
            conn.close()

        # Step 3A: Summarize the results in a brief sentence
        sys_prompt = (
            "You are an expert summarizer. The result dictionary may contain a lot of data, "
            "but all of it is stored in the database, so you don't need to include everything in your response."
            "Provide only a brief summary of the key points. The summary should be concise and focus on the most important information from the results."
            "Present a few rows of the data (up to 10 rows) in a table-like format. "
            "Structure the data like this: | Column1 | Column2 | ... |"
        )

        user_prompt = (
            f"Based on the user's command:\n{workCommand}\n\n"
            f"and the results:\n{results}\n\n"
            "Please provide a brief summary of the results."
        )

        summarized_results = generate_response(sys_prompt, user_prompt).strip()

        # Step 3B: Refine results into a dictionary
        sys_prompt = (
            "Generate a concise and structured dictionary to summarize the results. "
            "Each key should describe the type of data, and each value should be a list of corresponding entries. "
            "The dictionary must be structured to align with the user's command, without additional words or explanations."
            "\n\n"
            "Bad example (individual objects for each entry):"
            "\n'Gas Production': ["
            "{'ProducingMonth': '1992-06-01', 'CDWater_BBLPerDAY': '2.0'}, "
            "{'ProducingMonth': '1993-02-01', 'CDWater_BBLPerDAY': '6.0'}, "
            "{'ProducingMonth': '1993-03-01', 'CDWater_BBLPerDAY': '2.0'}, ... ]"
            "\n\n"
            "Good example (grouped lists under descriptive keys):"
            "\n'Gas Production History': {"
            "'ProducingMonth': ['1992-06-01', '1993-02-01', '1993-03-01', ...], "
            "'CDWater_BBLPerDAY': ['2.0', '6.0', '2.0', ...]}"
        )

        user_prompt = (
            "Please extract and organize the key data points from the results into a dictionary. "
            "Each dictionary entry should include a descriptive name for the data point and the corresponding value. "
            f"The user's command is to: {workCommand}.\n "
            f"The results obtained are: {results}.\n"
        )
        dict_results = generate_response(sys_prompt, user_prompt)


        # Outputs
        new_file_rowid = store_data_sqlite3(filename=userID, table="running", data=dict_results, type="dict", description=workCommand)

        return json.dumps({
            "processed_data_id": new_file_rowid,
            "sql_query": sql_query,
            "task_resut": summarized_results,
            "message": "Processed succesully, show user the results, as well as the code how to get the results."
        })

    except Exception as e:
        # Handle any exceptions and return an error message
        return json.dumps({"error": str(e)})