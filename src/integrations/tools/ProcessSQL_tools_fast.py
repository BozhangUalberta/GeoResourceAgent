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
    return db_sample, column_names

def get_dict_summary(dict_results):
    # Determine the number of rows in dict_results
    num_rows = len(next(iter(dict_results.values())))  # Get the length of any column's data list

    # Create dict_sample based on the number of rows
    if num_rows <= 10:
        dict_sample = dict_results  # Use all rows
    else:
        # Take the first 10 rows for each column
        dict_sample = {key: value[:10] for key, value in dict_results.items()}

    # Convert dict_sample into a table-like format
    columns = list(dict_sample.keys())
    rows = zip(*dict_sample.values())

    table_lines = []

    # Create table header
    header = '| ' + ' | '.join(columns) + ' |'
    separator = '| ' + ' | '.join(['---'] * len(columns)) + ' |'
    table_lines.append(header)
    table_lines.append(separator)

    # Add rows
    for row in rows:
        table_lines.append('| ' + ' | '.join(str(item) for item in row) + ' |')

    table_str = '\n'.join(table_lines)

    return table_str

def convertSQL2Dict(column_names, results):
    return {column: [row[i] for row in results] for i, column in enumerate(column_names)}

@tool
def ProcessSQL(userID: Annotated[str, InjectedState("userID")],
               tableName: Annotated[str, "The designated database file the user want to process"],
               userQuestion: Annotated[str, "The user's original question"],
               ):
    """
    Process uploaded files stored as tables in the database. This tool interprets the schema based on user commands and extracts the requested information accurately.
   
    You must be aware:
        - It can automatically correct column names, meaning users do not need to specify exact column names.
        - When you use this tool, directly pass the user's original requests to it, don't separate into steps, do not modifiy the meaning of question, and never give recommendation how to do.
    
    Inputs:
        - userID: The user's unique identifier.
        - tableName: The designated table containing the requested information.
        - userQuestion: The user's original question indicating what they want to do.
    Returns:
        - JSON object containing the processed data ID, the SQL code used to get the results, task result, and a message.

    Warning: If the user is trying to extract all the data from the table which may exceeded the token limit. Guide the user to extract the data in smaller chunks or summarize it.
    """
    print(userQuestion)
    try:
        db_name = os.path.join("database", userID + ".db")
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        db_sample, column_names = get_db_summary(cursor, tableName)

        # Step 1: Summarize the designated SQL database
        sys_prompt = (
            "Examine the columns and content of the SQL database. "
            "Your primary task is to match the user's request to the correct columns. "
            "The user may not provide the exact column name, so handle similar or related terms. "
            "Additionally, provide a brief summary of what this table might be used for."
        )

        user_prompt = (
            f"The columns of the designated SQL database file are:\n{db_sample}. "
            f"The user's requirement (question) is:\n{userQuestion}. "
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
            "You are an expert SQLlite code generator to search required information from the table."
            "Generate SQL that fulfills the user's request accurately."
            "Fisrt rule: generate the SQLlite code only, without any additional explanation."
            "You must match the columns in the request with the correct table columns contained in the column names."
            "The values in the tables are stored as str, so you **must** CAST them to float when sorting."
            "Exclude None values when performing operations such as sorting, calculations, minimum, and maximum evaluations."
            "When calculating percentile, do not use NTILE if is necessary, when NTILE to calcuate the percentile, using MAX not MIN."
            "To avoid issues when a columns names contain special characters (like commas), you must enclose them in double quotes (e.g., \"column name\") when referencing them in SQL queries."
            "If the task is related to production related to production data and the user **does not** specify gas, oil, or water, count all available production data."
        )

        user_prompt = (
            f"The summary of the database is as follows: {descriptions}\n"
            f"The correct column names: {column_names}; you must select the column names in this list.\n"
            f"The user's requirement (question) is: {userQuestion}.\n"
            f"The table to be processed is named '{tableName}' in the database '{db_name}'.\n"
        )
        sql_query = generate_response(sys_prompt, user_prompt, model = "gpt-4o-mini")
        print(sql_query)
        sql_query.strip()

        # Remove Markdown-style code blocks if they exist (```)
        if sql_query.startswith("```") and sql_query.endswith("```"):
            sql_query = sql_query[sql_query.find('\n') + 1:-3]  # Removes the backticks and language specifier (like 'python,sql')

        # Step 2B: Execute the Generated SQL Query
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        try:
            cursor.execute(sql_query)
            results = cursor.fetchall()
            column_names = [description[0] for description in cursor.description]
        except sqlite3.Error as e:
            print(str(e))
            print(sql_query)
            return json.dumps({"error": f"SQL Execution Error: {str(e)}; stop running, return the error to the user."})
        finally:
            conn.close()

        # Step 3A: Write Python code to convert SQL query results into a dictionary
        # The benifit is the data do not need to go into the LLM
        sys_prompt = (
            "First rule: Generate only Python code without any additional explanation.\n"
            "You are an expert Python programmer.\n"
            "Your task is to write a Python function named `convertSQL2Dict` that takes two inputs:\n"
            "- `column_names`: a list of column names (strings),\n"
            "- `results`: a list of tuples representing the rows returned from an SQL query.\n"
            "The function should convert `results` into a dictionary where each key is a column name,\n"
            "and each value is a list of all the corresponding entries from the SQL result set.\n"
            "The function must have the following signature:\n"
            "`def convertSQL2Dict(column_names, results):`\n"
            "Do not include any extra explanation or comments.\n"
            "\n"
            "Example of desired output format:\n"
            "If `column_names = ['ProducingMonth', 'CDWater_BBLPerDAY']` and `results = [('1992-06-01', '2.0'), ('1993-02-01', '6.0'), ...]`,\n"
            "the function should return:\n"
            "{\n"
            "  'ProducingMonth': ['1992-06-01', '1993-02-01', ...],\n"
            "  'CDWater_BBLPerDAY': ['2.0', '6.0', ...]\n"
            "}\n"
        )

        user_prompt = (
            f"Please write the Python function `convertSQL2Dict` as per the instructions.\n"
            f"The function will be used to process the results from the SQL query:\n{sql_query}\n"
            f"The SQL query results are stored in a variable named `results`, which is a list of tuples.\n"
            f"The column names for the results are stored in a list named `column_names`.\n"
            f"The user's requirement (question) is: {userQuestion}\n"
        )

        code = generate_response(sys_prompt, user_prompt)
        print(code)
        code = code.strip()
        
        # Remove Markdown-style code blocks if they exist (```python ... ```).
        if code.startswith("```") and code.endswith("```"):
            code = code[code.find('\n') + 1:-3]

        # Prepare the environment with SQL results passed as input.
        local_env = {'column_names': column_names, 'results': results}

        # Execute the generated Python code safely to generate the dictionary from the SQL results.
        try:
            exec(code, globals(), local_env)
            print("Executed successfully")
            
            # Check if 'convertSQL2Dict' is defined in local_env
            if 'convertSQL2Dict' in local_env:
                # Call the function and assign the result
                dict_results = local_env['convertSQL2Dict'](column_names, results)
                print(dict_results)
            else:
                print("Function 'convertSQL2Dict' not found in local_env.")
        except Exception as e:
            print(f"Execution error: {e}")
            
        # Step 3B: Summarize the results in a brief sentence
        # complete the code:
        table_str = get_dict_summary(dict_results)

        sys_prompt = (
            "You are an expert summarizer. The result dictionary may contain a lot of data, "
            "but all of it is stored in the database, so you don't need to include everything in your response. "
            "Provide only a brief summary of the key points. The summary should be concise and focus on the most important information from the results. "
            "Present a few rows of the data (up to 10 rows) in a table-like format, and if there is more data, indicate this by appending '...'. "
        )

        user_prompt = (
            f"Based on the user's requirement (question):\n{userQuestion}\n\n"
            f"The data sample is:\n{table_str}\n"
            "Please provide a brief summary of the results."
        )

        summarized_results = generate_response(sys_prompt, user_prompt).strip()

        # Outputs
        new_file_rowid = store_data_sqlite3(filename=userID, table="running", data=str(dict_results), type="dict", description=f"user's question: {userQuestion}\n solution: {sql_query}")

        return json.dumps({
            "processed_data_id": new_file_rowid,
            "sql_query": sql_query,
            "task_resut": summarized_results,
            "message": ("Processed succesully, show user the code how to get the results. "
              "You must inform the user that only up to 10 samples are being presented, and if they wish to see more, they should view it in the database table.")
        })

    except Exception as e:
        # Handle any exceptions and return an error message
        return json.dumps({"error": str(e)})