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
import ast
import numpy as np  # Import numpy to handle NaN values

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

@tool
def ProcessSQL(userID: Annotated[str, InjectedState("userID")],
               tableName: Annotated[str, "The designated table the user wants to process"],
               workCommand: Annotated[str, "The user's instructions on how to process the data."]):
    """
    Process uploaded files stored as tables in the database. This tool interprets the schema based on user commands and extracts the requested information accurately.
    With an integrated LLM, it can automatically correct column names, meaning users do not need to specify exact column names.
    Inputs:
        - userID: The user's unique identifier.
        - tableName: The designated table containing the requested information.
        - workCommand: The user's instructions on how to process the data.
    Returns:
        - JSON object containing the processed data ID, the Python code used to get the results, task result, and a message.
    """
    db_name = os.path.join("database", f"{userID}.db")
    conn = sqlite3.connect(db_name)

    # Read the specified table into a pandas DataFrame
    df = pd.read_sql_query(f"SELECT * FROM {tableName}", conn)

    # Handle NULL values in 'place' column by replacing NaN with empty strings
    if 'place' in df.columns:
        df['place'] = df['place'].fillna('')

    # Alternatively, replace NaN with empty strings in all object-type columns
    df = df.apply(lambda x: x.fillna('') if x.dtype == 'object' else x)

    # Create a summary of the DataFrame
    columns = df.columns.tolist()
    sample_data = df.head().to_dict(orient='records')

    # Step 1: Summarize the designated DataFrame
    sys_prompt = (
        "Examine the columns and content of the DataFrame. "
        "Your primary task is to match the user's request to the correct columns. "
        "The user may not provide the exact column name, so handle similar or related terms. "
        "Additionally, provide a brief summary of what this table might be used for. "
        "Note that some data may contain NULL or missing values."
    )

    user_prompt = (
        f"The DataFrame has the following columns:{columns}\n"
        f"Sample data:{sample_data}\n"
        f"The user's command is: {workCommand}\n"
        "Please match the user's requested columns with the original columns. "
        "If no match is found, respond with 'Exit'."
    )

    # Generate a response based on system and user prompts
    descriptions = generate_response(sys_prompt, user_prompt)

    # Handle the case where no matching columns are found
    if descriptions.strip() == "Exit":
        return json.dumps({"error": "The requested columns do not exist."})

    # Step 2: Process the DataFrame using the generated Python code
    sys_prompt = (
        "First rule: Generate only the code without any additional explanation.\n"
        "You are a code generator for the exec to run. "
        "Your task is to generate Python code to process a pandas DataFrame named 'df'. "
        "Ensure that the code is well-structured and includes error handling for missing or inconsistent data, such as NULL values. "
        "When performing calculations, replace NULL values with 0 to avoid errors. "
    )
    user_prompt = (
        f"The user's command is:\n{workCommand}\n\n"
        f"The DataFrame has the following columns:\n{columns}\n\n"
        "Please generate the Python code to accomplish this task. Remember to handle NULL values appropriately."
    )
    code = generate_response(sys_prompt, user_prompt)
    code = code.strip()
    # Remove Markdown-style code blocks if they exist (```)
    if code.startswith("```") and code.endswith("```"):
        code = code[code.find('\n') + 1:-3]  # Removes the backticks and language specifier (like 'python')
    # Print the cleaned code for debugging
    print('Cleaned Code:')
    print(repr(code))

    # Prepare the execution environment
    local_env = {'df': df, 'pd': pd, 'np': np}

    # Use exec to execute the code safely
    try:
        exec(code, globals(), local_env)
        print("exec correctly")
        results = local_env.get('result', None)
        print("code is running")
    except Exception as exec_error:
        print(str(exec_error))
        return json.dumps({"error": f"Error executing generated code: {str(exec_error)}"})

    # Step 3A: Summarize the results in a brief sentence
    sys_prompt = (
        "You are an expert summarizer. The result dictionary may contain a lot of data, "
        "but all of it is stored in the database, so you don't need to include everything in your response."
        " Display a few of the results (10 lines maximum). "
        "Provide only a brief summary of the key points. The summary should be concise and focus on the most important information from the results."
    )

    user_prompt = (
        f"Based on the user's command:\n{workCommand}\n\n"
        f"and the results:\n{results}\n\n"
        "Please provide a brief summary of the results."
    )

    summarized_results = generate_response(sys_prompt, user_prompt).strip()

    # Step 3B: Refine results into a dictionary
    sys_prompt = (
        "Summarize the obtained results concisely and create a dictionary aligned with the user's command. "
        "Generate the dictionary without any additional words. "
        "Each entry should include a descriptive key for the type of information and the corresponding value."
    )

    user_prompt = (
        "Please extract and organize the key data points from the results into a dictionary. "
        "Each dictionary entry should include a descriptive name for the data point and the corresponding value. "
        f"The user's command is to: {workCommand}.\n "
        f"The results obtained are: {results}.\n"
    )
    dict_results = generate_response(sys_prompt, user_prompt)

    # Store the results in the database
    new_file_rowid = store_data_sqlite3(
        filename=userID,
        table="running",
        data=dict_results,
        type="dict",
        description=workCommand
    )

    return json.dumps({
        "processed_data_id": new_file_rowid,
        "python_code": code,
        "task_result": summarized_results,
        "message": "Processed successfully. Show user the results, as well as the code how to get the results."
    })
