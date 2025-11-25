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

@tool
def ProcessCSV(userID: Annotated[str, InjectedState("userID")],
               rowid: Annotated[str, "User uploaded .csv file"],
               workCommand: Annotated[str, "The user's instructions on how to process the CSV."],
               ):
    """
    Process the uploaded CSV file, it can read the CSV contents and finish some analysis tasks for uploaded file.
    This tool has LLM built inside and can understand any of the command, do not need to let the user specify the specific column name.
    Inputs:
        - userID: The user's unique identifier.
        - rowid: The ID to retrieve the uploaded CSV file.
        - workCommand(optional): The user's instructions on how to process the CSV.
    Returns:
        - JSON object containing the processed data ID, the python code used to get the results, task result and a message.
    """
    try:
        user_save_dir = SAVE_DIR.format(userID=userID)
        if not os.path.exists(user_save_dir):
            os.makedirs(user_save_dir)

        # Get the CSV file path from the database using the provided rowid
        data_path = get_data_sqlite3(filename="test.db", table=userID, id=rowid, type="userinput")
        file_name = os.path.basename(data_path)
        data_path = f"src/static/{userID}/{file_name}"  # Ensure correct file path format

        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(data_path)

        # Step 1: Summarize the CSV file
        csv_sample = df.columns
        sys_prompt = (
            "Examine the columns and content of the CSV data. "
            "Your primary task is to match the user's request to the correct columns. "
            "The user may not provide the exact column name, but you need to handle similar or related terms. "
            "Additionally, provide a brief summary of what this file might be used for."
        )

        user_prompt = (
            f"The columns of the CSV file are:\n{csv_sample}. "
            f"The user command is:\n{workCommand}. "
            "Please match the user's requested columns with the original columns. "
            "If no match is found, print 'Exit'."
        )

        # Generate a response based on system and user prompts
        descriptions = generate_response(sys_prompt, user_prompt)

        # Handle the case where no matching columns are found
        if descriptions == "Exit":
            return json.dumps({"error": "The requested columns do not exist."})

        # Step2: process the csv file using python repl tool
        sys_prompt = (
            "First rule: Generate the code only, without speaking too much."
            "You are a code generator for the Python REPL tool. "
            "Your task is to generate Python code to load, analyze, and process CSV data files. "
            "The code should assume the file will contain tabular data with columns, which may include strings, numbers, or missing values. "
            "Ensure that the code is well-structured, includes error handling for missing or inconsistent data" 
            "If the task involves retrieving specific values or lists from the CSV, return only the requested values or list directly."
        )
        user_prompt = (
            f"{workCommand}. The file to be processed is located at: {data_path}. "
            f"It contains data with the following summary and columns: {descriptions}. "
            "Please locate the column that most closely matches the requested information. "
            "If no exact column name match is found, use your best judgment to find the closest match based on similar terms, "
            "and proceed with the task accordingly.")
        code = generate_response(sys_prompt, user_prompt)
        # use repl to compile the tool
        repl = PythonREPL()
        results = repl.run(code)

        # Step 3: Refine results
        sys_prompt = (
            "First rule, only genrate the disctionary without other words."
            "Extract the key values from the results and organize them into a dictionary format. "
            "Each entry in the dictionary should have a descriptive name that clearly indicates the type of information it holds, followed by the value. "
            "Summarize the main findings concisely and ensure the dictionary aligns with the goal of the task as specified in the original work command."
        )

        user_prompt = (
            "Please extract and organize the key data points from the results into a dictionary. "
            "Each dictionary entry should include a descriptive name for the data point and the corresponding value. "
            f"The user's question is to {workCommand}. "
            f"The results obtained are: {results}."
        )
        dict_results = generate_response(sys_prompt, user_prompt)

        # Outputs
        new_file_rowid = store_data_sqlite3(filename="test.db", table=userID, data=dict_results, type="processed", description=workCommand)

        return json.dumps({
            "processed_data_id": new_file_rowid,
            "code": code,
            "task_resut": results,
            "message": "Processed succesully, show user the results, as well as the code how to get the results."
        })

    except Exception as e:
        # Handle any exceptions and return an error message
        return json.dumps({"error": str(e)})