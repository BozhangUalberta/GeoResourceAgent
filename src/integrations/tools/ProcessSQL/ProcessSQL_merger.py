from langchain_core.tools import tool
import json
import uuid
from src.utils.db_utils import get_data_sqlite3, store_data_sqlite3
from typing import Annotated, Optional, List
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

def load_data_to_dataframe(db_name, table_name):
    # Load the entire table into a DataFrame
    conn = sqlite3.connect(db_name)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

def sanitize_table_name(name):
    # Remove invalid characters and replace spaces with underscores
    return ''.join(c if c.isalnum() or c == '_' else '_' for c in name)

@tool
def ProcessSQL_merger(
    userID: Annotated[str, InjectedState("userID")],
    merge_requirement: Annotated[str, "The way how to merge the table."],
    table_names: Annotated[List[str], "The designated database tables the user wants to process"],
    table_info: Annotated[List[str], "The summarized infomation from the tables"],
    user_question: Annotated[str, "How the user want to merge the data."],
    drop_na: Annotated[Optional[bool], "If drop the NULLs in the tables. Default is false."] = False,
    error_feedback: Annotated[Optional[str], "Feedback error from previous execution or None if first time run or no feedback"] = None,
):
    """
    Merge different tables into one based on the user's requirement.\n
    merge_requirement
    - "Vertial": Merge the data from the wells vertically, one after another, if no special requirement.
    - "Same_Column": Merge the data by using the same column.
    - "Other_specified": Merge the data by requirement.
    - "Other_not_specified": Will return error and let the user specicy the merging method.
    """
    print('========== Data Merger ==========')
    print(user_question)
    db_name = os.path.join("database", userID + ".db")

    # Load the tables into dataframes
    dfs = {}
    for i, table_name in enumerate(table_names, start=1):
        df = load_data_to_dataframe(db_name, table_name)
        dfs[f"df{i}"] = df

    # Ensure at least two tables for merging
    if len(dfs) < 2:
        return json.dumps({
            "new_table_name": None,
            "message": "Error: At least two tables are required for merging."
        })

    # Add DataFrames as a list for easier iteration
    dfs_list = list(dfs.values())

   # Prepare the system and user prompts
    sys_prompt = (
        "Your task is to write Python code that merges multiple pandas DataFrames "
        "according to the user's requirements. The input DataFrames are available "
        "as variables (df1, df2, ...) and in a list named 'dfs_list'. "
        "**The pandas library is available as 'pd'; do not include any import statements in your code.** "
        "Generate only the Python code to perform the merging operation. "
        "The final merged DataFrame should be named 'results_df'. Ensure the code is "
        "robust and handles edge cases like mismatched columns or empty DataFrames. "
        "Do not include any additional explanations or comments."
    )

    user_prompt = (
        f"The user's question is:\n{user_question}\n\n"
        f"The information of the tables:\n{table_info}\n\n"
    )
    if drop_na:
        user_prompt += "Drop NAs while after merging the data."

    if merge_requirement == "Vertical":
        user_prompt += "Merge the data from the wells vertically, one after another, if no special requirement."
    elif merge_requirement == "Same_column":
        user_prompt += "Merge the data by using the same column."
    elif merge_requirement == "Other_specified":
        user_prompt +="Use the specified method by the user."
    elif  merge_requirement == "Other_not_specified":
        return json.dumps({
            "error": "Please specify how do you want to merge the table."
        })

    if error_feedback:
        user_prompt += f"Feedback from the previous execution:\n{error_feedback}\n"

    # Generate the code
    code = generate_response(sys_prompt, user_prompt)
    code = code.strip()
    # Remove code block markers if present
    for _ in range(3):
        if code.startswith("```") and code.endswith("```"):
            code = code[code.find('\n') + 1:-3]

    print('Generated Code:')
    print(code)

    # Prepare the execution environment
    local_env = {
        **dfs,
        'dfs_list': dfs_list,
        'pd': pd,
        'results_df': None,
    }

    # Execute the code
    try:
        exec(code, local_env)
    except Exception as e:
        print(f"Error executing code: {e}")
        return json.dumps({
            "new_table_name": None,
            "message": f"An error occurred while executing the code: {e}"
        })

    # Get 'results_df' from local_env
    results_df = local_env.get('results_df')

    if results_df is None:
        print("No results dataframe generated.")
        return json.dumps({
            "new_table_name": None,
            "message": "No results dataframe generated."
        })

    # Proceed to generate new table name and return results
    sys_prompt = (
        "Summarize the task in a few words, creating a new table name by joining the original names with an appropriate suffix, separated by underscores. "
        "Do not include any symbols except underscores. "
        "The new name should be simple and descriptive of the merged data."
    )

    user_prompt = (
        f"Original table names: {', '.join(table_names)}. "
        "Generate a concise, new table name for the merged file. Must be less than 6 words."
    )

    # Generate a new table name for the merged DataFrame
    new_table_name = generate_response(sys_prompt, user_prompt).strip()
    new_table_name = sanitize_table_name(new_table_name)
    print(f"New table name: {new_table_name}")

    # Save the results dataframe to the database
    conn = sqlite3.connect(db_name)
    results_df.to_sql(new_table_name, conn, if_exists='replace', index=False)
    conn.close()

    # Store information about the new table
    store_data_sqlite3(
        filename=userID,
        table="running",
        data=f"Saved table: {new_table_name}",
        type="processed",
        description=f"Data generated by ProcessSQL_merger has been saved in the table named '{new_table_name}'. This data may be used for further processing."
    )

    return json.dumps({
        "new_table_name": new_table_name,
        "message": "Merge successful. The new table has been created."
    })