from langchain_core.tools import tool
import json
import uuid
from typing import Annotated, Tuple, Literal
from langgraph.prebuilt import InjectedState
import os
import pandas as pd
import sqlite3
from openai import OpenAI
from decouple import config
from src.utils.db_utils import store_data_sqlite3
import re
from src.utils.logger_utils import get_logger

SAVE_DIR = "src/static/{userID}"

def sanitize_table_name(name):
    # Allow only alphanumeric characters and underscores, and convert to lowercase
    return re.sub(r'\W+', '_', name).lower()

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

def validator(task_objective: str, sql_query: str, column_names: list, results_df) -> Tuple[str, str]:
    # Use the new prompt
    validation_sys_prompt = (
        "You are an experienced data analyst. Evaluate if the given results meet the user's requirements.\n"
        "Respond with a JSON object:\n"
        "{\n"
        "  \"status\": \"VALID\" or \"NOT VALID\",\n"
        "  \"reason\": \"A brief explanation\"\n"
        "}\n"
        "If the results are acceptable, appear logical and mostly fulfill the request, respond with, use \"VALID\".\n"
        "If there are significant issues in logic or the main requirements aren't met, use \"NOT VALID\" and provide an explanation."
    )

    validation_user_prompt = (
        f"User's question: {task_objective}\n"
        f"Results obtained:\n{results_df.head(20).to_string(index=False)}\n"
        f"SQL Query executed to get the results:\n{sql_query}\n"
        f"Original column names in the table:\n{column_names}\n"
        f"Please produce the JSON response as instructed."
    )

    # Generate response
    validation_response = generate_response(validation_sys_prompt, validation_user_prompt)
    
    # Parse the JSON from the response
    try:
        response_data = json.loads(validation_response.strip())
        status = response_data.get("status", "NOT VALID")
        reason = response_data.get("reason", "No reason provided.")
    except json.JSONDecodeError:
        # If JSON parsing fails, return NOT VALID
        status = "NOT VALID"
        reason = "The validator did not produce valid JSON."
    
    if status == "VALID":
        print("valid")
        return "VALID", "Results meet the requirements."
    else:
        print("invalid")
        return "NOT VALID", reason


@tool
def ProcessSQL_processor(
    userID: Annotated[str, InjectedState("userID")],
    tableName: Annotated[str, "The designated database table to be processed"],
    sql_code: Annotated[str, "The SQL code used to address the problem"],
    task_objective: Annotated[str, "The task objective of the task"],
    store_results: Annotated[bool, "True to save the extracted information; False otherwise"]
):
    """
    Executes SQL code to retrieve information from a designated table in the database.
    It validates the results, if the results are valid, returns them. If not, instructs the agent to call the coder again.
    In case of an error, generates a prompt for the coder explaining the problem.
    Data should be stored if the user request.

    Inputs:
        - userID: The user's unique identifier.
        - tableName: The designated table containing the requested information.
        - sql_code: The SQL code used to extract information, please pass the SQL query **only**.
        - task_objective: The task objective of the task (e.g., the user's original question).
        - store_results: "True to save the extracted information; False otherwise"

    Returns:
        - JSON with processed table name, SQL code, task result, and a message, or instructions to call the coder again.
    """
    print('========== Processor ===========')
    print(f"Executing SQL Code:\n{sql_code}")
    logger = get_logger(userID )
    logger.info(f"Processing the generated code....<br>")

    try:
        db_name = os.path.join("database", f"{userID}.db")
        # Sanitize and prepare SQL query
        sql_query = sql_code.strip()
        for _ in range(5):
            if sql_query.startswith("```") and sql_query.endswith("```"):
                sql_query = sql_query[sql_query.find('\n') + 1:-3]

        # Connect to database and execute query
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        try:
            cursor.execute(sql_query)
            results = cursor.fetchall()
            # Check if there are column names available
            if cursor.description is not None:
                column_names = [description[0] for description in cursor.description]
                results_df = pd.DataFrame(results, columns=column_names)
            else:
                # If no columns are returned, results_df is empty or a message is returned
                results_df = pd.DataFrame()
                print("No columns were returned by the SQL query, please check the correctness of the column name.")
        except sqlite3.Error as e:
            error_message = f"SQL Execution Error: {str(e)}"
            print(error_message)
            # Return instruction to call the coder again
            return json.dumps({
                "instruction": "SQL execution error occurred. Please call ProcessSQL_coder to generate new SQL code.",
                "error_message": error_message
            })

        # Validate results (e.g., check if results are not empty)
        if results_df.empty:
            print("Results are empty.")
            # Instruct agent to call the coder again
            return json.dumps({
                "instruction": "Results are empty. Please call ProcessSQL_coder to generate new SQL code.",
                "message": "No data was returned by the SQL query."
            })

        # Step: Use LLM to check if results fulfill user's requirement and are logical
        print('========== Validation ===========')
        validation_status, validation_response = validator(task_objective,sql_query, column_names, results_df)

        # Check the LLM's validation response
        if validation_status == "VALID":
            print(validation_response)
            show_max = 60
            if results_df.shape[0] > show_max:
                store_results = True
                print(f"Warning: Results contain more than {show_max} rows.")
                reason_for_storage = "Task objective explicitly requested data extraction."
            else:
                reason_for_storage = "The user ask to store the output results to table."

            if store_results:
                # Generate a new table name based on the task objective
                new_table_name = f'extracted_from_{tableName}_' + uuid.uuid4().hex[:4]
                new_table_name = sanitize_table_name(new_table_name)
                print("** store data **")
                print(f"new table name:{new_table_name}")
                try:
                    results_df.to_sql(new_table_name, conn, if_exists='replace', index=False)
                    store_data_sqlite3(filename=userID,
                            table="running",
                            data=f"Saved table: {new_table_name}",
                            type="processed",
                            description=f"Data generated by ProcessSQL_processor has been saved in the table named '{new_table_name}' to address the task: {task_objective}. This data may be used for further processing.")
                except Exception as e:
                    error_message = f"Error saving data to table '{new_table_name}': {str(e)}"
                    print(error_message)
                    return json.dumps({
                        "instruction": "An error occurred while saving data. Please verify the table name and try again.",
                        "error_message": error_message
                    })
                
                message = (
                    f"The data has been stored in a new table named '{new_table_name}'. The reason for storing the data is: {reason_for_storage}."
                )
            else:
                message = "The extracted information was not stored in the database table."

            # Commit and close database connection
            conn.commit()
            conn.close()

            # Return the results
            return json.dumps({
                "message": message,
                "result_sample": results_df.head(show_max).to_dict(orient="records"),  # Preview of up to show_max results
            })
        
        if validation_status == "NOT VALID":
            # LLM found issues with the results
            print(f"LLM validation failed.")
            print(f"The reason is: {validation_response}")
            # Return instruction to call the coder again
            return json.dumps({
                "instruction": (
                    "Results are invalid according to LLM validation. Please carefully review the validation response."
                    " If the issue pertains to incorrect column names, call 'ProcessSQL_summarizer' to re-summarize the column names."
                    f" If the issue is related to the SQL code logic, call 'ProcessSQL_coder' to generate a revised SQL query based on the feedback from the validaor: {validation_response}."
                )
            })

    except Exception as e:
        # Handle any general exceptions
        error_message = f"An unexpected error occurred: {str(e)}"
        print(error_message)
        # Return instruction to call the coder again
        return json.dumps({
            "error_message": error_message
        })
