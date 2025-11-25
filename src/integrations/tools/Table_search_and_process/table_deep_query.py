from langchain_core.tools import tool
import json
import uuid
from typing import Annotated, Tuple, Optional
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

def coder(userID: str,
          tableName: str,
          userQuestion: str,
          table_info: str,
          validation_feedback: Optional[str] = None,
          previous_solution: Optional[str] = None) -> str:
    """
    Generate SQL code based on user question, table info, and optional feedback.
    This replaces the previous ProcessSQL_coder tool, making it a callable function.
    """
    print('========== Coder ===========')
    print("question:", userQuestion)
    print("table_info:", table_info)
    print("feedback:", validation_feedback)
    logger = get_logger(userID)
    logger.info("Generating a new SQL query...<br>")

    db_name = os.path.join("database", f"{userID}.db")
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Fetch column names from the specified table
    cursor.execute(f"PRAGMA table_info({tableName})")
    column_info = cursor.fetchall()
    column_names = [column[1] for column in column_info]
    conn.close()

    sys_prompt = (
        "You are an expert SQL query generator tasked with creating SQL queries that fulfill the user's request based on the provided table information. "
        "First rule: generate the SQLite code only, without any additional explanation.\n"
        "Seccond rule: You can only extract info; you do not have the right to add, delete or modify the original table.\n"
        "Match the features in the user's request with the correct table columns using the detailed `table_info`. "
        "Use CAST operations if needed. Avoid NULL values in calculations. "
        "If production data is requested but not specifying oil/gas/water, include all available production.\n"
        "Incorporate suggestions from `validation_feedback` and reference the `previous_solution` if available.\n"
    )

    user_prompt = (
        f"The table information is as follows: {table_info}\n"
        f"The correct column names in the table are: {column_names}\n"
        f"User's requirement: {userQuestion}\n"
        f"Table name: '{tableName}' in '{db_name}'\n"
        f"Previous solution: {previous_solution if previous_solution else 'None'}\n"
        f"Validation feedback: {validation_feedback if validation_feedback else 'None'}\n"
    )

    sql_query = generate_response(sys_prompt, user_prompt, model="gpt-4o")
    sql_query = sql_query.strip()
    if sql_query.startswith("```") and sql_query.endswith("```"):
        sql_query = sql_query[sql_query.find('\n') + 1:-3]

    return sql_query

def validator(task_objective: str,
              sql_query: str,
              column_names: list,
              results_df: pd.DataFrame,
              be_less_strict: bool = False) -> Tuple[str, str]:
    """
    Validates the results using an LLM with a single attempt. Ensures a strictly formatted JSON response.
    If the LLM fails to produce a correct JSON with a recognized status, returns NOT VALID.
    """

    # Strictness Note
    strictness_note = (
        "Previously, you were too strict. Now, after multiple attempts, be more lenient. "
        "Accept results that are roughly aligned with the user's request unless there are major issues. "
        "For example, if the data is plausible and mostly fulfills the user's request, mark it as VALID. "
    )

    # System Prompt
    validation_sys_prompt = (
        "You are an experienced data analyst. Evaluate if the given results meet the user's requirements.\n"
        "You must respond with a JSON object and nothing else. The JSON must be strictly in this format:\n\n"
        "{\n"
        "  \"status\": \"VALID\" or \"NOT VALID\",\n"
        "  \"reason\": \"A brief explanation\"\n"
        "}\n\n"
        "Do not include any extra text, explanations, or commentary outside of the JSON object. "
        "Do not add new lines before or after the JSON. "
        "If the results mostly fulfill the request, set \"status\": \"VALID\". If not, use \"NOT VALID\" and explain why."
    )

    if be_less_strict:
        validation_sys_prompt += strictness_note

    # User Prompt
    validation_user_prompt = (
        f"User's question: {task_objective}\n"
        f"Results obtained (first 20 rows):\n{results_df.head(20).to_string(index=False)}\n"
        f"SQL Query executed:\n{sql_query}\n"
        f"Original column names in the table:\n{column_names}\n"
        "Please produce the JSON response as instructed."
    )

    # Single attempt to validate
    validation_response = generate_response(validation_sys_prompt, validation_user_prompt).strip()

    # Parse the JSON from the response
    try:
        response_data = json.loads(validation_response)
        status = response_data.get("status")
        reason = response_data.get("reason", "No reason provided.")

        if status == "VALID":
            return "VALID", reason
        elif status == "NOT VALID":
            return "NOT VALID", reason
        else:
            # Unrecognized status
            return "NOT VALID", "The validator did not produce a recognized status."
    except json.JSONDecodeError:
        # If JSON parsing fails, return NOT VALID
        return "NOT VALID", "The validator did not produce valid JSON."
    
def is_safe_query(sql_query: str) -> bool:
    # A simple check to ensure only SELECT is allowed:
    # Convert to uppercase to avoid case issues
    upper_query = sql_query.strip().upper()
    # Check if it starts with SELECT and does not contain modifying keywords
    if not upper_query.startswith("SELECT"):
        return False
    # Add checks for forbidden keywords
    forbidden_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE"]
    for keyword in forbidden_keywords:
        if keyword in upper_query:
            return False
    return True

@tool
def table_deep_query(
    userID: Annotated[str, InjectedState("userID")],
    tableName: Annotated[str, "The designated database table to be processed"],
    userQuestion: Annotated[str, "The user's original request"],
    table_info: Annotated[str, "The summarized information about the table"],
    store_results: Annotated[bool, "True to save the extracted information; False otherwise"]
):
    """
    The `table_deep_query` tool generates and executes SQL code to retrieve information from a specified
    database table based on the user's question. It can handle any query that can be formulated into SQL.
    The tool validates the results for logical consistency and ensures they align with the user's request.
    
    Important: The tool will NEVER modify the original table, only SELECT queries are allowed.
    """

    print('========== Deep Query ===========')
    logger = get_logger(userID)
    logger.info(f"Using the deep query to search the Table: {tableName}....<br>")

    iteration_count = 0
    max_iterations = 5
    validation_feedback = None
    previous_solution = None
    best_results = pd.DataFrame()
    best_sql = None

    db_name = os.path.join("database", f"{userID}.db")

    while iteration_count < max_iterations:
        iteration_count += 1
        # Generate SQL
        sql_query = coder(userID, tableName, userQuestion, table_info, validation_feedback, previous_solution)
        print(f"The validation iteration count: {iteration_count}")
        previous_solution = sql_query  # Store for next iteration if needed.

        # Check if the query is read-only
        if not is_safe_query(sql_query):
            validation_feedback = (
                "The generated query is not a SELECT-only query. It attempts to modify the table, which is not allowed. "
                "Please generate a SELECT-only query that extracts the required data without any modifications."
            )
            if iteration_count == max_iterations:
                return json.dumps({
                    "message": "No valid, non-modifying SQL query produced after multiple attempts."
                })
            # Move to the next iteration
            continue

        # Execute SQL
        try:
            conn = sqlite3.connect(db_name)
            cursor = conn.cursor()
            cursor.execute(sql_query)
            results = cursor.fetchall()
            if cursor.description is not None:
                column_names = [desc[0] for desc in cursor.description]
                results_df = pd.DataFrame(results, columns=column_names)
            else:
                results_df = pd.DataFrame()
                print("No columns returned. Possibly invalid query.")

        except sqlite3.Error as e:
            # If execution error, treat as feedback
            error_message = f"SQL Execution Error: {str(e)}"
            print(error_message)
            validation_feedback = error_message
            results_df = pd.DataFrame()
            continue
        finally:
            conn.close()

        # If empty results, try again
        if results_df.empty:
            validation_feedback = "The query returned no results. Please adjust the query."
            if iteration_count == max_iterations:
                return json.dumps({
                    "instruction": "No data returned after multiple attempts.",
                    "message": "Check your request or consider manual review."
                })
            continue

        # Validate results
        print('========== Validation ===========')
        logger.info("Validating the results....<br>")
        be_less_strict = iteration_count >= 3  # After 3 attempts, be less strict
        validation_status, validation_response = validator(
            task_objective=userQuestion,
            sql_query=sql_query,
            column_names=column_names,
            results_df=results_df,
            be_less_strict=be_less_strict
        )
        logger.info(f"Validation status: {validation_status}!<br>")

        if validation_status == "VALID":
            best_results = results_df
            best_sql = sql_query
            break
        else:
            # Not valid, use feedback
            print(validation_response)
            validation_feedback = validation_response
            if iteration_count == max_iterations:
                if best_results.empty:
                    return json.dumps({
                        "message": (
                            "The process failed to produce valid results after multiple attempts. "
                            f"Reason: {validation_feedback}. No partial results are available."
                        )
                    })
                else:
                    return json.dumps({
                        "message": (
                            "The process failed to produce valid results after multiple attempts. "
                            f"Reason: {validation_feedback}. "
                            f"Here are the last attempted results: {best_results.head().to_dict(orient='records')}"
                        )
                    })

    # Final output
    if best_results is not None and not best_results.empty:
        # Decide whether to store
        show_max = 60
        if best_results.shape[0] > show_max:
            store_results = True
            reason_for_storage = "Too many rows, storing for reference."
        else:
            reason_for_storage = "User requested storing the output."

        if store_results:
            conn = sqlite3.connect(db_name)
            new_table_name = f'extracted_from_{tableName}_' + uuid.uuid4().hex[:4]
            new_table_name = sanitize_table_name(new_table_name)
            try:
                # Storing results in a NEW table, not modifying the original
                best_results.to_sql(new_table_name, conn, if_exists='replace', index=False)
                all_columns = "\n".join([f"- '{col}'" for col in best_results.columns])
                store_data_sqlite3(
                    filename=userID,
                    table="running",
                    data=f"Saved table: {new_table_name}",
                    type="processed",
                    description=(
                        f"Data generated has been saved in '{new_table_name}'.\n"
                        f"The table includes the following columns:\n{all_columns}\n\n"
                    )
                )
                conn.commit()
            except Exception as e:
                error_message = f"Error saving data to table '{new_table_name}': {str(e)}"
                print(error_message)
                return json.dumps({
                    "instruction": "Error while saving data.",
                    "error_message": error_message
                })
            finally:
                conn.close()

            message = (
                "The processor successfully completed the searching task. "
                f"The data has been stored in the table '{new_table_name}'. "
                f"Reason for storage: {reason_for_storage}. "
                f"Total iterations performed to generate and validate the results: {iteration_count}."
            )
        else:
            message = (
                "The processor successfully completed the searching task. The data was not stored as per the settings. "
                f"Total iterations performed to generate and validate the results: {iteration_count}."
            )

        return json.dumps({
            "message": message,
            "sql_used": best_sql,
            "result_sample": best_results.head(show_max).to_dict(orient="records"),
        })
    else:
        return json.dumps({
            "instruction": "No valid results after multiple attempts.",
            "message": "User may need to review the request or the data."
        })