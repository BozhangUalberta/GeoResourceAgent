import os
import math
import json

import numexpr
import sqlite3

from typing import Annotated, Sequence, Optional
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState


@tool
def calculator(
        userID: Annotated[str, InjectedState("userID")],
        expression: str,
        is_store_into_running_table: bool = False,
        store_description: Optional[str] = None,
    ) -> str:
    """Calculate expression using Python's numexpr library.

    Expression should be a single line mathematical expression
    that solves the problem.

    Examples:
        "37593 * 67" for "37593 times 67"
        "37593**(1/5)" for "37593^(1/5)"
    """
    local_dict = {"pi": math.pi, "e": math.e}

    result = numexpr.evaluate(
        expression.strip(),
        global_dict={},  # restrict access to globals
        local_dict=local_dict,  # add common mathematical functions
    )

    # Return results directly
    if not is_store_into_running_table:
        return json.dumps({"status": "succeed", "result": str(result)})
    
    # Store into running table
    if is_store_into_running_table:
        result = numexpr.evaluate(
            expression.strip(),
            global_dict={},  # restrict access to globals
            local_dict=local_dict,  # add common mathematical functions
        )
        result_shape = str(result.shape)

        # Connect DB
        db_name = os.path.join("database", f"{userID}.db")
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Insert into running table
        cursor.execute('''
            INSERT INTO running (data, type, description)
            VALUES (?, ?, ?)
        ''', (result, "nparray", store_description))
        row_id = cursor.lastrowid

        # Commit the transaction and close the connection
        conn.commit()
        conn.close()

        return json.dumps({"status": "succeed", "row_id": row_id, "result_shape": result_shape})
    
    return json.dumps({"status": "failed", "reason": "param::is_store_into_running_table is neither True or False"})