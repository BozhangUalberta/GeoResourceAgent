import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import os
import uuid
import sqlite3
import json
from typing import Annotated, List, Optional, Dict
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langchain_experimental.utilities import PythonREPL
import numpy as np
from src.utils.logger_utils import get_logger
repl = PythonREPL()

IMAGE_DIR = "src/static/images"

def load_data_to_dataframe(db_name, table_name):
    """
    Load the entire table into a DataFrame.
    """
    conn = sqlite3.connect(db_name)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

from openai import OpenAI
from decouple import config 

def generate_response(sys_prompt, user_prompt, model="gpt-4o"):
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

################
##Table source##
################

@tool
def smart_plot_table_source(
        userID: Annotated[str, InjectedState("userID")],
        table_name: Annotated[str, "The correct table used for plot"],
        plot_y_col_name: Annotated[List[str], "List of column names for the y variables"],
        userRequest: Annotated[str, "The user's original plot request including the customized plot instruction"],
        plot_type: Annotated[str, "Plot tyep, line chart, histogram, pie chart or other specified curve"],
        plot_x_col_name: Annotated[Optional[str], "Column name for the x variable"] = None,
        grouped_by_col_name: Annotated[Optional[str], "Column name for the feature needs to be grouped by"] = None,
        plot_x_label: Annotated[Optional[str], 'x_label'] = None,
        plot_y_label: Annotated[Optional[str], 'y_label'] = None,
        plot_title: Annotated[Optional[str], 'plot_title'] = None,
):
    """
    Plot the data stored in the sql database.
    Use LLM to generate Python code for plotting multiple y-columns against an x-column in a designated table.
    This plot tool supports multiple y-columns; group the data to plot; single plot and multiple subplots.
    """
    print('========== Plot the smart production curves ===========')
    logger = get_logger(userID )
    logger.info(f"Plot the table data using the smart plotting tool ....<br>")

    # Generate unique file name and paths
    file_name = f"{uuid.uuid4().hex[:6]}.png"
    file_path = os.path.join(IMAGE_DIR, file_name)
    print(f"Absolute file path: {os.path.abspath(file_path)}")

    image_url = f"http://localhost:800/static/images/{file_name}"

    # Step 1: Load Data
    db_name = os.path.join("database", f"{userID}.db")
    df = load_data_to_dataframe(db_name, table_name)

    # Step 2: Filter Data
    if df.empty:
        return json.dumps({
            "image_url": None,
            "messages": "No data available. Check the table name correctness."
        })
    
    # If no plot_x_col_name is provided, create a default one based on the length of the dataframe
    if plot_x_col_name is None:
        plot_x_col_name = 'default_x_axis'
        df[plot_x_col_name] = range(len(df))

    # Group data if grouping column is provided
    if grouped_by_col_name:
        grouped = df.groupby(grouped_by_col_name)
        category_n = len(grouped.groups)  # Number of categories
    else:
        category_n = 1  # No grouping performed

    figsize = (9, 6)

    # System and User Prompt for LLM
    sys_prompt = (
        "You are an expert python code generator, specialized in generating Python code to create a plot using matplotlib.\n"
        "First rule: Only generate the python code, without any additional explanation.\n"
        "The data is provided in a pandas DataFrame called 'df'.\n"
        "If there are multiple categories, plot them as the user command (plot them in the same plot or in different subplots).\n"
        "Save the plot as a PNG file to the path provided as 'file_path'.\n"
        "Use the parameters given by the user for the plot.\n"
        "Use provided labels and title. Save the plot using plt.savefig(file_path)."
    )

    # Convert list of y-columns to a string representation for the prompt
    y_columns_str = ', '.join(plot_y_col_name)

    user_prompt = (
        f"User's request: {userRequest}\n"
        f"Plot type: {plot_type}\n"
        f"Save plot to: {file_path}\n"
        f"DataFrame columns:\n"
        f"X-axis: {plot_x_col_name}\n"
        f"Y-axis: {y_columns_str}\n"
        f"Labels and Title:\n"
        f"X-axis label: {plot_x_label}\n"
        f"Y-axis label: {plot_y_label}\n"
        f"Title: {plot_title}\n"
        f"figsize: {figsize}\n"
        f"number of category: {category_n}\n"
    )
    if grouped_by_col_name:
        user_prompt += f"grouped by: {grouped_by_col_name}\n"

    # Generate and execute the Python code
    try:
        code = generate_response(sys_prompt, user_prompt).strip()
        if code.startswith("```") and code.endswith("```"):
            code = code[code.find('\n') + 1:-3]

        print("Generated Code:\n", code)

        # Prepare the global namespace for exec
        global_namespace = {
            'df': df,
            'file_path': file_path,
            'plot_x_col_name': plot_x_col_name,
            'plot_y_col_name': plot_y_col_name,
            'plot_x_label': plot_x_label,
            'plot_y_label': plot_y_label,
            'plot_title': plot_title,
            'plt': plt,
            'pd': pd,
            'np': np,
        }

        if grouped_by_col_name:
            global_namespace['category_col_name'] = grouped_by_col_name

        # Execute the code
        exec(code, global_namespace)
        print("Code executed successfully.")
        print(f"Generated image URL: {image_url}")
    except Exception as e:
        error_message = f"Error executing generated code: {e}"
        print(error_message)
        return json.dumps({
            "error": error_message
        })

    return json.dumps({
        "image_url": image_url,
        "messages": "Plot generated successfully."
    })



#######################
##Manual input source##
#######################
@tool
def smart_plot_conversation_source(
        userID: Annotated[str, InjectedState("userID")],
        plot_y_value: Annotated[Dict[str, List[float]], "Dictionary where keys are plot names (str) and values are lists of float values"],
        userRequest: Annotated[str, "The user's original plot request including the customized plot instruction"],
        plot_type: Annotated[str, "Plot type, such as line chart, histogram, or pie chart"],
        plot_x_value: Annotated[Optional[Dict[str, List[float]]], "Dictionary where keys are plot names and values are lists of floats for the x-axis"] = None,
        plot_x_label: Annotated[Optional[str], 'Label for the x-axis'] = None,
        plot_y_label: Annotated[Optional[str], 'Label for the y-axis'] = None,
        plot_title: Annotated[Optional[str], 'Title of the plot'] = None,
):
    """
    plot_y_value example: {"plot_name1": [values1], "plot_name2": [values2]}
    plot_x_value example: {"plot_name1": [x_values_for_plot1], "plot_name2": [x_values_for_plot2]}

    If plot_x_value is not provided or a specific plot_name key is missing in plot_x_value,
    a default x-axis (0, 1, 2, ...) will be used for that plot_name.

    All series will be combined into a single DataFrame. If their lengths differ, 
    they will be padded with NaN to the length of the longest series.
    """
    print('========== Generating Plot from conversation ==========')
    logger = get_logger(userID)
    logger.info(f"Plot the conversation data using the smart plotting tool ....<br>")

    # Prepare data for plotting
    try:
        # Validate plot_y_value
        if not isinstance(plot_y_value, dict) or len(plot_y_value) == 0:
            raise ValueError("plot_y_value must be a non-empty dictionary with lists of numeric values.")

        # Check each y_data
        for name, y_data in plot_y_value.items():
            if not isinstance(y_data, list) or not all(isinstance(val, (int, float)) for val in y_data):
                raise ValueError(f"All values for '{name}' must be a list of numeric values (int or float).")
            if len(y_data) == 0:
                raise ValueError(f"The list of values for '{name}' cannot be empty.")

        # Determine the maximum length of all y-series
        max_length = max(len(y_data) for y_data in plot_y_value.values())

        # If plot_x_value is provided, it should be a dict or None
        if plot_x_value is not None and not isinstance(plot_x_value, dict):
            raise ValueError("plot_x_value must be a dictionary or None.")

        # Prepare a dictionary to store columns
        # For each plot_name, we will have two columns: x_plot_name and plot_name
        data_dict = {}
        plot_x_col_name = []
        plot_y_col_name = []

        print("preparing the data")

        for name, y_data in plot_y_value.items():
            # Validate or generate x_data
            if plot_x_value is not None and name in plot_x_value:
                x_data = plot_x_value[name]
                if not isinstance(x_data, list) or not all(isinstance(val, (int, float)) for val in x_data):
                    raise ValueError(f"X-values for '{name}' must be a list of numeric values (int or float).")
            else:
                # Generate a default x range if not provided
                x_data = list(range(len(y_data)))

            # Pad y_data if needed
            if len(y_data) < max_length:
                y_data = y_data + [np.nan]*(max_length - len(y_data))
            # Pad x_data if needed
            if len(x_data) < max_length:
                x_data = x_data + [np.nan]*(max_length - len(x_data))

            # Assign to the dictionary
            data_dict[f"x_{name}"] = x_data
            data_dict[name] = y_data
            plot_x_col_name.append(f"x_{name}")
            plot_y_col_name.append(name)

        # Create the DataFrame from data_dict
        df = pd.DataFrame(data_dict)

    except Exception as e:
        return {
            "error": f"Error preparing data: {e}"
        }

    # Generate unique file name and paths
    file_name = f"{uuid.uuid4().hex[:6]}.png"
    file_path = os.path.join(IMAGE_DIR, file_name)
    print(f"Absolute file path: {os.path.abspath(file_path)}")

    figsize = (9, 6)

    sys_prompt = (
        "You are an expert Python code generator specialized in creating plots using matplotlib.\n"
        "First rule: Only generate the python code, without any additional explanation.\n"
        "Generate Python code to plot the given y-columns against the x-column in a pandas DataFrame named 'df'.\n"
        "Save the plot as a PNG file to the path specified in 'file_path'.\n"
        "Use the provided labels and title for the plot."
    )

    x_columns_str = ', '.join(plot_x_col_name)
    y_columns_str = ', '.join(plot_y_col_name)

    user_prompt = (
        f"User's request: {userRequest}\n"
        f"Plot type: {plot_type}\n"
        f"Save plot to: {file_path}\n"
        f"X-axis: {x_columns_str}\n"
        f"Y-axis: {y_columns_str}\n"
        f"X-axis label: {plot_x_label}\n"
        f"Y-axis label: {plot_y_label}\n"
        f"Title: {plot_title}\n"
        f"figsize: {figsize}\n"
    )

    # Generate and execute the Python code
    try:
        code = generate_response(sys_prompt, user_prompt).strip()
        if code.startswith("```") and code.endswith("```"):
            code = code[code.find('\n') + 1:-3]

        print("Generated Code:\n", code)

        # Prepare the global namespace for exec
        global_namespace = {
            'df': df,
            'file_path': file_path,
            'plot_x_col_name': plot_x_col_name,
            'plot_y_col_name': plot_y_col_name,
            'plot_x_label': plot_x_label,
            'plot_y_label': plot_y_label,
            'plot_title': plot_title,
            'plt': plt,
            'pd': pd,
            'np': np,
        }

        # Execute the code
        exec(code, global_namespace)
        print("Code executed successfully.")
        image_url = f"http://localhost:800/static/images/{file_name}"

    except Exception as e:
        error_message = f"Error executing generated code: {e}"
        print(error_message)
        return {
            "error": error_message
        }

    return {
        "image_url": image_url,
        "message": "Plot generated successfully."
    }

