import os
import uuid
import json
import sqlite3

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt

from typing import Annotated, List, Optional
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langgraph.prebuilt import InjectedState

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


########################################
####### Dumb plot tool ################
########################################
@tool
def plot_production_curves_table_source(
        userID: Annotated[str, InjectedState("userID")],
        table_name: Annotated[str, "The table containing well production data"],
        plot_data_col_name: Annotated[str, "Column name for plotting"],
        userRequest: Annotated[str, "The user's original plot request"],
        well_names: Annotated[Optional[List[str]], "Wells to plot"] = None,
        well_identifier_col_name: Annotated[Optional[str], "Column name for well identifier"] = 'WellName',
        plot_x_label: Annotated[Optional[str], 'x_label'] = None,
        plot_y_label: Annotated[Optional[str], 'y_label'] = None,
        plot_title: Annotated[Optional[str], 'plot_title'] = None
):
    """
    Plot production curves for multiple wells in a single plot.
    """
    print('========== Plot the production curves ===========')
    db_name = os.path.join("database", f"{userID}.db")
    df = load_data_to_dataframe(db_name, table_name)

    # Check if required columns exist
    if plot_data_col_name not in df.columns or well_identifier_col_name not in df.columns:
        print("Required column(s) missing in the data.")
        return json.dumps({
            "image_url": None,
            "messages": "Required column(s) missing in the data."
        })
    
    # Filter wells if well_names are provided
    if well_names:
        df = df[df[well_identifier_col_name].isin(well_names)]
    
    # Handle empty DataFrame after filtering
    if df.empty:
        print("No data available for the specified wells.")
        return json.dumps({
            "image_url": None,
            "messages": "No data available for the specified wells."
        })
    
    # Group data by well identifier
    grouped = df.groupby(well_identifier_col_name)
    plot_data_bank = []
    for well_name, group in grouped:
        plot_data = group[plot_data_col_name].dropna().reset_index(drop=True)
        plot_data_bank.append((well_name, plot_data))
    
    # Plot the production data
    plt.figure(figsize=(9, 5))
    for well_name, plot_data in plot_data_bank:
        plot_data_x = list(range(len(plot_data)))
        plt.plot(plot_data_x, plot_data, label=well_name)

    # Add plot decorations
    plt.xlabel(plot_x_label or "Elapsed Production Time")
    plt.ylabel(plot_y_label or "Production Rate")
    plt.title(plot_title or "Production Curves for Wells")
    plt.legend()
    plt.grid(True)

    # Save the plot to a unique file
    file_name = f"{uuid.uuid4().hex[:6]}.png"
    file_path = os.path.join(IMAGE_DIR, file_name)
    plt.savefig(file_path)
    plt.close()
    image_url = f"http://localhost:800/static/images/{file_name}"
    print(f"Generated image URL: {image_url}")

    return json.dumps({
        "image_url": image_url,
        "messages": "Plot complete"
    })



########################################
####### Smart plot tool ################
########################################
from openai import OpenAI
from decouple import config 

def generate_response(sys_prompt, user_prompt, model="gpt-4o"):
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

########################################################
####### Bridging tool: plots in multiple tables ########
########################################################
@tool
def plot_curves_from_multiple_tables(
        table_names: Annotated[List[str], "The tables where the data is stored"],
        plot_x_col_name: Annotated[str, "Column name for the x variable"],
        plot_y_col_name: Annotated[str, "Column name for the y variable"],
):
    """
    This tool is designed to plot data from multiple tables. 
    Ensure that the provided table names are valid. 
    Ensure that the provided column names are valid, please use the Summarizer to check them if you are not sure.
    You MUST use this tool when the user requests to visualize data that is stored across different tables.
    """
    import json
    print('========== Merge bridging tool ===========')

    # Construct instructions for the agent workflow
    next_steps_message = (
        "Direct to the ProcessSQL_merger tool to merge the files based on the extracted requirement and use the smart_plot_production_curves_table_source to plot. "
    )

    # Return the extracted details and instructions
    return json.dumps({
        "table_names": table_names,
        "plot_x_col_name": plot_x_col_name,
        "plot_y_col_name":plot_y_col_name,
        "messages": next_steps_message
    })


####################################################
####### Smart plot tool --- multiple columns #######
####################################################
from openai import OpenAI
from decouple import config 

def generate_response(sys_prompt, user_prompt, model="gpt-4o"):
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
def smart_plot_production_curves_table_source(
        userID: Annotated[str, InjectedState("userID")],
        table_name: Annotated[str, "The correct table used for plot"],
        plot_y_col_name: Annotated[List[str], "List of column names for the y variables"],
        userRequest: Annotated[str, "The user's original plot request"],
        well_identifier_col_name: Annotated[Optional[str], "Column name for well identifier"] = None,
        well_names: Annotated[Optional[List[str]], "Wells to plot"] = None,
        datetime_col_name: Annotated[Optional[str], "Column name for the datetime column"] = None,
        plot_x_label: Annotated[Optional[str], 'x_label'] = None,
        plot_y_label: Annotated[Optional[str], 'y_label'] = None,
        plot_title: Annotated[Optional[str], 'plot_title'] = None,
        feedback: Annotated[Optional[str], "Feedback from the previous step or None if no feedback"] = None,
):
    """
    You must make sure the table used for plotting is the correct one.
    Use LLM to generate Python code for plotting multiple y-columns for production data in a designated table.
    The tool can be called multiple times with feedback from the previous step to improve the generated code.
    """
    host_ip = os.getenv("HOST_IP")
    host_port = int(os.getenv("HOST_PORT"))

    print('========== Plot the smart production curves ===========')
    print(f"datetime_col_name: {datetime_col_name}")
    logger = get_logger(userID)
    logger.info("Using the smart plotting tool...<br>")

    try:
        # Generate unique file name and paths
        file_name = f"{str(uuid.uuid4())}.png"
        file_path = os.path.join(IMAGE_DIR, file_name)
        print(f"Absolute file path: {os.path.abspath(file_path)}")

        image_url = f"http://{host_ip}:{host_port}/static/images/{file_name}"

        # Step 1: Load Data
        db_name = os.path.join("database", f"{userID}.db")
        df = load_data_to_dataframe(db_name, table_name)

        # Step 2: Filter Data
        # If well_identifier_col_name is not provided, add a default column 'well_id'
        if not well_identifier_col_name:
            print("well_identifier_col_name not provided. Adding a default column 'well_id' with value 1 for all rows.")
            well_identifier_col_name = 'well_id'
            df[well_identifier_col_name] = 1  # Add a default column with value 1

        # Filter DataFrame if both well_identifier_col_name and well_names are valid
        if well_names:
            print(f"Filtering DataFrame by well_identifier_col_name: {well_identifier_col_name}, well_names: {well_names}")
            df = df[df[well_identifier_col_name].isin(well_names)]

        if df.empty:
            return json.dumps({
                "image_url": None,
                "messages": (
                    "No data available for the specified wells. "
                    "Check that the well identifier column and well names are correct."
                )
            })

        # Include the modified sample plots
        sample_plot1 = """import matplotlib.pyplot as plt
            import pandas as pd

            # Group data by well identifier (API_UWI)
            grouped = df.groupby('API_UWI')  # Replace 'API_UWI' with your well_identifier_col_name if needed
            plot_data_bank = []

            # Process each well's data
            for well_name, group in grouped:
                # Drop missing values, reset index, and prepare the data
                plot_data = group[colume_name].dropna().reset_index(drop=True)
                plot_data_bank.append((well_name, plot_data))

            # Plot production data for each well
            plt.figure(figsize=(9, 6))
            for well_name, plot_data_original, plot_data_fitted in plot_data_bank:
                # Create separate X-axis ranges for original and fitted data
                plot_data_x = list(range(len(plot_data)))
                
                # Plot original and fitted data using their respective X-axes
                plt.plot(plot_data_x, plot_data, label=f'{well_name}_{colume_name}', linestyle='--')

            # Add plot decorations
            plt.xlabel("Elapsed Production Time")  # Replace with dynamic variable if needed (e.g., plot_x_label)
            plt.ylabel("Production Rate (MCF per DAY)")  # Replace with dynamic variable if needed (e.g., plot_y_label)
            plt.title("Production Curves for Wells")  # Replace with dynamic variable if needed (e.g., plot_title)
            plt.grid(True)

            # Save the plot
            plt.tight_layout()
            plt.savefig(file_path)
            plt.close() """

        sample_plot2 = """
            import matplotlib.pyplot as plt
            import pandas as pd

            # Group data by well identifier (API_UWI)
            grouped = df.groupby('API_UWI')

            # Define the y_cols to plot
            y_cols = ["y_col_1", "y_col_2"]

            # File path for saving the plot
            file_path = "synthetic_well_plot_combined.png"

            # Plot production data for each well, with all y_cols on the same plot
            plt.figure(figsize=(9, 6))

            for well_name, group in grouped:
                # Ensure ProducingMonth is in datetime format and sorted
                group['ProducingMonth'] = pd.to_datetime(group['ProducingMonth'])
                group = group.sort_values(by='ProducingMonth')

                # Plot data for each y_col
                for y_col in y_cols:
                    if y_col in group.columns:
                        # Drop NaN values for ProducingMonth and the current y_col
                        valid_group = group.dropna(subset=['ProducingMonth', y_col])

                        # Extract ProducingMonth and y_col data
                        producing_month = valid_group['ProducingMonth']
                        plot_data = valid_group[y_col]

                        # Plot data for the current y_col
                        plt.plot(producing_month, plot_data, label=f'{well_name}: {y_col}')

            # Add plot decorations
            plt.xlabel("Producing Month")
            plt.ylabel("Values")
            plt.title("Production Curves for Wells (Combined)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            # Save the plot
            plt.savefig(file_path)
            plt.show()
            """

        # System and User Prompt for LLM
        if datetime_col_name:
            sys_prompt = (
                "Generate Python code to create a plot using matplotlib.\n"
                "First rule: Only generate the python code, without any additional explanation.\n"
                "The well production data are stored vertically one by one; you cannot just directly plot them, you need to separate them and plot.\n"
                "When you are plotting multiple wells and the datetime column is given, you must use this template:\n "
                f"{sample_plot2}\n"
                "The data is provided in a pandas DataFrame called 'df'.\n"
                "Save the plot as a PNG file to the path provided as 'file_path'.\n"
                "Use the following parameters for the plot:\n"
                "- X-axis columns: 'plot_x_col_name'\n"
                "- Y-axis columns: 'plot_y_col_name'\n"
                "- Well identifier column: 'well_identifier_col_name'\n"
                "Use provided labels and title. Save the plot using plt.savefig(file_path)."
            )
        else:
            sys_prompt = (
                "Generate Python code to create a plot using matplotlib.\n"
                "First rule: Only generate the python code, without any additional explanation.\n"
                "The well production data are stored vertically one by one; you cannot just directly plot them, you need to separate them and plot.\n"
                "When you are plotting multiple wells and the datetime column is not given, you must use this template:\n"
                f"{sample_plot1}\n"
                "This ensures all plots start at the same point.\n"
                "The data is provided in a pandas DataFrame called 'df'.\n"
                "Save the plot as a PNG file to the path provided as 'file_path'.\n"
                "Use the following parameters for the plot:\n"
                "- Y-axis columns: 'plot_y_col_name'\n"
                "- Well identifier column: 'well_identifier_col_name'\n"
                "Use provided labels and title. Save the plot using plt.savefig(file_path)."
            )
        # Include feedback if provided
        feedback_text = f"\nFeedback from previous step: {feedback}\n" if feedback else ""

        # Convert list of y-columns to a string representation for the prompt
        y_columns_str = ', '.join(plot_y_col_name)

        user_prompt = (
            f"User's request: {userRequest}\n"
            f"Save plot to: {file_path}\n"
            f"DataFrame columns:\n"
            f"- Well identifier: {well_identifier_col_name}\n"
            f"- Y-axis: {y_columns_str}\n"
            f"Labels and Title:\n"
            f"- X-axis label: {plot_x_label}\n"
            f"- Y-axis label: {plot_y_label}\n"
            f"- Title: {plot_title}\n"
            f"{feedback_text}"
        )

        print("check point 1")

        if datetime_col_name is not None:
            user_prompt += f"- X-axis: {datetime_col_name}\n"

        print("check point 2")

    except Exception as e:
        error_message = f"Something wrong in the code: {e}"
        print(error_message)
        return json.dumps({
            "messages": error_message
        })

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
            'plot_y_col_name': plot_y_col_name,
            'well_identifier_col_name': well_identifier_col_name,
            'plot_x_label': plot_x_label,
            'plot_y_label': plot_y_label,
            'plot_title': plot_title,
            'plt': plt,
            'pd': pd,
            'np': np,
        }

        if datetime_col_name:
            global_namespace['plot_x_col_name'] = datetime_col_name

        # Execute the code
        exec(code, global_namespace)
        print("Code executed successfully.")
        print(f"Generated image URL: {image_url}")
    except Exception as e:
        error_message = f"Error executing generated code: {e}"
        print(error_message)
        return json.dumps({
            "image_url": None,
            "messages": error_message
        })

    return json.dumps({
        "image_url": image_url,
        "messages": "Plot generated successfully."
    })





