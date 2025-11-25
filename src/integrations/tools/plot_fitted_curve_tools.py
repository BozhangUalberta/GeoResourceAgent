import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from langchain_core.tools import tool
import json
from typing import Annotated, List, Optional, Dict
from langgraph.prebuilt import InjectedState
import os
import uuid

IMAGE_DIR = "src/static/images"

@tool
def plot_fitted_curve_tool(
        original_data: Annotated[List[float], "The original production curve."], 
        fitted_data: Annotated[List[float], "The fitted curve form DCA, if select_uploaded_file is 'NO'"],
        EUR: Annotated[Optional[float], "Cumulative production, retrieved from the conversation, if not available then None"] = None,
        b: Annotated[Optional[float], "b factor from the Decline analysis (DCA), if not available then None"] = None,
        Di: Annotated[Optional[float], "Di factor from the Decline analysis (DCA), if not available then None"] = None,
        plot_x_label: Annotated[Optional[str], 'x_label'] = None,
        plot_y_label: Annotated[Optional[str], 'y_label'] = None,
        plot_title: Annotated[Optional[str], 'plot_title'] = None
):
    """
    This tool is used to plot fitted production curves. The plot should include the original curve and the fitted curve.
    The input sequence will be a list of the original and the fitted production profile.
    The output contains the image unique url and plot message.
    """

    print('========== Plot the fitted curve, conversation source ===========')

    messages = {}
    # Set default labels if not provided
    if plot_x_label is None:
        plot_x_label = 'Elapsed Production Time'
        messages['x_label'] = 'x_label unit missing'
    if plot_y_label is None:
        plot_y_label = 'Production Rate'
        messages['y_label'] = 'y_label unit missing'
    if plot_title is None:
        plot_title = 'Production Curve'
    
    plt.figure(figsize=(8,5))  # Set figure size
    plt.grid(True) 

    if original_data is None:
        original_data = []
    if fitted_data == None or fitted_data == [] :
        fitted_data = []
        raise ValueError("The production data is empty, please input the production again.")
    
    original_length = len(original_data) 
    fiited_length = len(fitted_data)
    plt.plot(list(range(original_length)), original_data, 'k-', alpha=0.3, label='Original Data')
    plt.plot(list(range(fiited_length)), fitted_data, 'b-', alpha=0.3, label='Predicted Data')
        
    # Plot settings
    plt.title(plot_title)
    plt.xlabel(plot_x_label)
    plt.ylabel(plot_y_label)

    # Adding info on the curve
    if EUR is not None:
        plt.text(0.8, 0.85, f'EUR: {EUR:.3f}', transform=plt.gca().transAxes)
    if b is not None:
        plt.text(0.8, 0.8, f'b-factor: {b:.5f}', transform=plt.gca().transAxes)
    if Di is not None:
        plt.text(0.8, 0.75, f'Di: {Di:.5f}', transform=plt.gca().transAxes)

    plt.legend()

    # Save the plot to a unique file
    file_name = f"{uuid.uuid4().hex[:6]}.png"
    file_path = os.path.join(IMAGE_DIR, file_name)
    plt.savefig(file_path)
    plt.close()

    image_url = f"http://localhost:800/static/images/{file_name}"
    print(f"Generated image URL: {image_url}")

    return json.dumps({
        "image_url": f"Image URL: {image_url}",
        "messages": "Plot complete"
    })


########################################
####### DCA Table source ########
########################################

import sqlite3

def load_data_to_dataframe(db_name, table_name):
    # Load the entire table into a DataFrame
    conn = sqlite3.connect(db_name)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

@tool
def plot_fitted_curve_tool_table_source(
        userID: Annotated[str, InjectedState("userID")],
        table_name: Annotated[str, "The table containing the user's required production data"],
        org_data_col_name: Annotated[str, "Column name for original production data"],
        fitted_data_col_name: Annotated[str, "Column name for fitted production data"],
        EUR: Annotated[Optional[float], "Cumulative production, retrieved from the conversation, if not available then None"] = None,
        b: Annotated[Optional[float], "b factor from the Decline analysis (DCA), if not available then None"] = None,
        Di: Annotated[Optional[float], "Di factor from the Decline analysis (DCA), if not available then None"] = None,
        plot_x_label: Annotated[Optional[str], 'x_label'] = None,
        plot_y_label: Annotated[Optional[str], 'y_label'] = None,
        plot_title: Annotated[Optional[str], 'plot_title'] = None
):
    """
    This tool is used to plot fitted production curves. The plot should include the original curve and the fitted curve.
    The input sequence will be a list of the original and the fitted production profile.
    The output contains the image unique url and plot message.
    """

    print('========== Plot the fitted curve, table source ===========')
    db_name = os.path.join("database", f"{userID}.db")
    df = load_data_to_dataframe(db_name, table_name)
    df.dropna(axis=1, how='all', inplace=True)
    original_data = df[org_data_col_name]
    fitted_data = df[fitted_data_col_name]

    messages = {}
    # Set default labels if not provided
    if plot_x_label is None:
        plot_x_label = 'Elapsed Production Time'
        messages['x_label'] = 'x_label unit missing'
    if plot_y_label is None:
        plot_y_label = 'Production Rate'
        messages['y_label'] = 'y_label unit missing'
    if plot_title is None:
        plot_title = 'Production Curve'
    
    plt.figure(figsize=(8,5))  # Set figure size
    plt.grid(True) 
    
    original_length = len(original_data) 
    fiited_length = len(fitted_data)
    plt.plot(list(range(original_length)), original_data, 'k-', alpha=0.3, label='Original Data')
    plt.plot(list(range(fiited_length)), fitted_data, 'b-', alpha=0.3, label='Predicted Data')
        
    # Plot settings
    plt.title(plot_title)
    plt.xlabel(plot_x_label)
    plt.ylabel(plot_y_label)

    # Adding info on the curve
    if EUR is not None:
        plt.text(0.8, 0.85, f'EUR: {EUR:.3f}', transform=plt.gca().transAxes)
    if b is not None:
        plt.text(0.8, 0.8, f'b-factor: {b:.5f}', transform=plt.gca().transAxes)
    if Di is not None:
        plt.text(0.8, 0.75, f'Di: {Di:.5f}', transform=plt.gca().transAxes)

    plt.legend()

    # Save the plot to a unique file
    file_name = f"{uuid.uuid4().hex[:6]}.png"
    file_path = os.path.join(IMAGE_DIR, file_name)
    plt.savefig(file_path)
    plt.close()

    image_url = f"http://localhost:800/static/images/{file_name}"
    print(f"Generated image URL: {image_url}")

    return json.dumps({
        "image_url": f"Image URL: {image_url}",
        "messages": "Plot complete"
    })