from langchain_core.tools import tool
import json
import uuid
import lasio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
from typing import Annotated, Optional
from langgraph.prebuilt import InjectedState
from src.utils.db_utils import get_data_sqlite3
import pandas as pd

SAVE_DIR = "src/static/{userID}"

@tool
def las_reader(userID: Annotated[str, InjectedState("userID")],
               rowid: Annotated[str, "User uploaded .las file"],
               depth_top: Annotated[Optional[float], "Payzone top depth"] = None,
               depth_bottom: Annotated[Optional[float], "Payzone bottom depth"] = None,
               longitude: Annotated[Optional[float], "Well longitude"] = None,
               latitude: Annotated[Optional[float], "Well latitude"] = None,
               plot: Annotated[Optional[bool], "default False, set to True if you want to plot the data"] = False):
    """
    Used to process LAS file for well logs. Reads a LAS file, performs some analysis, and can plot the well log.
    
    Returns:
        - The processed data with a .dat format, and an information message.
    """
    try:
        # Create user-specific save directory if it doesn't exist
        user_save_dir = SAVE_DIR.format(userID=userID)
        if not os.path.exists(user_save_dir):
            os.makedirs(user_save_dir)

        # Get the LAS file path from the database using the provided rowid
        data_path = get_data_sqlite3(filename="test.db", table=userID, id=rowid, type="userinput")
        file_name = os.path.basename(data_path)
        data_path = f"src/static/{userID}/{file_name}"  # Ensure correct file path format

        # Read the LAS file
        las = lasio.read(data_path)
        
        df = pd.DataFrame(las.data,columns=las.keys())

        # Basic info about the LAS file
        na_info = df.isna().sum()
        col_info = df.columns
        max_depth = np.max(df['DEPT'])
        min_depth = np.min(df['DEPT'])
        print(f"MAX depth: {max_depth}")

        # Filter data by payzone if depth_top and depth_bottom are provided
        if depth_top is not None and depth_bottom is not None:
            focused_df = df[(df['DEPT'] >= depth_top) & (df['DEPT'] <= depth_bottom)]
        else:
            focused_df = df

        # Create a new .dat file with basic well information and payzone data
        new_file_id = f"las_processed_file_{uuid.uuid4()}.dat"
        processed_file_path = os.path.join(user_save_dir, new_file_id)
        with open(processed_file_path, 'w') as f:
            f.write(f"Longitude: {longitude}\n")
            f.write(f"Latitude: {latitude}\n")
            f.write(f"Depth range: {depth_top} - {depth_bottom}\n")
            f.write(f"Columns: {', '.join(focused_df.columns)}\n")
            focused_df.to_csv(f, sep='\t', index=False)

        # Plot if requested
        if plot:
            from matplotlib.ticker import MultipleLocator
            fig, axs = plt.subplots(1, 5, figsize=(10, 10))  # Create 5 subplots for clarity
            y_major_locator = MultipleLocator(10)
            y_value = df['DEPT']
            log_col = df.columns
            x_value = [df[i] for i in log_col]

            for i, df_value in enumerate(x_value[:5]):  # Limit to 5 columns for plot clarity
                unit = las.get_curve(log_col[i]).unit
                axs[i].plot(df_value, y_value, 'k', alpha=0.8)
                axs[i].invert_yaxis()  # Invert the y-axis to match depth log convention
                axs[i].set_title(log_col[i])
                axs[i].yaxis.set_major_locator(y_major_locator)
                axs[i].grid(True)
                axs[i].set_xlabel(unit)

            plot_file_name = f"{uuid.uuid4()}.png"
            plot_file_path = os.path.join(user_save_dir, plot_file_name)
            plt.savefig(plot_file_path)
            plt.close()

            image_url = f"http://localhost:800/static/{userID}/{plot_file_name}"
        else:
            image_url = "Plotting not requested."

        # Return the successful operation result
        return json.dumps({
            "image_url": image_url,
            "processed_las_id": new_file_id,
            "message": f"Analysis complete. The LAS file contains columns: {col_info}. Missing data info: {na_info}. "
                       f"Max depth: {max_depth}. Min depth: {min_depth}. Processed file stored as {new_file_id}."
        })

    except Exception as e:
        # Catch any error and return the error message
        return json.dumps({"error": str(e)})
