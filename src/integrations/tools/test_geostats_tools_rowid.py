import json
from langchain_core.tools import tool
import uuid
from typing import Annotated, Optional, Tuple
from langgraph.prebuilt import InjectedState
import os
import pandas as pd
import sqlite3
import ast
from pykrige.ok import OrdinaryKriging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

IMAGE_DIR = "src/static/images"

@tool
def geostats_interpolation(userID: Annotated[str, InjectedState("userID")],
                           rowid: Annotated[str, "The row in the running table containing the property to interpolate"],
                           lati_col_name: Annotated[str, "The column for latitude values"],
                           longi_col_name: Annotated[str, "The column for longitude values"],
                           property_col_name: Annotated[str, "The column for the feature to interpolate"],
                           target_mesh: Annotated[Optional[list[int]], "The output mesh size for the interpolated model, e.g., [20, 20]"]):
    """
    Perform 2D geostatistical interpolation, converting scattered points into a Cartesian mesh using Kriging.

    Inputs:
    - userID: The user's unique identifier.
    - rowid: The ID in the database to retrieve data.
    - lati_col_name, longi_col_name, property_col_name: Columns for latitude, longitude, and the target feature.
    - target_mesh: Int list specifying grid size for interpolation, e.g., [20, 20].

    Output:
    - JSON object containing the generated image URL and a success message.
    """
    try:
        # Database connection and data extraction
        db_name = os.path.join("database", f"{userID}.db")
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        table_name = "running"
        
        cursor.execute(f"SELECT data FROM {table_name} WHERE rowid = ?", (rowid,))
        row = cursor.fetchone()
        if not row:
            return json.dumps({"error": "No data found for the given rowid"})
        
        # Extracting data
        data = row[0]
        dictionary = ast.literal_eval(data)
        latitudes = [float(val) for val in dictionary[lati_col_name]]
        longitudes = [float(val) for val in dictionary[longi_col_name]]
        properties = [float(val) for val in dictionary[property_col_name]]
        
        # Variogram model configuration
        variogram_model = 'gaussian'
        variogram_parameters = {
            'sill': 1.0,
            'range': 0.5,
            'nugget': 0.1
        }

        # Kriging model instantiation
        kriging_model = OrdinaryKriging(
            longitudes, latitudes, properties,
            variogram_model=variogram_model,
            verbose=False,
            enable_plotting=False,
            variogram_parameters=variogram_parameters
        )
        
        # Creating the grid for interpolation
        grid_x = target_mesh[0] if target_mesh else 20
        grid_y = target_mesh[1] if target_mesh else 20
        lon_loc = [min(longitudes) + i * (max(longitudes) - min(longitudes)) / (grid_x - 1) for i in range(grid_x)]
        lat_loc = [min(latitudes) + i * (max(latitudes) - min(latitudes)) / (grid_y - 1) for i in range(grid_y)]
        
        # Execute interpolation
        z, ss = kriging_model.execute("grid", lon_loc, lat_loc)

        # Plotting the results
        plt.figure()
        plt.contourf(lon_loc, lat_loc, z, cmap="viridis")
        plt.colorbar(label=property_col_name)
        plt.grid(True, color="gray", linestyle="-", linewidth=0.5)

        plt.xticks(lon_loc)  # Set x-ticks at longitude locations
        plt.yticks(lat_loc) 

        file_name = f"{uuid.uuid4()}.png"
        file_path = os.path.join(IMAGE_DIR, file_name)
        plt.savefig(file_path)
        plt.close()

        # Generating image URL
        image_url = f"http://localhost:800/static/images/{file_name}"

        return json.dumps({
            "image_url": image_url,
            "message": "Interpolation and plot completed successfully."
        })
    except Exception as e:
        return json.dumps({"error": f"Error: {str(e)}"})
