import json
import os
import pandas as pd
from joblib import load
import pickle
from langchain_core.tools import tool
import uuid
from typing import Annotated, Optional, Union, List
from langgraph.prebuilt import InjectedState
from src.integrations.APIfunctions.geo_pred import geo_pred_Montney

import json
import matplotlib
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon

# Configure matplotlib to use a backend that doesn't require a display environment
matplotlib.use('Agg')

##########################################
####### Pretrained GeoPredict Tool #######
##########################################
IMAGE_DIR = "src/static/images"

@tool
def geopred_conv_source(play: Annotated[str, "Play name, e.g., Montney, Duvernay"],
                        longitude: Annotated[float, "Well longitude"],
                        latitude: Annotated[float, "Well latitude"],
                        interval: Annotated[str, "The interval"],
                        startYear: Annotated[int, "Year"],
                        month: Annotated[int, "Month"]):
    """
    Predicts geological features for Central Montney and Duvernay wells based on input coordinates and play name.
    The play must be selected from: [Montney, Duvernay].
    """

    # Prepare the model input
    model_input = {
        "Longitude": longitude,
        "Latitude": latitude,
        "ENVInterval": interval,
        "StartYear": startYear,
        "Month": month
    }

    # Load region edge data from JSON file
    try:
        with open("src/integrations/pretrained/play_region.json") as f:
            play_regions = json.load(f)

        play_edge = play_regions.get(play)
        
        # Check if play edge data exists for the specified play
        if play_edge is None:
            return json.dumps("Play not found in region data. Please check the play name.")
    except FileNotFoundError:
        return json.dumps("Region data file not found.")

    # Convert region boundary points to a Polygon
    region_polygon = Polygon(play_edge)  # Assuming play_edge is a list of (longitude, latitude) tuples

    # Plot the region boundary and well location
    plt.figure(figsize=(8, 6))
    y, x = zip(*play_edge)
    plt.fill(x, y, color='lightblue', alpha=0.5, label=f"{play} Region Boundary")
    plt.scatter(longitude, latitude, color='red', label="Well Location")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"{play} Region and Well Location")
    plt.legend()
    plt.show()

    # Check if the well location is within the region
    well_point = Point(latitude,longitude)
    if not region_polygon.contains(well_point):
        return json.dumps("Out of range, input again")

    # Perform prediction based on the play
    if play == "Montney":
        pred_results = geo_pred_Montney(model_input)
    else:
        return json.dumps("No such region")
    
    # Save plot
    file_name = f"{str(uuid.uuid4())}.png"
    file_path = os.path.join(IMAGE_DIR, file_name)
    plt.savefig(file_path)
    plt.close()

    # Generating image URL
    try:
        host_ip = os.getenv("HOST_IP")
        host_port = int(os.getenv("HOST_PORT"))
    except Exception as e:
        print(f"Error while finding ip and port from env variable when assembling image url.")
    
    image_url = f"http://{host_ip}:{host_port}/static/images/{file_name}"

    return json.dumps({
        "image_url": image_url,
        "pred_results": pred_results,
    })
