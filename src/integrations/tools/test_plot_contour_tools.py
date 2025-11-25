
from langchain_core.tools import tool
import json
from typing import Annotated, Optional

import json

@tool
def plot_contour_tool(
    file_id: Annotated[str, "The name of the CSV file."],
    property_name: Annotated[str, "The name of the feature (column) to plot."],
    grid: Annotated[Optional[str], "Triangular or Cartesian grid type. Default is Triangular."] = None,
    levels: Annotated[Optional[int], "The number of contour levels."] = None,
    smooth_level: Annotated[Optional[int], "Smoothness of the contour map."] = None
) -> Annotated[str, "A message indicating the plot status."]:
    """
    Generates a contour plot based on the provided CSV file and the specified property.
    
    Args:
        file_id (str): The name of the CSV file containing the data.
        property_name (str): The feature (column) to plot on the contour map.
        grid (str, optional): The type of grid to use (Triangular or Cartesian). Default is Triangular.
        levels (int, optional): The number of contour levels to use. Defaults to automatic levels if not specified.
        smooth_level (int, optional): The smoothness level for the contour map.

    Returns:
        - JSON object with a message indicating whether the plot was successful or if an error occurred.
    """
    try:
        # Simulate processing and plotting
        plot_status = f"Generated contour plot for '{property_name}' from file '{file_id}'."

        # Return success message
        output_results = {"Contour plot message": plot_status}
        return json.dumps(output_results)

    except Exception as e:
        # Handle any exceptions and return an error message
        return json.dumps({"Contour plot message": f"Error: {str(e)}"})