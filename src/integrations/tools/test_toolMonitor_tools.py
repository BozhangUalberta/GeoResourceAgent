import json
from langchain_core.tools import tool
from typing import Annotated, Optional

import json

@tool
def tool_monitor(
    las_file_id: str | None,
    dat_file_id: str | None,
    excel_file_id: str | None,
    location_info: str | None,
    play_info: str | None,
    production_data_file_id: str | None,
    dca_performed: bool | None
) -> Annotated[str, "A message indicating the availability of each tool."]:
    """
    Monitors the availability of various tools based on the dependencies provided.
    It checks if LAS data, location info, play info, and production data are available to determine which tools can be used.

    Args:
        las_file_id (str, optional): The identifier of the LAS data file.
        dat_file_id (str, optional): The rowid of the .dat CMG file.
        location_info (str, optional): Location information of the well.
        play_info (str, optional): Play information for the well.
        production_data_file_id (str, optional): The identifier of the production data file.
        dca_performed (bool, optional): Has the DCA tool been performed already?

    Returns:
        - JSON object indicating whether each tool is available or unavailable.
    """

    tools_availability = {}

    # Always ready tools
    tools_availability["user_input_parser"] = "available"
    tools_availability["csv_reader"] = "available"
    tools_availability["database_query"] = "available"

    # Check for LAS data (this is the base tool)
    if las_file_id:
        tools_availability["las_reader"] = "available"
    else:
        tools_availability["las_reader"] = "unavailable"

    # Check for .dat data
    if dat_file_id:
        tools_availability["CMG_dat_reader"] = "available"
    else:
        tools_availability["CMG_dat_reader"] = "unavailable"

    # Check for geostats_interpolation_tools_db and plot_contour_tool (require LAS and location info)
    if las_file_id and location_info:
        tools_availability["geostats_interpolation_tools_db"] = "available"
        tools_availability["plot_contour_tool"] = "available"
    else:
        tools_availability["geostats_interpolation_tools_db"] = "unavailable"
        tools_availability["plot_contour_tool"] = "unavailable"

    # Check for geo_pred_pretrained and DL_pred_pretrain (require LAS, location, and play info)
    if las_file_id and location_info and play_info:
        tools_availability["geo_pred_pretrained"] = "available"
        tools_availability["DL_pred_pretrain"] = "available"
    else:
        tools_availability["geo_pred_pretrained"] = "unavailable"
        tools_availability["DL_pred_pretrain"] = "unavailable"

    # Check for DCA_tool and plot_fitted_curve_tool (require production data)
    if production_data_file_id:
        tools_availability["DCA_tool"] = "available"
        tools_availability["plot_fitted_curve_tool"] = "available"
    else:
        tools_availability["DCA_tool"] = "unavailable"
        tools_availability["plot_fitted_curve_tool"] = "unavailable"

    # Check for NPV_tool (requires production data)
    if production_data_file_id:
        tools_availability["NPV_tool"] = "available"
    else:
        tools_availability["NPV_tool"] = "unavailable"

    # Check for online_fit (requires LAS, location, and production data)
    if las_file_id and location_info and production_data_file_id:
        tools_availability["online_fit"] = "available"
    else:
        tools_availability["online_fit"] = "unavailable"

    # Return the availability status of all tools as a JSON object
    return json.dumps(tools_availability)
