
from langchain_core.tools import tool
from src.integrations.APIfunctions.Methane_API import _methane_analysis_pipeline, plot_methane_emission_rate_static

@tool("methane_analysis_pipeline", return_direct=False)
def methane_analysis_pipeline(
        date_start: str,
        date_end: str,
        region_name=None,
        N=None,
        S=None,
        W=None,
        E=None):
    """
    This function analyzes methane bounded by datetime and area.
    Steps:

    Fetch monthly methane data and other parameters.
    Plot methane data.
    Calculate monthly methane emission rates:
    Emission Rate: tons/year.
    Background methane concentration: ppb.
    (Optional) Plot monthly methane emission rates.
    Methane Data Source: Sentinel-5 Precursor Level 2 Methane.
    Rate Calculation: Based on the mass balance of 𝑋𝐶𝐻4 over a controlled volume (V). Emission rate is calculated using the divergence theorem and cross-sectional flux method.

    Parameters:

    date_start (str): Start date ('YYYY-MM-DD').
    date_end (str): End date ('YYYY-MM-DD').
    region_name (str): Name of oil & gas region (key in regions).
    N, S, W, E (float): North, South, West, East for custom bounding box.
    Note: User must provide either region_name or NSWE coordinates.
    """
    print(
        f"methane_analysis_pipeline arguments: date_start={date_start}, date_end={date_end}, region_name={region_name}, N={N}, S={S}, W={W}, E={E}")
    tool_args = {
        'date_start': date_start,
        'date_end': date_end,
        'region_name': region_name,
        'N': N,
        'S': S,
        'W': W,
        'E': E
    }

    result = _methane_analysis_pipeline(**tool_args)
    return result

