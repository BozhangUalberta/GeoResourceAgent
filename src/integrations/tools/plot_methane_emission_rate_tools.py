
import json
import os
import uuid
from langchain_core.tools import tool
import matplotlib.pyplot as plt
from src.integrations.APIfunctions.Methane_API import plot_methane_emission_rate_static

@tool("plot_methane_emission_rate", return_direct=False)
def plot_methane_emission_rate(result):
    """
    Plot the monthly methane emission rate and background values from the Methane_Emission_tools results. using a static plot,
    save it locally, and return the URL to access it.

    Parameters:
    - result (dict): the output from Methane_Emission_tools, Dictionary containing 'Month', 'EmissionRate', and 'Background'.
    Use appropreate unit for EmissionRate values (i.e. thousand tons/year, million tons/year etc.)
    """
    # Extract data from the result dictionary
    months = result['Month']
    emission_rate = result['EmissionRate']
    background = result['Background']

    # Ensure all lists are the same length
    if not (len(months) == len(emission_rate) == len(background)):
        raise ValueError("The lengths of 'Month', 'EmissionRate', and 'Background' must match.")

    # Create figure and axis with increased dpi and flexible figure size
    fig, ax1 = plt.subplots(figsize=(max(10, len(months) * 0.5), 6)) #dpi=300

    # Plot methane emission rate
    color = 'royalblue'
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Methane Emission Rate (tons/year)', color=color)
    ax1.plot(months, emission_rate, color=color, marker='o', linestyle='-', linewidth=2, markersize=6,
             label='Methane Emission Rate (tons/year)')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Create secondary y-axis for background methane
    ax2 = ax1.twinx()
    color = 'seagreen'
    ax2.set_ylabel('Background Methane (ppb)', color=color)
    ax2.plot(months, background, color=color, marker='x', linestyle='--', linewidth=2, markersize=6,
             label='Background Methane (ppb)')
    ax2.tick_params(axis='y', labelcolor='black')

    # Add title and layout settings
    plt.title('Monthly Methane Emission Rate and Background')
    fig.tight_layout()  # Adjust layout to prevent overlap

    # Generate a unique filename and save the plot
    IMAGE_DIR = "src/static/images"  # Make sure this directory exists and is served by FastAPI
    file_name = f"{uuid.uuid4().hex[:6]}.png"
    file_path = os.path.join(IMAGE_DIR, file_name)
    plt.savefig(file_path)
    plt.close()

    # Construct the URL to access the saved image
    image_url = f"http://localhost:800/static/images/{file_name}"
    print(f"Generated image URL: {image_url}")

    # Return the URL in JSON format
    return json.dumps({
        "image_url": f"Image URL: {image_url}",
        "messages": "Plot complete"
    })
