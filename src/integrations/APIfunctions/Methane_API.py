import sqlite3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import time
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
import json
from typing import Annotated, List, Optional, Dict
import os
import uuid
from langgraph.prebuilt import InjectedState
from src.integrations.APIfunctions.Methane_cities import cities
from src.integrations.APIfunctions.Methane_regions import regions
from sqlalchemy import create_engine
import psycopg2

# Define the sqlite and postgresql connection string:
postgres_uri = "postgresql+psycopg2://postgres:!Rbk241001@database-1.cxss6cwyw2ck.us-east-2.rds.amazonaws.com:5433/postgres"
sqlite_db_path = "G:/Methane_Data/methane_data_monthly.db"

def fetch_and_process_data(date_start, 
                           date_end, 
                           db_type='sqlite', 
                           sqlite_db_path=sqlite_db_path, 
                           postgres_uri=postgres_uri, 
                           region_name=None, N=None, S=None, W=None, E=None):
    """
    Fetch monthly methane data and other necessary parameters from the specified database (SQLite or PostgreSQL),
    filter out low outliers, and prepare the data for plotting.

    Parameters:
    - date_start (str): Start date in 'YYYY-MM-DD' format.
    - date_end (str): End date in 'YYYY-MM-DD' format.
    - db_type (str): Type of database ("sqlite" or "postgres").
    - sqlite_db_path (str, optional): Path to the SQLite database file (required if db_type is "sqlite").
    - postgres_uri (str, optional): SQLAlchemy connection string for PostgreSQL (required if db_type is "postgres").
    - region_name (str, optional): Name of the oil and gas region to filter (optional, requires defined regions).
    - N, S, W, E (float, optional): North, South, West, East coordinates for a custom bounding box.

    Returns:
    - ch4_raw_month (np.array): Array of raw methane values for each month.
    - dry_air_density_month (np.array): Array of average dry air density for each month.
    - wind_u_month (np.array): Array of average zonal wind velocity for each month.
    - wind_v_month (np.array): Array of average meridional wind velocity for each month.
    - lat_cr (np.array): Latitude values.
    - lon_cr (np.array): Longitude values.
    - month_unique (np.array): Unique month values.
    """

    # Check if a region name is provided, otherwise expect custom NSEW coordinates
    if region_name:
            # Assuming 'regions' is a dictionary with pre-defined bounding boxes
            region = regions.get(region_name, {})
            N, S, W, E = region.get("N"), region.get("S"), region.get("W"), region.get("E")
    elif N is None or S is None or W is None or E is None:
            raise ValueError("For custom bounding boxes, N, S, W, and E must be provided.")

    # Parameterized SQL query
    table_name = "monthly_methane_measurements" if db_type == "sqlite" else "methane_monthly_backup"
    query = f"""
        SELECT year_month, latitude, longitude, avg_CH4_mixing_ratio_ground,
               avg_dry_air_density, avg_wind_u, avg_wind_v
        FROM {table_name}
        WHERE latitude BETWEEN ? AND ?
          AND longitude BETWEEN ? AND ?
          AND year_month BETWEEN ? AND ?;
    """

    if db_type == "postgres":
        # Replace placeholders for PostgreSQL
        query = query.replace("?", "%s")

    try:
        start_time_pull = time.time()

        if db_type == "sqlite":
            if not sqlite_db_path:
                raise ValueError("sqlite_db_path must be provided for SQLite connection.")
            # Connect to SQLite
            conn = sqlite3.connect(sqlite_db_path)
            conn.execute("PRAGMA cache_size = 10000;")
            conn.execute("PRAGMA synchronous = OFF;")
            conn.execute("PRAGMA journal_mode = MEMORY;")
            # Execute the query
            df = pd.read_sql_query(query, conn, params=(S, N, W, E, date_start, date_end))
            conn.close()

        elif db_type == "postgres":
            if not postgres_uri:
                raise ValueError("postgres_uri must be provided for PostgreSQL connection.")
            # Connect to PostgreSQL
            engine = create_engine(postgres_uri)
            with engine.connect() as conn:
                df = pd.read_sql_query(query, conn, params=(S, N, W, E, date_start, date_end))

        else:
            raise ValueError("db_type must be 'sqlite' or 'postgres'.")

        elapsed_time_pull = time.time() - start_time_pull
        print(f"Data pulled successfully from {db_type} in {elapsed_time_pull:.2f} seconds.")

    except Exception as e:
        print(f"Error fetching data from the {db_type} database: {e}")
        return None

    # Ensure 'year_month' is in datetime format
    if 'year_month' in df.columns:
        df['year_month'] = pd.to_datetime(df['year_month'])

    # Reshape and process the data as before
    month_unique = df['year_month'].unique()
    ch4_raw_month = []
    dry_air_density_month = []
    wind_u_month = []
    wind_v_month = []

    lat_cr = np.flipud(df['latitude'].unique())
    lon_cr = df['longitude'].unique()

    for month in month_unique:
        df_month = df[df['year_month'] == month]

        ch4_month_temp = df_month.pivot(index='latitude', columns='longitude',
                                        values='avg_CH4_mixing_ratio_ground').values
        dry_air_density_temp = df_month.pivot(index='latitude', columns='longitude',
                                              values='avg_dry_air_density').values
        wind_u_temp = df_month.pivot(index='latitude', columns='longitude', values='avg_wind_u').values
        wind_v_temp = df_month.pivot(index='latitude', columns='longitude', values='avg_wind_v').values

        low_threshold = np.nanpercentile(ch4_month_temp, 5)
        ch4_month_temp[ch4_month_temp < low_threshold] = np.nan

        ch4_raw_month.append(np.flipud(ch4_month_temp))
        dry_air_density_month.append(np.flipud(dry_air_density_temp))
        wind_u_month.append(np.flipud(wind_u_temp))
        wind_v_month.append(np.flipud(wind_v_temp))

    ch4_raw_month = np.stack(ch4_raw_month)
    dry_air_density_month = np.stack(dry_air_density_month)
    wind_u_month = np.stack(wind_u_month)
    wind_v_month = np.stack(wind_v_month)

    return ch4_raw_month, dry_air_density_month, wind_u_month, wind_v_month, lat_cr, lon_cr, month_unique



def plot_methane_data_plotly(ch4_raw_month, lat_cr, lon_cr, month_unique):
    """
    Plot methane data for each month on an interactive map with Plotly.

    Parameters:
    - ch4_raw_month (np.array): Array of methane values for each month.
    - lat_cr (np.array): Latitude values.
    - lon_cr (np.array): Longitude values.
    - month_unique (np.array): Unique month values.
    """
    for i, month in enumerate(month_unique):
        # Create figure
        fig = go.Figure()

        # Add methane data as heatmap
        fig.add_trace(go.Heatmap(
            z=ch4_raw_month[i],
            x=lon_cr,
            y=lat_cr,
            colorscale='Jet',
            colorbar=dict(title='CH4 Mixing Ratio (ppb)')
        ))

        # Add map features
        fig.update_geos(
            visible=True, resolution=50,
            showcountries=True, countrycolor='Black',
            showsubunits=True, subunitcolor='Blue',
            showcoastlines=True, coastlinecolor='LightBlue',
            showland=True, landcolor='rgb(229, 229, 229)',
            showlakes=True, lakecolor='LightBlue'
        )

        # Update layout for map visualization
        fig.update_layout(
            title=f'CH4 Mixing Ratio - {month.strftime("%Y-%m")}',
            title_font=dict(size=20, family='Arial', color='black'),
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            template='plotly_white',
            autosize=False,
            width=1000,
            height=800,
        )

        # Show interactive plot
        pio.renderers.default = 'browser'
        fig.show()


# Plot methane data and add map with cities
def plot_methane_data(ch4_raw_month, 
                      lat_cr, 
                      lon_cr, 
                      month_unique):
    """
    Plot methane concentration in the unit of ppb for each month on a map with Cartopy and add major cities and provincial boundaries.
    using a static plot, save each plot locally, and return a series of URLs to access all the plots.

    Parameters:
    - ch4_raw_month (np.array): Array of methane values for each month.
    - lat_cr (np.array): Latitude values.
    - lon_cr (np.array): Longitude values.
    - month_unique (np.array): Unique month values.
    """

    image_urls = []
    try:
        for i, month in enumerate(month_unique):
            # Create the scatter plot with Cartopy
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()}) # , dpi=300

            # Set the extent of the plot (map boundaries) based on lat/lon
            ax.set_extent([lon_cr.min(), lon_cr.max(), lat_cr.min(), lat_cr.max()], crs=ccrs.PlateCarree())

            # Calculate the latitude range for aspect ratio correction
            lat_range = lat_cr.max() - lat_cr.min()
            lon_range = lon_cr.max() - lon_cr.min()

            # Adjust aspect ratio based on lat/lon ranges to preserve real scale
            ax.set_aspect(abs(lon_range / lat_range))

            # Add high-resolution map features
            ax.add_feature(cfeature.LAND, edgecolor='black', zorder=1)
            ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=2)
            ax.add_feature(cfeature.COASTLINE, linewidth=1.0, zorder=3)
            ax.add_feature(cfeature.OCEAN, color='lightblue', zorder=1)
            ax.add_feature(cfeature.LAKES, color='lightblue', zorder=1)
            ax.add_feature(cfeature.RIVERS, zorder=2)

            # Add gridlines
            gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), color='gray', linestyle='--', linewidth=0.5)
            gl.top_labels = False
            gl.right_labels = False

            # Add provincial/state boundaries using Cartopy's Natural Earth dataset
            provinces = cfeature.NaturalEarthFeature(
                category='cultural',
                name='admin_1_states_provinces_lines',
                scale='10m',  # Use high resolution (10m)
                facecolor='none'
            )
            ax.add_feature(provinces, edgecolor='black', linewidth=1.0, zorder=4)

            # Plot methane data
            scatter = ax.pcolormesh(lon_cr, lat_cr, ch4_raw_month[i], cmap='jet', transform=ccrs.PlateCarree())

            # Add a color bar
            plt.colorbar(scatter, ax=ax, label='CH4 Mixing Ratio (ppb)')

            # Filter cities within the boundaries of the plot
            lon_min, lon_max = lon_cr.min(), lon_cr.max()
            lat_min, lat_max = lat_cr.min(), lat_cr.max()
            filtered_cities = [city for city in cities if
                            lon_min <= city['lon'] <= lon_max and lat_min <= city['lat'] <= lat_max]

            # Add city names to the plot
            for city in filtered_cities:
                ax.text(city['lon'], city['lat'], city['name'], transform=ccrs.PlateCarree(),
                        fontsize=10, fontweight='bold', color='black', zorder=5,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

            # Set title and labels
            plt.title(f"CH4 Mixing Ratio - {month.strftime('%Y-%m')}")
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')

            # Generate a unique filename and save the plot
            IMAGE_DIR = "src/static/images"  # Make sure this directory exists and is served by FastAPI
            file_name = f"{uuid.uuid4().hex[:6]}.png"
            file_path = os.path.join(IMAGE_DIR, file_name)
            plt.savefig(file_path)
            plt.close()
            # Check if the file was saved successfully
            if os.path.exists(file_path):
                print(f"File successfully saved: {file_path}")
            else:
                print(f"Error: File not found at {file_path}")

            # Construct the URL to access the saved image
            image_url = f"http://localhost:800/static/images/{file_name}"
            print(f"Generated image URL: {image_url}")

            # Append the URL to the list
            image_urls.append(image_url)

        return json.dumps({
                    "image_urls": image_urls,
                    "messages": "Plots complete"
                })

    except Exception as e:
        print(f"Error during plot generation: {e}")
        # Return the URLs in JSON format
        
    


# IDW Interpolation with Distance Cut-Off Radius
def idwfill(inputarray, d=0, p=2):
    """
    Perform Inverse Distance Weighting (IDW) to fill missing values in a 2D array.

    Parameters:
    - inputarray (np.array): 2D array with NaNs representing missing values.
    - d (float): Distance cut-off radius (in grid units). Use '0' for no limit.
    - p (int): Weighting power.

    Returns:
    - np.array: Filled 2D array with missing values interpolated.
    """
    X = inputarray
    Y = inputarray * 1  # Copy of the input array to store results

    # Find indices of NaN (missing) and non-NaN (available) values
    nan_idx = np.argwhere(np.isnan(X))
    valid_idx = np.argwhere(~np.isnan(X))

    rn, cn = nan_idx[:, 0], nan_idx[:, 1]  # Rows and columns of NaN values
    r, c = valid_idx[:, 0], valid_idx[:, 1]  # Rows and columns of valid values

    # If distance cut-off radius is specified, square it for distance comparisons
    if d > 0:
        d_squared = d ** 2

    # Iterate over all missing values to interpolate
    for k in range(len(rn)):
        # Calculate squared distance to all valid points
        D = (rn[k] - r) ** 2 + (cn[k] - c) ** 2

        if d > 0:
            # Apply cut-off radius
            within_cutoff = D < d_squared
            if any(within_cutoff):
                # Calculate weights and interpolate
                weights = 1 / (np.sqrt(D[within_cutoff]) ** p)
                Y[rn[k], cn[k]] = np.sum(X[r[within_cutoff], c[within_cutoff]] * weights) / np.sum(weights)
            else:
                # If no points within cut-off radius, set value to zero
                Y[rn[k], cn[k]] = 0
        else:
            # No cut-off radius, use all available points
            weights = 1 / (np.sqrt(D) ** p)
            Y[rn[k], cn[k]] = np.sum(X[r, c] * weights) / np.sum(weights)

    return Y


def calculate_methane_emission(ch4_raw_month, dry_air_density_month, wind_u_month, wind_v_month, lat_cr, lon_cr,
                               month_unique):
    """
    Estimate background methane concentration, obtain methane enhancements, and calculate methane emission rate.

    Parameters:
   - ch4_raw_month (np.array): Array of raw methane values for each month.
   - dry_air_density_month (np.array): Array of average dry air density for each month.
   - wind_u_month (np.array): Array of average zonal wind velocity for each month.
   - wind_v_month (np.array): Array of average meridional wind velocity for each month.
   - lat_cr (np.array): Latitude values.
   - lon_cr (np.array): Longitude values.
   - month_unique (np.array): Unique month values.

    Returns:
    - 'Month'(list): date_month
    - 'EmissionRate'(list): Calculated Methane Emission Rate in the unit of tons/year
    - 'Background'(list): ch4_bg_month(Estimated Background methane concentration, in the unit of ppb)
    """
    delta_lat = 5500  # grid size in meters
    delta_lon = 4700
    molar_ch4 = 16.04  # molar mass of methane (g/mol)
    unit_conversion = 1e-9 / 1e6 * 3600 * 24 * 365  # Convert from ppb to tons per year

    netflux = np.zeros(len(month_unique))
    ch4_bg_month = np.zeros(len(month_unique))

    # Iterate through each month to calculate methane enhancement and emissions
    for month in range(len(month_unique)):
        ch4_month_temp = ch4_raw_month[month, :, :]
        rho_temp = dry_air_density_month[month, :, :]
        windu_temp = wind_u_month[month, :, :]
        windv_temp = wind_v_month[month, :, :]

        # Calculate background methane (10th percentile)
        ch4_bg_month[month] = np.nanpercentile(ch4_month_temp, 10)

        # Calculate methane enhancement by subtracting background methane
        ch4_enhance_temp = ch4_month_temp - ch4_bg_month[month]
        ch4_enhance_temp[ch4_enhance_temp < -20] = np.nan  # Remove extreme low values

        # Interpolate to fill in missing values in methane enhancement, wind, and density
        ch4_enhance_temp = idwfill(ch4_enhance_temp, d=5, p=2)
        rho_temp = idwfill(rho_temp, d=0, p=2)
        windu_temp = idwfill(windu_temp, d=0, p=2)
        windv_temp = idwfill(windv_temp, d=0, p=2)

        # Calculate Outflow Flux for Methane Emission Estimation
        # Horizontal Flux - Left boundary
        idx_l = np.argwhere(windu_temp[:, 0] < 0)
        flux_outflow_l = 0
        if len(idx_l) > 0:
            flux_outflow_l = np.sum(
                np.abs(windu_temp[idx_l, 0]) * molar_ch4 * rho_temp[idx_l, 0] * delta_lat * ch4_enhance_temp[idx_l, 0])

        # Horizontal Flux - Right boundary
        idx_r = np.argwhere(windu_temp[:, -1] > 0)
        flux_outflow_r = 0
        if len(idx_r) > 0:
            flux_outflow_r = np.sum(
                np.abs(windu_temp[idx_r, -1]) * molar_ch4 * rho_temp[idx_r, -1] * delta_lat * ch4_enhance_temp[
                    idx_r, -1])

        # Vertical Flux - Upper boundary
        idx_u = np.argwhere(windv_temp[0, :] > 0)
        flux_outflow_u = 0
        if len(idx_u) > 0:
            flux_outflow_u = np.sum(
                np.abs(windv_temp[0, idx_u]) * molar_ch4 * rho_temp[0, idx_u] * delta_lon * ch4_enhance_temp[0, idx_u])

        # Vertical Flux - Lower boundary
        idx_d = np.argwhere(windv_temp[-1, :] < 0)
        flux_outflow_d = 0
        if len(idx_d) > 0:
            flux_outflow_d = np.sum(
                np.abs(windv_temp[-1, idx_d]) * molar_ch4 * rho_temp[-1, idx_d] * delta_lon * ch4_enhance_temp[
                    -1, idx_d])

        # Calculate net flux for the month
        net_outflow = flux_outflow_l + flux_outflow_r + flux_outflow_u + flux_outflow_d
        netflux[month] = net_outflow * unit_conversion

    # # Calculate annual averages
    # emission_avg = np.mean(netflux)
    # annual_avg = [np.mean(netflux[i:i+12]) for i in range(0, len(netflux), 12)]

    # Create result dataframe
    date_month = pd.date_range(month_unique[0], periods=len(month_unique), freq='MS').strftime("%Y-%m").tolist()
    result = {
        'Month': date_month,
        'EmissionRate': netflux.tolist(),
        'Background': ch4_bg_month.tolist()
    }

    return result


def plot_methane_emission_interactive(result):
    """
    Plot the monthly methane emission rate and background values using an interactive plot.

    Parameters:
    - result (dict): Dictionary containing 'Month', 'EmissionRate', and 'Background'.
    """
    months = result['Month']
    emission_rate = result['EmissionRate']
    background = result['Background']

    # Create a figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add methane emission rate trace
    fig.add_trace(
        go.Scatter(x=months, y=emission_rate, mode='lines+markers', name='Methane Emission Rate (tons/year)',
                   line=dict(color='royalblue', width=2), marker=dict(size=6)),
        secondary_y=False,
    )

    # Add background methane trace
    fig.add_trace(
        go.Scatter(x=months, y=background, mode='lines+markers', name='Background Methane (ppb)',
                   line=dict(color='seagreen', width=2, dash='dash'), marker=dict(size=6, symbol='x')),
        secondary_y=True,
    )

    # Update layout for better visualization
    fig.update_layout(
        title_text='Monthly Methane Emission Rate and Background',
        title_font=dict(size=20, family='Arial', color='black'),
        xaxis_title='Month',
        yaxis_title='Methane Emission Rate (tons/year)',
        yaxis2_title='Background Methane (ppb)',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.5)', bordercolor='black', borderwidth=1),
        template='plotly_white',
        xaxis_tickangle=-45,
    )

    # Update axes properties
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')

    # Show interactive plot
    pio.renderers.default = 'browser'
    fig.show()


def plot_methane_emission_rate_static(result):

    """
    Plot the monthly methane emission rate and background values using a static plot,
    save it locally, and return the URL to access it.

    Parameters:
    - result (dict): Dictionary containing 'Month', 'EmissionRate', and 'Background'.
    """
    # Extract data from the result dictionary
    months = result['Month']
    emission_rate = result['EmissionRate']
    background = result['Background']

    # Create figure and axis with increased dpi and flexible figure size
    fig, ax1 = plt.subplots(figsize=(max(10, len(months) * 0.5), 6), dpi=300)

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



def _methane_analysis_pipeline(date_start: str, date_end: str, region_name=None, N=None, S=None, W=None, E=None):
    # Debug statement
    print(
        f"_methane_analysis_pipeline received: date_start={date_start}, date_end={date_end}, region_name={region_name}, N={N}, S={S}, W={W}, E={E}")
    # Fetch and process data
    print("Step 1: Fetching and processing data...")
    if region_name:
        # Fetch data using the region name
        ch4_raw_month, dry_air_density_month, wind_u_month, wind_v_month, lat_cr, lon_cr, month_unique = fetch_and_process_data(
            date_start, date_end, region_name=region_name
        )
    else:
        # Validate that all coordinates are provided if no region name is given
        if N is None or S is None or W is None or E is None:
            raise ValueError("For a custom bounding box, all of N, S, W, and E must be provided.")

        # Fetch data using custom bounding box coordinates
        ch4_raw_month, dry_air_density_month, wind_u_month, wind_v_month, lat_cr, lon_cr, month_unique = fetch_and_process_data(
            date_start, date_end, N=N, S=S, W=W, E=E
        )
    print(f"ch4_raw_month: {ch4_raw_month.shape if hasattr(ch4_raw_month, 'shape') else len(ch4_raw_month)}")
    print(f"lat_cr: {lat_cr.shape if hasattr(lat_cr, 'shape') else len(lat_cr)}")
    print(f"lon_cr: {lon_cr.shape if hasattr(lon_cr, 'shape') else len(lon_cr)}")
    print(f"month_unique: {month_unique}")
    # Plot methane data
    # try:
    # # Visualize methane concentration
    #     print("Step 2: Visualizing methane mixing ratio...")
    #     plot_methane_data(ch4_raw_month, lat_cr, lon_cr, month_unique)
    # except Exception as e:
    #     print(f"Error during methane data plotting: {e}")

    # # Calculate methane emission
    # print("Step 3: Calculating methane emission...")
    # result = calculate_methane_emission(
    #     ch4_raw_month, dry_air_density_month, wind_u_month, wind_v_month, lat_cr, lon_cr, month_unique
    # )
    # return json.dumps(result)
       # Initialize the result dictionary
    result = {}

    # Plot methane data
    try:
        # Visualize methane concentration
        print("Step 2: Visualizing methane mixing ratio...")
        plot_results = plot_methane_data(ch4_raw_month, lat_cr, lon_cr, month_unique)
        
        # Parse image URLs from plot results
        if isinstance(plot_results, str):
            plot_results = json.loads(plot_results)
        result["image_urls"] = plot_results.get("image_urls", [])
    except Exception as e:
        print(f"Error during methane data plotting: {e}")
        result["image_urls"] = []
        result["plot_error"] = str(e)

    # Calculate methane emission
    print("Step 3: Calculating methane emission...")
    try:
        emission_result = calculate_methane_emission(
            ch4_raw_month, dry_air_density_month, wind_u_month, wind_v_month, lat_cr, lon_cr, month_unique
        )
        result["emission_data"] = emission_result
    except Exception as e:
        print(f"Error during methane emission calculation: {e}")
        result["emission_error"] = str(e)

    return json.dumps(result)