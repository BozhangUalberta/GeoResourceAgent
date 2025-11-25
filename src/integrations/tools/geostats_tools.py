import json
import os
import uuid
import re
import ast
import sqlite3
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from typing import Annotated, Optional, Union, List
from pykrige.ok import OrdinaryKriging
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

# Configure matplotlib to use a backend that doesn't require a display environment
matplotlib.use('Agg')


########################################
####### Geological Interpolatin ########
########################################
IMAGE_DIR = "src/static/images"

def load_data_to_dataframe(db_name, table_name):
    # Load the entire table into a DataFrame
    conn = sqlite3.connect(db_name)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

# def sanitize_image_name(name):
#     # Allow only alphanumeric characters and underscores, and convert to lowercase
#     return re.sub(r'\W+', '_', name).lower()

from scipy.spatial import ConvexHull
import numpy as np

@tool
def geostats_interpolation(userID: Annotated[str, InjectedState("userID")],
                           table_name: Annotated[str, "The table containing the user's required production data"],
                           lati_col_name: Annotated[str, "The column for latitude values"],
                           longi_col_name: Annotated[str, "The column for longitude values"],
                           property_col_name: Annotated[str, "The column for the feature to interpolate, only accept one at once."],
                           target_mesh: Annotated[Optional[list[int]], "The output mesh size for the interpolated model, e.g., [20, 20]"],
                           plot_wells: Annotated[Optional[bool], "Show the well location used to fit the model"] = False,
                           region_hull: Annotated[Optional[bool], "Show the edge of the fitted region"] = False):
    
    """
    Perform 2D geostatistical interpolation, converting scattered points into a Cartesian mesh using Kriging.
    
    If `region_hull` is set to True, only the region within the convex hull of well locations is considered valid.

    Default `plot_wells` and `region_hull` are False.
    
    Output:
    - JSON object containing the generated image URL and a success message.
    """
    print('========== Geological interpolation ===========')
    # Database connection and data extraction
    db_name = os.path.join("database", f"{userID}.db")
    df = load_data_to_dataframe(db_name, table_name)
    df_selected = df[[lati_col_name, longi_col_name, property_col_name]].dropna()
    if df_selected.empty:
        return json.dumps({"error": "No data available after dropping missing values."})
    
    latitudes = df_selected[lati_col_name].astype(float).tolist()
    longitudes = df_selected[longi_col_name].astype(float).tolist()
    properties = df_selected[property_col_name].astype(float).tolist()
    
    if not latitudes or not longitudes or not properties:
        return json.dumps({"error": "Latitude, longitude, or property data is missing or invalid."})
    
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
        variogram_parameters = variogram_parameters
    )
    
    # Creating the grid for interpolation
    grid_x = target_mesh[0] if target_mesh else 20
    grid_y = target_mesh[1] if target_mesh else 20
    lon_loc = [min(longitudes) + i * (max(longitudes) - min(longitudes)) / (grid_x - 1) for i in range(grid_x)]
    lat_loc = [min(latitudes) + i * (max(latitudes) - min(latitudes)) / (grid_y - 1) for i in range(grid_y)]
    
    # Execute interpolation
    z, _ = kriging_model.execute("grid", lon_loc, lat_loc)

    # Create the figure before plotting
    plt.figure(figsize=(8, 6))

    # Apply convex hull mask if region_hull is True
    if region_hull:
        points = np.column_stack((longitudes, latitudes))
        hull = ConvexHull(points)

        # Create a mask for the grid points inside the hull
        hull_path = plt.Polygon(points[hull.vertices], closed=True)
        grid_lon, grid_lat = np.meshgrid(lon_loc, lat_loc)
        grid_points = np.c_[grid_lon.ravel(), grid_lat.ravel()]
        mask = np.array([hull_path.contains_point(point) for point in grid_points])
        mask = mask.reshape(grid_lon.shape)

        # Mask the interpolated data outside the convex hull
        z = np.ma.masked_where(~mask, z)

        # Plot the hull boundary
        plt.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'r--', lw=2, label="Region Hull")

    # Plotting the results
    plt.contourf(lon_loc, lat_loc, z, cmap="viridis")
    plt.colorbar(label=property_col_name)
    plt.grid(True, color="gray", linestyle="-", linewidth=0.5)

    # Scatter plot well locations if plot_wells is True
    if plot_wells:
        plt.scatter(longitudes, latitudes, c='gray', marker='o', label='Well Locations')

    # Add legend if necessary
    if plot_wells or region_hull:
        plt.legend(loc='upper right')

    # Set xticks and yticks to show at most 5 ticks
    max_ticks = 8
    lon_ticks = np.linspace(min(lon_loc), max(lon_loc), min(max_ticks, grid_x))
    lat_ticks = np.linspace(min(lat_loc), max(lat_loc), min(max_ticks, grid_y))
    plt.xticks(lon_ticks)
    plt.yticks(lat_ticks)

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
        "message": "Interpolation and plot completed successfully."
    })



########################################
######## Correlation Analysis ##########
########################################
@tool
def correlation_analysis(userID: Annotated[str, InjectedState("userID")],
                        table_name: Annotated[str, "The table containing the user's required data"],
                        property_col_name: Optional[Annotated[List[str], "A list of column names to include in the correlation analysis, e.g., ['column1', 'column2']"]] = None,
                        exclude_columns: Optional[Annotated[List[str], "A list of column names to exclude in the correlation analysis, e.g., ['column1', 'column2']"]] = None
):
    """
    This tool performs correlation analysis on the specified columns using a correlation matrix. It processes 
    categorical data, normalizes numeric features, and generates a heatmap to visualize correlations.
    People commonly like to exclude the well names related features like API_UWI, WELL_NAME, Well_ID, you can remind the user to exclude them.

    Inputs:
    - `userID`: A unique identifier for the user.
    - `table_name`: The name of the table containing the required data.
    - `property_col_name`: A list of column names to include in the analysis, specified as `['column1', 'column2']`. 
    If you're unsure of the exact column names, use the Summarizer to identify the correct column names.
    If not provided, all numeric columns will be included by default.
    """
    try:
        print("============== Correlation Analysis ==============")
        print(property_col_name)
        # Load data
        db_name = os.path.join("database", f"{userID}.db")
        df = load_data_to_dataframe(db_name, table_name)
        df.dropna(axis=1, how='all', inplace=True)

        if exclude_columns is None:
            exclude_columns = []
        
        if property_col_name is not None and exclude_columns:
            exclude_columns = []

        # Exclude columns that should not be processed or used as features
        if property_col_name:
            df = df[property_col_name]
        else: 
            df = df.drop(columns=exclude_columns, errors='ignore')

        # Initialize a dictionary to store label encodings for reference
        label_encodings = {}
        
        # Step 1: Process categorical columns
        for column in df.columns:
            if exclude_columns and column in exclude_columns:
                print(f"Skipping column {column} (excluded from processing)")
                continue
            # Skip columns with all NaNs
            if df[column].isna().all():
                # print(f"Skipping column {column} (all values are NaN)")
                continue
            # Handle categorical columns
            if df[column].dtype == 'object':
                if set(df[column].dropna().unique()) <= {'Yes', 'NO'}:
                    # Process binary variables
                    # print(f"Processing binary column: {column}")
                    df[column] = df[column].replace({'Yes': 1, 'NO': 0})
                else:
                    # Process multi-class categorical variables
                    # print(f"Processing multi-class categorical column: {column}")
                    df[column] = df[column].fillna('Missing')  # Handle NaNs with a placeholder
                    le = LabelEncoder()
                    df[column] = le.fit_transform(df[column])
                    # Store label encoding mapping
                    label_encodings[column] = dict(zip(le.classes_, le.transform(le.classes_)))
        
        # Step 2: Normalize features
        features_to_scale = df.select_dtypes(include=[np.number]).columns.tolist()
        features_to_scale = [col for col in features_to_scale if col not in exclude_columns]
        scaler = StandardScaler()
        df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
        
        # Step 3: Re-fill any remaining NaNs with the column mean (post-encoding and normalization)
        # Only apply to numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].apply(lambda col: col.fillna(col.mean()), axis=0)
        
        # Step 4: Select features for dependency analysis
        features = df.columns.tolist()
        
        # Step 5: Compute the correlation matrix
        df_corr = df[features].select_dtypes(include=[np.number])
        corr_matrix = df_corr.corr()
        
        # Step 6: Plot the heatmap
        plt.figure(figsize=(9, 8))
        if len(features) < 20:
            sns.heatmap(corr_matrix, annot=True, annot_kws={"size":8}, cmap='coolwarm', center=0)
        else: 
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
        plt.title("Correlation Matrix Heatmap")
        plt.xlabel("Features")
        plt.ylabel("Features")
        
        # Save plot
        plot_file_name = f"{uuid.uuid4().hex[:6]}.png"
        file_path = os.path.join(IMAGE_DIR, plot_file_name)
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()

        try:
            host_ip = os.getenv("HOST_IP")
            host_port = int(os.getenv("HOST_PORT"))
        except Exception as e:
            print(f"Error while finding ip and port from env variable when assembling image url.")
                
        image_url = f"http://{host_ip}:{host_port}/static/images/{plot_file_name}"
        
        return json.dumps({
            "image_url": image_url,
            "message": "Correlation analysis complete.",
        })
    except Exception as e:
        print(f"Error occurred: {e}")
        return json.dumps({"error": str(e)})


########################################
######## Importance Analysis ###########
########################################
@tool
def importance_analysis(userID: Annotated[str, InjectedState("userID")],
                        table_name: Annotated[str, "The table containing the user's required production data"],
                        method_name: Annotated[str, "method name ('RF' or 'GB')"],
                        target_name: Annotated[str, "target of interest"],
                        property_col_name: Optional[Annotated[List[str], "A list of column names to include in the importance analysis, e.g., ['column1', 'column2']"]] = None,
                        exclude_columns: Optional[Annotated[List[str], "A list of column names to exclude in the importance analysis, e.g., ['column1', 'column2']"]] = None,
                        plot_type: Optional[Annotated[str, "plot chart type ('P' for pie, 'B' for bar)"]] = None
):
    """
    Importance analysis using Random Forest ("RF") or Gradient Boosting ("GB") with permutation.
    Two plot types are available: pie chart ("P") or bar chart ("B").
    People commonly like to exclude the well names related features like API_UWI, WELL_NAME, Well_ID, you can remind the user to exclude them.
    """
    print("============== Importance Analysis ==============")
    # Load data
    db_name = os.path.join("database", f"{userID}.db")
    df = load_data_to_dataframe(db_name, table_name)
    df.dropna(axis=1, how='all', inplace=True)

    # Initialize exclude_columns if None
    if exclude_columns is None:
        exclude_columns = []

    # Initialize property_col_name if None
    if property_col_name is None:
        property_col_name = []
    elif isinstance(property_col_name, str):
        property_col_name = [property_col_name]
    
    # Handle both property_col_name and exclude_columns provided
    if property_col_name and exclude_columns:
        print("Warning: Both 'property_col_name' and 'exclude_columns' are provided. 'exclude_columns' will be ignored.")
        exclude_columns = []
    
    # Initialize a dictionary to store label encodings for reference
    label_encodings = {}
    
    # Step 1: Process categorical columns
    for column in df.columns:
        if column in exclude_columns:
            print(f"Skipping column {column} (excluded from processing)")
            continue
        if df[column].isna().all():
            print(f"Skipping column {column} (all values are NaN)")
            continue
        # Handle categorical columns
        if df[column].dtype == 'object':
            if set(df[column].dropna().unique()) <= {'Yes', 'NO'}:
                print(f"Processing binary column: {column}")
                df[column] = df[column].replace({'Yes': 1, 'NO': 0})
            else:
                print(f"Processing multi-class categorical column: {column}")
                df[column] = df[column].fillna('Missing')
                le = LabelEncoder()
                df[column] = le.fit_transform(df[column])
                label_encodings[column] = dict(zip(le.classes_, le.transform(le.classes_)))
    
    # Step 2: Normalize features
    features_to_scale = df.select_dtypes(include=[np.number]).columns.tolist()
    features_to_scale = [col for col in features_to_scale if col not in exclude_columns + [target_name]]
    scaler = MinMaxScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    
    # Step 3: Fill remaining NaNs in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].apply(lambda col: col.fillna(col.mean()), axis=0)
    
    # Step 4: Define X and y
    if property_col_name:
        # Ensure property_col_name is a list
        if isinstance(property_col_name, str):
            property_col_name = [property_col_name]
        # Verify that all specified features exist in the dataframe
        missing_features = [col for col in property_col_name if col not in df.columns]
        if missing_features:
            raise ValueError(f"The following features specified in property_col_name do not exist in the data: {missing_features}")
        # Remove the target variable from property_col_name if present
        if target_name in property_col_name:
            print(f"Removing target '{target_name}' from property_col_name to avoid using it as a predictor.")
            property_col_name.remove(target_name)
        # Define X using only the specified features
        X = df[property_col_name]
    else:
        # Use all features except the target and excluded columns
        X = df.drop(columns=[target_name] + exclude_columns)
    y = df[target_name]
    
    # Check for NaNs in X and y
    if X.isnull().values.any():
        raise ValueError("NaN values found in predictors after preprocessing.")
    if y.isnull().values.any():
        raise ValueError("NaN values found in target variable after preprocessing.")
    
    # Step 5: Select model
    if method_name == "RF":
        model = RandomForestRegressor(random_state=42)
    elif method_name == "GB":
        model = GradientBoostingRegressor(random_state=42)
    else:
        raise ValueError("Invalid method_name. Choose 'RF' for Random Forest or 'GB' for Gradient Boosting.")
    
    # Fit model
    model.fit(X, y)
    
    # Calculate permutation importance
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=2)
    importance_scores = result.importances_mean
    feature_names = X.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance_scores})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Step 6: Plot importance
    plt.figure(figsize=(9, 8))
    if plot_type == "P":
        plt.pie(importance_df['Importance'], labels=importance_df['Feature'], autopct='%1.1f%%', startangle=140)
        plt.title("Feature Importance (Pie Chart)")
    else:  # Default to bar chart if plot_type is 'B' or any other value
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title("Feature Importance (Bar Chart)")
        plt.xlabel("Mean Importance Score")
        plt.ylabel("Feature")
    
    # Save plot
    plot_file_name = f"{str(uuid.uuid4())}.png"
    file_path = os.path.join(IMAGE_DIR, plot_file_name)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    
    try:
        host_ip = os.getenv("HOST_IP")
        host_port = int(os.getenv("HOST_PORT"))
    except Exception as e:
        print(f"Error while finding ip and port from env variable when assembling image url.")
        
    image_url = f"http://{host_ip}:{host_port}/static/images/{plot_file_name}"
    
    # Convert feature importance ranking to JSON format for return
    ranked_features = importance_df.to_dict(orient='records')
    
    return json.dumps({
        "image_url": image_url,
        "message": "Importance analysis complete.",
        "rank": ranked_features
    })


########################################
######### Distribution Plot ############
########################################
@tool
def plot_distribution(
    userID: Annotated[str, InjectedState("userID")],
    table_name: Annotated[str, "The name of the table containing the user's required production data"],
    property_col_name: Optional[Annotated[List[str], "A list of columns to plot distributions for, e.g., ['Latitude', 'Longitude']"]] = None,
    normalize: Optional[Annotated[bool, "Set to True to plot both PDF and CDF on the same plot for each column"]] = None,
    value_bin: Optional[Annotated[int, "Specify the number of bins for histograms of numeric data"]] = None
):
    """
    Plot distribution for the specified columns in `property_col_name`. 
    - `property_col_name`: A list of column names to plot distributions for (e.g., ['Latitude', 'Longitude']). 
    - `normalize`: If set to True, plots both PDF and CDF on the same plot for each column.
    - `value_bin`: The number of bins for numeric data histograms.
    
    If you're unsure of the exact column names, use the Summarizer to identify the correct names.
    """
    print("============== Plot Distribution ==============")
    print(property_col_name)
    x_ticks_max = 6
    # Load data
    db_name = os.path.join("database", f"{userID}.db")
    df = load_data_to_dataframe(db_name, table_name)
    df.dropna(axis=1, how='all', inplace=True)
    
    # Attempt to convert columns with numeric values stored as strings to numeric types
    for col in df.columns:
        # Convert column if all non-null values are numeric strings, including negative numbers
        if df[col].dropna().apply(lambda x: str(x).replace('-', '', 1).replace('.', '', 1).isdigit()).all():
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Determine columns to plot
    if property_col_name is None:
        columns_to_plot = df.columns
    elif isinstance(property_col_name, str):
        columns_to_plot = [property_col_name]
    else:
        columns_to_plot = property_col_name
    
    # Set up figure and axes
    num_plots = len(columns_to_plot)
    num_cols = 3  # Number of columns in subplot grid
    num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate rows needed for plots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(9, 3 * num_rows))
   
    axes = axes.flatten()  # Flatten the array if there are multiple plots
    
    # Plot each feature
    for idx, col in enumerate(columns_to_plot):
        data = df[col].dropna()  # Drop NaN values for the current column
        
        # Determine bin settings based on data type
        if pd.api.types.is_numeric_dtype(data):
            bins = value_bin if value_bin is not None else 6
            # Plot histogram for numeric data
            ax1 = axes[idx]
            sns.histplot(data, bins=bins, kde=False, ax=ax1, stat="density", color='skyblue', label='Histogram')
            ax1.set_ylabel("Density")
            ax1.set_xlabel(col)
            
            # Limit the number of x-axis labels to x_ticks_max for numeric data
            ax1.locator_params(axis="x", nbins=x_ticks_max)
            
            # If normalize, plot PDF and CDF on the same plot
            if normalize:
                # Plot PDF using KDE
                sns.kdeplot(data, ax=ax1, color="blue", label="PDF")
                
                # Create a twin axis sharing the x-axis
                ax2 = ax1.twinx()
                
                # Calculate and plot CDF
                sorted_data = np.sort(data)
                cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
                ax2.plot(sorted_data, cdf, color="orange", label="CDF")
                ax2.set_ylabel("Cumulative Probability")
                ax2.set_ylim(0, 1)
        else:
            # Plot histogram for categorical data (object type)
            sns.histplot(data, bins='auto', kde=False, ax=axes[idx], stat="count", color='skyblue')
            axes[idx].set_ylabel("Count")
            axes[idx].set_xlabel(col)
            # Limit the number of x-ticks for categorical data
            unique_values = data.unique()
            if len(unique_values) > x_ticks_max:
                displayed_ticks = np.linspace(0, len(unique_values) - 1, x_ticks_max, dtype=int)
                tick_labels = [str(val)[:3] + "..." if len(str(val)) > 3 else str(val) for val in unique_values[displayed_ticks]]
                axes[idx].set_xticks(displayed_ticks)
                axes[idx].set_xticklabels(tick_labels)
            else:
                tick_labels = [str(val)[:3] + "..." if len(str(val)) > 3 else str(val) for val in unique_values]
                axes[idx].set_xticks(range(len(unique_values)))
                axes[idx].set_xticklabels(tick_labels)
        
    # Hide any extra subplots if there are more axes than plots
    for i in range(num_plots, len(axes)):
        axes[i].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plot_file_name = f"{str(uuid.uuid4())}.png"
    file_path = os.path.join(IMAGE_DIR, plot_file_name)
    plt.savefig(file_path)
    plt.close()
    
    try:
        host_ip = os.getenv("HOST_IP")
        host_port = int(os.getenv("HOST_PORT"))
    except Exception as e:
        print(f"Error while finding ip and port from env variable when assembling image url.")
    
    image_url = f"http://{host_ip}:{host_port}/static/images/{plot_file_name}"
    
    return json.dumps({
        "image_url": image_url,
        "message": "Distribution analysis complete."
    })

