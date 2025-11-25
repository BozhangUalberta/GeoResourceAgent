import json
import os
import uuid
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from matplotlib.ticker import MultipleLocator
from typing import Annotated, Optional, List
from sklearn.preprocessing import StandardScaler
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
import lasio
from src.utils.db_utils import get_data_sqlite3

# Configure matplotlib to use a backend that doesn't require a display environment
matplotlib.use('Agg')

IMAGE_DIR = "src/static/images"

@tool
def auto_zonation(userID: Annotated[str, InjectedState("userID")],
                  row_id: Annotated[int, f"Row ID for LAS file location in 'running' table, it should be a user-uploaded file."], 
                  clustering_method: Annotated[str, "method used to apply zonation"],
                  depth_col_name: Annotated[str, "name of the depth column"],
                  k: Annotated[int, "number of zones"],
                  plot_col_names: Optional[Annotated[List[str], "columns to plot"]] = None
                 ):
    """
    Automatic zonation tool; Using all the logs in the .las file to do the cluster analysis.
    methods include KMeans:"Kmean", Agglomerative Clustering:"Agglomerative", and Gaussian Mixture Model:"Gaussian".
    Specific columns can be selected for plotting to visualize the zonation results.
    The commonly used depth_col_name involves DEPT, DEPTH, DEPTH_FT, when you use summarizer to check, you can keep an eye on the similar column names.
    """
    # Step 1: Read the LAS file as a DataFrame and clean up NaNs
    file_path = get_data_sqlite3(filename=userID, table="running", id=row_id, type="userinput")

    # Read the LAS file using lasio
    las = lasio.read(file_path, engine="normal")
    df = pd.DataFrame(las.data, columns=las.keys())

    df.dropna(axis=1, how='all', inplace=True)

    for column in df.columns:
        # Check if the first value is NaN, and if so, fill it with the first available value in that column
        if pd.isna(df[column].iloc[0]):
            first_valid_index = df[column].first_valid_index()
            if first_valid_index is not None:
                df.loc[0, column] = df[column].iloc[first_valid_index]
        
        # Forward fill the remaining NaN values using the previous valid value
        df[column] = df[column].ffill()

    # Step 2: Standardize features and apply chosen clustering method
    features = df.drop(columns=[depth_col_name])  # Drop Depth, keep only feature columns
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Apply selected clustering method
    if clustering_method == "Kmean":
        model = KMeans(n_clusters=k, random_state=42)
    elif clustering_method == "Agglomerative":
        model = AgglomerativeClustering(n_clusters=k)
    elif clustering_method == "Gaussian":
        model = GaussianMixture(n_components=k, random_state=42)
    else:
        return json.dumps({"error": "Invalid clustering method. Choose 'kmean', 'Agglomerative', or 'Gaussian'."})

    df['Zone'] = model.fit_predict(scaled_features)

    # Step 3: Plot logs and zonation
    if not plot_col_names:
        plot_col_names = features.columns[:4].tolist()

    if plot_col_names:
        n_cols = len(plot_col_names)
        fig, axs = plt.subplots(1, n_cols + 1, figsize=(n_cols * 2+2, 10), gridspec_kw={'width_ratios': [1] * n_cols + [0.5]})
        y_major_locator = MultipleLocator(10)
        y_value = df[depth_col_name]

        # Define depth range
        depth_min, depth_max = y_value.min(), y_value.max()

        # Plot each specified log column
        for i, col_name in enumerate(plot_col_names):
            unit = las.get_curve(col_name).unit if col_name in las.keys() else ''
            axs[i].plot(df[col_name], y_value, 'k', alpha=0.8)
            axs[i].invert_yaxis()
            axs[i].set_title(col_name)
            axs[i].yaxis.set_major_locator(y_major_locator)
            axs[i].grid(True)
            axs[i].set_xlabel(unit)
            axs[i].set_ylim(depth_max, depth_min) 

        # Plot zonation
        num_zones = df['Zone'].nunique()
        cmap = plt.get_cmap('rainbow', num_zones)
        zone_data = df['Zone'].values.reshape(-1, 1)

        # Plot the zones using imshow for a clear block display
        im = axs[-1].imshow(zone_data, aspect='auto', cmap=cmap, extent=[0, 1, depth_min, depth_max])
        axs[-1].invert_yaxis()
        axs[-1].set_title('Zone')
        axs[-1].set_xticks([])
        axs[-1].yaxis.set_major_locator(y_major_locator)

        # Optional: Add a color bar to indicate zone labels
        cbar = fig.colorbar(im, ax=axs[-1], orientation='vertical', label='Zone')
        cbar.set_ticks(range(num_zones))
        cbar.set_ticklabels([f'Zone {i}' for i in range(num_zones)])

    else:
        # If no columns specified, skip plotting logs
        fig, axs = plt.subplots(1, 1, figsize=(2, 10))
        axs.set_visible(False)

    # Save plot
    file_name = f"{str(uuid.uuid4())}.png"
    file_path = os.path.join(IMAGE_DIR, file_name)
    plt.tight_layout()
    plt.savefig(file_path)

    try:
        host_ip = os.getenv("HOST_IP")
        host_port = int(os.getenv("HOST_PORT"))
    except Exception as e:
        print(f"Error while finding ip and port from env variable when assembling image url.")
        
    image_url = f"http://{host_ip}:{host_port}/static/images/{file_name}"

    return json.dumps({
        "image_url": image_url,
        "message": "Autozonation complete."
    })