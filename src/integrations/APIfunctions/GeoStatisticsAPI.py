import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
from scipy.interpolate import Rbf
from scipy.spatial import cKDTree
from scipy.spatial import Delaunay

class GeoStats_Interpolate:
    def __init__(self):
        self.models = {}

    #%%
    def ordinary_kriging_interpolation(self, data: pd.DataFrame, value_column: str, locations: pd.DataFrame, variogram_model='linear',variogram_parameters = None):

        if not all(col in data.columns for col in ['longitude', 'latitude', value_column]):
            raise ValueError("Data must contain 'longitude', 'latitude', and the specified value column.")
        if not all(col in locations.columns for col in ['longitude', 'latitude']):
            raise ValueError("Locations data must contain 'longitude' and 'latitude' columns.")

        lon = data['longitude'].values
        lat = data['latitude'].values
        values = data[value_column].values

        if variogram_parameters == None:
            if variogram_model == 'gaussian':
                variogram_parameters = {
                    'sill': 1.0,   # Total variance of the field
                    'range': 0.5,  # Effective range (distance at which correlation drops significantly)
                    'nugget': 0.1  # Variance at zero distance (measurement error or microscale variance)
                }

        # Create Ordinary Kriging model
        kriging_model = OrdinaryKriging(
            lon, lat, values,
            variogram_model=variogram_model,
            verbose=False,
            enable_plotting=False,
            variogram_parameters = variogram_parameters
        )

        # Interpolate at the specified locations
        lon_loc = locations['longitude'].values
        lat_loc = locations['latitude'].values
        z, ss = kriging_model.execute("grid", lon_loc, lat_loc)

        # Create a DataFrame with the results
        result = locations.copy()
        result['predicted_value'] = z

        return result
    
    #%%
    def universal_kriging_interpolation(self, data: pd.DataFrame, value_column: str, locations: pd.DataFrame, variogram_model='linear', variogram_parameters=None, drift_terms=None):

        # Check if the required columns are present in data and locations
        if not all(col in data.columns for col in ['longitude', 'latitude', value_column]):
            raise ValueError("Data must contain 'longitude', 'latitude', and the specified value column.")
        if not all(col in locations.columns for col in ['longitude', 'latitude']):
            raise ValueError("Locations data must contain 'longitude' and 'latitude' columns.")

        lon = data['longitude'].values
        lat = data['latitude'].values
        values = data[value_column].values

        # If variogram_parameters is not provided, set default values based on the variogram_model
        if variogram_parameters is None:
            if variogram_model == 'gaussian':
                variogram_parameters = {
                    'sill': 1.0,
                    'range': 0.5,
                    'nugget': 0.1
                }
            # Add more conditions for other variogram models as needed

        # Create Universal Kriging model
        kriging_model = UniversalKriging(
            lon, lat, values,
            variogram_model=variogram_model,
            variogram_parameters=variogram_parameters,
            drift_terms=drift_terms,
            verbose=False,
            enable_plotting=False
        )

        # Interpolate at the specified locations
        lon_loc = locations['longitude'].values
        lat_loc = locations['latitude'].values
        z, ss = kriging_model.execute('points', lon_loc, lat_loc)

        # Create a DataFrame with the results
        result = locations.copy()
        result['predicted_value'] = z

        return result

    #%%
    def spline_interpolation(self, data: pd.DataFrame, value_column: str, locations: pd.DataFrame, function=None):

        if not all(col in data.columns for col in ['longitude', 'latitude', value_column]):
            raise ValueError("Data must contain 'longitude', 'latitude', and the specified value column.")
        if not all(col in locations.columns for col in ['longitude', 'latitude']):
            raise ValueError("Locations data must contain 'longitude' and 'latitude' columns.")

        lon = data['longitude'].values
        lat = data['latitude'].values
        values = data[value_column].values

        # Create Spline (RBF) model
        if function == None:
            function = 'thin_plate'
        spline_model = Rbf(lon, lat, values, function=function)

        # Interpolate at the specified locations
        lon_loc = locations['longitude'].values
        lat_loc = locations['latitude'].values
        predictions = spline_model(lon_loc, lat_loc)

        # Create a DataFrame with the results
        result = locations.copy()
        result['predicted_value'] = predictions

        return result
    #%%
    def natural_neighbor_interpolation(self, data: pd.DataFrame, value_column: str, locations: pd.DataFrame):
        if not all(col in data.columns for col in ['longitude', 'latitude', value_column]):
            raise ValueError("Data must contain 'longitude', 'latitude', and the specified value column.")
        if not all(col in locations.columns for col in ['longitude', 'latitude']):
            raise ValueError("Locations data must contain 'longitude' and 'latitude' columns.")

        lon = data['longitude'].values
        lat = data['latitude'].values
        values = data[value_column].values

        points = np.column_stack((lon, lat))
        triangulation = Delaunay(points)

        interpolated_values = []
        for point in np.column_stack((locations['longitude'].values, locations['latitude'].values)):
            simplex = triangulation.find_simplex(point)
            if simplex == -1:
                interpolated_values.append(np.nan)
            else:
                vertices = triangulation.simplices[simplex]
                barycentric_coords = triangulation.transform[simplex, :-1].dot(point - triangulation.transform[simplex, -1])
                weights = np.hstack((barycentric_coords, 1 - barycentric_coords.sum()))
                interpolated_values.append(np.dot(weights, values[vertices]))

        result = locations.copy()
        result['predicted_value'] = interpolated_values
        return result

    #%%
    def rbf_interpolation(self, data: pd.DataFrame, value_column: str, locations: pd.DataFrame, function=None):
        if not all(col in data.columns for col in ['longitude', 'latitude', value_column]):
            raise ValueError("Data must contain 'longitude', 'latitude', and the specified value column.")
        if not all(col in locations.columns for col in ['longitude', 'latitude']):
            raise ValueError("Locations data must contain 'longitude' and 'latitude' columns.")

        lon = data['longitude'].values
        lat = data['latitude'].values
        values = data[value_column].values

        # Create RBF model
        if function == None:
            function = 'multiquadric'
        rbf_model = Rbf(lon, lat, values, function=function)

        # Interpolate at the specified locations
        lon_loc = locations['longitude'].values
        lat_loc = locations['latitude'].values
        predictions = rbf_model(lon_loc, lat_loc)

        # Create a DataFrame with the results
        result = locations.copy()
        result['predicted_value'] = predictions

        return result
    #%%
    def trend_surface_interpolation(self, data: pd.DataFrame, value_column: str, locations: pd.DataFrame, degree: int):
        if not all(col in data.columns for col in ['longitude', 'latitude', value_column]):
            raise ValueError("Data must contain 'longitude', 'latitude', and the specified value column.")
        if not all(col in locations.columns for col in ['longitude', 'latitude']):
            raise ValueError("Locations data must contain 'longitude' and 'latitude' columns.")

        lon = data['longitude'].values
        lat = data['latitude'].values
        values = data[value_column].values

        A = np.vstack([lon**i * lat**j for i in range(degree + 1) for j in range(degree + 1 - i)]).T
        coeffs, _, _, _ = np.linalg.lstsq(A, values, rcond=None)

        lon_loc = locations['longitude'].values
        lat_loc = locations['latitude'].values
        A_pred = np.vstack([lon_loc**i * lat_loc**j for i in range(degree + 1) for j in range(degree + 1 - i)]).T
        predictions = A_pred.dot(coeffs)

        result = locations.copy()
        result['predicted_value'] = predictions
        return result
    #%%
    def moving_average_interpolation(self, data: pd.DataFrame, value_column: str, locations: pd.DataFrame, radius: float):
        if not all(col in data.columns for col in ['longitude', 'latitude', value_column]):
            raise ValueError("Data must contain 'longitude', 'latitude', and the specified value column.")
        if not all(col in locations.columns for col in ['longitude', 'latitude']):
            raise ValueError("Locations data must contain 'longitude' and 'latitude' columns.")

        interpolated_values = []
        for point in np.column_stack([locations['longitude'].values, locations['latitude'].values]):
            distances = np.sqrt(np.sum((data[['longitude', 'latitude']].values - point) ** 2, axis=1))
            within_radius = distances <= radius
            if np.any(within_radius):
                interpolated_values.append(data[value_column].values[within_radius].mean())
            else:
                interpolated_values.append(np.nan)

        result = locations.copy()
        result['predicted_value'] = interpolated_values
        return result
#%%
    def nearest_neighbor_interpolation(self, data: pd.DataFrame, value_column: str, locations: pd.DataFrame):
        if not all(col in data.columns for col in ['longitude', 'latitude', value_column]):
            raise ValueError("Data must contain 'longitude', 'latitude', and the specified value column.")
        if not all(col in locations.columns for col in ['longitude', 'latitude']):
            raise ValueError("Locations data must contain 'longitude' and 'latitude' columns.")

        lon = data['longitude'].values
        lat = data['latitude'].values
        values = data[value_column].values

        kdtree = cKDTree(np.column_stack([lon, lat]))

        distances, indices = kdtree.query(np.column_stack([locations['longitude'].values, locations['latitude'].values]), k=1)
        predictions = values[indices]

        result = locations.copy()
        result['predicted_value'] = predictions
        return result
