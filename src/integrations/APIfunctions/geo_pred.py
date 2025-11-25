import json
import pandas as pd
from joblib import load
import pickle
import numpy as np

def geo_pred_Montney(agent_inputs):
    """
    Predicts geological features for Central Montney wells based on input coordinates and formation interval.
    example input: {'Longitude': -120.37108, 'Latitude':  55.962475, 'ENVInterval': 'LOWER MONTNEY A', 'StartYear': 2015, 'Month': 1}
    """
    output_feature_keys = [
        'County', 'ElevationGL_FT', 'TotalOrganicCarbon_WTPCT', 'HeightOfHCPV_FT', 
        'HCPV_PCT', 'PhiH_FT', 'WaterSaturation_PCT', 'NonClayVolume_PCT', 'ClayVolume_PCT',
        'EffectivePorosity_PCT', 'DensityPorosity_PCT', 'Resistivity_OHMSM', 'BulkDensity_GPerCC',
        'GammaRay_API', 'Isopach_FT', 'SubseaBaseDepth_FT', 'SubseaTopDepth_FT', 'BottomOfZone_FT',
        'TopOfZone_FT', 'GasGravity_SG'
    ]

    def classify_and_update_feature(feature_name: str, feature_dict: dict, encoding_dict: dict) -> dict:
        """Classifies and updates a feature using the given encoding dictionary."""
        updated_features = feature_dict.copy()
        feature_value = updated_features.get(feature_name, "").upper()

        if feature_name in encoding_dict:
            if feature_value in encoding_dict[feature_name]:
                updated_features[feature_name] = encoding_dict[feature_name][feature_value]
            else:
                raise ValueError(
                    f"Value '{feature_value}' not found in label encodings for '{feature_name}'. "
                )
        else:
            raise ValueError(f"Feature '{feature_name}' not found in label encodings.")

        return updated_features
    
    # Function to convert non-ENVInterval elements to floats if they are strings
    def convert_to_float_if_needed(key, value):
        if key != 'ENVInterval':
            try:
                return float(value) if isinstance(value, str) else value
            except ValueError:
                raise ValueError(f"Unable to convert '{key}' value '{value}' to float.")
        return value  
    
    # Load the model and label encodings
    gbr_model = load('src/integrations/pretrained/multi_gbr_regressor_noscale.joblib')
    with open('src/integrations/pretrained/label_encodings.pkl', 'rb') as file:
        label_encodings = pickle.load(file)

    print(f"label encodings: {label_encodings}")

    # Define required input keys
    input_keys = ['Longitude', 'Latitude', 'ENVInterval', 'StartYear', 'Month']

    # Extract required input features from agent_inputs
    input_features = {key: agent_inputs.get(key) for key in input_keys if agent_inputs.get(key) is not None}
    input_features = {key: convert_to_float_if_needed(key, value) for key, value in input_features.items()}
    
    # Check for missing static inputs
    missing_keys = [key for key in input_keys if key not in input_features]
    
    if missing_keys:
        return json.dumps({
            "error": f"Missing required static inputs: {', '.join(missing_keys)}"
        })

    # Update the categorical features using encoding
    classified_input_features = classify_and_update_feature('ENVInterval', input_features, label_encodings)

    # Prepare DataFrame and drop 'Month'
    inputs_df = pd.DataFrame([classified_input_features]).drop(columns=['Month'])

    # Predict using the model
    gbr_pred = gbr_model.predict(inputs_df)
    gbr_pred_dict = dict(zip(output_feature_keys, gbr_pred.flatten()))
    combined_results = {**classified_input_features, **gbr_pred_dict}

    # Convert all numpy data types to native Python types
    for key in combined_results:
        if isinstance(combined_results[key], np.generic):
            combined_results[key] = combined_results[key].item()

    return combined_results
    
# # results = geo_pred_Montney('{
#     "Longitude": -120.37108,
#     "Latitude": 55.962475,
#     "ENVInterval": "LOWER MONTNEY A",
#     "StartYear": 2015,
#     "Month": 1
# }')
# print(results)