import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from joblib import load
import pickle
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta

class CentralMontneyRatePred:
    def __init__(self, static_inputs, prod_hist=None, shut_list=None):
        """
        Parameters:
        static_inputs (dict): A dictionary containing the static inputs provided by the user.
        prod_hist (array): A 2D numpy array representing historical production data with shape (n_timesteps, 3).
        shut_list (list): A list of indices representing shut-in periods (optional).
        """
        # Define the keys for peak input
        self.peak_input_keys = ['Longitude', 'Latitude', 'ENVInterval', 'StartYear', 'County', 'ElevationGL_FT',
            'TotalOrganicCarbon_WTPCT', 'HeightOfHCPV_FT', 'HCPV_PCT', 'PhiH_FT',
            'WaterSaturation_PCT', 'NonClayVolume_PCT', 'ClayVolume_PCT',
            'EffectivePorosity_PCT', 'DensityPorosity_PCT', 'Resistivity_OHMSM',
            'BulkDensity_GPerCC', 'GammaRay_API', 'Isopach_FT', 'SubseaBaseDepth_FT',
            'SubseaTopDepth_FT', 'BottomOfZone_FT', 'TopOfZone_FT', 'GasGravity_SG',
            'ENVOperator', 'WellPadDirection', 'AverageStageSpacing_FT', 'FracStages',
            'CompletionTime_DAYS', 'LateralLength_FT', 'MD_FT', 'TVD_FT']
        
        self.gbr_model_p = self._load_gbr_model()
        self.AI_model = self.load_model()
        self.static_inputs = static_inputs  # User-provided static inputs
        self.prod_hist = self._pad_prod_hist(prod_hist)  # Pad or initialize prod_hist to (60, 3)
        self.shut_ins = self._convert_shut_list(shut_list)  # Convert or initialize shut_list to (60, 1)
        self.hist_len = len(prod_hist) - 1 if prod_hist is not None else 0  # Determine hist_len
        self.dynamic_df = None  # Dynamic inputs generated from static inputs
        self.scalers = self._load_scalers()

        # Predict PeakTime and PeakRate if they are not provided
        if 'PeakTime' not in self.static_inputs or 'PeakRate' not in self.static_inputs:
            self.predict_peak()

    def _pad_prod_hist(self, prod_hist):
        if prod_hist is None:
            return np.zeros((60, 3))  # Default to zeros if no production history is provided

        padded_hist = np.zeros((60, 3))  # Initialize a (60, 3) array with zeros
        length = min(len(prod_hist), 60)  # Determine how much of prod_hist can be used
        padded_hist[:length, :] = prod_hist[:length, :]  # Fill with available production history
        return padded_hist

    def _convert_shut_list(self, shut_list):
        shut_ins = np.zeros((60, 1))  # Initialize a (60, 1) array with zeros
        if shut_list is not None:
            for index in shut_list:
                if 0 <= index < 60:  # Ensure the index is within bounds
                    shut_ins[index, 0] = 1
        return shut_ins

    def _load_gbr_model(self):
        try:
            return load('multi_gbr_regressor_peak.joblib')
        except Exception as e:
            raise FileNotFoundError("GBR model could not be loaded. Ensure the file path is correct.") from e

    def load_model(self):
        mask_special_val = -1e9
        threshold_active = 0

        def thresholded_tanh(x, threshold=threshold_active):
            thresholds = tf.constant(threshold, dtype=x.dtype)
            thresholds = tf.reshape(thresholds, (1, 1, -1))
            tanh_x = tf.tanh(x)
            adjusted_tanh = (tanh_x + 1) / 2 * (1 - thresholds) + thresholds  # Scales and shifts the output
            return adjusted_tanh

        def masked_loss_function(y_true, y_pred):
            mask = K.cast(K.not_equal(y_true, mask_special_val), K.floatx())
            y_true_masked = y_true * mask
            y_pred_masked = y_pred * mask
            loss = K.mean(K.square(y_pred_masked - y_true_masked), axis=-1)
            return loss

        try:
            model = keras.models.load_model(
                'MED_DUDS_denoise_ThTanh0430.keras',
                custom_objects={
                    'masked_loss_function': masked_loss_function,
                    'thresholded_tanh': thresholded_tanh
                }
            )
            return model
        except Exception as e:
            raise FileNotFoundError("AI model could not be loaded. Ensure the file path is correct.") from e

    def _load_scalers(self):
        try:
            scalers = {}
            scaler_paths = ['scaler_encoder.pkl', 'scaler_decoder_D.pkl', 'scaler_decoder_S.pkl', 'scaler_y_pred.pkl']
            scaler_names = ['scaler_encoder', 'scaler_decoder_D', 'scaler_decoder_S', 'scaler_y_pred']

            for name, path in zip(scaler_names, scaler_paths):
                with open(Path('ml_model_scalers') / path, 'rb') as file:
                    scalers[name] = pickle.load(file)
            return scalers

        except Exception as e:
            raise FileNotFoundError("One or more scaler files could not be loaded. Ensure the file paths are correct.") from e

    def predict_peak(self):
        peak_inputs = {key: self.static_inputs[key] for key in self.peak_input_keys}

        p_inputs = pd.DataFrame([peak_inputs])
        peaks = self.gbr_model_p.predict(p_inputs)
        self.static_inputs['PeakTime'] = peaks[0][0]
        self.static_inputs['PeakRate'] = peaks[0][1]

    def create_dynamic_df(self):
        start_year = int(self.static_inputs['StartYear'])
        start_month = int(self.static_inputs.get('Month', 1))  # Use January (1) as default if 'Month' is missing
        start_date = datetime(start_year, start_month, 1)
        date_list = [start_date + relativedelta(months=+i) for i in range(60)]

        self.dynamic_df = pd.DataFrame({
            'TotalProdMonths': list(range(1, 61)),
            'Year': [date.year for date in date_list],
            'Month': [date.month for date in date_list],
            'ShutIns': self.shut_ins.flatten()
        })

    def predict_profile(self):
        if not self.static_inputs or self.dynamic_df is None:
            raise ValueError("Static inputs must be set, and dynamic inputs must be generated before prediction.")

        # Prepare dynamic and static inputs
        dynamic_inputs = self.dynamic_df[['TotalProdMonths', 'Year', 'Month', 'ShutIns']].values

        # Encoder input preparation
        encoder_inputs = np.concatenate((dynamic_inputs, self.prod_hist), axis=1)
        encoder_inputs_scaled = self.scalers['scaler_encoder'].transform(encoder_inputs.reshape(-1, 7)).reshape(1, 60, 7)
        encoder_inputs_scaled[:, self.hist_len + 1:, :] = -1e9  # Apply mask beyond the history length

        # Decoder D inputs preparation
        decoder_D_inputs_scaled = self.scalers['scaler_decoder_D'].transform(dynamic_inputs.reshape(-1, 4)).reshape(1, 60, 4)
        decoder_D_inputs_scaled[:, :self.hist_len + 1, :] = -1e9  # Apply mask up to history length

        # Decoder S inputs preparation
        NN_static_inputs = {key: self.static_inputs[key] for key in self.peak_input_keys + ['PeakTime', 'PeakRate']}
        decoder_S_inputs = np.array(list(NN_static_inputs.values()), dtype='float32').reshape(1, -1)
        decoder_S_inputs_scaled = self.scalers['scaler_decoder_S'].transform(decoder_S_inputs)

        # Predict the profile
        y_pred = self.AI_model.predict([encoder_inputs_scaled, decoder_D_inputs_scaled, decoder_S_inputs_scaled])
        y_pred[:, 0, :] = 0  # Mask the first timestep as zero
        y_pred_restored = self.scalers['scaler_y_pred'].inverse_transform(y_pred.reshape(-1, 1)).reshape(1, 60, 1)

        return y_pred_restored
