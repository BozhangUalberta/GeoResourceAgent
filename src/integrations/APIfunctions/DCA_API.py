import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

class ProcessShutSeqs:
    def __init__(self, shut_threshold = 10, screen_ratio = 0.1, shut_signal=None, peak_t=None):
        self.shut_threshold = shut_threshold
        self.screen_ratio = screen_ratio
        self.shut_signal = shut_signal
        self.peak_t = peak_t
        
    def find_shutins(self, seq):
        # Make a copy of the sequence
        seq_copy = np.copy(seq)

        for i in range(1, len(seq) - 2):
            if seq_copy[i] != 0 and seq_copy[i + 1] != 0:
                q_ratio = np.abs(seq_copy[i + 1] / seq_copy[i])
                if q_ratio <= self.screen_ratio:
                    seq_copy[i + 1] = 0

            if seq_copy[i] != 0 and seq_copy[i + 1] != 0 and seq_copy[i + 2] != 0:
                q_ratio2 = np.abs(seq_copy[i + 2] / seq_copy[i])
                if q_ratio2 <= self.screen_ratio / 2:
                    seq_copy[i + 2] = 0

            if seq_copy[i] < self.shut_threshold:
                seq_copy[i] = 0

        shut_signal = np.copy(seq_copy)
        shut_signal[shut_signal != 0] = 1
        shutins_rev = 1 - shut_signal
        return shutins_rev

        
    def find_peak_t(self, seq):
        peak_t = np.argmax(seq)+1
        return peak_t

    def quality_check(self, seq):
        fixed_seq = np.copy(seq)
        if len(seq) > 1:
            if seq[1] > seq[0]:
                fixed_seq[0] = seq[1]
        elif len(seq) > 2:
            if seq[1] > seq[0] and seq[2] > seq[1]:
                fixed_seq[0] = seq[2]
        elif len(seq) > 3:
            if seq[1] > seq[0] and seq[2] > seq[1] and seq[3] > seq[2]:
                fixed_seq[0] = seq[3]
        return fixed_seq
    
    @staticmethod
    def simple_arps(original_curve, t_extend=None):
        t = np.arange(len(original_curve))  # Assuming time steps are uniform and start from 0
        qi = original_curve[0]  # Initial flow rate

        # Initial guess for Di and b, assuming some typical values for these parameters
        Di_initial_guess = 0.02  # This is a placeholder and should be estimated based on data or given as input
        b_initial_guess = 0.5  # Typical value for hyperbolic decline

        # Define the Arps decline function
        def arps_decline(t, Di, b):
            if b != 0:
                return qi / (1 + b * Di * t) ** (1/b)
            else:
                # For b=0, the formula simplifies to an exponential decline
                return qi * np.exp(-Di * t)

        # Objective function to minimize (mean squared error between the model and actual data)
        def objective(params):
            Di, b = params
            q_predicted = arps_decline(t, Di, b)
            return mean_squared_error(original_curve, q_predicted)

        # Initial parameter guess
        initial_params = [Di_initial_guess, b_initial_guess]

        # Optimization
        result = minimize(objective, initial_params, bounds=[(0.01, 1), (0, 2)])
        Di_opt, b_opt = result.x

        # Predict q using optimized parameters
        if t_extend == None:
            q = arps_decline(t, Di_opt, b_opt)
        else:
            q = arps_decline(np.arange(len(original_curve)+t_extend), Di_opt, b_opt)

        return q, Di_opt, b_opt

    def complex_arps(self, org_seq, t_extend=None):
        segments = []
        start = None
        self.shut_signal = self.find_shutins(org_seq) if self.shut_signal is None else self.shut_signal
        self.peak_t = self.find_peak_t(org_seq) if self.peak_t is None else self.peak_t

        # Identify segments where production occurs (shut_signal == 0)
        for i, signal in enumerate(self.shut_signal):
            if signal == 0 and start is None:
                start = i
            elif signal == 1 and start is not None:
                if i - start >= 1:  # Ensure segment has at least n steps
                    segments.append((start, i))
                start = None
        # Catch any segment that goes to the end of the array
        if start is not None and len(self.shut_signal) - start >= 1:
            segments.append((start, len(self.shut_signal)))

        # Initialize the processed sequence with zeros
        if t_extend is not None:
            extended_length = len(org_seq) + t_extend
        else:
            extended_length = len(org_seq)
            
        processed_seq = np.zeros(extended_length)
        Di_values = np.zeros(extended_length)
        b_values = np.zeros(extended_length)

        # Apply decline analysis to each production segment and update the processed sequence
        n = 0
        for start, end in segments:
            if start <= self.peak_t <= end:
                # If the peak time is within this segment, copy production up to the peak time
                processed_seq[start:self.peak_t] = org_seq[start:self.peak_t]
                
                # Apply decline analysis after the peak time
                prod_segment2 = org_seq[self.peak_t - 1:end]

                if len(segments) == 1:
                    # If there is only one segment, apply decline curve with extension if needed
                    segment_fitted, Di_value, b_value = self.simple_arps(prod_segment2,t_extend)
                    processed_seq[self.peak_t-1:end+t_extend] = segment_fitted
                    Di_values[self.peak_t-1:end+t_extend] = Di_value
                    b_values[self.peak_t-1:end+t_extend] = b_value
                else:
                    # Apply decline curve without extension for multiple segments
                    segment_fitted, Di_value, b_value = self.simple_arps(prod_segment2)
                    processed_seq[self.peak_t - 1:end] = segment_fitted
                    Di_values[self.peak_t - 1:end] = Di_value
                    b_values[self.peak_t - 1:end] = b_value

            else:
                # If no peak within this segment, analyze the entire segment
                prod_segment = org_seq[start:end]
                fixed_rate = self.quality_check(prod_segment)
                
                if n == len(segments)-1 and t_extend is not None: 
                # for the last element, special treat to extend if applies
                    segment_fitted, Di_value, b_value = self.simple_arps(fixed_rate,t_extend)
                    processed_seq[start:end+t_extend] = segment_fitted
                    Di_values[start:end+t_extend] = Di_value
                    b_values[start:end+t_extend] = b_value
                    
                else: 
                    # Standard decline curve fitting without extension
                    segment_fitted, Di_value, b_value = self.simple_arps(fixed_rate)
                    processed_seq[start:end] = segment_fitted
                    Di_values[start:end] = Di_value
                    b_values[start:end] = b_value

                if len(segments) > 1 and n < len(segments) - 1:
                    start_next, _ = segments[n + 1]

                    if end < len(processed_seq):
                        processed_seq[end:start_next] = org_seq[end:start_next]
            n += 1

        return processed_seq, Di_values, b_values