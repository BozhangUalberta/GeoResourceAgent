import numpy as np
from itertools import groupby
from scipy.optimize import fsolve, curve_fit

def detect_peak_smooth_z(prod_data:list,
                         lag:int,
                         threshold_local:float,
                         threshold_global:float,
                        influence:float) ->list:
    """
    Ref:
    https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data/54507329#54507329
    
    The algorithm takes 3 inputs:
    lag = the lag of the moving window,
    threshold = the z-score at which the algorithm signals and
    influence = the influence (between 0 and 1) of new signals on the mean and
    standard deviation.
    
    For example, a lag of 5 will use the last 5 observations to smooth the data.
    A threshold of 3.5 will signal if a data point is 3.5 standard deviations
    away from the moving mean. And an influence of 0.5 gives signals half of the
    influence that normal data points have. Likewise, an influence of 0 ignores
    signals completely for recalculating the new threshold. An influence of 0 is
    therefore the most robust option (but assumes stationarity);
    putting the influence option at 1 is least robust.
    
    For non-stationary data, the influence option should therefore be put
    somewhere between 0 and 1.
    """
    y = prod_data.copy()
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    
    for i in range(lag, len(y)):
        stdFilter_global = np.std(y[:i])
        if abs(y[i] - avgFilter[i-1]) > threshold_local * stdFilter[i-1] and \
           abs(y[i] - avgFilter[i-1]) > threshold_global * stdFilter_global:
            if y[i] > avgFilter[i-1]:
                signals[i] = 1
            else:
                signals[i] = 0  # use -1 if hope to check negative change
            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])
    
    # Post Processing to remove some single points
    for i in range(signals.size-3):
        if signals[i] == 1 and signals[i+1] == 0 and signals[i+2] == 1:
            signals[i+1] = 1
        elif signals[i] == 1 and np.sum(signals[i+1:i+3]) == 0 and signals[i+3] == 1:
            signals[i+1] = 1
            signals[i+2] = 1
    
    t_idx_count = [sum(1 for _ in group) for _, group in groupby(signals)]
    if len(t_idx_count) > 1:
        t_idx_sum = np.cumsum(t_idx_count)
        for i, idx in enumerate(t_idx_count):
            if signals[t_idx_sum[i]-1] == 0:
                continue
            if i == len(t_idx_count)-1:
                # if signal at the end of curve, ignore
                t_start = t_idx_sum[i-1]
                t_end = t_idx_sum[i]
                signals[t_start:t_end] = 0
            elif sum(t_idx_count[i:]) < 15:
                # if there are less than threshold data after last signal, ignore.
                t_start = t_idx_sum[i-1]
                t_end = t_idx_sum[i]
                signals[t_start:t_end] = 0
            elif idx == 1:
                t_idx = t_idx_sum[i-1]
                if (y[t_idx+1] < np.mean(y[t_idx-2:t_idx]) or \
                    y[t_idx+2] < np.mean(y[t_idx-2:t_idx])):
                    signals[t_idx] = 0
    
    return signals



def _calculate_boundary(prod_data:list,
                        var_upper:float=0.85,
                        var_lower:float=0.70):
    """This function prepares the dataset and calcualtes parameters"""

    # Take the max 3 values of the curve.
    # Calucualte the mean of these 3 max values.
    t_prod = [p for p in prod_data]
    t_prod.sort(reverse=True)
    t_max_avg = np.mean(t_prod[:3])
    # Calculate the upper and lower boundary for identifying peak.
    # If a data value is above the upper_boundary, it could be a peak.
    upper_boundary = t_max_avg*var_upper
    # If a data value drops below lower_boundary,
    #   it means an up-down trend is created,
    #   In other word, a peak is created.
    lower_boundary = t_max_avg*var_lower
    return upper_boundary, lower_boundary


def _detect_peaks(prod_data, upper_boundary, lower_boundary,
                  var_threshold_dec_per_month=0.03, var_decline_threshold=0.90):
    """This function detect all possible peaks in the production history"""

    # Create a copy of the original curve
    y = prod_data.copy()
    # Add a 0 to the end of original curve, this is just to help identify peak,
    # because sometimes the peak rate is the last point of curve.
    # This 0 will be removed at end of this function.
    y = np.append(y, 0)

    # Initialize the temp variables
    peak_months = []  # The month of peaks identified
    peak_rates = []  # The rate of peak months
    t_max_rate = 0  # Initial peak value is set to be 0
    t_max_idx = 0  # Initial peak month is set to be 0
    t_threshold = var_decline_threshold
    t_dummy = True    # dummy variable used to control the while loop
    var_pre_1_threshold = 1 + var_threshold_dec_per_month * 1
    
    while t_dummy:
        for idx, t_prod in enumerate(y):
            # Sequentially check each data point, the point is identified as
            # peak if the point is:
            # 1. above the upper boundary,
            # 2. is higher than 93% (t_threshold) of the last identified peak.

            if t_prod > t_max_rate * t_threshold and \
               t_prod > upper_boundary:
                t_max_idx = idx
                t_max_rate = t_prod

            # If the data value drops below the lower boundary
            # Output the peak identified in the previous section
            if t_prod < lower_boundary:
                if t_max_rate > upper_boundary:
                    # If the peak month is later than the 2nd month,
                    # and the previous month's rate is larger than a threshold,
                    # and the previous month rate is the highest among recent
                    # 3 months:
                    # Set the peak month to previous month.
                    if t_max_idx > 1 and y[t_max_idx-1] == np.max(
                            y[t_max_idx-2:t_max_idx+1]) and \
                            y[t_max_idx-1] > \
                            (y[t_max_idx] * var_pre_1_threshold):
                        t_max_idx = t_max_idx-1
                        t_max_rate = y[t_max_idx]
                    # If peak month is the 2nd month, and the first month's
                    # rate is larger than a threshold:
                    # Set the peak month to previous month.
                    elif t_max_idx == 1 and y[t_max_idx-1] > \
                         y[t_max_idx] * var_pre_1_threshold:
                        t_max_idx = t_max_idx-1
                        t_max_rate = y[t_max_idx]

                    # Output the identified peak
                    peak_months.append(t_max_idx)
                    peak_rates.append(t_max_rate)

                # Reset the peak month and peak value
                t_max_rate = 0
                t_max_idx = 0

        # If not peak identified with the current t_threshold,
        # increase by 0.01 and rerun the loop
        if not peak_months:
            t_threshold += 0.01
        # If any peak identified, end the loop
        else:
            t_dummy = False

    return peak_months, peak_rates

def _post_processing(prod_data, peak_months, peak_rates,
                     var_threshold_dec_per_month=0.03):
    
    var_pre_3_threshold = 1 + var_threshold_dec_per_month * 3
    """This function helps to optimize some cases, not all curves affected"""
    # If there are multiple peaks identified, If the last two peaks are
    # within 6 months, and the penultimate is higher than a threshold:
    # Set the peak month to the penultimate one
    if len(peak_months) > 1:
        if peak_months[-1] - peak_months[-2] <= 6 and \
           peak_rates[-2] > peak_rates[-1] * var_pre_3_threshold:
            peak_months = peak_months[:-1]

    # Check the local maximum, compare the current peak with the previous 6 
    # months data, if previous rate is higher and difference > a threshold, 
    # reset the peak month to previous one.
    check_pre_month = max(0, peak_months[-1]-6)
    check_pre_data = prod_data[check_pre_month:peak_months[-1]+1]
    check_pre_data_max = np.max(check_pre_data)
    if check_pre_data_max != prod_data[peak_months[-1]]:
        delta_month = peak_months[-1]-list(prod_data).index(check_pre_data_max)
        var_pre_delta_threshold = 1 + var_threshold_dec_per_month * delta_month
        if check_pre_data_max > prod_data[peak_months[-1]] * var_pre_delta_threshold:
            peak_months.append(list(prod_data).index(check_pre_data_max))

    # Output the last peak identified in the curve as the peak of entire curve.
    max_idx = peak_months[-1]
    return max_idx

def detect_peak_plateau(prod_data:list):
    """
    This function identifies one or more peaks of the curve, and output the last
    peak identified as the peak of entire curve.

    This peak(s) may not be the absolute maximum of the curve, but it should mark
    the starting of a normal decline trend.

    This function helps to deal with plateau/slow decline curves.
    """

    """
    Parameters used in this function

    var_upper: parameter used to define upper boundary for identifying peak.
    var_lower: parameter used to define lower boundary for identifying peak.
    var_decline_threshold:
                1. The threshold used to screen platue/slow decline. If there are
                two points and they are both above the upper_threshold, and the
                second one is > 93% (t_threshold) of the previous one, it is
                consider as slow decline/plateau, and the second is now the
                new peak.
                2. Usually range between 0.85~0.95, too high will miss shallow
                decline (plateau), too low will miss the real peak.

    var_pre_1_threshold: threshold to compare the peak rate and peak-1 month's rate
    var_pre_2_threshold: threshold to compare the peak rate and peak-2 month's rate
    var_pre_3_threshold: threshold to compare the peak rate and peak-3 month's rate
    """

    var_upper = 0.85
    var_lower = 0.70
    var_decline_threshold = 0.90
    var_threshold_dec_per_month = 0.03

    # Calculate the boundary of checking window
    upper_boundary, lower_boundary = _calculate_boundary(prod_data, var_upper,
                                                       var_lower)
    # Detect all potential peaks
    peak_months, peak_rates = _detect_peaks(prod_data, upper_boundary,
                                          lower_boundary,
                                          var_threshold_dec_per_month,
                                          var_decline_threshold)
    # Get the final selection of peak month
    max_idx = _post_processing(prod_data, peak_months, peak_rates,
                             var_threshold_dec_per_month)
    return max_idx



def get_peak_index(prod_data, LAG=5, THRESHOLD_LOCAL=3,THRESHOLD_GLOBAL=1, INFLUENCE=0.15):
    """
    This function detect the correct peak location to partition the curve by
    using three different signals
    """
    prod_data = np.array(prod_data)/np.max(prod_data)
    prod_data = np.round(prod_data, decimals=2)
    
    # Peak index 1
    peak_index_1 = detect_peak_plateau(prod_data)
    
    # Peak index 2
    peak_signal_2 = detect_peak_smooth_z(prod_data, LAG, THRESHOLD_LOCAL,
                                       THRESHOLD_GLOBAL, INFLUENCE)
    
    # Get the potential peak intervals from peak index 2
    p_sig_count = [sum(1 for _ in group) for _, group in groupby(peak_signal_2)]
    p_sig_interval = []
    
    if len(p_sig_count) > 1:
        p_sig_cs = np.cumsum(p_sig_count)
        for i, j in enumerate(p_sig_cs):
            if peak_signal_2[j-1] == 0:
                continue
            t_start = p_sig_cs[i-1] if i != 0 else 0
            t_end = p_sig_cs[i]-1
            p_sig_interval.append([t_start, t_end])
    
    # Post processing to get the true peak
    peak_final = peak_index_1
    if p_sig_interval:
        for idx, interval in reversed(list(enumerate(p_sig_interval))):
            if interval[-1] < peak_index_1:
                break
            
            if interval[0] > peak_index_1:
                # current period max
                t_idx = np.array(range(interval[0], interval[1]+1))
                if prod_data[t_idx].max() >= prod_data[t_idx][-1]*1.1:
                    t_interval_max = prod_data[t_idx].max()
                else:
                    t_interval_max = prod_data[t_idx][-1]
                t_interval_max_idx = len(t_idx)-list(prod_data[t_idx][::-1]).index(t_interval_max)-1
                
                # pre 6 month mean
                t_pre_idx = np.array(range(max(0, interval[0]-6), interval[0]))
                t_pre_interval_mean = prod_data[t_pre_idx].mean()
                
                # after period max
                t_after_idx = np.array(range(interval[1]+1, len(prod_data)))
                t_after_interval_max = prod_data[t_after_idx].max()
                
                # after 3 points max/mean
                t_interval_max_after_3_max = np.max(prod_data[interval[1]+1:min(len(prod_data), interval[1]+4)])
                t_interval_max_after_3_mean = np.mean(prod_data[interval[1]+1:min(len(prod_data), interval[1]+4)])
                
                # pre 3 points min/mean
                t_interval_max_pre_3_min = np.min(prod_data[interval[0]-3:interval[0]])
                t_interval_max_pre_3_mean = np.mean(prod_data[interval[0]-3:interval[0]])
                
                if t_interval_max_pre_3_min == 0:
                    peak_final = t_idx[t_interval_max_idx]
                    break 
                
                if (t_interval_max > t_after_interval_max*0.95 and
                    t_interval_max >= t_pre_interval_mean*1.5 and
                    t_interval_max >= prod_data[peak_index_1] * 0.4 and
                    t_interval_max_after_3_max >= t_interval_max*0.5 and
                    t_interval_max_pre_3_min >= t_pre_interval_mean*0.4 and
                    t_interval_max_after_3_mean > t_interval_max_pre_3_mean*1.2):
                    peak_final = t_idx[t_interval_max_idx]
                    break
    
    return peak_final



def adjust_peak(fitted_curve,
                no_downtime_data,
                first_TC_month_index,
                extend_length):
    """This function adjust the peak of fitted auto-curve"""
    tar_data = no_downtime_data
    t_max_idx = first_TC_month_index

    curve_ori = fitted_curve
    curve_adjust_peak = np.concatenate([tar_data[:t_max_idx+1],
                                             curve_ori[t_max_idx+1:]])
    # Reallocate the delta between actual qi and auto-curve qi to the
    # following part of data, use length of [post peak actual data],
    # +24 is to make sure we won't make drastic change in short period
    t_length = len(tar_data) - t_max_idx + min(48,extend_length) #
    # Last point only change 1% of the total diff
    t_ratio_last = 0.01
    # Calculate the change weight of first point
    t_ratio_first = 2 / t_length - t_ratio_last
    # Apply a arithmetic sequence, calculate the common_diff
    t_common_diff = (t_ratio_first - t_ratio_last) / (t_length-1)
    # The weights of all adjusted data points
    t_weights = [t_ratio_first - i* t_common_diff for i in range(t_length)]
    # Delta betwen acutal qi and auto-curve qi
    t_delta = curve_ori[t_max_idx] - tar_data[t_max_idx]
    # Calculate the adjusted values
    t_adjust = curve_adjust_peak[t_max_idx+1:(t_max_idx+1+t_length)] +\
        t_delta * np.array(t_weights)
    
    # If negative value appers after adjust the initial rate, ignore the adjustment 
    if min(t_adjust)<0:
        curve_adjust_peak = curve_ori
    else:
        # Replace the original values with adjusted values
        curve_adjust_peak[t_max_idx+1:(t_max_idx+1+t_length)] = t_adjust
      
    # If after the adjust, the next point is higher than peak
    # This happends when there is a extreme hi qi.
    if curve_adjust_peak[t_max_idx+1]>curve_adjust_peak[t_max_idx]:

        t_y_cum = np.cumsum(fitted_curve[t_max_idx:])
        t_y_cum_append = np.diff(t_y_cum)
        # compare the cum of actual data and auto curve, find the first segment where cum-actual==cum-auto
        t_ratio = t_y_cum[:(len(tar_data)-t_max_idx)]/np.cumsum(tar_data[t_max_idx:])
        if len(t_ratio)<=3:
           t_ind_cum_equal = len(t_ratio)
        else:
           t_ind_cum_equal = np.argmax(t_ratio[2:]<1)+2 if min(t_ratio)<1 else len(t_ratio)-1

        # assume we can rebuilt this segment with a straight line (same cum), calculate the ending point value 
        t_first_seg_end_value_auto = 2*(t_y_cum[t_ind_cum_equal-1])/(t_ind_cum_equal)-tar_data[t_max_idx]
        t_first_seg_end_value = 2*(sum(tar_data[t_max_idx:t_max_idx+t_ind_cum_equal]))/(t_ind_cum_equal)-tar_data[t_max_idx]
        if (t_first_seg_end_value > tar_data[t_max_idx] or t_first_seg_end_value_auto > tar_data[t_max_idx]): 
            # If ending point higher than the maximum value of actual curve, 
            # ignore the first segment of auto-curve and use actual data instead.
            # happens when less data point and bad ref curves
            t_first_seg_values = tar_data[t_max_idx:t_max_idx+t_ind_cum_equal]
        else:
            def myFunction(d):
                F = t_y_cum[t_ind_cum_equal-1]-(1-d**t_ind_cum_equal)/(1-d)
                return F
            dGuess = [0.95]
            t_delta = fsolve(myFunction,dGuess)[0]
            t_first_seg_values = np.full(t_ind_cum_equal, tar_data[t_max_idx])*(t_delta**np.array(range(t_ind_cum_equal)))
            t_last_adjust = t_delta**(t_ind_cum_equal-1)
            t_last_fit =  fitted_curve[t_max_idx+t_ind_cum_equal-1]
            if t_last_adjust> t_last_fit*1.03:
                while t_last_adjust> t_last_fit*1.03 and\
                    (t_last_adjust - t_last_fit)/t_last_fit/(1-t_delta) >= 1 and\
                    t_ind_cum_equal < t_length-48:
                    t_move_num = min(6, (t_last_adjust - t_last_fit)/t_last_fit/(1-t_delta))
                    t_ind_cum_equal = min(t_length-48, int(t_ind_cum_equal + t_move_num))
                    def myFunction(d):
                        F = t_y_cum[t_ind_cum_equal-1]-(1-d**t_ind_cum_equal)/(1-d)
                        return F
                    dGuess = [0.95]
                    t_delta = fsolve(myFunction,dGuess)[0]
                    t_first_seg_values = np.full(t_ind_cum_equal, tar_data[t_max_idx])*(t_delta**np.array(range(t_ind_cum_equal)))                         
                    t_last_adjust = t_delta**(t_ind_cum_equal-1)
                    t_last_fit =  fitted_curve[t_max_idx+t_ind_cum_equal-1]
            if t_last_adjust< t_last_fit*0.97:
                while t_last_adjust< t_last_fit*0.97 and\
                    (t_last_fit - t_last_adjust)/t_last_adjust/(1-t_delta) >= 1 and\
                    t_ind_cum_equal > 6:
                    t_move_num = min(6, (t_last_fit - t_last_adjust)/t_last_fit/(1-t_delta))
                    t_ind_cum_equal = max(6, int(t_ind_cum_equal - t_move_num))
                    
                    def myFunction(d):
                        F = t_y_cum[t_ind_cum_equal-1]-(1-d**t_ind_cum_equal)/(1-d)
                        return F
                    dGuess = [0.95]
                    t_delta = fsolve(myFunction,dGuess)[0]
                    t_first_seg_values = np.full(t_ind_cum_equal, tar_data[t_max_idx])*(t_delta**np.array(range(t_ind_cum_equal)))                         
                    t_last_adjust = t_delta**(t_ind_cum_equal-1)
                    t_last_fit =  fitted_curve[t_max_idx+t_ind_cum_equal-1]

        t_y_cum_append = t_y_cum_append[(t_ind_cum_equal-1):]
        curve_adjust_peak = np.concatenate((np.array(tar_data[:t_max_idx]),t_first_seg_values))
        curve_adjust_peak = np.concatenate((curve_adjust_peak,t_y_cum_append))
        
    return curve_adjust_peak


def Hyperbolic_Rate(t:int, qi:float, Di:float, b:float) -> float:
    """
    Hyperbolic decline rate function based on Arps equation.
    
    Parameters:
    - t: Time.
    - qi: Initial production rate.
    - Di: Initial decline rate.
    - b: Decline coefficient.
    
    Returns:
    - The production rate at time t.
    """
    return qi / ((1 + b * Di * t) ** (1/b))


def Harmonic_Rate(t: int, qi: float, Di: float) -> float:
    """
    Harmonic decline rate function based on Arps equation.

    Parameters:
    - t: Time.
    - qi: Initial production rate.
    - Di: Initial decline rate.

    Returns:
    - The production rate at time t.
    
    """
    return qi / (1 + Di * t)

def Exponential_Rate(t: int, qi: float, Di: float) -> float:
    """
    Exponential decline rate function based on Arps equation.

    Parameters:
    - t_month: Time.
    - qi: Initial production rate.
    - Di: Initial decline rate.

    Returns:
    - The production rate at time t.
    
    """
    return qi * np.exp(-Di * t)

def Constant_Rate(t: int, qi: float, Di: float) -> float:
    """
    Constant decline rate function.

    Parameters:
    - t_month: Time.
    - qi: Initial production rate.
    - Di: Initial decline rate.

    Returns:
    - The production rate at time t.
    """
    return qi * (1 - Di * t)


class ProcessShutSeqs:
    """
    A class for holding the following data of a well:
   
    This class does The following function:

    1. Find peak of the well
    2. Fit to Arps equation
    3. Generate decline curve
    4. Adjust peak (qi) or the decline curve
    5. Modify for terminal decline
    6. Add back downtime
    """

    def __init__(self,  raw_data, fit_equation='hyperbolic', forecast_length=360,
                 peak_month=None, downtime_threshold=20, extend_length=0):
        self.data_raw = raw_data # raw data
        # Parameter boundaries for [qi,b,Di] 
        self.fit_params = []
        self.fit_equation = fit_equation
        # The length of extra months decline-curve to be generated
        self.forecast_length = forecast_length
        # how many months to extend the curve as output
        self.extend_length = extend_length
        # If the user define a peak month
        self.peak_month = peak_month
        # The prod threshold to be identified as downtime
        self.downtime_threshold = downtime_threshold


        if self.fit_equation == 'hyperbolic':
            self.fit_func = Hyperbolic_Rate
        elif self.fit_equation == 'harmonic':
            self.fit_func = Harmonic_Rate
        elif self.fit_equation == 'exponential':
            self.fit_func = Exponential_Rate

    def find_peak(self):
        """
        This function finds the peak of the well.
        """
        if self.peak_month is None:
            self.peak_month = get_peak_index(self.data_raw)
        self.peak_rate = self.data_raw[self.peak_month]
        self.params_bound = ([0.5*self.peak_rate, 0.5, 0.05], [2*self.peak_rate, 1.5, 4])
        # self.data_norm = [dr / self.data_raw[self.peak] for dr in self.data_raw]
        self.data_post_peak = self.data_raw[self.peak_month:]


    def remove_downtime(self):
        """
        This function removes the downtime from the data.
        """
        self.data_post_peak_no_down = [p for p in self.data_post_peak if p >= self.downtime_threshold]
        self.data_post_peak_downtime = [i for i, p in enumerate(self.data_post_peak ) if p < self.downtime_threshold]


    def find_fit_params(self):
        """
        This function finds the fit parameters of the well.
        """

        t_months = np.arange(len(self.data_post_peak_no_down))
        t_prod = np.array(self.data_post_peak_no_down)

        try:
            params, cov = curve_fit(self.fit_func, t_months, t_prod,
                                    p0=[self.peak_rate, 0.8, 0.8], bounds=self.params_bound)
            self.fit_status = 'success'
            self.fit_params = params
            self.fit_cov = cov
        except:
            self.fit_params = np.array([-1, -1, -1])
            self.fit_cov = [-1]
            self.fit_status = 'fail'

    def generate_decline_curve(self):
        """
        This function generates the decline curve of the well.
        """

        t_months = np.arange(len(self.data_post_peak_no_down)+self.forecast_length)
        ori_fit_data = self.fit_func(t_months, *self.fit_params)

        self.ori_fit_data = ori_fit_data

    def adjust_peak(self):
        """
        This function adjusts the peak of the well.
        """
        self.adjust_fit_data = adjust_peak(self.ori_fit_data,
                                           self.data_post_peak_no_down,
                                           0,
                                           self.forecast_length)
    

    def add_terminal_decline(self):
        """ This function address for the terminal decline"""
        tar_curve = self.adjust_fit_data.copy()
        tar_data = self.data_post_peak_no_down.copy()
        t_max_idx = 0

        # Calculate effective yearly-decline
        t_dec_yearly = np.array([(tar_curve[i] - tar_curve[i+12])/tar_curve[i]\
                                for i in range(t_max_idx, len(tar_curve)-12)])
        # Due to the fluctuation of raw data, negative decline rate may happen
        t_dec_yearly[t_dec_yearly <= 0] = 0.1
        # If the yearly declin rate at end point of actual data is lower than
        # 5%, then apply 80% of the end decline rate for terminal decline,
        # otherwith, apply 5% as the terminal decline rate.
        if t_dec_yearly[len(tar_data)-t_max_idx-1] > 0.05:
            t_dec_yearly_terminal = 0.05
        else:
            t_dec_yearly_terminal = t_dec_yearly[len(tar_data)-t_max_idx-1]*0.8
        # Get the index of terminal decline period
        t_dec_idx, = np.where(t_dec_yearly < t_dec_yearly_terminal)
        # If at the end of auto-curve, the effective decline is lower than 5%,
        # apply terminal decline.
        if t_dec_idx.size != 0:
            t_dec_idx = t_dec_idx + t_max_idx
            # Modify for terminal decline
            t_dec_idx = np.concatenate([t_dec_idx, np.array(
                range(t_dec_idx[-1]+1, 360+len(tar_data)))])
            t_dec_coef = np.array([((1-t_dec_yearly_terminal)**(1/12))**i for
                                   i in range(len(t_dec_idx))])
            tar_curve[t_dec_idx] = tar_curve[t_dec_idx[0]] * t_dec_coef

        self.adjust_fit_data_with_terminal = tar_curve

    def add_back_downtime(self):
        """
        This function adds the downtime back to the data.
        """
        if len(self.data_post_peak_downtime) > 0:
            self.fit_data_post_peak = self.adjust_fit_data_with_terminal.copy()
            for index in self.data_post_peak_downtime:
                self.fit_data_post_peak = np.insert(self.fit_data_post_peak, index, self.data_post_peak[index])
        else:
            self.fit_data_post_peak = self.adjust_fit_data_with_terminal.copy()

    def add_back_pre_peak(self):
        """
        This function adds the pre-peak data back to the data.
        """
        self.fit_data_final = np.concatenate([self.data_raw[:self.peak_month], self.fit_data_post_peak])

    def prepare_final_output(self):
        """
        This function prepares the final output of the well.
        """
        self.fit_data_output = self.fit_data_final[:len(self.data_raw)+self.extend_length]