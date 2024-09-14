import pandas as pd
import numpy as np
from multiprocessing import Pool

def calculate_adi(data: pd.DataFrame, clip: bool=True) -> pd.Series:
    # The total number of periods
    total_periods = data.shape[1]  # Number of columns
    non_zero_counts_per_row = data.astype(bool).sum(axis=1)
    adi = total_periods / non_zero_counts_per_row
    adi.name = 'ADI'
    if clip:
        adi = adi.replace([np.inf, -np.inf], np.nan)
        return adi.fillna(adi.max() + 1)
    return adi

def calculate_cv(data: pd.DataFrame) -> pd.Series:
    result = data.std(axis=1) / data.mean(axis=1)
    max_cv = int(result.max()) + 1
    result = result.fillna(max_cv)  # Inf values from division by zero are set to max value + 1.
    return result

def croston_pattern(data):
    if not isinstance(data, pd.Series):
        raise TypeError("Expected a Pandas Series.")

    # Find indices where nonzero values occur
    nonzero_arrs = data[data > 0].tolist()
    nonzero_arrs = [int(value) for value in nonzero_arrs]
    nonzero_indices = data[data > 0].index

    if nonzero_indices.empty:
        nonzero_arrs = list()
        zero_intervals = [len(data)]
        return nonzero_arrs, zero_intervals

    nonzero_intervals = np.diff(nonzero_indices)
    zero_intervals = []
    zero_intervals.append(nonzero_indices[0])
    zero_intervals.extend(nonzero_indices[1:] - nonzero_indices[:-1] - 1)
    
    if nonzero_indices[-1] < len(data) - 1:
        zero_intervals.append(len(data) - nonzero_indices[-1] - 1)

    return nonzero_arrs, zero_intervals

def croston_worker(data: pd.Series):
    nonzero_arrs, zero_intervals = croston_pattern(data)
    nonzero_arrs = [int(value) for value in nonzero_arrs]
    
    if nonzero_arrs:
        freq_min = min(nonzero_arrs)
        freq_max = max(nonzero_arrs)
        freq_avg = sum(nonzero_arrs) / len(nonzero_arrs)
    else:
        freq_min = 0
        freq_max = 0
        freq_avg = 0
    
    if zero_intervals:
        cycle_recent = zero_intervals[-1]
        cycle_avg = sum(zero_intervals) / len(zero_intervals)
    else:
        cycle_recent = 0
        cycle_avg = 0
    
    return freq_min, freq_max, freq_avg, cycle_recent, cycle_avg

def calculate_croston(ts_features: pd.DataFrame)->pd.DataFrame:
    with Pool() as pool:
        croston_results = pool.map(croston_worker, [ts_features.loc[idx, :].reset_index(drop=True) for idx in range(ts_features.shape[0])])
    
    freq_min, freq_max, freq_avg, cycle_recent, cycle_avg = zip(*croston_results)
    
    return pd.DataFrame({
        'FreqMin': freq_min,
        'FreqMax': freq_max,
        'FreqAvg': freq_avg,
        'PurchaseCycleRecent': cycle_recent,
        'PurchaseCycleAvg': cycle_avg
    })

def extract_order_features(data: np.ndarray, short_term_window: int, long_term_window: int) -> np.array:
    """
    Extract order features from the given data.
    
    Parameters:
    - data: np.ndarray with order amounts.
    - short_term_window: int, the window size for the short term.
    - long_term_window: int, the window size for the long term.
    
    Returns:
    -  np.ndarray : ['Order Ratio', 'Order Probability', 'Order Frequency Count']
    """
    if len(data) < max(short_term_window, long_term_window):
        raise ValueError("Data length must be greater than or equal to the maximum window size.")
    sum_orders_short_term = np.sum(data[-short_term_window:])
    total_orders_long_term = np.sum(data[-long_term_window:])

    order_ratio = sum_orders_short_term / total_orders_long_term if total_orders_long_term != 0 else 0
    order_probability = round(np.count_nonzero(data[-short_term_window:]) / len(data[-long_term_window:]), 2)
    order_frequency_count = np.count_nonzero(data[-long_term_window:])
    
    return np.array([order_ratio, order_probability,order_frequency_count])

def naive_mean_model(data: np.ndarray) -> float:
    """
    Predict the next value using the naive mean model.
    
    Parameters:
    - data: np.ndarray, the time series data.
    
    Returns:
    - float: The predicted next value.
    """
    if len(data) == 0:
        raise ValueError("Data array is empty.")
    
    mean_value = np.mean(data)
    return mean_value