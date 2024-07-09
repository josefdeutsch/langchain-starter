import numpy as np

def calculate_hurst(ts):
    lags = range(2, 100)
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]*2.0

def calculate_hurst2(ts):
    lags = range(2, 100)
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    
    # Add a small constant to avoid log(0)
    tau = np.array(tau) + 1e-10
    lags = np.array(lags) + 1e-10
    
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]*2.0

def calculate_hurst3(ts):
    lags = range(2, 100)
    tau = []
    for lag in lags:
        if len(ts) > lag:
            lagged_diff = np.subtract(ts[lag:], ts[:-lag])
            if len(lagged_diff) > 1:
                tau.append(np.std(lagged_diff))
            else:
                tau.append(np.nan)
        else:
            tau.append(np.nan)
    
    tau = np.array(tau)
    valid = ~np.isnan(tau) & (tau > 0)
    
    tau = tau[valid] + 1e-10
    lags = np.array(lags)[valid] + 1e-10
    
    if len(tau) < 2:
        return np.nan
    
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]*2.0

def calculate_hurst4(ts):
    # Define range of lags
    lags = np.arange(2, 100)
    
    # Calculate the array of tau values
    tau = np.array([np.std(ts[lag:] - ts[:-lag]) for lag in lags])
    
    # Filter out invalid (zero or NaN) tau values
    valid = np.isfinite(tau) & (tau > 0)
    lags = lags[valid]
    tau = tau[valid]
    
    if len(tau) < 2:
        return np.nan
    
    # Perform linear regression on log-log scale
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    
    return poly[0] * 2.0

def calculate_hurst5(ts):
    lags = range(2, 100)
    tau = []
    
    for lag in lags:
        if len(ts) <= lag:
            break
        
        lagged_diff = ts[lag:] - ts[:-lag]
        if len(lagged_diff) > 1:
            tau.append(np.std(lagged_diff))
    
    tau = np.array(tau)
    
    # Filter out invalid (zero or NaN) tau values
    valid = (tau > 0)
    lags = np.array(lags[:len(tau)])[valid]
    tau = tau[valid]
    
    if len(tau) < 2:
        return np.nan
    
    # Perform linear regression on log-log scale
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

ts = np.array([11.17207184, 11.86265515, 12.26049217, 12.544247, 12.76605094, 
               12.94717416, 13.09597649, 13.22713356, 13.3407013, 13.441538, 
               13.53340476, 13.61806386, 13.69595148, 13.76682574, 13.83279078, 
               13.89456541, 13.95209501, 14.00663054, 14.05749465, 14.10374575, 
               14.14902495, 14.19167066, 14.23310015, 14.27205972, 14.30988786, 
               14.34633704, 14.38253735, 14.4175553, 14.4509589, 14.48232145, 
               14.51117775, 14.53903973, 14.56689427])


hurst_exponent = calculate_hurst3(ts)
hurst_exponent = calculate_hurst4(ts)
hurst_exponent = calculate_hurst5(ts)
print("Hurst Exponent:", hurst_exponent)
