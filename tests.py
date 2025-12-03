from functools import partial
from sklearn.svm import SVC
from joblib import Parallel, delayed
from scipy.stats import binomtest
from sklearn.preprocessing import MinMaxScaler

from utils import __semitest__

import numpy as np
import pandas as pd
import time

# reverse causality testing function
def rc_test(X, Y, base_classifier = partial(SVC, kernel = 'poly', degree = 3, coef0 = 0, probability = True), label_len = -1, job_rep = 100, normalize = False, y_is_binary = True):

    if label_len == -1:
        label_len = round(X.shape[0]/10)
    
    # Handle Y input (flatten if 2D)
    if Y.ndim > 1:
        y_values = Y[:, 0]  # Take first column if 2D
    else:
        y_values = Y        # Use as-is if 1D
    
    # Convert Y based on user flag
    if y_is_binary:
        caus = y_values
    else:
        print("Converting Y from continuous to binary using median split (user specified)")
        median_val = np.median(y_values)
        caus = y_values > median_val
        print(f"Median threshold: {median_val:.3f}, Above median: {np.sum(caus)}, Below median: {np.sum(~caus)}")
        
    effe = X  # Use all columns
    
    if normalize:
        effe = MinMaxScaler().fit_transform(effe)

    # Pass the function (not an instance) to create fresh classifiers in each process
    job_call = partial(__semitest__, base_classifier, effe, caus, label_len)
    results = np.array(Parallel(n_jobs = job_rep, backend = "threading")(map(delayed(job_call), range(job_rep))))

    # return two p-values
    return [
        binomtest(np.sum(results[:,0] < results[:,1]), np.sum(results[:,0] < results[:,1]) + np.sum(results[:,0] > results[:,1]), alternative='greater').pvalue, 
        binomtest(np.sum(results[:,0] < results[:,1]), results.shape[0], alternative='greater').pvalue
    ]

def main():
    # Record start time
    start_time = time.time()
    
    # load data
    df_wide = pd.read_excel('data/N1000_Wide.xlsx')
    
    # Convert to numpy arrays immediately
    x = df_wide[["jbmsall2", "jbmsall3", "jbmsall4", "jbmsall5"]].values
    y = df_wide[["jbmtuea1"]].values

    print(f"Data loaded. Shape: X={x.shape}, Y={y.shape}")
    print(f"Data types: X={type(x)}, Y={type(y)}")
    print(f"Running reverse causality test")
    
    # Test for reverse causality
    p_values = rc_test(
        X=x,  # Now numpy array
        Y=y,  # Now numpy array
        job_rep=1000,        #  bootstrap repetitions for timing test
        normalize=True,    # Normalize features
        y_is_binary=True   # Flag: True if Y is already binary, False if continuous
    )
    test_end = time.time()

    # Calculate and display timing results
    total_duration = test_end - start_time
    
    print(f"\n=== TIMING RESULTS ===")
    print(f"Total execution time: {total_duration:.2f} seconds")
    
    print(f"\n=== RESULTS ===")
    print(f"Evidence of reverse causality:")
    print(f"P-value (excluding ties): {p_values[0]:.4f}")
    print(f"P-value (including ties): {p_values[1]:.4f}")

if __name__ == '__main__':
    main()