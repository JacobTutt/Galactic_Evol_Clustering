import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import pandas as pd

from astropy.table import Table
from tqdm.notebook import tqdm
import pickle as pkl

from extreme_deconvolution import extreme_deconvolution
from sklearn.preprocessing import StandardScaler

from typing import List, Tuple, Optional, Union

def BICScore(log_likelihood, num_params, num_data_points):
    """
    Bayesian Information Criterion (BIC) 

    A model selection criterion that balances model fit and complexity by penalizing the number of parameters more strongly than AIC, especially for larger datasets.

    Parameters: 
        log_likelihood (float): Overall Log likelihood of the model
        num_params (int): Number of parameters in the model
        num_data_points (int): Number of data points in the dataset

    Returns:
        float: BIC score

    """
    return - 2 * log_likelihood + num_params * np.log(num_data_points)

def AICScore(log_likelihood, num_params):
    """
    Akaike Information Criterion (AIC) 

    A model selection criterion that balances model fit and complexity by penalizing the number of parameters with a constant factor (2 per parameter).

    Parameters: 
        log_likelihood (float): Overall Log likelihood of the model
        num_params (int): Number of parameters in the model

    Returns:
        float: AIC score

    """
    return - 2 * log_likelihood + 2 * num_params

def run_XD_BIC(data, data_keys: List[str], data_err_keys: List[str], component_range: Tuple[int, int] = (1, 10), max_iterations: int = int(1e10), n_repeats: int = 3, n_init: int = 100, save_path: Optional[str] = None) -> Tuple[dict, dict]:
    """
    Run Extreme Deconvolution with varying number of Gaussian components and multiple initialisations to ensure convergence.

    Parameters:
        data (Astropy Table): Full filtered dataset, including parameters and their errors with additional columns not used for XD.
        data_keys (list): List of keys for the parameters of interest in the data table.
        data_err_keys (list): List of keys for the errors of the parameters of interest in the data table. These must be in order and correspond to the data_keys.
        component_range (tuple): Range of gaussian components to test, (min, max).
        max_iterations (int): Maximum number of EM iterations for each run.
        n_repeats (int): Number of complete repetitions of the 100 initialisations.
        n_init (int): Number of random initializations per component count.
        save_path (str): Path to save the results to.

    Returns:
        dict: Contains BIC, AIC, and best-fit parameters for each component count.
        dict: Contains the best BIC score and the parameters in which it was achieved
    """

    # Entry checks
    # Check that the number of data keys and error keys match
    if len(data_keys) != len(data_err_keys):
        raise ValueError("Number of data keys and error keys must match")

    # Check if data is an Astropy Table or recarray and extract column names accordingly
    if hasattr(data, 'colnames'):  # For Astropy Table
        colnames = data.colnames
    elif hasattr(data, 'dtype'):   # For recarray (FITS_rec)
        colnames = data.dtype.names
    else:
        raise TypeError("Unsupported data type. Must be Astropy Table or FITS_rec/recarray.")
    
    # Check that the data keys and error keys are present in the data table
    # Print the missing keys if they are not present
    missing_keys = [key for key in data_keys + data_err_keys if key not in colnames]
    if missing_keys:
        raise ValueError(f"Keys {missing_keys} not found in data table")
    
    # Check that the component range is valid
    if not isinstance(component_range, tuple) or len(component_range) != 2:
        raise ValueError("Gaussian component range must be a tuple of form (min, max)")
    
    if component_range[0] > component_range[1]:
        raise ValueError("Invalid gaussian component range") 
    
    # Check that the number of repeats and initialisations are valid
    if not isinstance(n_repeats, int) or n_repeats <= 0:
        raise TypeError("n_repeats must be a positive integer")
    if not isinstance(n_init, int) or n_init <= 0:
        raise TypeError("n_init must be a positive integer")
    if not isinstance(max_iterations, int) or max_iterations <= 0:
        raise TypeError("max_iterations must be a positive integer")


    # Extract the data from the astropy table using the keys provided and stack them into a 2D array, (number_of_parameters, number_of_samples)
    data_array = np.vstack([data[key] for key in data_keys]).T

    # Extract the errors from the astropy table using the keys provided
    # We have no informatiom on the correlation between the errors so assume they are diagonal/uncorrelated. 
    errors_array = np.vstack([data[err_key] for err_key in data_err_keys]).T

    # Extract the number of samples and features from the data array
    # Alternatively, n_features = len(data_keys) and n_samples = len(data_array)
    n_samples, n_features = data_array.shape

    # Scale the data to have zero mean and unit variance - this will improve the convergence of the EM algorithm
    # Errors are scaled by the same factor to maintain the same relative uncertainty
    # Note this will require the gaussians means and covariances returned by XD to be scaled back to the original units for interpretation before saving
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_array)
    errors_scaled = errors_array / scaler.scale_

    # Calculate the extreme values of the data for initialisation randomisation
    extreme_data_values = (np.max(data_array, axis=0), np.min(data_array, axis=0))

    # Initialise the results dictionary for dynamic appending
    results = {
        # Repeat Number
        "repeat no.": [],
        # Intialisation Number
        "init no": [],
        # Number of gaussians fitted
        "n_gauss": [],
        # Log likelihood of the best-fit model
        "log_likelihood": [],
        # Bayesian Information Criterion
        "BIC": [],
        # Akaike Information Criterion
        "AIC": [],
        # Best-fit weights of the gaussians
        "weights": [],
        # Best-fit means of the gaussians
        "means": [],
        # Best-fit covariances of the gaussians
        "covariances": []
    }

    # Track the best BIC and initialise the best parameters
    best_BIC = np.inf
    best_params = {"BIC": None, "n_gauss": None, "repeat": None, "init": None, "means": None}

    # Iterate for a test range of number of gaussians
    for n_gauss in tqdm(range(component_range[0], component_range[1] + 1), desc="Number of Gaussian Components"):
        # Overall repeats 
        for n in tqdm(range(n_repeats), desc="Repeat Cycles", leave=False):
            # Random initialisations of input parameters
            for i in tqdm(range(n_init), desc="Initialisations", leave=False):

                # Random initiatlisation of weights
                # init_weights = np.random.dirichlet(np.ones(n_gauss))
                # Even initialisation of weights
                init_weights = np.ones(n_gauss) / n_gauss

                # Random initialisation of means - using the extreme values of each parameter
                init_mean = np.random.uniform(low=extreme_data_values[0], high=extreme_data_values[1], size=(n_gauss, n_features))

                # Covariances initialised as identity matrices
                init_covar = np.array([np.identity(n_features) for _ in range(n_gauss)])
                # Run XD
                try:
                    XD_avg_LL = extreme_deconvolution(
                        data_scaled, errors_scaled, init_weights, init_mean, init_covar, maxiter=max_iterations)
                    # Calculate the total log likelihood
                    total_LL = XD_avg_LL * n_samples

                    # Calculate the bic and aic scores
                    num_params = n_gauss * (1 + n_features + n_features * (n_features + 1) // 2) - 1
                    bic, aic = BICScore(total_LL, num_params, n_samples), AICScore(total_LL, num_params)

                    # Copy the updated weights, means and covariances
                    post_XD_weights, post_XD_means, post_XD_cov = init_weights.copy(), init_mean.copy(), init_covar.copy()

                    # Unscale the means and covariances to return them to their original/meaningful units
                    post_scaling_means = scaler.inverse_transform(post_XD_means)
                    post_scaling_cov = np.array([
                        np.dot(np.dot(np.diag(scaler.scale_), cov), np.diag(scaler.scale_))
                        for cov in post_XD_cov
                    ])

                    # Keep track of the best BIC and parameters in which it was achieved
                    if bic < best_BIC:
                        best_BIC = bic
                        best_params.update({"BIC": bic, "n_gauss": n_gauss, "repeat": n, "init": i, "means": post_scaling_means})

                    # Store the results
                    results["repeat no."].append(n)
                    results["init no"].append(i)
                    results["n_gauss"].append(n_gauss)
                    results["log_likelihood"].append(total_LL)
                    results["BIC"].append(bic)
                    results["AIC"].append(aic)
                    results["weights"].append(post_XD_weights)
                    results["means"].append(post_scaling_means)
                    results["covariances"].append(post_scaling_cov)


                except Exception as e:
                    print(f"XD failed for {n_gauss} components, on repeat: {n}, iteration: {i}: {e}")

                    # Store the results
                    results["repeat no."].append(n)
                    results["init no"].append(i)
                    results["n_gauss"].append(n_gauss)
                    results["log_likelihood"].append(None)
                    results["BIC"].append(None)
                    results["AIC"].append(None)
                    results["weights"].append(None)
                    results["means"].append(None)
                    results["covariances"].append(None)

        # Save the results if a path is provided
        # This is redone for each component count to ensure that the results are saved in case of a crash
        if save_path:
            try:
                with open(save_path, "wb") as f:
                    pkl.dump(results, f)
                print(f"Results saved successfully at {save_path}")
            except Exception as e:
                print(f"Failed to save results: {e}")

    # Final summary of best BIC
    print(f" Best BIC Score: {best_BIC:.4f}")
    print(f"   - Components (n_gauss): {best_params['n_gauss']}")
    print(f"   - Repeat cycle (n): {best_params['repeat']}")
    print(f"   - Initialisation (i): {best_params['init']}")
    print(f"   - Best initialisation of Mean values: \n{best_params['means']}")

    return results, best_params


def BIC_analysis(saved_path):
    """
    Analyse the results of the Extreme Deconvolution (XD) 
    
    Runs by computing and visualising the BIC/AIC scores for each Gaussian component count.

    This function:
    - Loads XD results from a pickle file.
    - Calculates failed XD runs per Gaussian component.
    - Computes mean and standard deviation of BIC and AIC scores.
    - Displays formatted tables of results.
    - Plots combined BIC and AIC curves with lowest, highest, and median scores.

    Parameters:
    ------------
    saved_path : str
        The file path to the pickle file containing XD results, including BIC and AIC scores.

    Returns:
    --------
    None
        Displays tables and plots summarizing the BIC/AIC analysis for each Gaussian component count.
    """

    # Loads the saved results from the XD run
    with open(saved_path, "rb") as f:
        XD_results = pkl.load(f)

    # Extracts unique Gaussian component counts and define the range
    component_range = (min(XD_results["n_gauss"]), max(XD_results["n_gauss"]))
    n_gauss_list = np.array([n for n in range(component_range[0], component_range[1] + 1)])

    # Determines the number of initializations and repeats
    n_init = max(XD_results["init no"]) + 1
    n_repeats = max(XD_results["repeat no."]) + 1

    # Creates a DataFrame summarising the number of failed XD runs per Gaussian component
    n_XD_failed = [
        sum([b is None for c, b in zip(XD_results["n_gauss"], XD_results["BIC"]) if c == n_gauss])
        for n_gauss in n_gauss_list
    ]
    n_runs_gauss = np.array([n_repeats * n_init for _ in n_gauss_list])
    n_failed_XD = pd.DataFrame({
        "No. Gaussians": n_gauss_list,
        "No. Failed XD runs": n_XD_failed,
        "Total No. Runs": n_runs_gauss
    })

    # Prints a formatted table of the number of failed XD runs
    print("Table of Number of Gaussians vs Number of Failed XD Runs")
    print(tabulate(n_failed_XD, headers='keys', tablefmt='psql'))

    # Reshapes BIC and AIC scores to 3D arrays: (n_gauss_components, n_repeats, n_init)
    BIC_scores = np.array(XD_results['BIC']).reshape((len(n_gauss_list), n_repeats, n_init))
    AIC_scores = np.array(XD_results['AIC']).reshape((len(n_gauss_list), n_repeats, n_init))

    # Compute means and standard deviations for BIC and AIC across initialisations
    BIC_means, BIC_stds = np.mean(BIC_scores, axis=2), np.std(BIC_scores, axis=2)
    AIC_means, AIC_stds = np.mean(AIC_scores, axis=2), np.std(AIC_scores, axis=2)

    # Format mean ± stddev for BIC and AIC into DataFrames
    BIC_means_stds_df = pd.DataFrame(
        np.array([[f"{mean:.5f} \\pm {std:.5f}" for mean, std in zip(mean_row, std_row)]
                  for mean_row, std_row in zip(BIC_means, BIC_stds)]),
        columns=[f"Repeat {i + 1}" for i in range(n_repeats)],
        index=n_gauss_list
    )
    BIC_means_stds_df.insert(0, "No. Gaussians", n_gauss_list)

    AIC_means_stds_df = pd.DataFrame(
        np.array([[f"{mean:.5f} \\pm {std:.5f}" for mean, std in zip(mean_row, std_row)]
                  for mean_row, std_row in zip(AIC_means, AIC_stds)]),
        columns=[f"Repeat {i + 1}" for i in range(n_repeats)],
        index=n_gauss_list
    )
    AIC_means_stds_df.insert(0, "No. Gaussians", n_gauss_list)

    # Print BIC and AIC summary tables
    print("Table of BIC Means and Stds")
    print(tabulate(BIC_means_stds_df, headers='keys', tablefmt='psql'))

    print("Table of AIC Means and Stds")
    print(tabulate(AIC_means_stds_df, headers='keys', tablefmt='psql'))

    # Plot the lowest, highest, and median BIC and AIC scores for each component count
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Calculate min, max, and median BIC and AIC scores across repeats and initialisations for each Gaussian component count ie from n_init * n_repeats values - (n_gauss_components)
    BIC_min, BIC_max, BIC_median = BIC_scores.min(axis=(1, 2)), BIC_scores.max(axis=(1, 2)), np.median(BIC_scores, axis=(1, 2))
    AIC_min, AIC_max, AIC_median = AIC_scores.min(axis=(1, 2)), AIC_scores.max(axis=(1, 2)), np.median(AIC_scores, axis=(1, 2))

    # Plot combined BIC & AIC
    fig, ax = plt.subplots(figsize=(10, 6))
    # BIC - Blue
    ax.plot(n_gauss_list, BIC_min, 'b-', label="BIC - Lowest (Solid)")
    ax.plot(n_gauss_list, BIC_max, 'b--', label="BIC - Highest (Dashed)")
    ax.plot(n_gauss_list, BIC_median, 'b:', label="BIC - Median (Dotted)")

    # AIC - Red
    ax.plot(n_gauss_list, AIC_min, 'r-', label="AIC - Lowest (Solid)")
    ax.plot(n_gauss_list, AIC_max, 'r--', label="AIC - Highest (Dashed)")
    ax.plot(n_gauss_list, AIC_median, 'r:', label="AIC - Median (Dotted)")

    ax.set_xlabel("Number of Gaussian Components", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("BIC and AIC Score Analysis for Gaussian Components", fontsize=14)
    ax.legend(loc='best')
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    return AIC_means_stds_df, BIC_means_stds_df