import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

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

# def BIC_analysis(saved_path):
#     """
#     Analyse the results of the XD and plot the BIC/AIC scores for each component count.

#     Parameters:
#         saved_path (str): Path to the pickle file containing BIC results.

#     Returns:
#         None
#     """
#     # Load the saved results from the XD Run
#     with open(saved_path, "rb") as f:
#         XD_results = pkl.load(f)

#     # Extract unique Gaussian component counts
#     component_range = XD_results_df["n_gauss"].dropna().unique()
#     component_range.sort()

#     # Determine number of initializations and repeats
#     n_init = XD_results_df["init no"].max() + 1
#     n_repeats = XD_results_df["repeat no."].max() + 1

#     # Create table of no gassians vs number of failed XD Runs
#     n_failed_XD = (
#         XD_results_df.groupby(["n_gauss", "repeat no."])        # Group by Gaussian components and the repeats
#         .agg(Total_Runs=("BIC", "count"),               # Count total runs (including successful and failed)
#             Failed_XD_Runs=("BIC", lambda x: x.isna().sum()))  # Count failed runs (BIC is NaN)
#         .reindex(component_range, fill_value=0)         # Ensure all Gaussian components are represented
#         .reset_index()                                  # Reset index for a clean DataFrame
#     )

#     # print the results - this will also be returned by the function
#     print("Table of Number of Gaussians vs Number of Failed XD Runs")
#     print(tabulate(n_failed_XD, headers='keys', tablefmt='psql'))

#     # Create a table of initialisation results across repeats
#     spread_results = = (
#         XD_results_df.groupby("n_gauss")
#     )


#     # Ensure consistentcy across repeats
#     # For each component and repeat work out mean and std error for each component across initialisations:
    



#     # Prepare data for plotting
#     BIC_means, BIC_lows, BIC_highs = [], [], []
#     AIC_means, AIC_lows, AIC_highs = [], [], []


#     for n_gauss in component_range:
#         BIC_scores = [b for c, b in zip(XD_results["n_gauss"], XD_results["BIC"]) if c == n_gauss and b is not None]
#         AIC_scores = [a for c, a in zip(XD_results["n_gauss"], XD_results["AIC"]) if c == n_gauss and a is not None]

#         if BIC_scores and AIC_scores:
#             BIC_means.append(np.min(BIC_scores))
#             BIC_lows.append(np.percentile(BIC_scores, 25))
#             BIC_highs.append(np.percentile(BIC_scores, 75))

#             AIC_means.append(np.min(AIC_scores))
#             AIC_lows.append(np.percentile(AIC_scores, 25))
#             AIC_highs.append(np.percentile(AIC_scores, 75))

#     # Plotting
#     fig, ax1 = plt.subplots(figsize=(10, 6))

#     ax1.plot(unique_components, AIC_means, label="AIC", color="red", linewidth=2)
#     ax1.fill_between(unique_components, AIC_lows, AIC_highs, color="red", alpha=0.3)
#     ax1.set_xlabel("Number of Gaussian Components")
#     ax1.set_ylabel("AIC", color="red")
#     ax1.tick_params(axis='y', labelcolor="red")

#     ax2 = ax1.twinx()
#     ax2.plot(unique_components, BIC_means, label="BIC", color="blue", linewidth=2)
#     ax2.fill_between(unique_components, BIC_lows, BIC_highs, color="blue", alpha=0.3)
#     ax2.set_ylabel("BIC", color="blue")
#     ax2.tick_params(axis='y', labelcolor="blue")

#     # Highlight optimal BIC component
#     optimal_component = unique_components[np.argmin(BIC_means)]
#     ax1.axvline(optimal_component, linestyle='--', color='gray', label=f"Optimal BIC = {optimal_component}")

#     # Add zoomed-in inset
#     from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#     axins = inset_axes(ax1, width="30%", height="30%", loc='upper right')
#     axins.plot(unique_components, BIC_means, color="blue")
#     axins.set_xlim(max(optimal_component - 1, unique_components[0]),
#                    min(optimal_component + 1, unique_components[-1]))
#     axins.set_ylim(min(BIC_means) - 500, min(BIC_means) + 500)
#     axins.tick_params(labelleft=False, labelbottom=False)

#     fig.suptitle("BIC and AIC Analysis for Extreme Deconvolution Results")
#     ax1.legend(loc="upper left")
#     plt.show()



    


#     # Convert the results to a pandas dataframe
#     results_df = pd.DataFrame(results)

#     # Group the results by the number of gaussians and calculate the mean BIC score
#     grouped_results = results_df.groupby("n_gauss").agg({"BIC": "mean"})

#     # Plot the BIC scores
#     plt.figure(figsize=(10, 6))
#     plt.plot(grouped_results.index, grouped_results["BIC"], marker="o")
#     plt.xlabel("Number of Gaussian Components")
#     plt.ylabel("BIC Score")
#     plt.title("BIC Score vs Number of Gaussian Components")
#     plt.grid(True)
#     plt.show()

#     return None