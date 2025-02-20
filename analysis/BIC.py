import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from extreme_deconvolution import extreme_deconvolution
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from astropy.table import Table
from sklearn.preprocessing import StandardScaler

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
    
def run_extreme_deconvolution(data, data_keys, data_err_keys, component_range=(1, 10), max_iterations=int(1e10), n_repeats=3, n_init=100):
    """
    Run Extreme Deconvolution with varying number of Gaussian components and multiple initialisations to ensure convergence.

    Parameters:
        data (Astropy Table): Full filtered dataset, including parameters and their errors with additional columns not used for XD.
        data_keys (list): List of keys for the parameters of interest in the data table.
        data_err_keys (list): List of keys for the errors of the parameters of interest in the data table. These must be in order and correspond to the data_keys.
        component_range (tuple): Range of components to test, (min, max).
        max_iterations (int): Maximum number of EM iterations for each run.
        n_repeats (int): Number of complete repetitions of the 100 initialisations.
        n_init (int): Number of random initializations per component count.

    Returns:
        dict: Contains BIC, AIC, and best-fit parameters for each component count.
    """

    # Extract the data from the astropy table using the keys provided and stack them into a 2D array, (number_of_parameters, number_of_samples)
    data_array = np.vstack([data[key].values for key in data_keys]).T

    # Extract the errors from the astropy table using the keys provided
    # We have no informatiom on the correlation between the errors so assume they are diagonal/uncorrelated. 
    errors_array = np.vstack([data[err_key].values for err_key in data_err_keys]).T

    # Extract the number of samples and features from the data array
    # Alternatively, n_features = len(data_keys) and n_samples = len(data_array)
    n_samples, n_features = data_array.shape

    # Scale the data to have zero mean and unit variance - this will improve the convergence of the EM algorithm
    # Errors are scaled by the same factor to maintain the same relative uncertainty
    # Note this will require the gaussians means and covariances returned by XD to be scaled back to the original units for interpretation

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_array)
    errors_scaled = errors_array / scaler.scale_

    # Calculate the extreme values of the data for initialisation randomisation
    extreme_data_values = (np.max(data_array, axis=0), np.min(data_array, axis=0))

    # Initialise the results dictionary for dynamic appending
    results = {
        # Number of gaussians fitted
        "n_components": [],
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


    # Iterate over range of number of gaussian components
    for n_components in range(component_range[0], component_range[1] + 1):
        best_log_likelihood = -np.inf
        best_weights, best_means, best_covariances = None, None, None
        best_means_original, best_covariances_original = None, None

        # Repeats and custom initializations
        for _ in range(n_repeats):
            for _ in range(n_init):

                initamp = np.random.dirichlet(np.ones(n_components))
                initmean = np.random.uniform(low=feature_min, high=feature_max, size=(n_components, n_features))
                initcovar = np.array([np.identity(n_features) for _ in range(n_components)])

                try:
                    XD_avg_LL = extreme_deconvolution(
                        data_scaled, errors_scaled, initamp, initmean, initcovar, maxiter=max_iterations
                    )
                    total_LL = XD_avg_LL * n_samples

                    if total_LL > best_log_likelihood:
                        best_log_likelihood = total_LL
                        best_weights, best_means, best_covariances = initamp.copy(), initmean.copy(), initcovar.copy()

                        # Unscale the means and covariances to return them to their original/meaningful units
                        means_original = scaler.inverse_transform(best_means)
                        covariances_original = np.array([
                            np.dot(np.dot(np.diag(scaler.scale_), cov), np.diag(scaler.scale_))
                            for cov in best_covariances
                        ])
                        best_means_original = means_original
                        best_covariances_original = covariances_original

                except Exception as e:
                    print(f"XD failed for {n_components} components: {e}")

        num_params = n_components * (1 + n_features + n_features * (n_features + 1) // 2) - 1
        bic = BICScore(best_log_likelihood, num_params, n_samples)
        aic = AICScore(best_log_likelihood, num_params)

        # Store best results
        results["n_components"].append(n_components)
        results["log_likelihood"].append(best_log_likelihood)
        results["BIC"].append(bic)
        results["AIC"].append(aic)
        results["weights"].append(best_weights)
        results["means"].append(best_means)
        results["covariances"].append(best_covariances)
        results["means_original"].append(best_means_original)
        results["covariances_original"].append(best_covariances_original)

        print(f"Completed {n_components} components: Log-Likelihood={best_log_likelihood:.2f}, BIC={bic:.2f}, AIC={aic:.2f}")

    return results
