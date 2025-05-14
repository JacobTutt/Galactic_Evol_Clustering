import numpy as np
from tabulate import tabulate
import pandas as pd
from collections import defaultdict, Counter

from astropy.table import Table
from astropy.io import fits
from tqdm.notebook import tqdm
import pickle as pkl

from extreme_deconvolution import extreme_deconvolution
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2, norm, gaussian_kde, multivariate_normal

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
from matplotlib.ticker import FuncFormatter
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


from typing import List, Tuple, Optional, Union

class XDPipeline:
    """
    A pipeline for performing Extreme Deconvolution (XD) using a Gaussian Mixture Model (GMM).

    Aims at analysing and fitting multi-dimensional stellar datasets.

    The pipeline follows these key steps:
    
    1. **Initialisation** (`__init__`):

       - Takes in stellar data as an Astropy Table, NumPy recarray, or Pandas DataFrame.
       - Extracts relevant features defined by `data_keys` and their errors `data_err_keys`.

    2. **Extreme Deconvolution (XD)** (`run_XD`):

       - Normalises the dataset for efficient convergence. (Optional: scaling)
       - Runs XD over a specified range of Gaussian components.
       - Iterates through multiple random initialisations to ensure robust fitting.
       - Uses BIC and AIC scores to evaluate model performance.
       - Optionally saves results to a file for later analysis.

    3. **Model Comparison & Selection** (`compare_XD`):

       - Compares different XD runs using Bayesian Information Criterion (BIC) or Akaike Information Criterion (AIC).
       - Identifies the best-fit model based on BIC or AIC scores.
       - Supports filtering results by a specific number of components or repeat cycle.
       - Generates a summary of failed runs and visualises scores across different components.

    4. **Star Assignment to Gaussian Components** (`assigment_XD`):

       - Computes each star's probability of belonging to each Gaussian component (responsibilities).
       - Assigns each star to the most probable Gaussian component.
       - Accounts for measurement uncertainties by modifying covariance matrices (error-aware).

    5. **Results Table** (`table_results_XD`):

       - Constructs a summary table showing the properties of each Gaussian component.
       - Displays and outputs estimated weights, assigned star counts, and mean ± standard deviation for each parameter.

    6. **Plotting Results** (`plot_XD`):

       - Generates a 2D scatter plot with color-coded Gaussian assignments.
       - Overlays Gaussian components as confidence ellipses (scaled by a z-score for different confidence levels).
       - Displays marginal histograms and Kernel Density Estimation (KDE) plots for feature distributions.
       - Includes a bar chart representing the relative weight of each Gaussian component.

    Parameters
    ----------
    star_data : Table, np.recarray, pd.DataFrame
        Input dataset containing stellar observations and features.
    data_keys : List[str]
        List of feature name keys to be used for fitting the Gaussian Mixture Model.
    data_err_keys : List[str]
        List of measurement uncertainty keys corresponding to feature keys.
    scaling : bool
        If True, standardise the features to have zero mean and unit variance.
        If False, no scaling is applied globally, but energy-related columns ('E_50' or 'Energy') are divided by 1e5 for consistency.

    Attributes
    ----------
    star_data : Table
        The input dataset - converted to an Astropy Table upon import.
    feature_data : np.ndarray
        Extracted feature values for model fitting.
    errors_data : np.ndarray
        Measurement errors associated with each feature.
    n_samples : int
        Number of stars (data points) in the dataset.
    n_features : int
        Number of features used in the XD model.
    results_XD : dict or None
        Dictionary storing the results of XD fitting across different initialisations.
    best_params : dict
        Best-performing XD model parameters based on the chosen optimisation metric.
    filtered_best_params : dict or None
        Best-performing parameters after applying user-defined filters (e.g., fixed number of components).
    assignment_metric : str or None
        Specifies whether the assignments were based on the "best" or "best filtered" model.

    Notes
    -----
    - The pipeline optionally scales input data using `StandardScaler` before fitting, ensuring numerical stability.
    - Measurement uncertainties are incorporated into the covariance matrices during model fitting.
    - The pipeline supports saving/loading XD results for reproducibility.
    """
    def __init__(self, star_data: Union[Table, np.recarray, pd.DataFrame], data_keys: List[str], data_err_keys: List[str], scaling: bool = True):
        """
        Initialise the XDPipeline with stellar data and keys of intrest for the Gaussian Mixture Model (GMM) - Extreme Deconvolution (XD) process, defining the parameter space of interest. 

        Parameters:
            star_data (Table, np.recarray, pd.DataFrame): Dataset containing stellar information, which can be an Astropy Table, NumPy recarray, or Pandas DataFrame.
            data_keys (List[str]): List of column names representing features used in the GMM fitting.
            data_err_keys (List[str]): List of column names representing measurement uncertainties, corresponding to `data_keys`.
            scaling (bool): If True, standardise the features to have zero mean and unit variance. If False, no scaling is applied globally, but energy-related columns ('E_50' or 'Energy') are divided by 1e5 for consistency.

        Raises:
            TypeError: If the input dataset is not a supported type.
            ValueError: If `data_keys` and `data_err_keys` have mismatched lengths or contain missing columns.
        """

        # Convert all inputs to an Astropy Table for consistency
        if isinstance(star_data, np.recarray):
            star_data = Table(star_data)  # Convert recarray to Table
        elif isinstance(star_data, pd.DataFrame):
            star_data = Table.from_pandas(star_data)  # Convert DataFrame to Table
        elif not isinstance(star_data, Table):
            raise TypeError("Unsupported data type. Must be an Astropy Table, NumPy recarray, or Pandas DataFrame.")
        
        # Extract column names
        colnames = star_data.colnames
        
        # Check that the number of data keys and error keys match
        if len(data_keys) != len(data_err_keys):
            raise ValueError("Size of data and error keys must match")
        
        # Check that the data keys and error keys are present in the data table
        # Print the missing keys if they are not present
        missing_keys = [key for key in data_keys + data_err_keys if key not in colnames]
        if missing_keys:
            raise ValueError(f"Keys {missing_keys} not found in data table")
        

        # Store the data, data keys, and data error keys
        self.star_data = star_data
        self.data_keys = data_keys
        self.data_err_keys = data_err_keys

        # Choose whether to scale the data or not - if not scaling the data then the energy values are divided by 1e5 
        self.scaling = scaling
        if not self.scaling:
            for key in self.data_keys:
                if key == 'E_50':
                    self.star_data['E_50'] /= 1e5
                    if 'E_err' in self.star_data.colnames:
                        self.star_data['E_err'] /= 1e5
                elif key == 'Energy':
                    self.star_data['Energy'] /= 1e5
                    if 'e_Energy' in self.star_data.colnames:
                        self.star_data['e_Energy'] /= 1e5

        # Extract the data and their errors from the data using the keys provided and stack them into a 2D array, (number_of_parameters, number_of_samples)
        self.feature_data = np.vstack([np.asarray(self.star_data[key]) for key in self.data_keys]).T

        # Extract the errors from the astropy table using the keys provided
        # We have no informatiom on the correlation between the errors so assume they are diagonal/uncorrelated. 
        self.errors_data = np.vstack([np.asarray(self.star_data[err_key]) for err_key in self.data_err_keys]).T

        # Extract the number of stars/ samples and features from the data array
        self.n_samples, self.n_features = self.feature_data.shape

        # Initialise the following attributes for future use in the XD pipeline
        # The range of Gaussian components to run for
        self.gauss_component_range= None
        # The maximum number of EM iterations for each GMM opitimisation
        self.max_iterations = None
        # The number of complete repetitions of the initialisations
        self.n_repeats = None
        # The number of random initialisations per component count
        self.n_init = None
        # The XD results for all runs
        self.results_XD = None
        # The path to save the XD results for all runs
        self.save_path_XD = None

        # The best (BIC/AIC) score and the parameters in which it was achieved
        self.best_params = {}
        # The best (BIC/AIC) score for a filtered data set and the parameters in which it was achieved
        self.filtered_best_params = None
        # Stores which of the above was used for the assignment
        self.assignment_metric = None


    def _BICScore(self, log_likelihood: float, num_params: int, num_data_points: int) -> float:
        """
        Compute the Bayesian Information Criterion (BIC) score.

        Parameters
        ----------
        log_likelihood : float
            Log-likelihood of the model.
        num_params : int
            Number of free parameters in the model.

        Returns
        -------
        float
            Computed BIC score.
        """
        return -2 * log_likelihood + num_params * np.log(num_data_points)
    
    def _AICScore(self, log_likelihood: float, num_params: int) -> float:
        """
        Compute the Akaike Information Criterion (AIC) score.

        Parameters
        ----------
        log_likelihood : float
            Log-likelihood of the model.
        num_params : int
            Number of free parameters in the model.

        Returns
        -------
        float
            Computed AIC score.
        """
        return - 2 * log_likelihood + 2 * num_params
    
    def run_XD(self, gauss_component_range: Tuple[int, int] = (1, 10), max_iterations: int = int(1e9), n_repeats: int = 3, n_init: int = 100, save_path: Optional[str] = None) -> None:
        """
        Initialise the XDPipeline with stellar data and define the parameter space for 
        Extreme Deconvolution (XD) using a specified set of features and their uncertainties.

        This constructor supports optional scaling of features using standardisation. If 
        `scaling=False`, the features are used in their original units; however, energy-related 
        parameters ('E_50' or 'Energy') are manually scaled by 1e5 for consistency.

        Parameters
        ----------
        star_data : Table, np.recarray, or pd.DataFrame
            Input dataset containing stellar properties. Accepted formats are Astropy Table,
            NumPy recarray, or Pandas DataFrame.

        data_keys : List[str]
            Column names representing the features to be used in the GMM-XD analysis.

        data_err_keys : List[str]
            Column names representing the corresponding measurement uncertainties for each
            feature in `data_keys`.

        scaling : bool, optional
            Whether to apply standard scaling (zero mean, unit variance) to the input features.
            If False, no scaling is applied globally, but energy-related columns ('E_50' or 'Energy')
            are manually divided by 1e5. Default is True.

        Raises
        ------
        TypeError
            If the input dataset is not a supported type.

        ValueError
            If `data_keys` and `data_err_keys` differ in length or contain missing columns 
            not present in the input dataset.
        """

        # Check that the component range is valid
        if not isinstance(gauss_component_range, tuple) or len(gauss_component_range) != 2:
            raise ValueError("Gaussian component range must be a tuple of form (min, max)")
        
        if gauss_component_range[0] > gauss_component_range[1]:
            raise ValueError("Invalid gaussian component range") 
        
        # Check that the number of repeats and initialisations are valid
        if not isinstance(n_repeats, int) or n_repeats <= 0:
            raise TypeError("n_repeats must be a positive integer")
        if not isinstance(n_init, int) or n_init <= 0:
            raise TypeError("n_init must be a positive integer")
        if not isinstance(max_iterations, int) or max_iterations <= 0:
            raise TypeError("max_iterations must be a positive integer")
        
        # Save attributes to the pipeline object
        self.gauss_component_range = gauss_component_range
        self.max_iterations = max_iterations
        self.n_repeats = n_repeats
        self.n_init = n_init
        self.save_path_XD = save_path


        # OPTIONAL - Scale the data to have zero mean and unit variance - this will improve the convergence of the EM algorithm
        # Errors are scaled by the same factor to maintain the same relative uncertainty
        # Note this will require the gaussians means and covariances returned by XD to be scaled back to the original units for interpretation before saving
        if self.scaling:
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(self.feature_data)
            errors_scaled = self.errors_data / scaler.scale_
        else:
            data_scaled = self.feature_data
            errors_scaled = self.errors_data

        # Calculate the extreme values of the data for initialisation randomisation
        # extreme_data_values = (np.max(self.feature_data, axis=0), np.min(self.feature_data, axis=0))
        # THIS MAY IN PART BE WHERE THE ERROR IS COMING FROM - THEY HAVE BEEN NORMALISED SO THE INITIALSIATION VALUES SHOULD BE NORMALISED TOO
        scaled_min = np.min(data_scaled, axis=0)
        scaled_max = np.max(data_scaled, axis=0)

        # Initialise the results dictionary for dynamic appending
        self.results_XD = {
            # Repeat Number
            "repeat no.": [],
            # Intialisation Number
            "init no.": [],
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

        # Iterate for a test range of number of gaussians
        for n_gauss in tqdm(range(self.gauss_component_range[0], self.gauss_component_range[1] + 1), desc="Number of Gaussian Components"):
            # Overall repeats 
            for n in tqdm(range(self.n_repeats), desc="Repeat Cycles", leave=False):
                # Random initialisations of input parameters
                for i in tqdm(range(self.n_init), desc="Initialisations", leave=False):

                    # Even initialisation of weights
                    #init_weights = np.ones(n_gauss) / n_gauss
                    init_weights = np.random.dirichlet(np.ones(n_gauss))

                    # Data has been scaled so means should be sampled from the scaled values.
                    
                    # Randomly Initialise means from a normal distribution with mean = 0 and std = 1
                    # init_mean = np.random.normal(loc=0, scale=1, size=(n_gauss, self.n_features))
                    
                    # Randomly initialise means from the scaled values range
                    init_mean = np.random.uniform(low=scaled_min, high=scaled_max, size=(n_gauss, self.n_features))

                    if self.scaling:
                        # Covariances initialised as identity matrices
                        init_covar = np.array([np.identity(self.n_features) for _ in range(n_gauss)])

                    else:
                    # Initialise the covariance matricies so each diagonal element has the variance of the data in that dimension
                        init_covar = np.array([np.diag(np.var(self.feature_data, axis=0)) for _ in range(n_gauss)])


                    # Run XD
                    try:
                        XD_avg_LL = extreme_deconvolution(
                            data_scaled, errors_scaled, init_weights, init_mean, init_covar, maxiter=max_iterations)
                        # Calculate the total log likelihood
                        total_LL = XD_avg_LL * self.n_samples

                        # Calculate the bic and aic scores
                        num_params = n_gauss * (1 + self.n_features + self.n_features * (self.n_features + 1) // 2) - 1
                        bic, aic = self._BICScore(total_LL, num_params, self.n_samples), self._AICScore(total_LL, num_params)

                        # Copy the updated weights, means and covariances
                        post_XD_weights, post_XD_means, post_XD_cov = init_weights.copy(), init_mean.copy(), init_covar.copy()

                        # If the scaling is on then the means and covariances are scaled and need to br reverting
                        # Unscale the means and covariances to return them to their original/meaningful units
                        if self.scaling:
                            post_scaling_means = scaler.inverse_transform(post_XD_means)
                            post_scaling_cov = np.array([
                                np.dot(np.dot(np.diag(scaler.scale_), cov), np.diag(scaler.scale_))
                                for cov in post_XD_cov
                            ])
                        else:
                            post_scaling_means = post_XD_means
                            post_scaling_cov = post_XD_cov

                        # Store the results
                        self.results_XD["repeat no."].append(n)
                        self.results_XD["init no."].append(i)
                        self.results_XD["n_gauss"].append(n_gauss)
                        self.results_XD["log_likelihood"].append(total_LL)
                        self.results_XD["BIC"].append(bic)
                        self.results_XD["AIC"].append(aic)
                        self.results_XD["weights"].append(post_XD_weights)
                        self.results_XD["means"].append(post_scaling_means)
                        self.results_XD["covariances"].append(post_scaling_cov)


                    except Exception as e:
                        print(f"XD failed for {n_gauss} components, on repeat: {n}, iteration: {i}: {e}")

                        # Store the results
                        self.results_XD["repeat no."].append(n)
                        self.results_XD["init no."].append(i)
                        self.results_XD["n_gauss"].append(n_gauss)
                        self.results_XD["log_likelihood"].append(None)
                        self.results_XD["BIC"].append(None)
                        self.results_XD["AIC"].append(None)
                        self.results_XD["weights"].append(None)
                        self.results_XD["means"].append(None)
                        self.results_XD["covariances"].append(None)

            # Save the results if a path is provided
            # This is redone for each component count to ensure that the results are saved in case of a crash
            if self.save_path_XD:
                try:
                    with open(self.save_path_XD, "wb") as f:
                        pkl.dump(self.results_XD, f)
                    print(f"Results saved successfully at {self.save_path_XD}")
                except Exception as e:
                    print(f"Failed to save results: {e}")

        return None
    
    def compare_XD(self, opt_metric = 'BIC', n_gauss_filter: Optional[int] = None, repeat_no_filter: Optional[int] = None, save_path: Optional[str] = None, zoom_in: Optional[List[int]] = None, display_full: bool = True) -> None:
        """
        Analyse Extreme Deconvolution (XD) results using BIC or AIC.
        This method identifies the best-fit model, summarizes failed runs, and visualizes score distributions.
        If no filters are applied, the analysis is performed on all results. Otherwise, it is performed on filtered results.

        Parameters
        ----------
        opt_metric : str
            Optimization metric ('BIC' or 'AIC').
        n_gauss_filter : Optional[int]
            Specific number of Gaussian components to filter results by.
        repeat_no_filter : Optional[int]
            Specific repeat cycle to filter results by.
        save_path : Optional[str]
            Path to load XD results if not already stored in the class.

        Raises
        ------
        ValueError
            If results are not available in the class and no valid `save_path` is given.
        ValueError
            If `opt_metric` is not 'BIC' or 'AIC'.
        ValueError
            If filter values (`n_gauss_filter`, `repeat_no_filter`) are outside valid ranges.
        """

        
        if self.results_XD is None:
            if save_path is None:
                raise ValueError("No XD results found in class and no path given to load from. Please run XD first or provide valid path to load from.")

            else:
                try:
                    self.save_path_XD = save_path
                    with open(self.save_path_XD, "rb") as f:
                        self.results_XD = pkl.load(f)
                except Exception as e:
                    print(f"Failed to save results: {e}")
                    raise ValueError("Failed to load results from the given path.")

            # Extract information from file data that will not be saved to the class if XD is not run in this instance

            self.gauss_component_range = (min(self.results_XD["n_gauss"]), max(self.results_XD["n_gauss"]))
            self.n_repeats = max(self.results_XD["repeat no."]) + 1
            self.n_init = max(self.results_XD["init no."]) + 1

        # Ensure input filters are valid for the XD Results that the class contains
        if n_gauss_filter is not None: 
            if n_gauss_filter < self.gauss_component_range[0] or n_gauss_filter > self.gauss_component_range[1]:
                raise ValueError(f"Invalid Filter: {n_gauss_filter} not in the range of Gaussian components: {self.gauss_component_range}")
        if repeat_no_filter is not None:
            if repeat_no_filter > self.n_repeats:
                raise ValueError(f"Invalid Filter: {repeat_no_filter} not in the range of repeat cycles: {self.n_repeats}")
        
        # Ensure valid optimisation metric is selected
        if opt_metric not in ['BIC', 'AIC']:
            raise ValueError("Invalid optimisation metric selected. Please select either 'BIC' or 'AIC'.")
        
        # Delete any columns in self.star_data that are already present from past analysis
        # Search for keys

        # Delete any existing probability assignments related entries to avoid conflicts
        colnames = self.star_data.colnames

        # Delete any existing probability assignments to avoid conflicts
        for key in colnames:
            if key.startswith('prob_gauss_') or key == 'max_gauss':
                del self.star_data[key]


        self.assignment_metric = None

        

        # This is preformed for all results ie all n_gauss, all repeats and all initialisations
        # Identifies the best BIC/AIC score - its index, correlated inputs and results
        self.best_params['metric'] = opt_metric
        self.best_params['score'] = min(b for b in self.results_XD[opt_metric] if b is not None)
        best_index = self.results_XD[opt_metric].index(self.best_params['score'])
        self.best_params['gauss_components'] = self.results_XD["n_gauss"][best_index]
        self.best_params['repeat'] = self.results_XD["repeat no."][best_index]
        self.best_params['init'] = self.results_XD["init no."][best_index]
        self.best_params['XD_weights'] = self.results_XD["weights"][best_index]
        self.best_params['XD_means'] = self.results_XD["means"][best_index]
        self.best_params['XD_covariances'] = self.results_XD["covariances"][best_index]

        # Prints summary of best performing Gaussian components
        if display_full:
            print(f" Best Overall {opt_metric} Score: {self.best_params['score']:.4f} occurred at:")
            print(f"   - Gaussian Components (n_gauss): {self.best_params['gauss_components']}")
            print(f"   - Repeat cycle (n): {self.best_params['repeat']}")
            print(f"   - Initialisation (i): {self.best_params['init']}")

        # This preforms it on the filtered data - ie a specific n_gauss and/ or repeat cycle
        if n_gauss_filter is not None or repeat_no_filter is not None:

            # Filter the dictionary and only keep relevent results
            mask = np.ones(len(self.results_XD["n_gauss"]), dtype=bool)
            if n_gauss_filter is not None:
                mask &= (np.array(self.results_XD["n_gauss"]) == n_gauss_filter)
            if repeat_no_filter is not None:
                mask &= (np.array(self.results_XD["repeat no."]) == repeat_no_filter)

            filtered_results = {}

            for key, values in self.results_XD.items():
                values_array = np.array(values, dtype=object)  # Ensure no shape enforcement
                filtered_results[key] = [values_array[i] for i in range(len(mask)) if mask[i]]
            # Filter the dictionary and only keep relevent results

            # Identifies the best BIC/AIC score
            self.filtered_best_params = {}
            self.filtered_best_params['filters'] = {"n_gauss": n_gauss_filter, "repeat": repeat_no_filter}
            self.filtered_best_params['metric'] = opt_metric 
            self.filtered_best_params['score'] = min(b for b in filtered_results[opt_metric] if b is not None)
            filted_index = filtered_results[opt_metric].index(self.filtered_best_params['score'])
            self.filtered_best_params['gauss_components'] = filtered_results["n_gauss"][filted_index]
            self.filtered_best_params['repeat'] = filtered_results["repeat no."][filted_index]
            self.filtered_best_params['init'] = filtered_results["init no."][filted_index]
            self.filtered_best_params['XD_weights'] = filtered_results["weights"][filted_index]
            self.filtered_best_params['XD_means'] = filtered_results["means"][filted_index]
            self.filtered_best_params['XD_covariances'] = filtered_results["covariances"][filted_index]
            
            # Prints summary of best performing Gaussian components
            # print the filters that can been applied
            print(f" The following filters were applied: {self.filtered_best_params['filters']}")
            # summary of results they returned
            print(f" Best {opt_metric} Score from filtered inputs: {self.filtered_best_params['score']:.4f} occurred at:")
            print(f"   - Gaussian Components (n_gauss): {self.filtered_best_params['gauss_components']}")
            print(f"   - Repeat cycle (n): {self.filtered_best_params['repeat']}")
            print(f"   - Initialisation (i): {self.filtered_best_params['init']}")

        else: 
            self.filtered_best_params = None
            print("No analysis was performed on filtered data as no filters inputed")


        # Analysis on overall code - not filtered
        n_gauss_list = np.array([n for n in range(self.gauss_component_range[0], self.gauss_component_range[1] + 1)])

        # Creates a DataFrame summarising the number of failed XD runs per Gaussian component
        n_XD_failed = [
            sum([b is None for c, b in zip(self.results_XD["n_gauss"], self.results_XD["BIC"]) if c == n_gauss])
            for n_gauss in n_gauss_list
        ]
        n_runs_gauss = np.array([self.n_repeats * self.n_init for _ in n_gauss_list])
        n_failed_XD = pd.DataFrame({
            "No. Gaussians": n_gauss_list,
            "No. Failed XD runs": n_XD_failed,
            "Total No. Runs": n_runs_gauss
        })

        if display_full:
            # Prints a formatted table of the number of failed XD runs
            print("Table of Number of Gaussians vs Number of Failed XD Runs")
            print(tabulate(n_failed_XD, headers='keys', tablefmt='psql'))


        # Reshapes BIC and AIC scores to 3D arrays: (n_gauss_components, n_repeats, n_init)
        BIC_scores = np.array([b if b is not None else np.nan for b in self.results_XD['BIC']]).reshape((len(n_gauss_list), self.n_repeats, self.n_init))
        AIC_scores = np.array([a if a is not None else np.nan for a in self.results_XD['AIC']]).reshape((len(n_gauss_list), self.n_repeats, self.n_init))

        # Compute means and standard deviations for BIC and AIC across initialisations
        BIC_means, BIC_stds = np.nanmean(BIC_scores, axis=2), np.nanstd(BIC_scores, axis=2)
        AIC_means, AIC_stds = np.nanmean(AIC_scores, axis=2), np.nanstd(AIC_scores, axis=2)

        # Format mean ± stddev for BIC and AIC into DataFrames
        BIC_means_stds_df = pd.DataFrame(
            np.array([[f"{mean:.5f} \\pm {std:.5f}" for mean, std in zip(mean_row, std_row)]
                    for mean_row, std_row in zip(BIC_means, BIC_stds)]),
            columns=[f"Repeat {i + 1}" for i in range(self.n_repeats)],
            index=n_gauss_list
        )
        BIC_means_stds_df.insert(0, "No. Gaussians", n_gauss_list)

        AIC_means_stds_df = pd.DataFrame(
            np.array([[f"{mean:.5f} \\pm {std:.5f}" for mean, std in zip(mean_row, std_row)]
                    for mean_row, std_row in zip(AIC_means, AIC_stds)]),
            columns=[f"Repeat {i + 1}" for i in range(self.n_repeats)],
            index=n_gauss_list
        )
        AIC_means_stds_df.insert(0, "No. Gaussians", n_gauss_list)

        if display_full:
            # Print BIC and AIC summary tables
            print("Table of BIC Means and Stds")
            print(tabulate(BIC_means_stds_df, headers='keys', tablefmt='psql'))

            print("Table of AIC Means and Stds")
            print(tabulate(AIC_means_stds_df, headers='keys', tablefmt='psql'))

        # Calculate min, max, and median BIC and AIC scores across repeats and initialisations for each Gaussian component count ie from n_init * n_repeats values - (n_gauss_components)
        BIC_min, BIC_max, BIC_median = BIC_scores.min(axis=(1, 2)), BIC_scores.max(axis=(1, 2)), np.nanmedian(BIC_scores, axis=(1, 2))
        AIC_min, AIC_max, AIC_median = AIC_scores.min(axis=(1, 2)), AIC_scores.max(axis=(1, 2)), np.nanmedian(AIC_scores, axis=(1, 2))

        if display_full:
            # Plot combined BIC & AIC
            fig, ax = plt.subplots(figsize=(6, 8))
            # BIC - Blue
            ax.plot(n_gauss_list, BIC_min, 'b-', label="BIC")
            ax.plot(n_gauss_list, BIC_max, 'b--') #, label="BIC - Highest")
            ax.plot(n_gauss_list, BIC_median, 'b:') #, label="BIC - Median")

            # AIC - Red
            ax.plot(n_gauss_list, AIC_min, 'r-', label="AIC")
            ax.plot(n_gauss_list, AIC_max, 'r--') #, label="AIC - Highest")
            ax.plot(n_gauss_list, AIC_median, 'r:') #, label="AIC - Median")

            ax.set_xlabel("Number of Gaussian Components", fontsize=12)
            ax.set_ylabel("Score", fontsize=12)
            ax.legend(loc='upper left')

            ax.grid(True)


            # Optional Zoom in Option for the top right
            if zoom_in:
                # Create inset axes for zoomed in view
                if opt_metric == 'BIC':
                    zoom_in_min = BIC_min
                    zoom_in_max = BIC_max
                    zoom_in_median = BIC_median
                    axins = inset_axes(ax, width="40%", height="30%", loc='upper right')
                    # BIC - Blue
                    axins.plot(n_gauss_list, zoom_in_min, 'b-')
                    axins.plot(n_gauss_list, zoom_in_max, 'b--') 
                    axins.plot(n_gauss_list, zoom_in_median, 'b:')

                if opt_metric == 'AIC':
                    zoom_in_min = AIC_min
                    zoom_in_max = AIC_max
                    zoom_in_median = AIC_median
                    axins = inset_axes(ax, width="40%", height="30%", loc='upper right')
                    # AIC - Red
                    axins.plot(n_gauss_list, zoom_in_min, 'r-')
                    axins.plot(n_gauss_list, zoom_in_max, 'r--')
                    axins.plot(n_gauss_list, zoom_in_median, 'r:')


                # Set x-axis limits from zoom_in list
                axins.set_xlim(min(zoom_in) - 0.5 , max(zoom_in) + 0.5)

                # Set y-axis limits based on BIC values in zoomed range
                mask = [(x in zoom_in) for x in n_gauss_list]
                if any(mask):
                    if opt_metric == 'BIC':
                        zoom_bic = [b for b, m in zip(BIC_min, mask) if m]
                        axins.set_ylim(min(zoom_bic) * 0.99, max(zoom_bic) * 1.02)
                    
                    if opt_metric == 'AIC':
                        zoom_aic = [a for a, m in zip(AIC_min, mask) if m]
                        axins.set_ylim(min(zoom_aic) * 0.99, max(zoom_aic) * 1.02)

                mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

            plt.tight_layout()
            plt.show()

        return None
    
    def assigment_XD(self, assignment_metric='best'):
        """
        Assign stars to Gaussian components based on the best-fit XD model.
        Computes the responsibility of each gaussians for each star and assigns it accordingly.

        This method performs assignment in scaled feature space using StandardScaler to reproduce
        the scaling used during XD fitting. Covariance matrices are adjusted to include measurement
        errors, and ill-conditioned matrices are regularized to ensure numerical stability.

        Parameters
        ----------
        assignment_metric : str
            Selection criteria for the best-fit model ('best' or 'best filtered').

        Raises
        ------
        ValueError
            If no XD results are available.
        ValueError
            If an invalid `assignment_metric` is specified.

        Returns
        -------
        None
            Updates `star_data` in place to include probability assignments:
                - `prob_gauss_{i}`: Probability of belonging to the i-th Gaussian component.
                - `max_gauss`: Index of the component with the highest probability (1-based index).
        """

        # Ensure that all previous analysis has been preformed. 
        if self.results_XD is None:
            raise ValueError("No XD results found in class. Please run XD and analysis first or provide valid path to load from in the analysis method.")
        
        if assignment_metric == 'best filtered' and self.filtered_best_params is None:
            raise ValueError("No filtered best parameters found. Please run filtered analysis first.")
        
        elif assignment_metric == 'best' and self.best_params == {}:
            raise ValueError("No best parameters found. Please run analysis first.")
        
        elif assignment_metric not in ['best', 'best filtered']:
            raise ValueError("Invalid assignment metric selected. Please select either 'best' or 'best filtered'.")
        
        # Delete any existing probability assignments related entries to avoid conflicts
        colnames = self.star_data.colnames

        # Delete any existing probability assignments to avoid conflicts
        for key in colnames:
            if key.startswith('prob_gauss_') or key == 'max_gauss':
                del self.star_data[key]

        # Allow storage of which assignment metric was used for future reference
        self.assignment_metric = assignment_metric
        
        # Extract the results of relevent analysis locally
        if assignment_metric == 'best filtered':
            assignment_params = self.filtered_best_params
        if assignment_metric == 'best':
            assignment_params = self.best_params

        # Print a summary of what this is preforming
        print(f"Assigning stars to Gaussian components based on the {assignment_metric} XD model.")
        print(f"This has been optimised for the {assignment_params['metric']} score and returned the results:")
        print(f" Best {assignment_params['metric']} Score: {assignment_params['score']:.4f} occurred at:")
        print(f"   - Gaussian Components (n_gauss): {assignment_params['gauss_components']}")
        print(f"   - Repeat cycle (n): {assignment_params['repeat']}")
        print(f"   - Initialisation (i): {assignment_params['init']}")

        # Error-Aware Explanation:
        # Cannot evaluate the probability density at position in parameter space deirectly
        # XD accounts for measurement errors by modifying the covariance matrices of the Gaussian components. T
        # Allows total uncertainty reflects both the model and the measurement noise.
        # Done by adding the measurement error covariance X to the intrinsic Gaussian variance V, so:
        #     T  = V + Xerr

        if self.scaling:
            # StandardScaler is used to reproduce scaling applied during XD fitting
            # Otherwise very ill-conditioned covariance matrices occur ie dominance from the energy term
            scaler = StandardScaler()
            scaler.fit(self.feature_data)

            scaling_factors = scaler.scale_
            D_inv = np.diag(1.0 / scaling_factors)

            # Rescale means and covariances to scaled space
            means_scaled = scaler.transform(assignment_params['XD_means'])
            covs_scaled = np.array([D_inv @ cov @ D_inv for cov in assignment_params['XD_covariances']])

            feature_data_scaled = scaler.transform(self.feature_data)
            errors_data_scaled = self.errors_data / scaling_factors

        else:
            # No scaling applied: use original data and model parameters directly
            means_scaled = np.array(assignment_params['XD_means'], dtype=np.float64)
            covs_scaled = np.array([
                np.array(cov, dtype=np.float64)
                for cov in assignment_params['XD_covariances']
            ])
            feature_data_scaled = self.feature_data
            errors_data_scaled = self.errors_data

        # Initialises columns for probabilities and assignments
        for i in range(assignment_params['gauss_components']):
            self.star_data[f'prob_gauss_{i+1}'] = np.zeros(len(self.star_data))
        self.star_data['max_gauss'] = np.zeros(len(self.star_data), dtype=int)

        # For each star calculate the probability of it belonging to each gaussian
        for star_index, star in enumerate(feature_data_scaled):
            probabilities = []

            # Extract the measurement error covariance for the current star and convert it to a diagonal matrix
            star_errors = np.diag(errors_data_scaled[star_index])

            # Cycle through each of the gaussians from the n_gauss
            for j in range(assignment_params['gauss_components']):
                # Mean covariance and weight for the jth gaussian
                mean_j = means_scaled[j]
                # Error-Aware Covariance Adjustment
                cov_j = covs_scaled[j] + star_errors
                weight_j = assignment_params['XD_weights'][j]

                # Calculate the probability of the data point given the gaussian
                try:
                    # Add regularisation if condition number is too high
                    if np.linalg.cond(cov_j) > 1e8:
                        cov_j += np.eye(cov_j.shape[0]) * 1e-2

                    prob = weight_j * multivariate_normal.pdf(star, mean=mean_j, cov=cov_j)

                # Handle singular covariance matrices
                except np.linalg.LinAlgError:
                    prob = 0 

                probabilities.append(prob)
                self.star_data[f'prob_gauss_{j+1}'][star_index] = prob

            # Assign the star to the gaussian with the highest probability
            self.star_data['max_gauss'][star_index] = np.argmax(probabilities) + 1

        return None
    
    def table_results_XD(self, component_name_dict: dict = None, combine: list = None, labels_combined: list = None) -> pd.DataFrame:
        """
        Generate a summary table of the Extreme Deconvolution (XD) results showing the mean and error values of each Gaussian in high-dimensional space.

        For each Gaussian the table includes:
        
        - Component Name (indexed numerically or custom if a mapping is provided)
        - XD assigned Weight (%)
        - Count of assigned stars
        - Count as a percentage of the total assigned stars
        - Mean values and standard deviations for each feature parameter

        Parameters
        ----------
        component_name_dict : dict, optional
            A dictionary mapping component indices (0-based) to custom names.
            The table will be ordered according to the order of keys in this dictionary if provided.
        combine : list of list of int, optional
            List of lists, where each inner list contains indices of components to be combined.
        labels_combined : list of str, optional
            Labels for the combined components. Must match the number of entries in `combine`.

        Returns
        -------
        pd.DataFrame
            A formatted summary of the Gaussian components fitted by XD.
        """


        # Ensure that the analysis has been run before generating the table
        if self.assignment_metric is None:
            raise ValueError("No assignment metric found. Please run assignment method first, which requires XD and analysis/comparison to be run first.")

        # Extract the relevant parameters depending on the assignment metric used during assignment_XD
        if self.assignment_metric == 'best':
            assignment_params = self.best_params 
        elif self.assignment_metric == 'best filtered':
            assignment_params = self.filtered_best_params

        # Extract Gaussian mixture parameters
        means = assignment_params['XD_means']
        covariances = assignment_params['XD_covariances']
        weights = assignment_params['XD_weights']
        n_components = assignment_params['gauss_components']

        # Ensure the parameters are in the correct format
        weights = np.array(weights, dtype=np.float64)
        means = np.array(means, dtype=np.float64)
        covariances = np.array(covariances, dtype=np.float64)

        # Count how many stars are assigned to each Gaussian component
        component_counts = np.array([(self.star_data['max_gauss'] == i+1).sum() for i in range(n_components)])
        count_percent = np.round((component_counts / component_counts.sum()) * 100, 1)

        # Construct base data dictionary
        table_data = {
            "Component": [f"Component {i+1}" for i in range(n_components)],
            "Weight (%)": np.round(weights * 100, 1),
            "Count": component_counts,
            "Count (%)": count_percent
        }

        # Add feature parameter summaries
        for i, key in enumerate(self.data_keys):
            mean_values = means[:, i]
            std_values = np.sqrt(covariances[:, i, i])
            table_data[key] = [f"{mean:.2f} ± {std:.2f}" for mean, std in zip(mean_values, std_values)]

        # Convert to DataFrame
        results_df = pd.DataFrame(table_data)

        # Override component names and order if a mapping is provided
        if component_name_dict:
            name_mapping = {f"Component {i+1}": name for i, name in component_name_dict.items()}
            results_df["Component"] = results_df["Component"].map(name_mapping)

            custom_order = [name_mapping[f"Component {i+1}"] for i in component_name_dict]
            results_df["Component"] = pd.Categorical(results_df["Component"], categories=custom_order, ordered=True)
            results_df = results_df.sort_values("Component").reset_index(drop=True)
        else:
            results_df = results_df.sort_values(by="Weight (%)", ascending=False).reset_index(drop=True)

        # Rename Energy
        if self.scaling:
            energy_rename_map = {'Energy': 'Scaled Energy ($\\times 10^5$)', 'E_50': 'Scaled Energy ($\\times 10^5$)'}
            results_df.rename(columns={k: v for k, v in energy_rename_map.items() if k in results_df.columns}, inplace=True)
        else:
            energy_rename_map = {'Energy': 'Energy ($\\times 10^5$)', 'E_50': 'Energy ($\\times 10^5$)'}
            results_df.rename(columns={k: v for k, v in energy_rename_map.items() if k in results_df.columns}, inplace=True)

        # Print formatted table
        print("\nSummary of GMM Fit Result for GALAH-Gaia Sample")
        print(tabulate(results_df, headers="keys", tablefmt="grid"))

        # Compute and print combined component table if requested
        if combine and labels_combined:
            # Ensure valid input: both lists must be the same length
            if len(combine) != len(labels_combined):
                raise ValueError("combine and labels_combined must have the same length.")

            print("\nCombined Component Summary")
            combined_rows = []

            # Loop over each subset of indicies and its label
            for group_indices, label in zip(combine, labels_combined):
                # Extract the weights and counts of the selected components
                w = weights[group_indices]
                # Calculate the total weight and counts - a simple sum
                total_weight = w.sum()
                total_count = component_counts[group_indices].sum()

                # Calculate overall weight and count percentages
                weight_pct = round(total_weight * 100, 1)
                count_pct = round((total_count / component_counts.sum()) * 100, 1)

                # Constructing the rows for the table with global values
                combined_row = {
                    "Component": label,
                    "Weight (%)": weight_pct,
                    "Count": total_count,
                    "Count (%)": count_pct
                }

                # Loop over each feature in the data, ie chemical abundances
                for i, key in enumerate(self.data_keys):
                    # Means and variances of this dimension for the selected components
                    means_group = means[group_indices, i]
                    covs_group = covariances[group_indices, i, i]

                    # ------ Combine the Gaussians ------
                    # Weighted average of means (centroid of the combined component)
                    mean_comb = np.average(means_group, weights=w)

                    # Variance combination (follows the law of total variance (weighted))
                    # Total variance = average of (individual variances + squared distance from group mean)
                    # Mathematically:
                    #     Var_total = Σ wᵢ [σᵢ² + (μᵢ - μ_combined)²] / Σ wᵢ
                    var_comb = np.average(covs_group + (means_group - mean_comb)**2, weights=w)

                    # Standard deviation - simple square root of variance
                    std_comb = np.sqrt(var_comb)

                    # Formatted string for display in the table
                    combined_row[key] = f"{mean_comb:.2f} ± {std_comb:.2f}"

                # Add the row to the combined table
                combined_rows.append(combined_row)

            # Convert all rows to a DataFrame for display
            combined_df = pd.DataFrame(combined_rows)

            # Rename Energy
            if self.scaling:
                energy_rename_map = {'Energy': 'Scaled Energy ($\\times 10^5$)', 'E_50': 'Scaled Energy ($\\times 10^5$)'}
                combined_df.rename(columns={k: v for k, v in energy_rename_map.items() if k in combined_df.columns}, inplace=True)
            else:
                energy_rename_map = {'Energy': 'Energy ($\\times 10^5$)', 'E_50': 'Energy ($\\times 10^5$)'}
                combined_df.rename(columns={k: v for k, v in energy_rename_map.items() if k in combined_df.columns}, inplace=True)

            print(tabulate(combined_df, headers="keys", tablefmt="grid"))

    

    def plot_XD(self, x_key: str, y_key: str, z_score: float = 2.0,
                full_survey_file: Optional[str] = None,
                color_palette: Optional[list] = None,
                xlim: Optional[tuple] = None,
                ylim: Optional[tuple] = None) -> None:
        """
        Creates a 2D plot of the Extreme Deconvolution (XD) results, displaying:
        - Individual stars colored by their assigned Gaussian component
        - Gaussian mixture model (GMM) components as confidence ellipses
        - Marginal histograms and KDE distributions for each axis
        - A bar chart representing the relative weight of each Gaussian component
        - Optional 2D histogram of full survey sample as grayscale background

        The confidence ellipses are scaled according to a given z-score, providing 
        a visual representation of the spread of each Gaussian component.

        Parameters
        ----------
        x_key : str
            The column name corresponding to the x-axis variable.
        y_key : str
            The column name corresponding to the y-axis variable.
        z_score : float, optional
            The z-score defining the confidence interval for the Gaussian ellipses.
            Defaults to 2, corresponding to a 95% confidence interval.
        full_survey_file : str, optional
            Path to FITS file of the full survey sample for reference background.
        color_palette : list, optional
            List of colors to use for each Gaussian component.
        xlim : tuple, optional
            Tuple (min, max) to manually set x-axis limits on the main plot.
        ylim : tuple, optional
            Tuple (min, max) to manually set y-axis limits on the main plot.

        Raises
        ------
        ValueError
            If the XD analysis has not been performed before plotting.
            If the provided x_key or y_key is not found in the dataset.
        """

        # Ensures analysis has been run before plotting
        if self.assignment_metric is None:
            raise ValueError("No assignment metric found. Please run assignment method first.")

        # Ensures that the keys provided are valid
        if x_key not in self.data_keys or y_key not in self.data_keys:
            raise ValueError("Invalid keys. Please provide valid keys from the data table (i.e., exist in data_keys).")

        # Extracts the relevant parameters depending on the assignment metric used during assignment_XD
        if self.assignment_metric == 'best':
            assignment_params = self.best_params 
        elif self.assignment_metric == 'best filtered':
            assignment_params = self.filtered_best_params

        # Extracts GMM parameters from the relevant best-fit analysis
        means = assignment_params['XD_means']
        covariances = assignment_params['XD_covariances']
        weights = assignment_params['XD_weights']
        n_components = assignment_params['gauss_components']

        # Ensures the parameters are in the correct format
        means = np.array(means, dtype=np.float64)
        covariances = np.array(covariances, dtype=np.float64)
        weights = np.array(weights, dtype=np.float64)


        # Retrieves relevant keys index positions
        x_index = self.data_keys.index(x_key)
        y_index = self.data_keys.index(y_key)

        # Extracts the individual star data
        x_data = self.feature_data[:, x_index]
        y_data = self.feature_data[:, y_index]
        assignments = self.star_data['max_gauss']

        # Set axis labels (custom formatting) - Use scaled or unscaled labels based on the scaling flag
        if self.scaling:
            axis_label_dict = {
                'fe_h': r'[Fe/H]', 'E_50': r'Energy', 'Energy': r'Energy',
                'alpha_m': r'[$\alpha$/Fe]', 'alpha_fe': r'[$\alpha$/Fe]', 'ce_fe': r'[Ce/Fe]',
                'al_fe': r'[Al/Fe]', 'Al_fe': r'[Al/Fe]', 'mg_mn': r'[Mg/Mn]',
                'Y_fe': r'[Y/Fe]', 'Mg_Mn': r'[Mg/Mn]', 'Mn_fe': r'[Mn/Fe]',
                'Ba_fe': r'[Ba/Fe]', 'Mg_Cu': r'[Mg/Cu]', 'Eu_fe': r'[Eu/Fe]',
                'Ba_Eu': r'[Ba/Eu]', 'Na_fe': r'[Na/Fe]'
            }
        else:
            axis_label_dict = {
                'fe_h': r'[Fe/H]', 'E_50': r'Energy ($\times 10^5$)', 'Energy': r'Energy ($\times 10^5$)',
                'alpha_m': r'[$\alpha$/Fe]', 'alpha_fe': r'[$\alpha$/Fe]', 'ce_fe': r'[Ce/Fe]',
                'al_fe': r'[Al/Fe]', 'Al_fe': r'[Al/Fe]', 'mg_mn': r'[Mg/Mn]',
                'Y_fe': r'[Y/Fe]', 'Mg_Mn': r'[Mg/Mn]', 'Mn_fe': r'[Mn/Fe]',
                'Ba_fe': r'[Ba/Fe]', 'Mg_Cu': r'[Mg/Cu]', 'Eu_fe': r'[Eu/Fe]',
                'Ba_Eu': r'[Ba/Eu]', 'Na_fe': r'[Na/Fe]'
            }

        xlabel = axis_label_dict.get(x_key, x_key)
        ylabel = axis_label_dict.get(y_key, y_key)

        # Defines colors for different components
        colors = color_palette if color_palette else sns.color_palette("husl", n_components)

        # Defines grid for subplots 
        fig = plt.figure(figsize=(8, 8))
        ax_main = plt.axes([0.1, 0.1, 0.6, 0.6])
        ax_histx = plt.axes([0.1, 0.71, 0.6, 0.19])
        ax_histy = plt.axes([0.71, 0.1, 0.19, 0.6])
        ax_bar = plt.axes([0.71, 0.71, 0.19, 0.19])

        # Background survey data
        if full_survey_file:
            with fits.open(full_survey_file) as hdul:
                survey_data = hdul[1].data

            # Capitalize only if file is from Apogee
            if "Apogee" in full_survey_file or "APOGEE" in full_survey_file:
                x_key_lookup = x_key.upper()
                y_key_lookup = y_key.upper()
            else:
                x_key_lookup = x_key
                y_key_lookup = y_key

            if x_key_lookup in survey_data.columns.names and y_key_lookup in survey_data.columns.names:
                full_x = survey_data[x_key_lookup]
                full_y = survey_data[y_key_lookup]

                # Remove NaNs
                mask = ~np.isnan(full_x) & ~np.isnan(full_y)

                # Plot background density
                ax_main.hist2d(
                    full_x[mask], full_y[mask],
                    bins=400, cmap='Greys', cmin=0, alpha=1,
                    norm=mcolors.PowerNorm(gamma=0.5)
                )

        # Main scatter plot - with stars coloured by their assigned Gaussian component
        ax_main.scatter(x_data, y_data, c=[colors[i-1] for i in assignments], s=3, alpha=0.5)
        ax_main.set_xlabel(xlabel, fontsize=13)
        ax_main.set_ylabel(ylabel, fontsize=13)
        ax_main.tick_params(axis='both', which='major', labelsize=12)

        if xlim:
            ax_main.set_xlim(xlim)
        if ylim:
            ax_main.set_ylim(ylim)

        # Plots Gaussian components as ellipses
        for i in range(n_components):
            mean = means[i, [x_index, y_index]]
            cov = covariances[i][[x_index, y_index], :][:, [x_index, y_index]]
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            eigvals, eigvecs = eigvals[order], eigvecs[:, order]
            angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
            width, height = 2 * np.sqrt(eigvals)
            scale = np.sqrt(chi2.ppf(norm.cdf(z_score) * 2 - 1, df=2))
            width *= scale
            height *= scale

            lw = np.clip(1.0 + 4.0 * weights[i], 1.0, 3.5)
            alpha = np.clip(0.4 + 0.7 * weights[i], 0.4, 0.9)

            ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                            edgecolor=colors[i], facecolor='none', linewidth=lw, alpha=alpha)
            ax_main.add_patch(ellipse)

        # Special formatting for energy
        if self.scaling:
            if y_key == 'E_50' or y_key == 'Energy':
                ax_main.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x * 1e-5:.1f}"))
                ax_main.set_ylabel(f"{ylabel} ($\\times 10^5$)", fontsize=13)

        # Histograms
        bins_x = np.linspace(np.min(x_data), np.max(x_data), 40)
        bins_y = np.linspace(np.min(y_data), np.max(y_data), 40)
        bin_width_x = np.diff(bins_x)[0]
        bin_width_y = np.diff(bins_y)[0]
        hist_x, _, _ = ax_histx.hist(x_data, bins=bins_x, color='gray', alpha=0.5)
        hist_y, _, _ = ax_histy.hist(y_data, bins=bins_y, color='gray', alpha=0.5, orientation='horizontal')
        ax_histx.set_xticks([])
        ax_histy.set_yticks([])
        if xlim:
            ax_histx.set_xlim(xlim)
        if ylim:
            ax_histy.set_ylim(ylim)
        ax_histx.tick_params(axis='y', which='major', labelsize=12)
        ax_histy.tick_params(axis='x', which='major', labelsize=12)

        # Gaussian overlays
        x_min = min(np.min(x_data), xlim[0]) if xlim else np.min(x_data)
        x_max = max(np.max(x_data), xlim[1]) if xlim else np.max(x_data)
        y_min = min(np.min(y_data), ylim[0]) if ylim else np.min(y_data)
        y_max = max(np.max(y_data), ylim[1]) if ylim else np.max(y_data)

        x_range = np.linspace(x_min, x_max, 300)
        y_range = np.linspace(y_min, y_max, 300)
        total_gauss_x = np.zeros_like(x_range)
        total_gauss_y = np.zeros_like(y_range)

        for i in range(n_components):
            mean_x, mean_y = means[i, x_index], means[i, y_index]
            std_x = np.sqrt(covariances[i][x_index, x_index])
            std_y = np.sqrt(covariances[i][y_index, y_index])
            weight = weights[i]
            scale_x = np.sum(hist_x) * weight * bin_width_x
            scale_y = np.sum(hist_y) * weight * bin_width_y
            gauss_x = norm.pdf(x_range, mean_x, std_x) * scale_x
            gauss_y = norm.pdf(y_range, mean_y, std_y) * scale_y
            total_gauss_x += gauss_x
            total_gauss_y += gauss_y
            ax_histx.plot(x_range, gauss_x, color=colors[i], alpha=0.7)
            ax_histy.plot(gauss_y, y_range, color=colors[i], alpha=0.7)

        ax_histx.plot(x_range, total_gauss_x, color='black', linewidth=2)
        ax_histy.plot(total_gauss_y, y_range, color='black', linewidth=2)

        # TOP RIGHT - Bar chart of relative weights
        sorted_indices = np.argsort(weights)[::-1]
        sorted_weights = weights[sorted_indices]
        sorted_colors = [colors[i] for i in sorted_indices]
        bar_heights = sorted_weights * 100
        bar_positions = np.arange(n_components)

        bars = ax_bar.bar(bar_positions, bar_heights, width=0.8, color=sorted_colors,
                        edgecolor='black', linewidth=0.8)

        for bar, weight in zip(bars, bar_heights):
            ax_bar.text(bar.get_x() + bar.get_width() / 2, weight + 2,
                        f"{weight:.1f}%", ha='center', va='bottom',
                        fontsize=11, color='black', rotation=90)

        ax_bar.set_ylim(0, 75)
        ax_bar.set_xlim(-0.5, n_components - 0.5)
        ax_bar.set_yticks([0, 20, 40, 60])
        ax_bar.set_yticklabels([])
        ax_bar.set_xticks([])
        ax_bar.yaxis.grid(True, linestyle='--', alpha=1)
        ax_bar.set_axisbelow(True)

        plt.show()

    def plot_nonXD(self, x_key: str, y_key: str, z_score: float = 2.0,
                full_survey_file: Optional[str] = None,
                color_palette: Optional[list] = None,
                xlim: Optional[tuple] = None,
                ylim: Optional[tuple] = None) -> None:
        """
        Creates a 2D diagnostic plot of the clustering results from Extreme Deconvolution (XD) assignments, 
        without relying on the original XD model parameters (means/covariances).

        Notes
        -----
        This method:

        * Displays individual stars colored by their assigned Gaussian component (from XD).
        * Fits new 2D Gaussians (empirically) to each component in the projection space (x_key vs y_key).
        * Overlays 2σ confidence ellipses from these fitted Gaussians.
        * Adds marginal histograms and overlaid Gaussian projections for each axis.
        * Plots a bar chart summarizing the relative weight of each component.
        * Optionally includes a background density reference (e.g. full APOGEE–Gaia sample) via a 2D histogram.

        
        Parameters
        ----------
        x_key : str
            The column name for the x-axis variable.
        y_key : str
            The column name for the y-axis variable.
        z_score : float, optional
            Confidence level scaling factor for ellipses (default is 2, ~95% confidence region).
        full_survey_file : str, optional
            Path to a FITS file containing a reference survey sample (e.g. total APOGEE) 
            to be plotted as a grayscale 2D histogram in the background.
        color_palette : list, optional
            Custom list of colors to assign to each Gaussian component.
        xlim : tuple, optional
            Limits for the x-axis, e.g. (-2, 0.5).
        ylim : tuple, optional
            Limits for the y-axis, e.g. (-0.5, 0.5).

        Raises
        ------
        ValueError
            If the XD assignment has not been performed before plotting.
            If the provided x_key or y_key is not present in the dataset.

        """

        # Ensures analysis has been run before plotting
        if self.assignment_metric is None:
            raise ValueError("No assignment metric found. Please run assignment method first.")

        # Ensures that the keys provided are valid
        if x_key not in self.star_data.colnames or y_key not in self.star_data.colnames:
            raise ValueError("Invalid keys. Please provide valid keys from the data table (i.e., exist in data_keys).")

        # Extracts the relevant parameters depending on the assignment metric used during assignment_XD
        if self.assignment_metric == 'best':
            assignment_params = self.best_params 
        elif self.assignment_metric == 'best filtered':
            assignment_params = self.filtered_best_params

        # Extracts GMM parameters from the relevant best-fit analysis
        weights = assignment_params['XD_weights']
        n_components = assignment_params['gauss_components']

        # Ensures the parameters are in the correct format
        weights = np.array(weights, dtype=np.float64)

        # Retrieves column index directly from the star_data table
        x_index = self.star_data.colnames.index(x_key)
        y_index = self.star_data.colnames.index(y_key)

        # Extracts the individual star data
        x_data = np.asarray(self.star_data[x_key])
        y_data = np.asarray(self.star_data[y_key])
        assignments = self.star_data['max_gauss']

        # Fit the data to a 2D Gaussian rather than using the XD results 
        # Convert to 2D array for easier indexing
        xy_data = np.vstack([x_data, y_data]).T
        # Initialize list to store calculated ellipses
        ellipse_params = []
        for i in range(1, n_components + 1):
            # Select data for component i
            cluster_points = xy_data[assignments == i]
            if len(cluster_points) < 2:
                continue  # Skip components with insufficient points
            # Compute mean and covariance
            mean = np.mean(cluster_points, axis=0)
            cov = np.cov(cluster_points, rowvar=False)
            ellipse_params.append((mean, cov))

        # Set axis labels (custom formatting) - Use scaled or unscaled labels based on the scaling flag
        if self.scaling:
            axis_label_dict = {
                'fe_h': r'[Fe/H]', 'E_50': r'Energy', 'Energy': r'Energy',
                'alpha_m': r'[$\alpha$/Fe]', 'alpha_fe': r'[$\alpha$/Fe]', 'ce_fe': r'[Ce/Fe]',
                'al_fe': r'[Al/Fe]', 'Al_fe': r'[Al/Fe]', 'mg_mn': r'[Mg/Mn]',
                'Y_fe': r'[Y/Fe]', 'Mg_Mn': r'[Mg/Mn]', 'Mn_fe': r'[Mn/Fe]',
                'Ba_fe': r'[Ba/Fe]', 'Mg_Cu': r'[Mg/Cu]', 'Eu_fe': r'[Eu/Fe]',
                'Ba_Eu': r'[Ba/Eu]', 'Na_fe': r'[Na/Fe]', 'Ni_fe': r'[Ni/Fe]', 'Eu_Mg': r'[Eu/Mg]', 'ni_fe': r'[Ni/Fe]'
            }
        else:
            axis_label_dict = {
                'fe_h': r'[Fe/H]', 'E_50': r'Energy ($\times 10^5$)', 'Energy': r'Energy ($\times 10^5$)',
                'alpha_m': r'[$\alpha$/Fe]', 'alpha_fe': r'[$\alpha$/Fe]', 'ce_fe': r'[Ce/Fe]',
                'al_fe': r'[Al/Fe]', 'Al_fe': r'[Al/Fe]', 'mg_mn': r'[Mg/Mn]',
                'Y_fe': r'[Y/Fe]', 'Mg_Mn': r'[Mg/Mn]', 'Mn_fe': r'[Mn/Fe]',
                'Ba_fe': r'[Ba/Fe]', 'Mg_Cu': r'[Mg/Cu]', 'Eu_fe': r'[Eu/Fe]',
                'Ba_Eu': r'[Ba/Eu]', 'Na_fe': r'[Na/Fe]', 'Ni_fe': r'[Ni/Fe]', 'Eu_Mg': r'[Eu/Mg]', 'ni_fe': r'[Ni/Fe]'
            }

        xlabel = axis_label_dict.get(x_key, x_key)
        ylabel = axis_label_dict.get(y_key, y_key)

        # Defines colors for different components
        colors = color_palette if color_palette else sns.color_palette("husl", n_components)

        # Defines grid for subplots 
        fig = plt.figure(figsize=(8, 8))
        ax_main = plt.axes([0.1, 0.1, 0.6, 0.6])
        ax_histx = plt.axes([0.1, 0.71, 0.6, 0.19])
        ax_histy = plt.axes([0.71, 0.1, 0.19, 0.6])
        ax_bar = plt.axes([0.71, 0.71, 0.19, 0.19])

        # Background survey data
        if full_survey_file:
            with fits.open(full_survey_file) as hdul:
                survey_data = hdul[1].data

            # Capitalize only if file is from Apogee
            if "Apogee" in full_survey_file or "APOGEE" in full_survey_file:
                x_key_lookup = x_key.upper()
                y_key_lookup = y_key.upper()
            else:
                x_key_lookup = x_key
                y_key_lookup = y_key

            if x_key_lookup in survey_data.columns.names and y_key_lookup in survey_data.columns.names:
                full_x = survey_data[x_key_lookup]
                full_y = survey_data[y_key_lookup]

                # Remove NaNs
                mask = ~np.isnan(full_x) & ~np.isnan(full_y)

                # Plot background density
                ax_main.hist2d(
                    full_x[mask], full_y[mask],
                    bins=400, cmap='Greys', cmin=0, alpha=1,
                    norm=mcolors.PowerNorm(gamma=0.5)
                )

        # Main scatter plot - with stars coloured by their assigned Gaussian component
        ax_main.scatter(x_data, y_data, c=[colors[i-1] for i in assignments], s=3, alpha=0.5)
        ax_main.set_xlabel(xlabel, fontsize=13)
        ax_main.set_ylabel(ylabel, fontsize=13)
        ax_main.tick_params(axis='both', which='major', labelsize=12)

        if xlim:
            ax_main.set_xlim(xlim)
        if ylim:
            ax_main.set_ylim(ylim)

        # Plots Gaussian components as ellipses
        for i, (mean, cov) in enumerate(ellipse_params):
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            eigvals, eigvecs = eigvals[order], eigvecs[:, order]
            angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
            width, height = 2 * np.sqrt(eigvals)

            scale = np.sqrt(chi2.ppf(norm.cdf(z_score) * 2 - 1, df=2))
            width *= scale
            height *= scale

            lw = np.clip(1.0 + 4.0 * weights[i], 1.0, 3.5)
            alpha = np.clip(0.4 + 0.7 * weights[i], 0.4, 0.9)

            ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                            edgecolor=colors[i], facecolor='none', linewidth=lw, alpha=alpha, linestyle='--')
            ax_main.add_patch(ellipse)

        # Special formatting for energy
        if self.scaling:
            if y_key == 'E_50' or y_key == 'Energy':
                ax_main.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x * 1e-5:.1f}"))
                ax_main.set_ylabel(f"{ylabel} ($\\times 10^5$)", fontsize=13)

        # Histograms
        bins_x = np.linspace(np.min(x_data), np.max(x_data), 40)
        bins_y = np.linspace(np.min(y_data), np.max(y_data), 40)
        bin_width_x = np.diff(bins_x)[0]
        bin_width_y = np.diff(bins_y)[0]
        hist_x, _, _ = ax_histx.hist(x_data, bins=bins_x, color='gray', alpha=0.5)
        hist_y, _, _ = ax_histy.hist(y_data, bins=bins_y, color='gray', alpha=0.5, orientation='horizontal')
        ax_histx.set_xticks([])
        ax_histy.set_yticks([])
        if xlim:
            ax_histx.set_xlim(xlim)
        if ylim:
            ax_histy.set_ylim(ylim)
        ax_histx.tick_params(axis='y', which='major', labelsize=12)
        ax_histy.tick_params(axis='x', which='major', labelsize=12)

        # Gaussian overlays
        x_range = np.linspace(np.min(x_data), np.max(x_data), 300)
        y_range = np.linspace(np.min(y_data), np.max(y_data), 300)
        total_gauss_x = np.zeros_like(x_range)
        total_gauss_y = np.zeros_like(y_range)
        
        for i, (mean, cov) in enumerate(ellipse_params):
            mean_x, mean_y = mean[0], mean[1]
            std_x = np.sqrt(cov[0, 0])
            std_y = np.sqrt(cov[1, 1])
            weight = weights[i]

            scale_x = np.sum(hist_x) * weight * bin_width_x
            scale_y = np.sum(hist_y) * weight * bin_width_y
            gauss_x = norm.pdf(x_range, mean_x, std_x) * scale_x
            gauss_y = norm.pdf(y_range, mean_y, std_y) * scale_y
            total_gauss_x += gauss_x
            total_gauss_y += gauss_y
            ax_histx.plot(x_range, gauss_x, color=colors[i], alpha=0.7,  ls ='--')
            ax_histy.plot(gauss_y, y_range, color=colors[i], alpha=0.7,  ls ='--')

        ax_histx.plot(x_range, total_gauss_x, color='black', linewidth=2, ls ='--')
        ax_histy.plot(total_gauss_y, y_range, color='black', linewidth=2, ls ='--')

        # TOP RIGHT - Bar chart of relative weights
        sorted_indices = np.argsort(weights)[::-1]
        sorted_weights = weights[sorted_indices]
        sorted_colors = [colors[i] for i in sorted_indices]
        bar_heights = sorted_weights * 100
        bar_positions = np.arange(n_components)

        bars = ax_bar.bar(bar_positions, bar_heights, width=0.8, color=sorted_colors,
                        edgecolor='black', linewidth=0.8)

        for bar, weight in zip(bars, bar_heights):
            ax_bar.text(bar.get_x() + bar.get_width() / 2, weight + 2,
                        f"{weight:.1f}%", ha='center', va='bottom',
                        fontsize=11, color='black', rotation=90)

        ax_bar.set_ylim(0, 75)
        ax_bar.set_xlim(-0.5, n_components - 0.5)
        ax_bar.set_yticks([0, 20, 40, 60])
        ax_bar.set_yticklabels([])
        ax_bar.set_xticks([])
        ax_bar.yaxis.grid(True, linestyle='--', alpha=1)
        ax_bar.set_axisbelow(True)

        plt.show()



def compare_assignments(results_table, target_label, label_map):
    """
    For stars primarily assigned to `target_label`, this function:

    Notes
    -----

    * Finds their second-best Gaussian component assignments
    * Summarizes how often each second-best component occurs
    * Calculates the mean, median, and standard deviation of:
        * The percent of the first-best probability that the second-best achieved
        * The absolute difference in probability between the first and second-best
        
    Parameters
    ----------
    results_table : Astropy Table
        Contains Gaussian Mixture Model results.

    target_label : str
        Name of the component youre analysing (e.g., "Aurora") to int.

    label_map : dict
        Maps of intenger indices (starting from 1) to astrophysical names of componets.
    """

    # Map the input target label to its corresponding Gaussian number
    target_gauss = next((k for k, v in label_map.items() if v == target_label), None)
    if target_gauss is None:
        raise ValueError(f"Label '{target_label}' not found in label_map.")

    # Identify all rows where this component is the maximum assignment ie items 'belonging to this assignment'
    matching_rows = results_table[results_table['max_gauss'] == target_gauss]

    if len(matching_rows) == 0:
        print(f"No stars found for primary label '{target_label}'.")
        return

    # Extract the probability assigned to each component from the results table and stack them into idependent columns 
    # Resultant shape: (n_stars, n_components)
    prob_cols = [col for col in results_table.colnames if col.startswith("prob_gauss_")]
    prob_array = np.vstack([matching_rows[col] for col in prob_cols]).T

    # For each remaining star row in this table, identify the indices of the best and second-best components
    # The best is simply the input target label 

    # Sort the indiceies of the column by increasing probability
    sorted_indices = np.argsort(prob_array, axis=1) 

    # Get the best and second-best indices so their value can be compared later
    best_indices = sorted_indices[:, -1]                    
    second_best_indices = sorted_indices[:, -2]       

    # Retrieve actual probabilities using these indices 
    best_probs = np.take_along_axis(prob_array, best_indices[:, None], axis=1).flatten()
    second_probs = np.take_along_axis(prob_array, second_best_indices[:, None], axis=1).flatten()

    # Calculate relative (percentage) and absolute differences in best and second-best probabilities
    percent_diffs = 100 * (second_probs / best_probs)        
    absolute_diffs = np.abs(best_probs - second_probs)      

    # Group percent and absolute differences by second-best component and store them so they can be displayed
    percent_stats = defaultdict(list)
    abs_stats = defaultdict(list)
    for idx, sec_idx in enumerate(second_best_indices):
        gauss_num = sec_idx + 1 
        percent_stats[gauss_num].append(percent_diffs[idx])
        abs_stats[gauss_num].append(absolute_diffs[idx])

    
    # Displaying the results

    # Build and display the output summary table
    print(f"\nDetailed second-best breakdown for stars primarily assigned to '{target_label}':")
    print(f"Total stars: {len(matching_rows)}\n")

    # Define headers and table structure
    table_headers = [
        "Second-Best Component", "# Stars", 
        "Mean % of 1st", "Std % of 1st", "Median % of 1st",
        "Mean Abs Diff", "Median Abs Diff", "Std Abs Diff"
    ]
    table_rows = []

    # Enter all the data into the table
    for gauss_num in sorted(percent_stats.keys(), key=lambda x: -len(percent_stats[x])):
        label = label_map.get(gauss_num, f"Gaussian {gauss_num}")
        percent_list = percent_stats[gauss_num]
        abs_list = abs_stats[gauss_num]
        table_rows.append([
            label,
            len(percent_list),
            f"{np.mean(percent_list):.2f}",
            f"{np.std(percent_list):.2f}",
            f"{np.median(percent_list):.2f}",
            f"{np.mean(abs_list):.6f}",
            f"{np.median(abs_list):.6f}",
            f"{np.std(abs_list):.6f}"
        ])

    # Display table
    print(tabulate(table_rows, headers=table_headers, tablefmt="github"))