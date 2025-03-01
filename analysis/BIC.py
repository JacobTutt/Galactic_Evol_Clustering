import numpy as np
from tabulate import tabulate
import pandas as pd

from astropy.table import Table
from tqdm.notebook import tqdm
import pickle as pkl

from extreme_deconvolution import extreme_deconvolution
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2, norm, gaussian_kde, multivariate_normal

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse

from typing import List, Tuple, Optional, Union

class XDPipeline:
    """
    """
    def __init__(self, star_data: Union[Table, np.recarray, pd.DataFrame], data_keys: List[str], data_err_keys: List[str]):
        """
        Initialise the XDPipeline with stellar data and keys of intrest for the Gaussian Mixture Model (GMM) - Extreme Deconvolution (XD) process, defining the parameter space of interest. 

        Parameters:
            star_data (Table, np.recarray, pd.DataFrame): Dataset containing stellar information, which can be an Astropy Table, NumPy recarray, or Pandas DataFrame.
            data_keys (List[str]): List of column names representing features used in the GMM fitting.
            data_err_keys (List[str]): List of column names representing measurement uncertainties, corresponding to `data_keys`.

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

        # Extract the data and their errors from the data using the keys provided and stack them into a 2D array, (number_of_parameters, number_of_samples)
        self.feature_data = np.vstack([self.star_data[key] for key in self.data_keys]).T

        # Extract the errors from the astropy table using the keys provided
        # We have no informatiom on the correlation between the errors so assume they are diagonal/uncorrelated. 
        self.errors_data = np.vstack([self.star_data[err_key] for err_key in self.data_err_keys]).T

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
        Run Extreme Deconvolution (XD) for a range of Gaussian components with multiple random initializations.
        The results are evaluated using BIC and AIC for model selection.

        Parameters
        ----------
        gauss_component_range : Tuple[int, int]
            Range of Gaussian components to test, specified as (min, max).
        max_iterations : int
            Maximum number of EM iterations per XD run.
        n_repeats : int
            Number of complete repetitions of the initialization process.
        n_init : int
            Number of random initializations per component count.
        save_path : Optional[str]
            Path to save XD results. If None, results are not saved.

        Raises
        ------
        ValueError
            If `gauss_component_range` is invalid.
        TypeError
            If `n_repeats`, `n_init`, or `max_iterations` are not positive integers.
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


        # Scale the data to have zero mean and unit variance - this will improve the convergence of the EM algorithm
        # Errors are scaled by the same factor to maintain the same relative uncertainty
        # Note this will require the gaussians means and covariances returned by XD to be scaled back to the original units for interpretation before saving
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(self.feature_data)
        errors_scaled = self.errors_data / scaler.scale_

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

                    # Covariances initialised as identity matrices
                    init_covar = np.array([np.identity(self.n_features) for _ in range(n_gauss)])
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

                        # Unscale the means and covariances to return them to their original/meaningful units
                        post_scaling_means = scaler.inverse_transform(post_XD_means)
                        post_scaling_cov = np.array([
                            np.dot(np.dot(np.diag(scaler.scale_), cov), np.diag(scaler.scale_))
                            for cov in post_XD_cov
                        ])

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
    
    def compare_XD(self, opt_metric = 'BIC', n_gauss_filter: Optional[int] = None, repeat_no_filter: Optional[int] = None, save_path: Optional[str] = None) -> None:
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

        # Print BIC and AIC summary tables
        print("Table of BIC Means and Stds")
        print(tabulate(BIC_means_stds_df, headers='keys', tablefmt='psql'))

        print("Table of AIC Means and Stds")
        print(tabulate(AIC_means_stds_df, headers='keys', tablefmt='psql'))

        # Calculate min, max, and median BIC and AIC scores across repeats and initialisations for each Gaussian component count ie from n_init * n_repeats values - (n_gauss_components)
        BIC_min, BIC_max, BIC_median = BIC_scores.min(axis=(1, 2)), BIC_scores.max(axis=(1, 2)), np.nanmedian(BIC_scores, axis=(1, 2))
        AIC_min, AIC_max, AIC_median = AIC_scores.min(axis=(1, 2)), AIC_scores.max(axis=(1, 2)), np.nanmedian(AIC_scores, axis=(1, 2))

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

        return None
    
    def assigment_XD(self, assignment_metric = 'best'): 
        """
        Assign stars to Gaussian components based on the best-fit XD model.
        Computes the responsibility of each gaussians for each star and assigns it accordingly.

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
        
        # Initialises columns for probabilities and assignments
        for i in range(assignment_params['gauss_components']):
            self.star_data[f'prob_gauss_{i+1}'] = np.zeros(len(self.star_data))
            self.star_data['max_gauss'] = np.zeros(len(self.star_data), dtype=int)

        # For each star calculate the probability of it belonging to each gaussian
        for star_index, star in enumerate(self.feature_data):
            probabilities = []
            # Extract the measurement error covariance for the current star and conver it to a diagonal matrix
            star_errors = np.diag(self.errors_data[star_index])
            # Cycle through each of the gaussinas from the n_gauss
            for j in range(assignment_params['gauss_components']):
                # Mean covariance and weight for the jth gaussian
                mean_j = assignment_params['XD_means'][j]
                # Error-Aware Covariance Adjustment
                cov_j = assignment_params['XD_covariances'][j] + star_errors
                weight_j = assignment_params['XD_weights'][j]

                # Calculate the probability of the data point given the gaussian
                try:
                    prob = weight_j * multivariate_normal.pdf(star, mean=mean_j, cov=cov_j)
                # Handle singular covariance matrices
                except np.linalg.LinAlgError:
                    prob = 0 

                probabilities.append(prob)
                self.star_data[f'prob_gauss_{j+1}'][star_index] = prob
            # Assign the star to the gaussian with the highest probability
            self.star_data['max_gauss'][star_index] = np.argmax(probabilities) + 1

        return None

    
    def plot_XD(self, x_key: str, y_key: str, z_score: Optional[float] = 2) -> None:
        """
        Generate a 2D plot of the Extreme Deconvolution (XD) results, showing the Gaussian components
        as ellipses and individual stars colored by their assigned Gaussian component.

        The plot mimics a corner plot style with marginal histograms for both x and y variables.

        Parameters
        ----------
        x_key : str
            The key for the x-axis parameter (must exist in data_keys).
        y_key : str
            The key for the y-axis parameter (must exist in data_keys).

        Raises
        ------
        ValueError
            If XD results are missing or if invalid keys are provided.
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

        # Retrieves relevant keys index positions - allows extraction of relevent data
        # Ie Star data features for the x and y axis , relevent means and covariance values from the XD analysis per gaussian
        x_index = self.data_keys.index(x_key)
        y_index = self.data_keys.index(y_key)

        # Extracts the individual star data
        x_data = self.feature_data[:, x_index]
        y_data = self.feature_data[:, y_index]

        # Extracts each stars assigned gaussian component - for colouring by component 
        assignments = self.star_data['max_gauss']

        # Defines colors for different components
        colors = sns.color_palette("husl", n_components)

        # Defines grid for subplots 
        fig = plt.figure(figsize=(8, 8))
        # Scatter plot - bottom left
        ax_main = plt.axes([0.1, 0.1, 0.65, 0.65]) 
        # Horizontal histogram - top left
        ax_histx = plt.axes([0.1, 0.76 , 0.65, 0.14]) 
        # Vertical histogram - bottom right
        ax_histy = plt.axes([0.76, 0.1, 0.14, 0.65])
        # Weight bar plot - top right
        ax_bar = plt.axes([0.76, 0.76, 0.14, 0.14])

        # Main scatter plot - with stars coloured by their assigned Gaussian component
        ax_main.scatter(x_data, y_data, c=[colors[i-1] for i in assignments], s=5, alpha=0.5)
        ax_main.set_xlabel(x_key)
        ax_main.set_ylabel(y_key)

        # Plots Gaussian components as ellipses
        for i in range(n_components):
            # Extracts relevant features (x, y) from the mean vector of the i-th Gaussian component
            mean = means[i, [x_index, y_index]]

            # Extracts the 2x2 covariance sub-matrix for (x, y) from the full covariance matrix of the i-th Gaussian component
            # Defines the spread and correlation for (x, y)
            cov = covariances[i][[x_index, y_index], :][:, [x_index, y_index]] 

            # Calculates the eigenvalues and eigenvectors of covariance matrix
            # Eigenvalues: length of major and minor axis of the ellipse
            # Eigenvectors: orientation of the major and minor axis of the ellipse - angles
            eigvals, eigvecs = np.linalg.eigh(cov)

            # Sorts eigenvalues in descending order - Larger eigenvalues: major axis variance, smaller eigenvalues: minor axis variance
            # Corresponding eigenvectors: major and minor axis orientation
            order = np.argsort(eigvals)[::-1]
            eigvals, eigvecs = eigvals[order], eigvecs[:, order]

            # Converts eigenvectors of major axis to angle in degrees
            angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))  

            # Computes width and height of the ellipse from variances (eigenvalues)
            width, height = 2 * np.sqrt(eigvals)  

            # Scales the ellipse using the desired z score/ confidence interval by scaling the width and height 
            scale_factor = np.sqrt(chi2.ppf(norm.cdf(z_score) * 2 - 1, df=2))
            width *= scale_factor
            height *= scale_factor

            # Adds the ellipse patch to the plot using the mean(centre), width, height and angle
            ellipse = Ellipse(
                xy=(mean[0], mean[1]),  
                width=width, height=height,  
                angle=angle, edgecolor=colors[i], facecolor='none', linewidth=2
            )
            ax_main.add_patch(ellipse)


        # Defines histogram bins edges using data range
        bins_x = np.linspace(np.min(x_data), np.max(x_data), 25)
        bins_y = np.linspace(np.min(y_data), np.max(y_data), 25)

        bin_width_x = np.diff(bins_x)[0]
        bin_width_y = np.diff(bins_y)[0]

        # Computes histograms for grey bars (scaled correctly)
        hist_x, _, _ = ax_histx.hist(x_data, bins=bins_x, color='gray', alpha=0.5)
        hist_y, _, _ = ax_histy.hist(y_data, bins=bins_y, color='gray', alpha=0.5, orientation='horizontal')
        ax_histx.set_xticks([])
        ax_histy.set_yticks([])

        # Defines a range for plotting 1D Gaussians
        x_1_gauss= np.linspace(np.min(x_data), np.max(x_data), 300)
        y_1_gauss = np.linspace(np.min(y_data), np.max(y_data), 300)

        # Initialises sum of Gaussians for total distribution
        total_gauss_x = np.zeros_like(x_1_gauss)
        total_gauss_y = np.zeros_like(y_1_gauss)

        # Plot individual 1D Gaussian components
        for i in range(n_components):
            # Retrieve mean and standard deviation for x and y from the i-th Gaussian component
            mean_x, mean_y = means[i, x_index], means[i, y_index]
            std_x, std_y = np.sqrt(covariances[i][x_index, x_index]), np.sqrt(covariances[i][y_index, y_index])

            # Fraction of total samples assigned to i-th Gaussian
            component_weight = weights[i]
            # Match histogram area
            scale_factor_gauss = np.sum(hist_x) * component_weight

            # Evaluate Gaussian over many full range of x and y
            gauss_x = norm.pdf(x_1_gauss, mean_x, std_x) * scale_factor_gauss * bin_width_x
            gauss_y = norm.pdf(y_1_gauss, mean_y, std_y) * scale_factor_gauss * bin_width_y

            # Add i-th Gaussian to overall gaussian sum distribution
            total_gauss_x += gauss_x
            total_gauss_y += gauss_y

            # Add plot of i-th Gaussian component
            ax_histx.plot(x_1_gauss, gauss_x, color=colors[i], alpha=0.7)
            ax_histy.plot(gauss_y, y_1_gauss, color=colors[i], alpha=0.7)

        # Plot overall gaussian sum distribution (black)
        ax_histx.plot(x_1_gauss, total_gauss_x, color='black', linewidth=2)
        ax_histy.plot(total_gauss_y, y_1_gauss, color='black', linewidth=2)

        # Initialise KDE probability distribution for each axis
        kde_x_func = gaussian_kde(x_data, bw_method='scott')
        kde_y_func = gaussian_kde(y_data, bw_method='scott')

        # Scale by area of histogram to match histrograma and 1D gaussian scales
        kde_x = kde_x_func(x_1_gauss) * np.sum(hist_x) * bin_width_x
        kde_y = kde_y_func(y_1_gauss) * np.sum(hist_y) * bin_width_y
        
        # Plot KDE distributions
        ax_histx.plot(x_1_gauss, kde_x, color='red', linestyle='dashed', linewidth=2, label="KDE from Histogram")
        ax_histy.plot(kde_y, y_1_gauss, color='red', linestyle='dashed', linewidth=2)

        # TOP RIGHT - Bar chart of relative weights
        # Reorder weights of components by ascending size and respective colors
        sort_weight_indices = np.argsort(weights)
        sorted_weights = weights[sort_weight_indices]
        sorted_colors = [colors[i] for i in sort_weight_indices]

        # Plot bats
        bars = ax_bar.bar(range(n_components), sorted_weights * 100, color=sorted_colors)

        # Add text labeling weight as percentage on top of each bar
        for bar, weight in zip(bars, sorted_weights * 100):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                0,
                f" {weight:.1f}%", 
                ha='center', va='bottom',
                fontsize=8,
                color='black',
                rotation=90 
            )

        # Remove axis labels and ticks
        ax_bar.set_yticks([])
        ax_bar.set_yticklabels([])
        ax_bar.set_ylabel("")
        ax_bar.set_xticks([])

        plt.show()

        return None
